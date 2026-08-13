"""Exact-child asynchronous process execution for the local agent bridge."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
import os
from pathlib import Path
import signal


LineCallback = Callable[[str, str], None]
# Existing fake-provider volume is 2,000 34-byte lines per stream (68 KiB).
# These limits retain headroom while bounding a provider-controlled transcript.
MAX_PROCESS_STREAM_LINES = 4_096
MAX_PROCESS_STREAM_BYTES = 256 * 1024
MAX_PROCESS_TOTAL_LINES = 8_192
MAX_PROCESS_TOTAL_BYTES = 512 * 1024
_OUTPUT_LIMIT_ERROR = "provider process output exceeded configured bounds"


class ProcessOutputLimitExceeded(RuntimeError):
    """A provider process exceeded the bounded output capture contract."""


@dataclass
class _OutputBudget:
    stream_lines: dict[str, int]
    stream_bytes: dict[str, int]
    total_lines: int = 0
    total_bytes: int = 0

    def accept(self, stream: str, raw_line: bytes) -> None:
        next_stream_lines = self.stream_lines[stream] + 1
        next_stream_bytes = self.stream_bytes[stream] + len(raw_line)
        next_total_lines = self.total_lines + 1
        next_total_bytes = self.total_bytes + len(raw_line)
        if (
            next_stream_lines > MAX_PROCESS_STREAM_LINES
            or next_stream_bytes > MAX_PROCESS_STREAM_BYTES
            or next_total_lines > MAX_PROCESS_TOTAL_LINES
            or next_total_bytes > MAX_PROCESS_TOTAL_BYTES
        ):
            raise ProcessOutputLimitExceeded(_OUTPUT_LIMIT_ERROR)
        self.stream_lines[stream] = next_stream_lines
        self.stream_bytes[stream] = next_stream_bytes
        self.total_lines = next_total_lines
        self.total_bytes = next_total_bytes


@dataclass(frozen=True)
class ProcessResult:
    run_id: str
    pid: int
    process_group_id: int
    exit_code: int
    stdout: tuple[str, ...]
    stderr: tuple[str, ...]
    interrupted: bool


@dataclass
class _ActiveRun:
    process: asyncio.subprocess.Process | None
    process_group_id: int | None
    launch_finished: asyncio.Event
    stop_lock: asyncio.Lock
    stop_requested: bool = False
    stop_completed: bool = False
    interrupted: bool = False


class ProcessRunner:
    """Run and stop only process groups created by this runner instance."""

    def __init__(self, *, stop_grace_seconds: float = 5.0) -> None:
        if stop_grace_seconds < 0:
            raise ValueError("stop_grace_seconds must be >= 0")
        if os.name != "posix":
            raise RuntimeError("ProcessRunner requires POSIX process groups")
        self._stop_grace_seconds = stop_grace_seconds
        self._active_runs: dict[str, _ActiveRun] = {}
        self._start_events: dict[str, asyncio.Event] = {}

    async def run(
        self,
        *,
        run_id: str,
        argv: Sequence[str],
        cwd: str | Path,
        env: Mapping[str, str],
        stdin: bytes | None,
        on_line: LineCallback,
        pass_fds: Sequence[int] = (),
    ) -> ProcessResult:
        """Synchronously await one exec child and collect its output lines."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        normalized_argv = tuple(argv)
        if not normalized_argv or not all(isinstance(argument, str) for argument in normalized_argv):
            raise ValueError("argv must be a non-empty sequence of strings")
        normalized_cwd = os.fspath(cwd)
        normalized_env = dict(env)
        normalized_pass_fds = tuple(pass_fds)
        if not all(
            isinstance(descriptor, int)
            and not isinstance(descriptor, bool)
            and descriptor >= 0
            for descriptor in normalized_pass_fds
        ):
            raise ValueError("pass_fds must contain open file descriptors")
        try:
            for descriptor in normalized_pass_fds:
                os.fstat(descriptor)
        except OSError as error:
            raise ValueError("pass_fds must contain open file descriptors") from error
        if stdin is not None and not isinstance(stdin, bytes):
            raise ValueError("stdin must be bytes or None")
        if not callable(on_line):
            raise ValueError("on_line must be callable")
        if run_id in self._active_runs or run_id in self._start_events:
            raise ValueError(f"run_id is already active: {run_id}")

        started = asyncio.Event()
        process: asyncio.subprocess.Process | None = None
        active = _ActiveRun(
            process=None,
            process_group_id=None,
            launch_finished=started,
            stop_lock=asyncio.Lock(),
        )
        stdout: list[str] = []
        stderr: list[str] = []
        output_budget = _OutputBudget(
            stream_lines={"stdout": 0, "stderr": 0},
            stream_bytes={"stdout": 0, "stderr": 0},
        )
        try:
            self._start_events[run_id] = started
            self._active_runs[run_id] = active
            launch_task = asyncio.create_task(asyncio.create_subprocess_exec(
                *normalized_argv,
                cwd=normalized_cwd,
                env=normalized_env,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                pass_fds=normalized_pass_fds,
            ))
            try:
                process = await asyncio.shield(launch_task)
            except asyncio.CancelledError:
                process = await launch_task
                active.process = process
                active.process_group_id = process.pid
                started.set()
                raise
            active.process = process
            active.process_group_id = process.pid
            started.set()
            if active.stop_requested:
                await self._stop_active(active)

            execution_tasks = (
                asyncio.create_task(
                    self._drain("stdout", process.stdout, stdout, on_line, output_budget)
                ),
                asyncio.create_task(
                    self._drain("stderr", process.stderr, stderr, on_line, output_budget)
                ),
                asyncio.create_task(self._write_stdin(process.stdin, stdin)),
                asyncio.create_task(process.wait()),
            )
            try:
                await asyncio.gather(*execution_tasks)
            except BaseException:
                for task in execution_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*execution_tasks, return_exceptions=True)
                raise
            if process.returncode is None:
                raise RuntimeError("child process did not report an exit code")
            return ProcessResult(
                run_id=run_id,
                pid=process.pid,
                process_group_id=process.pid,
                exit_code=process.returncode,
                stdout=tuple(stdout),
                stderr=tuple(stderr),
                interrupted=active.interrupted,
            )
        except BaseException:
            if (
                active.process_group_id is not None
                and self._process_group_exists(active.process_group_id)
            ):
                await self._stop_active(active)
            raise
        finally:
            if not started.is_set():
                started.set()
            if self._active_runs.get(run_id) is active:
                del self._active_runs[run_id]
            self._start_events.pop(run_id, None)

    async def stop(self, run_id: str) -> None:
        """Interrupt precisely the process group owned by ``run_id``."""
        active = self._active_runs.get(run_id)
        if active is None:
            raise KeyError(run_id)
        active.stop_requested = True
        active.interrupted = True
        await active.launch_finished.wait()
        if active.process is None:
            return
        await self._stop_active(active)

    async def wait_until_started(self, run_id: str) -> None:
        """Wait until a scheduled ``run`` has registered its exact child."""
        event = self._start_events.get(run_id)
        if event is None:
            await asyncio.sleep(0)
            event = self._start_events.get(run_id)
        if event is None:
            raise KeyError(run_id)
        await event.wait()
        if run_id not in self._active_runs:
            raise KeyError(run_id)

    def is_running(self, run_id: str) -> bool:
        active = self._active_runs.get(run_id)
        return active is not None and active.process is not None

    async def _stop_active(self, active: _ActiveRun) -> None:
        async with active.stop_lock:
            active.interrupted = True
            if active.stop_completed:
                return
            if active.process is None or active.process_group_id is None:
                return
            try:
                os.killpg(active.process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                await active.process.wait()
                active.stop_completed = True
                return
            if not await self._wait_for_process_group_exit(active.process_group_id):
                with suppress(ProcessLookupError):
                    os.killpg(active.process_group_id, signal.SIGKILL)
            await active.process.wait()
            active.stop_completed = True

    async def _wait_for_process_group_exit(self, process_group_id: int) -> bool:
        deadline = asyncio.get_running_loop().time() + self._stop_grace_seconds
        while self._process_group_exists(process_group_id):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(remaining, 0.01))
        return True

    @staticmethod
    def _process_group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        return True

    @staticmethod
    async def _drain(
        stream: str,
        reader: asyncio.StreamReader | None,
        captured: list[str],
        on_line: LineCallback,
        output_budget: _OutputBudget,
    ) -> None:
        if reader is None:
            raise RuntimeError(f"{stream} pipe was not created")
        while True:
            try:
                raw_line = await reader.readline()
            except ValueError:
                raise ProcessOutputLimitExceeded(_OUTPUT_LIMIT_ERROR) from None
            if not raw_line:
                break
            output_budget.accept(stream, raw_line)
            line = raw_line.decode("utf-8", errors="replace").removesuffix("\n").removesuffix("\r")
            captured.append(line)
            on_line(stream, line)

    @staticmethod
    async def _write_stdin(
        writer: asyncio.StreamWriter | None, stdin: bytes | None,
    ) -> None:
        if stdin is None:
            return
        if writer is None:
            raise RuntimeError("stdin pipe was not created")
        try:
            writer.write(stdin)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            writer.close()
            with suppress(BrokenPipeError, ConnectionResetError):
                await writer.wait_closed()
