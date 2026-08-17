"""Exact-child asynchronous process execution for the local agent bridge."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
import os
from pathlib import Path
import signal
import time


LineCallback = Callable[[str, str], None]
# Existing fake-provider volume is 2,000 34-byte lines per stream (68 KiB).
# These limits retain headroom while bounding a provider-controlled transcript.
MAX_PROCESS_STREAM_LINES = 4_096
MAX_PROCESS_STREAM_BYTES = 256 * 1024
MAX_PROCESS_TOTAL_LINES = 8_192
MAX_PROCESS_TOTAL_BYTES = 512 * 1024
MAX_TERMINAL_PROCESS_EXITS = 128
_OUTPUT_LIMIT_ERROR = "provider process output exceeded configured bounds"
_KILL_REAP_SECONDS = 1.0


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


@dataclass(frozen=True, slots=True)
class StopReceipt:
    """The exact local child state observed for one stop request."""

    run_id: str
    was_running: bool
    process_exited: bool


@dataclass
class _ActiveRun:
    process: asyncio.subprocess.Process | None
    process_group_id: int | None
    launch_finished: asyncio.Event
    process_exited: asyncio.Event
    stop_lock: asyncio.Lock
    stop_requested: bool = False
    termination_started: bool = False
    kill_sent: bool = False
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
        self._terminal_process_exits: OrderedDict[str, asyncio.Event] = OrderedDict()
        self._closed = False
        self._close_complete = False
        self._close_task: asyncio.Task[None] | None = None

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
        if self._closed:
            raise RuntimeError("ProcessRunner is closed")
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
        if (
            run_id in self._active_runs
            or run_id in self._start_events
            or run_id in self._terminal_process_exits
        ):
            raise ValueError(f"run_id is active or retained: {run_id}")

        started = asyncio.Event()
        process: asyncio.subprocess.Process | None = None
        active = _ActiveRun(
            process=None,
            process_group_id=None,
            launch_finished=started,
            process_exited=asyncio.Event(),
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
                asyncio.create_task(self._observe_process_exit(active, process)),
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
            active.process_exited.set()
            if self._active_runs.get(run_id) is active:
                del self._active_runs[run_id]
            self._start_events.pop(run_id, None)
            if process is not None:
                self._remember_terminal_process_exit(run_id, active.process_exited)

    async def stop(self, run_id: str, *, timeout_seconds: float) -> StopReceipt:
        """Interrupt precisely the process group owned by ``run_id``."""
        self._validate_run_id(run_id)
        self._validate_timeout(timeout_seconds)
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        active = self._active_runs.get(run_id)
        if active is None:
            return StopReceipt(run_id=run_id, was_running=False, process_exited=True)
        active.stop_requested = True
        active.interrupted = True
        if not await self._wait_for_event_until(active.launch_finished, deadline):
            return StopReceipt(
                run_id=run_id,
                was_running=True,
                process_exited=active.process_exited.is_set(),
            )
        if active.process is not None:
            await self._stop_active(active, deadline=deadline)
        return StopReceipt(
            run_id=run_id,
            was_running=True,
            process_exited=await self._wait_for_event_until(active.process_exited, deadline),
        )

    async def wait_process_exit(self, run_id: str, *, timeout_seconds: float) -> None:
        """Wait only for the exact local child registered under ``run_id``."""
        self._validate_run_id(run_id)
        self._validate_timeout(timeout_seconds)
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        active = self._active_runs.get(run_id)
        process_exited = (
            active.process_exited
            if active is not None
            else self._terminal_process_exits.get(run_id)
        )
        if process_exited is None:
            raise KeyError(run_id)
        if not await self._wait_for_event_until(process_exited, deadline):
            raise TimeoutError(run_id)

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

    async def aclose(self) -> None:
        """Stop and reap every exact child group before returning."""
        if self._close_complete:
            return
        existing = self._close_task
        if existing is not None:
            await asyncio.shield(existing)
            return
        self._closed = True
        closing = asyncio.create_task(self._close_active_runs())
        self._close_task = closing
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            # Keep the owned shutdown task observable for a later caller.
            raise
        except BaseException:
            if self._close_task is closing:
                self._close_task = None
            raise
        self._close_complete = True

    def close(self) -> None:
        """Synchronously stop/reap exact groups after the event loop has unwound."""
        if self._close_complete:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("await ProcessRunner.aclose() while an event loop is running")
        self._closed = True
        self._close_active_runs_synchronously()
        self._close_complete = True

    async def _close_active_runs(self) -> None:
        active_runs = tuple(self._active_runs.values())
        for active in active_runs:
            active.stop_requested = True
            active.interrupted = True
        await asyncio.gather(*(self._close_active(active) for active in active_runs))

    async def _close_active(self, active: _ActiveRun) -> None:
        if not active.launch_finished.is_set():
            deadline = asyncio.get_running_loop().time() + self._stop_grace_seconds
            if not await self._wait_for_event_until(active.launch_finished, deadline):
                raise RuntimeError("provider launch did not finish during shutdown")
        if active.process is not None:
            await self._stop_active(active)
        elif active.process_group_id is not None:
            raise RuntimeError("provider group has no exact child process")
        else:
            active.stop_completed = True

    def _close_active_runs_synchronously(self) -> None:
        active_runs = tuple(self._active_runs.values())
        for active in active_runs:
            active.stop_requested = True
            active.interrupted = True
            if not active.launch_finished.is_set():
                raise RuntimeError("provider launch is still in progress during synchronous shutdown")
            if active.process is None and active.process_group_id is not None:
                raise RuntimeError("provider group has no exact child process")

        groups = tuple(
            active.process_group_id
            for active in active_runs
            if active.process_group_id is not None and self._process_group_exists(active.process_group_id)
        )
        for process_group_id in groups:
            with suppress(ProcessLookupError):
                os.killpg(process_group_id, signal.SIGTERM)
        remaining_children, remaining_groups = self._wait_for_synchronous_shutdown(
            active_runs,
            groups,
            time.monotonic() + self._stop_grace_seconds,
        )
        for process_group_id in remaining_groups:
            with suppress(ProcessLookupError):
                os.killpg(process_group_id, signal.SIGKILL)
        remaining_children, remaining_groups = self._wait_for_synchronous_shutdown(
            remaining_children,
            remaining_groups,
            time.monotonic() + _KILL_REAP_SECONDS,
        )
        if remaining_children:
            raise RuntimeError("owned provider child did not reap during shutdown")
        if remaining_groups:
            raise RuntimeError("owned provider group did not exit during shutdown")
        for active in active_runs:
            active.stop_completed = True

    def _wait_for_synchronous_shutdown(
        self,
        active_runs: Sequence[_ActiveRun],
        groups: Sequence[int],
        deadline: float,
    ) -> tuple[tuple[_ActiveRun, ...], tuple[int, ...]]:
        """Poll exact children before treating their process groups as gone."""
        remaining_children = tuple(
            active for active in active_runs if active.process is not None
        )
        remaining_groups = tuple(
            group for group in groups if self._process_group_exists(group)
        )
        while True:
            remaining_children = tuple(
                active
                for active in remaining_children
                if not self._poll_exact_process_synchronously(active)
            )
            remaining_groups = tuple(
                group for group in remaining_groups if self._process_group_exists(group)
            )
            if not remaining_children and not remaining_groups:
                return remaining_children, remaining_groups
            if time.monotonic() >= deadline:
                return remaining_children, remaining_groups
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _poll_exact_process_synchronously(active: _ActiveRun) -> bool:
        assert active.process is not None
        try:
            reaped, _ = os.waitpid(active.process.pid, os.WNOHANG)
        except ChildProcessError:
            # The previous asyncio child watcher has already reaped this exact
            # child while its loop was unwinding.
            active.process_exited.set()
            return True
        if reaped == active.process.pid:
            active.process_exited.set()
            return True
        return False

    async def _stop_active(
        self, active: _ActiveRun, *, deadline: float | None = None,
    ) -> None:
        if deadline is None:
            await active.stop_lock.acquire()
        elif not await self._acquire_stop_lock_until(active.stop_lock, deadline):
            return
        try:
            active.interrupted = True
            if active.stop_completed:
                return
            if active.process is None or active.process_group_id is None:
                return
            if not active.termination_started:
                active.termination_started = True
                try:
                    os.killpg(active.process_group_id, signal.SIGTERM)
                except ProcessLookupError:
                    if deadline is None:
                        await self._observe_process_exit(active, active.process)
                        active.stop_completed = True
                    return
            grace_deadline = asyncio.get_running_loop().time() + self._stop_grace_seconds
            if deadline is not None:
                grace_deadline = min(grace_deadline, deadline)
            if not active.kill_sent and not await self._wait_for_process_group_exit(
                active.process_group_id, deadline=grace_deadline,
            ):
                with suppress(ProcessLookupError):
                    os.killpg(active.process_group_id, signal.SIGKILL)
                    active.kill_sent = True
            if deadline is None and active.kill_sent and not await self._wait_for_process_group_exit(
                active.process_group_id,
                deadline=asyncio.get_running_loop().time() + _KILL_REAP_SECONDS,
            ):
                raise RuntimeError("owned provider group did not exit during shutdown")
            if deadline is None:
                await self._observe_process_exit(active, active.process)
                active.stop_completed = True
        finally:
            active.stop_lock.release()

    async def _wait_for_process_group_exit(
        self, process_group_id: int, *, deadline: float,
    ) -> bool:
        while self._process_group_exists(process_group_id):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(remaining, 0.01))
        return True

    async def _wait_for_event_until(
        self, event: asyncio.Event, deadline: float,
    ) -> bool:
        if event.is_set():
            return True
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        try:
            await asyncio.wait_for(event.wait(), timeout=remaining)
        except TimeoutError:
            return event.is_set()
        return True

    async def _acquire_stop_lock_until(self, lock: asyncio.Lock, deadline: float) -> bool:
        if not lock.locked():
            await lock.acquire()
            return True
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return False
        try:
            await asyncio.wait_for(lock.acquire(), timeout=remaining)
        except TimeoutError:
            return False
        return True

    def _remember_terminal_process_exit(self, run_id: str, process_exited: asyncio.Event) -> None:
        self._terminal_process_exits[run_id] = process_exited
        while len(self._terminal_process_exits) > MAX_TERMINAL_PROCESS_EXITS:
            self._terminal_process_exits.popitem(last=False)

    @staticmethod
    async def _observe_process_exit(
        active: _ActiveRun, process: asyncio.subprocess.Process,
    ) -> None:
        await process.wait()
        active.process_exited.set()

    @staticmethod
    def _validate_timeout(timeout_seconds: float) -> None:
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be a finite number >= 0")

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

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
