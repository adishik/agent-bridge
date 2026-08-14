from __future__ import annotations

import asyncio
from contextlib import suppress
import os
from pathlib import Path
import signal
import sys

import pytest

from agent_bridge import process as process_module
from agent_bridge.process import ProcessResult, ProcessRunner


async def _run_sleeper(
    runner: ProcessRunner, run_id: str, tmp_path: Path,
) -> ProcessResult:
    return await runner.run(
        run_id=run_id,
        argv=(sys.executable, "-c", "import time; time.sleep(60)"),
        cwd=tmp_path,
        env=dict(os.environ),
        stdin=None,
        on_line=lambda stream, line: None,
    )


async def _run_exit(runner: ProcessRunner, run_id: str, tmp_path: Path) -> ProcessResult:
    return await runner.run(
        run_id=run_id,
        argv=(sys.executable, "-c", "pass"),
        cwd=tmp_path,
        env=dict(os.environ),
        stdin=None,
        on_line=lambda stream, line: None,
    )


def test_runner_streams_stdout_and_stderr(tmp_path: Path) -> None:
    async def scenario() -> None:
        script = tmp_path / "emit.py"
        script.write_text(
            "import sys\nprint('one', flush=True)\n"
            "print('warning', file=sys.stderr, flush=True)\nprint('two', flush=True)\n"
        )
        seen: list[tuple[str, str]] = []
        result = await ProcessRunner().run(
            run_id="run-1",
            argv=(sys.executable, str(script)),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: seen.append((stream, line)),
        )

        assert result.exit_code == 0
        assert ("stdout", "one") in seen
        assert ("stderr", "warning") in seen

    asyncio.run(scenario())


def test_stop_targets_only_the_named_run(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.2)
        first = asyncio.create_task(_run_sleeper(runner, "first", tmp_path))
        second = asyncio.create_task(_run_sleeper(runner, "second", tmp_path))
        try:
            await runner.wait_until_started("first")
            await runner.wait_until_started("second")
            receipt = await runner.stop("first", timeout_seconds=1)
            assert receipt == process_module.StopReceipt("first", was_running=True, process_exited=True)
            assert (await first).interrupted is True
            assert runner.is_running("second") is True
        finally:
            for run_id in ("first", "second"):
                with suppress(KeyError):
                    await runner.stop(run_id, timeout_seconds=1)
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(scenario())


def test_stop_receipt_signals_only_the_registered_run_and_stale_ids_do_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run ID is the only stop authority; no PID/PGID fallback is possible."""
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        signals: list[tuple[int, signal.Signals | int]] = []
        original_killpg = process_module.os.killpg

        def record_killpg(process_group_id: int, sig: signal.Signals | int) -> None:
            signals.append((process_group_id, sig))
            original_killpg(process_group_id, sig)

        monkeypatch.setattr(process_module.os, "killpg", record_killpg)
        first = asyncio.create_task(_run_sleeper(runner, "owned-first", tmp_path))
        second = asyncio.create_task(runner.run(
            run_id="owned-second",
            argv=(sys.executable, "-c", "import time; time.sleep(0.08)"),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: None,
        ))
        try:
            await runner.wait_until_started("owned-first")
            await runner.wait_until_started("owned-second")
            stopped = await runner.stop("owned-first", timeout_seconds=1)
            stale = await runner.stop("999999", timeout_seconds=0)

            assert stopped == process_module.StopReceipt("owned-first", True, True)
            assert stale == process_module.StopReceipt("999999", False, True)
            first_result = await first
            assert first_result.interrupted is True
            assert (await second).interrupted is False
            directed = [(group, sig) for group, sig in signals if sig != 0]
            assert directed == [(first_result.process_group_id, signal.SIGTERM)]
            with pytest.raises(ValueError, match="run_id"):
                await runner.stop(first_result.pid, timeout_seconds=0)  # type: ignore[arg-type]
            assert [(group, sig) for group, sig in signals if sig != 0] == directed
        finally:
            await runner.stop("owned-first", timeout_seconds=1)
            await runner.stop("owned-second", timeout_seconds=1)
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(scenario())


def test_stop_is_single_flight_and_wait_process_exit_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated Stop cannot duplicate signals, and exact exit observation is bounded."""
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        signals: list[tuple[int, signal.Signals | int]] = []
        original_killpg = process_module.os.killpg

        def record_killpg(process_group_id: int, sig: signal.Signals | int) -> None:
            signals.append((process_group_id, sig))
            original_killpg(process_group_id, sig)

        monkeypatch.setattr(process_module.os, "killpg", record_killpg)
        task = asyncio.create_task(_run_sleeper(runner, "single-flight", tmp_path))
        try:
            await runner.wait_until_started("single-flight")
            with pytest.raises(TimeoutError):
                await runner.wait_process_exit("single-flight", timeout_seconds=0.001)
            first, second = await asyncio.gather(
                runner.stop("single-flight", timeout_seconds=1),
                runner.stop("single-flight", timeout_seconds=1),
            )
            assert first == process_module.StopReceipt("single-flight", True, True)
            assert second == process_module.StopReceipt("single-flight", True, True)
            result = await task
            assert result.interrupted is True
            await runner.wait_process_exit("single-flight", timeout_seconds=0)
            assert [sig for _, sig in signals if sig != 0] == [signal.SIGTERM]
            assert runner.is_running("single-flight") is False
        finally:
            await runner.stop("single-flight", timeout_seconds=1)
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_stop_timeout_receipt_records_escalation_without_claiming_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero wait budget reports uncertainty after directing TERM then KILL."""
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.2)
        signals: list[signal.Signals | int] = []
        original_killpg = process_module.os.killpg

        def record_killpg(process_group_id: int, sig: signal.Signals | int) -> None:
            signals.append(sig)
            original_killpg(process_group_id, sig)

        monkeypatch.setattr(process_module.os, "killpg", record_killpg)
        task = asyncio.create_task(runner.run(
            run_id="timeout-escalation",
            argv=(
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(60)\n",
            ),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: None,
        ))
        try:
            await runner.wait_until_started("timeout-escalation")
            receipt = await runner.stop("timeout-escalation", timeout_seconds=0)
            assert receipt == process_module.StopReceipt("timeout-escalation", True, False)
            assert [sig for sig in signals if sig != 0] == [signal.SIGTERM, signal.SIGKILL]
            assert (await task).interrupted is True
            assert runner.is_running("timeout-escalation") is False
        finally:
            await runner.stop("timeout-escalation", timeout_seconds=1)
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_stop_launch_wait_uses_its_timeout_and_preserves_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registration barrier cannot extend Stop's deadline or lose its exact intent."""
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.2)
        original_create = process_module.asyncio.create_subprocess_exec
        spawned: asyncio.subprocess.Process | None = None
        registered = asyncio.Event()
        release_registration = asyncio.Event()

        async def held_registration(
            *args: object, **kwargs: object,
        ) -> asyncio.subprocess.Process:
            nonlocal spawned
            spawned = await original_create(*args, **kwargs)  # type: ignore[arg-type]
            registered.set()
            await release_registration.wait()
            return spawned

        monkeypatch.setattr(
            process_module.asyncio, "create_subprocess_exec", held_registration,
        )
        task = asyncio.create_task(_run_sleeper(runner, "held-registration", tmp_path))
        try:
            await registered.wait()
            started = asyncio.get_running_loop().time()
            receipt = await asyncio.wait_for(
                runner.stop("held-registration", timeout_seconds=0.03), timeout=0.12,
            )
            elapsed = asyncio.get_running_loop().time() - started

            assert receipt == process_module.StopReceipt("held-registration", True, False)
            assert elapsed < 0.075
            release_registration.set()
            assert (await asyncio.wait_for(task, timeout=1)).interrupted is True
        finally:
            release_registration.set()
            if not task.done():
                await runner.stop("held-registration", timeout_seconds=1)
            await asyncio.gather(task, return_exceptions=True)
            if spawned is not None:
                await asyncio.wait_for(spawned.wait(), timeout=1)

    asyncio.run(scenario())


def test_stop_uses_one_deadline_across_grace_escalation_and_exit_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed exact observer cannot make one Stop spend its budget repeatedly."""
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.2)
        release_observer = asyncio.Event()

        async def delayed_observe(
            active: object, process: asyncio.subprocess.Process,
        ) -> None:
            await process.wait()
            await release_observer.wait()
            active.process_exited.set()  # type: ignore[attr-defined]

        monkeypatch.setattr(
            ProcessRunner, "_observe_process_exit", staticmethod(delayed_observe),
        )
        task = asyncio.create_task(runner.run(
            run_id="one-deadline",
            argv=(
                sys.executable,
                "-c",
                "import signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(60)\n",
            ),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: None,
        ))
        try:
            await runner.wait_until_started("one-deadline")
            started = asyncio.get_running_loop().time()
            receipt = await runner.stop("one-deadline", timeout_seconds=0.03)
            elapsed = asyncio.get_running_loop().time() - started

            assert receipt == process_module.StopReceipt("one-deadline", True, False)
            assert elapsed < 0.075
        finally:
            release_observer.set()
            if not task.done():
                await runner.stop("one-deadline", timeout_seconds=1)
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_wait_process_exit_retains_known_completion_and_bounds_terminal_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exited runs remain exactly observable until the capped terminal record evicts them."""
    async def scenario() -> None:
        monkeypatch.setattr(process_module, "MAX_TERMINAL_PROCESS_EXITS", 2)
        runner = ProcessRunner()
        await _run_exit(runner, "completed-one", tmp_path)
        await _run_exit(runner, "completed-two", tmp_path)

        await runner.wait_process_exit("completed-one", timeout_seconds=0)
        await runner.wait_process_exit("completed-two", timeout_seconds=0)
        with pytest.raises(ValueError, match="retained"):
            await _run_exit(runner, "completed-one", tmp_path)
        with pytest.raises(KeyError):
            await runner.wait_process_exit("never-known", timeout_seconds=0)

        await _run_exit(runner, "completed-three", tmp_path)
        with pytest.raises(KeyError):
            await runner.wait_process_exit("completed-one", timeout_seconds=0)
        await runner.wait_process_exit("completed-two", timeout_seconds=0)
        await runner.wait_process_exit("completed-three", timeout_seconds=0)
        assert (await _run_exit(runner, "completed-one", tmp_path)).exit_code == 0

    asyncio.run(scenario())


def test_stop_kills_owned_descendant_after_direct_child_exits(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        ready = asyncio.Event()
        task = asyncio.create_task(runner.run(
            run_id="run-with-descendant",
            argv=(
                sys.executable,
                "-c",
                "import os, signal, sys, time\n"
                "if os.fork() == 0:\n"
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "    time.sleep(0.2)\n"
                "else:\n"
                "    print('ready', flush=True)\n",
            ),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: ready.set() if (stream, line) == ("stdout", "ready") else None,
        ))
        try:
            await runner.wait_until_started("run-with-descendant")
            await ready.wait()
            await asyncio.sleep(0.02)
            assert task.done() is False
            assert runner.is_running("run-with-descendant") is True
            receipt = await runner.stop("run-with-descendant", timeout_seconds=1)
            assert receipt.process_exited is True
            assert (await task).interrupted is True
        finally:
            with suppress(KeyError):
                await runner.stop("run-with-descendant", timeout_seconds=1)
            await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), timeout=1,
            )

    asyncio.run(scenario())


def test_cancelling_during_launch_reaps_the_spawned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        original_create = process_module.asyncio.create_subprocess_exec
        spawned: asyncio.subprocess.Process | None = None
        created = asyncio.Event()
        release = asyncio.Event()

        async def delayed_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            nonlocal spawned
            spawned = await original_create(*args, **kwargs)  # type: ignore[arg-type]
            created.set()
            await release.wait()
            return spawned

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", delayed_create)
        task = asyncio.create_task(runner.run(
            run_id="cancelled-launch",
            argv=(sys.executable, "-c", "import time; time.sleep(0.2)"),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: None,
        ))
        try:
            await created.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert spawned is not None
            assert spawned.returncode is not None
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            if spawned is not None:
                await asyncio.wait_for(spawned.wait(), timeout=1)

    asyncio.run(scenario())


def test_stop_during_real_child_launch_waits_and_terminates_the_exact_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        original_create = process_module.asyncio.create_subprocess_exec
        launch_entered = asyncio.Event()
        release_launch = asyncio.Event()
        spawned: asyncio.subprocess.Process | None = None

        async def gated_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            nonlocal spawned
            spawned = await original_create(*args, **kwargs)  # type: ignore[arg-type]
            launch_entered.set()
            await release_launch.wait()
            return spawned

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", gated_create)
        run_task = asyncio.create_task(_run_sleeper(runner, "launch-race", tmp_path))
        stop_task: asyncio.Task[StopReceipt] | None = None
        try:
            await launch_entered.wait()
            stop_task = asyncio.create_task(runner.stop("launch-race", timeout_seconds=1))
            await asyncio.sleep(0)
            returned_while_launch_pending = stop_task.done()
            release_launch.set()
            stop_result = await asyncio.gather(stop_task, return_exceptions=True)
            if stop_result[0] is not None:
                await runner.wait_until_started("launch-race")
                await runner.stop("launch-race", timeout_seconds=1)
            result = await asyncio.wait_for(run_task, timeout=1)

            assert returned_while_launch_pending is False
            assert stop_result == [
                process_module.StopReceipt("launch-race", was_running=True, process_exited=True)
            ]
            assert result.interrupted is True
            assert spawned is not None
            assert spawned.returncode is not None
        finally:
            release_launch.set()
            with suppress(KeyError):
                await runner.stop("launch-race", timeout_seconds=1)
            await asyncio.wait_for(
                asyncio.gather(run_task, return_exceptions=True), timeout=1,
            )
            if stop_task is not None:
                await asyncio.gather(stop_task, return_exceptions=True)

    asyncio.run(scenario())


def test_cancelling_after_leader_exit_stops_its_owned_descendant(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        ready = asyncio.Event()
        escaped = tmp_path / "descendant-escaped.txt"
        task = asyncio.create_task(runner.run(
            run_id="cancelled-descendant",
            argv=(
                sys.executable,
                "-c",
                "import os, signal, time\n"
                "if os.fork() == 0:\n"
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "    time.sleep(0.2)\n"
                f"    open({str(escaped)!r}, 'w').write('escaped')\n"
                "else:\n"
                "    print('ready', flush=True)\n",
            ),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: ready.set() if (stream, line) == ("stdout", "ready") else None,
        ))
        try:
            await runner.wait_until_started("cancelled-descendant")
            await ready.wait()
            await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.25)
            assert escaped.exists() is False
        finally:
            with suppress(KeyError):
                await runner.stop("cancelled-descendant", timeout_seconds=1)
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

    asyncio.run(scenario())


def test_stop_waits_the_configured_grace_for_a_live_descendant(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.08)
        ready = asyncio.Event()
        task = asyncio.create_task(runner.run(
            run_id="grace-descendant",
            argv=(
                sys.executable,
                "-c",
                "import os, signal, time\n"
                "if os.fork() == 0:\n"
                "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "    time.sleep(0.3)\n"
                "else:\n"
                "    print('ready', flush=True)\n",
            ),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: ready.set() if (stream, line) == ("stdout", "ready") else None,
        ))
        try:
            await runner.wait_until_started("grace-descendant")
            await ready.wait()
            await asyncio.sleep(0.02)
            started = asyncio.get_running_loop().time()
            await runner.stop("grace-descendant", timeout_seconds=1)
            assert asyncio.get_running_loop().time() - started >= 0.06
            assert (await task).interrupted is True
        finally:
            with suppress(KeyError):
                await runner.stop("grace-descendant", timeout_seconds=1)
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

    asyncio.run(scenario())


def test_failed_launch_normalization_does_not_reserve_the_run_id(tmp_path: Path) -> None:
    class InvalidCwd:
        def __fspath__(self) -> str:
            raise TypeError("invalid cwd")

    async def scenario() -> None:
        runner = ProcessRunner()
        with pytest.raises(TypeError, match="invalid cwd"):
            await runner.run(
                run_id="reusable",
                argv=(sys.executable, "-c", "pass"),
                cwd=InvalidCwd(),
                env=dict(os.environ),
                stdin=None,
                on_line=lambda stream, line: None,
            )
        result = await runner.run(
            run_id="reusable",
            argv=(sys.executable, "-c", "pass"),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: None,
        )
        assert result.exit_code == 0

    asyncio.run(scenario())


def test_runner_preserves_line_boundaries_and_replaces_invalid_utf8(tmp_path: Path) -> None:
    async def scenario() -> None:
        seen: list[tuple[str, str]] = []
        result = await ProcessRunner().run(
            run_id="line-contract",
            argv=(
                sys.executable,
                "-c",
                "import sys\n"
                "sys.stdout.buffer.write(b' first  \\r\\nsecond\\ninvalid\\xff\\nfinal')\n"
                "sys.stderr.buffer.write(b'warning\\r\\nlast')\n"
                "sys.stdout.flush(); sys.stderr.flush()\n",
            ),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: seen.append((stream, line)),
        )

        assert result.stdout == (" first  ", "second", "invalid�", "final")
        assert result.stderr == ("warning", "last")
        assert [line for stream, line in seen if stream == "stdout"] == list(result.stdout)
        assert [line for stream, line in seen if stream == "stderr"] == list(result.stderr)

    asyncio.run(scenario())


def test_runner_drains_both_streams_under_simultaneous_volume(tmp_path: Path) -> None:
    async def scenario() -> None:
        result = await ProcessRunner().run(
            run_id="volume",
            argv=(
                sys.executable,
                "-c",
                "import sys\n"
                "for index in range(2000):\n"
                "    print(f'out-{index:04d}-xxxxxxxxxxxxxxxxxxxxxxxx', flush=True)\n"
                "    print(f'err-{index:04d}-xxxxxxxxxxxxxxxxxxxxxxxx', file=sys.stderr, flush=True)\n",
            ),
            cwd=tmp_path,
            env=dict(os.environ),
            stdin=None,
            on_line=lambda stream, line: None,
        )

        assert len(result.stdout) == 2000
        assert len(result.stderr) == 2000
        assert result.stdout[0] == "out-0000-xxxxxxxxxxxxxxxxxxxxxxxx"
        assert result.stderr[-1] == "err-1999-xxxxxxxxxxxxxxxxxxxxxxxx"

    asyncio.run(scenario())


def test_runner_terminates_its_exact_process_group_when_output_exceeds_the_bound(
    tmp_path: Path,
) -> None:
    """Overflow must be a fixed failure and must not leave a TERM-ignoring child."""
    async def scenario() -> None:
        escaped = tmp_path / "descendant-escaped.txt"
        runner = ProcessRunner(stop_grace_seconds=0.02)
        with pytest.raises(RuntimeError, match="output") as raised:
            await runner.run(
                run_id="output-overflow",
                argv=(
                    sys.executable,
                    "-c",
                    "import os, signal, time\n"
                    "if os.fork() == 0:\n"
                    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "    time.sleep(0.3)\n"
                    f"    open({str(escaped)!r}, 'w').write('escaped')\n"
                    "else:\n"
                    "    for index in range(5000):\n"
                    "        print('SECRET_OUTPUT_' + str(index), flush=True)\n",
                ),
                cwd=tmp_path,
                env=dict(os.environ),
                stdin=None,
                on_line=lambda stream, line: None,
            )
        await asyncio.sleep(0.35)
        assert "SECRET_OUTPUT" not in str(raised.value)
        assert escaped.exists() is False
        assert runner.is_running("output-overflow") is False

    asyncio.run(scenario())
