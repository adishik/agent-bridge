from __future__ import annotations

import asyncio
from contextlib import suppress
import os
from pathlib import Path
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
            await runner.stop("first")
            assert (await first).interrupted is True
            assert runner.is_running("second") is True
        finally:
            for run_id in ("first", "second"):
                with suppress(KeyError):
                    await runner.stop(run_id)
            await asyncio.gather(first, second, return_exceptions=True)

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
            await runner.stop("run-with-descendant")
            assert (await task).interrupted is True
        finally:
            with suppress(KeyError):
                await runner.stop("run-with-descendant")
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
            launch_entered.set()
            await release_launch.wait()
            spawned = await original_create(*args, **kwargs)  # type: ignore[arg-type]
            return spawned

        monkeypatch.setattr(process_module.asyncio, "create_subprocess_exec", gated_create)
        run_task = asyncio.create_task(_run_sleeper(runner, "launch-race", tmp_path))
        stop_task: asyncio.Task[None] | None = None
        try:
            await launch_entered.wait()
            stop_task = asyncio.create_task(runner.stop("launch-race"))
            await asyncio.sleep(0)
            returned_while_launch_pending = stop_task.done()
            release_launch.set()
            stop_result = await asyncio.gather(stop_task, return_exceptions=True)
            if stop_result[0] is not None:
                await runner.wait_until_started("launch-race")
                await runner.stop("launch-race")
            result = await asyncio.wait_for(run_task, timeout=1)

            assert returned_while_launch_pending is False
            assert stop_result == [None]
            assert result.interrupted is True
            assert spawned is not None
            assert spawned.returncode is not None
        finally:
            release_launch.set()
            with suppress(KeyError):
                await runner.stop("launch-race")
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
                await runner.stop("cancelled-descendant")
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
            await runner.stop("grace-descendant")
            assert asyncio.get_running_loop().time() - started >= 0.06
            assert (await task).interrupted is True
        finally:
            with suppress(KeyError):
                await runner.stop("grace-descendant")
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
