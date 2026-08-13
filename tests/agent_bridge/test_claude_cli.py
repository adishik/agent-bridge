from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_bridge.adapters.claude_cli import (
    METERED_ENV_KEYS,
    METERED_ENV_PREFIXES,
    ClaudeCLI,
    ClaudeRunError,
    SubscriptionAuthError,
)
from agent_bridge.contracts import FableClarification, ReviewVerdict
from agent_bridge.process import ProcessResult, ProcessRunner


SAFE_ENV = {"AGENT_BRIDGE_TEST_FAKE": "1", "LANG": "C.UTF-8", "PATH": "/not-used"}
SECRET_SENTINELS = (
    "SECRET_EMAIL_SENTINEL",
    "SECRET_ORGANIZATION_SENTINEL",
    "SECRET_CREDENTIAL_SENTINEL",
    "SECRET_ASSISTANT_TEXT_SENTINEL",
    "SECRET_USER_TEXT_SENTINEL",
    "SECRET_STREAM_TEXT_SENTINEL",
    "SECRET_UNKNOWN_EVENT_SENTINEL",
    "SECRET_NONSTRING_TYPE_SENTINEL",
    "SECRET_RESULT_FIELD_SENTINEL",
    "SECRET_STDERR_SENTINEL",
)


def _adapter(fake_claude: Path, tmp_path: Path, **extra_env: str) -> ClaudeCLI:
    return ClaudeCLI(
        fake_claude,
        ProcessRunner(stop_grace_seconds=0.05),
        env={**SAFE_ENV, **extra_env},
        cwd=tmp_path,
    )


@pytest.mark.parametrize("executable", ("claude", "bin/claude"))
def test_fable_rejects_bare_and_relative_executable_names(
    executable: str, tmp_path: Path,
) -> None:
    with pytest.raises(ValueError) as raised:
        ClaudeCLI(executable, ProcessRunner(), env=SAFE_ENV, cwd=tmp_path)
    assert str(raised.value) == "executable must be an absolute path"


@pytest.mark.parametrize("kind", ("directory", "non_executable"))
def test_fable_requires_an_absolute_executable_file(kind: str, tmp_path: Path) -> None:
    executable = tmp_path / "candidate"
    if kind == "directory":
        executable.mkdir()
    else:
        executable.write_text("not executable", encoding="utf-8")
        executable.chmod(0o600)

    with pytest.raises(ValueError) as raised:
        ClaudeCLI(executable, ProcessRunner(), env=SAFE_ENV, cwd=tmp_path)
    assert str(raised.value) == "executable must be an executable file"


async def _wait_for_model_invocation(tmp_path: Path) -> None:
    deadline = asyncio.get_running_loop().time() + 1
    while True:
        argv_path = tmp_path / "captured-argv.json"
        if argv_path.exists() and "--safe-mode" in json.loads(argv_path.read_text()):
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("fake Claude model invocation did not start")
        await asyncio.sleep(0.005)


async def _wait_for_fake_signal(path: Path) -> None:
    deadline = asyncio.get_running_loop().time() + 1
    while not path.exists():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"fake Claude signal did not appear: {path.name}")
        await asyncio.sleep(0.005)


def _assert_no_secret_sentinel(value: object) -> None:
    representation = repr(value)
    for sentinel in SECRET_SENTINELS:
        assert sentinel not in representation


def test_fable_requires_subscription_and_sanitizes_environment(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        env = {
            **SAFE_ENV,
            "ANTHROPIC_API_KEY": "must-not-pass",
            "ANTHROPIC_AUTH_TOKEN": "must-not-pass",
            "ANTHROPIC_BASE_URL": "https://metered.invalid",
            "ANTHROPIC_FUTURE_PROVIDER_KEY": "must-not-pass",
            "CLAUDE_CODE_OAUTH_TOKEN": "must-not-pass",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "CLAUDE_CODE_USE_FUTURE_PROVIDER": "1",
            "AWS_BEARER_TOKEN_BEDROCK": "must-not-pass",
            "GOOGLE_APPLICATION_CREDENTIALS": "/metered/credentials.json",
        }
        adapter = ClaudeCLI(fake_claude, ProcessRunner(), env=env, cwd=tmp_path)
        status = await adapter.preflight()

        assert status.logged_in is True
        assert status.auth_method == "claude.ai"
        assert status.api_provider == "firstParty"
        assert status.subscription_type == "max"
        captured = json.loads((tmp_path / "captured-env.json").read_text())
        assert METERED_ENV_PREFIXES == ("ANTHROPIC_", "CLAUDE_CODE_USE_")
        assert METERED_ENV_KEYS == frozenset({
            "CLAUDE_CODE_OAUTH_TOKEN",
            "AWS_BEARER_TOKEN_BEDROCK",
            "GOOGLE_APPLICATION_CREDENTIALS",
        })
        assert not ({
            "CLAUDE_CODE_OAUTH_TOKEN",
            "AWS_BEARER_TOKEN_BEDROCK",
            "GOOGLE_APPLICATION_CREDENTIALS",
        } & set(captured))
        assert not any(
            key.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_")) for key in captured
        )
        assert not (set(captured) & METERED_ENV_KEYS)
        assert not any(key.startswith(METERED_ENV_PREFIXES) for key in captured)
        assert captured["AGENT_BRIDGE_TEST_FAKE"] == "1"

    asyncio.run(scenario())


def test_fable_accepts_pretty_printed_subscription_auth_status(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        auth_status = {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "max",
        }
        status = await _adapter(
            fake_claude,
            tmp_path,
            FAKE_CLAUDE_AUTH_STATUS=json.dumps(auth_status, indent=2),
        ).preflight()

        assert status.logged_in is True
        assert status.auth_method == "claude.ai"
        assert status.api_provider == "firstParty"
        assert status.subscription_type == "max"

    asyncio.run(scenario())


def test_fable_rejects_multiple_auth_json_documents(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        auth_status = json.dumps({
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "max",
        })
        with pytest.raises(SubscriptionAuthError, match="could not be verified"):
            await _adapter(
                fake_claude,
                tmp_path,
                FAKE_CLAUDE_AUTH_STATUS=f"{auth_status}\n{auth_status}",
            ).preflight()

    asyncio.run(scenario())


@pytest.mark.parametrize("stdout", ((), ("[]",), ("null",)))
def test_fable_rejects_empty_or_non_object_auth_document(
    fake_claude: Path, tmp_path: Path, stdout: tuple[str, ...],
) -> None:
    result = ProcessResult(
        run_id="auth-result",
        pid=123,
        process_group_id=123,
        exit_code=0,
        stdout=stdout,
        stderr=(),
        interrupted=False,
    )

    with pytest.raises(SubscriptionAuthError, match="could not be verified"):
        _adapter(fake_claude, tmp_path)._parse_auth_status(result)


@pytest.mark.parametrize(
    "auth_status",
    [
        {"loggedIn": True, "authMethod": method, "apiProvider": "firstParty", "subscriptionType": "max"}
        for method in ("api_key", "console", "bedrock", "vertex", "foundry", "gateway")
    ] + [
        {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty"},
        {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "   "},
        {"loggedIn": False, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "max"},
        {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "gateway", "subscriptionType": "max"},
        {"loggedIn": 1, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "max"},
        {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": 1},
    ],
)
def test_fable_rejects_every_non_subscription_auth_shape_without_model_call(
    fake_claude: Path, tmp_path: Path, auth_status: dict[str, object],
) -> None:
    async def scenario() -> None:
        adapter = _adapter(
            fake_claude,
            tmp_path,
            FAKE_CLAUDE_AUTH_STATUS=json.dumps({
                **auth_status,
                "email": "identity@example.invalid",
                "organization": "secret-organization",
            }),
        )
        with pytest.raises(SubscriptionAuthError) as raised:
            await adapter.plan(
                run_id="run-rejected",
                task_id="task-1",
                prompt="Plan it",
                context="AGENTS rules and repository context",
            )
        assert "identity@example.invalid" not in str(raised.value)
        assert "secret-organization" not in str(raised.value)
        assert json.loads((tmp_path / "captured-argv.json").read_text()) == [
            "auth", "status", "--json",
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("extra_env", "error_match"),
    [
        ({"FAKE_CLAUDE_AUTH_STATUS": "malformed"}, "subscription"),
        ({"FAKE_CLAUDE_AUTH_EXIT": "9"}, "subscription"),
    ],
)
def test_fable_fails_closed_on_unusable_auth_status(
    fake_claude: Path,
    tmp_path: Path,
    extra_env: dict[str, str],
    error_match: str,
) -> None:
    async def scenario() -> None:
        with pytest.raises(SubscriptionAuthError, match=error_match):
            await _adapter(fake_claude, tmp_path, **extra_env).preflight()

    asyncio.run(scenario())


def test_fable_argv_is_safe_read_only_and_not_bare(fake_claude: Path, tmp_path: Path) -> None:
    async def scenario() -> None:
        await _adapter(fake_claude, tmp_path).plan(
            run_id="run-1",
            task_id="task-1",
            prompt="Plan it",
            context="AGENTS rules",
        )
        argv = json.loads((tmp_path / "captured-argv.json").read_text())
        assert argv[:3] == ["--safe-mode", "-p", "--model"]
        assert ["--model", "fable"] == argv[argv.index("--model"):argv.index("--model") + 2]
        assert ["--permission-mode", "plan"] == argv[
            argv.index("--permission-mode"):argv.index("--permission-mode") + 2
        ]
        assert ["--tools", "Read,Glob,Grep"] == argv[
            argv.index("--tools"):argv.index("--tools") + 2
        ]
        assert "--no-chrome" in argv
        assert ["--output-format", "stream-json"] == argv[
            argv.index("--output-format"):argv.index("--output-format") + 2
        ]
        assert "--verbose" in argv
        assert "--include-partial-messages" in argv
        assert "--bare" not in argv
        assert "Bash" not in argv and "Edit" not in argv and "Write" not in argv

    asyncio.run(scenario())


def test_plan_uses_same_sanitized_environment_and_returns_normalized_contract(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        adapter = _adapter(
            fake_claude,
            tmp_path,
            ANTHROPIC_API_KEY="must-not-pass",
            CLAUDE_CODE_USE_BEDROCK="1",
            FAKE_CLAUDE_SECRET_OUTPUT="1",
        )
        result = await adapter.plan(
            run_id="run-1",
            task_id="task-1",
            prompt="Plan it",
            context="AGENTS.md:\nRead-only rules.\nRepository context:\nbranch feat/agent-bridge",
        )

        assert result.run_id == "run-1"
        assert result.cli_session_id == "fable-session-1"
        assert result.payload is not None
        assert result.payload["task_id"] == "task-1"
        assert result.interrupted is False
        assert tuple(dict(event) for event in result.events) == (
            {"type": "system", "subtype": "init", "session_id": "fable-session-1"},
            {"type": "assistant"},
            {"type": "user"},
            {"type": "stream_event"},
            {"type": "result", "has_structured_output": True},
        )
        assert result.stderr == ("stderr_lines=1",)
        _assert_no_secret_sentinel(result)
        history = json.loads((tmp_path / "captured-env-history.json").read_text())
        assert len(history) == 2
        assert history[0] == history[1]
        assert not (set(history[1]) & METERED_ENV_KEYS)
        assert not any(key.startswith(METERED_ENV_PREFIXES) for key in history[1])
        prompt = json.loads((tmp_path / "captured-argv.json").read_text())[-1]
        assert "TaskBrief" in prompt
        assert "task-1" in prompt
        assert "AGENTS.md" in prompt
        assert "branch feat/agent-bridge" in prompt
        assert "Only JSON" in prompt

    asyncio.run(scenario())


def test_clarify_and_review_resume_the_exact_session_with_their_contracts(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        adapter = _adapter(fake_claude, tmp_path)
        clarification = await adapter.clarify(
            run_id="clarify-1",
            session_id="session-exact:/[]",
            prompt="Resolve Sol's question for task-1.",
        )
        clarify_argv = json.loads((tmp_path / "captured-argv.json").read_text())
        assert clarify_argv[-3] == "--resume"
        assert clarify_argv[-2] == "session-exact:/[]"
        assert "FableClarification" in clarify_argv[-1]
        assert clarification.payload is not None
        assert clarification.payload["status"] == "answered"

        verdict = await adapter.review(
            run_id="review-1",
            session_id="session-exact:/[]",
            prompt="Review evidence for task-1.",
        )
        review_argv = json.loads((tmp_path / "captured-argv.json").read_text())
        assert review_argv[-3:-1] == ["--resume", "session-exact:/[]"]
        assert "ReviewVerdict" in review_argv[-1]
        assert verdict.payload is not None
        assert verdict.payload["status"] == "approved"

    asyncio.run(scenario())


def test_interrupted_plan_resume_uses_exact_session_and_task_brief_contract(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        adapter = _adapter(fake_claude, tmp_path)
        result = await adapter.resume_plan(
            run_id="run-resume-plan",
            session_id="fable-session-exact",
            task_id="task-1",
            prompt="Plan it",
            context="Binding AGENTS rules.",
        )

        assert result.cli_session_id == "fable-session-exact"
        assert result.payload is not None
        assert result.payload["task_id"] == "task-1"
        argv = json.loads((tmp_path / "captured-argv.json").read_text())
        assert argv[-3:-1] == ["--resume", "fable-session-exact"]
        assert "TaskBrief" in argv[-1]
        assert "task-1" in argv[-1]
        assert "Binding AGENTS rules" in argv[-1]

    asyncio.run(scenario())


def test_plan_rejects_a_task_id_other_than_the_coordinator_generated_id(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with pytest.raises(ClaudeRunError, match="task ID"):
            await _adapter(
                fake_claude, tmp_path, FAKE_CLAUDE_TASK_ID="different-task",
            ).plan(
                run_id="run-wrong-task",
                task_id="task-1",
                prompt="Plan it",
                context="AGENTS.md and repository context",
            )

    asyncio.run(scenario())


def test_resume_rejects_a_different_session_reported_by_claude(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with pytest.raises(ClaudeRunError, match="session"):
            await _adapter(
                fake_claude, tmp_path, FAKE_CLAUDE_SESSION_ID="different-session",
            ).clarify(
                run_id="run-wrong-session",
                session_id="stored-session",
                prompt="Resolve the question for task-1.",
            )

    asyncio.run(scenario())


def test_legacy_clarification_keeps_its_existing_first_session_projection(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        result = await _adapter(
            fake_claude,
            tmp_path,
            FAKE_CLAUDE_MODE="conflicting_session",
        ).clarify(
            run_id="legacy-conflicting-session",
            session_id="stored-session",
            prompt="Resolve the existing clarification.",
        )

        assert result.cli_session_id == "stored-session"
        assert result.payload is not None
        assert result.payload["status"] == "answered"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("missing_init", "init"),
        ("missing_result", "result"),
        ("invalid_payload", "schema"),
        ("malformed_json", "JSON"),
    ],
)
def test_completed_fable_run_rejects_incomplete_or_invalid_output(
    fake_claude: Path, tmp_path: Path, mode: str, match: str,
) -> None:
    async def scenario() -> None:
        with pytest.raises(ClaudeRunError, match=match) as raised:
            await _adapter(fake_claude, tmp_path, FAKE_CLAUDE_MODE=mode).plan(
                run_id=f"run-{mode}",
                task_id="task-1",
                prompt="Plan it",
                context="AGENTS.md and repository context",
            )
        result = raised.value.result
        assert result is not None
        assert result.run_id == f"run-{mode}"
        assert result.payload is None
        assert result.interrupted is False
        assert all("type" in event for event in result.events)

    asyncio.run(scenario())


def test_nonzero_fable_run_error_retains_audit_events_and_summarizes_stderr(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with pytest.raises(ClaudeRunError, match="exit") as raised:
            await _adapter(
                fake_claude,
                tmp_path,
                FAKE_CLAUDE_MODE="nonzero",
                FAKE_CLAUDE_SECRET_OUTPUT="1",
            ).plan(
                run_id="run-nonzero",
                task_id="task-1",
                prompt="Plan it",
                context="AGENTS.md and repository context",
            )

        result = raised.value.result
        assert result is not None
        assert result.run_id == "run-nonzero"
        assert result.exit_code == 7
        assert result.payload is None
        assert tuple(dict(event) for event in result.events) == (
            {"type": "system", "subtype": "init", "session_id": "fable-session-1"},
            {"type": "assistant"},
            {"type": "user"},
            {"type": "stream_event"},
            {"type": "result", "has_structured_output": True},
        )
        assert result.stderr == ("stderr_lines=2",)
        assert "controlled model failure" not in str(raised.value)
        _assert_no_secret_sentinel(raised.value)
        _assert_no_secret_sentinel(result)

    asyncio.run(scenario())


def test_interrupted_fable_run_preserves_partial_events_and_session(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        adapter = ClaudeCLI(
            fake_claude,
            runner,
            env={
                **SAFE_ENV,
                "FAKE_CLAUDE_MODE": "slow_after_init",
                "FAKE_CLAUDE_SECRET_OUTPUT": "1",
            },
            cwd=tmp_path,
        )
        run = asyncio.create_task(adapter.plan(
            run_id="run-interrupted",
            task_id="task-1",
            prompt="Plan it",
            context="AGENTS.md and repository context",
        ))
        await _wait_for_model_invocation(tmp_path)
        await _wait_for_fake_signal(tmp_path / "fake-claude-partials-ready.json")
        await runner.stop("run-interrupted")
        result = await run

        assert result.interrupted is True
        assert result.cli_session_id == "fable-session-1"
        assert result.payload is None
        assert tuple(dict(event) for event in result.events) == (
            {"type": "system", "subtype": "init", "session_id": "fable-session-1"},
            {"type": "assistant"},
            {"type": "user"},
            {"type": "stream_event"},
        )
        assert result.stderr == ("stderr_lines=1",)
        _assert_no_secret_sentinel(result)

    asyncio.run(scenario())


def test_interrupted_fable_run_before_init_has_no_session_or_payload(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        adapter = ClaudeCLI(
            fake_claude,
            runner,
            env={**SAFE_ENV, "FAKE_CLAUDE_MODE": "slow_before_init"},
            cwd=tmp_path,
        )
        run = asyncio.create_task(adapter.plan(
            run_id="run-before-init",
            task_id="task-1",
            prompt="Plan it",
            context="AGENTS.md and repository context",
        ))
        await _wait_for_model_invocation(tmp_path)
        await runner.stop("run-before-init")
        result = await run

        assert result.interrupted is True
        assert result.cli_session_id is None
        assert result.payload is None
        assert result.events == ()

    asyncio.run(scenario())


def test_interrupted_resume_rejects_an_observed_different_session(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        adapter = ClaudeCLI(
            fake_claude,
            runner,
            env={
                **SAFE_ENV,
                "FAKE_CLAUDE_MODE": "slow_after_init",
                "FAKE_CLAUDE_SESSION_ID": "different-session",
                "FAKE_CLAUDE_SECRET_OUTPUT": "1",
            },
            cwd=tmp_path,
        )
        run = asyncio.create_task(adapter.clarify(
            run_id="run-interrupted-mismatch",
            session_id="stored-session",
            prompt="Resolve the question for task-1.",
        ))
        await _wait_for_model_invocation(tmp_path)
        await _wait_for_fake_signal(tmp_path / "fake-claude-partials-ready.json")
        await runner.stop("run-interrupted-mismatch")

        with pytest.raises(ClaudeRunError, match="different session") as raised:
            await run
        result = raised.value.result
        assert result is not None
        assert result.interrupted is True
        assert result.cli_session_id == "different-session"
        assert result.payload is None
        assert result.stderr == ("stderr_lines=1",)
        _assert_no_secret_sentinel(raised.value)
        _assert_no_secret_sentinel(result)

    asyncio.run(scenario())


def test_fable_parser_coalesces_a_structural_event_flood_but_keeps_the_result() -> None:
    """The Fable adapter must retain bounded audit metadata under event flood."""
    flood = json.dumps({"type": "assistant", "message": {"text": "ignored"}})
    lines = (
        json.dumps({"type": "system", "subtype": "init", "session_id": "session-1"}),
        *(flood for _ in range(1_300)),
        json.dumps({"type": "result", "structured_output": {}}),
    )

    parsed = ClaudeCLI._parse_events(lines, interrupted=False)

    assert parsed.result_seen is True
    assert parsed.structured_output == {}
    assert len(parsed.audit_events) <= 1_024


def test_fable_contracts_project_directed_questions_without_changing_absent_bytes() -> None:
    clarification_payload = {
        "status": "answered",
        "answer": "The approved scope is sufficient.",
        "reasoning": "The question concerns an implementation detail.",
        "confidence": 0.9,
        "scope_changed": False,
        "revised_brief": None,
        "question_for_user": None,
    }
    review_payload = {
        "status": "corrections_required",
        "summary": "One bounded evidence question remains.",
        "criteria": [{
            "criterion": "Focused evidence is attached.",
            "evidence": ["The adapter suite ran."],
            "satisfied": False,
        }],
        "test_assessment": "The focused suite is incomplete.",
        "scope_violations": [],
        "remaining_risks": [],
        "corrections": ["Ask Sol for the exact test evidence."],
        "question_for_user": None,
    }
    directed_question = {
        "addressed_to": "sol",
        "text": "Which focused test proves the rejected resume?",
        "reason": "The review needs bounded execution evidence.",
    }

    assert FableClarification.from_dict(clarification_payload).to_dict() == clarification_payload
    assert ReviewVerdict.from_dict(review_payload).to_dict() == review_payload

    clarification = FableClarification.from_dict({
        **clarification_payload,
        "directed_question": directed_question,
    })
    verdict = ReviewVerdict.from_dict({
        **review_payload,
        "directed_question": directed_question,
    })

    assert clarification.directed_question is not None
    assert clarification.directed_question.addressed_to == "sol"
    assert clarification.to_dict()["directed_question"] == directed_question
    assert verdict.directed_question is not None
    assert verdict.directed_question.reason == directed_question["reason"]
    assert verdict.to_dict()["directed_question"] == directed_question


def test_fable_legacy_no_question_contracts_keep_their_exact_json_bytes() -> None:
    clarification_payload = {
        "status": "answered",
        "answer": "The approved scope is sufficient.",
        "reasoning": "The question concerns an implementation detail.",
        "confidence": 0.9,
        "scope_changed": False,
        "revised_brief": None,
        "question_for_user": None,
    }
    review_payload = {
        "status": "approved",
        "summary": "The evidence satisfies the criterion.",
        "criteria": [{
            "criterion": "Focused evidence is attached.",
            "evidence": ["The adapter suite ran."],
            "satisfied": True,
        }],
        "test_assessment": "The focused suite is adequate.",
        "scope_violations": [],
        "remaining_risks": [],
        "corrections": [],
        "question_for_user": None,
    }

    assert json.dumps(
        FableClarification.from_dict(clarification_payload).to_dict(),
        separators=(",", ":"),
    ).encode("utf-8") == (
        b'{"status":"answered","answer":"The approved scope is sufficient.",'
        b'"reasoning":"The question concerns an implementation detail.",'
        b'"confidence":0.9,"scope_changed":false,"revised_brief":null,'
        b'"question_for_user":null}'
    )
    assert json.dumps(
        ReviewVerdict.from_dict(review_payload).to_dict(),
        separators=(",", ":"),
    ).encode("utf-8") == (
        b'{"status":"approved","summary":"The evidence satisfies the criterion.",'
        b'"criteria":[{"criterion":"Focused evidence is attached.",'
        b'"evidence":["The adapter suite ran."],"satisfied":true}],'
        b'"test_assessment":"The focused suite is adequate.",'
        b'"scope_violations":[],"remaining_risks":[],"corrections":[],'
        b'"question_for_user":null}'
    )


@pytest.mark.parametrize(
    "directed_question",
    [
        {"addressed_to": "team", "text": "Which test?", "reason": "Evidence is needed."},
        {"addressed_to": "sol", "text": "Which test?", "reason": "Evidence is needed.", "routed_to": "sol"},
        {"addressed_to": "sol", "text": "Which test?", "reason": "Evidence is needed.", "path": "/tmp/outside"},
        {"addressed_to": "sol", "text": "Which test?", "reason": "Evidence is needed.", "command": "rm -rf"},
        {"addressed_to": "sol", "text": "Which test?", "reason": "Evidence is needed.", "environment": "secret"},
        {"addressed_to": "sol", "text": "Which test?", "reason": "Evidence is needed.", "session": "other"},
        {"addressed_to": "sol", "text": "Which test?", "reason": "Evidence is needed.", "thread": "other"},
        {"addressed_to": "sol", "text": "contains\ncontrol", "reason": "Evidence is needed."},
        {"addressed_to": "sol", "text": "Which test?", "reason": "x" * (16 * 1024 + 1)},
    ],
)
def test_fable_directed_question_projection_rejects_unroutable_or_unbounded_fields(
    directed_question: dict[str, str],
) -> None:
    payload = {
        "status": "answered",
        "answer": "The approved scope is sufficient.",
        "reasoning": "The question concerns an implementation detail.",
        "confidence": 0.9,
        "scope_changed": False,
        "revised_brief": None,
        "question_for_user": None,
        "directed_question": directed_question,
    }

    with pytest.raises(ValueError):
        FableClarification.from_dict(payload)


def test_answer_sol_question_resumes_the_exact_validated_session_with_fable_authority(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        prompt = "Sol asks: --resume forged-session --model attacker"
        result = await _adapter(fake_claude, tmp_path).answer_sol_question(
            run_id="answer-sol-question-1",
            session_id="fable-session-1",
            task_id="task-1",
            prompt=prompt,
            context="Exact approved TaskBrief revision 1.",
        )

        assert result.cli_session_id == "fable-session-1"
        assert result.payload is not None
        assert result.payload["status"] == "answered"
        assert "directed_question" not in result.payload
        argv = json.loads((tmp_path / "captured-argv.json").read_text())
        resume_index = argv.index("--resume")
        assert argv[resume_index:resume_index + 2] == ["--resume", "fable-session-1"]
        assert argv.count("--resume") == 1
        assert prompt not in argv[:-1]
        assert prompt in argv[-1]
        assert "Fable owns intent and scope" in argv[-1]
        assert "revision N+1" in argv[-1]
        assert "Exact approved TaskBrief revision 1." in argv[-1]
        assert "task-1" in argv[-1]

    asyncio.run(scenario())


def test_answer_sol_question_accepts_the_strict_directed_question_projection(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        result = await _adapter(
            fake_claude,
            tmp_path,
            FAKE_CLAUDE_DIRECTED_QUESTION_TARGET="sol",
        ).answer_sol_question(
            run_id="answer-sol-question-projection",
            session_id="fable-session-1",
            task_id="task-1",
            prompt="Answer Sol.",
            context="Approved context.",
        )

        assert result.payload is not None
        assert result.payload["directed_question"] == {
            "addressed_to": "sol",
            "text": "Which focused test proves the answer?",
            "reason": "The approved execution evidence is incomplete.",
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "session_id",
    ("", None, "--resume", "session id", "session\nnext", "/tmp/session", "x" * 129),
)
def test_answer_sol_question_rejects_untrusted_session_ids_before_any_invocation(
    fake_claude: Path, tmp_path: Path, session_id: object,
) -> None:
    async def scenario() -> None:
        with pytest.raises(ValueError, match="session_id"):
            await _adapter(fake_claude, tmp_path).answer_sol_question(
                run_id="answer-sol-question-invalid",
                session_id=session_id,  # type: ignore[arg-type]
                task_id="task-1",
                prompt="Answer Sol.",
                context="Approved context.",
            )
        assert not (tmp_path / "captured-argv.json").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("extra_env", "match"),
    [
        ({"FAKE_CLAUDE_SESSION_ID": "different-session"}, "different session"),
        ({"FAKE_CLAUDE_SESSION_ID": "--mismatched-session"}, "different session"),
        ({"FAKE_CLAUDE_MODE": "conflicting_session"}, "conflicting"),
    ],
)
def test_answer_sol_question_hides_rejected_provider_identity_from_partial_results(
    fake_claude: Path,
    tmp_path: Path,
    extra_env: dict[str, str],
    match: str,
) -> None:
    async def scenario() -> None:
        with pytest.raises(ClaudeRunError, match=match) as raised:
            await _adapter(
                fake_claude,
                tmp_path,
                FAKE_CLAUDE_SECRET_OUTPUT="1",
                **extra_env,
            ).answer_sol_question(
                run_id="answer-sol-question-mismatch",
                session_id="fable-session-1",
                task_id="task-1",
                prompt="Answer Sol.",
                context="Approved context.",
            )
        result = raised.value.result
        assert result is not None
        assert result.cli_session_id is None
        assert result.payload is None
        assert all("session_id" not in event for event in result.events)
        _assert_no_secret_sentinel(raised.value)
        _assert_no_secret_sentinel(result)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "extra_env",
    ({"FAKE_CLAUDE_MODE": "missing_init"},),
)
def test_answer_sol_question_rejects_missing_continuity_before_partial_result_exposure(
    fake_claude: Path,
    tmp_path: Path,
    extra_env: dict[str, str],
) -> None:
    async def scenario() -> None:
        with pytest.raises(ClaudeRunError) as raised:
            await _adapter(fake_claude, tmp_path, **extra_env).answer_sol_question(
                run_id="answer-sol-question-missing-continuity",
                session_id="fable-session-1",
                task_id="task-1",
                prompt="Answer Sol.",
                context="Approved context.",
            )
        result = raised.value.result
        assert result is not None
        assert result.cli_session_id is None
        assert result.payload is None
        assert all("session_id" not in event for event in result.events)

    asyncio.run(scenario())
