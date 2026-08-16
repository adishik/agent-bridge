from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent_bridge.app import (
    BootstrapStatus,
    InMemoryEventBroadcaster,
    create_app,
    create_hub_app,
)
from agent_bridge.contracts import (
    ConversationActor,
    ConversationEnvelope,
    ConversationMessageType,
    ConversationTarget,
    DirectedAgentQuestion,
    StreamEvent,
    TaskBrief,
)
from agent_bridge.coordinator import Coordinator, InterventionIntent
from agent_bridge.hub import (
    ActiveAgentLease,
    HubWorkflowOrchestrator,
    LeaseToken,
    ProjectRegistry,
    RuntimeReadiness,
    RuntimeStatus,
)
from agent_bridge.projects import project_id_for_root
from agent_bridge.process import ProcessRunner, StopReceipt
from agent_bridge.state_machine import TaskState
from agent_bridge.store import InterventionStatus, SQLiteStore


SESSION_ID = "session-1"
SESSION_KEY = "session-secret"
CSRF_TOKEN = "csrf-secret"
EXPECTED_EVENT_REPLAY_PAGE_SIZE = 100
EXPECTED_MAX_INITIAL_REPLAY_EVENTS = 300
_HANGING_PROBE = object()


class _Clock:
    def __init__(self) -> None:
        self._tick = 0

    def __call__(self) -> str:
        self._tick += 1
        return f"2026-08-10T00:00:{self._tick:02d}Z"


@dataclass
class _RealIds:
    task_number: int = 0

    def new_task_id(self) -> str:
        self.task_number += 1
        return f"real-task-{self.task_number}"

    def new_run_id(self) -> str:
        self.task_number += 1
        return f"real-run-{self.task_number}"


class RecordingCoordinator:
    def __init__(
        self,
        *,
        store: SQLiteStore,
        broadcaster: InMemoryEventBroadcaster,
    ) -> None:
        self.store = store
        self.broadcaster = broadcaster
        self.calls: list[tuple[object, ...]] = []
        self.fail_actions: set[str] = set()
        self.block_actions: set[str] = set()
        self.release = threading.Event()
        self.emit_message_events = False

    async def _record(self, action: str, *values: object) -> None:
        self.calls.append((action, *values))
        if action in self.block_actions:
            while not self.release.is_set():
                await asyncio.sleep(0.005)
        if action in self.fail_actions:
            raise RuntimeError(f"{action} failed with deliberately unsafe detail")

    async def handle_user_request(self, session_id: str, text: str) -> str:
        await self._record("message", session_id, text)
        if self.emit_message_events:
            self.store.append_event(
                session_id,
                None,
                "fable",
                "message",
                {"text": f"planned: {text}"},
            )
        return "task-generated"

    async def approve_task(self, task_id: str, revision: int) -> None:
        await self._record("approve", task_id, revision)

    async def edit_task(self, task_id: str, brief: TaskBrief) -> None:
        await self._record("edit", task_id, brief.to_dict())

    async def reject_task(self, task_id: str) -> None:
        await self._record("reject", task_id)

    async def answer_user_question(self, task_id: str, answer: str) -> None:
        await self._record("answer", task_id, answer)

    async def stop_task(self, task_id: str) -> None:
        await self._record("stop", task_id)

    async def resume_task(self, task_id: str) -> None:
        await self._record("resume", task_id)


@dataclass(frozen=True)
class WebHarness:
    app: Any
    store: SQLiteStore
    coordinator: RecordingCoordinator
    broadcaster: InMemoryEventBroadcaster
    static_dir: Path
    status_provider: "_StatusProvider"


@dataclass
class _StatusProvider:
    status: BootstrapStatus

    def __call__(self) -> BootstrapStatus:
        return self.status


@pytest.fixture
def web_harness(tmp_path: Path, valid_brief: TaskBrief) -> WebHarness:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text(
        "<!doctype html><title>Agent Bridge test</title>", encoding="utf-8"
    )
    (static_dir / "app.css").write_text("body { color: black; }\n", encoding="utf-8")
    (static_dir / "app.js").write_text("globalThis.bridgeLoaded = true;\n", encoding="utf-8")
    store = SQLiteStore(
        tmp_path / "bridge.sqlite3",
        clock=_Clock(),
        check_same_thread=False,
    )
    store.create_session(SESSION_ID, "/repo")
    store.save_task(SESSION_ID, valid_brief, TaskState.AWAITING_USER_APPROVAL)
    broadcaster = InMemoryEventBroadcaster()
    coordinator = RecordingCoordinator(store=store, broadcaster=broadcaster)
    status_provider = _StatusProvider(BootstrapStatus(
        session_id=SESSION_ID,
        fable_ready=True,
        fable_status="subscription_ready",
        sol_status="ready",
        repository="/repo",
        branch="feat/agent-bridge",
    ))
    app = create_app(
        coordinator=coordinator,
        store=store,
        static_dir=static_dir,
        session_key=SESSION_KEY,
        csrf_token=CSRF_TOKEN,
        broadcaster=broadcaster,
        bootstrap_status=status_provider,
    )
    harness = WebHarness(
        app=app,
        store=store,
        coordinator=coordinator,
        broadcaster=broadcaster,
        static_dir=static_dir,
        status_provider=status_provider,
    )
    try:
        yield harness
    finally:
        coordinator.release.set()
        store.close()


def _authenticated_client(harness: WebHarness) -> TestClient:
    client = TestClient(harness.app)
    response = client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
    if response.status_code != 303 or response.headers.get("location") != "/":
        raise RuntimeError("test client authentication failed")
    return client


def _csrf(client: TestClient) -> str:
    response = client.get("/api/bootstrap")
    if response.status_code != 200:
        raise RuntimeError("test bootstrap failed")
    token = response.json().get("csrf_token")
    if not isinstance(token, str):
        raise RuntimeError("test bootstrap omitted its CSRF token")
    return token


def _acknowledge_model_usage(client: TestClient) -> str:
    token = _csrf(client)
    response = client.post(
        "/api/settings/usage-credits-acknowledgement",
        json={"acknowledged": True},
        headers={"X-CSRF-Token": token},
    )
    if response.status_code != 202:
        raise RuntimeError("test usage-credit acknowledgement failed")
    return token


def _wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before the test deadline")


def _edited_brief(valid_brief: TaskBrief) -> dict[str, object]:
    payload = valid_brief.to_dict()
    payload["revision"] = 2
    payload["title"] = "Edited bridge task"
    return payload


def test_keyed_index_sets_strict_http_only_cookie_and_mutations_require_csrf(
    web_harness: WebHarness,
) -> None:
    with TestClient(web_harness.app) as client:
        assert client.get("/").status_code == 403
        keyed = client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
        assert keyed.status_code == 303
        assert keyed.headers["location"] == "/"
        assert keyed.headers["cache-control"] == "no-store"
        assert keyed.headers["referrer-policy"] == "no-referrer"
        assert "agent_bridge_session" in keyed.cookies
        set_cookie = keyed.headers["set-cookie"]
        assert "HttpOnly" in set_cookie
        assert "SameSite=strict" in set_cookie
        assert SESSION_KEY not in keyed.text
        clean = client.get("/")
        assert clean.status_code == 200
        assert clean.request.url.path == "/"
        assert clean.request.url.query == b""
        assert clean.headers["cache-control"] == "no-store"
        assert clean.headers["referrer-policy"] == "no-referrer"

        missing = client.post(
            f"/api/sessions/{SESSION_ID}/messages", json={"text": "plan it"}
        )
        assert missing.status_code == 403
        bootstrap = client.get("/api/bootstrap")
        assert bootstrap.status_code == 200
        assert bootstrap.json()["csrf_token"] == CSRF_TOKEN
        _acknowledge_model_usage(client)
        accepted = client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "plan it"},
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
        assert accepted.status_code == 202
        _wait_until(
            lambda: ("message", SESSION_ID, "plan it")
            in web_harness.coordinator.calls
        )


def test_followed_key_redirect_finishes_on_clean_url_without_key_referrer(
    web_harness: WebHarness,
) -> None:
    with TestClient(web_harness.app) as client:
        response = client.get(f"/?key={SESSION_KEY}")
    assert response.status_code == 200
    assert response.url.path == "/"
    assert response.url.query == b""
    assert len(response.history) == 1
    assert response.history[0].status_code == 303
    assert response.history[0].headers["location"] == "/"
    assert response.history[0].headers["referrer-policy"] == "no-referrer"
    assert response.request.headers.get("referer") is None
    assert SESSION_KEY not in response.text


def test_public_static_assets_cannot_bypass_index_authentication(
    web_harness: WebHarness,
) -> None:
    with TestClient(web_harness.app) as client:
        assert client.get("/static/index.html").status_code != 200
        assert client.get("/static/%2e%2e/index.html").status_code != 200
        css = client.get("/static/app.css")
        javascript = client.get("/static/app.js")

    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert css.text == "body { color: black; }\n"
    assert javascript.status_code == 200
    assert "javascript" in javascript.headers["content-type"]
    assert javascript.text == "globalThis.bridgeLoaded = true;\n"


def test_overlong_public_asset_name_fails_closed_as_not_found(
    web_harness: WebHarness,
) -> None:
    with TestClient(web_harness.app) as client:
        response = client.get(f"/static/{'a' * 5000}")

    assert response.status_code == 404


def test_wrong_key_cookie_and_websocket_cookie_fail_closed(
    web_harness: WebHarness,
) -> None:
    with TestClient(web_harness.app) as client:
        assert client.get("/?key=wrong").status_code == 403
        assert client.get("/api/bootstrap").status_code == 403
        client.cookies.set("agent_bridge_session", "wrong")
        assert client.get("/api/bootstrap").status_code == 403
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                f"/ws?session_id={SESSION_ID}&after=0"
            ):
                pass
        assert caught.value.code == 1008


def test_tokens_require_non_empty_ascii_configuration(
    web_harness: WebHarness,
) -> None:
    for name, session_key, csrf_token in (
        ("empty session", "", CSRF_TOKEN),
        ("unicode session", "snowman-☃", CSRF_TOKEN),
        ("empty csrf", SESSION_KEY, ""),
        ("unicode csrf", SESSION_KEY, "csrf-☃"),
    ):
        with pytest.raises(ValueError, match="ASCII|non-empty"):
            create_app(
                coordinator=web_harness.coordinator,
                store=web_harness.store,
                static_dir=web_harness.static_dir,
                session_key=session_key,
                csrf_token=csrf_token,
                broadcaster=web_harness.broadcaster,
            )


def test_non_ascii_auth_candidates_fail_closed_without_server_error(
    web_harness: WebHarness,
) -> None:
    raw_cookie = [(b"cookie", b"agent_bridge_session=\xff")]
    authenticated_headers = [
        (b"cookie", f"agent_bridge_session={SESSION_KEY}".encode("ascii")),
        (b"x-csrf-token", b"\xff"),
    ]
    with TestClient(web_harness.app) as client:
        assert client.get("/?key=%E2%98%83").status_code == 403
        assert client.get("/api/bootstrap", headers=raw_cookie).status_code == 403
        assert client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "plan it"},
            headers=authenticated_headers,
        ).status_code == 403
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                f"/ws?session_id={SESSION_ID}&after=0",
                headers={b"cookie": b"agent_bridge_session=\xff"},
            ):
                pass
        assert caught.value.code == 1008
    assert web_harness.coordinator.calls == []


def test_unknown_session_is_rejected_before_work_or_socket_acceptance(
    web_harness: WebHarness,
) -> None:
    with _authenticated_client(web_harness) as client:
        response = client.post(
            "/api/sessions/missing-session/messages",
            json={"text": "plan it"},
            headers={"X-CSRF-Token": _csrf(client)},
        )
        assert response.status_code == 404
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                "/ws?session_id=missing-session&after=0"
            ):
                pass
        assert caught.value.code == 1008
    assert web_harness.coordinator.calls == []


def test_health_check_is_loopback_liveness_not_a_session_or_secret_disclosure(
    web_harness: WebHarness,
) -> None:
    with TestClient(web_harness.app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert SESSION_KEY not in response.text
    assert CSRF_TOKEN not in response.text


def test_bootstrap_is_complete_authoritative_and_omits_runtime_secrets(
    web_harness: WebHarness,
) -> None:
    with _authenticated_client(web_harness) as client:
        payload = client.get("/api/bootstrap").json()

    assert payload == {
        "csrf_token": CSRF_TOKEN,
        "usage_credits_acknowledged": False,
        "session_id": SESSION_ID,
        "fable_ready": True,
        "fable_status": "subscription_ready",
        "sol_status": "ready",
        "repository": "/repo",
        "branch": "feat/agent-bridge",
        "replay_after": 0,
        "tasks": [
            {
                "task_id": "task-1",
                "revision": 1,
                "state": "awaiting_user_approval",
                "brief": web_harness.store.task_brief("task-1", 1).to_dict(),
                "approved_at": None,
                "correction_count": 0,
                "continuation_state": None,
                "updated_at": None,
                "active_agent": None,
                "active_started_at": None,
                "revision_start_sequence": None,
                "outcome": None,
                "review": None,
                "clarification": None,
                "activity_kind": None,
                "activity": None,
            }
        ],
    }
    serialized = str(payload)
    for forbidden in (
        "fable_session_id",
        "sol_thread_id",
        "baseline_id",
        "process_group_id",
        "executable",
        "environment",
        "raw_command",
    ):
        assert forbidden not in serialized


def test_bootstrap_keeps_current_revision_evidence_outside_recent_chat_window(
    web_harness: WebHarness,
) -> None:
    brief = web_harness.store.task_brief("task-1", 1)
    web_harness.store.append_event(
        SESSION_ID, brief.task_id, "sol", "outcome", {"summary": "old outcome"},
    )
    boundary = web_harness.store.append_event(
        SESSION_ID, brief.task_id, "fable", "task_brief",
        {"brief": brief.to_dict()},
    )
    evidence = {
        "outcome": {"summary": "durable outcome"},
        "review": {"summary": "durable review"},
        "clarification": {"reasoning": "durable clarification"},
        "agent_event": {"status": "completed", "command_sha256": "a" * 64},
    }
    for kind, payload in evidence.items():
        web_harness.store.append_event(
            SESSION_ID, brief.task_id,
            "sol" if kind in {"outcome", "agent_event"} else "fable",
            kind, payload,
        )
    for index in range(EXPECTED_MAX_INITIAL_REPLAY_EVENTS + 5):
        web_harness.store.append_event(
            SESSION_ID, None, "coordinator", "message", {"text": str(index)},
        )

    with _authenticated_client(web_harness) as client:
        task = client.get("/api/bootstrap").json()["tasks"][0]

    assert task["revision_start_sequence"] == boundary.sequence
    assert task["outcome"] == evidence["outcome"]
    assert task["review"] == evidence["review"]
    assert task["clarification"] == evidence["clarification"]
    assert task["activity_kind"] == "agent_event"
    assert task["activity"] == evidence["agent_event"]


@pytest.mark.parametrize("kind", ("action_error", "stop_error", "agent_event", "resume_drift"))
def test_bootstrap_activity_projection_is_allowlisted_and_never_reloads_raw_details(
    web_harness: WebHarness,
    kind: str,
) -> None:
    brief = web_harness.store.task_brief("task-1", 1)
    web_harness.store.append_event(
        SESSION_ID, brief.task_id, "fable", "task_brief", {"brief": brief.to_dict()},
    )
    web_harness.store.append_event(
        SESSION_ID, brief.task_id, "coordinator", kind,
        {
            "status": "completed",
            "command_sha256": "a" * 64,
            "run_id": "provider-run-secret",
            "thread_id": "provider-thread-secret",
            "session_id": "provider-session-secret",
            "command": "rm -rf never-render-this",
            "raw_output": "provider output must not reload",
            "extra": "hostile-extra",
            "hostile": '<img src=x onerror="globalThis.pwned=true">' * 200,
        },
    )

    with _authenticated_client(web_harness) as client:
        task = client.get("/api/bootstrap").json()["tasks"][0]

    assert task["activity_kind"] == kind
    assert task["activity"] == (
        {"status": "completed", "command_sha256": "a" * 64}
        if kind == "agent_event" else {}
    )
    serialized = json.dumps(task)
    for forbidden in (
        "provider-run-secret", "provider-thread-secret", "provider-session-secret",
        "rm -rf never-render-this", "provider output must not reload", "hostile-extra",
        "globalThis.pwned",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize("kind", ("action_error", "stop_error", "agent_event", "resume_drift"))
def test_hub_bootstrap_activity_projection_is_allowlisted_and_project_scoped(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
    kind: str,
) -> None:
    runtime = hub_harness.runtimes["project-a"]
    runtime.store.save_task("chat-a", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    runtime.store.append_event(
        "chat-a", valid_brief.task_id, "fable", "task_brief", {"brief": valid_brief.to_dict()},
    )
    runtime.store.append_event(
        "chat-a", valid_brief.task_id, "coordinator", kind,
        {"status": "completed", "command_sha256": "b" * 64, "run_id": "foreign-provider-run"},
    )
    with _authenticated_hub_client(hub_harness) as client:
        payload = client.get("/api/projects/project-a/chats/chat-a/bootstrap").json()
    task = next(item for item in payload["tasks"] if item["task_id"] == valid_brief.task_id)
    assert task["activity_kind"] == kind
    assert task["activity"] == (
        {"status": "completed", "command_sha256": "b" * 64}
        if kind == "agent_event" else {}
    )
    assert "foreign-provider-run" not in json.dumps(payload)


def test_bootstrap_activity_projection_drops_invalid_agent_fields(
    web_harness: WebHarness,
) -> None:
    brief = web_harness.store.task_brief("task-1", 1)
    web_harness.store.append_event(
        SESSION_ID, brief.task_id, "fable", "task_brief", {"brief": brief.to_dict()},
    )
    web_harness.store.append_event(
        SESSION_ID, brief.task_id, "coordinator", "agent_event",
        {
            "status": "COMPLETED",
            "command_sha256": "A" * 64,
            "run_id": "invalid-digest-provider-run",
        },
    )

    with _authenticated_client(web_harness) as client:
        task = client.get("/api/bootstrap").json()["tasks"][0]

    assert task["activity_kind"] == "agent_event"
    assert task["activity"] == {}
    assert "invalid-digest-provider-run" not in json.dumps(task)


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        *[(status, {"status": status, "command_sha256": "c" * 64}) for status in (
            "completed", "declined", "failed", "in_progress", "interrupted",
        )],
        *[(status, {"command_sha256": "c" * 64}) for status in (
            "running", "pending", "success", "error", "COMPLETED",
        )],
    ),
)
def test_bootstrap_activity_projection_matches_coordinator_agent_event_statuses(
    web_harness: WebHarness,
    status: str,
    expected: dict[str, str],
) -> None:
    brief = web_harness.store.task_brief("task-1", 1)
    web_harness.store.append_event(
        SESSION_ID, brief.task_id, "fable", "task_brief", {"brief": brief.to_dict()},
    )
    web_harness.store.append_event(
        SESSION_ID, brief.task_id, "sol", "agent_event",
        {"status": status, "command_sha256": "c" * 64, "run_id": "never-project"},
    )
    with _authenticated_client(web_harness) as client:
        task = client.get("/api/bootstrap").json()["tasks"][0]
    assert task["activity_kind"] == "agent_event"
    assert task["activity"] == expected
    assert "never-project" not in json.dumps(task)


@pytest.mark.parametrize(
    "status", ("completed", "declined", "failed", "in_progress", "interrupted"),
)
def test_coordinator_producer_characterizes_the_agent_event_status_contract(
    status: str,
) -> None:
    digest = "d" * 64
    event = Coordinator._sol_structural_event({
        "type": "item.completed",
        "item_type": "command_execution",
        "status": status,
        "command_sha256": digest,
    })
    assert event is not None
    assert event["status"] == status
    assert event["command_sha256"] == digest


@pytest.mark.parametrize(
    ("event_type", "expected_status"),
    (("item.started", None), ("item.updated", None), ("item.completed", "completed")),
)
def test_coordinator_producer_only_attaches_status_to_completed_agent_events(
    event_type: str,
    expected_status: str | None,
) -> None:
    event = Coordinator._sol_structural_event({
        "type": event_type,
        "item_type": "command_execution",
        "status": "completed",
        "command_sha256": "e" * 64,
    })
    assert event is not None
    assert event.get("status") == expected_status
    assert event["command_sha256"] == "e" * 64


def test_default_bootstrap_shape_is_complete_and_fail_closed(
    web_harness: WebHarness,
) -> None:
    app = create_app(
        coordinator=web_harness.coordinator,
        store=web_harness.store,
        static_dir=web_harness.static_dir,
        session_key=SESSION_KEY,
        csrf_token=CSRF_TOKEN,
        broadcaster=web_harness.broadcaster,
    )
    with TestClient(app) as client:
        client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
        payload = client.get("/api/bootstrap").json()

    assert payload == {
        "csrf_token": CSRF_TOKEN,
        "usage_credits_acknowledged": False,
        "session_id": None,
        "fable_ready": False,
        "fable_status": "checking",
        "sol_status": "checking",
        "repository": None,
        "branch": None,
        "replay_after": 0,
        "tasks": [],
    }


def test_bootstrap_status_requires_subscription_ready_proof() -> None:
    with pytest.raises(ValueError, match="subscription_ready"):
        BootstrapStatus(
            session_id=SESSION_ID,
            fable_ready=True,
            fable_status="api_key_ready",
            sol_status="ready",
            repository="/repo",
            branch="main",
        )


def test_model_starting_mutations_require_current_session_readiness_and_acknowledgement(
    web_harness: WebHarness,
    valid_brief: TaskBrief,
) -> None:
    model_actions = (
        (f"/api/sessions/{SESSION_ID}/messages", {"text": "plan"}),
        ("/api/tasks/task-1/approve", {"revision": 1}),
        ("/api/tasks/task-1/answer", {"answer": "answer"}),
        ("/api/tasks/task-1/resume", None),
    )
    with _authenticated_client(web_harness) as client:
        csrf = _csrf(client)
        headers = {"X-CSRF-Token": csrf}

        # These actions do not initiate or resume model execution.
        assert client.post(
            "/api/tasks/task-1/edit",
            json=_edited_brief(valid_brief),
            headers=headers,
        ).status_code == 202
        assert client.post(
            "/api/tasks/task-1/reject", headers=headers
        ).status_code == 202
        assert client.post(
            "/api/tasks/task-1/stop", headers=headers
        ).status_code == 202

        for path, body in model_actions:
            response = client.post(path, json=body, headers=headers)
            assert response.status_code == 409
            assert "Fable subscription readiness" in response.json()["detail"]
            assert "Sol readiness" in response.json()["detail"]
            assert "usage-credit acknowledgement" in response.json()["detail"]

        _acknowledge_model_usage(client)
        web_harness.status_provider.status = BootstrapStatus(
            session_id=SESSION_ID,
            fable_ready=False,
            fable_status="subscription_unavailable",
            sol_status="ready",
            repository="/repo",
            branch="feat/agent-bridge",
        )
        for path, body in model_actions:
            assert client.post(path, json=body, headers=headers).status_code == 409

        for sol_status in ("checking", "blocked", "unavailable"):
            web_harness.status_provider.status = BootstrapStatus(
                session_id=SESSION_ID,
                fable_ready=True,
                fable_status="subscription_ready",
                sol_status=sol_status,
                repository="/repo",
                branch="feat/agent-bridge",
            )
            for path, body in model_actions:
                response = client.post(path, json=body, headers=headers)
                assert response.status_code == 409
                assert "model actions require" in response.json()["detail"]

        web_harness.store.create_session("session-2", "/repo")
        web_harness.status_provider.status = BootstrapStatus(
            session_id="session-2",
            fable_ready=True,
            fable_status="subscription_ready",
            sol_status="ready",
            repository="/repo",
            branch="feat/agent-bridge",
        )
        for path, body in model_actions:
            assert client.post(path, json=body, headers=headers).status_code == 409

        web_harness.status_provider.status = BootstrapStatus(
            session_id=SESSION_ID,
            fable_ready=True,
            fable_status="subscription_ready",
            sol_status="ready",
            repository="/repo",
            branch="feat/agent-bridge",
        )
        for path, body in model_actions:
            assert client.post(path, json=body, headers=headers).status_code == 202

    _wait_until(lambda: len(web_harness.coordinator.calls) == 7)


def test_browser_controller_uses_exact_bootstrap_and_recovers_from_initial_failure(
    web_harness: WebHarness,
) -> None:
    persisted_event = web_harness.store.append_event(
        SESSION_ID,
        "task-1",
        "coordinator",
        "task_state",
        {"state": "sol_running", "revision": 1},
    ).to_dict()
    with _authenticated_client(web_harness) as client:
        assert client.post(
            "/api/settings/usage-credits-acknowledgement",
            json={"acknowledged": True},
            headers={"X-CSRF-Token": _csrf(client)},
        ).status_code == 202
        bootstrap = client.get("/api/bootstrap").json()

    module_uri = (
        Path("src/agent_bridge/static/app.js").resolve().as_uri()
    )
    unsafe = '<img src=x onerror="globalThis.pwned=true">'
    harness = f"""
      import * as bridge from {json.dumps(module_uri)};
      const bootstrap = {json.dumps(bootstrap)};
      const hostileActivity = "<img src=x onerror=globalThis.pwned=true>".repeat(200);
      const projectBootstrap = {{...bootstrap, project_id: "project-a", tasks: [{{...bootstrap.tasks[0], exchange_allowance: 2, exchange_consumed: 1, activity_kind: "agent_event", activity: {{status: hostileActivity, command_sha256: hostileActivity, run_id: hostileActivity, raw_output: hostileActivity}}}}, ...bootstrap.tasks.slice(1), {{task_id: "intervene-task", revision: 1, continuation_generation: 1, state: "fable_planning"}}]}};
      let projectPayload = {{csrf_token: {json.dumps(CSRF_TOKEN)}, usage_credits_acknowledged: true, projects: [{{
        project_id: "project-a", label: "PROJECT-A", branch: "feat/agent-bridge",
        readiness: {{fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
      }}], active_lease: null}};
      const chats = {{chats: [{{session_id: "session-1", title: "New chat", latest_sequence: 0}}]}};
      const persistedEvent = {json.dumps(persisted_event)};
      const unsafe = {json.dumps(unsafe)};
      let interactionAllowed = () => true;
      const flush = async () => {{ for (let tick = 0; tick < 12; tick += 1) await Promise.resolve(); }};

      class ClassList {{
        constructor(node) {{ this.node = node; this.values = new Set(); }}
        add(value) {{ this.values.add(value); this.node.className = [...this.values].join(" "); }}
        remove(value) {{ this.values.delete(value); this.node.className = [...this.values].join(" "); }}
        contains(value) {{ return this.values.has(value); }}
        toggle(value, force) {{ force ? this.add(value) : this.remove(value); }}
      }}
      let documentRoot;
      class Node {{
        constructor(tag, id = "") {{
          this.tag = tag; this.id = id; this.children = []; this.attributes = {{}};
          this.dataset = {{}}; this.className = ""; this.classList = new ClassList(this);
          this._text = ""; this.value = ""; this.disabled = false; this.hidden = false;
          this.checked = false; this.open = false; this.inert = false; this.listeners = {{}};
        }}
        set textContent(value) {{ this._text = String(value); this.children = []; }}
        get textContent() {{ return this._text + this.children.map((child) => child.textContent).join(""); }}
        append(...children) {{ for (const child of children) {{ child.parent = this; this.children.push(child); }} }}
        replaceChildren(...children) {{ this.children = []; this.append(...children); this._text = ""; }}
        removeChild(child) {{ this.children.splice(this.children.indexOf(child), 1); }}
        remove() {{ this.parent?.removeChild(this); }}
        setAttribute(name, value) {{ this.attributes[name] = String(value); }}
        removeAttribute(name) {{ delete this.attributes[name]; if (name === "open") this.open = false; }}
        getAttribute(name) {{ return this.attributes[name] ?? null; }}
        addEventListener(kind, listener) {{ (this.listeners[kind] ??= []).push(listener); }}
            async emit(kind, event = {{}}) {{
              if (!interactionAllowed(this)) return;
              event.preventDefault ??= () => {{}};
              for (const listener of this.listeners[kind] ?? []) await listener(event);
              await this.onclick?.(event);
        }}
        focus() {{ documentRoot.activeElement = this; this.focusCount = (this.focusCount ?? 0) + 1; }}
        showModal() {{ this.open = true; this.showModalCount = (this.showModalCount ?? 0) + 1; }}
        close() {{ this.open = false; this.closeCount = (this.closeCount ?? 0) + 1; }}
        querySelectorAll(selector) {{
          const selectors = selector.split(",").map((entry) => entry.trim());
          const matches = (node) => selectors.some((entry) => (
            entry === node.tag || entry === "[tabindex]" && node.attributes.tabindex !== undefined
              || entry.startsWith("#") && node.id === entry.slice(1)
          ));
          const result = [];
          const stack = [...this.children];
          while (stack.length) {{
            const child = stack.shift();
            if (matches(child)) result.push(child);
            stack.push(...child.children);
          }}
          return result;
        }}
        querySelector(selector) {{
          return this.querySelectorAll(selector)[0] ?? null;
        }}
      }}

      const ids = [
        "task-list", "conversation", "conversation-shell", "project-navigation",
        "task-inspector", "task-inspector-panel", "composer", "message-input",
        "composer-submit", "composer-guidance", "usage-modal", "usage-credits-form",
        "usage-credits-confirm", "usage-credits-acknowledge", "usage-error",
        "toast-region", "fable-status", "sol-status", "repository-status",
        "connection-status", "task-drawer-toggle", "inspector-drawer-toggle",
        "bootstrap-retry", "project-list", "chat-list", "new-chat",
            "selected-project-name", "selected-chat-name", "task-inspector-summary", "composer-recipient",
            "composer-label", "task-inspector-empty",
            "task-controls", "activity-audit", "intervention-context", "intervene-control",
            "stop-control", "conversation-status", "conversation-context",
      ];
      const nodes = Object.fromEntries(ids.map((id) => [id, new Node(
        id === "usage-modal" ? "dialog" : id.includes("toggle") || id.includes("submit") || id === "bootstrap-retry" || id === "new-chat" ? "button" : id === "composer" || id === "usage-credits-form" ? "form" : "div",
        id,
      )]));
      nodes["message-input"].tag = "textarea";
      nodes["intervene-control"].tag = "button"; nodes["stop-control"].tag = "button";
      nodes["activity-audit"].tag = "details";
      nodes["composer-recipient"].tag = "select";
      nodes["composer-recipient"].value = "sol";
      nodes["composer-recipient"].options = ["fable", "sol", "team"].map((value) => {{
        const option = new Node("option"); option.value = value; return option;
      }});
      nodes["usage-credits-confirm"].tag = "input";
      nodes["task-drawer-toggle"].setAttribute("aria-expanded", "false");
      nodes["inspector-drawer-toggle"].setAttribute("aria-expanded", "false");
      nodes["usage-modal"].append(nodes["usage-credits-form"]);
      nodes["usage-credits-form"].append(
        nodes["usage-credits-confirm"], nodes["usage-credits-acknowledge"],
        nodes["usage-error"], nodes["bootstrap-retry"],
      );
      const disabledDrawerControl = new Node("button", "disabled-drawer-control");
      disabledDrawerControl.disabled = true;
      nodes["project-navigation"].append(disabledDrawerControl, nodes["task-list"]);
      nodes["task-inspector-panel"].append(
        nodes["task-inspector"], nodes["task-inspector-summary"], nodes["task-controls"],
        nodes["task-inspector-empty"], nodes["activity-audit"],
      );
      nodes["composer"].append(
        nodes["composer-label"], nodes["message-input"], nodes["composer-recipient"],
        nodes["composer-submit"], nodes["composer-guidance"],
      );
      nodes["conversation-context"].append(
        nodes["intervention-context"], nodes["conversation-status"],
        nodes["intervene-control"], nodes["stop-control"],
      );
      nodes["conversation-shell"].append(nodes["conversation-context"], nodes["composer"]);
      const isDescendantOf = (node, ancestor) => {{
        for (let current = node; current; current = current.parent) {{
          if (current === ancestor) return true;
        }}
        return false;
      }};
      const hasInertAncestor = (node) => {{
        for (let current = node; current; current = current.parent) {{
          if (current.inert) return true;
        }}
        return false;
      }};
      interactionAllowed = (node) => (!nodes["usage-modal"].open
        || isDescendantOf(node, nodes["usage-modal"])) && !hasInertAncestor(node);
      const launcher = new Node("button", "launcher");
      documentRoot = {{
        activeElement: launcher,
        listeners: {{}},
        createElement(tag) {{ return new Node(tag); }},
        querySelector(selector) {{ return selector.startsWith("#") ? nodes[selector.slice(1)] ?? null : null; }},
        addEventListener(kind, listener) {{ (this.listeners[kind] ??= []).push(listener); }},
        async emit(kind, event) {{ for (const listener of this.listeners[kind] ?? []) await listener(event); }},
      }};

      const fetchCalls = [];
      let projectFetchCount = 0;
      let bootstrapFetchCount = 0;
      let interventionAttempts = 0;
      let acknowledgementAttempts = 0;
      let releaseReconnectBootstrap;
      const scheduled = [];
      class FakeSocket {{
        static instances = [];
        constructor(url) {{ this.url = url; this.listeners = {{}}; FakeSocket.instances.push(this); }}
        addEventListener(kind, listener) {{ this.listeners[kind] = listener; }}
        close() {{ this.closed = true; }}
      }}
      const media = {{
        matches: true,
        listeners: [],
        addEventListener(kind, listener) {{ if (kind === "change") this.listeners.push(listener); }},
        emit() {{ for (const listener of this.listeners) listener({{matches: this.matches}}); }},
      }};
      const windowRoot = {{
        location: {{protocol: "http:", host: "127.0.0.1:8765"}},
        WebSocket: FakeSocket,
        matchMedia: () => media,
        fetch: async (url, options = {{}}) => {{
          fetchCalls.push({{url, options}});
          if (url === "/api/projects") {{
            projectFetchCount += 1;
            if (projectFetchCount === 1) return {{ok: false, status: 503, json: async () => ({{}})}};
            return {{ok: true, status: 200, json: async () => projectPayload}};
          }}
          if (url === "/api/projects/project-a/chats?limit=50") {{
            return {{ok: true, status: 200, json: async () => chats}};
          }}
          if (url === "/api/projects/project-a/chats/session-1/bootstrap") {{
            bootstrapFetchCount += 1;
            if (bootstrapFetchCount === 2) {{
              return await new Promise((resolve) => {{ releaseReconnectBootstrap = resolve; }});
            }}
            return {{ok: true, status: 200, json: async () => projectBootstrap}};
          }}
          if (url.endsWith("/tasks/intervene-task/intervene")) {{
            interventionAttempts += 1;
            return interventionAttempts === 1
              ? {{ok: false, status: 409, json: async () => ({{detail: "task revision conflicts"}})}}
              : {{ok: true, status: 202, json: async () => ({{}})}};
          }}
          if (url.endsWith("/authorize-retry")) {{
            acknowledgementAttempts += 1;
            return acknowledgementAttempts === 1
              ? {{ok: false, status: 503, json: async () => ({{}})}}
              : acknowledgementAttempts === 4
                ? {{ok: false, status: 409, json: async () => ({{detail: "resume generation conflicts"}})}}
              : {{ok: true, status: 202, json: async () => ({{}})}};
          }}
          return {{ok: true, status: 202, json: async () => ({{}})}};
        }},
        setTimeout: (callback, delay) => {{ scheduled.push({{callback, delay}}); return scheduled.length; }},
        clearTimeout: () => {{}},
        addEventListener: () => {{}},
      }};

      const controller = bridge.startBrowserApp(documentRoot, windowRoot);
      if (nodes["usage-modal"].showModalCount !== 1) process.exit(2);
      if (documentRoot.activeElement !== nodes["usage-credits-confirm"]) process.exit(3);
      if (await controller.ready) process.exit(4);
      if (nodes["bootstrap-retry"].hidden || !nodes["message-input"].disabled) process.exit(5);
      if (nodes["task-inspector-empty"].hidden) process.exit(53);
      if (!interactionAllowed(nodes["bootstrap-retry"])) process.exit(25);
      if (!nodes["usage-error"].textContent.includes("No project chats")) process.exit(26);

      await nodes["bootstrap-retry"].emit("click");
      if (!(await controller.ready)) process.exit(6);
      if (nodes["message-input"].disabled || nodes["composer-submit"].disabled) process.exit(7);
      if (!nodes["task-inspector-summary"].textContent.includes("Question budget2 remaining · 1 consumed")) process.exit(34);
      if (!nodes["task-controls"].hidden) process.exit(49);
      if (!nodes["task-inspector-empty"].hidden) process.exit(52);
      if (nodes["usage-modal"].closeCount !== 1 || documentRoot.activeElement !== launcher) process.exit(8);
      media.matches = false;
      media.emit();
      if (nodes["repository-status"].textContent !== "Project: PROJECT-A · Branch: feat/agent-bridge") {{
        process.exit(9);
      }}

      const approve = nodes["task-inspector"].children[0].querySelector("button");
      await approve.emit("click");
      await Promise.resolve();
      const approval = fetchCalls.find((call) => call.url === "/api/projects/project-a/chats/session-1/tasks/task-1/approve");
      if (approval.options.body !== '{{"revision":1}}') process.exit(10);
      if (approval.options.headers["X-CSRF-Token"] !== {json.dumps(CSRF_TOKEN)}) process.exit(11);

      nodes["message-input"].value = unsafe;
      await nodes["composer"].emit("submit");
      const message = fetchCalls.find((call) => call.url === "/api/projects/project-a/chats/session-1/messages");
      if (JSON.parse(message.options.body).text !== unsafe) process.exit(12);
      if (message.url.includes(unsafe)) process.exit(13);
      const socket = FakeSocket.instances[0];
      socket.listeners.message({{data: JSON.stringify(persistedEvent)}});
      socket.listeners.message({{data: JSON.stringify(persistedEvent)}});
      if (controller.state.tasks[0].state !== "sol_running") process.exit(21);
      if (controller.state.tasks[0].history.length !== 1) process.exit(22);
      if (nodes["sol-status"].textContent !== "Sol · Running") process.exit(23);
          const projectButton = nodes["project-list"].children[0].children[0];
          const chatButton = nodes["chat-list"].children[0].children[0];
          await projectButton.emit("click");
          await chatButton.emit("click");
          projectButton.focus();
          socket.listeners.message({{data: JSON.stringify({{...persistedEvent, sequence: persistedEvent.sequence + 1}})}});
          await flush();
      if (documentRoot.activeElement !== projectButton) process.exit(29);
      projectPayload = {{...projectPayload, active_lease: {{project_id: "project-a", session_id: "session-1", task_id: "task-1"}}}};
          socket.listeners.message({{data: JSON.stringify({{...persistedEvent, sequence: persistedEvent.sequence + 2}})}});
          await flush();
      if (!nodes["new-chat"].disabled || !nodes["project-list"].children[0].children[0].disabled) process.exit(30);
      projectPayload = {{...projectPayload, active_lease: null}};
          socket.listeners.message({{data: JSON.stringify({{...persistedEvent, sequence: persistedEvent.sequence + 3}})}});
          await flush();
      if (nodes["new-chat"].disabled || nodes["project-list"].children[0].children[0].disabled) process.exit(31);
      const reconnectFocus = nodes["chat-list"].children[0].children[0];
      reconnectFocus.focus();
      socket.listeners.close();
      scheduled[0].callback();
      const reconnecting = FakeSocket.instances[1].listeners.open();
      await Promise.resolve();
      if (!nodes["message-input"].disabled
          || !nodes["connection-status"].textContent.includes("refresh")) process.exit(27);
      releaseReconnectBootstrap({{ok: true, status: 200, json: async () => projectBootstrap}});
      await reconnecting;
      if (documentRoot.activeElement !== reconnectFocus) process.exit(24);
      if (nodes["message-input"].disabled) process.exit(28);

      media.matches = true;
      media.emit();
      const firstDrawerFocusable = nodes["task-list"].children[1].children[0].querySelector("button");
      await nodes["task-drawer-toggle"].emit("click");
      if (!nodes["project-navigation"].classList.contains("drawer-open") || !nodes["conversation-shell"].inert || nodes["project-navigation"].attributes.role !== "dialog" || nodes["project-navigation"].attributes["aria-modal"] !== "true") process.exit(14);
      if (nodes["task-drawer-toggle"].attributes["aria-expanded"] !== "true") process.exit(15);
      if (documentRoot.activeElement !== firstDrawerFocusable) process.exit(35);
      launcher.focus();
      await documentRoot.emit("keydown", {{key: "Tab", preventDefault() {{ this.prevented = true; }}}});
      if (documentRoot.activeElement !== firstDrawerFocusable) process.exit(36);
      const blockedCalls = fetchCalls.length;
      nodes["message-input"].value = "This cannot send while navigation is inert.";
      await nodes["composer"].emit("submit");
      if (fetchCalls.length !== blockedCalls) process.exit(54);
      await documentRoot.emit("keydown", {{key: "Escape", preventDefault() {{}}}});
      if (nodes["project-navigation"].classList.contains("drawer-open") || nodes["project-navigation"].attributes.role !== undefined || nodes["project-navigation"].attributes["aria-modal"] !== undefined) process.exit(16);
      if (documentRoot.activeElement !== nodes["task-drawer-toggle"]) process.exit(17);
      if (globalThis.pwned === true) process.exit(18);
      await nodes["inspector-drawer-toggle"].emit("click");
      nodes["activity-audit"].open = true;
      await nodes["activity-audit"].emit("toggle");
      if (!nodes["activity-audit"].textContent.includes("Agent Event") || nodes["activity-audit"].textContent.includes(hostileActivity)) process.exit(37);
      if (!nodes["activity-audit"].textContent.includes("TypeAgent Event")) process.exit(50);
      media.matches = false;
      media.emit();
      if (nodes["project-navigation"].inert || nodes["task-inspector-panel"].inert || nodes["conversation-shell"].inert) process.exit(19);
      media.matches = true;
      media.emit();
      await nodes["task-drawer-toggle"].emit("click");
      const taskButton = nodes["task-list"].children[1].children[1].querySelector("button");
      await taskButton.emit("click");
      if (!nodes["project-navigation"].inert || nodes["project-navigation"].classList.contains("drawer-open") || nodes["conversation-shell"].inert) process.exit(20);
      if (!nodes["activity-audit"].textContent.includes("No structured activity recorded") || nodes["activity-audit"].textContent.includes(hostileActivity)) process.exit(38);
      projectBootstrap.fable_ready = false;
      projectBootstrap.fable_status = "unavailable";
      projectBootstrap.sol_status = "unavailable";
      const activeSocket = FakeSocket.instances.at(-1);
      const refreshFromEvent = async (sequence) => {{
        activeSocket.listeners.message({{data: JSON.stringify({{...persistedEvent, sequence, task_id: "intervene-task", kind: "conversation", payload: {{text: "refresh"}}}})}});
        await flush();
        scheduled.at(-1).callback();
        await flush();
      }};
      await refreshFromEvent(persistedEvent.sequence + 50);
      const safeDigest = "a".repeat(64);
      media.matches = false;
      media.emit();
      const taskOneButton = nodes["task-list"].children[1].children[0].querySelector("button");
      await taskOneButton.emit("click");
      for (const [index, [kind, expected]] of [
        ["action_error", "Action Error"], ["stop_error", "Stop Error"],
        ["agent_event", "Agent Event"], ["resume_drift", "Resume Drift"],
      ].entries()) {{
        projectBootstrap.tasks[0] = {{...projectBootstrap.tasks[0], activity_kind: kind, activity: {{
          status: "completed", command_sha256: safeDigest, run_id: hostileActivity,
          raw_output: hostileActivity, extra: hostileActivity,
        }}}};
        await refreshFromEvent(persistedEvent.sequence + 60 + index);
        const auditText = nodes["activity-audit"].textContent;
        if (!auditText.includes(expected) || auditText.includes(hostileActivity)) process.exit(55 + index);
        if (kind !== "agent_event" && (auditText.includes("Completed") || auditText.includes(safeDigest))) process.exit(56);
        if (kind === "agent_event" && (!auditText.includes("Completed") || !auditText.includes(safeDigest))) process.exit(57);
      }}
      await nodes["task-list"].children[1].children[1].querySelector("button").emit("click");
      if (controller.state.gate.canCompose || !nodes["message-input"].disabled) process.exit(41);
      await nodes["intervene-control"].emit("click");
      if (nodes["composer-recipient"].value !== "fable" || !nodes["composer-recipient"].options[1].disabled) process.exit(47);
      nodes["message-input"].value = "Keep scope exact.";
      await nodes["message-input"].emit("input");
      if (nodes["composer-submit"].disabled) process.exit(32);
      await nodes["composer"].emit("submit");
      const intervention = fetchCalls.find((call) => call.url === "/api/projects/project-a/chats/session-1/tasks/intervene-task/intervene");
      if (!intervention || JSON.parse(intervention.options.body).message !== "Keep scope exact.") process.exit(33);
      if (!nodes["message-input"].disabled || !nodes["composer-submit"].disabled) process.exit(39);
      if (!nodes["toast-region"].textContent.includes("task revision conflicts")) process.exit(51);
      await nodes["intervene-control"].emit("click");
      nodes["message-input"].value = "Use the fresh intervention identity.";
      await nodes["message-input"].emit("input");
      await nodes["composer"].emit("submit");
      const interventionRequests = fetchCalls.filter((call) => call.url === intervention.url);
      const firstPayload = JSON.parse(interventionRequests[0].options.body);
      const secondPayload = JSON.parse(interventionRequests[1].options.body);
      if (interventionRequests.length !== 2 || firstPayload.intervention_id === secondPayload.intervention_id || secondPayload.message !== "Use the fresh intervention identity.") process.exit(40);
      projectBootstrap.tasks[1] = {{...projectBootstrap.tasks[1], state: "completed", intervention: {{
        intervention_id: "unknown-a", status: "resume_outcome_unknown", resume_generation: 4,
        warning: "may have executed", eligible: false,
      }}}};
      await refreshFromEvent(persistedEvent.sequence + 101);
      const warningFocuses = nodes["intervention-context"].focusCount;
      const firstAcknowledge = nodes["conversation-context"].querySelector("#intervention-acknowledge-control");
      await firstAcknowledge.emit("click");
      const acknowledgements = () => fetchCalls.filter((call) => call.url.endsWith("/authorize-retry"));
      const firstAcknowledgement = JSON.parse(acknowledgements()[0].options.body);
      projectBootstrap.tasks[1].intervention = {{...projectBootstrap.tasks[1].intervention, resume_generation: 5}};
      await refreshFromEvent(persistedEvent.sequence + 102);
      if (nodes["intervention-context"].focusCount <= warningFocuses) process.exit(43);
      await nodes["conversation-context"].querySelector("#intervention-acknowledge-control").emit("click");
      const secondAcknowledgement = JSON.parse(acknowledgements()[1].options.body);
      if (firstAcknowledgement.acknowledgment_id === secondAcknowledgement.acknowledgment_id || secondAcknowledgement.expected_resume_generation !== 5 || secondAcknowledgement.acknowledge_possible_prior_execution !== true) process.exit(42);
      const focusBeforeNewGeneration = nodes["intervention-context"].focusCount;
      projectBootstrap.tasks[1].intervention = {{...projectBootstrap.tasks[1].intervention, resume_generation: 6}};
      await refreshFromEvent(persistedEvent.sequence + 103);
      if (nodes["intervention-context"].focusCount <= focusBeforeNewGeneration || focusBeforeNewGeneration < warningFocuses) process.exit(43);
      await nodes["conversation-context"].querySelector("#intervention-acknowledge-control").emit("click");
      const thirdAcknowledgement = JSON.parse(acknowledgements()[2].options.body);
      if (thirdAcknowledgement.acknowledgment_id === secondAcknowledgement.acknowledgment_id || thirdAcknowledgement.expected_resume_generation !== 6) process.exit(44);
      projectBootstrap.tasks[1].intervention = {{...projectBootstrap.tasks[1].intervention, resume_generation: 7}};
      await refreshFromEvent(persistedEvent.sequence + 104);
      await nodes["conversation-context"].querySelector("#intervention-acknowledge-control").emit("click");
      const conflictAcknowledgement = JSON.parse(acknowledgements()[3].options.body);
      await nodes["conversation-context"].querySelector("#intervention-acknowledge-control").emit("click");
      const freshAcknowledgement = JSON.parse(acknowledgements()[4].options.body);
      if (conflictAcknowledgement.acknowledgment_id === freshAcknowledgement.acknowledgment_id || freshAcknowledgement.expected_resume_generation !== 7 || freshAcknowledgement.acknowledge_possible_prior_execution !== true) process.exit(48);
      for (const status of ["pending_stop", "ready", "resuming", "resume_outcome_unknown"]) {{
        projectBootstrap.tasks[1].intervention = {{
          intervention_id: `stop-${{status}}`, status, resume_generation: 6, eligible: false,
        }};
        await refreshFromEvent(persistedEvent.sequence + 105 + acknowledgements().length);
        if (nodes["stop-control"].hidden || nodes["stop-control"].disabled) process.exit(45);
        await nodes["stop-control"].emit("click");
      }}
      const stopRequests = fetchCalls.filter((call) => call.url.endsWith("/tasks/intervene-task/stop"));
      if (stopRequests.length !== 4 || stopRequests.some((call) => call.options.body !== "null")) process.exit(46);
      projectBootstrap.fable_ready = true;
      projectBootstrap.fable_status = "subscription_ready";
      projectBootstrap.sol_status = "ready";
      projectBootstrap.tasks[1] = {{...projectBootstrap.tasks[1], state: "fable_planning", intervention: {{
        intervention_id: "safe-resume", status: "ready", resume_generation: 8, eligible: true,
      }}}};
      await refreshFromEvent(persistedEvent.sequence + 140);
      await nodes["intervene-control"].emit("click");
      await flush();
      const safeResume = fetchCalls.find((call) => call.url.endsWith("/interventions/safe-resume/resume"));
      if (!safeResume || safeResume.options.body !== '{{"expected_resume_generation":8}}'
          || !nodes["conversation-status"].textContent.includes("resume accepted")) process.exit(67);

      projectBootstrap.tasks[0] = {{...projectBootstrap.tasks[0], revision: 1,
        continuation_generation: 8, state: "awaiting_user_input", intervention: null,
        pending_question: {{question_id: "browser-question-1", asked_by: "sol", addressed_to: "user",
          routed_to: "user", text: "Which exact approved option should Sol use?",
          revision: 1, continuation_generation: 8}},
        exchange_permission: {{request_id: "browser-grant-1", revision: 1, continuation_generation: 8}},
      }};
      await refreshFromEvent(persistedEvent.sequence + 141);
      await nodes["task-list"].children[1].children[0].querySelector("button").emit("click");
      const actionCards = nodes["conversation"].children.filter((node) => node.className === "conversation-action-card");
      const questionCard = actionCards.find((card) => card.textContent.includes("Which exact approved option"));
      const permissionCard = actionCards.find((card) => card.textContent.includes("Automatic exchange limit"));
      if (!questionCard || !permissionCard || !questionCard.textContent.includes("Sol → You")
          || !permissionCard.textContent.includes("Allow 3 more exchanges")) process.exit(68);
      await questionCard.querySelector("button").emit("click");
      nodes["message-input"].value = "Use the exact option already approved.";
      await nodes["composer"].emit("submit");
      await flush();
      const answer = fetchCalls.find((call) => call.url.endsWith("/tasks/task-1/answer"));
      if (!answer || answer.options.body !== '{{"text":"Use the exact option already approved.","revision":1,"question_id":"browser-question-1","continuation_generation":8}}') process.exit(69);
      const refreshedPermission = nodes["conversation"].children.find((card) => card.className === "conversation-action-card" && card.textContent.includes("Automatic exchange limit"));
      await refreshedPermission.querySelectorAll("button").at(-1).emit("click");
      await flush();
      const grant = fetchCalls.find((call) => call.url.endsWith("/tasks/task-1/exchanges/grant"));
      if (!grant || grant.options.body !== '{{"revision":1,"continuation_generation":8,"request_id":"browser-grant-1"}}') process.exit(70);

      projectBootstrap.tasks[0] = {{...projectBootstrap.tasks[0], revision: 2,
        continuation_generation: 9, state: "awaiting_user_approval", pending_question: null,
        exchange_permission: null, brief: {{...projectBootstrap.tasks[0].brief, revision: 2}},
      }};
      await refreshFromEvent(persistedEvent.sequence + 142);
      await nodes["task-list"].children[1].children[0].querySelector("button").emit("click");
      await nodes["task-inspector"].children[0].querySelector("button").emit("click");
      await flush();
      const approvals = fetchCalls.filter((call) => call.url.endsWith("/tasks/task-1/approve"));
      if (approvals.at(-1).options.body !== '{{"revision":2}}') process.exit(71);
    """
    result = subprocess.run(
        [
            "node",
            "--experimental-default-type=module",
            "--input-type=module",
            "-e",
            harness,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_browser_startup_creates_first_chat_without_a_bootstrap_and_ignores_stale_new_chat(
) -> None:
    module_uri = Path("src/agent_bridge/static/app.js").resolve().as_uri()
    harness = f"""
      import * as bridge from {json.dumps(module_uri)};
      let documentRoot;
      let interactionAllowed = () => true;
      class ClassList {{
        constructor() {{ this.values = new Set(); }}
        add(value) {{ this.values.add(value); }}
        remove(value) {{ this.values.delete(value); }}
        contains(value) {{ return this.values.has(value); }}
        toggle(value, force) {{ force ? this.add(value) : this.remove(value); }}
      }}
      class Node {{
        constructor(tag, id = "") {{
          this.tag = tag; this.id = id; this.children = []; this.attributes = {{}};
          this.dataset = {{}}; this.classList = new ClassList(); this.listeners = {{}};
          this._text = ""; this.disabled = false; this.hidden = false; this.open = false;
          this.value = ""; this.checked = false; this.inert = false;
        }}
        set textContent(value) {{ this._text = String(value); this.children = []; }}
        get textContent() {{ return this._text; }}
        append(...children) {{ for (const child of children) {{ child.parent = this; this.children.push(child); }} }}
        replaceChildren(...children) {{ this.children = []; this.append(...children); this._text = ""; }}
        removeChild(child) {{ this.children.splice(this.children.indexOf(child), 1); }}
        remove() {{ this.parent?.removeChild(this); }}
        setAttribute(name, value) {{ this.attributes[name] = String(value); }}
        removeAttribute(name) {{ delete this.attributes[name]; }}
        addEventListener(kind, listener) {{ (this.listeners[kind] ??= []).push(listener); }}
        async emit(kind, event = {{}}) {{
          if (!interactionAllowed(this)) return;
          event.preventDefault ??= () => {{}};
          for (const listener of this.listeners[kind] ?? []) await listener(event);
          await this.onclick?.(event);
        }}
        focus() {{ documentRoot.activeElement = this; }}
        showModal() {{ this.open = true; }}
        close() {{ this.open = false; }}
        querySelector() {{ return this.children.find((child) => child.tag === "button") ?? null; }}
      }}
      const ids = [
        "task-list", "conversation", "task-inspector", "composer", "message-input",
        "composer-submit", "composer-guidance", "usage-modal", "usage-credits-form",
        "usage-credits-confirm", "usage-credits-acknowledge", "usage-error", "toast-region",
        "fable-status", "sol-status", "repository-status", "connection-status",
        "task-drawer-toggle", "inspector-drawer-toggle", "bootstrap-retry", "project-list",
        "chat-list", "new-chat", "selected-project-name", "selected-chat-name",
      ];
      const nodes = Object.fromEntries(ids.map((id) => [id, new Node(
        id === "usage-modal" ? "dialog" : id === "composer" || id === "usage-credits-form" ? "form" : id.includes("button") || id.includes("toggle") || id.includes("submit") || id === "new-chat" || id === "bootstrap-retry" ? "button" : "div",
        id,
      )]));
      nodes["message-input"].tag = "textarea";
      nodes["usage-credits-confirm"].tag = "input";
      nodes["usage-modal"].append(nodes["usage-credits-form"]);
      nodes["usage-credits-form"].append(nodes["usage-credits-confirm"], nodes["usage-credits-acknowledge"], nodes["usage-error"], nodes["bootstrap-retry"]);
      const isDescendantOf = (node, ancestor) => {{
        for (let current = node; current; current = current.parent) {{
          if (current === ancestor) return true;
        }}
        return false;
      }};
      interactionAllowed = (node) => !nodes["usage-modal"].open
        || isDescendantOf(node, nodes["usage-modal"]);
      const launcher = new Node("button", "launcher");
      documentRoot = {{
        activeElement: launcher, listeners: {{}},
        createElement(tag) {{ return new Node(tag); }},
        querySelector(selector) {{ return selector.startsWith("#") ? nodes[selector.slice(1)] ?? null : null; }},
        addEventListener(kind, listener) {{ (this.listeners[kind] ??= []).push(listener); }},
      }};
      const calls = [];
      const creates = [];
      let acknowledgementFailures = 1;
      let acknowledged = false;
      let createCount = 0;
      const scheduled = [];
      class Socket {{
        static instances = [];
        constructor(url) {{ this.url = url; this.listeners = {{}}; Socket.instances.push(this); }}
        addEventListener(kind, listener) {{ this.listeners[kind] = listener; }}
        close() {{ this.closed = true; }}
      }}
      const windowRoot = {{
        location: {{protocol: "http:", host: "bridge.test"}}, WebSocket: Socket,
        matchMedia: () => ({{matches: false, addEventListener() {{}}}}),
        setTimeout(callback, delay) {{ scheduled.push({{callback, delay}}); return scheduled.length; }}, clearTimeout() {{}}, addEventListener() {{}},
        fetch(url, options = {{}}) {{
          calls.push({{url, options}});
          if (url === "/api/projects") return Promise.resolve({{ok: true, status: 200, json: async () => ({{
            csrf_token: "csrf-empty", usage_credits_acknowledged: acknowledged, projects: [
              {{project_id: "empty", label: "Empty", branch: "main", readiness: {{fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}}}},
              {{project_id: "other", label: "Other", branch: "next", readiness: {{fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}}}},
            ], active_lease: null,
          }})}});
          if (url === "/api/settings/usage-credits-acknowledgement" && options.method === "POST") {{
            if (acknowledgementFailures > 0) {{
              acknowledgementFailures -= 1;
              return Promise.resolve({{ok: false, status: 503, json: async () => ({{}})}});
            }}
            acknowledged = true;
            return Promise.resolve({{ok: true, status: 202, json: async () => ({{}})}});
          }}
          if (url === "/api/projects/empty/chats?limit=50") return Promise.resolve({{ok: true, status: 200, json: async () => ({{chats: []}})}});
          if (url === "/api/projects/other/chats?limit=50") return Promise.resolve({{ok: true, status: 200, json: async () => ({{chats: [{{session_id: "other-chat", title: "Other chat", latest_sequence: 0}}]}})}});
          if (url === "/api/projects/empty/chats/first-empty/bootstrap") return Promise.resolve({{ok: true, status: 200, json: async () => ({{csrf_token: "csrf-empty", usage_credits_acknowledged: true, project_id: "empty", session_id: "first-empty", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "main", replay_after: 0, tasks: []}})}});
          if (url === "/api/projects/other/chats/other-chat/bootstrap") return Promise.resolve({{ok: true, status: 200, json: async () => ({{csrf_token: "csrf-empty", usage_credits_acknowledged: true, project_id: "other", session_id: "other-chat", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "next", replay_after: 0, tasks: []}})}});
          if (url === "/api/projects/empty/chats" && options.method === "POST") {{
            createCount += 1;
            if (createCount === 1) return Promise.resolve({{ok: true, status: 201, json: async () => ({{session_id: "first-empty", title: "First empty", latest_sequence: 0}})}});
            return new Promise((resolve) => creates.push(resolve));
          }}
          throw new Error(`unexpected ${{options.method ?? "GET"}} ${{url}}`);
        }},
      }};
      const app = bridge.startBrowserApp(documentRoot, windowRoot);
      if (!(await app.ready) || app.state.projectId !== "empty" || app.state.sessionId !== null || !nodes["usage-modal"].open) process.exit(2);
      if (documentRoot.activeElement !== nodes["usage-credits-confirm"] || nodes["fable-status"].textContent !== "Fable · Subscription · ready" || nodes["sol-status"].textContent !== "Sol · Ready") process.exit(7);
      await nodes["new-chat"].emit("click");
      await Promise.resolve();
      if (createCount !== 0 || !nodes["usage-modal"].open) process.exit(3);
      nodes["usage-credits-confirm"].checked = true;
      await nodes["usage-credits-confirm"].emit("change");
      await nodes["usage-credits-form"].emit("submit");
      if (!nodes["usage-modal"].open || app.state.gate.acknowledged || !nodes["usage-error"].textContent.includes("503")) process.exit(8);
      await nodes["usage-credits-form"].emit("submit");
      for (let tick = 0; tick < 12; tick += 1) await Promise.resolve();
      const acknowledgements = calls.filter((call) => call.url === "/api/settings/usage-credits-acknowledgement");
      if (nodes["usage-modal"].open || !app.state.gate.acknowledged || nodes["new-chat"].disabled || !interactionAllowed(nodes["new-chat"]) || documentRoot.activeElement !== launcher || acknowledgements.length !== 2 || acknowledgements.some((call) => call.options.headers["X-CSRF-Token"] !== "csrf-empty")) process.exit(9);

      await nodes["new-chat"].emit("click");
      for (let tick = 0; tick < 12; tick += 1) await Promise.resolve();
      const firstCreate = calls.find((call) => call.url === "/api/projects/empty/chats" && call.options.method === "POST");
      if (!firstCreate || firstCreate.options.headers["X-CSRF-Token"] !== "csrf-empty" || app.state.projectId !== "empty" || app.state.sessionId !== "first-empty" || Socket.instances.length !== 1 || !Socket.instances[0].url.includes("session_id=first-empty")) process.exit(5);

      await nodes["new-chat"].emit("click");
      await Promise.resolve();
      if (creates.length !== 1) process.exit(6);
      await nodes["project-list"].children[1].children[0].emit("click");
      for (let tick = 0; tick < 12; tick += 1) await Promise.resolve();
      creates.shift()({{ok: true, status: 201, json: async () => ({{session_id: "late-empty", title: "Late", latest_sequence: 0}})}});
      for (let tick = 0; tick < 12; tick += 1) await Promise.resolve();
      const activeSockets = Socket.instances.filter((socket) => !socket.closed);
      if (app.state.projectId !== "other" || app.state.sessionId !== "other-chat" || activeSockets.length !== 1 || scheduled.length !== 0 || documentRoot.activeElement !== launcher || !activeSockets[0].url.includes("project_id=other")) process.exit(4);
    """
    result = subprocess.run(
        ["node", "--experimental-default-type=module", "--input-type=module", "-e", harness],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/settings/usage-credits-acknowledgement", {"acknowledged": True}),
        (f"/api/sessions/{SESSION_ID}/messages", {"text": "plan it"}),
        ("/api/tasks/task-1/approve", {"revision": 1}),
        ("/api/tasks/task-1/edit", "EDITED_BRIEF"),
        ("/api/tasks/task-1/reject", None),
        ("/api/tasks/task-1/answer", {"answer": "use the existing seam"}),
        ("/api/tasks/task-1/stop", None),
        ("/api/tasks/task-1/resume", None),
    ],
)
def test_every_mutation_rejects_missing_and_wrong_csrf(
    web_harness: WebHarness,
    valid_brief: TaskBrief,
    path: str,
    body: object,
) -> None:
    payload = _edited_brief(valid_brief) if body == "EDITED_BRIEF" else body
    with _authenticated_client(web_harness) as client:
        assert client.post(path, json=payload).status_code == 403
        assert client.post(
            path,
            json=payload,
            headers={"X-CSRF-Token": "wrong"},
        ).status_code == 403
    assert web_harness.coordinator.calls == []
    assert web_harness.store.get_setting("usage_credits_acknowledged") is None


def test_approval_requires_the_latest_exact_revision_before_scheduling(
    web_harness: WebHarness,
) -> None:
    with _authenticated_client(web_harness) as client:
        csrf = _acknowledge_model_usage(client)
        response = client.post(
            "/api/tasks/task-1/approve",
            json={"revision": 2},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 409
        assert "latest exact revision" in response.json()["detail"]
        assert client.post(
            "/api/tasks/missing/approve",
            json={"revision": 1},
            headers={"X-CSRF-Token": CSRF_TOKEN},
        ).status_code == 404
        assert client.post(
            "/api/tasks/task-1/approve",
            json={"revision": True},
            headers={"X-CSRF-Token": CSRF_TOKEN},
        ).status_code == 422
    assert not any(call[0] == "approve" for call in web_harness.coordinator.calls)


def test_approval_rejects_unresolved_open_questions_before_scheduling(
    web_harness: WebHarness,
    valid_brief: TaskBrief,
) -> None:
    unresolved = replace(
        valid_brief,
        revision=2,
        open_questions=("Which path is authoritative?",),
    )
    web_harness.store.save_task(
        SESSION_ID,
        unresolved,
        TaskState.AWAITING_USER_APPROVAL,
    )
    with _authenticated_client(web_harness) as client:
        csrf = _acknowledge_model_usage(client)
        response = client.post(
            "/api/tasks/task-1/approve",
            json={"revision": 2},
            headers={"X-CSRF-Token": csrf},
        )
    assert response.status_code == 409
    assert response.json() == {
        "detail": "resolve the TaskBrief open questions before approval"
    }
    assert not any(call[0] == "approve" for call in web_harness.coordinator.calls)


def test_edit_reject_answer_stop_and_resume_schedule_exact_coordinator_calls(
    web_harness: WebHarness,
    valid_brief: TaskBrief,
) -> None:
    edited = _edited_brief(valid_brief)
    actions = (
        ("/api/tasks/task-1/approve", {"revision": 1}),
        ("/api/tasks/task-1/edit", edited),
        ("/api/tasks/task-1/reject", None),
        ("/api/tasks/task-1/answer", {"answer": "use option one"}),
        ("/api/tasks/task-1/stop", None),
        ("/api/tasks/task-1/resume", None),
    )
    with _authenticated_client(web_harness) as client:
        headers = {"X-CSRF-Token": _acknowledge_model_usage(client)}
        for path, body in actions:
            assert client.post(path, json=body, headers=headers).status_code == 202
        _wait_until(lambda: len(web_harness.coordinator.calls) == len(actions))

    assert ("approve", "task-1", 1) in web_harness.coordinator.calls
    assert ("edit", "task-1", edited) in web_harness.coordinator.calls
    assert ("reject", "task-1") in web_harness.coordinator.calls
    assert ("answer", "task-1", "use option one") in web_harness.coordinator.calls
    assert ("stop", "task-1") in web_harness.coordinator.calls
    assert ("resume", "task-1") in web_harness.coordinator.calls


def test_edit_rejects_wrong_identity_revision_and_repository_root_field(
    web_harness: WebHarness,
    valid_brief: TaskBrief,
) -> None:
    with _authenticated_client(web_harness) as client:
        headers = {"X-CSRF-Token": _csrf(client)}
        wrong_task = _edited_brief(valid_brief)
        wrong_task["task_id"] = "another-task"
        assert client.post(
            "/api/tasks/task-1/edit", json=wrong_task, headers=headers
        ).status_code == 409

        stale = valid_brief.to_dict()
        assert client.post(
            "/api/tasks/task-1/edit", json=stale, headers=headers
        ).status_code == 409

        with_repo_root = _edited_brief(valid_brief)
        with_repo_root["repo_root"] = "/tmp/attacker-selected-repo"
        assert client.post(
            "/api/tasks/task-1/edit", json=with_repo_root, headers=headers
        ).status_code == 422

        assert client.post(
            "/api/tasks/task-1/stop?repo_root=/tmp/attacker-selected-repo",
            headers=headers,
        ).status_code == 422
    assert web_harness.coordinator.calls == []


def test_message_and_answer_text_are_body_only_and_preserved_verbatim(
    web_harness: WebHarness,
) -> None:
    text = "question? key=session-secret & path=/repo/<script>"
    with _authenticated_client(web_harness) as client:
        headers = {"X-CSRF-Token": _acknowledge_model_usage(client)}
        response = client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": text},
            headers=headers,
        )
        assert response.status_code == 202
        assert response.request.url.query == b""
        answer = client.post(
            "/api/tasks/task-1/answer",
            json={"answer": text},
            headers=headers,
        )
        assert answer.status_code == 202
        assert answer.request.url.query == b""
        _wait_until(lambda: len(web_harness.coordinator.calls) == 2)
    assert ("message", SESSION_ID, text) in web_harness.coordinator.calls
    assert ("answer", "task-1", text) in web_harness.coordinator.calls


def test_strict_request_bodies_reject_extra_fields_and_blank_text(
    web_harness: WebHarness,
) -> None:
    with _authenticated_client(web_harness) as client:
        headers = {"X-CSRF-Token": _csrf(client)}
        assert client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "plan", "repo_root": "/tmp/repo"},
            headers=headers,
        ).status_code == 422
        assert client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "   "},
            headers=headers,
        ).status_code == 422
        assert client.post(
            "/api/tasks/task-1/answer",
            json={"answer": ""},
            headers=headers,
        ).status_code == 422
    assert web_harness.coordinator.calls == []


def test_usage_credit_acknowledgement_is_true_only_and_persisted(
    web_harness: WebHarness,
) -> None:
    with _authenticated_client(web_harness) as client:
        headers = {"X-CSRF-Token": _csrf(client)}
        assert client.get("/api/bootstrap").json()[
            "usage_credits_acknowledged"
        ] is False
        assert client.get("/api/bootstrap").headers["cache-control"] == "no-store"
        assert client.post(
            "/api/settings/usage-credits-acknowledgement",
            json={"acknowledged": False},
            headers=headers,
        ).status_code == 422
        assert client.post(
            "/api/settings/usage-credits-acknowledgement",
            json={"acknowledged": True},
            headers=headers,
        ).status_code == 202
        assert client.get("/api/bootstrap").json()[
            "usage_credits_acknowledged"
        ] is True
    assert web_harness.store.get_setting("usage_credits_acknowledged") is True


def test_scheduled_coroutine_is_retained_until_it_finishes(
    web_harness: WebHarness,
) -> None:
    web_harness.coordinator.block_actions.add("message")
    with _authenticated_client(web_harness) as client:
        csrf = _acknowledge_model_usage(client)
        response = client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "wait for release"},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 202
        _wait_until(lambda: len(web_harness.app.state.active_coroutines) == 1)
        task = next(iter(web_harness.app.state.active_coroutines))
        assert task.done() is False
        web_harness.coordinator.release.set()
        _wait_until(lambda: web_harness.app.state.active_coroutines == set())


def test_lifespan_cancels_and_awaits_active_actions_without_error_event(
    web_harness: WebHarness,
) -> None:
    web_harness.coordinator.block_actions.add("message")
    client = TestClient(web_harness.app)
    with client:
        keyed = client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
        assert keyed.status_code == 303
        csrf = _acknowledge_model_usage(client)
        response = client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "still running at shutdown"},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 202
        _wait_until(lambda: len(web_harness.app.state.active_coroutines) == 1)

    assert web_harness.app.state.shutting_down is True
    assert web_harness.app.state.active_coroutines == set()
    assert not any(
        event.kind == "action_error"
        for event in web_harness.store.events_after(SESSION_ID, 0)
    )
    assert web_harness.app.state.coroutine_observation_failures == []


def test_new_actions_are_rejected_once_shutdown_has_started(
    web_harness: WebHarness,
) -> None:
    with _authenticated_client(web_harness) as client:
        csrf = _acknowledge_model_usage(client)
        web_harness.app.state.shutting_down = True
        response = client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "must not schedule"},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 503
        assert web_harness.app.state.active_coroutines == set()
    assert web_harness.coordinator.calls == []


def test_scheduled_coroutine_exception_becomes_a_safe_coordinator_event(
    web_harness: WebHarness,
) -> None:
    web_harness.coordinator.fail_actions.add("message")
    with _authenticated_client(web_harness) as client:
        csrf = _acknowledge_model_usage(client)
        response = client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "trigger failure"},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 202
        _wait_until(
            lambda: any(
                event.kind == "action_error"
                for event in web_harness.store.events_after(SESSION_ID, 0)
            )
        )
        _wait_until(lambda: web_harness.app.state.active_coroutines == set())

    errors = tuple(
        event
        for event in web_harness.store.events_after(SESSION_ID, 0)
        if event.kind == "action_error"
    )
    assert len(errors) == 1
    assert errors[0].actor == "coordinator"
    assert errors[0].payload == {
        "action": "message",
        "error_type": "RuntimeError",
    }
    assert "deliberately unsafe detail" not in str(errors[0].to_dict())
    assert web_harness.app.state.coroutine_observation_failures == []


def test_websocket_replays_only_events_after_sequence(
    web_harness: WebHarness,
) -> None:
    first = web_harness.store.append_event(
        SESSION_ID, None, "user", "message", {"text": "one"}
    )
    second = web_harness.store.append_event(
        SESSION_ID, None, "fable", "message", {"text": "two"}
    )
    with _authenticated_client(web_harness) as client:
        with client.websocket_connect(
            f"/ws?session_id={SESSION_ID}&after={first.sequence}"
        ) as socket:
            event = socket.receive_json()
    assert event["sequence"] == second.sequence
    assert event["payload"] == {"text": "two"}


def test_websocket_replay_reads_bounded_pages_but_preserves_explicit_cursor(
    web_harness: WebHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = tuple(
        web_harness.store.append_event(
            SESSION_ID, None, "coordinator", "message", {"text": str(index)}
        )
        for index in range(EXPECTED_EVENT_REPLAY_PAGE_SIZE + 5)
    )
    original_events_after = web_harness.store.events_after
    observed_limits: list[int | None] = []

    def recorded_events_after(
        session_id: str,
        sequence: int,
        *,
        limit: int | None = None,
    ) -> tuple[StreamEvent, ...]:
        observed_limits.append(limit)
        return original_events_after(session_id, sequence, limit=limit)

    monkeypatch.setattr(web_harness.store, "events_after", recorded_events_after)
    with _authenticated_client(web_harness) as client:
        with client.websocket_connect(
            f"/ws?session_id={SESSION_ID}&after=0"
        ) as socket:
            received = [socket.receive_json() for _ in events]

    assert [event["sequence"] for event in received] == [
        event.sequence for event in events
    ]
    assert observed_limits
    assert all(limit == EXPECTED_EVENT_REPLAY_PAGE_SIZE for limit in observed_limits)


def test_bootstrap_exposes_recent_replay_floor_for_fresh_browser(
    web_harness: WebHarness,
) -> None:
    events = tuple(
        web_harness.store.append_event(
            SESSION_ID, None, "coordinator", "message", {"text": str(index)}
        )
        for index in range(EXPECTED_MAX_INITIAL_REPLAY_EVENTS + 5)
    )
    with _authenticated_client(web_harness) as client:
        bootstrap = client.get("/api/bootstrap").json()

    assert bootstrap["replay_after"] == events[-EXPECTED_MAX_INITIAL_REPLAY_EVENTS - 1].sequence


def test_websocket_reconnect_uses_last_sequence_without_duplication(
    web_harness: WebHarness,
) -> None:
    first = web_harness.store.append_event(
        SESSION_ID, None, "user", "message", {"text": "one"}
    )
    second = web_harness.store.append_event(
        SESSION_ID, None, "fable", "message", {"text": "two"}
    )
    with _authenticated_client(web_harness) as client:
        with client.websocket_connect(
            f"/ws?session_id={SESSION_ID}&after=0"
        ) as socket:
            assert socket.receive_json()["sequence"] == first.sequence
            assert socket.receive_json()["sequence"] == second.sequence

        third = web_harness.store.append_event(
            SESSION_ID, None, "sol", "message", {"text": "three"}
        )
        with client.websocket_connect(
            f"/ws?session_id={SESSION_ID}&after={second.sequence}"
        ) as socket:
            replayed = socket.receive_json()
    assert replayed["sequence"] == third.sequence
    assert replayed["payload"] == {"text": "three"}


def test_websocket_replays_before_live_fanout_and_preserves_agent_text(
    web_harness: WebHarness,
) -> None:
    replay = web_harness.store.append_event(
        SESSION_ID, None, "coordinator", "status", {"text": "ready"}
    )
    web_harness.coordinator.emit_message_events = True
    unsafe_text = '<img src=x onerror="globalThis.pwned=true">'
    with _authenticated_client(web_harness) as client:
        csrf = _acknowledge_model_usage(client)
        with client.websocket_connect(
            f"/ws?session_id={SESSION_ID}&after=0"
        ) as socket:
            assert socket.receive_json()["sequence"] == replay.sequence
            accepted = client.post(
                f"/api/sessions/{SESSION_ID}/messages",
                json={"text": unsafe_text},
                headers={"X-CSRF-Token": csrf},
            )
            assert accepted.status_code == 202
            live = socket.receive_json()
    assert live["sequence"] > replay.sequence
    assert live["payload"] == {"text": f"planned: {unsafe_text}"}


def test_websocket_reads_reentrant_store_events_from_sqlite_in_sequence_order(
    web_harness: WebHarness,
) -> None:
    inner_events: list[StreamEvent] = []

    def append_inner(event: StreamEvent) -> None:
        if event.kind == "outer":
            inner_events.append(
                web_harness.store.append_event(
                    SESSION_ID,
                    None,
                    "coordinator",
                    "inner",
                    {"text": "inner"},
                )
            )

    listener_token = web_harness.store.add_event_listener(append_inner)
    try:
        with _authenticated_client(web_harness) as client:
            with client.websocket_connect(
                f"/ws?session_id={SESSION_ID}&after=0"
            ) as socket:
                outer = web_harness.store.append_event(
                    SESSION_ID,
                    None,
                    "coordinator",
                    "outer",
                    {"text": "outer"},
                )
                received = [socket.receive_json(), socket.receive_json()]
    finally:
        web_harness.store.remove_event_listener(listener_token)

    assert len(inner_events) == 1
    assert [event["sequence"] for event in received] == [
        outer.sequence,
        inner_events[0].sequence,
    ]
    assert [event["kind"] for event in received] == ["outer", "inner"]


def test_app_registers_store_live_listener_only_for_its_lifespan(
    web_harness: WebHarness,
) -> None:
    published: list[StreamEvent] = []
    original_publish = web_harness.broadcaster.publish

    def record_publish(event: StreamEvent) -> None:
        published.append(event)
        original_publish(event)

    web_harness.broadcaster.publish = record_publish  # type: ignore[method-assign]
    with TestClient(web_harness.app):
        live = web_harness.store.append_event(
            SESSION_ID, None, "coordinator", "status", {"text": "live"}
        )
        _wait_until(lambda: published == [live])

    web_harness.store.append_event(
        SESSION_ID, None, "coordinator", "status", {"text": "after shutdown"}
    )
    assert published == [live]


def test_websocket_rejects_invalid_session_id_and_negative_cursor(
    web_harness: WebHarness,
) -> None:
    with _authenticated_client(web_harness) as client:
        with pytest.raises(WebSocketDisconnect) as bad_session:
            with client.websocket_connect("/ws?session_id=agent%20text&after=0"):
                pass
        assert bad_session.value.code == 1008
        with pytest.raises(WebSocketDisconnect) as bad_cursor:
            with client.websocket_connect(
                f"/ws?session_id={SESSION_ID}&after=-1"
            ):
                pass
        assert bad_cursor.value.code == 1008

        for invalid in ("not-an-integer", str(2**63)):
            with pytest.raises(WebSocketDisconnect) as malformed:
                with client.websocket_connect(
                    f"/ws?session_id={SESSION_ID}&after={invalid}"
                ):
                    pass
            assert malformed.value.code == 1008

        with client.websocket_connect(
            f"/ws?session_id={SESSION_ID}&after={2**53 - 1}"
        ):
            pass
        with pytest.raises(WebSocketDisconnect) as unsafe_integer:
            with client.websocket_connect(
                f"/ws?session_id={SESSION_ID}&after={2**53}"
            ):
                pass
        assert unsafe_integer.value.code == 1008


def test_broadcaster_coalesces_5000_cross_thread_publishes_to_one_wakeup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = StreamEvent(
        sequence=1,
        session_id=SESSION_ID,
        task_id=None,
        actor="coordinator",
        kind="status",
        payload={"text": "wake"},
        created_at="2026-08-10T00:01:00Z",
    )
    broadcaster = InMemoryEventBroadcaster(max_queue_size=1)

    async def scenario() -> None:
        scheduled_calls = 0
        loop = asyncio.get_running_loop()
        original_call_soon_threadsafe = loop.call_soon_threadsafe

        def counted_call_soon_threadsafe(callback: Any, *args: Any) -> Any:
            nonlocal scheduled_calls
            scheduled_calls += 1
            return original_call_soon_threadsafe(callback, *args)

        monkeypatch.setattr(
            loop, "call_soon_threadsafe", counted_call_soon_threadsafe
        )
        subscription = broadcaster.subscribe(SESSION_ID)
        async with subscription:
            def publish_many() -> None:
                for _ in range(5000):
                    broadcaster.publish(event)

            publisher = threading.Thread(target=publish_many)
            publisher.start()
            publisher.join(timeout=2)
            assert publisher.is_alive() is False
            assert scheduled_calls == 1
            assert subscription._queue.qsize() == 0

            await asyncio.sleep(0)
            assert subscription._queue.qsize() == 1
            assert subscription._wake_pending is True
            assert await anext(subscription) == event
            assert subscription._queue.qsize() == 0
            assert subscription._wake_pending is False

        calls_before_cleanup_check = scheduled_calls
        after_disconnect = threading.Thread(target=lambda: broadcaster.publish(event))
        after_disconnect.start()
        after_disconnect.join(timeout=2)
        assert after_disconnect.is_alive() is False
        assert scheduled_calls == calls_before_cleanup_check
        assert broadcaster._subscribers == {}

    asyncio.run(scenario())


def test_broadcaster_rejects_unbounded_or_invalid_queue_sizes() -> None:
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="max_queue_size"):
            InMemoryEventBroadcaster(max_queue_size=invalid)


def test_live_broadcaster_ignores_other_sessions_and_duplicate_sequences(
    web_harness: WebHarness,
) -> None:
    existing = web_harness.store.append_event(
        SESSION_ID, None, "user", "message", {"text": "existing"}
    )
    with _authenticated_client(web_harness) as client:
        with client.websocket_connect(
            f"/ws?session_id={SESSION_ID}&after={existing.sequence}"
        ) as socket:
            duplicate = StreamEvent(
                sequence=existing.sequence,
                session_id=SESSION_ID,
                task_id=None,
                actor="coordinator",
                kind="status",
                payload={"text": "duplicate"},
                created_at="2026-08-10T00:01:00Z",
            )
            other = StreamEvent(
                sequence=existing.sequence + 1,
                session_id="other-session",
                task_id=None,
                actor="coordinator",
                kind="status",
                payload={"text": "other"},
                created_at="2026-08-10T00:01:01Z",
            )
            web_harness.broadcaster.publish(duplicate)
            web_harness.broadcaster.publish(other)
            live = web_harness.store.append_event(
                SESSION_ID,
                None,
                "coordinator",
                "status",
                {"text": "live"},
            )
            received = socket.receive_json()
    assert received["sequence"] == live.sequence
    assert received["payload"] == {"text": "live"}


@dataclass
class _HubCoordinator:
    store: SQLiteStore
    calls: list[tuple[object, ...]]

    async def edit_task(self, task_id: str, brief: TaskBrief) -> None:
        self.calls.append(("edit", task_id, brief.revision))

    async def reject_task(self, task_id: str) -> None:
        self.calls.append(("reject", task_id))


@dataclass
class _HubRuntime:
    project_id: str
    label: str
    repository: str
    branch: str
    store: SQLiteStore
    coordinator: _HubCoordinator
    broadcaster: InMemoryEventBroadcaster
    readiness: object


@dataclass
class _Prepared:
    preparation_id: str
    project_id: str
    session_id: str
    task_id: str
    revision: int = 0


class _HubWorkflows:
    """Route-level fake: persistence stays in the real selected SQLite store."""

    def __init__(self, runtimes: dict[str, _HubRuntime]) -> None:
        self.runtimes = runtimes
        self.prepared: list[_Prepared] = []
        self.preparation_calls: list[tuple[str, dict[str, object]]] = []
        self.runs: list[str] = []
        self.aborts: list[tuple[str, str]] = []
        self.stops: list[tuple[str, str, str]] = []
        self.reject_preparation = False
        self.probe_calls = 0
        self.active_lease: LeaseToken | None = None
        self.run_failure = False
        self.block_run = False

    def active_lease_snapshot(self) -> LeaseToken | None:
        return self.active_lease

    def require_no_active_lease(self) -> None:
        if self.active_lease is not None:
            raise RuntimeError("another workflow already owns the active agent lease")

    def require_navigation_allowed(
        self, *, project_id: str, session_id: str,
    ) -> LeaseToken | None:
        token = self.active_lease
        if token is not None and (
            token.project_id != project_id or token.session_id != session_id
        ):
            raise RuntimeError("active workflow belongs to another project or chat")
        return token

    def reserve_stop(
        self, *, project_id: str, session_id: str, task_id: str,
    ) -> LeaseToken:
        token = self.active_lease
        if (
            token is None
            or token.project_id != project_id
            or token.session_id != session_id
            or token.task_id != task_id
        ):
            raise RuntimeError("stop requires the exact active workflow")
        return token

    def cancel_stop_reservation(self, reservation: LeaseToken) -> None:
        if self.active_lease != reservation:
            raise RuntimeError("stop requires the exact active workflow")

    def _prepare(
        self,
        *,
        action: str,
        project_id: str,
        session_id: str,
        task_id: str | None,
        text: str | None = None,
    ) -> _Prepared:
        self.require_no_active_lease()
        if self.reject_preparation:
            raise RuntimeError("another workflow already owns the active agent lease")
        self.probe_calls += 1
        if text is not None:
            self.runtimes[project_id].store.append_event(
                session_id, None, "user", "message", {"text": text}
            )
        prepared = _Prepared(
            preparation_id=f"{action}-{len(self.prepared) + 1}",
            project_id=project_id,
            session_id=session_id,
            task_id=(task_id or f"task-{len(self.prepared) + 1}"),
        )
        self.prepared.append(prepared)
        self.active_lease = LeaseToken(
            len(self.prepared), project_id, session_id, prepared.task_id,
        )
        return prepared

    async def prepare_new_request(self, **kwargs: object) -> _Prepared:
        return self._prepare(
            action="new",
            project_id=str(kwargs["project_id"]),
            session_id=str(kwargs["session_id"]),
            task_id=None,
            text=str(kwargs["text"]),
        )

    async def prepare_approval(self, **kwargs: object) -> _Prepared:
        return self._prepare(
            action="approval",
            project_id=str(kwargs["project_id"]),
            session_id=str(kwargs["session_id"]),
            task_id=str(kwargs["task_id"]),
        )

    async def prepare_answer(self, **kwargs: object) -> _Prepared:
        return self._prepare(
            action="answer",
            project_id=str(kwargs["project_id"]),
            session_id=str(kwargs["session_id"]),
            task_id=str(kwargs["task_id"]),
        )

    async def prepare_resume(self, **kwargs: object) -> _Prepared:
        return self._prepare(
            action="resume",
            project_id=str(kwargs["project_id"]),
            session_id=str(kwargs["session_id"]),
            task_id=str(kwargs["task_id"]),
        )

    async def prepare_continuation_message(self, **kwargs: object) -> _Prepared:
        self._require_task(kwargs)
        self.preparation_calls.append(("continuation", dict(kwargs)))
        return self._prepare(
            action="continuation",
            project_id=str(kwargs["project_id"]),
            session_id=str(kwargs["session_id"]),
            task_id=str(kwargs["task_id"]),
        )

    async def prepare_question_answer(self, **kwargs: object) -> _Prepared:
        self._require_task(kwargs)
        self.preparation_calls.append(("question_answer", dict(kwargs)))
        return self._prepare(
            action="question_answer",
            project_id=str(kwargs["project_id"]),
            session_id=str(kwargs["session_id"]),
            task_id=str(kwargs["task_id"]),
        )

    async def prepare_exchange_grant(self, **kwargs: object) -> _Prepared:
        self._require_task(kwargs)
        self.preparation_calls.append(("exchange_grant", dict(kwargs)))
        return self._prepare(
            action="exchange_grant",
            project_id=str(kwargs["project_id"]),
            session_id=str(kwargs["session_id"]),
            task_id=str(kwargs["task_id"]),
        )

    def _require_task(self, kwargs: object) -> None:
        if not isinstance(kwargs, dict):
            raise ValueError("typed preparation arguments are invalid")
        try:
            task = self.runtimes[str(kwargs["project_id"])].store.get_task(
                str(kwargs["task_id"]), int(kwargs["revision"]),
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise LookupError("task not found") from error
        if task.session_id != kwargs["session_id"]:
            raise LookupError("task not found")

    async def run(self, prepared: _Prepared) -> None:
        self.runs.append(prepared.preparation_id)
        if self.active_lease is not None and self.active_lease.task_id == prepared.task_id:
            self.active_lease = None
        if self.block_run:
            await asyncio.Future()
        if self.run_failure:
            raise RuntimeError("deliberately unsafe prepared failure")

    def abort_prepared(self, prepared: _Prepared, *, reason: str) -> None:
        self.aborts.append((prepared.preparation_id, reason))

    async def stop(self, *, reservation: LeaseToken) -> None:
        if self.active_lease != reservation:
            raise RuntimeError("stop requires the exact active workflow")
        self.stops.append((
            reservation.project_id, reservation.session_id, reservation.task_id,
        ))
        self.active_lease = None


class _HubStoreFake:
    def __init__(self) -> None:
        self.acknowledged = False

    def usage_credits_acknowledged(self) -> bool:
        return self.acknowledged

    def acknowledge_usage_credits(self) -> None:
        self.acknowledged = True


@dataclass
class _ProbePlan:
    fable_results: list[object]
    sol_results: list[object]
    fable_calls: int = 0
    sol_calls: int = 0

    async def fable_probe(self) -> tuple[bool, str]:
        self.fable_calls += 1
        result = self.fable_results.pop(0)
        if result is _HANGING_PROBE:
            await asyncio.Future()
        if isinstance(result, BaseException):
            raise result
        return result  # type: ignore[return-value]

    async def sol_probe(self) -> str:
        self.sol_calls += 1
        result = self.sol_results.pop(0)
        if result is _HANGING_PROBE:
            await asyncio.Future()
        if isinstance(result, BaseException):
            raise result
        return result  # type: ignore[return-value]


@dataclass
class _RealHubHarness:
    app: Any
    runtimes: dict[str, _HubRuntime]
    workflows: HubWorkflowOrchestrator
    lease: ActiveAgentLease
    probes: dict[str, _ProbePlan]
    hub_store: _HubStoreFake

    def close(self) -> None:
        for runtime in self.runtimes.values():
            runtime.coordinator.close()
            runtime.store.close()


class _InterventionRunner(ProcessRunner):
    """Controlled process edge; the Hub, Coordinator, and Store stay real."""

    def __init__(self) -> None:
        super().__init__(stop_grace_seconds=0)
        self.stores: dict[str, SQLiteStore] = {}
        self.stops: list[str] = []
        self.stop_started = asyncio.Event()
        self.release_stop = asyncio.Event()
        self.block_stop = False

    async def stop(self, run_id: str, *, timeout_seconds: float) -> StopReceipt:
        assert timeout_seconds > 0
        self.stops.append(run_id)
        self.stop_started.set()
        if self.block_stop:
            await self.release_stop.wait()
        self.stores[run_id].finish_agent_run(
            run_id, status="interrupted", exit_code=-15,
        )
        return StopReceipt(run_id=run_id, was_running=True, process_exited=True)


def _real_hub_harness(
    tmp_path: Path,
    *,
    fable_results: tuple[object, ...] = ((True, "subscription_ready"),),
    sol_results: tuple[object, ...] = ("ready",),
    fable: object | None = None,
    repository: object | None = None,
    runner: object | None = None,
) -> _RealHubHarness:
    static_dir = tmp_path / "real-static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html><title>real hub</title>")
    runtimes: dict[str, _HubRuntime] = {}
    probes: dict[str, _ProbePlan] = {}
    for label in ("a", "b"):
        repo_root = tmp_path / f"repository-{label}"
        repo_root.mkdir()
        project_id = project_id_for_root(repo_root)
        store = SQLiteStore(
            tmp_path / f"real-{label}.sqlite3",
            clock=_Clock(),
            check_same_thread=False,
        )
        store.create_session("shared-chat", str(repo_root))
        probe = _ProbePlan(list(fable_results), list(sol_results))
        coordinator = Coordinator(
            store=store,
            repository=object() if repository is None else repository,
            runner=object() if runner is None else runner,
            fable=object() if fable is None else fable,
            sol=object(),
            ids=_RealIds(),
            repo_root=repo_root,
            repo_context="local test repository",
            trusted_shells={"sh": "/bin/sh"},
        )
        runtime = _HubRuntime(
            project_id=project_id,
            label=f"REAL-{label.upper()}",
            repository=str(repo_root),
            branch="main",
            store=store,
            coordinator=coordinator,  # type: ignore[arg-type]
            broadcaster=InMemoryEventBroadcaster(),
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(False, "checking", "checking"),
                fable_probe=probe.fable_probe,
                sol_probe=probe.sol_probe,
                timeout_seconds=0.05,
            ),
        )
        runtimes[label] = runtime
        probes[label] = probe
    hub_store = _HubStoreFake()
    hub_store.acknowledge_usage_credits()
    registry = ProjectRegistry(tuple(runtimes.values()))
    lease = ActiveAgentLease()
    workflows = HubWorkflowOrchestrator(
        registry=registry,
        lease=lease,
        usage_credits_acknowledged=hub_store.usage_credits_acknowledged,
    )
    app = create_hub_app(
        registry=registry,
        hub_store=hub_store,
        workflows=workflows,
        static_dir=static_dir,
        session_key=SESSION_KEY,
        csrf_token=CSRF_TOKEN,
    )
    return _RealHubHarness(app, runtimes, workflows, lease, probes, hub_store)


class _DelayedStopWorkflow:
    """Delay only the scheduled Stop body; all Hub logic stays real."""

    def __init__(self, inner: HubWorkflowOrchestrator) -> None:
        self._inner = inner
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    async def stop(self, **kwargs: object) -> None:
        self.entered.set()
        await self.release.wait()
        await self._inner.stop(**kwargs)  # type: ignore[arg-type]


class _BlockingFable:
    """Local provider edge fake used solely to hold a real prepared workflow."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def plan(self, **kwargs: object) -> object:
        self.started.set()
        await asyncio.Future()


@dataclass(frozen=True)
class _LocalBaseline:
    baseline_id: str
    allowed_paths: tuple[str, ...]


class _LocalRepository:
    """The narrow local repository edge required by real approval preparation."""

    def capture(self, brief: TaskBrief) -> _LocalBaseline:
        return _LocalBaseline("local-baseline", brief.allowed_paths)

    def baseline_manifest(self, baseline: _LocalBaseline) -> dict[str, object]:
        return {"baseline_id": baseline.baseline_id}

    def discard_baseline(self, baseline: _LocalBaseline) -> None:
        return None


@dataclass(frozen=True)
class _HubHarness:
    app: Any
    runtimes: dict[str, _HubRuntime]
    workflows: _HubWorkflows
    hub_store: _HubStoreFake
    static_dir: Path


@pytest.fixture
def hub_harness(tmp_path: Path, valid_brief: TaskBrief) -> _HubHarness:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html><title>hub</title>")
    runtimes: dict[str, _HubRuntime] = {}
    for project_id in ("project-a", "project-b"):
        store = SQLiteStore(
            tmp_path / f"{project_id}.sqlite3",
            clock=_Clock(),
            check_same_thread=False,
        )
        session_id = "chat-a" if project_id == "project-a" else "chat-b"
        store.create_session(session_id, f"/repositories/{project_id}")
        if project_id == "project-b":
            store.save_task(session_id, valid_brief, TaskState.AWAITING_USER_APPROVAL)
        coordinator = _HubCoordinator(store, [])
        runtimes[project_id] = _HubRuntime(
            project_id=project_id,
            label=project_id.upper(),
            repository=f"/repositories/{project_id}",
            branch="main",
            store=store,
            coordinator=coordinator,
            broadcaster=InMemoryEventBroadcaster(),
            readiness=SimpleNamespace(
                snapshot=lambda: RuntimeStatus(True, "subscription_ready", "ready")
            ),
        )
    workflows = _HubWorkflows(runtimes)
    hub_store = _HubStoreFake()
    app = create_hub_app(
        registry=ProjectRegistry(tuple(runtimes.values())),
        hub_store=hub_store,
        workflows=workflows,  # type: ignore[arg-type]
        static_dir=static_dir,
        session_key=SESSION_KEY,
        csrf_token=CSRF_TOKEN,
    )
    harness = _HubHarness(app, runtimes, workflows, hub_store, static_dir)
    try:
        yield harness
    finally:
        for runtime in runtimes.values():
            runtime.store.close()


def _authenticated_hub_client(harness: _HubHarness) -> TestClient:
    client = TestClient(harness.app)
    response = client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
    if response.status_code != 303:
        raise RuntimeError("hub authentication failed")
    return client


def test_hub_routes_resolve_project_before_any_foreign_chat_or_task_lookup(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing route-selected lookup would query or mutate project B."""
    foreign = hub_harness.runtimes["project-b"].store
    foreign_calls: list[str] = []

    def foreign_query(*args: object, **kwargs: object) -> object:
        foreign_calls.append("foreign")
        raise AssertionError("foreign store must not be queried")

    for name in (
        "session_exists", "chat", "latest_task", "get_task", "events_after",
        "latest_task_overviews", "browser_replay_floor", "append_event",
    ):
        monkeypatch.setattr(foreign, name, foreign_query)

    edited = _edited_brief(valid_brief)
    mutations = (
        ("/api/projects/project-a/chats/chat-b/messages", {"text": "plan"}),
        ("/api/projects/project-a/chats/chat-a/tasks/task-1/approve", {"revision": 1}),
        ("/api/projects/project-a/chats/chat-a/tasks/task-1/edit", edited),
        ("/api/projects/project-a/chats/chat-a/tasks/task-1/reject", None),
        ("/api/projects/project-a/chats/chat-a/tasks/task-1/answer", {"text": "yes", "revision": 1, "question_id": "question-1", "continuation_generation": 1}),
        ("/api/projects/project-a/chats/chat-a/tasks/task-1/stop", None),
        ("/api/projects/project-a/chats/chat-a/tasks/task-1/resume", None),
    )
    with _authenticated_hub_client(hub_harness) as client:
        headers = {"X-CSRF-Token": CSRF_TOKEN}
        for path, body in mutations:
            expected = 409 if path.endswith("/stop") else 404
            assert client.post(path, json=body, headers=headers).status_code == expected
        assert client.get(
            "/api/projects/project-a/chats/chat-b/bootstrap"
        ).status_code == 404
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                "/ws?project_id=project-a&session_id=chat-b&after=0"
            ):
                pass
    assert caught.value.code == 1008
    assert foreign_calls == []
    assert hub_harness.workflows.prepared == []


def test_hub_duplicate_chat_and_task_ids_remain_bound_to_the_route_project(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A global task lookup would run B when both projects use the same IDs."""
    shared_brief = replace(valid_brief, task_id="shared-task")
    for runtime in hub_harness.runtimes.values():
        runtime.store.create_session("shared-chat", runtime.repository)
        runtime.store.save_task(
            "shared-chat", shared_brief, TaskState.AWAITING_USER_APPROVAL
        )
    foreign = hub_harness.runtimes["project-b"].store

    def foreign_query(*args: object, **kwargs: object) -> object:
        raise AssertionError("duplicate identifiers must not select project B")

    monkeypatch.setattr(foreign, "latest_task", foreign_query)
    monkeypatch.setattr(foreign, "chat", foreign_query)
    hub_harness.hub_store.acknowledge_usage_credits()
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            "/api/projects/project-a/chats/shared-chat/tasks/shared-task/approve",
            json={"revision": 1},
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
        assert response.status_code == 202
        _wait_until(lambda: bool(hub_harness.workflows.runs))
    assert hub_harness.workflows.prepared[0].project_id == "project-a"
    assert hub_harness.workflows.prepared[0].session_id == "shared-chat"


def test_hub_directed_routes_require_exact_bound_request_shapes(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
) -> None:
    """Directed browser input must use its dedicated exact-bound endpoints."""
    runtime = hub_harness.runtimes["project-a"]
    task = replace(valid_brief, task_id="directed-task")
    runtime.store.save_task("chat-a", task, TaskState.AWAITING_USER_INPUT)
    headers = {"X-CSRF-Token": CSRF_TOKEN}
    base = "/api/projects/project-a/chats/chat-a/tasks/directed-task"
    with _authenticated_hub_client(hub_harness) as client:
        ordinary = client.post(
            "/api/projects/project-a/chats/chat-a/messages",
            json={"text": "please plan this", "addressed_to": "sol"},
            headers=headers,
        )
        _wait_until(lambda: not hub_harness.workflows.active_lease)
        continuation = client.post(
            f"{base}/messages",
            json={
                "text": "continue exactly this task",
                "addressed_to": "fable",
                "revision": 1,
                "continuation_generation": 1,
            },
            headers=headers,
        )
        _wait_until(lambda: not hub_harness.workflows.active_lease)
        answer = client.post(
            f"{base}/answer",
            json={
                "text": "the exact answer",
                "revision": 1,
                "question_id": "question-1",
                "continuation_generation": 1,
            },
            headers=headers,
        )
        _wait_until(lambda: not hub_harness.workflows.active_lease)
        grant = client.post(
            f"{base}/exchanges/grant",
            json={
                "revision": 1,
                "continuation_generation": 1,
                "request_id": "grant-1",
            },
            headers=headers,
        )
    assert ordinary.status_code == 202
    assert continuation.status_code == 202
    assert answer.status_code == 202
    assert grant.status_code == 202
    assert hub_harness.workflows.preparation_calls == [
        ("continuation", {"project_id": "project-a", "session_id": "chat-a", "task_id": "directed-task", "revision": 1, "continuation_generation": 1, "text": "continue exactly this task", "addressed_to": ConversationTarget.FABLE}),
        ("question_answer", {"project_id": "project-a", "session_id": "chat-a", "task_id": "directed-task", "revision": 1, "continuation_generation": 1, "question_id": "question-1", "answer": "the exact answer"}),
        ("exchange_grant", {"project_id": "project-a", "session_id": "chat-a", "task_id": "directed-task", "revision": 1, "continuation_generation": 1, "request_id": "grant-1"}),
    ]


@pytest.mark.parametrize(
    ("suffix", "payload"),
    (
        ("messages", {"text": " ", "addressed_to": "fable", "revision": 1, "continuation_generation": 1}),
        ("messages", {"text": "x", "addressed_to": "team", "revision": 1, "continuation_generation": 1}),
        ("messages", {"text": "x", "addressed_to": "fable", "revision": True, "continuation_generation": 1}),
        ("messages", {"text": "x", "addressed_to": "fable", "revision": 1, "continuation_generation": 1, "routed_to": "sol"}),
        ("answer", {"text": "x", "revision": 1, "question_id": "question-1", "continuation_generation": 1, "sender": "user"}),
        ("answer", {"text": "x", "revision": 1}),
        ("answer", {"text": "x", "revision": 1, "question_id": "bad/id", "continuation_generation": 1}),
        ("exchanges/grant", {"revision": 1, "continuation_generation": 1, "request_id": "grant-1", "continuation": "forbidden"}),
        ("exchanges/grant", {"revision": 1, "continuation_generation": "1", "request_id": "grant-1"}),
    ),
)
def test_hub_directed_requests_reject_untrusted_or_nonexact_browser_fields(
    hub_harness: _HubHarness,
    suffix: str,
    payload: dict[str, object],
) -> None:
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            f"/api/projects/project-a/chats/chat-a/tasks/task-1/{suffix}",
            json=payload,
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
    assert response.status_code == 422
    assert hub_harness.workflows.preparation_calls == []
    assert hub_harness.workflows.prepared == []


def test_hub_ordinary_message_rejects_directed_control_fields_and_oversized_text(
    hub_harness: _HubHarness,
) -> None:
    payloads = (
        {"text": "plan", "routed_to": "fable"},
        {"text": "plan", "sender": "user"},
        {"text": "plan", "continuation_generation": 1},
        {"text": "plan\nwith control"},
        {"text": "x" * (16 * 1024 + 1)},
    )
    with _authenticated_hub_client(hub_harness) as client:
        for payload in payloads:
            response = client.post(
                "/api/projects/project-a/chats/chat-a/messages",
                json=payload,
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
            assert response.status_code == 422
    assert hub_harness.workflows.prepared == []


@pytest.mark.parametrize("route", ("messages", "messages-bound", "answer"))
@pytest.mark.parametrize("invalid_text", ("secret /private/repository/token", "bad\x00control"))
def test_hub_task5_text_validation_never_reflects_untrusted_values_or_schedules(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
    route: str,
    invalid_text: str,
) -> None:
    runtime = hub_harness.runtimes["project-a"]
    runtime.store.save_task("chat-a", valid_brief, TaskState.AWAITING_USER_INPUT)
    if route == "messages":
        path = "/api/projects/project-a/chats/chat-a/messages"
        payload = {"text": invalid_text, "addressed_to": "fable", "sender": "user"}
    elif route == "messages-bound":
        path = "/api/projects/project-a/chats/chat-a/tasks/task-1/messages"
        payload = {
            "text": invalid_text, "addressed_to": "fable", "revision": 1,
            "continuation_generation": 1, "routed_to": "fable",
        }
    else:
        path = "/api/projects/project-a/chats/chat-a/tasks/task-1/answer"
        payload = {
            "text": invalid_text, "revision": 1, "question_id": "question-1",
            "continuation_generation": 1, "sender": "user",
        }
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            path, json=payload, headers={"X-CSRF-Token": CSRF_TOKEN},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid request"}
    assert invalid_text not in response.text
    assert "/private/repository/token" not in response.text
    assert hub_harness.workflows.prepared == []
    assert hub_harness.workflows.preparation_calls == []
    assert runtime.store.events_after("chat-a", 0) == ()
    assert hub_harness.app.state.active_coroutines == set()


@pytest.mark.parametrize("route", ("messages", "messages-bound", "answer"))
def test_hub_task5_text_validation_rejects_lone_surrogates_without_reflection(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
    route: str,
) -> None:
    hub_harness.runtimes["project-a"].store.save_task(
        "chat-a", valid_brief, TaskState.AWAITING_USER_INPUT,
    )
    if route == "messages":
        path = "/api/projects/project-a/chats/chat-a/messages"
        body = b'{"text":"\\ud800","addressed_to":"fable"}'
    elif route == "messages-bound":
        path = "/api/projects/project-a/chats/chat-a/tasks/task-1/messages"
        body = b'{"text":"\\ud800","addressed_to":"fable","revision":1,"continuation_generation":1}'
    else:
        path = "/api/projects/project-a/chats/chat-a/tasks/task-1/answer"
        body = b'{"text":"\\ud800","revision":1,"question_id":"question-1","continuation_generation":1}'
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            path,
            content=body,
            headers={"X-CSRF-Token": CSRF_TOKEN, "content-type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid request"}
    assert "ud800" not in response.text
    assert hub_harness.workflows.prepared == []
    assert hub_harness.workflows.preparation_calls == []
    assert hub_harness.runtimes["project-a"].store.events_after("chat-a", 0) == ()
    assert hub_harness.app.state.active_coroutines == set()


def test_hub_directed_duplicate_identifiers_stay_in_the_route_project(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 2's deferred route harness covers all directed bound endpoints."""
    foreign = hub_harness.runtimes["project-b"].store
    for runtime in hub_harness.runtimes.values():
        runtime.store.create_session("shared-directed-chat", runtime.repository)
        for task_id in ("shared-continuation", "shared-question", "shared-grant"):
            runtime.store.save_task(
                "shared-directed-chat", replace(valid_brief, task_id=task_id),
                TaskState.AWAITING_USER_INPUT,
            )

    def foreign_access(*args: object, **kwargs: object) -> object:
        raise AssertionError("directed route must not probe the foreign project")

    for name in ("session_exists", "get_task", "latest_task", "chat", "append_event"):
        monkeypatch.setattr(foreign, name, foreign_access)

    base = "/api/projects/project-a/chats/shared-directed-chat/tasks"
    with _authenticated_hub_client(hub_harness) as client:
        headers = {"X-CSRF-Token": CSRF_TOKEN}
        responses = []
        for suffix, payload in (
            ("shared-continuation/messages", {"text": "continue", "addressed_to": "fable", "revision": 1, "continuation_generation": 1}),
            ("shared-question/answer", {"text": "answer", "revision": 1, "question_id": "shared-question-id", "continuation_generation": 1}),
            ("shared-grant/exchanges/grant", {"revision": 1, "continuation_generation": 1, "request_id": "shared-grant-id"}),
        ):
            responses.append(client.post(f"{base}/{suffix}", json=payload, headers=headers))
            _wait_until(lambda: hub_harness.workflows.active_lease is None)
    assert [response.status_code for response in responses] == [202, 202, 202]
    assert [call[1]["project_id"] for call in hub_harness.workflows.preparation_calls] == [
        "project-a", "project-a", "project-a",
    ]
    assert all(
        prepared.project_id == "project-a" for prepared in hub_harness.workflows.prepared
    )


def test_hub_bootstrap_projects_only_safe_exact_user_question_projection(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
) -> None:
    runtime = hub_harness.runtimes["project-a"]
    task = replace(valid_brief, task_id="question-card-task")
    runtime.store.save_task("chat-a", task, TaskState.SOL_RUNNING)
    runtime.store.pause_for_question(
        session_id="chat-a",
        task_id=task.task_id,
        revision=task.revision,
        expected_generation=1,
        question_id="question-card-1",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        text="Which exact approved option should Sol use?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": "secret-run", "prompt": "private prompt"},
        event=ConversationEnvelope(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.USER,
            routed_to=ConversationTarget.USER,
            message_type=ConversationMessageType.QUESTION,
            text="Which exact approved option should Sol use?",
            task_id=task.task_id,
            revision=task.revision,
            continuation_generation=1,
            question_id="question-card-1",
        ),
    )
    with _authenticated_hub_client(hub_harness) as client:
        payload = client.get(
            "/api/projects/project-a/chats/chat-a/bootstrap"
        ).json()
    projected = next(item for item in payload["tasks"] if item["task_id"] == "question-card-task")
    assert projected["continuation_generation"] == 1
    assert projected["exchange_allowance"] == 3
    assert projected["exchange_consumed"] == 0
    assert projected["pending_question"] == {
        "question_id": "question-card-1",
        "asked_by": "sol",
        "addressed_to": "user",
        "routed_to": "user",
        "text": "Which exact approved option should Sol use?",
        "revision": 1,
        "continuation_generation": 1,
    }
    assert "secret-run" not in json.dumps(payload)
    assert "private prompt" not in json.dumps(payload)


def test_hub_bootstrap_projects_only_current_ungranted_exchange_permission(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
) -> None:
    def pause_permission(runtime: _HubRuntime, session_id: str) -> tuple[TaskBrief, str]:
        task = replace(valid_brief, task_id="permission-card-task")
        runtime.store.save_task(session_id, task, TaskState.SOL_RUNNING)
        runtime.store._connection.execute(  # noqa: SLF001 - exact exhausted fixture
            "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
            (task.task_id, task.revision),
        )
        runtime.store.pause_for_exchange_permission(
            session_id=session_id,
            task_id=task.task_id,
            revision=task.revision,
            expected_generation=1,
            attempted_question=DirectedAgentQuestion(
                addressed_to="fable",
                text="A fourth question needs permission.",
                reason="The exchange allowance is exhausted.",
            ),
            continuation_state=TaskState.SOL_RUNNING,
            pending_action={"provider_id": "must-not-project"},
            event=ConversationEnvelope(
                sender=ConversationActor.SYSTEM,
                addressed_to=ConversationTarget.USER,
                routed_to=ConversationTarget.USER,
                message_type=ConversationMessageType.STATUS,
                text="Automatic exchange limit reached. Allow three more internal exchanges to continue.",
                task_id=task.task_id,
                revision=task.revision,
                continuation_generation=1,
            ),
        )
        permission_id = runtime.store._connection.execute(  # noqa: SLF001 - fixture identity
            "SELECT permission_id FROM exchange_permissions"
        ).fetchone()[0]
        return task, permission_id

    project_a = hub_harness.runtimes["project-a"]
    project_b = hub_harness.runtimes["project-b"]
    task_a, permission_a = pause_permission(project_a, "chat-a")
    _task_b, permission_b = pause_permission(project_b, "chat-b")

    with _authenticated_hub_client(hub_harness) as client:
        first = client.get("/api/projects/project-a/chats/chat-a/bootstrap").json()
    projected = next(item for item in first["tasks"] if item["task_id"] == task_a.task_id)
    assert projected["exchange_permission"] == {
        "request_id": permission_a,
        "revision": 1,
        "continuation_generation": 1,
    }
    assert permission_b not in json.dumps(first)
    assert "provider_id" not in json.dumps(first)

    project_a.store._connection.execute(  # noqa: SLF001 - stale pause must not project
        "UPDATE tasks SET continuation_generation = 2 WHERE task_id = ? AND revision = ?",
        (task_a.task_id, task_a.revision),
    )
    with _authenticated_hub_client(hub_harness) as client:
        stale = client.get("/api/projects/project-a/chats/chat-a/bootstrap").json()
    assert next(item for item in stale["tasks"] if item["task_id"] == task_a.task_id)["exchange_permission"] is None

    project_a.store._connection.execute(  # noqa: SLF001 - restore exact paused fixture
        "UPDATE tasks SET continuation_generation = 1 WHERE task_id = ? AND revision = ?",
        (task_a.task_id, task_a.revision),
    )
    assert project_a.store.grant_internal_exchanges(
        session_id="chat-a", task_id=task_a.task_id, revision=1,
        expected_generation=1, request_id=permission_a,
    ) == 3
    with _authenticated_hub_client(hub_harness) as client:
        granted = client.get("/api/projects/project-a/chats/chat-a/bootstrap").json()
    assert next(item for item in granted["tasks"] if item["task_id"] == task_a.task_id)["exchange_permission"] is None


def test_hub_directed_scheduler_rejection_aborts_the_exact_preparation(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_harness.runtimes["project-a"].store.save_task(
        "chat-a", valid_brief, TaskState.AWAITING_USER_INPUT,
    )
    def reject_scheduler(coroutine: object, **kwargs: object) -> object:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            "/api/projects/project-a/chats/chat-a/tasks/task-1/messages",
            json={
                "text": "continue exactly this task", "addressed_to": "fable",
                "revision": 1, "continuation_generation": 1,
            },
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
    assert response.status_code == 503
    assert hub_harness.workflows.aborts == [("continuation-1", "scheduler_unavailable")]
    assert hub_harness.app.state.active_coroutines == set()


@pytest.mark.parametrize(
    ("addressed_to", "expected_target"),
    (("sol", ConversationTarget.SOL), ("team", ConversationTarget.TEAM)),
)
def test_real_ordinary_recipient_is_persisted_but_always_routed_to_fable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    addressed_to: str,
    expected_target: ConversationTarget,
) -> None:
    harness = _real_hub_harness(tmp_path)

    def reject_scheduler(coroutine: object, **kwargs: object) -> object:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
    runtime = harness.runtimes["a"]
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/messages",
                json={"text": "Plan this exact request", "addressed_to": addressed_to},
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert response.status_code == 503
        event = next(
            event for event in runtime.store.events_after("shared-chat", 0)
            if event.actor == "user"
        )
        assert event.actor == "user"
        assert event.kind == "conversation"
        assert event.payload == ConversationEnvelope(
            sender=ConversationActor.USER,
            addressed_to=expected_target,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.STATEMENT,
            text="Plan this exact request",
        ).to_dict()
        assert harness.lease.snapshot() is None
        assert harness.app.state.active_coroutines == set()
    finally:
        harness.close()


@pytest.mark.parametrize("addressed_to", ("fable", "sol", "team"))
def test_real_hub_first_ordinary_recipient_message_titles_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    addressed_to: str,
) -> None:
    harness = _real_hub_harness(
        tmp_path,
        fable_results=((True, "subscription_ready"),) * 2,
        sol_results=("ready",) * 2,
    )

    def reject_scheduler(coroutine: object, **kwargs: object) -> object:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
    runtime = harness.runtimes["a"]
    path = f"/api/projects/{runtime.project_id}/chats/shared-chat/messages"
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            headers = {"X-CSRF-Token": CSRF_TOKEN}
            first = client.post(
                path,
                json={"text": "  First directed chat title.  ", "addressed_to": addressed_to},
                headers=headers,
            )
            second = client.post(
                path,
                json={"text": "Later request cannot rename this chat.", "addressed_to": "fable"},
                headers=headers,
            )
        assert (first.status_code, second.status_code) == (503, 503)
        chat = runtime.store.chat("shared-chat")
        assert chat is not None
        assert chat.title == "First directed chat title."
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("acknowledged", "fable_results", "sol_results", "expected_probes"),
    (
        (False, ((True, "subscription_ready"),), ("ready",), (0, 0)),
        (True, ((False, "subscription_unavailable"),), ("ready",), (1, 1)),
        (True, ((True, "subscription_ready"),), ("unavailable",), (1, 1)),
    ),
)
def test_real_directed_preparation_runs_the_complete_gate_before_store_or_scheduler(
    tmp_path: Path,
    valid_brief: TaskBrief,
    acknowledged: bool,
    fable_results: tuple[object, ...],
    sol_results: tuple[object, ...],
    expected_probes: tuple[int, int],
) -> None:
    harness = _real_hub_harness(
        tmp_path,
        fable_results=fable_results,
        sol_results=sol_results,
    )
    runtime = harness.runtimes["a"]
    task = replace(valid_brief, task_id="gated-directed-task")
    runtime.store.save_task("shared-chat", task, TaskState.SOL_RUNNING)
    runtime.store.set_sol_thread(task.task_id, task.revision, "gated-sol-thread")
    runtime.store.pause_for_continuation(
        task.task_id,
        task.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"sol_run_id": "gated-sol-run", "prompt": "Continue exactly."},
    )
    runtime.store._connection.execute(  # noqa: SLF001 - seed exact approved route state
        "UPDATE tasks SET approved_at = ? WHERE task_id = ? AND revision = ?",
        ("2026-08-10T12:00:00Z", task.task_id, task.revision),
    )
    if not acknowledged:
        harness.hub_store.acknowledged = False
    before_events = runtime.store.events_after("shared-chat", 0)
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/tasks/{task.task_id}/messages",
                json={
                    "text": "continue exactly", "addressed_to": "sol",
                    "revision": 1, "continuation_generation": 1,
                },
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert response.status_code == 409
        assert (harness.probes["a"].fable_calls, harness.probes["a"].sol_calls) == expected_probes
        assert runtime.store.events_after("shared-chat", 0) == before_events
        assert runtime.store.latest_prepared_action_for_task(
            project_id=runtime.project_id, session_id="shared-chat",
            task_id=task.task_id, revision=1,
        ) is None
        assert harness.lease.snapshot() is None
        assert harness.app.state.active_coroutines == set()
    finally:
        harness.close()


def test_every_duplicate_identifier_route_stays_inside_project_a(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A global duplicate-ID lookup would query B or schedule B's coordinator."""
    shared_brief = replace(valid_brief, task_id="shared-task")
    for runtime in hub_harness.runtimes.values():
        runtime.store.create_session("shared-chat", runtime.repository)
        runtime.store.save_task(
            "shared-chat", shared_brief, TaskState.AWAITING_USER_APPROVAL,
        )
    foreign = hub_harness.runtimes["project-b"]

    def foreign_access(*args: object, **kwargs: object) -> object:
        raise AssertionError("duplicate identifiers must not touch project B")

    for name in (
        "session_exists", "chat", "latest_task", "get_task", "append_event",
        "events_after", "latest_task_overviews", "browser_replay_floor",
    ):
        monkeypatch.setattr(foreign.store, name, foreign_access)

    hub_harness.hub_store.acknowledge_usage_credits()
    with _authenticated_hub_client(hub_harness) as client:
        headers = {"X-CSRF-Token": CSRF_TOKEN}
        assert client.get(
            "/api/projects/project-a/chats/shared-chat/bootstrap"
        ).status_code == 200
        assert client.post(
            "/api/projects/project-a/chats/shared-chat/messages",
            json={"text": "plan in A"}, headers=headers,
        ).status_code == 202
        _wait_until(lambda: hub_harness.workflows.runs == ["new-1"])
        for path, body in (
            ("/api/projects/project-a/chats/shared-chat/tasks/shared-task/approve", {"revision": 1}),
            ("/api/projects/project-a/chats/shared-chat/tasks/shared-task/edit", _edited_brief(shared_brief)),
            ("/api/projects/project-a/chats/shared-chat/tasks/shared-task/reject", None),
            ("/api/projects/project-a/chats/shared-chat/tasks/shared-task/answer", {"text": "yes", "revision": 1, "question_id": "question-1", "continuation_generation": 1}),
            ("/api/projects/project-a/chats/shared-chat/tasks/shared-task/resume", None),
        ):
            assert client.post(path, json=body, headers=headers).status_code == 202
        _wait_until(lambda: len(hub_harness.workflows.runs) == 4)
        hub_harness.workflows.active_lease = LeaseToken(
            9, "project-a", "shared-chat", "shared-task",
        )
        assert client.post(
            "/api/projects/project-a/chats/shared-chat/tasks/shared-task/stop",
            headers=headers,
        ).status_code == 202
        _wait_until(lambda: hub_harness.workflows.stops == [
            ("project-a", "shared-chat", "shared-task"),
        ])
        own_event = hub_harness.runtimes["project-a"].store.append_event(
            "shared-chat", None, "user", "message", {"text": "A only"},
        )
        with client.websocket_connect(
            f"/ws?project_id=project-a&session_id=shared-chat&after={own_event.sequence - 1}"
        ) as socket:
            replay = socket.receive_json()
    assert replay["sequence"] == own_event.sequence
    assert foreign.coordinator.calls == []
    assert all(
        prepared.project_id == "project-a"
        for prepared in hub_harness.workflows.prepared
    )


def test_real_duplicate_identifier_mutations_persist_only_in_selected_project(
    tmp_path: Path,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing lookup to a global task/session source would mutate B's rows too."""
    harness = _real_hub_harness(
        tmp_path,
        fable_results=((True, "subscription_ready"),) * 4,
        sol_results=("ready",) * 4,
        repository=_LocalRepository(),
    )
    runtime_a = harness.runtimes["a"]
    runtime_b = harness.runtimes["b"]
    task_ids = {
        "approval": "same-approval",
        "edit": "same-edit",
        "reject": "same-reject",
        "answer": "same-answer",
        "resume": "same-resume",
        "stop": "same-stop",
    }

    def seed(runtime: _HubRuntime) -> None:
        store = runtime.store
        approval = replace(valid_brief, task_id=task_ids["approval"])
        edit = replace(valid_brief, task_id=task_ids["edit"])
        reject = replace(valid_brief, task_id=task_ids["reject"])
        answer = replace(valid_brief, task_id=task_ids["answer"])
        resume = replace(valid_brief, task_id=task_ids["resume"])
        stop = replace(valid_brief, task_id=task_ids["stop"])
        store.save_task("shared-chat", approval, TaskState.AWAITING_USER_APPROVAL)
        store.save_task("shared-chat", edit, TaskState.AWAITING_USER_APPROVAL)
        store.set_fable_session(edit.task_id, edit.revision, "fable-edit-session")
        store.save_task("shared-chat", reject, TaskState.AWAITING_USER_APPROVAL)
        store.save_task("shared-chat", answer, TaskState.SOL_RUNNING)
        store.set_sol_thread(answer.task_id, answer.revision, "sol-answer-thread")
        store.pause_for_continuation(
            answer.task_id,
            answer.revision,
            expected=TaskState.SOL_RUNNING,
            target=TaskState.AWAITING_USER_INPUT,
            continuation_state=TaskState.SOL_RUNNING,
            pending={"prompt": "Answer exactly.", "sol_run_id": "sol-answer-run"},
        )
        store._connection.execute(  # noqa: SLF001 - seed exact approved routing state
            "UPDATE tasks SET approved_at = ? WHERE task_id = ? AND revision = ?",
            ("2026-08-10T12:00:00Z", answer.task_id, answer.revision),
        )
        store.save_task("shared-chat", resume, TaskState.FABLE_PLANNING)
        store.mark_interrupted(
            resume.task_id, resume.revision, continuation=TaskState.FABLE_PLANNING,
        )
        store.save_task("shared-chat", stop, TaskState.FABLE_PLANNING)

    def snapshot(runtime: _HubRuntime) -> tuple[object, ...]:
        rows = tuple(
            (
                task_id,
                runtime.store.latest_task(task_id).state.value,  # type: ignore[union-attr]
                runtime.store.latest_task(task_id).revision,  # type: ignore[union-attr]
                None if runtime.store.latest_task(task_id).continuation_state is None  # type: ignore[union-attr]
                else runtime.store.latest_task(task_id).continuation_state.value,  # type: ignore[union-attr]
            )
            for task_id in task_ids.values()
        )
        records = tuple(
            (
                task_id,
                None
                if (record := runtime.store.latest_prepared_action_for_task(
                    project_id=runtime.project_id,
                    session_id="shared-chat",
                    task_id=task_id,
                    revision=1,
                )) is None
                else (record.action, record.status, record.generation),
            )
            for task_id in task_ids.values()
        )
        return rows, records, tuple(event.to_dict() for event in runtime.store.events_after("shared-chat", 0))

    seed(runtime_a)
    seed(runtime_b)
    before_b = snapshot(runtime_b)

    def reject_scheduler(coroutine: object, **kwargs: object) -> object:
        raise RuntimeError("scheduler unavailable")

    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            headers = {"X-CSRF-Token": CSRF_TOKEN}
            edited = _edited_brief(replace(valid_brief, task_id=task_ids["edit"]))
            assert client.post(
                f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/{task_ids['edit']}/edit",
                json=edited, headers=headers,
            ).status_code == 202
            # TestClient owns the route coroutine on its portal thread; wait
            # for it instead of probing this Store connection concurrently.
            _wait_until(lambda: harness.app.state.active_coroutines == set())
            edited_task = runtime_a.store.latest_task(task_ids["edit"])
            assert edited_task is not None and edited_task.revision == 2
            assert client.post(
                f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/{task_ids['reject']}/reject",
                headers=headers,
            ).status_code == 202
            _wait_until(lambda: harness.app.state.active_coroutines == set())
            rejected_task = runtime_a.store.latest_task(task_ids["reject"])
            assert rejected_task is not None and rejected_task.state is TaskState.FAILED

            monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
            for path, body in (
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/messages", {"text": "only A message"}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/{task_ids['approval']}/approve", {"revision": 1}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/{task_ids['answer']}/messages", {"text": "only A continuation", "addressed_to": "sol", "revision": 1, "continuation_generation": 1}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/{task_ids['resume']}/resume", None),
            ):
                assert client.post(path, json=body, headers=headers).status_code == 503
            monkeypatch.undo()

            harness.lease.acquire(
                project_id=runtime_a.project_id,
                session_id="shared-chat",
                task_id=task_ids["stop"],
            )
            active_stop = harness.lease.snapshot()
            assert active_stop is not None
            assert (
                active_stop.project_id,
                active_stop.session_id,
                active_stop.task_id,
            ) == (runtime_a.project_id, "shared-chat", task_ids["stop"])
            assert client.post(
                f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/{task_ids['stop']}/stop",
                headers=headers,
            ).status_code == 202
            _wait_until(lambda: harness.lease.snapshot() is None)

            selected_event = runtime_a.store.append_event(
                "shared-chat", None, "user", "message", {"text": "selected A only"},
            )
            assert client.get(
                f"/api/projects/{runtime_a.project_id}/chats/shared-chat/bootstrap"
            ).status_code == 200
            with client.websocket_connect(
                f"/ws?project_id={runtime_a.project_id}&session_id=shared-chat&after={selected_event.sequence - 1}"
            ) as socket:
                assert socket.receive_json()["sequence"] == selected_event.sequence

        assert snapshot(runtime_b) == before_b
        a_rows, _, a_events = snapshot(runtime_a)
        assert {task_id: values for task_id, *values in a_rows} == {
            task_ids["approval"]: ["interrupted", 1, "sol_running"],
            task_ids["edit"]: ["awaiting_user_approval", 2, None],
            task_ids["reject"]: ["failed", 1, None],
            task_ids["answer"]: ["interrupted", 1, "sol_running"],
            task_ids["resume"]: ["interrupted", 1, "fable_planning"],
            task_ids["stop"]: ["interrupted", 1, "fable_planning"],
        }
        assert set(task_ids.values()).issubset({event["task_id"] for event in a_events})
        for route in ("approval", "answer", "resume"):
            record = runtime_a.store.latest_prepared_action_for_task(
                project_id=runtime_a.project_id,
                session_id="shared-chat",
                task_id=task_ids[route],
                revision=1,
            )
            assert record is not None and record.status == "ABORTED"
        assert harness.lease.snapshot() is None
    finally:
        harness.close()


def test_real_b_only_identifiers_cannot_change_selected_project_or_foreign_store(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """Resolving B before the route project would make these B-only IDs observable."""
    harness = _real_hub_harness(tmp_path)
    runtime_a = harness.runtimes["a"]
    runtime_b = harness.runtimes["b"]
    runtime_b.store.create_session("b-only-chat", runtime_b.repository)
    runtime_b.store.save_task(
        "b-only-chat", replace(valid_brief, task_id="b-only-task"),
        TaskState.AWAITING_USER_APPROVAL,
    )
    before_a = tuple(event.to_dict() for event in runtime_a.store.events_after("shared-chat", 0))
    before_b = tuple(event.to_dict() for event in runtime_b.store.events_after("b-only-chat", 0))
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            headers = {"X-CSRF-Token": CSRF_TOKEN}
            for path, body in (
                (f"/api/projects/{runtime_a.project_id}/chats/b-only-chat/messages", {"text": "never"}),
                (f"/api/projects/{runtime_a.project_id}/chats/b-only-chat/tasks/b-only-task/approve", {"revision": 1}),
                (f"/api/projects/{runtime_a.project_id}/chats/b-only-chat/tasks/b-only-task/edit", _edited_brief(replace(valid_brief, task_id="b-only-task"))),
                (f"/api/projects/{runtime_a.project_id}/chats/b-only-chat/tasks/b-only-task/reject", None),
                (f"/api/projects/{runtime_a.project_id}/chats/b-only-chat/tasks/b-only-task/answer", {"text": "never", "revision": 1, "question_id": "question-1", "continuation_generation": 1}),
                (f"/api/projects/{runtime_a.project_id}/chats/b-only-chat/tasks/b-only-task/resume", None),
            ):
                assert client.post(path, json=body, headers=headers).status_code == 404
            assert client.post(
                f"/api/projects/{runtime_a.project_id}/chats/b-only-chat/tasks/b-only-task/stop",
                headers=headers,
            ).status_code == 409
            assert client.get(
                f"/api/projects/{runtime_a.project_id}/chats/b-only-chat/bootstrap"
            ).status_code == 404
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(
                    f"/ws?project_id={runtime_a.project_id}&session_id=b-only-chat&after=0"
                ):
                    pass
        assert caught.value.code == 1008
        assert tuple(event.to_dict() for event in runtime_a.store.events_after("shared-chat", 0)) == before_a
        assert tuple(event.to_dict() for event in runtime_b.store.events_after("b-only-chat", 0)) == before_b
        assert runtime_b.store.get_task("b-only-task", 1).state is TaskState.AWAITING_USER_APPROVAL
        assert harness.lease.snapshot() is None
    finally:
        harness.close()


def test_hub_chats_are_project_local_and_first_message_updates_only_its_title(
    hub_harness: _HubHarness,
) -> None:
    with TestClient(hub_harness.app) as anonymous:
        assert anonymous.get("/api/projects").status_code == 403
    with _authenticated_hub_client(hub_harness) as client:
        headers = {"X-CSRF-Token": CSRF_TOKEN}
        projects = client.get("/api/projects")
        assert projects.status_code == 200
        assert projects.headers["cache-control"] == "no-store"
        assert projects.json() == {
            "csrf_token": CSRF_TOKEN,
            "usage_credits_acknowledged": False,
            "projects": [
                {
                    "project_id": "project-a", "label": "PROJECT-A", "branch": "main",
                    "readiness": {
                        "fable_ready": True,
                        "fable_status": "subscription_ready",
                        "sol_status": "ready",
                    },
                },
                {
                    "project_id": "project-b", "label": "PROJECT-B", "branch": "main",
                    "readiness": {
                        "fable_ready": True,
                        "fable_status": "subscription_ready",
                        "sol_status": "ready",
                    },
                },
            ],
            "active_lease": None,
        }
        created = client.post("/api/projects/project-a/chats", headers=headers)
        assert created.status_code == 201
        chat = created.json()
        assert chat["title"] == "New chat"
        assert chat["session_id"] not in {"chat-a", "chat-b"}
        message = client.post(
            f"/api/projects/project-a/chats/{chat['session_id']}/messages",
            json={"text": "  Build a concise status dashboard for the handoff.  "},
            headers=headers,
        )
        assert message.status_code == 202
        _wait_until(lambda: bool(hub_harness.workflows.runs))
        page = client.get("/api/projects/project-a/chats?limit=2")
        assert page.status_code == 200
        assert page.json()["chats"][0]["session_id"] == chat["session_id"]
        assert page.json()["chats"][0]["title"] == "Build a concise status dashboard for the handoff."
        reopened = client.get(
            f"/api/projects/project-a/chats/{chat['session_id']}/bootstrap"
        )
        assert reopened.status_code == 200
        assert reopened.json()["session_id"] == chat["session_id"]
        assert client.get("/api/projects/project-b/chats?limit=50").json()["chats"] == [
            {
                "session_id": "chat-b",
                "title": "New chat",
                "created_at": "2026-08-10T00:00:01Z",
                "updated_at": "2026-08-10T00:00:01Z",
                "latest_sequence": 0,
            }
        ]


def test_project_navigation_public_payloads_need_no_repository_path(
    hub_harness: _HubHarness,
) -> None:
    """The static controller receives labels and opaque IDs, never a local path."""
    with _authenticated_hub_client(hub_harness) as client:
        projects = client.get("/api/projects")
        assert projects.status_code == 200
        payload = projects.json()
        assert payload["csrf_token"] == CSRF_TOKEN
        assert projects.headers["cache-control"] == "no-store"
        assert payload["usage_credits_acknowledged"] is False
        assert payload["active_lease"] is None
        assert payload["projects"] == [
            {
                "project_id": "project-a", "label": "PROJECT-A", "branch": "main",
                "readiness": {
                    "fable_ready": True,
                    "fable_status": "subscription_ready",
                    "sol_status": "ready",
                },
            },
            {
                "project_id": "project-b", "label": "PROJECT-B", "branch": "main",
                "readiness": {
                    "fable_ready": True,
                    "fable_status": "subscription_ready",
                    "sol_status": "ready",
                },
            },
        ]
        assert "repository" not in str(payload)
        chats = client.get("/api/projects/project-a/chats?limit=50")
        assert chats.status_code == 200
        assert chats.json()["chats"][0]["session_id"] == "chat-a"
        bootstrap = client.get(
            "/api/projects/project-a/chats/chat-a/bootstrap"
        )
        assert bootstrap.status_code == 200
        assert bootstrap.json()["project_id"] == "project-a"
        assert bootstrap.json()["session_id"] == "chat-a"
        assert "repository" not in str(bootstrap.json())


def test_hub_model_preparation_rejects_foreign_lease_without_probe_and_stop_is_exact_owner_only(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
) -> None:
    runtime = hub_harness.runtimes["project-a"]
    runtime.store.save_task("chat-a", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    hub_harness.hub_store.acknowledge_usage_credits()
    hub_harness.workflows.reject_preparation = True
    with _authenticated_hub_client(hub_harness) as client:
        headers = {"X-CSRF-Token": CSRF_TOKEN}
        denied = client.post(
            "/api/projects/project-a/chats/chat-a/tasks/task-1/approve",
            json={"revision": 1}, headers=headers,
        )
        assert denied.status_code == 409
        assert client.post(
            "/api/projects/project-a/chats/chat-a/tasks/task-1/stop",
            headers=headers,
        ).status_code == 409
    assert hub_harness.workflows.probe_calls == 0
    assert hub_harness.workflows.stops == []


def test_active_lease_synchronously_gates_navigation_model_mutations_and_exact_stop(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
) -> None:
    runtime = hub_harness.runtimes["project-a"]
    runtime.store.save_task("chat-a", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    hub_harness.hub_store.acknowledge_usage_credits()
    hub_harness.workflows.active_lease = LeaseToken(
        7, "project-b", "chat-b", "foreign-task",
    )
    with _authenticated_hub_client(hub_harness) as client:
        headers = {"X-CSRF-Token": CSRF_TOKEN}
        projects = client.get("/api/projects").json()
        assert projects["active_lease"] == {
            "project_id": "project-b", "session_id": "chat-b", "task_id": "foreign-task",
        }
        assert client.post("/api/projects/project-a/chats", headers=headers).status_code == 409
        assert client.get("/api/projects/project-a/chats/chat-a/bootstrap").status_code == 409
        assert client.get("/api/projects/project-b/chats/chat-b/bootstrap").status_code == 200
        for path, body in (
            ("/api/projects/project-a/chats/chat-a/messages", {"text": "plan"}),
            ("/api/projects/project-a/chats/chat-a/tasks/task-1/approve", {"revision": 1}),
            ("/api/projects/project-a/chats/chat-a/tasks/task-1/answer", {"text": "yes", "revision": 1, "question_id": "question-1", "continuation_generation": 1}),
            ("/api/projects/project-a/chats/chat-a/tasks/task-1/resume", None),
            ("/api/projects/project-a/chats/chat-a/tasks/task-1/stop", None),
        ):
            assert client.post(path, json=body, headers=headers).status_code == 409
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                "/ws?project_id=project-a&session_id=chat-a&after=0"
            ):
                pass
        assert caught.value.code == 1008

        hub_harness.workflows.active_lease = LeaseToken(
            8, "project-a", "chat-a", "task-1",
        )
        assert client.post(
            "/api/projects/project-a/chats/chat-a/tasks/other-task/stop",
            headers=headers,
        ).status_code == 409
        assert client.post(
            "/api/projects/project-a/chats/chat-a/tasks/task-1/stop",
            headers=headers,
        ).status_code == 202
        _wait_until(lambda: hub_harness.workflows.stops == [("project-a", "chat-a", "task-1")])
    assert hub_harness.workflows.probe_calls == 0
    assert hub_harness.workflows.prepared == []


def test_stop_route_reserves_generation_one_before_a_delayed_child_can_start(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """A route that schedules before reserving would let generation two replace Stop."""
    harness = _real_hub_harness(tmp_path)
    runtime = harness.runtimes["a"]
    runtime.store.save_task(
        "shared-chat", valid_brief, TaskState.FABLE_PLANNING,
    )
    token = harness.lease.acquire(
        project_id=runtime.project_id,
        session_id="shared-chat",
        task_id=valid_brief.task_id,
    )
    delayed = _DelayedStopWorkflow(harness.workflows)
    app = create_hub_app(
        registry=ProjectRegistry(tuple(harness.runtimes.values())),
        hub_store=harness.hub_store,
        workflows=delayed,  # type: ignore[arg-type]
        static_dir=tmp_path / "real-static",
        session_key=SESSION_KEY,
        csrf_token=CSRF_TOKEN,
    )
    try:
        with TestClient(app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/tasks/{valid_brief.task_id}/stop",
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
            assert response.status_code == 202
            client.portal.call(delayed.entered.wait)
            harness.lease.release(token)
            assert harness.lease.snapshot() == token
            with pytest.raises(RuntimeError, match="another workflow"):
                harness.lease.acquire(
                    project_id=runtime.project_id,
                    session_id="shared-chat",
                    task_id=valid_brief.task_id,
                )
            client.portal.call(delayed.release.set)
            _wait_until(lambda: not app.state.active_coroutines)
        task = runtime.store.get_task(valid_brief.task_id, valid_brief.revision)
        assert task.state is TaskState.INTERRUPTED
        assert harness.lease.snapshot() is None
        assert runtime.store.events_after("shared-chat", 0)[-1].kind == "task_state"
        assert app.state.coroutine_observation_failures == []
    finally:
        harness.close()


@pytest.mark.parametrize("owner_finishes", (False, True))
def test_stop_scheduler_rejection_cancels_only_its_reservation_without_mutating_the_task(
    tmp_path: Path,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
    owner_finishes: bool,
) -> None:
    """A failed outer schedule must not clear the still-active exact workflow."""
    harness = _real_hub_harness(tmp_path)
    runtime = harness.runtimes["a"]
    runtime.store.save_task("shared-chat", valid_brief, TaskState.FABLE_PLANNING)
    token = harness.lease.acquire(
        project_id=runtime.project_id,
        session_id="shared-chat",
        task_id=valid_brief.task_id,
    )

    def reject_scheduler(coroutine: object, **kwargs: object) -> object:
        if owner_finishes:
            harness.lease.release(token)
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/tasks/{valid_brief.task_id}/stop",
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert response.status_code == 503
        assert runtime.store.get_task(valid_brief.task_id, valid_brief.revision).state is TaskState.FABLE_PLANNING
        assert runtime.store.events_after("shared-chat", 0) == ()
        # A deferred owner finalizer cannot surrender the exact retry authority
        # while this failed Stop reservation is being cancelled.
        assert harness.lease.snapshot() is token
        assert harness.app.state.active_coroutines == set()
    finally:
        harness.close()


def test_real_foreign_lease_rejects_navigation_and_model_preparation_before_store_work(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """Foreign routes leave both SQLite runtimes untouched; exact Stop changes B only."""
    harness = _real_hub_harness(tmp_path, repository=_LocalRepository())
    runtime_a = harness.runtimes["a"]
    runtime_b = harness.runtimes["b"]
    approval = replace(valid_brief, task_id="approval-task")
    answer = replace(valid_brief, task_id="answer-task")
    resume = replace(valid_brief, task_id="resume-task")
    runtime_a.store.save_task("shared-chat", approval, TaskState.AWAITING_USER_APPROVAL)
    runtime_a.store.save_task("shared-chat", answer, TaskState.SOL_RUNNING)
    runtime_a.store.set_sol_thread(answer.task_id, answer.revision, "sol-answer-thread")
    runtime_a.store.pause_for_continuation(
        answer.task_id,
        answer.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"prompt": "Answer exactly.", "sol_run_id": "sol-answer-run"},
    )
    runtime_a.store.save_task("shared-chat", resume, TaskState.FABLE_PLANNING)
    runtime_a.store.mark_interrupted(
        resume.task_id,
        resume.revision,
        continuation=TaskState.FABLE_PLANNING,
    )
    owner_token = harness.lease.acquire(
        project_id=runtime_b.project_id,
        session_id="shared-chat",
        task_id="foreign-task",
    )
    foreign_prepared = runtime_b.coordinator.prepare_new_request(
        session_id="shared-chat",
        task_id="foreign-task",
        text="owner already running",
        generation=owner_token.generation,
    )

    def runtime_inventory(runtime: _HubRuntime) -> dict[str, tuple[tuple[object, ...], ...]]:
        """Read the persisted rows directly; no route or workflow helper builds these."""
        store = runtime.store
        chats = tuple(
            (
                chat.session_id,
                chat.repo_root,
                chat.title,
                chat.created_at,
                chat.updated_at,
                chat.latest_sequence,
            )
            for chat in store.list_chats(limit=50)
        )
        sessions = tuple(
            tuple(row)
            for row in store._connection.execute(  # noqa: SLF001 - read-only test evidence
                """
                SELECT session_id, repo_root, title, created_at, updated_at
                FROM sessions
                ORDER BY session_id
                """
            )
        )
        tasks = tuple(
            (
                row["task_id"],
                int(row["revision"]),
                row["session_id"],
                row["state"],
                None if row["brief_json"] is None else row["brief_json"].encode(),
                row["approved_at"],
                row["fable_session_id"],
                row["sol_thread_id"],
                row["baseline_id"],
                int(row["correction_count"]),
                row["continuation_state"],
                None if row["pending_json"] is None else row["pending_json"].encode(),
            )
            for row in store._connection.execute(  # noqa: SLF001 - read-only test evidence
                """
                SELECT task_id, revision, session_id, state, brief_json, approved_at,
                       fable_session_id, sol_thread_id, baseline_id, correction_count,
                       continuation_state, pending_json
                FROM tasks
                ORDER BY task_id, revision
                """
            )
        )
        events = tuple(
            (
                int(row["sequence"]),
                row["session_id"],
                row["task_id"],
                row["actor"],
                row["kind"],
                row["payload_json"].encode(),
                row["created_at"],
            )
            for row in store._connection.execute(  # noqa: SLF001 - read-only test evidence
                """
                SELECT sequence, session_id, task_id, actor, kind, payload_json, created_at
                FROM events
                ORDER BY sequence
                """
            )
        )
        prepared = tuple(
            (
                row["preparation_id"],
                row["project_id"],
                row["session_id"],
                row["task_id"],
                int(row["revision"]),
                row["action"],
                row["payload_json"].encode(),
                row["source_state"],
                row["active_state"],
                row["continuation_state"],
                None
                if row["pending_context_json"] is None
                else row["pending_context_json"].encode(),
                row["previous_preparation_id"],
                row["status"],
                row["reason"],
                int(row["generation"]),
            )
            for row in store._connection.execute(  # noqa: SLF001 - read-only test evidence
                """
                SELECT preparation_id, project_id, session_id, task_id, revision, action,
                       payload_json, source_state, active_state, continuation_state,
                       pending_context_json, previous_preparation_id, status, reason,
                       generation
                FROM prepared_actions
                ORDER BY rowid
                """
            )
        )
        return {
            "chats": chats,
            "sessions": sessions,
            "tasks": tasks,
            "events": events,
            "prepared": prepared,
        }

    def lease_projection() -> tuple[object, object]:
        token = harness.lease.snapshot()
        reservation = harness.lease._stop_reservation  # noqa: SLF001 - exact claim state

        def token_projection(value: LeaseToken | None) -> tuple[object, ...] | None:
            if value is None:
                return None
            return (
                value.generation,
                value.project_id,
                value.session_id,
                value.task_id,
            )

        return (
            token_projection(token),
            None if reservation is None else (
                reservation._claim_id,  # noqa: SLF001 - exact claim state
                token_projection(reservation._token),  # noqa: SLF001 - exact claim state
            ),
        )

    def hub_inventory() -> dict[str, object]:
        return {
            "a": runtime_inventory(runtime_a),
            "b": runtime_inventory(runtime_b),
            "lease": lease_projection(),
            "probes": tuple(
                (label, harness.probes[label].fable_calls, harness.probes[label].sol_calls)
                for label in ("a", "b")
            ),
        }

    def task_projection(
        inventory: dict[str, tuple[tuple[object, ...], ...]],
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (task[0], task[1], task[2], task[3], task[10], task[11])
            for task in inventory["tasks"]
        )

    before = hub_inventory()
    assert before["lease"] == (
        (1, runtime_b.project_id, "shared-chat", "foreign-task"),
        None,
    )
    assert before["probes"] == (("a", 0, 0), ("b", 0, 0))
    assert before["a"]["chats"] == (  # type: ignore[index]
        ("shared-chat", runtime_a.repository, "New chat", "2026-08-10T00:00:01Z",
         "2026-08-10T00:00:01Z", 0),
    )
    assert before["a"]["sessions"] == (  # type: ignore[index]
        ("shared-chat", runtime_a.repository, "New chat", "2026-08-10T00:00:01Z",
         "2026-08-10T00:00:01Z"),
    )
    assert task_projection(before["a"]) == (  # type: ignore[arg-type]
        ("answer-task", 1, "shared-chat", "awaiting_user_input", "sol_running",
         b'{"prompt":"Answer exactly.","sol_run_id":"sol-answer-run"}'),
        ("approval-task", 1, "shared-chat", "awaiting_user_approval", None, None),
        ("resume-task", 1, "shared-chat", "interrupted", "fable_planning", None),
    )
    assert before["a"]["events"] == ()  # type: ignore[index]
    assert before["a"]["prepared"] == ()  # type: ignore[index]
    assert before["b"]["chats"] == (  # type: ignore[index]
        ("shared-chat", runtime_b.repository, "owner already running",
         "2026-08-10T00:00:01Z", "2026-08-10T00:00:02Z", 1),
    )
    assert before["b"]["sessions"] == (  # type: ignore[index]
        ("shared-chat", runtime_b.repository, "owner already running",
         "2026-08-10T00:00:01Z", "2026-08-10T00:00:02Z"),
    )
    assert task_projection(before["b"]) == (  # type: ignore[arg-type]
        ("foreign-task", 0, "shared-chat", "fable_planning", None, None),
    )
    assert before["b"]["events"] == (  # type: ignore[index]
        (1, "shared-chat", "foreign-task", "user", "message",
         b'{"text":"owner already running"}', "2026-08-10T00:00:02Z"),
    )
    assert before["b"]["prepared"] == (  # type: ignore[index]
        (
            foreign_prepared.preparation_id, runtime_b.project_id, "shared-chat",
            "foreign-task", 0, "new_request",
            b'{"kind":"new_request","text":"owner already running"}',
            "fable_planning", "fable_planning", None, None, None, "PREPARED", None, 1,
        ),
    )

    def assert_no_side_effects(route_before: dict[str, object]) -> dict[str, object]:
        after = hub_inventory()
        # Each rejection may mutate neither runtime's chat/session/task/event/action rows.
        for label in ("a", "b"):
            before_runtime = route_before[label]  # type: ignore[index]
            after_runtime = after[label]  # type: ignore[index]
            assert after_runtime["chats"] == before_runtime["chats"]
            assert after_runtime["sessions"] == before_runtime["sessions"]
            assert after_runtime["tasks"] == before_runtime["tasks"]
            assert after_runtime["events"] == before_runtime["events"]
            assert after_runtime["prepared"] == before_runtime["prepared"]
        assert after["lease"] == route_before["lease"]
        assert after["probes"] == (("a", 0, 0), ("b", 0, 0))
        assert harness.app.state.active_coroutines == set()
        return after

    def assert_rejected_without_side_effects(
        response: object,
        *,
        route_before: dict[str, object],
    ) -> None:
        assert getattr(response, "status_code") == 409
        assert_no_side_effects(route_before)

    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            headers = {"X-CSRF-Token": CSRF_TOKEN}
            route_before = hub_inventory()
            assert_rejected_without_side_effects(
                client.post(f"/api/projects/{runtime_a.project_id}/chats", headers=headers),
                route_before=route_before,
            )
            route_before = hub_inventory()
            assert_rejected_without_side_effects(
                client.get(
                    f"/api/projects/{runtime_a.project_id}/chats/shared-chat/bootstrap"
                ),
                route_before=route_before,
            )
            owner_bootstrap = client.get(
                f"/api/projects/{runtime_b.project_id}/chats/shared-chat/bootstrap"
            )
            assert owner_bootstrap.status_code == 200
            assert owner_bootstrap.json()["project_id"] == runtime_b.project_id
            assert owner_bootstrap.json()["session_id"] == "shared-chat"
            assert_no_side_effects(before)
            for path, body in (
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/messages", {"text": "plan"}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/approval-task/approve", {"revision": 1}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/answer-task/messages", {"text": "continue", "addressed_to": "sol", "revision": 1, "continuation_generation": 1}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/answer-task/answer", {"text": "yes", "revision": 1, "question_id": "question-1", "continuation_generation": 1}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/resume-task/resume", None),
            ):
                route_before = hub_inventory()
                assert_rejected_without_side_effects(
                    client.post(path, json=body, headers=headers), route_before=route_before,
                )
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(
                    f"/ws?project_id={runtime_a.project_id}&session_id=shared-chat&after=0"
                ):
                    pass
            assert caught.value.code == 1008
            assert_no_side_effects(before)
            before_stop = hub_inventory()
            response = client.post(
                f"/api/projects/{runtime_b.project_id}/chats/shared-chat/tasks/foreign-task/stop",
                headers=headers,
            )
            assert response.status_code == 202
            _wait_until(
                lambda: harness.lease.snapshot() is None
                and harness.app.state.active_coroutines == set()
            )
        after_stop = hub_inventory()
        # Stop's explicit allowed delta is the owning B task/action/event/session only.
        assert after_stop["a"] == before_stop["a"]
        assert after_stop["lease"] == (None, None)
        assert after_stop["probes"] == (("a", 0, 0), ("b", 0, 0))
        assert after_stop["b"]["chats"] == (  # type: ignore[index]
            ("shared-chat", runtime_b.repository, "owner already running",
             "2026-08-10T00:00:01Z", "2026-08-10T00:00:03Z", 2),
        )
        assert after_stop["b"]["sessions"] == (  # type: ignore[index]
            ("shared-chat", runtime_b.repository, "owner already running",
             "2026-08-10T00:00:01Z", "2026-08-10T00:00:03Z"),
        )
        expected_stop_pending = (
            '{"prepared_action":{"action":"new_request","context":null,'
            f'"preparation_id":"{foreign_prepared.preparation_id}","reason":"stop"}}}}'
        ).encode()
        assert after_stop["b"]["tasks"] == (  # type: ignore[index]
            (
                "foreign-task", 0, "shared-chat", "interrupted", None, None, None,
                None, None, 0, "fable_planning", expected_stop_pending,
            ),
        )
        assert after_stop["b"]["events"] == (  # type: ignore[index]
            (1, "shared-chat", "foreign-task", "user", "message",
             b'{"text":"owner already running"}', "2026-08-10T00:00:02Z"),
            (2, "shared-chat", "foreign-task", "coordinator", "task_state",
             b'{"revision":0,"state":"interrupted"}', "2026-08-10T00:00:03Z"),
        )
        assert after_stop["b"]["prepared"] == (  # type: ignore[index]
            (
                foreign_prepared.preparation_id, runtime_b.project_id, "shared-chat",
                "foreign-task", 0, "new_request",
                b'{"kind":"new_request","text":"owner already running"}',
                "fable_planning", "fable_planning", None, None, None, "INTERRUPTED",
                "stop", 1,
            ),
        )
        assert harness.app.state.active_coroutines == set()
    finally:
        harness.close()


def test_real_concurrent_project_preparations_probe_only_the_lease_winner(
    tmp_path: Path,
) -> None:
    """Moving lease acquisition after readiness would make both real runtimes probe."""
    harness = _real_hub_harness(tmp_path)

    async def exercise() -> None:
        barrier = asyncio.Barrier(2)

        async def prepare(label: str) -> object:
            runtime = harness.runtimes[label]
            await barrier.wait()
            return await harness.workflows.prepare_new_request(
                project_id=runtime.project_id,
                session_id="shared-chat",
                text=f"plan for {label}",
                ids=_RealIds(),
            )

        results = await asyncio.gather(
            prepare("a"), prepare("b"), return_exceptions=True,
        )
        winners = [result for result in results if not isinstance(result, BaseException)]
        losers = [result for result in results if isinstance(result, BaseException)]
        assert len(winners) == 1
        assert len(losers) == 1
        assert isinstance(losers[0], RuntimeError)
        assert sum(probe.fable_calls for probe in harness.probes.values()) == 1
        assert sum(probe.sol_calls for probe in harness.probes.values()) == 1
        stores_with_message = [
            runtime.store
            for runtime in harness.runtimes.values()
            if runtime.store.events_after("shared-chat", 0)
        ]
        assert len(stores_with_message) == 1
        winner = winners[0]
        harness.workflows.abort_prepared(winner, reason="scheduler_unavailable")  # type: ignore[arg-type]
        assert harness.lease.snapshot() is None

    try:
        asyncio.run(exercise())
    finally:
        harness.close()


def test_real_http_preparation_refreshes_fable_readiness_after_each_lease_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reusing a ready snapshot would accept the second unavailable Fable result."""
    harness = _real_hub_harness(
        tmp_path,
        fable_results=((True, "subscription_ready"), (False, "subscription_unavailable")),
        sol_results=("ready", "ready"),
    )

    def reject_scheduler(coroutine: object, **kwargs: object) -> object:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
    runtime = harness.runtimes["a"]
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            headers = {"X-CSRF-Token": CSRF_TOKEN}
            first = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/messages",
                json={"text": "first plan"}, headers=headers,
            )
            assert first.status_code == 503
            assert runtime.readiness.snapshot() == RuntimeStatus(
                True, "subscription_ready", "ready",
            )
            second = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/messages",
                json={"text": "second plan"}, headers=headers,
            )
            assert second.status_code == 409
        assert harness.probes["a"].fable_calls == 2
        assert harness.probes["a"].sol_calls == 2
        assert runtime.readiness.snapshot() == RuntimeStatus(
            False, "subscription_unavailable", "ready",
        )
        assert harness.lease.snapshot() is None
        assert harness.app.state.active_coroutines == set()
        assert [event.kind for event in runtime.store.events_after("shared-chat", 0)] == [
            "conversation", "task_state",
        ]
    finally:
        harness.close()


def test_hub_bootstrap_immediately_projects_only_invalidated_fable_subscription(
    tmp_path: Path,
) -> None:
    harness = _real_hub_harness(tmp_path)
    failed = harness.runtimes["a"]
    unaffected = harness.runtimes["b"]
    try:
        failed.readiness.invalidate_fable_subscription()
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            failed_payload = client.get(
                f"/api/projects/{failed.project_id}/chats/shared-chat/bootstrap"
            ).json()
            unaffected_payload = client.get(
                f"/api/projects/{unaffected.project_id}/chats/shared-chat/bootstrap"
            ).json()

        assert (failed_payload["fable_ready"], failed_payload["fable_status"]) == (
            False, "subscription_unavailable",
        )
        assert (unaffected_payload["fable_ready"], unaffected_payload["fable_status"]) == (
            False, "checking",
        )
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("fable_result", "sol_result", "expected_status"),
    (
        (_HANGING_PROBE, "ready", RuntimeStatus(False, "subscription_unavailable", "unavailable")),
        ("malformed-fable-result", "ready", RuntimeStatus(False, "subscription_unavailable", "unavailable")),
        ((True, "subscription_ready"), _HANGING_PROBE, RuntimeStatus(False, "subscription_unavailable", "unavailable")),
        ((True, "subscription_ready"), 7, RuntimeStatus(False, "subscription_unavailable", "unavailable")),
        ((False, "subscription_unavailable"), "ready", RuntimeStatus(False, "subscription_unavailable", "ready")),
        ((True, "subscription_ready"), "unavailable", RuntimeStatus(True, "subscription_ready", "unavailable")),
    ),
    ids=(
        "fable-timeout", "fable-malformed", "sol-timeout", "sol-malformed",
        "fable-unavailable", "sol-unavailable",
    ),
)
def test_real_http_readiness_failures_release_lease_without_durable_preparation(
    tmp_path: Path,
    fable_result: object,
    sol_result: object,
    expected_status: RuntimeStatus,
) -> None:
    """Any failed fresh gate must leave the real selected store untouched."""
    harness = _real_hub_harness(
        tmp_path, fable_results=(fable_result,), sol_results=(sol_result,),
    )
    runtime = harness.runtimes["a"]
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/messages",
                json={"text": "must not persist"},
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert response.status_code == 409
        assert runtime.readiness.snapshot() == expected_status
        assert harness.lease.snapshot() is None
        assert harness.app.state.active_coroutines == set()
        assert runtime.store.events_after("shared-chat", 0) == ()
        assert runtime.store.latest_task("real-task-1") is None
    finally:
        harness.close()


def test_lifespan_cancellation_terminalizes_a_real_prepared_workflow_and_releases_its_lease(
    tmp_path: Path,
) -> None:
    """Dropping tracked tasks at shutdown would leak the claimed row or global lease."""
    blocking_fable = _BlockingFable()
    harness = _real_hub_harness(tmp_path, fable=blocking_fable)
    runtime = harness.runtimes["a"]
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/messages",
                json={"text": "hold until shutdown"},
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
            assert response.status_code == 202
            client.portal.call(blocking_fable.started.wait)
            event = runtime.store.events_after("shared-chat", 0)[0]
            assert event.task_id is not None
            prepared = runtime.store.latest_prepared_action_for_task(
                project_id=runtime.project_id,
                session_id="shared-chat",
                task_id=event.task_id,
                revision=0,
            )
            assert prepared is not None and prepared.status == "CLAIMED"
            assert harness.lease.snapshot() is not None
        terminal = runtime.store.prepared_action(prepared.preparation_id)
        assert terminal is not None
        assert terminal.status == "INTERRUPTED"
        assert terminal.reason == "adapter_interrupted"
        assert runtime.store.get_task(event.task_id, 0).state is TaskState.INTERRUPTED
        assert harness.lease.snapshot() is None
        assert harness.app.state.active_coroutines == set()
        assert harness.app.state.coroutine_observation_failures == []
    finally:
        harness.close()


def test_hub_scheduler_rejection_aborts_the_exact_durable_preparation(
    hub_harness: _HubHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_harness.hub_store.acknowledge_usage_credits()

    def reject_task(coroutine: object, **kwargs: object) -> object:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_task)
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            "/api/projects/project-a/chats/chat-a/messages",
            json={"text": "make a plan"},
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
    assert response.status_code == 503
    assert response.json()["detail"]["state"] == "recoverable"
    assert hub_harness.workflows.aborts == [("new-1", "scheduler_unavailable")]
    assert hub_harness.workflows.runs == []
    assert hub_harness.app.state.active_coroutines == set()


def test_prepared_task_failures_and_lifespan_cancellation_are_observed_without_raw_event_persistence(
    hub_harness: _HubHarness,
) -> None:
    hub_harness.hub_store.acknowledge_usage_credits()
    hub_harness.workflows.run_failure = True
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            "/api/projects/project-a/chats/chat-a/messages",
            json={"text": "fail locally"},
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
        assert response.status_code == 202
        _wait_until(lambda: not hub_harness.app.state.active_coroutines)
    assert hub_harness.app.state.coroutine_observation_failures == [
        {"stage": "prepared", "error_type": "RuntimeError"},
    ]
    assert [event.kind for event in hub_harness.runtimes["project-a"].store.events_after("chat-a", 0)] == [
        "message",
    ]

    hub_harness.workflows.run_failure = False
    hub_harness.workflows.block_run = True
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            "/api/projects/project-a/chats/chat-a/messages",
            json={"text": "cancel locally"},
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
        assert response.status_code == 202
        _wait_until(lambda: bool(hub_harness.app.state.active_coroutines))
    assert hub_harness.app.state.active_coroutines == set()
    assert hub_harness.app.state.coroutine_observation_failures == [
        {"stage": "prepared", "error_type": "RuntimeError"},
    ]


def test_real_hub_scheduler_rejection_returns_exact_resume_identity_without_duplicate_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The durable retry is Resume, not a repeated user message."""
    repo_root = tmp_path / "repository"
    repo_root.mkdir()
    project_id = project_id_for_root(repo_root)
    store = SQLiteStore(
        tmp_path / "project.sqlite3", clock=_Clock(), check_same_thread=False,
    )
    store.create_session("chat-real", str(repo_root))

    async def fable_probe() -> tuple[bool, str]:
        return True, "subscription_ready"

    async def sol_probe() -> str:
        return "ready"

    coordinator = Coordinator(
        store=store,
        repository=object(),  # Preparation/abort needs no repository operation.
        runner=object(),
        fable=object(),
        sol=object(),
        ids=_RealIds(),
        repo_root=repo_root,
        repo_context="local test repository",
        trusted_shells={"sh": "/bin/sh"},
    )
    runtime = _HubRuntime(
        project_id=project_id,
        label="REAL",
        repository=str(repo_root),
        branch="main",
        store=store,
        coordinator=coordinator,  # type: ignore[arg-type]
        broadcaster=InMemoryEventBroadcaster(),
        readiness=RuntimeReadiness(
            initial=RuntimeStatus(True, "subscription_ready", "ready"),
            fable_probe=fable_probe,
            sol_probe=sol_probe,
        ),
    )
    hub_store = _HubStoreFake()
    hub_store.acknowledge_usage_credits()
    workflows = HubWorkflowOrchestrator(
        registry=ProjectRegistry((runtime,)),
        lease=ActiveAgentLease(),
        usage_credits_acknowledged=hub_store.usage_credits_acknowledged,
    )
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html><title>real hub</title>")
    app = create_hub_app(
        registry=ProjectRegistry((runtime,)),
        hub_store=hub_store,
        workflows=workflows,
        static_dir=static_dir,
        session_key=SESSION_KEY,
        csrf_token=CSRF_TOKEN,
    )

    def reject_scheduler(coroutine: object, **kwargs: object) -> object:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
    try:
        with TestClient(app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{project_id}/chats/chat-real/messages",
                json={"text": "make exactly one durable plan"},
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert response.status_code == 503
        recovery = response.json()["detail"]
        assert recovery == {
            "state": "recoverable",
            "preparation_id": recovery["preparation_id"],
            "project_id": project_id,
            "session_id": "chat-real",
            "task_id": recovery["task_id"],
            "revision": 0,
        }
        original = store.prepared_action(recovery["preparation_id"])
        assert original is not None
        assert original.status == "ABORTED"
        assert store.get_task(recovery["task_id"], 0).state is TaskState.INTERRUPTED
        assert workflows.active_lease_snapshot() is None
        assert app.state.active_coroutines == set()
        assert [event.kind for event in store.events_after("chat-real", 0)] == [
            "conversation", "task_state",
        ]

        with TestClient(app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            existing_rejection = client.post(
                f"/api/projects/{project_id}/chats/chat-real/tasks/{recovery['task_id']}/resume",
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert existing_rejection.status_code == 503
        existing_recovery = existing_rejection.json()["detail"]
        assert existing_recovery["project_id"] == project_id
        assert existing_recovery["session_id"] == "chat-real"
        assert existing_recovery["task_id"] == recovery["task_id"]
        assert existing_recovery["revision"] == 0
        existing = store.prepared_action(existing_recovery["preparation_id"])
        assert existing is not None
        assert existing.action == "resume"
        assert existing.status == "ABORTED"
        assert existing.previous_preparation_id == original.preparation_id
        assert workflows.active_lease_snapshot() is None
        assert app.state.active_coroutines == set()
        conversations = [
            event for event in store.events_after("chat-real", 0)
            if event.actor == "user" and event.kind == "conversation"
        ]
        assert len(conversations) == 1
        assert conversations[0].payload == ConversationEnvelope(
            sender=ConversationActor.USER,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.STATEMENT,
            text="make exactly one durable plan",
        ).to_dict()

        monkeypatch.undo()
        with TestClient(app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            resumed = client.post(
                f"/api/projects/{project_id}/chats/chat-real/tasks/{recovery['task_id']}/resume",
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
            assert resumed.status_code == 202
            _wait_until(lambda: not app.state.active_coroutines)
        retry = store.latest_prepared_action_for_task(
            project_id=project_id,
            session_id="chat-real",
            task_id=recovery["task_id"],
            revision=0,
        )
        assert retry is not None
        assert retry.action == "resume"
        assert retry.previous_preparation_id == existing.preparation_id
        assert len([
            event for event in store.events_after("chat-real", 0)
            if event.actor == "user" and event.kind == "conversation"
        ]) == 1
    finally:
        coordinator.close()
        store.close()


def test_hub_websocket_replay_is_session_filtered_inside_the_selected_project(
    hub_harness: _HubHarness,
) -> None:
    first = hub_harness.runtimes["project-a"].store.append_event(
        "chat-a", None, "user", "message", {"text": "only A"}
    )
    hub_harness.runtimes["project-b"].store.append_event(
        "chat-b", None, "user", "message", {"text": "only B"}
    )
    with _authenticated_hub_client(hub_harness) as client:
        with client.websocket_connect(
            "/ws?project_id=project-a&session_id=chat-a&after=0"
        ) as socket:
            replay = socket.receive_json()
    assert replay["sequence"] == first.sequence
    assert replay["payload"] == {"text": "only A"}


def test_compatibility_factory_is_one_non_owning_runtime(web_harness: WebHarness) -> None:
    runtime = web_harness.app.state.compatibility_runtime
    assert runtime.store is web_harness.store
    assert runtime.coordinator is web_harness.coordinator
    assert runtime.broadcaster is web_harness.broadcaster
    assert web_harness.app.state.project_registry.projects() == (runtime,)
    assert not hasattr(runtime, "close")


def test_compatibility_model_starts_use_the_injected_fresh_readiness_check(
    web_harness: WebHarness,
) -> None:
    async def fresh_readiness() -> BootstrapStatus:
        return BootstrapStatus(
            session_id=SESSION_ID,
            fable_ready=False,
            fable_status="subscription_unavailable",
            sol_status="ready",
            repository="/repo",
            branch="main",
        )

    app = create_app(
        coordinator=web_harness.coordinator,
        store=web_harness.store,
        static_dir=web_harness.static_dir,
        session_key=SESSION_KEY,
        csrf_token=CSRF_TOKEN,
        broadcaster=web_harness.broadcaster,
        bootstrap_status=web_harness.status_provider,
        readiness_check=fresh_readiness,
    )
    with TestClient(app) as client:
        assert client.get(f"/?key={SESSION_KEY}", follow_redirects=False).status_code == 303
        csrf = _acknowledge_model_usage(client)
        response = client.post(
            f"/api/sessions/{SESSION_ID}/messages",
            json={"text": "must use the fresh check"},
            headers={"X-CSRF-Token": csrf},
        )
    assert response.status_code == 409
    assert web_harness.coordinator.calls == []


def test_intervention_request_body_is_strict_before_any_workflow_lookup(
    hub_harness: _HubHarness,
) -> None:
    with _authenticated_hub_client(hub_harness) as client:
        response = client.post(
            "/api/projects/project-a/chats/chat-a/tasks/task-1/intervene",
            json={
                "intervention_id": "intervention-1", "message": "Keep scope exact.",
                "addressed_to": "fable", "revision": 1,
                "continuation_generation": 1, "routed_to": "sol",
            },
            headers={"X-CSRF-Token": CSRF_TOKEN},
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid request"}
    assert hub_harness.workflows.prepared == []


def _seed_live_fable_intervention_source(
    harness: _RealHubHarness,
    *,
    runtime_key: str,
    brief: TaskBrief,
    runner: _InterventionRunner,
    run_id: str,
    lease: ActiveAgentLease | None = None,
) -> tuple[_HubRuntime, LeaseToken]:
    """Seed only the real persisted source run that an Intervene route owns."""
    runtime = harness.runtimes[runtime_key]
    runtime.store.save_task("shared-chat", brief, TaskState.FABLE_PLANNING)
    runtime.store.start_agent_run(run_id, brief.task_id, brief.revision, "fable")
    runner.stores[run_id] = runtime.store
    owner = harness.lease if lease is None else lease
    return runtime, owner.acquire(
        project_id=runtime.project_id,
        session_id="shared-chat",
        task_id=brief.task_id,
    )


def _seed_unknown_intervention(
    runtime: _HubRuntime,
    *,
    brief: TaskBrief,
) -> object:
    """Build a genuine UNKNOWN record without invoking a provider."""
    store = runtime.store
    source_run_id = "unknown-source-run"
    sol_thread_id = "123e4567-e89b-12d3-a456-426614174000"
    store.save_task("shared-chat", brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(brief.task_id, brief.revision, sol_thread_id)
    store.set_pending_context(
        brief.task_id,
        brief.revision,
        expected=TaskState.SOL_RUNNING,
        pending={"sol_run_id": source_run_id, "prompt": "Continue exactly."},
    )
    store.start_agent_run(source_run_id, brief.task_id, brief.revision, "sol")
    store.set_agent_run_session(source_run_id, sol_thread_id)
    created = store.create_intervention_and_request_stop(
        intervention_id="unknown-intervention",
        session_id="shared-chat",
        task_id=brief.task_id,
        revision=brief.revision,
        expected_source_generation=1,
        message="Keep the approved boundary exact.",
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        run_id=source_run_id,
    )
    store.finish_agent_run(source_run_id, status="interrupted", exit_code=-15)
    ready = store.mark_intervention_ready(created.intervention_id, run_id=source_run_id)
    store.begin_intervention_resume(
        ready.intervention_id,
        expected_resume_generation=ready.resume_generation,
        resume_attempt_id="unknown-attempt",
        resume_run_id="unknown-resume-run",
    )
    unknown = store.mark_resume_outcome_unknown(
        ready.intervention_id,
        resume_attempt_id="unknown-attempt",
        resume_run_id="unknown-resume-run",
    )
    store.recover_active_tasks()
    return unknown


def test_intervention_routes_require_keyed_csrf_usage_and_strict_exact_inputs(
    hub_harness: _HubHarness,
    valid_brief: TaskBrief,
) -> None:
    """A route that accepts any browser input could route hidden process authority."""
    runtime = hub_harness.runtimes["project-a"]
    runtime.store.save_task("chat-a", valid_brief, TaskState.FABLE_PLANNING)
    payload = {
        "intervention_id": "intervention-1",
        "message": "Keep the approved scope exact.",
        "addressed_to": "fable",
        "revision": valid_brief.revision,
        "continuation_generation": 1,
    }
    path = "/api/projects/project-a/chats/chat-a/tasks/task-1/intervene"
    with TestClient(hub_harness.app) as unauthenticated:
        assert unauthenticated.post(path, json=payload).status_code == 403
    with _authenticated_hub_client(hub_harness) as client:
        assert client.post(path, json=payload).status_code == 403
        headers = {"X-CSRF-Token": CSRF_TOKEN}
        assert client.post(path, json=payload, headers=headers).status_code == 409
        hub_harness.hub_store.acknowledge_usage_credits()
        assert client.post(
            path,
            json={**payload, "revision": valid_brief.revision + 1},
            headers=headers,
        ).status_code == 409
        invalid_payloads = (
            {**payload, "intervention_id": "x" * 129},
            {**payload, "message": " \t"},
            {**payload, "message": "unsafe\x00text"},
            {**payload, "message": "x" * (16 * 1024 + 1)},
            {**payload, "addressed_to": "team"},
            {**payload, "continuation_generation": 0},
            {**payload, "routed_to": "sol"},
            {**payload, "run_id": "run-1"},
            {**payload, "provider_session_id": "provider-1"},
            {**payload, "continuation": "fable_planning"},
            {**payload, "path": "/repo"},
            {**payload, "command": "unsafe"},
            {**payload, "env": {"HOME": "/tmp"}},
        )
        for invalid in invalid_payloads:
            response = client.post(path, json=invalid, headers=headers)
            assert response.status_code == 422
            assert response.json() == {"detail": "invalid request"}
    assert hub_harness.workflows.prepared == []


def test_intervene_commits_before_blocked_stop_and_rechecks_model_gate(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """If scheduling owned the write, a blocked source could make HTTP lie about intent."""
    runner = _InterventionRunner()
    runner.block_stop = True
    harness = _real_hub_harness(
        tmp_path,
        runner=runner,
        fable_results=((False, "subscription_unavailable"),),
    )
    runtime, owner = _seed_live_fable_intervention_source(
        harness,
        runtime_key="a",
        brief=valid_brief,
        runner=runner,
        run_id="blocked-source-run",
    )
    path = (
        f"/api/projects/{runtime.project_id}/chats/shared-chat/"
        f"tasks/{valid_brief.task_id}/intervene"
    )
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            stale = client.post(
                path,
                json={
                    "intervention_id": "stale-generation-intervention",
                    "message": "This stale generation must not claim the source.",
                    "addressed_to": "fable",
                    "revision": valid_brief.revision,
                    "continuation_generation": owner.generation + 1,
                },
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
            assert stale.status_code == 409
            assert runtime.store.intervention("stale-generation-intervention") is None
            assert runner.stops == []
            response = client.post(
                path,
                json={
                    "intervention_id": "blocked-intervention",
                    "message": "Pause at this exact boundary.",
                    "addressed_to": "fable",
                    "revision": valid_brief.revision,
                    "continuation_generation": owner.generation,
                },
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
            assert response.status_code == 202
            accepted = response.json()
            assert accepted["scheduled"] is True
            assert accepted["intervention"] == {
                "intervention_id": "blocked-intervention",
                "message": "Pause at this exact boundary.",
                "addressed_to": "fable",
                "routed_to": "fable",
                "status": "pending_stop",
                "task_id": valid_brief.task_id,
                "revision": valid_brief.revision,
                "source_generation": owner.generation,
                "resume_generation": owner.generation + 1,
                "eligible": True,
                "visible_discontinuity": True,
                "warning": None,
            }
            committed = runtime.store.intervention("blocked-intervention")
            assert committed is not None
            assert committed.status is InterventionStatus.PENDING_STOP
            assert runtime.store.get_task(
                valid_brief.task_id, valid_brief.revision,
            ).state is TaskState.INTERRUPTED
            client.portal.call(runner.stop_started.wait)
            assert runner.stops == ["blocked-source-run"]
            assert harness.lease.snapshot() == owner
            client.portal.call(runner.release_stop.set)
            _wait_until(lambda: not harness.app.state.active_coroutines)
        ready = runtime.store.intervention("blocked-intervention")
        assert ready is not None
        assert ready.status is InterventionStatus.READY
        assert ready.resume_attempt_id is None
        assert ready.resume_run_id is None
        assert harness.lease.snapshot() is None
    finally:
        harness.close()


@pytest.mark.parametrize("final_status", ("pending", "ready"))
def test_intervention_scheduler_rejection_preserves_source_and_releases_recovery_lease(
    tmp_path: Path,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
    final_status: str,
) -> None:
    """Failed installation must not release a source token or retain a recovery token."""
    runner = _InterventionRunner()
    harness = _real_hub_harness(
        tmp_path,
        runner=runner,
        fable_results=((False, "subscription_unavailable"),),
    )
    runtime, source_owner = _seed_live_fable_intervention_source(
        harness,
        runtime_key="a",
        brief=valid_brief,
        runner=runner,
        run_id="recovery-source-run",
    )
    path = (
        f"/api/projects/{runtime.project_id}/chats/shared-chat/"
        f"tasks/{valid_brief.task_id}/intervene"
    )

    def reject_scheduler(coroutine: object, **kwargs: object) -> object:
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            accepted = client.post(
                path,
                json={
                    "intervention_id": "recovery-intervention",
                    "message": "Preserve this durable stop intent.",
                    "addressed_to": "fable",
                    "revision": valid_brief.revision,
                    "continuation_generation": source_owner.generation,
                },
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert accepted.status_code == 202
        assert accepted.json()["scheduled"] is False
        record = runtime.store.intervention("recovery-intervention")
        assert record is not None
        assert record.status is InterventionStatus.PENDING_STOP
        assert harness.lease.snapshot() == source_owner
        assert runner.stops == []
        assert harness.app.state.active_coroutines == set()

        if final_status == "ready":
            runtime.store.finish_agent_run(
                "recovery-source-run", status="interrupted", exit_code=-15,
            )
            record = runtime.store.mark_intervention_ready(
                record.intervention_id, run_id=record.run_id,
            )

        restarted_lease = ActiveAgentLease()
        restarted_workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry(tuple(harness.runtimes.values())),
            lease=restarted_lease,
            usage_credits_acknowledged=harness.hub_store.usage_credits_acknowledged,
        )
        restarted_app = create_hub_app(
            registry=ProjectRegistry(tuple(harness.runtimes.values())),
            hub_store=harness.hub_store,
            workflows=restarted_workflows,
            static_dir=tmp_path / "real-static",
            session_key=SESSION_KEY,
            csrf_token=CSRF_TOKEN,
        )
        resume_path = (
            f"/api/projects/{runtime.project_id}/chats/shared-chat/"
            f"interventions/{record.intervention_id}/resume"
        )
        with TestClient(restarted_app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            rejected = client.post(
                resume_path,
                json={"expected_resume_generation": record.resume_generation},
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert rejected.status_code == 202
        assert rejected.json()["scheduled"] is False
        assert runtime.store.intervention(record.intervention_id) == record
        assert restarted_lease.snapshot() is None
        assert runner.stops == []
        assert restarted_app.state.active_coroutines == set()

        monkeypatch.undo()
        with TestClient(restarted_app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            recovered = client.post(
                resume_path,
                json={"expected_resume_generation": record.resume_generation},
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
            assert recovered.status_code == 202
            assert recovered.json()["scheduled"] is True
            _wait_until(lambda: not restarted_app.state.active_coroutines)
        recovered_record = runtime.store.intervention(record.intervention_id)
        assert recovered_record is not None
        assert recovered_record.status is InterventionStatus.READY
        assert recovered_record.resume_attempt_id is None
        assert restarted_lease.snapshot() is None
    finally:
        harness.close()


def test_intervention_persistence_failure_is_bounded_and_never_scheduled(
    tmp_path: Path,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error before durable commit must not start an unowned stop coroutine."""
    runner = _InterventionRunner()
    harness = _real_hub_harness(tmp_path, runner=runner)
    runtime, owner = _seed_live_fable_intervention_source(
        harness,
        runtime_key="a",
        brief=valid_brief,
        runner=runner,
        run_id="failed-persistence-source",
    )

    def fail_persistence(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected persistence failure")

    monkeypatch.setattr(
        runtime.store, "create_intervention_and_request_stop", fail_persistence,
    )
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/"
                f"tasks/{valid_brief.task_id}/intervene",
                json={
                    "intervention_id": "failed-persistence-intervention",
                    "message": "Never schedule an uncommitted stop.",
                    "addressed_to": "fable",
                    "revision": valid_brief.revision,
                    "continuation_generation": owner.generation,
                },
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert response.status_code == 409
        assert response.json() == {"detail": "workflow is not currently available"}
        assert runtime.store.intervention("failed-persistence-intervention") is None
        assert runtime.store.get_task(
            valid_brief.task_id, valid_brief.revision,
        ).state is TaskState.FABLE_PLANNING
        assert runner.stops == []
        assert harness.lease.snapshot() == owner
        assert harness.app.state.active_coroutines == set()
    finally:
        harness.close()


def test_resume_rejects_an_unauthenticated_intervention_before_scheduling(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """Using the raw row here would let a tampered record obtain a recovery lease."""
    runner = _InterventionRunner()
    runner.block_stop = True
    harness = _real_hub_harness(tmp_path, runner=runner)
    runtime, source_owner = _seed_live_fable_intervention_source(
        harness,
        runtime_key="a",
        brief=valid_brief,
        runner=runner,
        run_id="tampered-source-run",
    )
    try:
        prepared = harness.workflows.prepare_intervention(
            project_id=runtime.project_id,
            session_id="shared-chat",
            task_id=valid_brief.task_id,
            intent=InterventionIntent(
                intervention_id="tampered-intervention",
                message="Keep this authenticated.",
                addressed_to=ConversationTarget.FABLE,
                revision=valid_brief.revision,
                continuation_generation=source_owner.generation,
            ),
        )
        runtime.store._connection.execute(
            "UPDATE interventions SET message = 'tampered' WHERE intervention_id = ?",
            (prepared.record.intervention_id,),
        )
        restarted_lease = ActiveAgentLease()
        restarted_workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry(tuple(harness.runtimes.values())),
            lease=restarted_lease,
            usage_credits_acknowledged=harness.hub_store.usage_credits_acknowledged,
        )
        restarted_app = create_hub_app(
            registry=ProjectRegistry(tuple(harness.runtimes.values())),
            hub_store=harness.hub_store,
            workflows=restarted_workflows,
            static_dir=tmp_path / "real-static",
            session_key=SESSION_KEY,
            csrf_token=CSRF_TOKEN,
        )
        with TestClient(restarted_app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            response = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/"
                "interventions/tampered-intervention/resume",
                json={"expected_resume_generation": prepared.record.resume_generation},
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
            missing = client.post(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/"
                "interventions/missing-intervention/resume",
                json={"expected_resume_generation": prepared.record.resume_generation},
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert response.status_code == 404
        assert missing.status_code == 404
        assert restarted_lease.snapshot() is None
        assert runner.stops == []
        assert restarted_app.state.active_coroutines == set()
    finally:
        harness.close()


def test_unknown_retry_requires_literal_acknowledgement_and_is_not_resume(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """An ordinary resume must never silently authorize a possibly executed retry."""
    harness = _real_hub_harness(
        tmp_path,
        fable_results=((False, "subscription_unavailable"),),
    )
    runtime = harness.runtimes["a"]
    unknown = _seed_unknown_intervention(runtime, brief=valid_brief)
    assert runtime.store.authenticated_intervention("unknown-intervention") == unknown
    base = (
        f"/api/projects/{runtime.project_id}/chats/shared-chat/"
        "interventions/unknown-intervention"
    )
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            headers = {"X-CSRF-Token": CSRF_TOKEN}
            ordinary = client.post(
                f"{base}/resume",
                json={"expected_resume_generation": unknown.resume_generation},
                headers=headers,
            )
            assert ordinary.status_code == 409
            for invalid_acknowledgement in (False, "true"):
                rejected = client.post(
                    f"{base}/authorize-retry",
                    json={
                        "expected_resume_generation": unknown.resume_generation,
                        "acknowledgment_id": "acknowledgment-1",
                        "acknowledge_possible_prior_execution": invalid_acknowledgement,
                    },
                    headers=headers,
                )
                assert rejected.status_code == 422
            authorized = client.post(
                f"{base}/authorize-retry",
                json={
                    "expected_resume_generation": unknown.resume_generation,
                    "acknowledgment_id": "acknowledgment-1",
                    "acknowledge_possible_prior_execution": True,
                },
                headers=headers,
            )
            assert authorized.status_code == 202
            assert authorized.json()["scheduled"] is True
            _wait_until(lambda: not harness.app.state.active_coroutines)
        retried = runtime.store.authenticated_intervention("unknown-intervention")
        assert retried is not None
        assert retried.status is InterventionStatus.READY
        assert retried.resume_generation == unknown.resume_generation + 1
        assert retried.resume_attempt_id is None
        assert retried.resume_run_id is None
    finally:
        harness.close()


def test_intervention_routes_select_only_the_named_project_and_conflicts_do_not_schedule(
    tmp_path: Path,
    valid_brief: TaskBrief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hub-wide ID lookup would let one project observe or alter another's intent."""
    runner = _InterventionRunner()
    harness = _real_hub_harness(tmp_path, runner=runner)
    runtime_a, owner_a = _seed_live_fable_intervention_source(
        harness,
        runtime_key="a",
        brief=valid_brief,
        runner=runner,
        run_id="project-a-source",
    )
    try:
        record_a = harness.workflows.prepare_intervention(
            project_id=runtime_a.project_id,
            session_id="shared-chat",
            task_id=valid_brief.task_id,
            intent=InterventionIntent(
                intervention_id="shared-intervention",
                message="Project A guidance.",
                addressed_to=ConversationTarget.FABLE,
                revision=valid_brief.revision,
                continuation_generation=owner_a.generation,
            ),
        ).record
        harness.lease.release(owner_a)
        restarted_lease = ActiveAgentLease()
        restarted_workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry(tuple(harness.runtimes.values())),
            lease=restarted_lease,
            usage_credits_acknowledged=harness.hub_store.usage_credits_acknowledged,
        )
        restarted_app = create_hub_app(
            registry=ProjectRegistry(tuple(harness.runtimes.values())),
            hub_store=harness.hub_store,
            workflows=restarted_workflows,
            static_dir=tmp_path / "real-static",
            session_key=SESSION_KEY,
            csrf_token=CSRF_TOKEN,
        )
        runtime_b, owner_b = _seed_live_fable_intervention_source(
            harness,
            runtime_key="b",
            brief=valid_brief,
            runner=runner,
            run_id="project-b-source",
            lease=restarted_lease,
        )

        def reject_scheduler(coroutine: object, **kwargs: object) -> object:
            raise RuntimeError("scheduler unavailable")

        with monkeypatch.context() as scoped:
            def foreign_query(*args: object, **kwargs: object) -> object:
                raise AssertionError("the project A Store must not be queried")

            scoped.setattr(runtime_a.store, "authenticated_intervention", foreign_query)
            scoped.setattr("agent_bridge.app.asyncio.create_task", reject_scheduler)
            with TestClient(restarted_app) as client:
                client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
                created_b = client.post(
                    f"/api/projects/{runtime_b.project_id}/chats/shared-chat/"
                    f"tasks/{valid_brief.task_id}/intervene",
                    json={
                        "intervention_id": "shared-intervention",
                        "message": "Project B guidance.",
                        "addressed_to": "fable",
                        "revision": valid_brief.revision,
                        "continuation_generation": owner_b.generation,
                    },
                    headers={"X-CSRF-Token": CSRF_TOKEN},
                )
                repeated_b = client.post(
                    f"/api/projects/{runtime_b.project_id}/chats/shared-chat/"
                    f"tasks/{valid_brief.task_id}/intervene",
                    json={
                        "intervention_id": "shared-intervention",
                        "message": "Project B guidance.",
                        "addressed_to": "fable",
                        "revision": valid_brief.revision,
                        "continuation_generation": owner_b.generation,
                    },
                    headers={"X-CSRF-Token": CSRF_TOKEN},
                )
            assert created_b.status_code == 202
            assert created_b.json()["scheduled"] is False
            assert repeated_b.status_code == 202
            assert repeated_b.json() == created_b.json()
        record_b = runtime_b.store.intervention("shared-intervention")
        assert record_b is not None
        assert record_b.message == "Project B guidance."
        assert runtime_a.store.intervention("shared-intervention") == record_a
        assert runner.stops == []

        with TestClient(restarted_app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            conflict = client.post(
                f"/api/projects/{runtime_b.project_id}/chats/shared-chat/"
                f"tasks/{valid_brief.task_id}/intervene",
                json={
                    "intervention_id": "shared-intervention",
                    "message": "Conflicting B guidance.",
                    "addressed_to": "fable",
                    "revision": valid_brief.revision,
                    "continuation_generation": owner_b.generation,
                },
                headers={"X-CSRF-Token": CSRF_TOKEN},
            )
        assert conflict.status_code == 409
        assert runtime_b.store.intervention("shared-intervention") == record_b
        assert runner.stops == []
        assert restarted_app.state.active_coroutines == set()

        with monkeypatch.context() as scoped:
            def foreign_resume_query(*args: object, **kwargs: object) -> object:
                raise AssertionError("the project B Store must not be queried")

            scoped.setattr(runtime_b.store, "intervention", foreign_resume_query)
            with TestClient(restarted_app) as client:
                client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
                foreign_resume = client.post(
                    f"/api/projects/{runtime_a.project_id}/chats/shared-chat/"
                    "interventions/shared-intervention/resume",
                    json={"expected_resume_generation": record_a.resume_generation},
                    headers={"X-CSRF-Token": CSRF_TOKEN},
                )
        assert foreign_resume.status_code == 409
        assert runtime_a.store.intervention("shared-intervention") == record_a
        assert runtime_b.store.intervention("shared-intervention") == record_b
        assert restarted_app.state.active_coroutines == set()
    finally:
        harness.close()


def test_hub_bootstrap_intervention_projection_is_allowlisted_and_warns_unknown(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """The bootstrap must not turn a durable control record into process disclosure."""
    harness = _real_hub_harness(tmp_path)
    runtime = harness.runtimes["a"]
    unknown = _seed_unknown_intervention(runtime, brief=valid_brief)
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            task = client.get(
                f"/api/projects/{runtime.project_id}/chats/shared-chat/bootstrap"
            ).json()["tasks"][0]
        intervention = task["intervention"]
        assert set(intervention) == {
            "intervention_id", "message", "addressed_to", "routed_to", "status",
            "task_id", "revision", "source_generation", "resume_generation",
            "eligible", "visible_discontinuity", "warning",
        }
        assert intervention == {
            "intervention_id": "unknown-intervention",
            "message": "Keep the approved boundary exact.",
            "addressed_to": "fable",
            "routed_to": "fable",
            "status": "resume_outcome_unknown",
            "task_id": valid_brief.task_id,
            "revision": valid_brief.revision,
            "source_generation": 1,
            "resume_generation": unknown.resume_generation,
            "eligible": False,
            "visible_discontinuity": False,
            "warning": "prior resume outcome is unknown and may have executed",
        }
        serialized = str(intervention)
        for forbidden in (
            "run_id", "resume_attempt_id", "fable_session_id", "sol_thread_id",
            "pending", "baseline", "raw_result", "command", "process_group",
        ):
            assert forbidden not in serialized
    finally:
        harness.close()
