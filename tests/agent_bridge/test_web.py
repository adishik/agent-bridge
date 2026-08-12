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
from agent_bridge.contracts import StreamEvent, TaskBrief
from agent_bridge.coordinator import Coordinator
from agent_bridge.hub import (
    ActiveAgentLease,
    HubWorkflowOrchestrator,
    LeaseToken,
    ProjectRegistry,
    RuntimeReadiness,
    RuntimeStatus,
)
from agent_bridge.projects import project_id_for_root
from agent_bridge.state_machine import TaskState
from agent_bridge.store import SQLiteStore


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
        "agent_event": {"status": "durable activity", "command_sha256": "safe-hash"},
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
    assert task["activity"] == evidence["agent_event"]


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
      const persistedEvent = {json.dumps(persisted_event)};
      const unsafe = {json.dumps(unsafe)};
      let interactionAllowed = () => true;

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
        get textContent() {{ return this._text; }}
        append(...children) {{ for (const child of children) {{ child.parent = this; this.children.push(child); }} }}
        replaceChildren(...children) {{ this.children = []; this.append(...children); this._text = ""; }}
        removeChild(child) {{ this.children.splice(this.children.indexOf(child), 1); }}
        remove() {{ this.parent?.removeChild(this); }}
        setAttribute(name, value) {{ this.attributes[name] = String(value); }}
        removeAttribute(name) {{ delete this.attributes[name]; if (name === "open") this.open = false; }}
        addEventListener(kind, listener) {{ (this.listeners[kind] ??= []).push(listener); }}
        async emit(kind, event = {{}}) {{
          if (!interactionAllowed(this)) return;
          event.preventDefault ??= () => {{}};
          for (const listener of this.listeners[kind] ?? []) await listener(event);
        }}
        focus() {{ documentRoot.activeElement = this; this.focusCount = (this.focusCount ?? 0) + 1; }}
        showModal() {{ this.open = true; this.showModalCount = (this.showModalCount ?? 0) + 1; }}
        close() {{ this.open = false; this.closeCount = (this.closeCount ?? 0) + 1; }}
        querySelector(selector) {{
          const wanted = selector.includes("button") ? "button" : null;
          const stack = [...this.children];
          while (stack.length) {{
            const child = stack.shift();
            if (wanted && child.tag === wanted && !child.disabled) return child;
            stack.push(...child.children);
          }}
          return null;
        }}
      }}

      const ids = [
        "task-list", "conversation", "task-inspector", "composer", "message-input",
        "composer-submit", "composer-guidance", "usage-modal", "usage-credits-form",
        "usage-credits-confirm", "usage-credits-acknowledge", "usage-error",
        "toast-region", "fable-status", "sol-status", "repository-status",
        "connection-status", "task-drawer-toggle", "inspector-drawer-toggle",
        "bootstrap-retry",
      ];
      const nodes = Object.fromEntries(ids.map((id) => [id, new Node(
        id === "usage-modal" ? "dialog" : id.includes("toggle") || id.includes("submit") || id === "bootstrap-retry" ? "button" : id === "composer" || id === "usage-credits-form" ? "form" : "div",
        id,
      )]));
      nodes["message-input"].tag = "textarea";
      nodes["usage-credits-confirm"].tag = "input";
      nodes["task-drawer-toggle"].setAttribute("aria-expanded", "false");
      nodes["inspector-drawer-toggle"].setAttribute("aria-expanded", "false");
      nodes["usage-modal"].append(nodes["usage-credits-form"]);
      nodes["usage-credits-form"].append(
        nodes["usage-credits-confirm"], nodes["usage-credits-acknowledge"],
        nodes["usage-error"], nodes["bootstrap-retry"],
      );
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
        activeElement: launcher,
        listeners: {{}},
        createElement(tag) {{ return new Node(tag); }},
        querySelector(selector) {{ return selector.startsWith("#") ? nodes[selector.slice(1)] ?? null : null; }},
        addEventListener(kind, listener) {{ (this.listeners[kind] ??= []).push(listener); }},
        async emit(kind, event) {{ for (const listener of this.listeners[kind] ?? []) await listener(event); }},
      }};

      const fetchCalls = [];
      let bootstrapFetchCount = 0;
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
          if (url === "/api/bootstrap") {{
            bootstrapFetchCount += 1;
            if (bootstrapFetchCount === 1) return {{ok: false, status: 503, json: async () => ({{}})}};
            if (bootstrapFetchCount === 3) {{
              return await new Promise((resolve) => {{ releaseReconnectBootstrap = resolve; }});
            }}
            return {{ok: true, status: 200, json: async () => bootstrap}};
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
      if (!interactionAllowed(nodes["bootstrap-retry"])) process.exit(25);
      if (!nodes["usage-error"].textContent.includes("Bootstrap failed")) process.exit(26);

      await nodes["bootstrap-retry"].emit("click");
      if (!(await controller.ready)) process.exit(6);
      if (nodes["message-input"].disabled || nodes["composer-submit"].disabled) process.exit(7);
      if (nodes["usage-modal"].closeCount !== 1 || documentRoot.activeElement !== launcher) process.exit(8);
      if (nodes["repository-status"].textContent !== "Repository: /repo · Branch: feat/agent-bridge") {{
        process.exit(9);
      }}

      const approve = nodes["task-inspector"].children[0].querySelector("button");
      await approve.emit("click");
      await Promise.resolve();
      const approval = fetchCalls.find((call) => call.url === "/api/tasks/task-1/approve");
      if (approval.options.body !== '{{"revision":1}}') process.exit(10);
      if (approval.options.headers["X-CSRF-Token"] !== {json.dumps(CSRF_TOKEN)}) process.exit(11);

      nodes["message-input"].value = unsafe;
      await nodes["composer"].emit("submit");
      const message = fetchCalls.find((call) => call.url === "/api/sessions/session-1/messages");
      if (JSON.parse(message.options.body).text !== unsafe) process.exit(12);
      if (message.url.includes(unsafe)) process.exit(13);
      const socket = FakeSocket.instances[0];
      socket.listeners.message({{data: JSON.stringify(persistedEvent)}});
      socket.listeners.message({{data: JSON.stringify(persistedEvent)}});
      if (controller.state.tasks[0].state !== "sol_running") process.exit(21);
      if (controller.state.tasks[0].history.length !== 1) process.exit(22);
      if (nodes["sol-status"].textContent !== "Sol · Running") process.exit(23);
      nodes["message-input"].focus();
      socket.listeners.close();
      scheduled[0].callback();
      const reconnecting = FakeSocket.instances[1].listeners.open();
      await Promise.resolve();
      if (!nodes["message-input"].disabled
          || !nodes["connection-status"].textContent.includes("refresh")) process.exit(27);
      releaseReconnectBootstrap({{ok: true, status: 200, json: async () => bootstrap}});
      await reconnecting;
      if (documentRoot.activeElement !== nodes["message-input"]) process.exit(24);
      if (nodes["message-input"].disabled) process.exit(28);

      await nodes["task-drawer-toggle"].emit("click");
      if (!nodes["task-list"].classList.contains("drawer-open")) process.exit(14);
      if (nodes["task-drawer-toggle"].attributes["aria-expanded"] !== "true") process.exit(15);
      await documentRoot.emit("keydown", {{key: "Escape", preventDefault() {{}}}});
      if (nodes["task-list"].classList.contains("drawer-open")) process.exit(16);
      if (documentRoot.activeElement !== nodes["task-drawer-toggle"]) process.exit(17);
      if (globalThis.pwned === true) process.exit(18);
      media.matches = false;
      media.emit();
      if (nodes["task-list"].inert || nodes["task-inspector"].inert) process.exit(19);
      const taskButton = nodes["task-list"].children[1].querySelector("button");
      await taskButton.emit("click");
      if (nodes["task-list"].inert) process.exit(20);
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

    def require_exact_stop_owner(
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

    async def stop(
        self, *, project_id: str, session_id: str, task_id: str, token: LeaseToken,
    ) -> None:
        if token != self.require_exact_stop_owner(
            project_id=project_id, session_id=session_id, task_id=task_id,
        ):
            raise RuntimeError("stop requires the exact active workflow")
        self.stops.append((project_id, session_id, task_id))
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


def _real_hub_harness(
    tmp_path: Path,
    *,
    fable_results: tuple[object, ...] = ((True, "subscription_ready"),),
    sol_results: tuple[object, ...] = ("ready",),
    fable: object | None = None,
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
            repository=object(),  # Preparation/abort never reaches an adapter edge.
            runner=object(),
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
        ("/api/projects/project-a/chats/chat-a/tasks/task-1/answer", {"answer": "yes"}),
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
            ("/api/projects/project-a/chats/shared-chat/tasks/shared-task/answer", {"answer": "yes"}),
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


def test_hub_chats_are_project_local_and_first_message_updates_only_its_title(
    hub_harness: _HubHarness,
) -> None:
    with _authenticated_hub_client(hub_harness) as client:
        headers = {"X-CSRF-Token": CSRF_TOKEN}
        projects = client.get("/api/projects")
        assert projects.status_code == 200
        assert projects.json() == {
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
            ("/api/projects/project-a/chats/chat-a/tasks/task-1/answer", {"answer": "yes"}),
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


def test_stop_route_carries_generation_one_token_into_a_delayed_same_identity_reacquire(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """Passing only IDs here would let the delayed Stop terminate generation two."""
    harness = _real_hub_harness(tmp_path)
    runtime = harness.runtimes["a"]
    runtime.store.save_task(
        "shared-chat", valid_brief, TaskState.AWAITING_USER_APPROVAL,
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
            successor = harness.lease.acquire(
                project_id=runtime.project_id,
                session_id="shared-chat",
                task_id=valid_brief.task_id,
            )
            assert successor.generation == token.generation + 1
            client.portal.call(delayed.release.set)
            _wait_until(lambda: not app.state.active_coroutines)
        task = runtime.store.get_task(valid_brief.task_id, valid_brief.revision)
        assert task.state is TaskState.AWAITING_USER_APPROVAL
        assert harness.lease.snapshot() == successor
        assert [event.kind for event in runtime.store.events_after("shared-chat", 0)] == [
            "action_error",
        ]
        assert app.state.coroutine_observation_failures == []
    finally:
        harness.close()


def test_real_foreign_lease_rejects_navigation_and_model_preparation_before_store_work(
    tmp_path: Path,
    valid_brief: TaskBrief,
) -> None:
    """Removing route guards would let foreign preparation create durable rows."""
    harness = _real_hub_harness(tmp_path)
    runtime_a = harness.runtimes["a"]
    runtime_b = harness.runtimes["b"]
    approval = replace(valid_brief, task_id="approval-task")
    answer = replace(valid_brief, task_id="answer-task")
    resume = replace(valid_brief, task_id="resume-task")
    runtime_a.store.save_task("shared-chat", approval, TaskState.AWAITING_USER_APPROVAL)
    runtime_a.store.save_task("shared-chat", answer, TaskState.AWAITING_USER_INPUT)
    runtime_a.store.save_task("shared-chat", resume, TaskState.INTERRUPTED)
    token = harness.lease.acquire(
        project_id=runtime_b.project_id,
        session_id="shared-chat",
        task_id="foreign-task",
    )
    try:
        with TestClient(harness.app) as client:
            client.get(f"/?key={SESSION_KEY}", follow_redirects=False)
            headers = {"X-CSRF-Token": CSRF_TOKEN}
            assert client.post(
                f"/api/projects/{runtime_a.project_id}/chats", headers=headers,
            ).status_code == 409
            assert client.get(
                f"/api/projects/{runtime_a.project_id}/chats/shared-chat/bootstrap"
            ).status_code == 409
            assert client.get(
                f"/api/projects/{runtime_b.project_id}/chats/shared-chat/bootstrap"
            ).status_code == 200
            for path, body in (
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/messages", {"text": "plan"}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/approval-task/approve", {"revision": 1}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/answer-task/answer", {"answer": "yes"}),
                (f"/api/projects/{runtime_a.project_id}/chats/shared-chat/tasks/resume-task/resume", None),
            ):
                assert client.post(path, json=body, headers=headers).status_code == 409
        assert harness.probes["a"].fable_calls == 0
        assert harness.probes["a"].sol_calls == 0
        assert harness.app.state.active_coroutines == set()
        assert runtime_a.store.events_after("shared-chat", 0) == ()
        assert runtime_a.store.latest_prepared_action_for_task(
            project_id=runtime_a.project_id,
            session_id="shared-chat",
            task_id="approval-task",
            revision=1,
        ) is None
        assert harness.lease.snapshot() == token
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
            "message", "task_state",
        ]
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("fable_result", "sol_result", "expected_status"),
    (
        (_HANGING_PROBE, "ready", RuntimeStatus(False, "subscription_unavailable", "unavailable")),
        ("malformed-fable-result", "ready", RuntimeStatus(False, "subscription_unavailable", "unavailable")),
        ((False, "subscription_unavailable"), "ready", RuntimeStatus(False, "subscription_unavailable", "ready")),
        ((True, "subscription_ready"), "unavailable", RuntimeStatus(True, "subscription_ready", "unavailable")),
    ),
    ids=("timeout", "malformed", "fable-unavailable", "sol-unavailable"),
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
            "message", "task_state",
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
        assert [event.kind for event in store.events_after("chat-real", 0)].count("message") == 1

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
        assert [event.kind for event in store.events_after("chat-real", 0)].count("message") == 1
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
