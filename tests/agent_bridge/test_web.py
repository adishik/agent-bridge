from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent_bridge.app import BootstrapStatus, InMemoryEventBroadcaster, create_app
from agent_bridge.contracts import StreamEvent, TaskBrief
from agent_bridge.state_machine import TaskState
from agent_bridge.store import SQLiteStore


SESSION_ID = "session-1"
SESSION_KEY = "session-secret"
CSRF_TOKEN = "csrf-secret"
EXPECTED_EVENT_REPLAY_PAGE_SIZE = 100
EXPECTED_MAX_INITIAL_REPLAY_EVENTS = 300


class _Clock:
    def __init__(self) -> None:
        self._tick = 0

    def __call__(self) -> str:
        self._tick += 1
        return f"2026-08-10T00:00:{self._tick:02d}Z"


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
