from __future__ import annotations

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

from agent_bridge.contracts import (
    ConversationActor,
    ConversationEnvelope,
    ConversationMessageType,
    ConversationTarget,
)
from agent_bridge.store import NewRequestPayload, SQLiteStore


STATIC = Path("src/agent_bridge/static")


class _RenderedLayout(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.ids: set[str] = set()
        self.id_counts: dict[str, int] = {}
        self.text: list[str] = []
        self.ancestors: dict[str, tuple[str, ...]] = {}
        self._open_elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        self.elements.append((tag, values))
        if values.get("id") is not None:
            element_id = str(values["id"])
            self.ids.add(element_id)
            self.id_counts[element_id] = self.id_counts.get(element_id, 0) + 1
            self.ancestors[element_id] = tuple(
                str(open_attrs["id"])
                for _open_tag, open_attrs in self._open_elements
                if open_attrs.get("id") is not None
            )
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "source",
            "track",
            "wbr",
        }:
            self._open_elements.append((tag, values))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._open_elements) - 1, -1, -1):
            if self._open_elements[index][0] == tag:
                del self._open_elements[index:]
                return

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def element(self, tag: str, element_id: str) -> dict[str, str | None]:
        for candidate_tag, attrs in self.elements:
            if candidate_tag == tag and attrs.get("id") == element_id:
                return attrs
        raise AssertionError(f"missing <{tag} id={element_id!r}>")


def _run_module_harness(source: str) -> None:
    module_uri = (STATIC / "app.js").resolve().as_uri()
    result = subprocess.run(
        [
            "node",
            "--experimental-default-type=module",
            "--input-type=module",
            "-e",
            f"import * as bridge from {json.dumps(module_uri)};\n{source}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_index_uses_one_semantic_three_pane_application_workspace() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    rendered = _RenderedLayout()
    rendered.feed(html)

    application_landmarks = [
        attrs
        for _tag, attrs in rendered.elements
        if attrs.get("role") == "application"
    ]
    assert application_landmarks == [
        {"id": "workspace", "role": "application", "aria-label": "Team workspace"}
    ]
    assert {
        "app-header",
        "workspace",
        "project-navigation",
        "project-list",
        "chat-list",
        "new-chat",
        "task-list",
        "conversation",
        "conversation-heading",
        "conversation-status",
        "conversation-context",
        "task-inspector",
        "task-inspector-heading",
        "activity-audit",
        "composer",
        "message-input",
        "usage-modal",
        "usage-credits-confirm",
        "usage-credits-acknowledge",
        "bootstrap-retry",
        "task-drawer-toggle",
        "inspector-drawer-toggle",
        "repository-authority-note",
    } <= rendered.ids
    assert rendered.element("nav", "project-navigation")["aria-label"] == "Projects and chats"
    assert rendered.element("ul", "project-list")["aria-label"] == "Projects"
    assert rendered.element("ul", "chat-list")["aria-label"] == "Chats"
    assert rendered.element("button", "new-chat")["type"] == "button"
    assert rendered.element("main", "conversation-shell")["aria-labelledby"] == "conversation-heading"
    assert rendered.element("p", "conversation-status")["aria-live"] == "polite"
    assert (
        rendered.element("aside", "task-inspector-panel")["aria-labelledby"]
        == "task-inspector-heading"
    )
    assert "open" not in rendered.element("details", "activity-audit")
    assert rendered.element("button", "task-drawer-toggle")["aria-controls"] == (
        "project-navigation"
    )
    assert rendered.element("button", "inspector-drawer-toggle")[
        "aria-controls"
    ] == "task-inspector-panel"
    assert "open" not in rendered.element("dialog", "usage-modal")
    modal_markup = html[html.index('<dialog\n      id="usage-modal"'):html.index("</dialog>")]
    assert 'id="bootstrap-retry"' in modal_markup
    assert rendered.element("button", "composer-submit")["type"] == "submit"
    assert rendered.element("script", "bridge-module")["type"] == "module"
    assert rendered.element("script", "bridge-module")["src"] == "/static/app.js"
    assert any(
        tag == "link"
        and attrs.get("rel") == "stylesheet"
        and attrs.get("href") == "/static/styles.css"
        for tag, attrs in rendered.elements
    )
    page_text = " ".join(" ".join(rendered.text).split())
    assert "Fable · Subscription · checking" in page_text
    assert (
        "Selected at server startup. Messages cannot change repository authority."
        in page_text
    )
    assert "Claude account usage credits are disabled" in page_text
    assert "cannot verify or change this account setting" in page_text


def test_index_accessibility_contract_has_explicit_labels_and_safe_hooks() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    rendered = _RenderedLayout()
    rendered.feed(html)

    assert [tag for tag, _attrs in rendered.elements].count("h1") == 1
    page_text = " ".join(" ".join(rendered.text).split())
    assert all(label in page_text for label in ("Projects", "Chats", "Conversation", "Task inspector"))
    assert all(
        attrs.get("type") is not None
        for tag, attrs in rendered.elements
        if tag == "button"
    )
    assert rendered.element("span", "fable-avatar")["aria-label"] == "Fable"
    assert rendered.element("span", "sol-avatar")["aria-label"] == "Sol"
    assert "Fable · Subscription · checking" in page_text
    assert "Sol · checking" in page_text
    assert any(
        attrs.get("aria-current") == "true"
        for _tag, attrs in rendered.elements
    )

    live_regions = [
        attrs
        for _tag, attrs in rendered.elements
        if attrs.get("aria-live") is not None
    ]
    assert all(attrs.get("role") in {"status", "alert"} for attrs in live_regions)
    assert re.search(r"<[^>]+\s(?:on[a-z]+|style)=", html, flags=re.IGNORECASE) is None
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert all(
        sink not in script
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(")
    )


def test_index_keeps_conversation_first_with_single_mobile_drawer_controls() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    rendered = _RenderedLayout()
    rendered.feed(html)

    element_ids = [attrs.get("id") for _tag, attrs in rendered.elements]
    assert element_ids.index("conversation") < element_ids.index("project-navigation")
    assert element_ids.index("conversation") < element_ids.index("task-inspector")
    for panel_id, heading_id in (
        ("project-navigation", "project-navigation-heading"),
        ("task-inspector-panel", "task-inspector-heading"),
    ):
        panel = next(
            attrs
            for _tag, attrs in rendered.elements
            if attrs.get("id") == panel_id
        )
        assert panel["data-drawer"] == "mobile"
        assert panel["aria-labelledby"] == heading_id
        assert panel["tabindex"] == "-1"
    assert element_ids.count("task-drawer-toggle") == 1
    assert element_ids.count("inspector-drawer-toggle") == 1

    script = (STATIC / "app.js").read_text(encoding="utf-8")
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")
    assert 'event.key !== "Escape"' in script
    assert "panel.inert" in script
    assert "focusTarget?.focus()" in script
    assert "@media (prefers-reduced-motion: reduce)" in styles


def test_drawer_controls_target_complete_unique_panels_and_resolvable_idrefs() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    rendered = _RenderedLayout()
    rendered.feed(html)

    drawer_contracts = (
        (
            "task-drawer-toggle",
            "project-navigation",
            ("project-list", "chat-list", "new-chat", "fable-status", "sol-status"),
        ),
        (
            "inspector-drawer-toggle",
            "task-inspector-panel",
            ("task-inspector", "task-inspector-summary", "task-controls", "activity-audit"),
        ),
    )
    for toggle_id, panel_id, required_children in drawer_contracts:
        toggle = rendered.element("button", toggle_id)
        assert toggle["aria-controls"] == panel_id
        assert rendered.id_counts[panel_id] == 1
        for child_id in required_children:
            assert panel_id in rendered.ancestors[child_id]

    for _tag, attrs in rendered.elements:
        for attribute in ("aria-controls", "aria-describedby", "aria-labelledby", "for"):
            value = attrs.get(attribute)
            if value is None:
                continue
            for reference in value.split():
                assert rendered.id_counts[reference] == 1
        href = attrs.get("href")
        if href is not None and href.startswith("#"):
            assert rendered.id_counts[href[1:]] == 1


def test_static_shells_survive_controller_replacement_roots() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    rendered = _RenderedLayout()
    rendered.feed(html)

    assert rendered.element("main", "conversation-shell")["aria-labelledby"] == (
        "conversation-heading"
    )
    assert rendered.element("section", "conversation")["aria-label"] == (
        "Conversation history"
    )
    assert "conversation-shell" in rendered.ancestors["conversation"]
    for persistent_id in (
        "conversation-heading",
        "selected-project-name",
        "selected-chat-name",
        "conversation-status",
        "conversation-context",
        "composer",
    ):
        assert "conversation" not in rendered.ancestors[persistent_id]

    assert rendered.element("aside", "task-inspector-panel")["aria-labelledby"] == (
        "task-inspector-heading"
    )
    assert rendered.element("div", "task-inspector")["aria-labelledby"] == (
        "task-inspector-heading"
    )
    assert "task-inspector-panel" in rendered.ancestors["task-inspector"]
    for persistent_id in (
        "task-inspector-heading",
        "task-inspector-summary",
        "task-controls",
        "activity-audit",
    ):
        assert "task-inspector" not in rendered.ancestors[persistent_id]

    assert "fable-status" not in rendered.ancestors["fable-avatar"]
    assert "sol-status" not in rendered.ancestors["sol-avatar"]
    assert "slack" not in html.lower()


def test_safe_rendering_preserves_untrusted_task_and_message_text() -> None:
    unsafe = '<img src=x onerror="globalThis.pwned=true">'
    harness = f"""
      const unsafe = {json.dumps(unsafe)};

      class Node {{
        constructor(tag) {{
          this.tag = tag;
          this.children = [];
          this.attributes = {{}};
          this.dataset = {{}};
          this.className = "";
          this._text = "";
          this.hidden = false;
          this.disabled = false;
        }}
        set textContent(value) {{ this._text = String(value); this.children = []; }}
        get textContent() {{ return this._text; }}
        set innerHTML(_value) {{ throw new Error("innerHTML must not be used"); }}
        append(...children) {{ this.children.push(...children); }}
        replaceChildren(...children) {{ this.children = [...children]; this._text = ""; }}
        setAttribute(name, value) {{ this.attributes[name] = String(value); }}
        addEventListener() {{}}
      }}

      const conversation = new Node("section");
      const taskList = new Node("aside");
      const inspector = new Node("aside");
      const roots = {{
        "#conversation": conversation,
        "#task-list": taskList,
        "#task-inspector": inspector,
      }};
      const created = [];
      const documentRoot = {{
        createElement(tag) {{ const node = new Node(tag); created.push(node); return node; }},
        querySelector(selector) {{ return roots[selector] ?? null; }},
      }};

      bridge.renderMessage(documentRoot, {{
        actor: "fable",
        kind: "message",
        payload: {{text: unsafe}},
        created_at: "2026-08-10T01:02:03Z",
      }});
      if (conversation.children[0].children[1].textContent !== unsafe) process.exit(2);
      if (conversation.children[0].attributes["aria-label"] !== "Fable message") process.exit(3);

      const brief = {{
        task_id: "task-1", revision: 4, title: unsafe, objective: unsafe,
        context: [unsafe], constraints: [unsafe], allowed_paths: [unsafe],
        out_of_scope: [unsafe], acceptance_criteria: [unsafe],
        required_tests: [unsafe], risks: [unsafe], open_questions: [unsafe],
        confidence: 0.7, confidence_rationale: unsafe,
      }};
      bridge.renderTaskList(documentRoot, [{{
          task_id: "task-1", revision: 4, state: "awaiting_user_approval", brief,
          approved_at: "2026-08-10T01:00:00Z", correction_count: 2,
          continuation_state: "sol_correcting", updated_at: "2026-08-10T01:02:03Z",
          active_agent: "sol", active_started_at: "2026-08-10T01:01:00Z",
      }}], "task-1", () => {{}});
      bridge.renderTaskInspector(documentRoot, {{
        task_id: "task-1", revision: 4, state: "awaiting_user_approval", brief,
        approved_at: "2026-08-10T01:00:00Z", correction_count: 2,
        continuation_state: "sol_correcting", updated_at: "2026-08-10T01:02:03Z",
        active_agent: "sol", active_started_at: "2026-08-10T01:01:00Z",
        outcome: {{
          summary: "Sol summary", changed_files: ["changed.py"],
          known_failures: ["known failure"], remaining_risks: ["Sol risk"],
          architecture_docs: "Architecture impact",
          question: {{
            ambiguity: "Ambiguous scope", why_it_matters: "Could widen changes",
            options: ["Option A", "Option B"], recommendation: "Option A",
            can_continue_safely: false,
          }},
          command_claims: [{{command_sha256: "abc123", exit_code: 0}}],
        }},
        clarification: {{
          status: "escalate_to_user", reasoning: "Clarification reasoning",
          confidence: 0.4, scope_changed: false, question_for_user: "Clarification question",
        }},
        activity: {{type: "command_execution", status: "completed", command_sha256: "activity-hash"}},
        review: {{
          status: "corrections_required", summary: "Review summary",
          test_assessment: "Test assessment", scope_violations: ["scope issue"],
          remaining_risks: ["Fable risk"], corrections: ["fix it"],
          criteria: [{{criterion: "criterion", evidence: ["criterion proof"], satisfied: false}}],
          question_for_user: "Review question",
        }},
      }}, {{fableReady: true, acknowledged: true, onAction: () => {{}}}});

      const allText = created.map((node) => node.textContent).join("\\n");
      if (!allText.includes(unsafe)) process.exit(4);
      if (created.some((node) => node.tag === "script" || node.tag === "img")) process.exit(5);
      if (created.some((node) => Object.values(node.attributes).includes(unsafe))) process.exit(6);
      if (globalThis.pwned === true) process.exit(7);
      for (const evidence of [
        "changed.py", "known failure", "Sol risk", "Test assessment",
        "scope issue", "Fable risk", "fix it", "criterion proof",
        "Architecture impact", "Ambiguous scope", "Could widen changes", "Option B",
            "Clarification reasoning", "Clarification question", "Review question",
            "activity-hash",
            "Approved at 2026-08-10T01:00:00Z", "Correction count", "2",
            "Sol Correcting", "2026-08-10T01:02:03Z", "Sol · started 2026-08-10T01:01:00Z",
      ]) {{
        if (!allText.includes(evidence)) process.exit(8);
      }}
      const firstNewNode = created.length;
      for (const state of ["awaiting_user_approval", "awaiting_scope_approval"]) {{
        bridge.renderTaskInspector(documentRoot, {{
          task_id: `missing-${{state}}`, revision: 1, state, brief: null,
        }}, {{fableReady: true, acknowledged: true, onAction: () => {{}}}});
      }}
      const nullBriefButtons = created.slice(firstNewNode)
        .filter((node) => node.tag === "button")
        .map((node) => node.textContent);
      if (JSON.stringify(nullBriefButtons) !== JSON.stringify(["Reject", "Reject"])) process.exit(9);
      const nullBriefText = created.slice(firstNewNode).map((node) => node.textContent).join("\\n");
      if (!nullBriefText.includes("Active run") || !nullBriefText.includes("No active run")) process.exit(11);
      const runningStart = created.length;
      bridge.renderTaskInspector(documentRoot, {{
        task_id: "running-task", revision: 2, state: "sol_running",
        brief: {{...brief, task_id: "different-task", revision: 1}},
      }}, {{fableReady: false, acknowledged: false, onAction: () => {{}}}});
      const runningButtons = created.slice(runningStart)
        .filter((node) => node.tag === "button")
        .map((node) => node.textContent);
      if (!runningButtons.includes("Stop") || runningButtons.includes("Approve & run") || runningButtons.includes("Edit")) process.exit(10);
      const unresolvedStart = created.length;
      bridge.renderTaskInspector(documentRoot, {{
        task_id: "unresolved-task", revision: 1, state: "awaiting_user_approval",
        brief: {{
          ...brief,
          task_id: "unresolved-task",
          revision: 1,
          open_questions: ["Which path is authoritative?"],
        }},
      }}, {{gate: {{ready: true}}, onAction: () => {{}}}});
      const buttons = created.slice(unresolvedStart).filter((node) => node.tag === "button");
      const approve = buttons.find((node) => node.textContent === "Approve & run");
      const edit = buttons.find((node) => node.textContent === "Edit");
      const reject = buttons.find((node) => node.textContent === "Reject");
      if (!approve || !approve.disabled) process.exit(30);
      if (!edit || edit.disabled) process.exit(31);
      if (!reject || reject.disabled) process.exit(32);
      const unresolvedText = created.slice(unresolvedStart).map((node) => node.textContent).join("\\n");
      if (!unresolvedText.includes("Resolve or remove the open questions in Edit before approval.")) {{
        process.exit(33);
      }}
      const failedStart = created.length;
      bridge.renderTaskInspector(documentRoot, {{
        task_id: "failed-task", revision: 1, state: "failed",
        brief: {{
          ...brief,
          task_id: "failed-task",
          revision: 1,
          open_questions: ["Which path is authoritative?"],
        }},
      }}, {{gate: {{ready: true}}, onAction: () => {{}}}});
      const failedButtons = created.slice(failedStart).filter((node) => node.tag === "button");
      if (failedButtons.some((node) => node.textContent === "Approve & run")) {{
        process.exit(34);
      }}
    """
    _run_module_harness(harness)


def test_revision_payloads_bind_approval_and_edit_to_the_displayed_contract() -> None:
    harness = r"""
      const displayed = {
        task_id: "task-7",
        revision: 8,
        title: "Old title",
        objective: "Old objective",
        context: ["old context"],
        constraints: ["old constraint"],
        allowed_paths: ["old/path"],
        out_of_scope: ["old scope"],
        acceptance_criteria: ["old criterion"],
        required_tests: ["old test"],
        risks: ["old risk"],
        open_questions: ["old question"],
        confidence: 0.4,
        confidence_rationale: "old rationale",
      };
      const formValues = {
        title: "New title",
        objective: "New objective",
        context: "context one\ncontext two",
        constraints: "constraint one\nconstraint two",
        allowed_paths: "src/agent_bridge\ntests/agent_bridge",
        out_of_scope: "outside-project\ndata",
        acceptance_criteria: "criterion one\ncriterion two",
        required_tests: "test one\ntest two",
        risks: "risk one\nrisk two",
        open_questions: "question one\nquestion two",
        confidence: "0.93",
        confidence_rationale: "New rationale",
      };
      const expectedEdit = {
        task_id: "task-7",
        revision: 9,
        title: "New title",
        objective: "New objective",
        context: ["context one", "context two"],
        constraints: ["constraint one", "constraint two"],
        allowed_paths: ["src/agent_bridge", "tests/agent_bridge"],
        out_of_scope: ["outside-project", "data"],
        acceptance_criteria: ["criterion one", "criterion two"],
        required_tests: ["test one", "test two"],
        risks: ["risk one", "risk two"],
        open_questions: ["question one", "question two"],
        confidence: 0.93,
        confidence_rationale: "New rationale",
      };
      if (JSON.stringify(bridge.approvalPayload(displayed)) !== '{"revision":8}') {
        process.exit(2);
      }
      const edited = bridge.editedRevision(displayed, formValues);
      if (JSON.stringify(edited) !== JSON.stringify(expectedEdit)) process.exit(3);
      if (Object.keys(edited).length !== 14) process.exit(4);
      displayed.revision = 100;
      if (edited.revision !== 9) process.exit(5);

      const mismatch = {
        task_id: "task-7", revision: 9,
        state: "awaiting_user_approval", brief: {...displayed, revision: 8},
      };
      if (bridge.canonicalTaskBrief(mismatch) !== null) process.exit(6);
      let failedClosed = false;
      try { bridge.approvalPayload(bridge.canonicalTaskBrief(mismatch)); }
      catch { failedClosed = true; }
      if (!failedClosed) process.exit(7);
    """
    _run_module_harness(harness)


def test_state_controls_and_subscription_acknowledgement_gate_fail_closed() -> None:
    harness = """
      const absent = bridge.subscriptionGate({});
      if (absent.ready || absent.canCompose || !absent.guidance.includes("checking")) {
        process.exit(2);
      }
      const unacknowledged = bridge.subscriptionGate({
        fable_ready: true,
        fable_status: "subscription_ready",
      });
      if (!unacknowledged.fableReady || unacknowledged.canCompose) process.exit(3);
      const ready = bridge.subscriptionGate({
        fable_ready: true,
        fable_status: "subscription_ready",
        sol_status: "ready",
        usage_credits_acknowledged: true,
      });
      if (!ready.ready || !ready.canCompose || !ready.solReady) process.exit(4);
      const solUnavailable = bridge.subscriptionGate({
        fable_ready: true,
        fable_status: "subscription_ready",
        sol_status: "unavailable",
        usage_credits_acknowledged: true,
      });
      if (solUnavailable.ready || solUnavailable.canCompose || solUnavailable.solReady) {
        process.exit(18);
      }
      if (!solUnavailable.guidance.includes("Sol")) process.exit(19);
      const subscriptionStatus = bridge.subscriptionGate({
        fable_status: {ready: true, authentication: "subscription"},
        usage_credits_acknowledged: true,
      });
      if (subscriptionStatus.ready || subscriptionStatus.canCompose) process.exit(5);
      const missingFlag = bridge.subscriptionGate({
        fable_status: "subscription_ready",
        usage_credits_acknowledged: true,
      });
      if (missingFlag.ready || missingFlag.canCompose) process.exit(16);
      const apiStatus = bridge.subscriptionGate({
        fable_status: {ready: true, authentication: "api_key"},
        usage_credits_acknowledged: true,
      });
      if (apiStatus.ready) process.exit(6);
      const inconsistent = bridge.subscriptionGate({
        fable_ready: true,
        fable_status: "subscription_unavailable",
        usage_credits_acknowledged: true,
      });
      if (inconsistent.ready || inconsistent.canCompose) process.exit(15);

      const waiting = bridge.controlsForState("awaiting_user_approval", absent);
      if (!waiting.approve.visible || waiting.approve.enabled) process.exit(7);
      if (!waiting.edit.visible || waiting.edit.enabled) process.exit(8);
      if (waiting.stop.visible || waiting.resume.visible) process.exit(9);
      if (!waiting.reject.visible || !waiting.reject.enabled) process.exit(17);

      const running = bridge.controlsForState("sol_running", absent);
      if (!running.stop.visible || !running.stop.enabled) process.exit(10);
      if (running.approve.visible || running.resume.visible) process.exit(11);

      const interrupted = bridge.controlsForState("interrupted", ready);
      if (!interrupted.resume.visible || !interrupted.resume.enabled) process.exit(12);
      if ("reviewState" in interrupted || interrupted.stop.visible) process.exit(13);

      const answering = bridge.controlsForState("awaiting_user_input", ready);
      if (!answering.answer.visible || !answering.answer.enabled) process.exit(14);
    """
    _run_module_harness(harness)


def test_injected_bootstrap_and_events_build_latest_task_state_without_live_calls() -> None:
    harness = r"""
      const bootstrap = {
        csrf_token: "csrf",
        session_id: "session-1",
        fable_ready: true,
        fable_status: "subscription_ready",
        usage_credits_acknowledged: true,
        sol_status: "ready",
        repository: "/repo",
        branch: "feat/bridge",
        replay_after: 42,
        tasks: [],
      };
      const initial = bridge.applyBootstrap({lastSequence: 0}, bootstrap);
      if (!initial.gate.ready || initial.sessionId !== "session-1") process.exit(2);
      if (initial.csrfToken !== "csrf" || initial.tasks.length !== 0) process.exit(3);
      if (initial.lastSequence !== 42) process.exit(18);

      const brief = {
        task_id: "task-1", revision: 1, title: "First", objective: "Build it",
        context: [], constraints: [], allowed_paths: ["src/agent_bridge"],
        out_of_scope: [], acceptance_criteria: ["works"], required_tests: ["test.py"],
        risks: [], open_questions: [], confidence: 0.9, confidence_rationale: "clear",
      };
      let tasks = bridge.reduceTaskEvent([], {
        task_id: "task-1", kind: "task_brief", payload: {brief},
      });
      tasks = bridge.reduceTaskEvent(tasks, {
        task_id: "task-2", kind: "task_state",
        payload: {state: "fable_planning", revision: 0},
      });
      tasks = bridge.reduceTaskEvent(tasks, {
        task_id: "task-1", kind: "task_state",
        payload: {state: "sol_running", revision: 1},
      });
      if (tasks[0].task_id !== "task-1" || tasks[0].state !== "sol_running") {
        process.exit(4);
      }
      if (tasks[0].brief.title !== "First" || tasks[1].task_id !== "task-2") {
        process.exit(5);
      }

      const repaired = bridge.applyBootstrap(
        {selectedTaskId: "vanished"},
        {...bootstrap, tasks},
      );
      if (repaired.selectedTaskId !== "task-1") process.exit(6);
      const sequenceOrdered = bridge.applyBootstrap(
        {},
        {...bootstrap, tasks: [
          {...tasks[0], updated_at: "2026-08-10T00:00:01Z"},
          {...tasks[1], updated_at: "2026-08-10T00:00:59Z"},
        ]},
      ).tasks;
      if (sequenceOrdered[0].task_id !== "task-1") process.exit(20);
      if (bridge.deriveSolStatus(tasks, "ready") !== "running") process.exit(7);
      if (bridge.elapsedLabel(
        "2026-08-10T00:00:00Z",
        Date.parse("2026-08-10T01:01:01Z"),
      ) !== "1h 1m") process.exit(8);
      if (bridge.elapsedLabel("not-a-date", 0) !== "elapsed unavailable") process.exit(9);
      if (bridge.repairTaskSelection(tasks, "vanished") !== "task-1") process.exit(10);
      if (bridge.repairTaskSelection([], "task-1") !== null) process.exit(11);

      const revisionOne = {
        task_id: "revision-task", revision: 1, state: "completed", brief: {...brief, task_id: "revision-task"},
        revision_start_sequence: 80,
        outcome: {summary: "rev1 outcome"}, review: {summary: "rev1 review"},
        clarification: {reasoning: "rev1 clarification"}, activity: {status: "rev1 activity"},
        history: [{sequence: 90, kind: "outcome"}],
      };
      const sameRevision = bridge.applyBootstrap(
        {tasks: [revisionOne], selectedTaskId: "revision-task"},
        {...bootstrap, tasks: [{...revisionOne, outcome: {summary: "authoritative outcome"}, history: undefined}]},
      ).tasks[0];
      if (sameRevision.outcome.summary !== "authoritative outcome" || sameRevision.history.length !== 1) process.exit(12);
      const revisionTwo = bridge.applyBootstrap(
        {tasks: [revisionOne], selectedTaskId: "revision-task"},
        {...bootstrap, tasks: [{
          ...revisionOne, revision: 2, revision_start_sequence: 100,
          brief: {...revisionOne.brief, revision: 2}, outcome: null, review: null,
          clarification: null, activity: null,
        }]},
      ).tasks[0];
      for (const field of ["outcome", "review", "clarification", "activity"]) {
        if (revisionTwo[field] !== null) process.exit(13);
      }
      if (revisionTwo.history?.length) process.exit(14);

      let replayRevision = bridge.reduceTaskEvent([], {
        sequence: 1, task_id: "replay-revision", actor: "fable", kind: "task_brief",
        payload: {brief: {...brief, task_id: "replay-revision"}}, created_at: "2026-08-10T00:00:00Z",
      });
      replayRevision = bridge.reduceTaskEvent(replayRevision, {
        sequence: 2, task_id: "replay-revision", actor: "sol", kind: "outcome",
        payload: {summary: "rev1"}, created_at: "2026-08-10T00:00:01Z",
      });
      replayRevision[0] = {
        ...replayRevision[0], approved_at: "2026-08-10T00:00:01Z",
        correction_count: 2, continuation_state: "sol_correcting",
        active_agent: "sol", active_started_at: "2026-08-10T00:00:01Z",
      };
      replayRevision = bridge.reduceTaskEvent(replayRevision, {
        sequence: 3, task_id: "replay-revision", actor: "fable", kind: "task_brief",
        payload: {brief: {...brief, task_id: "replay-revision", revision: 2}}, created_at: "2026-08-10T00:00:02Z",
      });
      if (replayRevision[0].outcome !== undefined || replayRevision[0].history.length !== 1 || replayRevision[0].history[0].sequence !== 3) process.exit(15);
      if (replayRevision[0].approved_at !== null || replayRevision[0].correction_count !== 0
          || replayRevision[0].continuation_state !== null || replayRevision[0].active_agent !== null
          || replayRevision[0].active_started_at !== null) process.exit(19);
      replayRevision = bridge.reduceTaskEvent(replayRevision, {
        sequence: 4, task_id: "replay-revision", actor: "sol", kind: "outcome",
        payload: {summary: "rev2"}, created_at: "2026-08-10T00:00:03Z",
      });
      replayRevision = bridge.reduceTaskEvent(replayRevision, {
        sequence: 5, task_id: "replay-revision", actor: "coordinator", kind: "task_state",
        payload: {state: "sol_running", revision: 3}, created_at: "2026-08-10T00:00:04Z",
      });
      if (replayRevision[0].outcome !== undefined || replayRevision[0].history.length !== 1 || replayRevision[0].history[0].sequence !== 5) process.exit(17);

      let boundedReplay = [{
        ...revisionOne, task_id: "bounded-replay", state: "sol_running",
        brief: {...brief, task_id: "bounded-replay"}, revision_start_sequence: 10,
      }];
      for (const event of [
        {sequence: 9, kind: "outcome", payload: {summary: "older outcome"}},
        {kind: "review", payload: {summary: "unknown review"}},
        {sequence: 9, kind: "task_state", payload: {state: "completed", revision: 1}},
      ]) {
        boundedReplay = bridge.reduceTaskEvent(boundedReplay, {
          ...event, task_id: "bounded-replay", actor: "coordinator",
          created_at: "2026-08-10T00:00:09Z",
        });
      }
      if (boundedReplay[0].outcome.summary !== "rev1 outcome"
          || boundedReplay[0].review.summary !== "rev1 review"
          || boundedReplay[0].state !== "sol_running") process.exit(21);
      boundedReplay = bridge.reduceTaskEvent(boundedReplay, {
        sequence: 12, task_id: "bounded-replay", actor: "fable", kind: "task_brief",
        payload: {brief: {...brief, task_id: "bounded-replay"}},
        created_at: "2026-08-10T00:00:12Z",
      });
      if (boundedReplay[0].revision_start_sequence !== 12
          || boundedReplay[0].outcome !== undefined || boundedReplay[0].review !== undefined
          || boundedReplay[0].history.length !== 1
          || boundedReplay[0].history[0].sequence !== 12) process.exit(22);
      if (bridge.associatedRevisionForEvent({sequence: 9, payload: {}}, boundedReplay[0]) !== null) process.exit(23);
      if (bridge.associatedRevisionForEvent({payload: {}}, boundedReplay[0]) !== null) process.exit(24);
      if (bridge.associatedRevisionForEvent({sequence: 13, payload: {}}, boundedReplay[0]) !== 1) process.exit(25);
    """
    _run_module_harness(harness)


def test_every_persisted_event_kind_reduces_and_renders_full_safe_details() -> None:
    harness = r"""
      class Node {
        constructor(tag) {
          this.tag = tag; this.children = []; this.attributes = {};
          this.className = ""; this._text = "";
        }
        set textContent(value) { this._text = String(value); this.children = []; }
        get textContent() { return this._text; }
        append(...children) { this.children.push(...children); }
        removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
        replaceChildren(...children) { this.children = [...children]; this._text = ""; }
        setAttribute(name, value) { this.attributes[name] = String(value); }
      }
      const conversation = new Node("section");
      const created = [];
      const documentRoot = {
        createElement(tag) { const node = new Node(tag); created.push(node); return node; },
        querySelector(selector) { return selector === "#conversation" ? conversation : null; },
      };
      const events = [
        ["message", {text: "hello"}],
        ["task_brief", {brief: {
          task_id: "task-1", revision: 1, title: "Task", objective: "Objective",
          context: [], constraints: [], allowed_paths: ["tools"], out_of_scope: [],
          acceptance_criteria: ["works"], required_tests: ["test.py"], risks: [],
          open_questions: [], confidence: 1, confidence_rationale: "clear",
        }}],
        ["task_state", {state: "sol_running", revision: 1}],
        ["outcome", {status: "question", summary: "Need input", architecture_docs: "unchanged", question: {ambiguity: "which", why_it_matters: "scope", options: ["a", "b"], recommendation: "a", can_continue_safely: false}}],
        ["clarification", {status: "escalate_to_user", reasoning: "uncertain", question_for_user: "Choose one", confidence: 0.4, scope_changed: false, revised_brief: null, answer: null}],
        ["review", {status: "escalate_to_user", summary: "review", test_assessment: "tests", criteria: [], scope_violations: [], remaining_risks: ["risk"], corrections: [], question_for_user: "Review question"}],
        ["resume_drift", {changed_paths: ["changed.py"], unexpected_paths: []}],
        ["stop_error", {run_id: "run-1"}],
        ["action_error", {action: "approve", error_type: "RuntimeError"}],
        ["task_rejected", {revision: 1}],
        ["agent_event", {type: "command_execution", status: "completed", command_sha256: "hash-only", exit_code: 0}],
      ].map(([kind, payload], index) => ({
        sequence: index + 1, session_id: "session-1", task_id: "task-1",
        actor: kind === "outcome" ? "sol" : kind === "review" ? "fable" : "coordinator",
        kind, payload, created_at: `2026-08-10T00:00:${String(index).padStart(2, "0")}Z`,
      }));

      let tasks = [];
      const agentEvent = events.find((event) => event.kind === "agent_event");
      const taskState = events.find((event) => event.kind === "task_state");
      if (bridge.conversationPresentation(agentEvent) !== "hidden") process.exit(20);
      if (bridge.conversationPresentation(taskState) !== "status") process.exit(21);
      if (bridge.conversationPresentation({kind: "future_event"}) !== "hidden") process.exit(22);
      for (const event of events) {
        tasks = bridge.reduceTaskEvent(tasks, event);
        const task = tasks.find((item) => item.task_id === event.task_id);
        bridge.renderConversationEvent(
          documentRoot,
          event,
          bridge.associatedRevisionForEvent(event, task),
        );
      }
      const task = tasks[0];
      if (task.state !== "failed") process.exit(2);
      if (task.history.length !== events.length - 1 || task.history[0].sequence !== 2) process.exit(3);
      if (task.activity.command_sha256 !== "hash-only") process.exit(23);
      if (task.outcome.architecture_docs !== "unchanged") process.exit(4);
      if (task.clarification.question_for_user !== "Choose one") process.exit(5);
      if (task.review.question_for_user !== "Review question") process.exit(6);
      const fullText = created.map((node) => node.textContent).join("\n");
      for (const expected of ["why_it_matters", "changed.py", "RuntimeError", "Review question"]) {
        if (!fullText.includes(expected)) process.exit(7);
      }
      if (created.some((node) => node.textContent === "Command Execution")) process.exit(24);
      const statusRows = conversation.children.filter(
        (node) => node.className === "conversation-status",
      );
      if (statusRows.length !== 1 || !statusRows[0].textContent.includes("Sol Running")) {
        process.exit(25);
      }
      if (created.filter((node) => node.tag === "details").length !== events.length - 2) process.exit(8);
      if (!fullText.includes("#1") || !fullText.includes("task-1") || !fullText.includes("message")) process.exit(9);
      if (!fullText.includes('"sequence": 1') || !fullText.includes('"kind": "message"')) process.exit(10);
      if (!fullText.includes("r1") || !fullText.includes('"revision": 1')) process.exit(11);
      const outcomeArticle = conversation.children[3];
      if (!outcomeArticle.children[2].textContent.includes("r1")) process.exit(12);
      if (!outcomeArticle.children[3].children[1].textContent.includes('"revision": 1')) process.exit(13);

      conversation.children = [];
      for (const event of [
        {actor: "user", kind: "message", payload: {text: "Planning request"}},
        {actor: "coordinator", kind: "agent_event", payload: {type: "command_execution"}},
        {actor: "coordinator", kind: "agent_event", payload: {type: "tool_call"}},
        {actor: "coordinator", kind: "agent_event", payload: {type: "tool_result"}},
        {actor: "fable", kind: "task_brief", payload: {brief: {title: "Task", revision: 1}}},
      ]) {
        bridge.renderConversationEvent(documentRoot, event);
      }
      if (conversation.children.length !== 2) process.exit(26);
      if (conversation.children[0].children[1].textContent !== "Planning request") process.exit(27);
      if (conversation.children[1].children[1].textContent !== "Task brief ready: Task · revision 1") process.exit(28);
    """
    _run_module_harness(harness)


def test_conversation_is_bounded_and_metadata_classes_fail_closed() -> None:
    harness = r"""
      class Node {
        constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.className = ""; this._text = ""; }
        set textContent(value) { this._text = String(value); this.children = []; }
        get textContent() { return this._text; }
        append(...children) { this.children.push(...children); }
        removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
        replaceChildren(...children) { this.children = [...children]; this._text = ""; }
        setAttribute(name, value) { this.attributes[name] = String(value); }
        remove() { this.removed = true; }
      }
      const conversation = new Node("section");
      const inspector = new Node("aside");
      const created = [];
      const documentRoot = {
        createElement(tag) { const node = new Node(tag); created.push(node); return node; },
        querySelector(selector) {
          if (selector === "#conversation") return conversation;
          if (selector === "#task-inspector") return inspector;
          return null;
        },
      };
      for (let sequence = 1; sequence <= bridge.MAX_CONVERSATION_MESSAGES + 5; sequence += 1) {
        bridge.renderMessage(documentRoot, {
          sequence, actor: "user", kind: "message", payload: {text: String(sequence)},
          created_at: sequence === 1 ? "not-a-date" : "2026-08-10T00:00:00Z",
        });
      }
      if (conversation.children.length !== bridge.MAX_CONVERSATION_MESSAGES) process.exit(2);
      const invalidTime = created.find((node) => node.tag === "time" && node.textContent === "not-a-date");
      if (invalidTime) process.exit(3);

      const brief = {task_id: "task-1", revision: 1, title: "Task", objective: "Objective", context: [], constraints: [], allowed_paths: ["tools"], out_of_scope: [], acceptance_criteria: ["works"], required_tests: ["test.py"], risks: [], open_questions: [], confidence: 1, confidence_rationale: "clear"};
      bridge.renderTaskInspector(documentRoot, {task_id: "task-1", revision: 1, state: "bad class payload", brief}, {fableReady: true, acknowledged: true});
      if (created.some((node) => node.className.includes("bad class payload"))) process.exit(4);

      let tasks = [];
      for (let sequence = 1; sequence <= bridge.MAX_TASK_HISTORY + 5; sequence += 1) {
        tasks = bridge.reduceTaskEvent(tasks, {
          sequence, task_id: "task-2", actor: "coordinator", kind: "action_error",
          payload: {action: "test", error_type: "Error"},
          created_at: "2026-08-10T00:00:00Z",
        });
      }
      if (tasks[0].history.length !== bridge.MAX_TASK_HISTORY) process.exit(5);
      if (tasks[0].history[0].sequence !== 6) process.exit(6);
      const oversizedHistory = Array.from(
        {length: bridge.MAX_TASK_HISTORY + 5},
        (_, index) => ({sequence: index + 1}),
      );
      const bootstrapped = bridge.applyBootstrap(
        {tasks: [{task_id: "task-3", revision: 0, history: oversizedHistory}]},
        {session_id: "session-1", tasks: [{task_id: "task-3", revision: 0, state: "fable_planning", brief: null}]},
      );
      if (bootstrapped.tasks[0].history.length !== bridge.MAX_TASK_HISTORY) process.exit(7);
      if (bootstrapped.tasks[0].history[0].sequence !== 6) process.exit(8);
      let replayTasks = [];
      for (let index = 0; index < bridge.MAX_TASK_OVERVIEWS + 5; index += 1) {
        replayTasks = bridge.reduceTaskEvent(replayTasks, {
          sequence: index + 1, task_id: `replay-${String(index).padStart(3, "0")}`,
          actor: "coordinator", kind: "task_state",
          payload: {state: "fable_planning", revision: 0},
          created_at: "2026-08-10T00:00:00Z",
        });
      }
      if (replayTasks.length !== bridge.MAX_TASK_OVERVIEWS) process.exit(9);
      if (replayTasks[0].task_id !== `replay-${String(bridge.MAX_TASK_OVERVIEWS + 4).padStart(3, "0")}`) process.exit(10);
    """
    _run_module_harness(harness)


def test_directed_conversation_cards_use_safe_envelopes_and_exact_bindings() -> None:
    harness = r"""
      class Node {
        constructor(tag) {
          this.tag = tag; this.children = []; this.attributes = {}; this.dataset = {};
          this.className = ""; this._text = ""; this.listeners = {}; this.disabled = false;
        }
        set textContent(value) { this._text = String(value); this.children = []; }
        get textContent() { return this._text; }
        append(...children) { this.children.push(...children); }
        replaceChildren(...children) { this.children = [...children]; this._text = ""; }
        setAttribute(name, value) { this.attributes[name] = String(value); }
        addEventListener(name, listener) { this.listeners[name] = listener; }
        remove() { this.removed = true; }
      }
      const conversation = new Node("section");
      const roots = {"#conversation": conversation, "#conversation-empty": null};
      const documentRoot = {
        createElement(tag) { return new Node(tag); },
        querySelector(selector) { return roots[selector] ?? null; },
      };
      const solQuestion = {
        kind: "conversation", actor: "sol", task_id: "task-a", sequence: 8,
        payload: {
              sender: "sol", addressed_to: "fable", routed_to: "fable", message_type: "question",
              text: "Need the exact test evidence.", task_id: "task-a", revision: 2,
              continuation_generation: 3, question_id: "question-a", reply_to_question_id: null,
        },
      };
      const card = bridge.renderConversationEvent(documentRoot, solQuestion);
      if (!card || !card.className.includes("message-sol") || !card.className.includes("target-fable")) process.exit(2);
      const cardText = card.children.map((node) => node.textContent).join("\n");
      if (!cardText.includes("Sol → Fable") || !cardText.includes("Task task-a · r2") || !cardText.includes("Question question-a")) process.exit(3);
      if (!cardText.includes("S")) process.exit(4);
      const routed = bridge.renderConversationEvent(documentRoot, {
        kind: "conversation", actor: "user", task_id: null, payload: {
              sender: "user", addressed_to: "sol", routed_to: "fable", message_type: "statement",
              text: "Please plan this.", task_id: null, revision: null,
              continuation_generation: null, question_id: null, reply_to_question_id: null,
        },
      });
      if (!routed || !routed.children.map((node) => node.textContent).join("\n").includes("User → Fable")) process.exit(14);
      if (!routed.children.map((node) => node.textContent).join("\n").includes("Addressed to Sol · routed to Fable before approval")) process.exit(15);
      if (bridge.renderConversationEvent(documentRoot, {kind: "conversation", payload: {sender: "<img>", text: "unsafe"}}) !== null) process.exit(5);
      if (bridge.renderConversationEvent(documentRoot, {kind: "conversation", actor: "sol", payload: {sender: "fable", addressed_to: "user", routed_to: "user", message_type: "statement", text: "ambiguous"}}) !== null) process.exit(16);
      if (bridge.renderConversationEvent(documentRoot, {kind: "agent_event", payload: {type: "command"}}) !== null) process.exit(6);

      const bindings = [];
      bridge.renderPendingConversationCards(documentRoot, [
        {task_id: "task-a", revision: 2, continuation_generation: 3, pending_question: {
          question_id: "question-a", asked_by: "sol", addressed_to: "user", routed_to: "user",
          text: "Which test?", revision: 2, continuation_generation: 3,
        }},
        {task_id: "task-b", revision: 5, continuation_generation: 7, pending_question: {
          question_id: "question-b", asked_by: "fable", addressed_to: "user", routed_to: "user",
          text: "Which scope?", revision: 5, continuation_generation: 7,
        }},
        {task_id: "task-c", revision: 1, continuation_generation: 4, exchange_permission: {
          request_id: "permission-c", revision: 1, continuation_generation: 4,
        }},
      ], {
        onReply(binding) { bindings.push(binding); },
        onGrant(binding) { bindings.push(binding); },
      }, {projectId: "project-a", sessionId: "chat-a"});
      const actionCards = conversation.children.filter((node) => node.className === "conversation-action-card");
      if (actionCards.length !== 3) process.exit(7);
      const buttons = actionCards.flatMap((card) => card.children).filter((node) => node.tag === "button");
      buttons.find((button) => button.textContent === "Reply").listeners.click();
      buttons.find((button) => button.textContent === "Allow 3 more exchanges").listeners.click();
      if (bindings.length !== 2 || bindings[0].taskId !== "task-a" || bindings[0].questionId !== "question-a") process.exit(8);
      if (bindings[1].requestId !== "permission-c" || bindings[1].continuationGeneration !== 4) process.exit(9);

      const state = {projectId: "project-a", sessionId: "chat-a"};
      const ordinary = bridge.composerRequest(state, null, "Plan this", "sol");
      if (ordinary.path !== "/api/projects/project-a/chats/chat-a/messages" || JSON.stringify(ordinary.payload) !== '{"text":"Plan this","addressed_to":"sol"}') process.exit(10);
      const answer = bridge.composerRequest(state, bindings[0], "Use the focused lane.", "fable");
      if (answer.path !== "/api/projects/project-a/chats/chat-a/tasks/task-a/answer" || JSON.stringify(answer.payload) !== '{"text":"Use the focused lane.","revision":2,"question_id":"question-a","continuation_generation":3}') process.exit(11);
      const grant = bridge.exchangeGrantRequest(state, bindings[1]);
      if (grant.path !== "/api/projects/project-a/chats/chat-a/tasks/task-c/exchanges/grant" || JSON.stringify(grant.payload) !== '{"revision":1,"continuation_generation":4,"request_id":"permission-c"}') process.exit(12);
      let staleFailedClosed = false;
      try { bridge.composerRequest({...state, sessionId: "other-chat"}, bindings[0], "wrong", "fable"); }
      catch { staleFailedClosed = true; }
      if (!staleFailedClosed) process.exit(13);
    """
    _run_module_harness(harness)


def test_directed_envelope_schema_matrix_matches_all_six_contract_types() -> None:
    harness = r"""
      class Node {
        constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; this.dataset = {}; this.className = ""; this._text = ""; }
        set textContent(value) { this._text = String(value); this.children = []; }
        get textContent() { return this._text; }
        append(...children) { this.children.push(...children); }
        setAttribute(name, value) { this.attributes[name] = String(value); }
        addEventListener() {}
      }
      const conversation = new Node("section");
      const documentRoot = {createElement(tag) { return new Node(tag); }, querySelector(selector) { return selector === "#conversation" ? conversation : null; }};
      const base = {sender: "user", addressed_to: "team", routed_to: "fable", text: "safe", task_id: null, revision: null, continuation_generation: null, question_id: null, reply_to_question_id: null};
      const valid = [
        {...base, message_type: "statement"},
        {...base, message_type: "intervention", task_id: "task-1", revision: 1, continuation_generation: 2},
        {...base, message_type: "approval", text: "Allowed 3 more exchanges.", task_id: "task-1", revision: 1},
        {...base, sender: "system", addressed_to: "user", routed_to: "user", message_type: "status", task_id: "task-1", revision: 1, continuation_generation: 2},
        {...base, sender: "sol", addressed_to: "fable", routed_to: "fable", message_type: "question", task_id: "task-1", revision: 1, continuation_generation: 2, question_id: "question-1"},
        {...base, sender: "fable", addressed_to: "sol", routed_to: "sol", message_type: "answer", task_id: "task-1", revision: 1, continuation_generation: 2, reply_to_question_id: "question-1"},
      ];
      for (const payload of valid) {
        const event = {kind: "conversation", actor: payload.sender, task_id: payload.task_id, payload};
        const card = bridge.renderConversationEvent(documentRoot, event);
        if (bridge.conversationPresentation(event) === "hidden" || card === null) process.exit(2);
        if (payload.message_type === "approval" && !card.children.map((node) => node.textContent).join("\n").includes("Allowed 3 more exchanges.")) process.exit(4);
      }
      const invalid = [
        {...base, message_type: "approval", task_id: "task-1", revision: 1, continuation_generation: 2},
        {...base, message_type: "question", task_id: "task-1", revision: 1, continuation_generation: 2},
        {...base, message_type: "answer", task_id: "task-1", revision: 1, continuation_generation: 2, question_id: "question-1"},
        {...base, message_type: "approval", task_id: "task-1", revision: 1, question_id: "question-1"},
        {...base, sender: "fable", message_type: "status"},
        {...base, message_type: "statement", task_id: "task-1", revision: null, continuation_generation: null},
        {...base, message_type: "statement", text: "unsafe\ncontrol"},
        {...base, message_type: "statement", unexpected: "audit only"},
      ];
      for (const payload of invalid) {
        const event = {kind: "conversation", actor: payload.sender, task_id: payload.task_id, payload};
        if (bridge.conversationPresentation(event) !== "hidden" || bridge.renderConversationEvent(documentRoot, event) !== null) process.exit(3);
      }
    """
    _run_module_harness(harness)


def test_directed_envelopes_match_real_store_outer_associations_and_text_contract(
    tmp_path,
) -> None:
    store = SQLiteStore(tmp_path / "directed-events.sqlite3", clock=lambda: "2026-08-12T00:00:00Z")
    store.create_session("chat-real", "/not-a-real-repository")
    store.prepare_new_request_action(
        project_id="project-real",
        session_id="chat-real",
        task_id="new-task",
        generation=1,
        payload=NewRequestPayload("Route this through the team.", ConversationTarget.TEAM),
    )
    store.append_event(
        "chat-real", "bound-task", "sol", "conversation", ConversationEnvelope(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.QUESTION,
            text="Which exact task contract applies?",
            task_id="bound-task",
            revision=2,
            continuation_generation=3,
            question_id="question-real",
        ).to_dict(),
    )
    store.append_event(
        "chat-real", "bound-task", "fable", "conversation", ConversationEnvelope(
            sender=ConversationActor.FABLE,
            addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL,
            message_type=ConversationMessageType.ANSWER,
            text="Use the persisted task contract.",
            task_id="bound-task",
            revision=2,
            continuation_generation=3,
            reply_to_question_id="question-real",
        ).to_dict(),
    )
    produced = [event.to_dict() for event in store.events_after("chat-real", 0)]
    store.close()
    harness = f"""
      const produced = {json.dumps(produced)};
      for (const event of produced) {{
        if (bridge.conversationPresentation(event) === "hidden") process.exit(2);
      }}
      const unbound = produced[0];
      if (unbound.task_id !== "new-task" || unbound.payload.task_id !== null) process.exit(3);
      const mismatchedOuter = {{...produced[1], task_id: "other-task"}};
      if (bridge.conversationPresentation(mismatchedOuter) !== "hidden") process.exit(4);
      const unsafeOuter = {{...produced[1], task_id: "bad/task"}};
      if (bridge.conversationPresentation(unsafeOuter) !== "hidden") process.exit(5);
      const base = {{...unbound.payload, message_type: "statement"}};
      const valid = [
        "x".repeat(16384),
        "é".repeat(8192),
        "😀".repeat(4096),
      ];
      for (const text of valid) {{
        if (bridge.conversationPresentation({{...unbound, payload: {{...base, text}}}}) === "hidden") process.exit(6);
      }}
      const originalTextEncoder = globalThis.TextEncoder;
      try {{
        delete globalThis.TextEncoder;
        if (bridge.conversationPresentation({{...unbound, payload: {{...base, text: "safe"}}}}) !== "hidden") process.exit(8);
        globalThis.TextEncoder = {{}};
        if (bridge.conversationPresentation({{...unbound, payload: {{...base, text: "safe"}}}}) !== "hidden") process.exit(9);
        globalThis.TextEncoder = class {{ encode() {{ throw new Error("encoder failed"); }} }};
        if (bridge.conversationPresentation({{...unbound, payload: {{...base, text: "safe"}}}}) !== "hidden") process.exit(10);
      }} finally {{
        globalThis.TextEncoder = originalTextEncoder;
      }}
      const invalid = [
        "   ", "\\u0085", "\\u2003", "x".repeat(16385), "é".repeat(8193), "😀".repeat(4097),
        "line\\nfeed", "bad\\u007fdelete", "high\\ud800", "low\\udc00",
      ];
      for (const text of invalid) {{
        if (bridge.conversationPresentation({{...unbound, payload: {{...base, text}}}}) !== "hidden") process.exit(7);
      }}
    """
    _run_module_harness(harness)


def test_composer_guidance_preserves_recipient_routing_through_lease_and_binding() -> None:
    harness = r"""
      const ready = {sessionId: "chat-a", gate: {canCompose: true, guidance: "Ready for a new Fable plan."}, activeLease: null};
      if (!bridge.composerGuidance(ready, null, "fable").includes("Fable is the direct planner")) process.exit(2);
      if (!bridge.composerGuidance(ready, null, "sol").includes("addressed to Sol are visibly routed through Fable")) process.exit(3);
      if (!bridge.composerGuidance(ready, null, "team").includes("addressed to Team are visibly routed through Fable")) process.exit(4);
      const leased = {...ready, activeLease: {projectId: "project-a"}};
      const leasedGuidance = bridge.composerGuidance(leased, null, "sol");
      if (!leasedGuidance.includes("An agent is active") || !leasedGuidance.includes("routed through Fable")) process.exit(5);
      const boundGuidance = bridge.composerGuidance(leased, {kind: "question"}, "sol");
      if (!boundGuidance.includes("exact task and continuation") || boundGuidance.includes("An agent is active")) process.exit(6);
      const sol = bridge.composerPresentation(ready, null, "sol");
      if (sol.disabled || sol.recipientDisabled || sol.label !== "Message Sol" || sol.submit !== "Send to Sol" || !sol.guidance.includes("routed through Fable")) process.exit(7);
      const lease = bridge.composerPresentation(leased, null, "team");
      if (!lease.disabled || !lease.recipientDisabled || !lease.guidance.includes("routed through Fable")) process.exit(8);
      const bound = bridge.composerPresentation(leased, {kind: "question"}, "sol");
      if (bound.disabled || !bound.recipientDisabled || bound.label !== "Bound reply" || bound.submit !== "Send reply") process.exit(9);
    """
    _run_module_harness(harness)


def test_project_chat_events_coalesce_selected_bootstrap_refresh_and_drop_stale_switch() -> None:
    harness = r"""
      const scheduled = [];
      const sockets = [];
      class Socket {
        constructor() { this.listeners = {}; sockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        close() { this.closed = true; }
      }
      const pendingTask = {
        task_id: "task-a", revision: 1, state: "awaiting_user_input", continuation_generation: 2,
        exchange_allowance: 0, exchange_consumed: 3, pending_question: {
          question_id: "question-a", asked_by: "sol", addressed_to: "user", routed_to: "user",
          text: "Which exact option?", revision: 1, continuation_generation: 2,
        }, exchange_permission: {request_id: "permission-a", revision: 1, continuation_generation: 2},
      };
      let alphaBootstraps = 0;
      const fetchFunction = (url) => {
        if (url === "/api/projects") return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf", usage_credits_acknowledged: true, projects: [
          {project_id: "alpha", label: "Alpha", branch: "main", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
          {project_id: "beta", label: "Beta", branch: "next", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
        ], active_lease: null})});
        if (url === "/api/projects/alpha/chats?limit=50") return Promise.resolve({ok: true, status: 200, json: async () => ({chats: [{session_id: "chat-a", title: "A", latest_sequence: 0}]})});
        if (url === "/api/projects/beta/chats?limit=50") return Promise.resolve({ok: true, status: 200, json: async () => ({chats: [{session_id: "chat-b", title: "B", latest_sequence: 0}]})});
        if (url === "/api/projects/alpha/chats/chat-a/bootstrap") {
          alphaBootstraps += 1;
          return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf", usage_credits_acknowledged: true, project_id: "alpha", session_id: "chat-a", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "main", replay_after: 0, tasks: alphaBootstraps === 1 ? [{task_id: "task-a", revision: 1, state: "sol_running"}] : [pendingTask]})});
        }
        if (url === "/api/projects/beta/chats/chat-b/bootstrap") return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf", usage_credits_acknowledged: true, project_id: "beta", session_id: "chat-b", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "next", replay_after: 0, tasks: []})});
        throw new Error(`unexpected ${url}`);
      };
      const controller = bridge.createProjectChatController({
        fetchFunction, WebSocketCtor: Socket,
        schedule(callback) { scheduled.push(callback); return scheduled.length; }, cancelSchedule() {},
        location: {protocol: "http:", host: "bridge.test"}, onState() {}, onEvent() {}, onStatus() {},
      });
      await controller.bootstrapInitial();
      sockets[0].listeners.message({data: JSON.stringify({sequence: 1, kind: "conversation", actor: "sol", task_id: "task-a", payload: {}})});
      sockets[0].listeners.message({data: JSON.stringify({sequence: 2, kind: "task_state", actor: "coordinator", task_id: "task-a", payload: {state: "awaiting_user_input", revision: 1}})});
      if (scheduled.length !== 1) process.exit(2);
      scheduled.shift()();
      for (let tick = 0; tick < 8; tick += 1) await Promise.resolve();
      if (alphaBootstraps !== 2 || controller.state.tasks[0].pending_question?.question_id !== "question-a" || controller.state.tasks[0].exchange_permission?.request_id !== "permission-a") process.exit(3);
      sockets[0].listeners.message({data: JSON.stringify({sequence: 3, kind: "task_state", actor: "coordinator", task_id: "task-a", payload: {state: "awaiting_user_input", revision: 1}})});
      if (scheduled.length !== 1) process.exit(4);
      await controller.selectProject("beta");
      scheduled.shift()();
      for (let tick = 0; tick < 8; tick += 1) await Promise.resolve();
      if (controller.state.projectId !== "beta" || controller.state.sessionId !== "chat-b" || alphaBootstraps !== 2) process.exit(5);
    """
    _run_module_harness(harness)


def test_project_chat_refresh_is_single_flight_cursor_guarded_and_cancellable() -> None:
    harness = r"""
      const scheduled = [];
      const sockets = [];
      const deferred = [];
      class Socket {
        constructor() { this.listeners = {}; sockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        close() { this.closed = true; }
      }
      const bootstrap = (projectId, sessionId, tasks = []) => ({
        csrf_token: "csrf", usage_credits_acknowledged: true, project_id: projectId, session_id: sessionId,
        fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", replay_after: 0, tasks,
      });
      const baseTask = {task_id: "task-a", revision: 1, state: "sol_running"};
      const freshTask = {task_id: "task-a", revision: 1, state: "awaiting_user_input", continuation_generation: 2,
        pending_question: {question_id: "question-a", asked_by: "sol", addressed_to: "user", routed_to: "user", text: "Which option?", revision: 1, continuation_generation: 2},
        exchange_permission: {request_id: "permission-a", revision: 1, continuation_generation: 2}};
      let alphaCalls = 0;
      const fetchFunction = (url) => {
        if (url === "/api/projects") return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf", usage_credits_acknowledged: true, projects: [
          {project_id: "alpha", label: "Alpha", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
          {project_id: "beta", label: "Beta", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
        ], active_lease: null})});
        if (url === "/api/projects/alpha/chats?limit=50") return Promise.resolve({ok: true, status: 200, json: async () => ({chats: [{session_id: "chat-a", title: "A", latest_sequence: 0}]})});
        if (url === "/api/projects/beta/chats?limit=50") return Promise.resolve({ok: true, status: 200, json: async () => ({chats: [{session_id: "chat-b", title: "B", latest_sequence: 0}]})});
        if (url === "/api/projects/beta/chats/chat-b/bootstrap") return Promise.resolve({ok: true, status: 200, json: async () => bootstrap("beta", "chat-b")});
        if (url === "/api/projects/alpha/chats/chat-a/bootstrap") {
          alphaCalls += 1;
          if (alphaCalls === 1 || alphaCalls === 4 || alphaCalls === 6 || alphaCalls === 8) return Promise.resolve({ok: true, status: 200, json: async () => bootstrap("alpha", "chat-a", [baseTask])});
          return new Promise((resolve) => deferred.push(resolve));
        }
        throw new Error(`unexpected ${url}`);
      };
      const flush = async () => { for (let tick = 0; tick < 12; tick += 1) await Promise.resolve(); };
      const controller = bridge.createProjectChatController({
        fetchFunction, WebSocketCtor: Socket,
        schedule(callback) { scheduled.push(callback); return scheduled.length; }, cancelSchedule() {},
        location: {protocol: "http:", host: "bridge.test"}, onState() {}, onEvent() {}, onStatus() {},
      });
      await controller.bootstrapInitial();
      const emit = (socket, sequence) => socket.listeners.message({data: JSON.stringify({sequence, kind: "conversation", actor: "sol", task_id: "task-a", payload: {}})});
      emit(sockets[0], 2);
      emit(sockets[0], 1);
      if (scheduled.length !== 1 || controller.state.lastSequence !== 2) process.exit(2);
      void controller.refreshSelectedBootstrap();
      if (alphaCalls !== 1 || scheduled.length !== 1) process.exit(14);
      scheduled.shift()();
      await flush();
      if (alphaCalls !== 2 || deferred.length !== 1) process.exit(3);
      emit(sockets[0], 3);
      emit(sockets[0], 4);
      if (scheduled.length !== 0) process.exit(4);
      deferred.shift()({ok: true, status: 200, json: async () => bootstrap("alpha", "chat-a", [baseTask])});
      await flush();
      if (controller.state.lastSequence !== 4 || scheduled.length !== 1) process.exit(5);
      scheduled.shift()();
      await flush();
      if (alphaCalls !== 3 || deferred.length !== 1) process.exit(6);
      deferred.shift()({ok: true, status: 200, json: async () => bootstrap("alpha", "chat-a", [freshTask])});
      await flush();
      if (controller.state.tasks[0].pending_question?.question_id !== "question-a") process.exit(7);
      emit(sockets[0], 5);
      if (scheduled.length !== 1) process.exit(8);
      controller.stop();
      scheduled.shift()();
      await flush();
      if (alphaCalls !== 3) process.exit(9);
      await controller.selectChat("alpha", "chat-a");
      void controller.refreshSelectedBootstrap();
      await flush();
      if (alphaCalls !== 5 || deferred.length !== 1) process.exit(10);
      emit(sockets[1], 1);
      deferred.shift()({ok: true, status: 200, json: async () => bootstrap("alpha", "chat-a", [baseTask])});
      await flush();
      if (scheduled.length !== 1 || controller.state.lastSequence !== 1) process.exit(11);
      scheduled.shift()();
      await flush();
      if (alphaCalls !== 6) process.exit(15);
      void controller.refreshSelectedBootstrap();
      await flush();
      if (alphaCalls !== 7 || deferred.length !== 1) process.exit(16);
      controller.stop();
      deferred.shift()({ok: true, status: 200, json: async () => bootstrap("alpha", "chat-a", [baseTask])});
      await flush();
      if (scheduled.length !== 0 || controller.state.lastSequence !== 1) process.exit(17);
      await controller.selectChat("alpha", "chat-a");
      void controller.refreshSelectedBootstrap();
      await flush();
      if (alphaCalls !== 9 || deferred.length !== 1) process.exit(12);
      const switchPromise = controller.selectProject("beta");
      deferred.shift()({ok: true, status: 200, json: async () => bootstrap("alpha", "chat-a", [freshTask])});
      await switchPromise;
      await flush();
      if (controller.state.projectId !== "beta" || controller.state.sessionId !== "chat-b" || scheduled.length !== 0) process.exit(13);
    """
    _run_module_harness(harness)


def test_action_requests_send_csrf_json_without_agent_text_in_urls() -> None:
    unsafe = "agent says /?key=secret#fragment <script>"
    harness = f"""
      const calls = [];
      const fakeFetch = async (url, options) => {{
        calls.push({{url, options}});
        return {{ok: true, status: 202, json: async () => ({{}})}};
      }};
      await bridge.postJson(
        fakeFetch,
        bridge.taskActionPath("task-1", "approve"),
        {{revision: 3}},
        "csrf-token",
      );
      await bridge.postJson(
        fakeFetch,
        bridge.sessionMessagePath("session-1"),
        {{text: {json.dumps(unsafe)}}},
        "csrf-token",
      );
      await bridge.postJson(
        fakeFetch,
        bridge.taskActionPath("task-1", "answer"),
        {{answer: {json.dumps(unsafe)}}},
        "csrf-token",
      );
      const brief = {{
        task_id: "task-1", revision: 3, title: "Title", objective: "Objective",
        context: [], constraints: [], allowed_paths: ["tools"], out_of_scope: [],
        acceptance_criteria: ["works"], required_tests: ["test.py"], risks: [],
        open_questions: [], confidence: 1, confidence_rationale: "clear",
      }};
      const edited = bridge.editedRevision(brief, {{
        title: {json.dumps(unsafe)}, objective: "Objective", context: "", constraints: "",
        allowed_paths: "tools", out_of_scope: "", acceptance_criteria: "works",
        required_tests: "test.py", risks: "", open_questions: "", confidence: "1",
        confidence_rationale: "clear",
      }});
      await bridge.postJson(
        fakeFetch,
        bridge.taskActionPath("task-1", "edit"),
        edited,
        "csrf-token",
      );
      const call = calls[0];
      if (call.url !== "/api/tasks/task-1/approve") process.exit(2);
      if (calls.some((item) => item.url.includes("agent says") || item.url.includes("secret"))) process.exit(3);
      if (call.options.method !== "POST") process.exit(4);
      if (call.options.headers["X-CSRF-Token"] !== "csrf-token") process.exit(5);
      if (call.options.headers["Content-Type"] !== "application/json") process.exit(6);
      if (call.options.body !== '{{"revision":3}}') process.exit(7);
      if (JSON.parse(calls[1].options.body).text !== {json.dumps(unsafe)}) process.exit(9);
      if (JSON.parse(calls[2].options.body).answer !== {json.dumps(unsafe)}) process.exit(10);
      if (JSON.parse(calls[3].options.body).title !== {json.dumps(unsafe)}) process.exit(11);
      let rejected = false;
      try {{ bridge.taskActionPath("task/../../escape", "approve"); }} catch {{ rejected = true; }}
      if (!rejected) process.exit(8);
    """
    _run_module_harness(harness)


def test_event_stream_replays_from_cursor_deduplicates_and_rebootstraps() -> None:
    harness = """
      if (JSON.stringify([0, 1, 2, 3, 4, 99].map(bridge.reconnectDelay)) !==
          JSON.stringify([500, 1000, 2000, 5000, 10000, 10000])) process.exit(2);
      if (bridge.websocketPath("session-1", 12) !==
          "/ws?session_id=session-1&after=12") process.exit(3);
      if (bridge.websocketUrl("session-1", 12, {protocol: "https:", host: "bridge.test"}) !==
          "wss://bridge.test/ws?session_id=session-1&after=12") process.exit(12);

      const sockets = [];
      class FakeSocket {
        constructor(url) { this.url = url; this.listeners = {}; sockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        emit(kind, value = {}) { if (kind === "close") this.closed = true; return this.listeners[kind]?.(value); }
        close() { this.closed = true; }
      }
      const scheduled = [];
      const cancelled = [];
      const received = [];
      let bootstrapCount = 0;
      const stream = bridge.createEventStream({
        sessionId: "session-1",
        initialSequence: 4,
        WebSocketCtor: FakeSocket,
        schedule(callback, delay) { scheduled.push({callback, delay}); return scheduled.length; },
        cancelSchedule(timer) { cancelled.push(timer); },
        location: {protocol: "http:", host: "127.0.0.1:9000"},
        bootstrap: async () => { bootstrapCount += 1; },
        onEvent(event) { received.push(event.sequence); },
        onStatus() {},
      });
      stream.connect();
      stream.connect();
      if (sockets.length !== 1) process.exit(13);
      if (sockets[0].url !== "ws://127.0.0.1:9000/ws?session_id=session-1&after=4") process.exit(4);
      sockets[0].emit("message", {data: JSON.stringify({sequence: 5, payload: {text: "new"}})});
      sockets[0].emit("message", {data: JSON.stringify({sequence: 5, payload: {text: "duplicate"}})});
      sockets[0].emit("message", {data: JSON.stringify({sequence: 3, payload: {text: "old"}})});
      if (JSON.stringify(received) !== "[5]" || stream.lastSequence !== 5) process.exit(5);
      sockets[0].emit("close");
      sockets[0].emit("close");
      if (scheduled.length !== 1 || scheduled[0].delay !== 500) process.exit(6);
      scheduled[0].callback();
      if (sockets[1].url !== "ws://127.0.0.1:9000/ws?session_id=session-1&after=5") process.exit(7);
      await sockets[1].emit("open");
      await Promise.resolve();
      if (bootstrapCount !== 1) process.exit(8);
      sockets[1].emit("close");
      if (scheduled[1].delay !== 500) process.exit(9);
      stream.stop();
      if (!sockets[1].closed) process.exit(10);
      if (cancelled.length !== 1) process.exit(14);
      const count = scheduled.length;
      sockets[1].emit("close");
      if (scheduled.length !== count) process.exit(11);
      await sockets[1].emit("open");
      if (bootstrapCount !== 1) process.exit(17);
      if (!bridge.acceptSequence(Number.MAX_SAFE_INTEGER - 1, {sequence: Number.MAX_SAFE_INTEGER}).accepted) process.exit(15);
      if (bridge.acceptSequence(Number.MAX_SAFE_INTEGER, {sequence: Number.MAX_SAFE_INTEGER + 1}).accepted) process.exit(16);

      const raceSockets = [];
      const raceScheduled = [];
      let releaseBootstrap;
      let staleBootstrapApplied = null;
      let raceBootstrapCount = 0;
      const raceStatuses = [];
      class RaceSocket {
        constructor(url) { this.url = url; this.listeners = {}; raceSockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        emit(kind, value = {}) { return this.listeners[kind]?.(value); }
        close() {}
      }
      const raceStream = bridge.createEventStream({
        sessionId: "session-1", initialSequence: 8, WebSocketCtor: RaceSocket,
        schedule(callback, delay) { raceScheduled.push({callback, delay}); return raceScheduled.length; },
        cancelSchedule() {}, location: {protocol: "http:", host: "127.0.0.1:9000"},
        bootstrap: async (isCurrent) => {
          raceBootstrapCount += 1;
          if (raceBootstrapCount === 1) {
            await new Promise((resolve) => { releaseBootstrap = resolve; });
          }
          staleBootstrapApplied = isCurrent();
          return staleBootstrapApplied;
        },
        onEvent() {}, onStatus(status) { raceStatuses.push(status); },
      });
      raceStream.connect();
      raceSockets[0].emit("close");
      raceScheduled[0].callback();
      const reconnecting = raceSockets[1].emit("open");
      await Promise.resolve();
      raceSockets[1].emit("message", {data: JSON.stringify({sequence: 9, payload: {text: "newer"}})});
      releaseBootstrap();
      await reconnecting;
      if (staleBootstrapApplied !== false || raceStream.lastSequence !== 9) process.exit(18);
      if (!raceStatuses.includes("bootstrap_stale")) process.exit(19);
      if (raceScheduled.length !== 2 || raceScheduled[1].delay !== 500) process.exit(20);
      raceScheduled[1].callback();
      await Promise.resolve();
      await Promise.resolve();
      if (raceBootstrapCount !== 2 || raceStatuses.at(-1) !== "connected") process.exit(21);

      const overlapSockets = [];
      const overlapScheduled = [];
      const pendingRefreshes = [];
      const overlapApplied = [];
      const overlapStatuses = [];
      class OverlapSocket {
        constructor(url) { this.url = url; this.listeners = {}; overlapSockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        emit(kind, value = {}) { return this.listeners[kind]?.(value); }
        close() {}
      }
      let refreshId = 0;
      const overlapStream = bridge.createEventStream({
        sessionId: "session-1", initialSequence: 20, WebSocketCtor: OverlapSocket,
        schedule(callback, delay) { overlapScheduled.push({callback, delay}); return overlapScheduled.length; },
        cancelSchedule() {}, location: {protocol: "http:", host: "127.0.0.1:9000"},
        bootstrap: (isCurrent) => {
          const id = ++refreshId;
          return new Promise((resolve) => pendingRefreshes.push(() => {
            const current = isCurrent();
            if (current) overlapApplied.push(id);
            resolve(current);
          }));
        },
        onEvent() {}, onStatus(status) { overlapStatuses.push(status); },
      });
      overlapStream.connect();
      overlapSockets[0].emit("close");
      overlapScheduled[0].callback();
      const olderResponse = overlapSockets[1].emit("open");
      const newerResponse = overlapSockets[1].emit("open");
      await Promise.resolve();
      pendingRefreshes[1]();
      await newerResponse;
      pendingRefreshes[0]();
      await olderResponse;
      if (JSON.stringify(overlapApplied) !== "[2]") process.exit(22);
      if (overlapScheduled.length !== 1 || overlapStatuses.at(-1) !== "connected") process.exit(23);

      const errorSockets = [];
      const errorScheduled = [];
      const errorStatuses = [];
      let errorBootstrapCount = 0;
      class ErrorSocket {
        constructor(url) { this.url = url; this.listeners = {}; errorSockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        emit(kind, value = {}) { return this.listeners[kind]?.(value); }
        close() {}
      }
      const errorStream = bridge.createEventStream({
        sessionId: "session-1", initialSequence: 30, WebSocketCtor: ErrorSocket,
        schedule(callback, delay) { errorScheduled.push({callback, delay}); return errorScheduled.length; },
        cancelSchedule() {}, location: {protocol: "http:", host: "127.0.0.1:9000"},
        bootstrap: async () => {
          errorBootstrapCount += 1;
          if (errorBootstrapCount === 1) throw new Error("refresh failed");
          return true;
        },
        onEvent() {}, onStatus(status) { errorStatuses.push(status); },
      });
      errorStream.connect();
      errorSockets[0].emit("close");
      errorScheduled[0].callback();
      await errorSockets[1].emit("open");
      if (errorStatuses.at(-1) !== "bootstrap_error"
          || errorScheduled.length !== 2 || errorScheduled[1].delay !== 500) process.exit(24);
      errorScheduled[1].callback();
      await Promise.resolve();
      await Promise.resolve();
      if (errorBootstrapCount !== 2 || errorStatuses.at(-1) !== "connected") process.exit(25);
    """
    _run_module_harness(harness)


def test_static_assets_avoid_executable_html_sinks_and_define_responsive_grid() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    forbidden = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(")
    assert all(token not in script for token in forbidden)
    assert re.search(r"grid-template-columns\s*:\s*18rem\s+minmax\(0,\s*1fr\)\s+22rem", styles)
    assert re.search(r"@media\s*\(max-width:\s*899px\)", styles)
    assert "#task-list.drawer-open" in styles
    assert "#task-inspector.drawer-open" in styles
    assert ".message-user" in styles
    assert ".message-fable" in styles
    assert ".message-sol" in styles
    assert ".message-coordinator" in styles


def test_project_chat_navigation_markup_is_semantic_and_never_exposes_paths() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    rendered = _RenderedLayout()
    rendered.feed(html)

    assert {
        "project-navigation",
        "project-list",
        "chat-list",
        "new-chat",
        "selected-project-name",
        "selected-chat-name",
    } <= rendered.ids
    assert rendered.element("nav", "project-navigation")["aria-label"] == "Projects and chats"
    assert rendered.element("ul", "project-list")["aria-label"] == "Projects"
    assert rendered.element("ul", "chat-list")["aria-label"] == "Chats"
    assert rendered.element("button", "new-chat")["type"] == "button"
    assert rendered.element("button", "new-chat")["disabled"] is None
    assert rendered.element("main", "conversation-shell")["id"] == "conversation-shell"
    assert rendered.element("aside", "task-inspector-panel")["id"] == "task-inspector-panel"
    assert "Fable" in " ".join(rendered.text)
    assert "Sol" in " ".join(rendered.text)
    assert not any(
        tag == "input" and "path" in " ".join(
            value or "" for value in attrs.values()
        ).lower()
        for tag, attrs in rendered.elements
    )


def test_project_chat_controller_scopes_selection_routes_and_stream_lifecycle() -> None:
    harness = r"""
      const calls = [];
      const replies = new Map([
        ["GET /api/projects", [
          {csrf_token: "csrf", projects: [
            {project_id: "project-a", label: "Alpha", branch: "main", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
            {project_id: "project-b", label: "Beta", branch: "release", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
          ], active_lease: null},
          {csrf_token: "csrf", projects: [
            {project_id: "project-a", label: "Alpha", branch: "main", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
            {project_id: "project-b", label: "Beta", branch: "release", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
          ], active_lease: null},
        ]],
        ["GET /api/projects/project-a/chats?limit=50", [
          {chats: [
            {session_id: "shared-chat", title: "Shared Alpha", latest_sequence: 0},
            {session_id: "old-chat", title: "Old Alpha", latest_sequence: 0},
          ]},
        ]],
        ["GET /api/projects/project-a/chats/shared-chat/bootstrap", [
          {csrf_token: "csrf", usage_credits_acknowledged: true, project_id: "project-a", session_id: "shared-chat", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "main", replay_after: 2, tasks: []},
        ]],
        ["GET /api/projects/project-a/chats/old-chat/bootstrap", [
          {csrf_token: "csrf", usage_credits_acknowledged: true, project_id: "project-a", session_id: "old-chat", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "main", replay_after: 0, tasks: []},
        ]],
        ["POST /api/projects/project-a/chats", [
          {session_id: "random-chat-99", title: "New chat", latest_sequence: 0},
        ]],
        ["GET /api/projects/project-a/chats/random-chat-99/bootstrap", [
          {csrf_token: "csrf", usage_credits_acknowledged: true, project_id: "project-a", session_id: "random-chat-99", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "main", replay_after: 0, tasks: []},
        ]],
        ["GET /api/projects/project-b/chats?limit=50", [
          {chats: [{session_id: "shared-chat", title: "Shared Beta", latest_sequence: 0}]},
        ]],
        ["GET /api/projects/project-b/chats/shared-chat/bootstrap", [
          {csrf_token: "csrf", usage_credits_acknowledged: true, project_id: "project-b", session_id: "shared-chat", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "release", replay_after: 0, tasks: []},
          {csrf_token: "csrf", usage_credits_acknowledged: true, project_id: "project-b", session_id: "shared-chat", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "release", replay_after: 7, tasks: []},
        ]],
      ]);
      const fetchFunction = async (url, options = {}) => {
        const key = `${options.method ?? "GET"} ${url}`;
        calls.push({url, options});
        const reply = replies.get(key)?.shift();
        if (!reply) throw new Error(`unexpected ${key}`);
        return {ok: true, status: options.method === "POST" ? 201 : 200, json: async () => reply};
      };
      const sockets = [];
      class Socket {
        constructor(url) { this.url = url; this.listeners = {}; sockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        emit(kind, value = {}) { return this.listeners[kind]?.(value); }
        close() { this.closed = true; }
      }
      const timers = [];
      const cancelled = [];
      const states = [];
      const chatResets = [];
      const controller = bridge.createProjectChatController({
        fetchFunction,
        WebSocketCtor: Socket,
        schedule(callback, delay) { timers.push({callback, delay}); return timers.length; },
        cancelSchedule(timer) { cancelled.push(timer); },
        location: {protocol: "https:", host: "bridge.test"},
        onState(state) { states.push(state); },
        onEvent() {},
        onStatus() {},
        onChatReset() { chatResets.push("reset"); },
      });

      await controller.bootstrapInitial();
      if (controller.state.projectId !== "project-a" || controller.state.sessionId !== "shared-chat") process.exit(2);
      if (sockets.length !== 1 || !sockets[0].url.includes("project_id=project-a") || !sockets[0].url.includes("session_id=shared-chat") || !sockets[0].url.endsWith("after=2")) process.exit(3);

      await controller.selectChat("project-a", "old-chat");
      if (!sockets[0].closed || controller.state.sessionId !== "old-chat" || controller.state.lastSequence !== 0) process.exit(4);

      await controller.createChat();
      if (controller.state.sessionId !== "random-chat-99" || !calls.some((call) => call.url === "/api/projects/project-a/chats" && call.options.headers["X-CSRF-Token"] === "csrf")) process.exit(5);
      if (sockets.filter((socket) => socket.closed).length !== 2) process.exit(6);

      await controller.selectProject("project-b");
      if (controller.state.projectId !== "project-b" || controller.state.sessionId !== "shared-chat") process.exit(7);
      if (!sockets[2].closed || !sockets[3].url.includes("project_id=project-b") || !sockets[3].url.includes("session_id=shared-chat")) process.exit(8);
      if (chatResets.length !== 3) process.exit(20);

      sockets[3].emit("message", {data: JSON.stringify({sequence: 7, task_id: null, payload: {text: "safe"}})});
      sockets[3].emit("close");
      if (timers.length !== 1 || timers[0].delay !== 500) process.exit(9);
      timers[0].callback();
      if (sockets.length !== 5 || !sockets[4].url.endsWith("after=7")) process.exit(10);
      await sockets[4].emit("open");
      if (controller.state.projectId !== "project-b" || controller.state.sessionId !== "shared-chat" || controller.state.lastSequence !== 7) process.exit(11);
      if (cancelled.length !== 0 || states.some((state) => state.projectId === "project-a" && state.sessionId === "shared-chat" && state.lastSequence === 7)) process.exit(12);

      if (bridge.projectChatMessagePath("project:a", "chat.2") !== "/api/projects/project%3Aa/chats/chat.2/messages") process.exit(13);
      if (bridge.projectTaskActionPath("project-a", "chat-a", "task-a", "approve") !== "/api/projects/project-a/chats/chat-a/tasks/task-a/approve") process.exit(14);
      if (bridge.projectWebsocketPath("project-a", "chat-a", 9) !== "/ws?project_id=project-a&session_id=chat-a&after=9") process.exit(15);
      let unsafeRejected = false;
      try { bridge.projectChatBootstrapPath("project-a", "chat/escape"); } catch { unsafeRejected = true; }
      if (!unsafeRejected) process.exit(16);

      class Node {
        constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.attributes = {}; this.disabled = false; this._text = ""; }
        set textContent(value) { this._text = String(value); this.children = []; }
        get textContent() { return this._text; }
        append(...children) { this.children.push(...children); }
        replaceChildren(...children) { this.children = [...children]; this._text = ""; }
        setAttribute(name, value) { this.attributes[name] = String(value); }
        addEventListener() {}
      }
      const projectList = new Node("ul");
      const chatList = new Node("ul");
      const newChat = new Node("button");
      const selectedProject = new Node("strong");
      const selectedChat = new Node("strong");
      const roots = {
        "#project-list": projectList, "#chat-list": chatList, "#new-chat": newChat,
        "#selected-project-name": selectedProject, "#selected-chat-name": selectedChat,
      };
      const documentRoot = {
        createElement(tag) { return new Node(tag); },
        querySelector(selector) { return roots[selector] ?? null; },
      };
      bridge.renderProjectNavigation(documentRoot, {
        projectId: "project-0", projectLabel: "<script>Alpha</script>", sessionId: "chat-0",
        projects: Array.from({length: bridge.MAX_NAV_PROJECTS + 5}, (_, index) => ({projectId: `project-${index}`, label: index === 0 ? "<script>Alpha</script>" : `Project ${index}`})),
        chats: Array.from({length: bridge.MAX_NAV_CHATS + 5}, (_, index) => ({sessionId: `chat-${index}`, title: index === 0 ? "<img>" : `Chat ${index}`})),
        activeLease: {projectId: "project-0", sessionId: "chat-0", taskId: "task-0"},
        csrfToken: "csrf",
      }, {onProject() {}, onChat() {}, onNewChat() {}});
      if (projectList.children.length !== bridge.MAX_NAV_PROJECTS || chatList.children.length !== bridge.MAX_NAV_CHATS) process.exit(17);
      if (!newChat.disabled || projectList.children.some((item) => !item.children[0].disabled) || chatList.children.some((item) => !item.children[0].disabled)) process.exit(18);
      if (selectedProject.textContent !== "<script>Alpha</script>" || selectedChat.textContent !== "<img>") process.exit(19);
    """
    _run_module_harness(harness)


def test_project_chat_controller_keeps_empty_chat_csrf_selection_refresh_and_focus_safe() -> None:
    harness = r"""
      const payloads = {
        empty: {
          csrf_token: "csrf-empty",
          projects: [
            {project_id: "empty", label: "Empty", branch: "main", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
            {project_id: "other", label: "Other", branch: "next", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
          ],
          active_lease: null,
        },
        leased: {
          csrf_token: "csrf-empty",
          projects: [
            {project_id: "empty", label: "Empty", branch: "main", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
            {project_id: "other", label: "Other", branch: "next", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
          ],
          active_lease: {project_id: "other", session_id: "other-chat", task_id: "task-1"},
        },
      };
      const projectRequests = [];
      const deferredCreates = [];
      const flush = async () => { for (let tick = 0; tick < 12; tick += 1) await Promise.resolve(); };
      const fetchFunction = (url, options = {}) => {
        if (url === "/api/projects") {
          return new Promise((resolve) => projectRequests.push(resolve));
        }
        if (url === "/api/projects/empty/chats?limit=50") {
          return Promise.resolve({ok: true, status: 200, json: async () => ({chats: []})});
        }
        if (url === "/api/projects/other/chats?limit=50") {
          return Promise.resolve({ok: true, status: 200, json: async () => ({chats: [{session_id: "other-chat", title: "Other chat", latest_sequence: 0}]})});
        }
        if (url === "/api/projects/other/chats/other-chat/bootstrap") {
          return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf-empty", usage_credits_acknowledged: true, project_id: "other", session_id: "other-chat", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "next", replay_after: 0, tasks: []})});
        }
        if (url === "/api/projects/empty/chats" && options.method === "POST") {
          return new Promise((resolve) => deferredCreates.push(resolve));
        }
        throw new Error(`unexpected ${options.method ?? "GET"} ${url}`);
      };
      const sockets = [];
      class Socket {
        constructor(url) { this.url = url; this.listeners = {}; sockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        close() { this.closed = true; }
      }
      const controller = bridge.createProjectChatController({
        fetchFunction, WebSocketCtor: Socket, schedule() { return 1; }, cancelSchedule() {},
        location: {protocol: "http:", host: "bridge.test"}, onState() {}, onEvent() {}, onStatus() {},
      });
      const initial = controller.bootstrapInitial();
      if (projectRequests.length !== 1) process.exit(2);
      projectRequests.shift()({ok: true, status: 200, json: async () => payloads.empty});
      await initial;
      if (controller.state.projectId !== "empty" || controller.state.sessionId !== null || controller.state.csrfToken !== "csrf-empty") process.exit(3);

      const create = controller.createChat();
      if (deferredCreates.length !== 1) process.exit(4);
      const switchProject = controller.selectProject("other");
      await switchProject;
      deferredCreates.shift()({ok: true, status: 201, json: async () => ({session_id: "late-empty", title: "Late empty", latest_sequence: 0})});
      await create;
      if (controller.state.projectId !== "other" || controller.state.sessionId !== "other-chat" || sockets.length !== 1 || !sockets[0].url.includes("project_id=other")) process.exit(5);

      const refreshOne = controller.refreshProjects();
      const refreshTwo = controller.refreshProjects();
      if (projectRequests.length !== 1) process.exit(6);
      projectRequests.shift()({ok: true, status: 200, json: async () => payloads.leased});
      await flush();
      if (controller.state.activeLease !== null || projectRequests.length !== 1) process.exit(7);
      projectRequests.shift()({ok: true, status: 200, json: async () => payloads.empty});
      await Promise.all([refreshOne, refreshTwo]);
      if (controller.state.activeLease !== null || controller.state.navigationLocked !== false) process.exit(8);

      const malformed = controller.refreshProjects();
      projectRequests.shift()({ok: true, status: 200, json: async () => ({csrf_token: "csrf-empty", projects: payloads.empty.projects, active_lease: {project_id: "not a safe id"}})});
      await malformed;
      if (controller.state.navigationLocked !== true || controller.state.activeLease !== null) process.exit(9);

      const missingLease = controller.refreshProjects();
      projectRequests.shift()({ok: true, status: 200, json: async () => ({csrf_token: "csrf-empty", projects: payloads.empty.projects})});
      await missingLease;
      if (controller.state.navigationLocked !== true || controller.state.activeLease !== null) process.exit(19);

      const malformedCsrf = controller.refreshProjects();
      projectRequests.shift()({ok: true, status: 200, json: async () => ({csrf_token: "bad\ncsrf", projects: payloads.empty.projects, active_lease: null})});
      await malformedCsrf;
      if (controller.state.navigationLocked !== true || controller.state.csrfToken !== "") process.exit(27);

      const recovered = controller.refreshProjects();
      projectRequests.shift()({ok: true, status: 200, json: async () => payloads.empty});
      await recovered;
      if (controller.state.navigationLocked !== false || controller.state.activeLease !== null) process.exit(20);

      const oldReleased = controller.refreshProjects();
      const newerLeased = controller.refreshProjects();
      if (projectRequests.length !== 1) process.exit(21);
      projectRequests.shift()({ok: true, status: 200, json: async () => payloads.empty});
      await flush();
      if (controller.state.activeLease !== null || projectRequests.length !== 1) process.exit(22);
      projectRequests.shift()({ok: true, status: 200, json: async () => payloads.leased});
      await Promise.all([oldReleased, newerLeased]);
      if (controller.state.activeLease?.projectId !== "other" || controller.state.navigationLocked !== false) process.exit(23);

      const malformedLiveLease = controller.refreshProjects();
      projectRequests.shift()({ok: true, status: 200, json: async () => ({csrf_token: "csrf-empty", projects: payloads.empty.projects, active_lease: {project_id: "not a safe id"}})});
      await malformedLiveLease;
      if (controller.state.navigationLocked !== true || controller.state.activeLease?.projectId !== "other") process.exit(28);

      const failedRefresh = controller.refreshProjects();
      projectRequests.shift()({ok: false, status: 503, json: async () => ({})});
      await failedRefresh;
      if (controller.state.navigationLocked !== true || controller.state.activeLease?.projectId !== "other") process.exit(24);
      const recoveredLease = controller.refreshProjects();
      projectRequests.shift()({ok: true, status: 200, json: async () => payloads.empty});
      await recoveredLease;
      if (controller.state.navigationLocked !== false || controller.state.activeLease !== null) process.exit(25);

      const sameCreates = [];
      const sameController = bridge.createProjectChatController({
        fetchFunction(url, options = {}) {
          if (url === "/api/projects") return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf-same", projects: [{project_id: "same", label: "Same", branch: "main", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}}], active_lease: null})});
          if (url === "/api/projects/same/chats?limit=50") return Promise.resolve({ok: true, status: 200, json: async () => ({chats: [{session_id: "old", title: "Old", latest_sequence: 0}, {session_id: "newer", title: "Newer", latest_sequence: 0}]})});
          if (url === "/api/projects/same/chats/old/bootstrap" || url === "/api/projects/same/chats/newer/bootstrap") {
            const sessionId = url.includes("/old/") ? "old" : "newer";
            return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf-same", usage_credits_acknowledged: true, project_id: "same", session_id: sessionId, fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "main", replay_after: 0, tasks: []})});
          }
          if (url === "/api/projects/same/chats" && options.method === "POST") return new Promise((resolve) => sameCreates.push(resolve));
          throw new Error(`unexpected ${options.method ?? "GET"} ${url}`);
        },
        WebSocketCtor: Socket, schedule() { return 1; }, cancelSchedule() {}, location: {protocol: "http:", host: "bridge.test"}, onState() {}, onEvent() {}, onStatus() {},
      });
      await sameController.bootstrapInitial();
      const sameCreate = sameController.createChat();
      await sameController.selectChat("same", "newer");
      sameCreates.shift()({ok: true, status: 201, json: async () => ({session_id: "late-same", title: "Late same", latest_sequence: 0})});
      await sameCreate;
      if (sameController.state.projectId !== "same" || sameController.state.sessionId !== "newer" || sameController.state.lastSequence !== 0) process.exit(26);

      class Node {
        constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.attributes = {}; this.disabled = false; this._text = ""; }
        set textContent(value) { this._text = String(value); this.children = []; }
        get textContent() { return this._text; }
        append(...children) { this.children.push(...children); }
        replaceChildren(...children) { this.children = [...children]; this._text = ""; }
        setAttribute(name, value) { this.attributes[name] = String(value); }
        addEventListener() {}
      }
      const projectList = new Node("ul");
      const chatList = new Node("ul");
      const newChat = new Node("button");
      const projectName = new Node("strong");
      const chatName = new Node("strong");
      const roots = {"#project-list": projectList, "#chat-list": chatList, "#new-chat": newChat, "#selected-project-name": projectName, "#selected-chat-name": chatName};
      const documentRoot = {createElement(tag) { return new Node(tag); }, querySelector(selector) { return roots[selector] ?? null; }};
      const navigationState = {projectId: "empty", projectLabel: "Empty", sessionId: null, csrfToken: "csrf-empty", activeLease: null, navigationLocked: false, navigationPending: false, projects: [{projectId: "empty", label: "Empty"}], chats: []};
      bridge.renderProjectNavigation(documentRoot, navigationState, {});
      const focusedButton = projectList.children[0].children[0];
      bridge.renderProjectNavigation(documentRoot, {...navigationState}, {});
      if (projectList.children[0].children[0] !== focusedButton || newChat.disabled) process.exit(10);
      bridge.renderProjectNavigation(documentRoot, {...navigationState, navigationLocked: true}, {});
      if (!newChat.disabled || !projectList.children[0].children[0].disabled) process.exit(11);
    """
    _run_module_harness(harness)


def test_project_chat_controller_replaces_invalidated_lease_refresh_before_unlocking() -> None:
    harness = r"""
      const released = {
        csrf_token: "csrf-refresh",
        usage_credits_acknowledged: true,
        projects: [
          {project_id: "project-a", label: "Alpha", branch: "main", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
          {project_id: "project-b", label: "Beta", branch: "next", readiness: {fable_ready: true, fable_status: "subscription_ready", sol_status: "ready"}},
        ],
        active_lease: null,
      };
      const leased = {
        ...released,
        active_lease: {project_id: "project-b", session_id: "chat-b", task_id: "task-b"},
      };
      const projectRequests = [];
      let projectFetches = 0;
      const flush = async () => { for (let tick = 0; tick < 12; tick += 1) await Promise.resolve(); };
      const fetchFunction = (url) => {
        if (url === "/api/projects") {
          projectFetches += 1;
          if (projectFetches === 1) return Promise.resolve({ok: true, status: 200, json: async () => released});
          return new Promise((resolve) => projectRequests.push(resolve));
        }
        if (url === "/api/projects/project-a/chats?limit=50") return Promise.resolve({ok: true, status: 200, json: async () => ({chats: [{session_id: "chat-a", title: "Alpha chat", latest_sequence: 0}]})});
        if (url === "/api/projects/project-b/chats?limit=50") return Promise.resolve({ok: true, status: 200, json: async () => ({chats: [{session_id: "chat-b", title: "Beta chat", latest_sequence: 0}]})});
        if (url === "/api/projects/project-a/chats/chat-a/bootstrap") return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf-refresh", usage_credits_acknowledged: true, project_id: "project-a", session_id: "chat-a", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "main", replay_after: 0, tasks: []})});
        if (url === "/api/projects/project-b/chats/chat-b/bootstrap") return Promise.resolve({ok: true, status: 200, json: async () => ({csrf_token: "csrf-refresh", usage_credits_acknowledged: true, project_id: "project-b", session_id: "chat-b", fable_ready: true, fable_status: "subscription_ready", sol_status: "ready", branch: "next", replay_after: 0, tasks: []})});
        throw new Error(`unexpected GET ${url}`);
      };
      const sockets = [];
      class Socket {
        constructor(url) { this.url = url; this.listeners = {}; sockets.push(this); }
        addEventListener(kind, listener) { this.listeners[kind] = listener; }
        close() { this.closed = true; }
      }
      const controller = bridge.createProjectChatController({
        fetchFunction, WebSocketCtor: Socket, schedule() { return 1; }, cancelSchedule() {},
        location: {protocol: "http:", host: "bridge.test"}, onState() {}, onEvent() {}, onStatus() {},
      });
      await controller.bootstrapInitial();
      if (controller.state.projectId !== "project-a" || controller.state.sessionId !== "chat-a" || controller.state.navigationPending) process.exit(2);

      const staleReleased = controller.refreshProjects();
      if (projectRequests.length !== 1) process.exit(3);
      await controller.selectProject("project-b");
      if (controller.state.projectId !== "project-b" || controller.state.sessionId !== "chat-b" || !controller.state.navigationPending) process.exit(4);
      projectRequests.shift()({ok: true, status: 200, json: async () => released});
      await flush();
      if (projectRequests.length !== 1 || controller.state.activeLease !== null || !controller.state.navigationPending) process.exit(5);
      projectRequests.shift()({ok: true, status: 200, json: async () => leased});
      await staleReleased;
      if (controller.state.activeLease?.projectId !== "project-b" || controller.state.navigationPending || controller.state.navigationLocked) process.exit(6);

      const staleFailure = controller.refreshProjects();
      if (projectRequests.length !== 1) process.exit(7);
      await controller.selectProject("project-a");
      if (!controller.state.navigationPending) process.exit(8);
      projectRequests.shift()({ok: true, status: 200, json: async () => leased});
      await flush();
      if (projectRequests.length !== 1 || !controller.state.navigationPending) process.exit(9);
      projectRequests.shift()({ok: false, status: 503, json: async () => ({})});
      await staleFailure;
      if (!controller.state.navigationLocked || controller.state.navigationPending) process.exit(10);

      const validRelease = controller.refreshProjects();
      projectRequests.shift()({ok: true, status: 200, json: async () => released});
      await validRelease;
      if (controller.state.navigationLocked || controller.state.activeLease !== null || controller.state.navigationPending) process.exit(11);

      const staleMalformed = controller.refreshProjects();
      if (projectRequests.length !== 1) process.exit(12);
      await controller.selectProject("project-b");
      projectRequests.shift()({ok: true, status: 200, json: async () => released});
      await flush();
      if (projectRequests.length !== 1 || !controller.state.navigationPending) process.exit(13);
      projectRequests.shift()({ok: true, status: 200, json: async () => ({...released, active_lease: {project_id: "bad id"}})});
      await staleMalformed;
      if (!controller.state.navigationLocked || controller.state.navigationPending) process.exit(14);

      const finalRelease = controller.refreshProjects();
      projectRequests.shift()({ok: true, status: 200, json: async () => released});
      await finalRelease;
      const activeSockets = sockets.filter((socket) => !socket.closed);
      if (controller.state.navigationLocked || controller.state.activeLease !== null || controller.state.navigationPending || activeSockets.length !== 1 || !activeSockets[0].url.includes("project_id=project-b")) process.exit(15);
    """
    _run_module_harness(harness)
