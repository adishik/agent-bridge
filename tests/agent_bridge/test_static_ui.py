from __future__ import annotations

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path


STATIC = Path("src/agent_bridge/static")


class _RenderedLayout(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.ids: set[str] = set()
        self.text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        self.elements.append((tag, values))
        if values.get("id") is not None:
            self.ids.add(str(values["id"]))

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


def test_index_renders_semantic_option_a_layout_and_accessible_mobile_drawers() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    rendered = _RenderedLayout()
    rendered.feed(html)

    assert {
        "app-header",
        "workspace",
        "task-list",
        "conversation",
        "task-inspector",
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
    assert rendered.element("aside", "task-list")["aria-label"] == "Tasks"
    assert rendered.element("section", "conversation")["aria-live"] == "polite"
    assert (
        rendered.element("aside", "task-inspector")["aria-label"]
        == "Task inspector"
    )
    assert rendered.element("button", "task-drawer-toggle")["aria-controls"] == (
        "task-list"
    )
    assert rendered.element("button", "inspector-drawer-toggle")[
        "aria-controls"
    ] == "task-inspector"
    assert "open" not in rendered.element("dialog", "usage-modal")
    modal_markup = html[html.index('<dialog\n      id="usage-modal"'):html.index("</dialog>")]
    assert 'id="bootstrap-retry"' in modal_markup
    assert "disabled" in rendered.element("button", "composer-submit")
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
