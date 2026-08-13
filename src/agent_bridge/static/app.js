const ACTOR_LABELS = Object.freeze({
  user: "You",
  fable: "Fable",
  sol: "Sol",
  coordinator: "Coordinator",
});

const ACTIVE_STATES = new Set([
  "fable_planning",
  "sol_running",
  "fable_clarifying",
  "fable_reviewing",
  "sol_correcting",
]);

const KNOWN_STATES = new Set([
  "idle",
  "fable_planning",
  "awaiting_user_approval",
  "sol_running",
  "fable_clarifying",
  "awaiting_user_input",
  "awaiting_scope_approval",
  "fable_reviewing",
  "sol_correcting",
  "completed",
  "failed",
  "interrupted",
]);

export const MAX_CONVERSATION_MESSAGES = 300;
export const MAX_TASK_HISTORY = 100;
export const MAX_TASK_OVERVIEWS = 200;
export const MAX_NAV_PROJECTS = 100;
export const MAX_NAV_CHATS = 50;

const MESSAGE_EVENT_KINDS = new Set([
  "message",
  "task_brief",
  "clarification",
  "outcome",
  "review",
  "task_rejected",
  "action_error",
  "stop_error",
  "resume_drift",
]);

const APPROVAL_STATES = new Set([
  "awaiting_user_approval",
  "awaiting_scope_approval",
]);

const TASK_ACTIONS = new Set([
  "approve",
  "edit",
  "reject",
  "answer",
  "stop",
  "resume",
]);

const LIST_FIELDS = Object.freeze([
  "context",
  "constraints",
  "allowed_paths",
  "out_of_scope",
  "acceptance_criteria",
  "required_tests",
  "risks",
  "open_questions",
]);

const EDITABLE_FIELDS = Object.freeze([
  "title",
  "objective",
  ...LIST_FIELDS,
  "confidence",
  "confidence_rationale",
]);

const FIELD_LABELS = Object.freeze({
  objective: "Objective",
  context: "Context",
  constraints: "Constraints",
  allowed_paths: "Allowed paths",
  out_of_scope: "Out of scope",
  acceptance_criteria: "Acceptance criteria",
  required_tests: "Required tests",
  risks: "Risks",
  open_questions: "Open questions",
  confidence_rationale: "Confidence rationale",
});

const RECONNECT_DELAYS = Object.freeze([500, 1000, 2000, 5000, 10000]);
const BOOTSTRAP_REFRESH_DELAY = 500;
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const SAFE_CSRF_TOKEN = /^[\x21-\x7e]+$/;


function requireSafeId(value, name) {
  if (typeof value !== "string" || !SAFE_ID.test(value)) {
    throw new Error(`${name} must be a safe identifier`);
  }
  return value;
}


function asObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : {};
}


function actorKey(actor) {
  return Object.hasOwn(ACTOR_LABELS, actor) ? actor : "coordinator";
}


function actorLabel(actor) {
  return ACTOR_LABELS[actorKey(actor)];
}


function stateLabel(value) {
  if (typeof value !== "string" || value.length === 0) {
    return "Unknown";
  }
  return value
    .split("_")
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}


function eventText(event) {
  const payload = asObject(event?.payload);
  if (typeof payload.text === "string") {
    return payload.text;
  }
  if (event?.kind === "task_brief") {
    const brief = asObject(payload.brief);
    const title = typeof brief.title === "string" ? brief.title : "Untitled task";
    const revision = Number.isInteger(brief.revision) ? brief.revision : "?";
    return `Task brief ready: ${title} · revision ${revision}`;
  }
  if (event?.kind === "task_state" && typeof payload.state === "string") {
    return `Task state: ${stateLabel(payload.state)}`;
  }
  if (event?.kind === "task_rejected") {
    return `Task revision ${String(payload.revision ?? "unknown")} rejected.`;
  }
  if (event?.kind === "agent_event") {
    const type = typeof payload.type === "string" ? stateLabel(payload.type) : "Agent activity";
    const status = typeof payload.status === "string" ? ` · ${stateLabel(payload.status)}` : "";
    return `${type}${status}`;
  }
  if (event?.kind === "resume_drift") {
    return "Repository state checked before resume.";
  }
  if (event?.kind === "stop_error") {
    return "The exact task process could not be stopped cleanly.";
  }
  if (event?.kind === "action_error") {
    return `Action failed: ${String(payload.action ?? "unknown")} · ${String(payload.error_type ?? "unknown error")}`;
  }
  if (typeof payload.summary === "string") {
    return payload.summary;
  }
  if (typeof payload.answer === "string") {
    return payload.answer;
  }
  if (typeof payload.question_for_user === "string") {
    return payload.question_for_user;
  }
  if (typeof payload.reasoning === "string") {
    return payload.reasoning;
  }
  if (typeof payload.status === "string") {
    return `${stateLabel(String(event?.kind ?? "update"))}: ${stateLabel(payload.status)}`;
  }
  return stateLabel(String(event?.kind ?? "update"));
}


function element(documentRoot, tag, text, className = "") {
  const node = documentRoot.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = String(text);
  }
  return node;
}


function appendConversationNode(documentRoot, conversation, node) {
  const intro = documentRoot.querySelector("#conversation-empty");
  if (intro && typeof intro.remove === "function") {
    intro.remove();
  }
  conversation.append(node);
  if (conversation.children.length > MAX_CONVERSATION_MESSAGES) {
    conversation.removeChild(conversation.children[0]);
  }
  return node;
}


export function conversationPresentation(event) {
  if (event?.kind === "task_state") return "status";
  if (MESSAGE_EVENT_KINDS.has(event?.kind)) return "message";
  return "hidden";
}


export function renderMessage(documentRoot, event, associatedRevision = null) {
  const conversation = documentRoot.querySelector("#conversation");
  if (!conversation) {
    throw new Error("conversation region is missing");
  }
  const key = actorKey(event?.actor);
  const article = element(documentRoot, "article", undefined, `message message-${key}`);
  article.setAttribute("aria-label", `${actorLabel(key)} message`);
  const actor = element(documentRoot, "strong", actorLabel(key));
  const messageNode = element(documentRoot, "p", eventText(event));
  const payload = asObject(event?.payload);
  const revision = Number.isInteger(associatedRevision)
    ? associatedRevision
    : (Number.isInteger(payload.revision)
      ? payload.revision
      : (Number.isInteger(asObject(payload.brief).revision) ? asObject(payload.brief).revision : null));
  const sequence = Number.isSafeInteger(event?.sequence) ? `#${event.sequence}` : "#?";
  const taskId = typeof event?.task_id === "string" ? event.task_id : "global";
  const kind = typeof event?.kind === "string" ? event.kind : "event";
  const revisionText = revision === null ? "" : ` · r${revision}`;
  const metadata = element(
    documentRoot,
    "p",
    `${sequence} · ${kind} · ${taskId}${revisionText}`,
    "message-metadata",
  );
  article.append(actor, messageNode, metadata);
  const projection = {
    sequence: Number.isSafeInteger(event?.sequence) ? event.sequence : null,
    session_id: typeof event?.session_id === "string" ? event.session_id : null,
    task_id: typeof event?.task_id === "string" ? event.task_id : null,
    revision,
    actor: typeof event?.actor === "string" ? event.actor : null,
    kind: typeof event?.kind === "string" ? event.kind : null,
    payload,
    created_at: typeof event?.created_at === "string" ? event.created_at : null,
  };
  const details = element(documentRoot, "details");
  details.append(
    element(documentRoot, "summary", "Inspect structured event"),
    element(documentRoot, "pre", JSON.stringify(projection, null, 2)),
  );
  article.append(details);
  if (
    typeof event?.created_at === "string"
    && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(event.created_at)
    && Number.isFinite(Date.parse(event.created_at))
  ) {
    const time = element(documentRoot, "time", event.created_at);
    time.setAttribute("datetime", event.created_at);
    article.append(time);
  }
  return appendConversationNode(documentRoot, conversation, article);
}


function renderConversationStatus(documentRoot, event, associatedRevision) {
  const conversation = documentRoot.querySelector("#conversation");
  if (!conversation) {
    throw new Error("conversation region is missing");
  }
  const payload = asObject(event?.payload);
  const revision = Number.isInteger(associatedRevision)
    ? associatedRevision
    : (Number.isInteger(payload.revision) ? payload.revision : null);
  const sequence = Number.isSafeInteger(event?.sequence) ? `#${event.sequence}` : "#?";
  const taskId = typeof event?.task_id === "string" ? event.task_id : "global";
  const revisionText = revision === null ? "" : ` · r${revision}`;
  const status = element(
    documentRoot,
    "p",
    `${eventText(event)} · ${sequence} · ${taskId}${revisionText}`,
    "conversation-status",
  );
  return appendConversationNode(documentRoot, conversation, status);
}


export function renderConversationEvent(documentRoot, event, associatedRevision = null) {
  const presentation = conversationPresentation(event);
  if (presentation === "hidden") return null;
  if (presentation === "status") {
    return renderConversationStatus(documentRoot, event, associatedRevision);
  }
  return renderMessage(documentRoot, event, associatedRevision);
}


export function canonicalTaskBrief(task) {
  const candidate = asObject(task?.brief);
  if (Object.keys(candidate).length === 0) {
    return null;
  }
  if (
    typeof task?.task_id !== "string"
    || !Number.isInteger(task?.revision)
    || candidate.task_id !== task.task_id
    || candidate.revision !== task.revision
  ) {
    return null;
  }
  return candidate;
}


function taskBrief(task) {
  return canonicalTaskBrief(task);
}


function taskIdentity(task) {
  const brief = taskBrief(task);
  const taskId = typeof task?.task_id === "string" ? task.task_id : brief?.task_id;
  return typeof taskId === "string" ? taskId : "unknown-task";
}


function taskRevision(task) {
  const brief = taskBrief(task);
  if (Number.isInteger(task?.revision)) {
    return task.revision;
  }
  return Number.isInteger(brief?.revision) ? brief.revision : 0;
}


function taskTitle(task) {
  const brief = taskBrief(task);
  return typeof brief?.title === "string" && brief.title
    ? brief.title
    : "Planning task";
}


function taskRecency(task) {
  return typeof task?.updated_at === "string" && task.updated_at
    ? task.updated_at
    : "No activity yet";
}


export function elapsedLabel(startedAt, nowMillis = Date.now()) {
  if (
    typeof startedAt !== "string"
    || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(startedAt)
    || !Number.isFinite(Date.parse(startedAt))
    || !Number.isFinite(nowMillis)
  ) {
    return "elapsed unavailable";
  }
  const seconds = Math.floor(Math.max(0, nowMillis - Date.parse(startedAt)) / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes === 0 ? `${hours}h` : `${hours}h ${remainingMinutes}m`;
}


function stateClass(state) {
  return KNOWN_STATES.has(state) ? `state-${state}` : "state-unknown";
}


export function renderTaskList(documentRoot, tasks, selectedTaskId, onSelect) {
  const container = documentRoot.querySelector("#task-list");
  if (!container) {
    throw new Error("task list region is missing");
  }
  const heading = element(documentRoot, "div", undefined, "panel-heading");
  const labelGroup = element(documentRoot, "div");
  labelGroup.append(
    element(documentRoot, "p", "Work queue", "eyebrow"),
    element(documentRoot, "h2", "Tasks"),
  );
  heading.append(labelGroup);

  const safeTasks = Array.isArray(tasks) ? tasks : [];
  if (safeTasks.length === 0) {
    container.replaceChildren(
      heading,
      element(
        documentRoot,
        "p",
        "Tasks will appear after Fable prepares a brief.",
        "empty-state",
      ),
    );
    return container;
  }

  const list = element(documentRoot, "ul", undefined, "task-list-items");
  for (const task of safeTasks) {
    const taskId = taskIdentity(task);
    const item = element(documentRoot, "li");
    const button = element(documentRoot, "button", undefined, "task-list-button");
    button.type = "button";
    button.dataset.taskId = taskId;
    button.setAttribute("aria-current", taskId === selectedTaskId ? "true" : "false");
    button.append(
      element(documentRoot, "span", taskTitle(task), "task-list-title"),
      element(
        documentRoot,
        "span",
        `${stateLabel(String(task?.state ?? "unknown"))} · r${taskRevision(task)} · ${taskRecency(task)}`,
        "task-meta",
      ),
    );
    button.addEventListener("click", () => onSelect(taskId));
    item.append(button);
    list.append(item);
  }
  container.replaceChildren(heading, list);
  return container;
}


function appendTextSection(documentRoot, parent, headingText, value) {
  const section = element(documentRoot, "section", undefined, "task-section");
  section.append(element(documentRoot, "h3", headingText));
  const content = typeof value === "string" && value ? value : "None recorded";
  const paragraph = element(documentRoot, "p", content);
  if (content === "None recorded") {
    paragraph.className = "task-section-empty";
  }
  section.append(paragraph);
  parent.append(section);
}


function appendListSection(documentRoot, parent, headingText, values) {
  const section = element(documentRoot, "section", undefined, "task-section");
  section.append(element(documentRoot, "h3", headingText));
  if (!Array.isArray(values) || values.length === 0) {
    section.append(element(documentRoot, "p", "None recorded", "task-section-empty"));
    parent.append(section);
    return;
  }
  const list = element(documentRoot, "ul");
  for (const value of values) {
    list.append(element(documentRoot, "li", String(value)));
  }
  section.append(list);
  parent.append(section);
}


function control(visible, enabled) {
  return Object.freeze({visible, enabled});
}


export function controlsForState(state, gate) {
  const gateReady = gate?.ready === true;
  const approval = APPROVAL_STATES.has(state);
  return Object.freeze({
    approve: control(approval, approval && gateReady),
    edit: control(approval, approval && gateReady),
    reject: control(approval, approval),
    answer: control(state === "awaiting_user_input", state === "awaiting_user_input" && gateReady),
    stop: control(ACTIVE_STATES.has(state), ACTIVE_STATES.has(state)),
    resume: control(state === "interrupted", state === "interrupted" && gateReady),
  });
}


function actionButton(documentRoot, label, action, descriptor, onAction, className = "") {
  if (!descriptor.visible) {
    return null;
  }
  const button = element(documentRoot, "button", label, `button ${className}`.trim());
  button.type = "button";
  button.dataset.action = action;
  button.disabled = !descriptor.enabled;
  button.addEventListener("click", () => onAction(action));
  return button;
}


function normalizeInspectorGate(options) {
  if (options?.gate) {
    return options.gate;
  }
  const fableReady = options?.fableReady === true;
  const solReady = options?.solReady === true;
  const acknowledged = options?.acknowledged === true;
  return {
    fableReady,
    solReady,
    acknowledged,
    ready: fableReady && solReady && acknowledged,
  };
}


export function renderTaskInspector(documentRoot, task, options = {}) {
  const container = documentRoot.querySelector("#task-inspector");
  if (!container) {
    throw new Error("task inspector region is missing");
  }
  if (!task) {
    const heading = element(documentRoot, "div", undefined, "panel-heading");
    heading.append(element(documentRoot, "h2", "Task inspector"));
    container.replaceChildren(
      heading,
      element(
        documentRoot,
        "p",
        "Select a task to review its contract, scope, evidence, and controls.",
        "empty-state",
      ),
    );
    return container;
  }

  const brief = taskBrief(task);
  const approvalState = APPROVAL_STATES.has(String(task?.state ?? ""));
  const hasOpenQuestions = brief !== null && brief.open_questions.length > 0;
  const approvalBlocked = approvalState && hasOpenQuestions;
  const hasUnusableBrief = brief === null && (
    approvalState || (task?.brief !== null && task?.brief !== undefined)
  );
  const card = element(documentRoot, "article", undefined, "task-card");
  const heading = element(documentRoot, "header", undefined, "task-heading");
  heading.append(
    element(documentRoot, "p", `Task · ${taskIdentity(task)} · revision ${taskRevision(task)}`, "eyebrow"),
    element(documentRoot, "h2", taskTitle(task)),
  );
  const badge = element(
    documentRoot,
    "span",
    stateLabel(String(task?.state ?? "unknown")),
    `state-badge ${stateClass(String(task?.state ?? "unknown"))}`,
  );
  heading.append(badge);
  card.append(heading);

  if (hasUnusableBrief) {
    appendTextSection(
      documentRoot,
      card,
      "Contract mismatch",
      "The approval-state TaskBrief is missing or its identity differs from the task record. Approval and editing are disabled.",
    );
  } else if (brief) {
    appendTextSection(documentRoot, card, FIELD_LABELS.objective, brief.objective);
    for (const field of LIST_FIELDS) {
      appendListSection(documentRoot, card, FIELD_LABELS[field], brief[field]);
    }
    appendTextSection(
      documentRoot,
      card,
      "Confidence",
      `${String(brief.confidence ?? "Not recorded")} · ${String(brief.confidence_rationale ?? "")}`,
    );
  } else {
    appendTextSection(documentRoot, card, "Planning", "Fable is preparing the task contract.");
  }
  if (approvalBlocked) {
    appendTextSection(
      documentRoot,
      card,
      "Approval blocked",
      "Resolve or remove the open questions in Edit before approval.",
    );
  }

  if (task?.outcome) {
    const outcome = asObject(task.outcome);
    appendTextSection(documentRoot, card, "Sol outcome", outcome.summary);
    appendListSection(documentRoot, card, "Changed files", outcome.changed_files);
    appendListSection(documentRoot, card, "Known failures", outcome.known_failures);
    appendListSection(documentRoot, card, "Sol remaining risks", outcome.remaining_risks);
    appendTextSection(documentRoot, card, "Architecture impact", outcome.architecture_docs);
    const question = asObject(outcome.question);
    if (Object.keys(question).length > 0) {
      appendTextSection(documentRoot, card, "Sol question", question.ambiguity);
      appendTextSection(documentRoot, card, "Why it matters", question.why_it_matters);
      appendListSection(documentRoot, card, "Options", question.options);
      appendTextSection(documentRoot, card, "Recommendation", question.recommendation);
      appendTextSection(
        documentRoot,
        card,
        "Can continue safely",
        question.can_continue_safely === true ? "Yes" : "No",
      );
    }
    const commandClaims = Array.isArray(outcome.command_claims)
      ? outcome.command_claims.map((claim) => {
        const value = asObject(claim);
        return `Command ${String(value.command_sha256 ?? "unknown")} · exit ${String(value.exit_code ?? "unknown")}`;
      })
      : [];
    appendListSection(documentRoot, card, "Command evidence", commandClaims);
  }
  if (task?.clarification) {
    const clarification = asObject(task.clarification);
    appendTextSection(
      documentRoot,
      card,
      "Fable clarification",
      `${stateLabel(String(clarification.status ?? "unknown"))} · ${String(clarification.reasoning ?? "")}`,
    );
    appendTextSection(documentRoot, card, "Fable answer", clarification.answer);
    appendTextSection(documentRoot, card, "Question for you", clarification.question_for_user);
    appendTextSection(
      documentRoot,
      card,
      "Clarification confidence",
      `${String(clarification.confidence ?? "Not recorded")} · scope changed: ${clarification.scope_changed === true ? "yes" : "no"}`,
    );
  }
  if (task?.review) {
    const review = asObject(task.review);
    appendTextSection(
      documentRoot,
      card,
      "Fable review",
      `${stateLabel(String(review.status ?? "unknown"))} · ${String(review.summary ?? "")}`,
    );
    appendTextSection(documentRoot, card, "Test assessment", review.test_assessment);
    appendListSection(documentRoot, card, "Scope violations", review.scope_violations);
    appendListSection(documentRoot, card, "Fable remaining risks", review.remaining_risks);
    appendListSection(documentRoot, card, "Requested corrections", review.corrections);
    appendTextSection(documentRoot, card, "Review question", review.question_for_user);
    const criteria = Array.isArray(review.criteria)
      ? review.criteria.flatMap((item) => {
        const criterion = asObject(item);
        const status = criterion.satisfied === true ? "satisfied" : "not satisfied";
        const evidence = Array.isArray(criterion.evidence) ? criterion.evidence : [];
        return [
          `${String(criterion.criterion ?? "Criterion")} · ${status}`,
          ...evidence.map((entry) => `Evidence: ${String(entry)}`),
        ];
      })
      : [];
    appendListSection(documentRoot, card, "Acceptance evidence", criteria);
  }
  if (task?.activity) {
    const activity = element(documentRoot, "details", undefined, "task-section");
    activity.append(
      element(documentRoot, "summary", "Latest agent activity"),
      element(documentRoot, "pre", JSON.stringify(asObject(task.activity), null, 2)),
    );
    card.append(activity);
  }
  appendTextSection(documentRoot, card, "Last activity", taskRecency(task));
  appendTextSection(
    documentRoot,
    card,
    "Approval",
    typeof task?.approved_at === "string"
      ? `Approved at ${task.approved_at}`
      : "Not approved",
  );
  appendTextSection(
    documentRoot,
    card,
    "Continuation",
    typeof task?.continuation_state === "string"
      ? stateLabel(task.continuation_state)
      : "No persisted continuation",
  );
  appendTextSection(
    documentRoot,
    card,
    "Correction count",
    Number.isInteger(task?.correction_count) ? String(task.correction_count) : "Not recorded",
  );
  if (typeof task?.active_agent === "string" || typeof task?.active_started_at === "string") {
    const activeAgent = typeof task.active_agent === "string"
      ? stateLabel(task.active_agent)
      : "Agent unavailable";
    const activeStart = typeof task.active_started_at === "string"
      ? `started ${task.active_started_at} · elapsed ${elapsedLabel(task.active_started_at)}`
      : "start time unavailable";
    appendTextSection(
      documentRoot,
      card,
      "Active run",
      `${activeAgent} · ${activeStart}`,
    );
  } else {
    appendTextSection(documentRoot, card, "Active run", "No active run");
  }
  if (Array.isArray(task?.history) && task.history.length > 0) {
    appendListSection(
      documentRoot,
      card,
      "Lifecycle history",
      task.history.map((entry) => {
        const history = asObject(entry);
        return `${String(history.created_at ?? "Unknown time")} · ${String(history.summary ?? stateLabel(String(history.kind ?? "event")))}`;
      }),
    );
  }

  const gate = normalizeInspectorGate(options);
  const controls = controlsForState(String(task?.state ?? ""), gate);
  const onAction = typeof options.onAction === "function" ? options.onAction : () => {};
  const actions = element(documentRoot, "div", undefined, "task-actions");
  const hiddenControl = control(false, false);
  const definitions = [
    ["Approve & run", "approve", hasUnusableBrief ? hiddenControl : (approvalBlocked ? control(true, false) : controls.approve), "button-primary"],
    ["Edit", "edit", hasUnusableBrief ? hiddenControl : controls.edit, ""],
    ["Reject", "reject", controls.reject, "button-danger"],
    ["Stop", "stop", controls.stop, "button-danger"],
    ["Resume", "resume", controls.resume, "button-primary"],
  ];
  for (const [label, action, descriptor, className] of definitions) {
    const button = actionButton(
      documentRoot,
      label,
      action,
      descriptor,
      () => onAction(action, task),
      className,
    );
    if (button) {
      actions.append(button);
    }
  }

  if (controls.answer.visible) {
    const answerForm = element(documentRoot, "form", undefined, "task-section");
    const answerLabel = element(documentRoot, "label", "Answer the agents");
    const answer = element(documentRoot, "textarea");
    answer.id = `task-answer-${taskIdentity(task)}`;
    answerLabel.setAttribute("for", answer.id);
    answer.name = "answer";
    answer.required = true;
    answer.disabled = !controls.answer.enabled;
    const send = element(documentRoot, "button", "Send answer", "button button-primary");
    send.type = "submit";
    send.disabled = !controls.answer.enabled;
    answerForm.append(answerLabel, answer, send);
    answerForm.addEventListener("submit", (event) => {
      event.preventDefault();
      onAction("answer", task, {answer: String(answer.value ?? "")});
    });
    card.append(answerForm);
  }

  card.append(actions);
  container.replaceChildren(card);
  return container;
}


function listValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  if (typeof value !== "string") {
    throw new Error("list fields must be newline-delimited text or arrays");
  }
  if (value === "") {
    return [];
  }
  return value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}


export function approvalPayload(displayedBrief) {
  const brief = asObject(displayedBrief);
  if (!Number.isInteger(brief.revision) || brief.revision < 1) {
    throw new Error("displayed task revision is invalid");
  }
  return {revision: brief.revision};
}


export function editedRevision(displayedBrief, formValues) {
  const brief = asObject(displayedBrief);
  const values = asObject(formValues);
  requireSafeId(brief.task_id, "task_id");
  if (!Number.isInteger(brief.revision) || brief.revision < 1) {
    throw new Error("displayed task revision is invalid");
  }
  for (const field of EDITABLE_FIELDS) {
    if (!Object.hasOwn(values, field)) {
      throw new Error(`edited task is missing ${field}`);
    }
  }
  const confidence = Number(values.confidence);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new Error("confidence must be between 0 and 1");
  }
  return {
    task_id: brief.task_id,
    revision: brief.revision + 1,
    title: String(values.title),
    objective: String(values.objective),
    context: listValue(values.context),
    constraints: listValue(values.constraints),
    allowed_paths: listValue(values.allowed_paths),
    out_of_scope: listValue(values.out_of_scope),
    acceptance_criteria: listValue(values.acceptance_criteria),
    required_tests: listValue(values.required_tests),
    risks: listValue(values.risks),
    open_questions: listValue(values.open_questions),
    confidence,
    confidence_rationale: String(values.confidence_rationale),
  };
}


function fieldControl(documentRoot, field, value) {
  const group = element(documentRoot, "div", undefined, "field-group");
  const label = element(documentRoot, "label", FIELD_LABELS[field] ?? stateLabel(field));
  label.setAttribute("for", `edit-${field}`);
  const multiline = LIST_FIELDS.includes(field) || field === "objective" || field === "confidence_rationale";
  const input = element(documentRoot, multiline ? "textarea" : "input");
  input.id = `edit-${field}`;
  input.name = field;
  input.required = !["out_of_scope", "risks", "open_questions"].includes(field);
  if (field === "confidence") {
    input.type = "number";
    input.min = "0";
    input.max = "1";
    input.step = "0.01";
  }
  input.value = Array.isArray(value) ? value.join("\n") : String(value ?? "");
  group.append(label, input);
  return {group, input};
}


export function renderTaskEditor(documentRoot, task, onSave, onCancel) {
  const container = documentRoot.querySelector("#task-inspector");
  const brief = taskBrief(task);
  if (!container || !brief) {
    throw new Error("a displayed task brief is required for editing");
  }
  const form = element(documentRoot, "form", undefined, "task-edit-form");
  form.append(
    element(documentRoot, "p", `New revision ${brief.revision + 1}`, "eyebrow"),
    element(documentRoot, "h2", "Edit task contract"),
  );
  const inputs = {};
  for (const field of EDITABLE_FIELDS) {
    const controlNode = fieldControl(documentRoot, field, brief[field]);
    inputs[field] = controlNode.input;
    form.append(controlNode.group);
  }
  const actions = element(documentRoot, "div", undefined, "task-actions");
  const save = element(documentRoot, "button", "Save revision", "button button-primary");
  save.type = "submit";
  const cancel = element(documentRoot, "button", "Cancel", "button");
  cancel.type = "button";
  cancel.addEventListener("click", () => onCancel());
  actions.append(save, cancel);
  form.append(actions);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const values = {};
    for (const field of EDITABLE_FIELDS) {
      values[field] = inputs[field].value;
    }
    onSave(editedRevision(brief, values));
  });
  container.replaceChildren(form);
  return form;
}


function explicitSubscriptionReady(bootstrap) {
  return bootstrap.fable_ready === true
    && bootstrap.fable_status === "subscription_ready";
}


function explicitSolReady(bootstrap) {
  return bootstrap.sol_status === "ready";
}


export function subscriptionGate(bootstrapData) {
  const bootstrap = asObject(bootstrapData);
  const fableReady = explicitSubscriptionReady(bootstrap);
  const solReady = explicitSolReady(bootstrap);
  const acknowledged = bootstrap.usage_credits_acknowledged === true;
  let guidance = "Ready for a new Fable plan.";
  if (!fableReady) {
    const hasStatus = Object.hasOwn(bootstrap, "fable_ready")
      || Object.hasOwn(bootstrap, "fable_status");
    guidance = hasStatus
      ? "Fable subscription authentication is unavailable. Check the foreground server guidance."
      : "Fable subscription status is checking. Actions stay disabled until startup preflight reports readiness.";
  } else if (!solReady) {
    guidance = bootstrap.sol_status === "checking" || !Object.hasOwn(bootstrap, "sol_status")
      ? "Sol CLI status is checking. Actions stay disabled until startup preflight reports readiness."
      : "Sol CLI is unavailable. Check the foreground server guidance.";
  } else if (!acknowledged) {
    guidance = "Confirm that Claude account usage credits are disabled before sending or approving work.";
  }
  return Object.freeze({
    fableReady,
    solReady,
    acknowledged,
    ready: fableReady && solReady && acknowledged,
    canCompose: fableReady && solReady && acknowledged,
    guidance,
  });
}


export function csrfHeaders(token) {
  if (typeof token !== "string" || token.length === 0) {
    throw new Error("CSRF token is unavailable");
  }
  return {"Content-Type": "application/json", "X-CSRF-Token": token};
}


export function taskActionPath(taskId, action) {
  requireSafeId(taskId, "task_id");
  if (!TASK_ACTIONS.has(action)) {
    throw new Error("unsupported task action");
  }
  return `/api/tasks/${encodeURIComponent(taskId)}/${action}`;
}


export function sessionMessagePath(sessionId) {
  requireSafeId(sessionId, "session_id");
  return `/api/sessions/${encodeURIComponent(sessionId)}/messages`;
}


export function projectChatsPath(projectId) {
  requireSafeId(projectId, "project_id");
  return `/api/projects/${encodeURIComponent(projectId)}/chats?limit=${MAX_NAV_CHATS}`;
}


export function projectNewChatPath(projectId) {
  requireSafeId(projectId, "project_id");
  return `/api/projects/${encodeURIComponent(projectId)}/chats`;
}


export function projectChatBootstrapPath(projectId, sessionId) {
  requireSafeId(projectId, "project_id");
  requireSafeId(sessionId, "session_id");
  return `/api/projects/${encodeURIComponent(projectId)}/chats/${encodeURIComponent(sessionId)}/bootstrap`;
}


export function projectChatMessagePath(projectId, sessionId) {
  requireSafeId(projectId, "project_id");
  requireSafeId(sessionId, "session_id");
  return `/api/projects/${encodeURIComponent(projectId)}/chats/${encodeURIComponent(sessionId)}/messages`;
}


export function projectTaskActionPath(projectId, sessionId, taskId, action) {
  requireSafeId(projectId, "project_id");
  requireSafeId(sessionId, "session_id");
  requireSafeId(taskId, "task_id");
  if (!TASK_ACTIONS.has(action)) {
    throw new Error("unsupported task action");
  }
  return `/api/projects/${encodeURIComponent(projectId)}/chats/${encodeURIComponent(sessionId)}/tasks/${encodeURIComponent(taskId)}/${action}`;
}


export async function postJson(fetchFunction, path, payload, csrfToken) {
  if (typeof path !== "string" || !path.startsWith("/api/") || path.includes("?")) {
    throw new Error("mutation path must be a body-only API path");
  }
  const response = await fetchFunction(path, {
    method: "POST",
    credentials: "same-origin",
    cache: "no-store",
    headers: csrfHeaders(csrfToken),
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      }
    } catch (_error) {
      // The status is sufficient when an error response has no JSON body.
    }
    throw new Error(detail);
  }
  return response;
}


export function acceptSequence(lastSequence, event) {
  const sequence = event?.sequence;
  if (!Number.isSafeInteger(sequence) || sequence < 1 || sequence <= lastSequence) {
    return {accepted: false, lastSequence};
  }
  return {accepted: true, lastSequence: sequence};
}


export function reconnectDelay(attempt) {
  const index = Number.isInteger(attempt) && attempt >= 0 ? attempt : 0;
  return RECONNECT_DELAYS[Math.min(index, RECONNECT_DELAYS.length - 1)];
}


export function websocketPath(sessionId, lastSequence) {
  requireSafeId(sessionId, "session_id");
  if (!Number.isSafeInteger(lastSequence) || lastSequence < 0) {
    throw new Error("last sequence must be a non-negative safe integer");
  }
  return `/ws?session_id=${encodeURIComponent(sessionId)}&after=${lastSequence}`;
}


export function websocketUrl(sessionId, lastSequence, locationValue) {
  const location = asObject(locationValue);
  if (!['http:', 'https:'].includes(location.protocol) || typeof location.host !== "string") {
    throw new Error("browser location must be loopback HTTP or HTTPS");
  }
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}${websocketPath(sessionId, lastSequence)}`;
}


export function projectWebsocketPath(projectId, sessionId, lastSequence) {
  requireSafeId(projectId, "project_id");
  requireSafeId(sessionId, "session_id");
  if (!Number.isSafeInteger(lastSequence) || lastSequence < 0) {
    throw new Error("last sequence must be a non-negative safe integer");
  }
  return `/ws?project_id=${encodeURIComponent(projectId)}&session_id=${encodeURIComponent(sessionId)}&after=${lastSequence}`;
}


export function projectWebsocketUrl(projectId, sessionId, lastSequence, locationValue) {
  const location = asObject(locationValue);
  if (!['http:', 'https:'].includes(location.protocol) || typeof location.host !== "string") {
    throw new Error("browser location must be loopback HTTP or HTTPS");
  }
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}${projectWebsocketPath(projectId, sessionId, lastSequence)}`;
}


export function createEventStream({
  sessionId,
  initialSequence = 0,
  WebSocketCtor,
  schedule,
  cancelSchedule,
  location,
  socketUrl = websocketUrl,
  bootstrap,
  onEvent,
  onStatus,
}) {
  requireSafeId(sessionId, "session_id");
  if (!Number.isSafeInteger(initialSequence) || initialSequence < 0) {
    throw new Error("initial sequence must be a non-negative safe integer");
  }
  if (
    typeof WebSocketCtor !== "function"
    || typeof schedule !== "function"
    || typeof cancelSchedule !== "function"
    || typeof socketUrl !== "function"
  ) {
    throw new Error("event stream dependencies are invalid");
  }
  let lastSequence = initialSequence;
  let reconnectAttempt = 0;
  let socket = null;
  let reconnectTimer = null;
  let refreshTimer = null;
  let refreshToken = 0;
  let refreshInFlightToken = null;
  let generation = 0;
  let stopped = false;

  function invalidateRefresh() {
    refreshToken += 1;
    refreshInFlightToken = null;
    if (refreshTimer !== null) {
      cancelSchedule(refreshTimer);
      refreshTimer = null;
    }
  }

  function scheduleRefresh(ownGeneration) {
    if (
      stopped
      || ownGeneration !== generation
      || socket === null
      || refreshTimer !== null
    ) {
      return;
    }
    refreshTimer = schedule(() => {
      refreshTimer = null;
      if (!stopped && ownGeneration === generation && socket !== null) {
        void refreshBootstrap(ownGeneration);
      }
    }, BOOTSTRAP_REFRESH_DELAY);
  }

  async function refreshBootstrap(ownGeneration) {
    if (stopped || ownGeneration !== generation || socket === null) {
      return false;
    }
    const ownToken = refreshToken + 1;
    refreshToken = ownToken;
    refreshInFlightToken = ownToken;
    const bootstrapSequence = lastSequence;
    const isCurrent = () => (
      !stopped
      && ownGeneration === generation
      && ownToken === refreshToken
      && lastSequence === bootstrapSequence
      && socket !== null
    );
    onStatus("bootstrap_refreshing");
    let refreshed;
    try {
      refreshed = await bootstrap(isCurrent);
    } catch (_error) {
      if (ownToken === refreshToken && ownGeneration === generation && !stopped) {
        refreshInFlightToken = null;
        onStatus("bootstrap_error");
        scheduleRefresh(ownGeneration);
      }
      return false;
    }
    if (ownToken !== refreshToken || ownGeneration !== generation || stopped) {
      return false;
    }
    refreshInFlightToken = null;
    if (refreshed === false || lastSequence !== bootstrapSequence) {
      onStatus("bootstrap_stale");
      scheduleRefresh(ownGeneration);
      return false;
    }
    onStatus("connected");
    return true;
  }

  function connect(isReconnect = false) {
    if (stopped) {
      return null;
    }
    if (socket !== null || reconnectTimer !== null) {
      return socket;
    }
    generation += 1;
    const ownGeneration = generation;
    const ownSocket = new WebSocketCtor(socketUrl(sessionId, lastSequence, location));
    socket = ownSocket;
    ownSocket.addEventListener("open", async () => {
      if (stopped || ownGeneration !== generation) {
        return;
      }
      reconnectAttempt = 0;
      if (isReconnect) {
        await refreshBootstrap(ownGeneration);
      } else {
        onStatus("connected");
      }
    });
    ownSocket.addEventListener("message", (message) => {
      if (stopped || ownGeneration !== generation) {
        return;
      }
      let event;
      try {
        event = JSON.parse(String(message.data));
      } catch (_error) {
        onStatus("invalid_event");
        return;
      }
      const accepted = acceptSequence(lastSequence, event);
      if (!accepted.accepted) {
        return;
      }
      lastSequence = accepted.lastSequence;
      onEvent(event);
      if (refreshInFlightToken !== null) {
        refreshToken += 1;
        refreshInFlightToken = null;
        onStatus("bootstrap_stale");
        scheduleRefresh(ownGeneration);
      }
    });
    ownSocket.addEventListener("error", () => {
      if (!stopped && ownGeneration === generation) {
        onStatus("connection_error");
      }
    });
    ownSocket.addEventListener("close", () => {
      if (stopped || ownGeneration !== generation || reconnectTimer !== null) {
        return;
      }
      invalidateRefresh();
      generation += 1;
      socket = null;
      onStatus("reconnecting");
      const delay = reconnectDelay(reconnectAttempt);
      reconnectAttempt += 1;
      reconnectTimer = schedule(() => {
        reconnectTimer = null;
        connect(true);
      }, delay);
    });
    return socket;
  }

  return {
    connect,
    stop() {
      stopped = true;
      invalidateRefresh();
      generation += 1;
      if (reconnectTimer !== null) {
        cancelSchedule(reconnectTimer);
        reconnectTimer = null;
      }
      if (socket && typeof socket.close === "function") {
        socket.close();
      }
    },
    get lastSequence() {
      return lastSequence;
    },
  };
}


function navigationProject(value) {
  const candidate = asObject(value);
  if (typeof candidate.project_id !== "string" || !SAFE_ID.test(candidate.project_id)) {
    return null;
  }
  return {
    projectId: candidate.project_id,
    label: typeof candidate.label === "string" && candidate.label
      ? candidate.label
      : candidate.project_id,
    branch: typeof candidate.branch === "string" ? candidate.branch : null,
    readiness: asObject(candidate.readiness),
  };
}


function navigationChat(value) {
  const candidate = asObject(value);
  if (typeof candidate.session_id !== "string" || !SAFE_ID.test(candidate.session_id)) {
    return null;
  }
  return {
    sessionId: candidate.session_id,
    title: typeof candidate.title === "string" && candidate.title
      ? candidate.title
      : "New chat",
    latestSequence: Number.isSafeInteger(candidate.latest_sequence)
      ? candidate.latest_sequence
      : 0,
  };
}


function navigationLease(value) {
  const candidate = asObject(value);
  if (
    typeof candidate.project_id !== "string"
    || typeof candidate.session_id !== "string"
    || typeof candidate.task_id !== "string"
    || !SAFE_ID.test(candidate.project_id)
    || !SAFE_ID.test(candidate.session_id)
    || !SAFE_ID.test(candidate.task_id)
  ) {
    return null;
  }
  return {
    projectId: candidate.project_id,
    sessionId: candidate.session_id,
    taskId: candidate.task_id,
  };
}


function navigationLeaseState(payload) {
  if (!Object.hasOwn(payload, "active_lease")) {
    return {activeLease: null, navigationLocked: true};
  }
  if (payload.active_lease === null) {
    return {activeLease: null, navigationLocked: false};
  }
  const activeLease = navigationLease(payload.active_lease);
  return activeLease === null
    ? {activeLease: null, navigationLocked: true}
    : {activeLease, navigationLocked: false};
}


function navigationCsrfToken(payload) {
  return typeof payload.csrf_token === "string" && SAFE_CSRF_TOKEN.test(payload.csrf_token)
    ? payload.csrf_token
    : null;
}


function boundedNavigationRecords(records, normalizer, maximum) {
  if (!Array.isArray(records)) {
    return [];
  }
  return records
    .map(normalizer)
    .filter((record) => record !== null)
    .slice(0, maximum);
}


function navigationProjectReadiness(projects, projectId) {
  return asObject(projects.find((project) => project.projectId === projectId)?.readiness);
}


function navigationProjectGate(projects, projectId, acknowledged) {
  const readiness = navigationProjectReadiness(projects, projectId);
  return subscriptionGate({
    fable_ready: readiness.fable_ready,
    fable_status: readiness.fable_status,
    sol_status: readiness.sol_status,
    usage_credits_acknowledged: acknowledged === true,
  });
}


function navigationState(previousState, payload) {
  const payloadObject = asObject(payload);
  const lease = navigationLeaseState(payloadObject);
  const csrfToken = navigationCsrfToken(payloadObject);
  const projects = boundedNavigationRecords(
    payloadObject.projects,
    navigationProject,
    MAX_NAV_PROJECTS,
  );
  const acknowledged = payloadObject.usage_credits_acknowledged === true;
  return {
    ...previousState,
    csrfToken: csrfToken ?? "",
    projects,
    activeLease: lease.navigationLocked
      ? (previousState.activeLease ?? null)
      : lease.activeLease,
    navigationLocked: lease.navigationLocked || csrfToken === null,
    gate: navigationProjectGate(projects, previousState.projectId, acknowledged),
    solStatus: navigationProjectReadiness(projects, previousState.projectId).sol_status ?? null,
  };
}


function chatListState(previousState, payload) {
  return {
    ...previousState,
    chats: boundedNavigationRecords(asObject(payload).chats, navigationChat, MAX_NAV_CHATS),
  };
}


function projectLabelFor(state, projectId) {
  return state.projects.find((project) => project.projectId === projectId)?.label
    ?? projectId
    ?? null;
}


function navigationAttribute(node, name) {
  if (typeof node?.getAttribute === "function") {
    return node.getAttribute(name);
  }
  return node?.attributes?.[name] ?? null;
}


function navigationListMatches(container, records, selectedId, locked, idName, labelName) {
  if (!container || container.children.length !== records.length) {
    return false;
  }
  return records.every((record, index) => {
    const button = container.children[index]?.children?.[0];
    return button?.dataset?.[idName] === record[idName]
      && button.textContent === record[labelName]
      && button.disabled === locked
      && navigationAttribute(button, "aria-current") === (
        record[idName] === selectedId ? "true" : "false"
      );
  });
}


function renderNavigationList(
  documentRoot,
  container,
  records,
  selectedId,
  locked,
  idName,
  labelName,
  onSelect,
) {
  if (navigationListMatches(container, records, selectedId, locked, idName, labelName)) {
    return;
  }
  const items = [];
  for (const record of records) {
    if (typeof record?.[idName] !== "string") {
      continue;
    }
    const item = element(documentRoot, "li");
    const button = element(documentRoot, "button", record[labelName], "project-navigation-button");
    button.type = "button";
    button.disabled = locked;
    button.dataset[idName] = record[idName];
    button.setAttribute("aria-current", record[idName] === selectedId ? "true" : "false");
    button.addEventListener("click", () => onSelect(record[idName]));
    item.append(button);
    items.push(item);
  }
  container?.replaceChildren(...items);
}


export function renderProjectNavigation(documentRoot, currentState, handlers = {}) {
  const state = asObject(currentState);
  const projectList = documentRoot.querySelector("#project-list");
  const chatList = documentRoot.querySelector("#chat-list");
  const newChat = documentRoot.querySelector("#new-chat");
  const projectName = documentRoot.querySelector("#selected-project-name");
  const chatName = documentRoot.querySelector("#selected-chat-name");
  const projects = Array.isArray(state.projects)
    ? state.projects.slice(0, MAX_NAV_PROJECTS)
    : [];
  const chats = Array.isArray(state.chats)
    ? state.chats.slice(0, MAX_NAV_CHATS)
    : [];
  const locked = state.activeLease !== null && state.activeLease !== undefined
    || state.navigationLocked === true
    || state.navigationPending === true;
  const selectedProject = projects.find((project) => project?.projectId === state.projectId);
  const selectedChat = chats.find((chat) => chat?.sessionId === state.sessionId);
  if (projectName) {
    projectName.textContent = typeof selectedProject?.label === "string"
      ? selectedProject.label
      : (typeof state.projectLabel === "string" ? state.projectLabel : "No project selected");
  }
  if (chatName) {
    chatName.textContent = typeof selectedChat?.title === "string"
      ? selectedChat.title
      : "No chat selected";
  }
  if (projectList) {
    renderNavigationList(
      documentRoot, projectList, projects, state.projectId, locked,
      "projectId", "label", (projectId) => handlers.onProject?.(projectId),
    );
  }
  if (chatList) {
    renderNavigationList(
      documentRoot, chatList, chats, state.sessionId, locked,
      "sessionId", "title", (sessionId) => handlers.onChat?.(sessionId),
    );
  }
  if (newChat) {
    newChat.disabled = locked
      || typeof state.projectId !== "string"
      || typeof state.csrfToken !== "string"
      || state.csrfToken.length === 0;
    newChat.onclick = () => handlers.onNewChat?.();
  }
}


export function createProjectChatController({
  fetchFunction,
  WebSocketCtor,
  schedule,
  cancelSchedule,
  location,
  onState = () => {},
  onEvent = () => {},
  onStatus = () => {},
  onChatReset = () => {},
}) {
  if (typeof fetchFunction !== "function") {
    throw new Error("project chat fetch function is required");
  }
  let state = {
    csrfToken: "",
    projectId: null,
    projectLabel: null,
    sessionId: null,
    projects: [],
    chats: [],
    activeLease: null,
    navigationLocked: true,
    navigationPending: false,
    tasks: [],
    selectedTaskId: null,
    gate: subscriptionGate({}),
    lastSequence: 0,
  };
  let stream = null;
  let selectionGeneration = 0;
  let projectRefreshGeneration = 0;
  let projectRefreshInFlight = null;
  let projectRefreshPending = false;
  let selectionPending = false;
  let projectRefreshBlocksNavigation = false;

  function publish(nextState) {
    state = nextState;
    onState(state);
  }

  async function getJson(path) {
    const response = await fetchFunction(path, {
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    return response.json();
  }

  function selected(projectId, sessionId, generation) {
    return selectionGeneration === generation
      && state.projectId === projectId
      && state.sessionId === sessionId;
  }

  function stopStream() {
    stream?.stop();
    stream = null;
  }

  function navigationPending() {
    return selectionPending || projectRefreshBlocksNavigation;
  }

  function invalidateProjectRefresh() {
    projectRefreshGeneration += 1;
    if (projectRefreshInFlight !== null) {
      projectRefreshPending = true;
      projectRefreshBlocksNavigation = true;
    }
  }

  function beginChatSelection(projectId, sessionId) {
    stopStream();
    invalidateProjectRefresh();
    selectionPending = true;
    if (state.sessionId !== null && state.sessionId !== sessionId) {
      onChatReset();
    }
    selectionGeneration += 1;
    publish({
      ...state,
      projectId,
      projectLabel: projectLabelFor(state, projectId),
      sessionId,
      tasks: [],
      selectedTaskId: null,
      gate: navigationProjectGate(state.projects, projectId, state.gate.acknowledged),
      solStatus: navigationProjectReadiness(state.projects, projectId).sol_status ?? null,
      lastSequence: 0,
      navigationPending: navigationPending(),
    });
    return selectionGeneration;
  }

  function startStream(projectId, sessionId) {
    if (stream || !selected(projectId, sessionId, selectionGeneration)) {
      return;
    }
    stream = createEventStream({
      sessionId,
      initialSequence: state.lastSequence,
      WebSocketCtor,
      schedule,
      cancelSchedule,
      location,
      socketUrl: (_streamSessionId, lastSequence, locationValue) => (
        projectWebsocketUrl(projectId, sessionId, lastSequence, locationValue)
      ),
      bootstrap: (isCurrent) => bootstrapSelected(
        projectId,
        sessionId,
        selectionGeneration,
        isCurrent,
      ),
      onEvent: (event) => {
        const previousSelection = state.selectedTaskId;
        const tasks = reduceTaskEvent(state.tasks, event);
        const selectedTaskId = repairTaskSelection(tasks, previousSelection);
        const sequence = Number.isSafeInteger(event?.sequence) && event.sequence >= 0
          ? Math.max(state.lastSequence, event.sequence)
          : state.lastSequence;
        publish({...state, tasks, selectedTaskId, lastSequence: sequence});
        const associatedTask = tasks.find((task) => taskIdentity(task) === event?.task_id);
        onEvent(event, associatedTask);
        void refreshProjects().catch(() => {});
      },
      onStatus: (status) => {
        if (["bootstrap_refreshing", "bootstrap_stale", "bootstrap_error"].includes(status)) {
          publish({
            ...state,
            gate: {...state.gate, ready: false, canCompose: false},
          });
        }
        onStatus(status);
      },
    });
    stream.connect();
  }

  async function bootstrapSelected(projectId, sessionId, generation, isCurrent = () => true) {
    const bootstrapData = await getJson(projectChatBootstrapPath(projectId, sessionId));
    if (!isCurrent() || !selected(projectId, sessionId, generation)) {
      return false;
    }
    const applied = applyBootstrap(state, bootstrapData);
    if (applied.projectId !== projectId || applied.sessionId !== sessionId) {
      throw new Error("bootstrap selection did not match the requested project chat");
    }
    selectionPending = false;
    publish({
      ...applied,
      projectLabel: projectLabelFor(applied, projectId),
      navigationPending: navigationPending(),
    });
    startStream(projectId, sessionId);
    return true;
  }

  async function selectChat(projectId, sessionId) {
    requireSafeId(projectId, "project_id");
    requireSafeId(sessionId, "session_id");
    if (state.projectId !== projectId) {
      return selectProject(projectId, sessionId);
    }
    if (state.sessionId === sessionId && stream !== null) {
      return true;
    }
    const generation = beginChatSelection(projectId, sessionId);
    return bootstrapSelected(projectId, sessionId, generation);
  }

  async function selectProject(projectId, requestedSessionId = null) {
    requireSafeId(projectId, "project_id");
    if (requestedSessionId !== null) {
      requireSafeId(requestedSessionId, "session_id");
    }
    if (state.projectId === projectId && requestedSessionId === null && stream !== null) {
      return true;
    }
    const currentSessionId = state.projectId === projectId ? state.sessionId : null;
    stopStream();
    invalidateProjectRefresh();
    selectionPending = true;
    if (state.sessionId !== null) {
      onChatReset();
    }
    selectionGeneration += 1;
    const generation = selectionGeneration;
    publish({
      ...state,
      projectId,
      projectLabel: projectLabelFor(state, projectId),
      sessionId: null,
      chats: [],
      tasks: [],
      selectedTaskId: null,
      gate: navigationProjectGate(state.projects, projectId, state.gate.acknowledged),
      solStatus: navigationProjectReadiness(state.projects, projectId).sol_status ?? null,
      lastSequence: 0,
      navigationPending: navigationPending(),
    });
    const chatPayload = await getJson(projectChatsPath(projectId));
    if (generation !== selectionGeneration || state.projectId !== projectId) {
      return false;
    }
    const withChats = chatListState(state, chatPayload);
    publish(withChats);
    const leasedSessionId = state.activeLease?.projectId === projectId
      ? state.activeLease.sessionId
      : null;
    const targetSessionId = requestedSessionId
      ?? leasedSessionId
      ?? (withChats.chats.some((chat) => chat.sessionId === currentSessionId)
        ? currentSessionId
        : withChats.chats[0]?.sessionId);
    if (typeof targetSessionId !== "string") {
      selectionPending = false;
      publish({...withChats, navigationPending: navigationPending()});
      return true;
    }
    return selectChat(projectId, targetSessionId);
  }

  async function performProjectRefresh() {
    const ownGeneration = projectRefreshGeneration;
    let refreshed = false;
    try {
      const payload = await getJson("/api/projects");
      if (ownGeneration === projectRefreshGeneration) {
        projectRefreshBlocksNavigation = false;
        const nextState = navigationState(state, payload);
        const label = typeof nextState.projectId === "string"
          ? projectLabelFor(nextState, nextState.projectId)
          : null;
        publish({...nextState, projectLabel: label, navigationPending: navigationPending()});
        refreshed = true;
      }
    } catch (_error) {
      if (ownGeneration === projectRefreshGeneration) {
        projectRefreshBlocksNavigation = false;
        publish({...state, navigationLocked: true, navigationPending: navigationPending()});
      }
    }
    if (projectRefreshPending) {
      projectRefreshPending = false;
      return performProjectRefresh();
    }
    return refreshed;
  }

  function refreshProjects() {
    projectRefreshGeneration += 1;
    if (projectRefreshInFlight !== null) {
      projectRefreshPending = true;
      return projectRefreshInFlight;
    }
    const refresh = performProjectRefresh();
    projectRefreshInFlight = refresh;
    void refresh.finally(() => {
      if (projectRefreshInFlight === refresh) {
        projectRefreshInFlight = null;
      }
    });
    return refresh;
  }

  async function bootstrapInitial() {
    await refreshProjects();
    const leasedProjectId = state.activeLease?.projectId ?? null;
    const selectedProjectId = leasedProjectId
      ?? (state.projects.some((project) => project.projectId === state.projectId)
        ? state.projectId
        : state.projects[0]?.projectId);
    if (typeof selectedProjectId !== "string") {
      return false;
    }
    return selectProject(selectedProjectId);
  }

  async function refreshSelectedBootstrap() {
    if (typeof state.projectId !== "string" || typeof state.sessionId !== "string") {
      return false;
    }
    return bootstrapSelected(state.projectId, state.sessionId, selectionGeneration);
  }

  async function createChat() {
    if (
      state.activeLease !== null
      || state.navigationLocked
      || state.navigationPending
      || typeof state.projectId !== "string"
      || typeof state.csrfToken !== "string"
      || state.csrfToken.length === 0
    ) {
      throw new Error("new chat is unavailable while navigation is locked");
    }
    const projectId = state.projectId;
    const sessionId = state.sessionId;
    const generation = selectionGeneration;
    const response = await postJson(
      fetchFunction,
      projectNewChatPath(projectId),
      {},
      state.csrfToken,
    );
    const created = navigationChat(await response.json());
    if (created === null) {
      throw new Error("new chat response was invalid");
    }
    if (
      generation !== selectionGeneration
      || state.projectId !== projectId
      || state.sessionId !== sessionId
    ) {
      return false;
    }
    publish({
      ...state,
      chats: [created, ...state.chats.filter((chat) => chat.sessionId !== created.sessionId)]
        .slice(0, MAX_NAV_CHATS),
    });
    return selectChat(projectId, created.sessionId);
  }

  return {
    bootstrapInitial,
    createChat,
    refreshProjects,
    refreshSelectedBootstrap,
    selectChat,
    selectProject,
    stop: stopStream,
    get state() {
      return state;
    },
  };
}


export function applyBootstrap(previousState, bootstrapData) {
  const previous = asObject(previousState);
  const bootstrap = asObject(bootstrapData);
  const projectId = typeof bootstrap.project_id === "string" && SAFE_ID.test(bootstrap.project_id)
    ? bootstrap.project_id
    : null;
  const sessionId = typeof bootstrap.session_id === "string" && SAFE_ID.test(bootstrap.session_id)
    ? bootstrap.session_id
    : null;
  const previousTasks = Array.isArray(previous.tasks) ? previous.tasks : [];
  const tasks = Array.isArray(bootstrap.tasks)
    ? bootstrap.tasks.slice(0, MAX_TASK_OVERVIEWS).map((task) => {
      const prior = previousTasks.find((candidate) => taskIdentity(candidate) === taskIdentity(task));
      const rawIncoming = asObject(task);
      const boundary = Number.isSafeInteger(rawIncoming.revision_start_sequence)
        && rawIncoming.revision_start_sequence >= 1
        ? rawIncoming.revision_start_sequence
        : null;
      const incoming = {
        ...rawIncoming,
        revision_start_sequence: boundary,
        outcome: safeEvidenceProjection(rawIncoming.outcome),
        review: safeEvidenceProjection(rawIncoming.review),
        clarification: safeEvidenceProjection(rawIncoming.clarification),
        activity: safeEvidenceProjection(rawIncoming.activity),
      };
      const priorBoundary = Number.isSafeInteger(prior?.revision_start_sequence)
        && prior.revision_start_sequence >= 1
        ? prior.revision_start_sequence
        : null;
      const sameBoundary = prior
        && Number.isInteger(prior.revision)
        && Number.isInteger(incoming.revision)
        && prior.revision === incoming.revision
        && priorBoundary === boundary;
      return {
        ...incoming,
        history: sameBoundary && Array.isArray(prior.history)
          ? prior.history.slice(-MAX_TASK_HISTORY)
          : [],
      };
    })
    : [];
  const selectedTaskId = repairTaskSelection(tasks, previous.selectedTaskId);
  const previousSequence = Number.isSafeInteger(previous.lastSequence)
    ? previous.lastSequence
    : 0;
  const replayAfter = Number.isSafeInteger(bootstrap.replay_after) && bootstrap.replay_after >= 0
    ? bootstrap.replay_after
    : 0;
  return {
    ...previous,
    csrfToken: typeof bootstrap.csrf_token === "string" ? bootstrap.csrf_token : "",
    projectId,
    sessionId,
    tasks,
    selectedTaskId,
    solStatus: bootstrap.sol_status ?? null,
    repository: bootstrap.repository ?? bootstrap.repo_root ?? null,
    branch: bootstrap.branch ?? null,
    gate: subscriptionGate(bootstrap),
    lastSequence: Math.max(previousSequence, replayAfter),
  };
}


export function repairTaskSelection(tasks, selectedTaskId) {
  const safeTasks = Array.isArray(tasks) ? tasks : [];
  if (safeTasks.some((task) => taskIdentity(task) === selectedTaskId)) {
    return selectedTaskId;
  }
  return safeTasks[0] ? taskIdentity(safeTasks[0]) : null;
}


export function deriveSolStatus(tasks, configuredStatus) {
  const safeTasks = Array.isArray(tasks) ? tasks : [];
  if (safeTasks.some((task) => ["sol_running", "sol_correcting"].includes(task?.state))) {
    return "running";
  }
  if (safeTasks.some((task) => ["awaiting_user_input", "awaiting_scope_approval"].includes(task?.state))) {
    return "blocked";
  }
  return typeof configuredStatus === "string" ? configuredStatus : "checking";
}


function withoutRevisionEvidence(task) {
  const {
    outcome: _outcome,
    review: _review,
    clarification: _clarification,
    activity: _activity,
    history: _history,
    ...snapshot
  } = asObject(task);
  return snapshot;
}


function safeEvidenceProjection(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value
    : null;
}


export function associatedRevisionForEvent(event, task) {
  const snapshot = asObject(task);
  const boundary = snapshot.revision_start_sequence;
  const sequence = event?.sequence;
  if (Number.isSafeInteger(boundary) && boundary >= 1) {
    if (!Number.isSafeInteger(sequence) || sequence < boundary) {
      return null;
    }
  }
  const payload = asObject(event?.payload);
  if (Number.isInteger(payload.revision)) {
    return payload.revision;
  }
  const briefRevision = asObject(payload.brief).revision;
  if (Number.isInteger(briefRevision)) {
    return briefRevision;
  }
  return Number.isSafeInteger(boundary)
    && Number.isSafeInteger(sequence)
    && sequence >= boundary
    && Number.isInteger(snapshot.revision)
    ? snapshot.revision
    : null;
}


function upsertTask(tasks, incoming) {
  const taskId = taskIdentity(incoming);
  const next = tasks.filter((task) => taskIdentity(task) !== taskId);
  return [incoming, ...next].slice(0, MAX_TASK_OVERVIEWS);
}


export function reduceTaskEvent(tasks, event) {
  if (typeof event?.task_id !== "string") {
    return tasks;
  }
  const existing = tasks.find((task) => taskIdentity(task) === event.task_id) ?? {
    task_id: event.task_id,
    revision: 0,
    state: "fable_planning",
    brief: null,
  };
  const payload = asObject(event.payload);
  const briefPayload = event.kind === "task_brief" ? asObject(payload.brief) : {};
  const isTaskBrief = event.kind === "task_brief";
  const incomingRevision = event.kind === "task_brief" && Number.isInteger(briefPayload.revision)
    ? briefPayload.revision
    : (event.kind === "task_state" && Number.isInteger(payload.revision)
      ? payload.revision
      : existing.revision);
  if (Number.isInteger(incomingRevision) && incomingRevision < existing.revision) {
    return tasks;
  }
  const currentBoundary = existing.revision_start_sequence;
  if (
    !isTaskBrief
    && Number.isSafeInteger(currentBoundary)
    && currentBoundary >= 1
    && (!Number.isSafeInteger(event.sequence) || event.sequence < currentBoundary)
  ) {
    return tasks;
  }
  const revisionChanged = Number.isInteger(incomingRevision)
    && incomingRevision !== existing.revision;
  const startsRevisionBoundary = isTaskBrief;
  const clearsRevisionEvidence = revisionChanged || startsRevisionBoundary;
  const base = clearsRevisionEvidence ? withoutRevisionEvidence(existing) : existing;
  const history = clearsRevisionEvidence
    ? []
    : (Array.isArray(existing.history) ? existing.history : []);
  let updated = {
    ...base,
    revision: incomingRevision,
    ...(revisionChanged ? {
      brief: null,
      approved_at: null,
      correction_count: 0,
      continuation_state: null,
      active_agent: null,
      active_started_at: null,
    } : {}),
    ...(startsRevisionBoundary ? {
      revision_start_sequence: Number.isSafeInteger(event.sequence)
        ? event.sequence
        : null,
    } : (revisionChanged ? {revision_start_sequence: null} : {})),
    updated_at: typeof event.created_at === "string" ? event.created_at : existing.updated_at,
    history: [
      ...history,
      {
        sequence: event.sequence,
        actor: event.actor,
        kind: event.kind,
        created_at: event.created_at,
        summary: eventText(event),
      },
    ].slice(-MAX_TASK_HISTORY),
  };
  if (event.kind === "task_brief") {
    const brief = briefPayload;
    updated = {
      ...updated,
      task_id: event.task_id,
      revision: Number.isInteger(brief.revision) ? brief.revision : existing.revision,
      state: existing.state === "awaiting_scope_approval"
        ? "awaiting_scope_approval"
        : "awaiting_user_approval",
      brief,
    };
  } else if (event.kind === "task_state" && typeof payload.state === "string") {
    updated = {
      ...updated,
      state: payload.state,
      revision: Number.isInteger(payload.revision) ? payload.revision : existing.revision,
    };
  } else if (event.kind === "task_rejected") {
    updated = {...updated, state: "failed"};
  } else if (event.kind === "outcome") {
    updated = {...updated, outcome: payload};
  } else if (event.kind === "clarification") {
    updated = {...updated, clarification: payload};
  } else if (event.kind === "review") {
    updated = {...updated, review: payload};
  } else if (["agent_event", "resume_drift", "stop_error", "action_error"].includes(event.kind)) {
    updated = {...updated, activity: payload};
  }
  return upsertTask(tasks, updated);
}


function setStatus(node, text, variant) {
  node.textContent = text;
  node.className = `status-pill status-${variant}`;
}


function showToast(documentRoot, text, isError = false) {
  const region = documentRoot.querySelector("#toast-region");
  if (!region) {
    return;
  }
  const toast = element(
    documentRoot,
    "div",
    text,
    `toast${isError ? " toast-error" : ""}`,
  );
  toast.setAttribute("role", isError ? "alert" : "status");
  region.replaceChildren(toast);
}


function closeDrawer(documentRoot, id, button, isMobileDrawer) {
  const panel = documentRoot.querySelector(id);
  if (panel) {
    panel.classList.remove("drawer-open");
    panel.inert = isMobileDrawer();
  }
  button.setAttribute("aria-expanded", "false");
}


function wireDrawer(
  documentRoot,
  buttonId,
  panelId,
  otherButtonId,
  otherPanelId,
  isMobileDrawer,
) {
  const button = documentRoot.querySelector(buttonId);
  const panel = documentRoot.querySelector(panelId);
  const otherButton = documentRoot.querySelector(otherButtonId);
  if (!button || !panel || !otherButton) {
    return;
  }
  button.addEventListener("click", () => {
    const opening = !panel.classList.contains("drawer-open");
    closeDrawer(documentRoot, otherPanelId, otherButton, isMobileDrawer);
    panel.classList.toggle("drawer-open", opening);
    panel.inert = isMobileDrawer() && !opening;
    button.setAttribute("aria-expanded", opening ? "true" : "false");
    if (opening) {
      const focusTarget = panel.querySelector("button, textarea, input, select, [tabindex]");
      focusTarget?.focus();
    }
  });
}


export function startBrowserApp(documentRoot, windowRoot) {
  let state = {
    csrfToken: "",
    projectId: null,
    projectLabel: null,
    sessionId: null,
    projects: [],
    chats: [],
    activeLease: null,
    navigationPending: false,
    tasks: [],
    selectedTaskId: null,
    gate: subscriptionGate({}),
    lastSequence: 0,
  };
  let controller = null;
  let ready = Promise.resolve(false);

  const composer = documentRoot.querySelector("#composer");
  const messageInput = documentRoot.querySelector("#message-input");
  const composerSubmit = documentRoot.querySelector("#composer-submit");
  const guidance = documentRoot.querySelector("#composer-guidance");
  const usageModal = documentRoot.querySelector("#usage-modal");
  const usageForm = documentRoot.querySelector("#usage-credits-form");
  const usageCheckbox = documentRoot.querySelector("#usage-credits-confirm");
  const usageSubmit = documentRoot.querySelector("#usage-credits-acknowledge");
  const usageError = documentRoot.querySelector("#usage-error");
  const bootstrapRetry = documentRoot.querySelector("#bootstrap-retry");
  const focusBeforeModal = documentRoot.activeElement;
  const drawerMedia = typeof windowRoot.matchMedia === "function"
    ? windowRoot.matchMedia("(max-width: 899px)")
    : {matches: false};
  const isMobileDrawer = () => drawerMedia.matches === true;

  if (!usageModal.open && typeof usageModal.showModal === "function") {
    usageModal.showModal();
  }
  usageCheckbox.focus();

  function selectedTask() {
    return state.tasks.find((task) => taskIdentity(task) === state.selectedTaskId) ?? null;
  }

  function renderStatus() {
    const fableNode = documentRoot.querySelector("#fable-status");
    const solNode = documentRoot.querySelector("#sol-status");
    const repoNode = documentRoot.querySelector("#repository-status");
    if (fableNode) {
      setStatus(
        fableNode,
        state.gate.fableReady
          ? "Fable · Subscription · ready"
          : "Fable · Subscription · unavailable",
        state.gate.fableReady ? "ready" : "error",
      );
    }
    if (solNode) {
      const configured = typeof state.solStatus === "string"
        ? state.solStatus
        : asObject(state.solStatus).status;
      const solValue = deriveSolStatus(state.tasks, configured);
      const label = typeof solValue === "string" ? stateLabel(solValue) : "checking";
      setStatus(solNode, `Sol · ${label}`, label === "Ready" ? "ready" : "checking");
    }
    if (repoNode) {
      const project = typeof state.projectLabel === "string" ? state.projectLabel : "checking";
      const branch = typeof state.branch === "string" ? state.branch : "checking";
      repoNode.textContent = `Project: ${project} · Branch: ${branch}`;
    }
    messageInput.disabled = !state.gate.canCompose || state.sessionId === null;
    composerSubmit.disabled = messageInput.disabled;
    guidance.textContent = state.sessionId === null
      ? `${state.gate.guidance} Waiting for the server session identifier.`
      : state.gate.guidance;
  }

  function renderWorkspace() {
    renderProjectNavigation(documentRoot, state, {
      onProject: (projectId) => {
        void controller?.selectProject(projectId).catch((error) => {
          showToast(documentRoot, String(error.message ?? error), true);
        });
      },
      onChat: (sessionId) => {
        if (typeof state.projectId !== "string") {
          return;
        }
        void controller?.selectChat(state.projectId, sessionId).catch((error) => {
          showToast(documentRoot, String(error.message ?? error), true);
        });
      },
      onNewChat: () => {
        void controller?.createChat().catch((error) => {
          showToast(documentRoot, String(error.message ?? error), true);
        });
      },
    });
    renderTaskList(documentRoot, state.tasks, state.selectedTaskId, (taskId) => {
      state.selectedTaskId = taskId;
      renderWorkspace();
      const toggle = documentRoot.querySelector("#task-drawer-toggle");
      if (toggle) {
        closeDrawer(documentRoot, "#task-list", toggle, isMobileDrawer);
      }
    });
    renderTaskInspector(documentRoot, selectedTask(), {
      gate: state.gate,
      onAction: handleTaskAction,
    });
  }

  function syncUsageModal() {
    if (state.gate.acknowledged) {
      if (usageModal.open) {
        if (typeof usageModal.close === "function") {
          usageModal.close();
        } else {
          usageModal.removeAttribute("open");
        }
        focusBeforeModal?.focus();
      }
    } else if (!usageModal.open && typeof usageModal.showModal === "function") {
      usageModal.showModal();
    } else {
      usageModal.setAttribute("open", "");
    }
  }

  function taskActionPayload(action, task, supplied) {
    if (action === "approve") {
      return approvalPayload(taskBrief(task));
    }
    if (action === "answer") {
      return supplied;
    }
    return null;
  }

  async function sendTaskAction(action, task, supplied) {
    if (typeof state.projectId !== "string" || typeof state.sessionId !== "string") {
      throw new Error("project chat selection is unavailable");
    }
    await postJson(
      windowRoot.fetch.bind(windowRoot),
      projectTaskActionPath(state.projectId, state.sessionId, taskIdentity(task), action),
      taskActionPayload(action, task, supplied),
      state.csrfToken,
    );
    showToast(documentRoot, `${stateLabel(action)} accepted.`);
  }

  function handleTaskAction(action, task, supplied) {
    if (action === "edit") {
      renderTaskEditor(
        documentRoot,
        task,
        async (edited) => {
          try {
            if (typeof state.projectId !== "string" || typeof state.sessionId !== "string") {
              throw new Error("project chat selection is unavailable");
            }
            await postJson(
              windowRoot.fetch.bind(windowRoot),
              projectTaskActionPath(state.projectId, state.sessionId, taskIdentity(task), "edit"),
              edited,
              state.csrfToken,
            );
            showToast(documentRoot, `Revision ${edited.revision} submitted for approval.`);
          } catch (error) {
            showToast(documentRoot, String(error.message ?? error), true);
          }
        },
        renderWorkspace,
      );
      return Promise.resolve();
    }
    return sendTaskAction(action, task, supplied).catch((error) => {
      showToast(documentRoot, String(error.message ?? error), true);
    });
  }

  function handleEvent(event, associatedTask) {
    const rendered = renderConversationEvent(
      documentRoot,
      event,
      associatedRevisionForEvent(event, associatedTask),
    );
    const conversation = documentRoot.querySelector("#conversation");
    if (rendered && conversation) {
      conversation.scrollTop = conversation.scrollHeight;
    }
  }

  function updateConnection(status) {
    const connection = documentRoot.querySelector("#connection-status");
    if (!connection) {
      return;
    }
    const labels = {
      connected: ["Connection · live", "ready"],
      reconnecting: ["Connection · reconnecting", "running"],
      connection_error: ["Connection · degraded", "error"],
      bootstrap_error: ["Connection · live · status refresh failed", "error"],
      bootstrap_stale: ["Connection · live · status refresh deferred", "running"],
      bootstrap_refreshing: ["Connection · live · refreshing status", "running"],
      invalid_event: ["Connection · invalid event ignored", "error"],
    };
    const [label, variant] = labels[status] ?? ["Connection · checking", "checking"];
    setStatus(connection, label, variant);
  }

  composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = String(messageInput.value ?? "");
    if (!state.gate.canCompose || !state.projectId || !state.sessionId || !text.trim()) {
      return;
    }
    messageInput.disabled = true;
    composerSubmit.disabled = true;
    try {
      await postJson(
        windowRoot.fetch.bind(windowRoot),
        projectChatMessagePath(state.projectId, state.sessionId),
        {text},
        state.csrfToken,
      );
      messageInput.value = "";
    } catch (error) {
      showToast(documentRoot, String(error.message ?? error), true);
    } finally {
      renderStatus();
    }
  });

  usageCheckbox.addEventListener("change", () => {
    usageSubmit.disabled = !usageCheckbox.checked;
  });
  usageModal.addEventListener("cancel", (event) => event.preventDefault());
  usageForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!usageCheckbox.checked || !state.csrfToken) {
      return;
    }
    usageSubmit.disabled = true;
    usageError.textContent = "";
    try {
      await postJson(
        windowRoot.fetch.bind(windowRoot),
        "/api/settings/usage-credits-acknowledgement",
        {acknowledged: true},
        state.csrfToken,
      );
      const refreshed = await controller.refreshProjects();
      if (!refreshed || !state.gate.acknowledged) {
        throw new Error("usage-credit acknowledgement could not be confirmed");
      }
    } catch (error) {
      usageError.textContent = String(error.message ?? error);
      usageSubmit.disabled = false;
    }
  });

  wireDrawer(
    documentRoot,
    "#task-drawer-toggle",
    "#task-list",
    "#inspector-drawer-toggle",
    "#task-inspector",
    isMobileDrawer,
  );
  wireDrawer(
    documentRoot,
    "#inspector-drawer-toggle",
    "#task-inspector",
    "#task-drawer-toggle",
    "#task-list",
    isMobileDrawer,
  );

  function syncDrawerMode() {
    for (const [panelId, buttonId] of [
      ["#task-list", "#task-drawer-toggle"],
      ["#task-inspector", "#inspector-drawer-toggle"],
    ]) {
      const panel = documentRoot.querySelector(panelId);
      const button = documentRoot.querySelector(buttonId);
      if (!panel || !button) {
        continue;
      }
      if (!isMobileDrawer()) {
        panel.classList.remove("drawer-open");
        button.setAttribute("aria-expanded", "false");
      }
      panel.inert = isMobileDrawer() && !panel.classList.contains("drawer-open");
    }
  }
  syncDrawerMode();
  if (typeof drawerMedia.addEventListener === "function") {
    drawerMedia.addEventListener("change", syncDrawerMode);
  } else if (typeof drawerMedia.addListener === "function") {
    drawerMedia.addListener(syncDrawerMode);
  }

  documentRoot.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    for (const [panelId, buttonId] of [
      ["#task-list", "#task-drawer-toggle"],
      ["#task-inspector", "#inspector-drawer-toggle"],
    ]) {
      const panel = documentRoot.querySelector(panelId);
      const button = documentRoot.querySelector(buttonId);
      if (panel?.classList.contains("drawer-open") && button) {
        event.preventDefault();
        closeDrawer(documentRoot, panelId, button, isMobileDrawer);
        button.focus();
      }
    }
  });

  controller = createProjectChatController({
    fetchFunction: windowRoot.fetch.bind(windowRoot),
    WebSocketCtor: windowRoot.WebSocket,
    schedule: windowRoot.setTimeout.bind(windowRoot),
    cancelSchedule: windowRoot.clearTimeout.bind(windowRoot),
    location: windowRoot.location,
    onState(nextState) {
      state = nextState;
      renderStatus();
      renderWorkspace();
      syncUsageModal();
    },
    onEvent: handleEvent,
    onStatus: updateConnection,
    onChatReset() {
      const conversation = documentRoot.querySelector("#conversation");
      conversation?.replaceChildren();
    },
  });

  function attemptBootstrap() {
    bootstrapRetry.hidden = true;
    usageError.textContent = "";
    ready = controller.bootstrapInitial()
      .then((bootstrapped) => {
        if (!bootstrapped) {
          throw new Error("No project chats are available.");
        }
        return true;
      })
      .catch((error) => {
        bootstrapRetry.hidden = false;
        updateConnection("connection_error");
        renderStatus();
        const message = String(error.message ?? error);
        usageError.textContent = message;
        showToast(documentRoot, message, true);
        return false;
      });
    return ready;
  }

  bootstrapRetry.addEventListener("click", () => attemptBootstrap());
  attemptBootstrap();

  windowRoot.addEventListener("beforeunload", () => controller?.stop());
  return {
    get ready() {
      return ready;
    },
    get state() {
      return state;
    },
    retry: attemptBootstrap,
    stop() {
      controller?.stop();
    },
  };
}


if (typeof document !== "undefined" && typeof window !== "undefined") {
  startBrowserApp(document, window);
}
