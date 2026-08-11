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
  const intro = documentRoot.querySelector("#conversation-empty");
  if (intro && typeof intro.remove === "function") {
    intro.remove();
  }
  conversation.append(article);
  if (conversation.children.length > MAX_CONVERSATION_MESSAGES) {
    conversation.removeChild(conversation.children[0]);
  }
  return article;
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
    ["Approve & run", "approve", hasUnusableBrief ? hiddenControl : controls.approve, "button-primary"],
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


export function createEventStream({
  sessionId,
  initialSequence = 0,
  WebSocketCtor,
  schedule,
  cancelSchedule,
  location,
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
    const ownSocket = new WebSocketCtor(websocketUrl(sessionId, lastSequence, location));
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


export function applyBootstrap(previousState, bootstrapData) {
  const previous = asObject(previousState);
  const bootstrap = asObject(bootstrapData);
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
    sessionId: null,
    tasks: [],
    selectedTaskId: null,
    gate: subscriptionGate({}),
    lastSequence: 0,
  };
  let stream = null;
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
      const repository = typeof state.repository === "string" ? state.repository : "Repository";
      const branch = typeof state.branch === "string" ? ` · ${state.branch}` : "";
      repoNode.textContent = `${repository}${branch}`;
    }
    messageInput.disabled = !state.gate.canCompose || state.sessionId === null;
    composerSubmit.disabled = messageInput.disabled;
    guidance.textContent = state.sessionId === null
      ? `${state.gate.guidance} Waiting for the server session identifier.`
      : state.gate.guidance;
  }

  function renderWorkspace() {
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

  async function loadBootstrap(isCurrent = () => true) {
    const response = await windowRoot.fetch("/api/bootstrap", {
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`Bootstrap failed with status ${response.status}`);
    }
    const bootstrapData = await response.json();
    if (!isCurrent()) {
      return false;
    }
    state = applyBootstrap(state, bootstrapData);
    renderStatus();
    renderWorkspace();
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
    return true;
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
    await postJson(
      windowRoot.fetch.bind(windowRoot),
      taskActionPath(taskIdentity(task), action),
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
            await postJson(
              windowRoot.fetch.bind(windowRoot),
              taskActionPath(taskIdentity(task), "edit"),
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

  function handleEvent(event) {
    const previousTasks = state.tasks;
    const previousSelection = state.selectedTaskId;
    state.tasks = reduceTaskEvent(state.tasks, event);
    state.selectedTaskId = repairTaskSelection(state.tasks, state.selectedTaskId);
    const associatedTask = state.tasks.find((task) => taskIdentity(task) === event?.task_id);
    renderMessage(documentRoot, event, associatedRevisionForEvent(event, associatedTask));
    if (state.tasks !== previousTasks || state.selectedTaskId !== previousSelection) {
      renderWorkspace();
      renderStatus();
    }
    const conversation = documentRoot.querySelector("#conversation");
    if (conversation) {
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
    if (["bootstrap_refreshing", "bootstrap_stale", "bootstrap_error"].includes(status)) {
      state = {...state, gate: subscriptionGate({})};
      renderStatus();
      renderWorkspace();
    }
    const [label, variant] = labels[status] ?? ["Connection · checking", "checking"];
    setStatus(connection, label, variant);
  }

  function beginStream() {
    if (!state.sessionId || stream) {
      return;
    }
    stream = createEventStream({
      sessionId: state.sessionId,
      initialSequence: state.lastSequence,
      WebSocketCtor: windowRoot.WebSocket,
      schedule: windowRoot.setTimeout.bind(windowRoot),
      cancelSchedule: windowRoot.clearTimeout.bind(windowRoot),
      location: windowRoot.location,
      bootstrap: loadBootstrap,
      onEvent: handleEvent,
      onStatus: updateConnection,
    });
    stream.connect();
  }

  composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = String(messageInput.value ?? "");
    if (!state.gate.canCompose || !state.sessionId || !text.trim()) {
      return;
    }
    messageInput.disabled = true;
    composerSubmit.disabled = true;
    try {
      await postJson(
        windowRoot.fetch.bind(windowRoot),
        sessionMessagePath(state.sessionId),
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
      await loadBootstrap();
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

  function attemptBootstrap() {
    bootstrapRetry.hidden = true;
    usageError.textContent = "";
    ready = loadBootstrap()
      .then(() => {
        beginStream();
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

  windowRoot.addEventListener("beforeunload", () => stream?.stop());
  return {
    get ready() {
      return ready;
    },
    get state() {
      return state;
    },
    retry: attemptBootstrap,
    stop() {
      stream?.stop();
    },
  };
}


if (typeof document !== "undefined" && typeof window !== "undefined") {
  startBrowserApp(document, window);
}
