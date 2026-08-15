const ACTOR_LABELS = Object.freeze({
  user: "You",
  fable: "Fable",
  sol: "Sol",
  coordinator: "Coordinator",
});

const DIRECTED_ACTORS = Object.freeze({
  user: Object.freeze({label: "User", avatar: "U", className: "user"}),
  fable: Object.freeze({label: "Fable", avatar: "F", className: "fable"}),
  sol: Object.freeze({label: "Sol", avatar: "S", className: "sol"}),
  system: Object.freeze({label: "System", avatar: "•", className: "coordinator"}),
});

const DIRECTED_TARGETS = Object.freeze({
  user: Object.freeze({label: "You", className: "user"}),
  fable: Object.freeze({label: "Fable", className: "fable"}),
  sol: Object.freeze({label: "Sol", className: "sol"}),
  team: Object.freeze({label: "Team", className: "coordinator"}),
});

const DIRECTED_MESSAGE_TYPES = new Set([
  "statement",
  "question",
  "answer",
  "approval",
  "intervention",
  "status",
]);

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
  "intervene",
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


function safeConversationText(value) {
  if (typeof value !== "string" || /^\p{White_Space}*$/u.test(value)) {
    return false;
  }
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code <= 0x1f || code === 0x7f) {
      return false;
    }
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        return false;
      }
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return false;
    }
  }
  const TextEncoderCtor = globalThis.TextEncoder;
  if (typeof TextEncoderCtor !== "function") {
    return false;
  }
  try {
    const encoded = new TextEncoderCtor().encode(value);
    return encoded instanceof Uint8Array && encoded.length <= 16 * 1024;
  } catch (_error) {
    return false;
  }
}


export function conversationPresentation(event) {
  if (event?.kind === "conversation") {
    const envelope = directedEnvelope(event);
    return envelope === null ? "hidden" : (envelope.messageType === "status" ? "status" : "message");
  }
  if (event?.kind === "task_state") return "status";
  if (MESSAGE_EVENT_KINDS.has(event?.kind)) return "message";
  return "hidden";
}


function directedEnvelope(event) {
  const payload = asObject(event?.payload);
  const envelopeFields = [
    "sender", "addressed_to", "routed_to", "message_type", "text", "task_id",
    "revision", "continuation_generation", "question_id", "reply_to_question_id",
  ];
  const sender = typeof payload.sender === "string" ? DIRECTED_ACTORS[payload.sender] : null;
  const addressedTo = typeof payload.addressed_to === "string"
    ? DIRECTED_TARGETS[payload.addressed_to]
    : null;
  const routedTo = typeof payload.routed_to === "string"
    ? DIRECTED_TARGETS[payload.routed_to]
    : null;
  if (
    event?.kind !== "conversation"
    || Object.keys(payload).length !== envelopeFields.length
    || !envelopeFields.every((field) => Object.hasOwn(payload, field))
    || sender === undefined || sender === null
    || addressedTo === undefined || addressedTo === null
    || routedTo === undefined || routedTo === null
    || !DIRECTED_MESSAGE_TYPES.has(payload.message_type)
    || typeof payload.text !== "string"
    || typeof event?.actor !== "string"
    || event.actor !== payload.sender
  ) {
    return null;
  }
  const taskId = typeof payload.task_id === "string" && SAFE_ID.test(payload.task_id)
    ? payload.task_id
    : null;
  const outerTaskId = event?.task_id === null
    ? null
    : (typeof event?.task_id === "string" && SAFE_ID.test(event.task_id)
      ? event.task_id
      : undefined);
  if (outerTaskId === undefined) {
    return null;
  }
  const revision = Number.isInteger(payload.revision) && payload.revision >= 1
    ? payload.revision
    : null;
  const generation = Number.isInteger(payload.continuation_generation)
    && payload.continuation_generation >= 1
    ? payload.continuation_generation
    : null;
  const questionId = typeof payload.question_id === "string" && SAFE_ID.test(payload.question_id)
    ? payload.question_id
    : null;
  const replyToQuestionId = typeof payload.reply_to_question_id === "string"
    && SAFE_ID.test(payload.reply_to_question_id)
    ? payload.reply_to_question_id
    : null;
  const textIsSafe = safeConversationText(payload.text);
  const hasUnboundFields = payload.task_id === null
    && payload.revision === null
    && payload.continuation_generation === null;
  const hasBoundFields = taskId !== null && revision !== null && generation !== null;
  const hasApprovalBinding = taskId !== null && revision !== null
    && payload.continuation_generation === null;
  const hasBoundBinding = payload.message_type === "approval"
    ? hasApprovalBinding
    : hasBoundFields;
  if (
    (payload.question_id !== undefined && payload.question_id !== null && questionId === null)
    || (payload.reply_to_question_id !== undefined && payload.reply_to_question_id !== null && replyToQuestionId === null)
    || !textIsSafe
    || (payload.message_type === "question" && (questionId === null || replyToQuestionId !== null))
    || (payload.message_type === "answer" && (questionId !== null || replyToQuestionId === null))
    || (!["question", "answer"].includes(payload.message_type)
      && (questionId !== null || replyToQuestionId !== null))
    || (payload.message_type === "approval" && !hasApprovalBinding)
    || (payload.message_type !== "approval" && !hasUnboundFields && !hasBoundFields)
    || ((payload.message_type === "question" || payload.message_type === "answer") && !hasBoundFields)
    || (hasBoundBinding && outerTaskId !== taskId)
    || (payload.message_type === "status" && payload.sender !== "system")
  ) {
    return null;
  }
  return Object.freeze({
    sender,
    addressedTo,
    routedTo,
    messageType: payload.message_type,
    text: payload.text,
    taskId,
    revision,
    generation,
    questionId,
    replyToQuestionId,
  });
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


function renderDirectedMessage(documentRoot, event, envelope) {
  const conversation = documentRoot.querySelector("#conversation");
  if (!conversation) {
    throw new Error("conversation region is missing");
  }
  const article = element(
    documentRoot,
    "article",
    undefined,
    `message message-${envelope.sender.className} target-${envelope.routedTo.className}`,
  );
  article.setAttribute("aria-label", `${envelope.sender.label} to ${envelope.routedTo.label}`);
  const avatar = element(documentRoot, "span", envelope.sender.avatar, "message-avatar");
  avatar.setAttribute("aria-hidden", "true");
  const heading = element(
    documentRoot,
    "strong",
    `${envelope.sender.label} → ${envelope.routedTo.label}`,
    "message-route",
  );
  const text = element(documentRoot, "p", envelope.text);
  const context = [];
  if (envelope.taskId !== null) {
    context.push(`Task ${envelope.taskId}`);
  }
  if (envelope.revision !== null) {
    context.push(`r${envelope.revision}`);
  }
  if (envelope.questionId !== null) {
    context.push(`Question ${envelope.questionId}`);
  }
  if (envelope.replyToQuestionId !== null) {
    context.push(`Reply to question ${envelope.replyToQuestionId}`);
  }
  if (envelope.generation !== null) {
    context.push(`generation ${envelope.generation}`);
  }
  if (envelope.addressedTo !== envelope.routedTo) {
    context.push(`Addressed to ${envelope.addressedTo.label} · routed to ${envelope.routedTo.label} before approval`);
  }
  const metadata = element(
    documentRoot,
    "p",
    context.length > 0 ? context.join(" · ") : "Conversation message",
    "message-metadata",
  );
  article.append(avatar, heading, text, metadata);
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
  if (event?.kind === "conversation") {
    const envelope = directedEnvelope(event);
    if (envelope === null) return null;
    if (envelope.messageType === "status") {
      return renderConversationStatus(documentRoot, event, associatedRevision);
    }
    return renderDirectedMessage(documentRoot, event, envelope);
  }
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


function activitySummary(task) {
  const activity = asObject(task?.activity);
  const allowed = new Set(["action_error", "stop_error", "agent_event", "resume_drift"]);
  if (typeof task?.activity_kind !== "string" || !allowed.has(task.activity_kind)) {
    return "No structured activity recorded";
  }
  const agentStatuses = new Set(["completed", "failed", "running", "pending", "success", "error"]);
  return [
    stateLabel(task.activity_kind),
    typeof activity.status === "string" && agentStatuses.has(activity.status)
      ? stateLabel(activity.status) : null,
    typeof activity.command_sha256 === "string" && /^[a-f0-9]{64}$/.test(activity.command_sha256)
      ? activity.command_sha256 : null,
  ].filter(Boolean).join(" · ");
}


function activityRows(task) {
  const activity = asObject(task?.activity);
  const allowed = new Set(["action_error", "stop_error", "agent_event", "resume_drift"]);
  const agentStatuses = new Set(["completed", "failed", "running", "pending", "success", "error"]);
  if (typeof task?.activity_kind !== "string" || !allowed.has(task.activity_kind)) {
    return [["Type", "No structured activity recorded"]];
  }
  const type = stateLabel(task.activity_kind);
  const rows = [["Type", type]];
  if (typeof activity.status === "string" && agentStatuses.has(activity.status)) {
    rows.push(["Status", stateLabel(activity.status)]);
  }
  if (typeof activity.command_sha256 === "string" && /^[a-f0-9]{64}$/.test(activity.command_sha256)) {
    rows.push(["Command digest", activity.command_sha256]);
  }
  return rows;
}


function renderActivityAudit(documentRoot, task) {
  const audit = documentRoot.querySelector("#activity-audit");
  if (!audit || audit.open !== true) {
    return;
  }
  const rows = element(documentRoot, "dl", undefined, "activity-audit-rows");
  rows.append(...activityRows(task).flatMap(([label, value]) => [
    element(documentRoot, "dt", label), element(documentRoot, "dd", value),
  ]));
  audit.replaceChildren(
    element(documentRoot, "summary", "Activity and audit"),
    element(documentRoot, "p", activitySummary(task)),
    rows,
  );
}


function boundedInspectorList(value) {
  if (!Array.isArray(value)) {
    return "None recorded";
  }
  const values = value
    .filter((entry) => typeof entry === "string" && entry.trim())
    .slice(0, 20);
  if (values.length === 0) {
    return "None recorded";
  }
  return values.join(" · ");
}


function renderPersistentInspector(documentRoot, task, options) {
  const summary = documentRoot.querySelector("#task-inspector-summary");
  const controls = documentRoot.querySelector("#task-controls");
  const empty = documentRoot.querySelector("#task-inspector-empty");
  if (!task) {
    if (empty) {
      empty.hidden = false;
    }
    if (summary) {
      const rows = [
        ["Task revision", "Awaiting task selection"],
        ["State", "Awaiting task selection"],
        ["Scope", "Awaiting task selection"],
        ["Allowed paths", "Awaiting task selection"],
        ["Required tests", "Awaiting task selection"],
        ["Question budget", "Awaiting task selection"],
      ];
      summary.replaceChildren(...rows.flatMap(([label, value]) => [
        element(documentRoot, "dt", label),
        element(documentRoot, "dd", value),
      ]));
    }
    if (controls) {
      controls.hidden = false;
      const heading = element(documentRoot, "h3", "Task controls");
      heading.id = "task-controls-heading";
      controls.replaceChildren(
        heading,
        element(documentRoot, "p", "Task-specific controls appear after task selection."),
      );
    }
    return;
  }
  if (empty) {
    empty.hidden = true;
  }
  const brief = taskBrief(task);
  const identity = taskIdentity(task);
  const questionBudget = Number.isInteger(task?.exchange_allowance)
    && task.exchange_allowance >= 0
    ? `${task.exchange_allowance} remaining${Number.isInteger(task.exchange_consumed) && task.exchange_consumed >= 0 ? ` · ${task.exchange_consumed} consumed` : ""}`
    : "Not recorded";
  if (summary) {
    const rows = [
      ["Task revision", `${identity} · r${taskRevision(task)}`],
      ["State", stateLabel(String(task?.state ?? "unknown"))],
      ["Scope", brief?.objective ?? "None recorded"],
      ["Allowed paths", boundedInspectorList(brief?.allowed_paths)],
      ["Required tests", boundedInspectorList(brief?.required_tests)],
      ["Question budget", questionBudget],
    ];
    summary.replaceChildren(...rows.flatMap(([label, value]) => [
      element(documentRoot, "dt", label),
      element(documentRoot, "dd", value),
    ]));
  }
  if (!controls) {
    return;
  }
  controls.hidden = true;
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


export function projectTaskMessagePath(projectId, sessionId, taskId) {
  requireSafeId(projectId, "project_id");
  requireSafeId(sessionId, "session_id");
  requireSafeId(taskId, "task_id");
  return `/api/projects/${encodeURIComponent(projectId)}/chats/${encodeURIComponent(sessionId)}/tasks/${encodeURIComponent(taskId)}/messages`;
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


export function projectInterventionPath(projectId, sessionId, interventionId, action) {
  requireSafeId(projectId, "project_id");
  requireSafeId(sessionId, "session_id");
  requireSafeId(interventionId, "intervention_id");
  if (action !== "resume" && action !== "authorize-retry") {
    throw new Error("intervention action is unavailable");
  }
  return `/api/projects/${encodeURIComponent(projectId)}/chats/${encodeURIComponent(sessionId)}/interventions/${encodeURIComponent(interventionId)}/${action}`;
}


function composerIdentity(state) {
  const current = asObject(state);
  return {
    projectId: requireSafeId(current.projectId, "project_id"),
    sessionId: requireSafeId(current.sessionId, "session_id"),
  };
}


function userRecipient(value, allowTeam = true) {
  if (value !== "fable" && value !== "sol" && (value !== "team" || !allowTeam)) {
    throw new Error("recipient is unavailable");
  }
  return value;
}


function exactBinding(state, binding) {
  const current = composerIdentity(state);
  const candidate = asObject(binding);
  if (
    candidate.projectId !== current.projectId
    || candidate.sessionId !== current.sessionId
    || typeof candidate.taskId !== "string"
    || !SAFE_ID.test(candidate.taskId)
    || !Number.isInteger(candidate.revision)
    || candidate.revision < 1
    || !Number.isInteger(candidate.continuationGeneration)
    || candidate.continuationGeneration < 1
  ) {
    throw new Error("bound reply is stale");
  }
  return {current, candidate};
}


export function composerRequest(state, binding, text, recipient = "fable") {
  if (typeof text !== "string" || text.trim().length === 0) {
    throw new Error("message text is required");
  }
  if (binding === null || binding === undefined) {
    const current = composerIdentity(state);
    return {
      path: projectChatMessagePath(current.projectId, current.sessionId),
      payload: {text, addressed_to: userRecipient(recipient)},
    };
  }
  const {current, candidate} = exactBinding(state, binding);
  if (candidate.kind === "question") {
    requireSafeId(candidate.questionId, "question_id");
    return {
      path: projectTaskActionPath(current.projectId, current.sessionId, candidate.taskId, "answer"),
      payload: {
        text,
        revision: candidate.revision,
        question_id: candidate.questionId,
        continuation_generation: candidate.continuationGeneration,
      },
    };
  }
  if (candidate.kind === "continuation") {
    return {
      path: projectTaskMessagePath(current.projectId, current.sessionId, candidate.taskId),
      payload: {
        text,
        addressed_to: userRecipient(recipient, false),
        revision: candidate.revision,
        continuation_generation: candidate.continuationGeneration,
      },
    };
  }
  throw new Error("bound reply is unavailable");
}


export function exchangeGrantRequest(state, binding) {
  const {current, candidate} = exactBinding(state, binding);
  if (candidate.kind !== "exchange_permission") {
    throw new Error("exchange permission is unavailable");
  }
  requireSafeId(candidate.requestId, "request_id");
  return {
    path: `/api/projects/${encodeURIComponent(current.projectId)}/chats/${encodeURIComponent(current.sessionId)}/tasks/${encodeURIComponent(candidate.taskId)}/exchanges/grant`,
    payload: {
      revision: candidate.revision,
      continuation_generation: candidate.continuationGeneration,
      request_id: candidate.requestId,
    },
  };
}


export function composerGuidance(state, binding, recipient = "fable") {
  const current = asObject(state);
  const gate = asObject(current.gate);
  const readiness = typeof gate.guidance === "string" ? gate.guidance : "Actions are unavailable.";
  if (binding !== null && binding !== undefined) {
    return `${readiness} This reply remains bound to its exact task and continuation.`;
  }
  const route = recipient === "sol"
    ? "Before approval, messages addressed to Sol are visibly routed through Fable."
    : (recipient === "team"
      ? "Before approval, messages addressed to Team are visibly routed through Fable."
      : "Fable is the direct planner for new tasks.");
  if (current.activeLease !== null && current.activeLease !== undefined) {
    return `An agent is active. Select an exact Reply card to answer its bound question. ${route}`;
  }
  return `${readiness} ${route}`;
}


export function composerPresentation(state, binding, recipient = "fable") {
  const current = asObject(state);
  const gate = asObject(current.gate);
  const boundReply = binding !== null && binding !== undefined;
  const ordinaryLeaseLocked = current.activeLease !== null
    && current.activeLease !== undefined
    && !boundReply;
  const selectedRecipient = recipient === "sol"
    ? "Sol"
    : (recipient === "team" ? "Team" : "Fable");
  const disabled = gate.canCompose !== true
    || current.sessionId === null
    || current.sessionId === undefined
    || ordinaryLeaseLocked;
  return Object.freeze({
    disabled,
    recipientDisabled: disabled || binding?.kind === "question",
    label: boundReply ? "Bound reply" : `Message ${selectedRecipient}`,
    submit: boundReply ? "Send reply" : "Send",
    guidance: composerGuidance(current, binding, recipient),
  });
}


function interventionIdentity(task) {
  const candidate = asObject(task);
  if (
    typeof candidate.task_id !== "string"
    || !SAFE_ID.test(candidate.task_id)
    || !Number.isInteger(candidate.revision)
    || candidate.revision < 1
    || !Number.isInteger(candidate.continuation_generation)
    || candidate.continuation_generation < 1
  ) {
    throw new Error("active task identity is unavailable");
  }
  return candidate;
}


export function interventionRecipients(task) {
  const active = interventionIdentity(task);
  return Object.freeze(["sol_running", "sol_correcting"].includes(active.state)
    ? ["fable", "sol"]
    : ["fable"]);
}


export function interventionDraft(task, interventionId, addressedTo, message) {
  const active = interventionIdentity(task);
  requireSafeId(interventionId, "intervention_id");
  if (!interventionRecipients(active).includes(addressedTo)) {
    throw new Error("intervention recipient is unavailable");
  }
  if (typeof message !== "string" || !message.trim() || message.length > 16 * 1024) {
    throw new Error("intervention message is required");
  }
  return Object.freeze({
    taskId: active.task_id,
    revision: active.revision,
    sourceGeneration: active.continuation_generation,
    interventionId,
    addressedTo,
    message,
    submitted: false,
  });
}


export function interventionWarningKey(intervention) {
  const record = asObject(intervention);
  if (
    typeof record.intervention_id !== "string"
    || !SAFE_ID.test(record.intervention_id)
    || !Number.isInteger(record.resume_generation)
    || record.resume_generation < 1
  ) {
    throw new Error("intervention warning identity is unavailable");
  }
  return `${record.intervention_id}:${record.resume_generation}`;
}


export function interventionRequest(state, task, message, interventionId, recipient = "fable") {
  const current = composerIdentity(state);
  const active = interventionIdentity(task);
  requireSafeId(interventionId, "intervention_id");
  if (!interventionRecipients(active).includes(recipient)) {
    throw new Error("intervention recipient is unavailable");
  }
  if (typeof message !== "string" || !message.trim() || message.length > 16 * 1024) {
    throw new Error("intervention message is required");
  }
  return Object.freeze({
    path: projectTaskActionPath(current.projectId, current.sessionId, active.task_id, "intervene"),
    payload: Object.freeze({
      intervention_id: interventionId,
      message,
      addressed_to: recipient,
      revision: active.revision,
      continuation_generation: active.continuation_generation,
    }),
  });
}


export function interventionPresentation(state, task) {
  const current = composerIdentity(state);
  const active = interventionIdentity(task);
  const record = asObject(active.intervention);
  if (Object.keys(record).length === 0) {
    return Object.freeze({kind: "new", submit: "Intervene", warning: null});
  }
  if (
    typeof record.intervention_id !== "string"
    || !SAFE_ID.test(record.intervention_id)
    || !Number.isInteger(record.resume_generation)
    || record.resume_generation < 1
  ) {
    return Object.freeze({kind: "unavailable", submit: "Intervention unavailable", warning: null});
  }
  if (record.status === "pending_stop" || record.status === "resuming") {
    return Object.freeze({kind: "pending", submit: "Intervention pending", warning: null});
  }
  if (record.status === "canceled_by_stop") {
    return Object.freeze({kind: "canceled", submit: "Intervention canceled by Stop", warning: null});
  }
  if (record.status === "ready" && record.eligible === true) {
    return Object.freeze({
      kind: "resume",
      submit: "Resume intervention",
      warning: null,
      interventionId: record.intervention_id,
      resumeGeneration: record.resume_generation,
      path: projectInterventionPath(current.projectId, current.sessionId, record.intervention_id, "resume"),
      payload: Object.freeze({expected_resume_generation: record.resume_generation}),
    });
  }
  if (record.status === "resume_outcome_unknown") {
    const warning = typeof record.warning === "string" && record.warning
      ? record.warning
      : "The prior resume outcome is unknown and may have executed.";
    return Object.freeze({
      kind: "unknown",
      submit: "Acknowledge possible prior execution",
      warning,
      interventionId: record.intervention_id,
      resumeGeneration: record.resume_generation,
    });
  }
  return Object.freeze({kind: "unavailable", submit: "Intervention unavailable", warning: null});
}


export function interventionRetryRequest(state, presentation, acknowledgmentId) {
  const current = composerIdentity(state);
  const unknown = asObject(presentation);
  if (
    unknown.kind !== "unknown"
    || typeof unknown.interventionId !== "string"
    || !SAFE_ID.test(unknown.interventionId)
    || !Number.isInteger(unknown.resumeGeneration)
    || unknown.resumeGeneration < 1
  ) {
    throw new Error("unknown intervention acknowledgement is unavailable");
  }
  requireSafeId(acknowledgmentId, "acknowledgment_id");
  return Object.freeze({
    path: projectInterventionPath(current.projectId, current.sessionId, unknown.interventionId, "authorize-retry"),
    payload: Object.freeze({
      expected_resume_generation: unknown.resumeGeneration,
      acknowledgment_id: acknowledgmentId,
      acknowledge_possible_prior_execution: true,
    }),
  });
}


function pendingQuestionBinding(state, task) {
  const snapshot = asObject(task);
  const question = asObject(snapshot.pending_question);
  const identity = composerIdentity(state);
  if (
    typeof snapshot.task_id !== "string"
    || !SAFE_ID.test(snapshot.task_id)
    || !Number.isInteger(snapshot.revision)
    || snapshot.revision < 1
    || !Number.isInteger(snapshot.continuation_generation)
    || snapshot.continuation_generation < 1
    || typeof question.question_id !== "string"
    || !SAFE_ID.test(question.question_id)
    || question.addressed_to !== "user"
    || question.routed_to !== "user"
    || !["fable", "sol"].includes(question.asked_by)
    || typeof question.text !== "string"
    || question.revision !== snapshot.revision
    || question.continuation_generation !== snapshot.continuation_generation
  ) {
    return null;
  }
  return Object.freeze({
    kind: "question",
    ...identity,
    taskId: snapshot.task_id,
    revision: snapshot.revision,
    questionId: question.question_id,
    continuationGeneration: snapshot.continuation_generation,
    askedBy: question.asked_by,
    text: question.text,
  });
}


function exchangePermissionBinding(state, task) {
  const snapshot = asObject(task);
  const permission = asObject(snapshot.exchange_permission);
  const identity = composerIdentity(state);
  if (
    typeof snapshot.task_id !== "string"
    || !SAFE_ID.test(snapshot.task_id)
    || !Number.isInteger(snapshot.revision)
    || snapshot.revision < 1
    || !Number.isInteger(snapshot.continuation_generation)
    || snapshot.continuation_generation < 1
    || typeof permission.request_id !== "string"
    || !SAFE_ID.test(permission.request_id)
    || permission.revision !== snapshot.revision
    || permission.continuation_generation !== snapshot.continuation_generation
  ) {
    return null;
  }
  return Object.freeze({
    kind: "exchange_permission",
    ...identity,
    taskId: snapshot.task_id,
    revision: snapshot.revision,
    continuationGeneration: snapshot.continuation_generation,
    requestId: permission.request_id,
  });
}


function clearConversationActionCards(conversation) {
  for (const node of Array.from(conversation?.children ?? [])) {
    if (node.className !== "conversation-action-card") {
      continue;
    }
    if (typeof node.remove === "function") {
      node.remove();
    } else if (typeof conversation.removeChild === "function") {
      conversation.removeChild(node);
    }
  }
}


export function renderPendingConversationCards(documentRoot, tasks, handlers = {}, state = {}) {
  const conversation = documentRoot.querySelector("#conversation");
  if (!conversation) {
    throw new Error("conversation region is missing");
  }
  clearConversationActionCards(conversation);
  for (const task of Array.isArray(tasks) ? tasks : []) {
    const question = pendingQuestionBinding(state, task);
    if (question !== null) {
      const card = element(documentRoot, "article", undefined, "conversation-action-card");
      card.setAttribute("aria-label", `${actorLabel(question.askedBy)} question for you`);
      card.append(
        element(documentRoot, "strong", `${actorLabel(question.askedBy)} → You`, "message-route"),
        element(documentRoot, "p", question.text),
        element(documentRoot, "p", `Task ${question.taskId} · r${question.revision} · question ${question.questionId} · generation ${question.continuationGeneration}`, "message-metadata"),
      );
      const reply = element(documentRoot, "button", "Reply", "button button-primary");
      reply.type = "button";
      reply.addEventListener("click", () => handlers.onReply?.(question));
      card.append(reply);
      appendConversationNode(documentRoot, conversation, card);
    }
    const permission = exchangePermissionBinding(state, task);
    if (permission !== null) {
      const card = element(documentRoot, "article", undefined, "conversation-action-card");
      card.setAttribute("aria-label", "Exchange permission for you");
      card.append(
        element(documentRoot, "strong", "System → You", "message-route"),
        element(documentRoot, "p", "Automatic exchange limit reached. Your direction is needed."),
        element(documentRoot, "p", `Task ${permission.taskId} · r${permission.revision} · generation ${permission.continuationGeneration}`, "message-metadata"),
      );
      const reply = element(documentRoot, "button", "Reply", "button");
      reply.type = "button";
      reply.addEventListener("click", () => handlers.onReply?.(Object.freeze({
        kind: "continuation",
        projectId: permission.projectId,
        sessionId: permission.sessionId,
        taskId: permission.taskId,
        revision: permission.revision,
        continuationGeneration: permission.continuationGeneration,
      })));
      const grant = element(documentRoot, "button", "Allow 3 more exchanges", "button button-primary");
      grant.type = "button";
      grant.addEventListener("click", () => handlers.onGrant?.(permission));
      card.append(reply, grant);
      appendConversationNode(documentRoot, conversation, card);
    }
  }
  return conversation;
}


export class HttpError extends Error {
  constructor(status, detail) {
    super(detail);
    this.name = "HttpError";
    this.status = status;
  }
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
    throw new HttpError(response.status, detail);
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
  let selectedBootstrapRefreshTimer = null;
  let selectedBootstrapRefreshInFlight = null;
  let selectedBootstrapRefreshPending = false;
  let selectedBootstrapRefreshGeneration = 0;

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
    invalidateSelectedBootstrapRefresh();
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

  function invalidateSelectedBootstrapRefresh() {
    selectedBootstrapRefreshGeneration += 1;
    if (selectedBootstrapRefreshTimer !== null) {
      cancelSchedule(selectedBootstrapRefreshTimer);
      selectedBootstrapRefreshTimer = null;
    }
    selectedBootstrapRefreshInFlight = null;
    selectedBootstrapRefreshPending = false;
  }

  function scheduleSelectedBootstrapRefresh(projectId, sessionId, generation) {
    if (selectedBootstrapRefreshInFlight !== null) {
      selectedBootstrapRefreshPending = true;
      return;
    }
    if (selectedBootstrapRefreshTimer !== null) {
      return;
    }
    const refreshGeneration = selectedBootstrapRefreshGeneration;
    selectedBootstrapRefreshTimer = schedule(() => {
      selectedBootstrapRefreshTimer = null;
      if (
        refreshGeneration !== selectedBootstrapRefreshGeneration
        || !selected(projectId, sessionId, generation)
      ) {
        return;
      }
      void requestSelectedBootstrapRefresh(
        projectId,
        sessionId,
        generation,
        false,
      ).catch(() => {});
    }, BOOTSTRAP_REFRESH_DELAY);
  }

  function requestSelectedBootstrapRefresh(projectId, sessionId, generation, explicit) {
    if (!selected(projectId, sessionId, generation)) {
      return Promise.resolve(false);
    }
    if (selectedBootstrapRefreshInFlight !== null || selectedBootstrapRefreshTimer !== null) {
      if (explicit || selectedBootstrapRefreshInFlight !== null) {
        selectedBootstrapRefreshPending = true;
      }
      return Promise.resolve(false);
    }
    return refreshSelectedBootstrapSingleFlight(
      projectId,
      sessionId,
      generation,
      selectedBootstrapRefreshGeneration,
    );
  }

  async function refreshSelectedBootstrapSingleFlight(
    projectId, sessionId, generation, refreshGeneration,
  ) {
    if (
      refreshGeneration !== selectedBootstrapRefreshGeneration
      || !selected(projectId, sessionId, generation)
      || selectedBootstrapRefreshInFlight !== null
    ) {
      return false;
    }
    const cursor = state.lastSequence;
    const flight = Object.freeze({projectId, sessionId, generation, refreshGeneration, cursor});
    selectedBootstrapRefreshInFlight = flight;
    let shouldFollowUp = false;
    try {
      const bootstrapData = await getJson(projectChatBootstrapPath(projectId, sessionId));
      const stillSelected = refreshGeneration === selectedBootstrapRefreshGeneration
        && selected(projectId, sessionId, generation);
      if (!stillSelected) {
        return false;
      }
      if (state.lastSequence !== cursor) {
        shouldFollowUp = true;
        return false;
      }
      const applied = applyBootstrap(state, bootstrapData);
      if (applied.projectId !== projectId || applied.sessionId !== sessionId) {
        throw new Error("bootstrap selection did not match the requested project chat");
      }
      publish({
        ...applied,
        projectLabel: projectLabelFor(applied, projectId),
        navigationPending: navigationPending(),
      });
      startStream(projectId, sessionId);
      return true;
    } finally {
      if (selectedBootstrapRefreshInFlight !== flight) {
        return;
      }
      selectedBootstrapRefreshInFlight = null;
      if (shouldFollowUp || selectedBootstrapRefreshPending) {
        selectedBootstrapRefreshPending = false;
        if (
          refreshGeneration === selectedBootstrapRefreshGeneration
          && selected(projectId, sessionId, generation)
        ) {
          scheduleSelectedBootstrapRefresh(projectId, sessionId, generation);
        }
      }
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
    const streamSelectionGeneration = selectionGeneration;
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
        streamSelectionGeneration,
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
        if (selectedBootstrapRefreshInFlight !== null) {
          selectedBootstrapRefreshPending = true;
        }
        const taskState = asObject(event?.payload).state;
        if (
          event?.kind === "conversation"
          || (event?.kind === "task_state"
            && ["awaiting_user_input", "awaiting_scope_approval"].includes(taskState))
        ) {
          scheduleSelectedBootstrapRefresh(projectId, sessionId, streamSelectionGeneration);
        }
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
    return requestSelectedBootstrapRefresh(
      state.projectId,
      state.sessionId,
      selectionGeneration,
      true,
    );
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
    updated = {...updated, activity_kind: event.kind, activity: payload};
  }
  return upsertTask(tasks, updated);
}


function setStatus(node, text, variant) {
  node.textContent = text;
  node.className = `status-pill status-${variant}`;
}


function activeTaskForControls(state) {
  const leaseTaskId = typeof state?.activeLease?.taskId === "string"
    ? state.activeLease.taskId
    : null;
  const tasks = Array.isArray(state?.tasks) ? state.tasks : [];
  return tasks.find((task) => taskIdentity(task) === leaseTaskId)
    ?? tasks.find((task) => ACTIVE_STATES.has(task?.state))
    ?? tasks.find((task) => Object.keys(asObject(task?.intervention)).length > 0)
    ?? null;
}


let generatedControlId = 0;


function newControlId(prefix) {
  const random = globalThis.crypto?.randomUUID?.();
  if (typeof random === "string" && SAFE_ID.test(random)) {
    return `${prefix}-${random}`;
  }
  generatedControlId += 1;
  return `${prefix}-${Date.now().toString(36)}-${generatedControlId}`;
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


function closeDrawer(documentRoot, id, button, isMobileDrawer, restoreFocus = false) {
  const panel = documentRoot.querySelector(id);
  if (panel) {
    panel.classList.remove("drawer-open");
    panel.inert = isMobileDrawer();
    panel.removeAttribute?.("role");
    panel.removeAttribute?.("aria-modal");
  }
  button.setAttribute("aria-expanded", "false");
  if (restoreFocus) {
    button.focus?.();
  }
}


function drawerFocusables(panel) {
  const selector = "button, textarea, input, select, a, summary, [tabindex]";
  const candidates = typeof panel?.querySelectorAll === "function"
    ? Array.from(panel.querySelectorAll(selector))
    : [panel?.querySelector?.(selector)].filter(Boolean);
  return candidates.filter((node) => (
    node.disabled !== true
    && node.hidden !== true
    && node.getAttribute?.("tabindex") !== "-1"
  ));
}


function wireDrawer(
  documentRoot,
  buttonId,
  panelId,
  otherButtonId,
  otherPanelId,
  isMobileDrawer,
  syncInert,
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
    if (opening && isMobileDrawer()) {
      panel.setAttribute("role", "dialog");
      panel.setAttribute("aria-modal", "true");
    } else {
      panel.removeAttribute?.("role");
      panel.removeAttribute?.("aria-modal");
    }
    button.setAttribute("aria-expanded", opening ? "true" : "false");
    syncInert?.();
    if (opening) {
      const focusTarget = drawerFocusables(panel)[0];
      focusTarget?.focus();
    } else {
      button.focus?.();
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
  let composerBinding = null;
  let activeInterventionDraft = null;
  let acknowledgementDraft = null;
  let unknownWarningId = null;

  const composer = documentRoot.querySelector("#composer");
  const messageInput = documentRoot.querySelector("#message-input");
  const composerSubmit = documentRoot.querySelector("#composer-submit");
  const composerLabel = documentRoot.querySelector("#composer-label");
  const composerRecipient = documentRoot.querySelector("#composer-recipient");
  const composerBindingNode = documentRoot.querySelector("#composer-binding");
  const composerBindingText = documentRoot.querySelector("#composer-binding-text");
  const composerClearBinding = documentRoot.querySelector("#composer-clear-binding");
  const guidance = documentRoot.querySelector("#composer-guidance");
  const interventionContext = documentRoot.querySelector("#intervention-context");
  const interveneControl = documentRoot.querySelector("#intervene-control");
  const stopControl = documentRoot.querySelector("#stop-control");
  const conversationStatus = documentRoot.querySelector("#conversation-status");
  const conversationContext = documentRoot.querySelector("#conversation-context");
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

  function bindingText(binding) {
    if (binding?.kind === "question") {
      return `Replying to ${actorLabel(binding.askedBy)} · task ${binding.taskId} · r${binding.revision} · question ${binding.questionId} · generation ${binding.continuationGeneration}`;
    }
    if (binding?.kind === "continuation") {
      return `Replying on task ${binding.taskId} · r${binding.revision} · generation ${binding.continuationGeneration}`;
    }
    return "";
  }

  function setComposerBinding(binding) {
    composerBinding = binding;
    renderStatus();
    if (binding !== null) {
      messageInput.focus();
    }
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
    let activeTask = activeTaskForControls(state);
    let activePresentation = null;
    if (activeTask !== null) {
      try {
        activePresentation = interventionPresentation(state, activeTask);
      } catch (_error) {
        // A websocket state update can arrive before its bounded bootstrap
        // projection contains the exact continuation identity.
        activeTask = null;
      }
    }
    if (activeInterventionDraft !== null && (
      activeTask === null
      || activeInterventionDraft.taskId !== taskIdentity(activeTask)
      || activeInterventionDraft.revision !== activeTask.revision
      || activeInterventionDraft.sourceGeneration !== activeTask.continuation_generation
    )) {
      activeInterventionDraft = null;
    }
    if (activeInterventionDraft !== null && activePresentation?.kind !== "new") {
      activeInterventionDraft = null;
    }
    const interventionMode = activeInterventionDraft !== null;
    const boundReply = composerBinding !== null;
    const selectedRecipient = composerRecipient?.value ?? "fable";
    const presentation = composerPresentation(state, composerBinding, selectedRecipient);
    messageInput.disabled = interventionMode ? activeInterventionDraft.submitted : presentation.disabled;
    composerSubmit.disabled = interventionMode
      ? activeInterventionDraft.submitted || !String(messageInput.value ?? "").trim()
      : presentation.disabled;
    if (composerRecipient) {
      composerRecipient.disabled = interventionMode ? activeInterventionDraft.submitted : presentation.recipientDisabled;
      const eligibleRecipients = interventionMode ? interventionRecipients(activeTask) : null;
      for (const option of Array.from(composerRecipient.options ?? [])) {
        option.disabled = interventionMode && !eligibleRecipients.includes(option.value);
      }
      if (interventionMode && !eligibleRecipients.includes(composerRecipient.value)) {
        composerRecipient.value = "fable";
      }
    }
    if (composerLabel) {
      composerLabel.textContent = interventionMode ? "Intervention guidance" : presentation.label;
    }
    composerSubmit.textContent = interventionMode ? "Intervene" : presentation.submit;
    if (composerBindingNode) {
      composerBindingNode.hidden = !boundReply;
    }
    if (composerBindingText) {
      composerBindingText.textContent = boundReply ? bindingText(composerBinding) : "";
    }
    if (composerClearBinding) {
      composerClearBinding.disabled = !boundReply;
    }
    guidance.textContent = state.sessionId === null
      ? `${presentation.guidance} Waiting for the server session identifier.`
      : presentation.guidance;
    if (interventionMode) {
      guidance.textContent = "Guidance will interrupt the exact active run. Choose Fable or Sol; the server verifies eligibility.";
    }
    renderInterventionControls(activeTask, activePresentation);
  }

  function removeUnknownAcknowledgement() {
    const existing = conversationContext?.querySelector?.("#intervention-acknowledge-control");
    existing?.remove?.();
  }

  function renderInterventionControls(activeTask, presentation) {
    if (!interveneControl || !stopControl) {
      return;
    }
    const activeRun = activeTask !== null && ACTIVE_STATES.has(activeTask.state);
    const interventionStatus = asObject(activeTask?.intervention).status;
    const stoppableIntervention = [
      "pending_stop", "ready", "resuming", "resume_outcome_unknown",
    ].includes(interventionStatus);
    interveneControl.hidden = activeTask === null;
    stopControl.hidden = !(activeRun || stoppableIntervention);
    stopControl.disabled = !(activeRun || stoppableIntervention);
    removeUnknownAcknowledgement();
    if (activeTask === null || presentation === null) {
      interventionContext?.removeAttribute?.("role");
      interventionContext?.removeAttribute?.("tabindex");
      interventionContext && (interventionContext.textContent = "Intervention controls appear for an active run.");
      return;
    }
    interveneControl.textContent = presentation.submit;
    interveneControl.disabled = presentation.kind === "pending"
      || presentation.kind === "canceled"
      || presentation.kind === "unavailable";
    if (presentation.kind === "new") {
      interventionContext?.removeAttribute?.("role");
      interventionContext?.removeAttribute?.("tabindex");
      interventionContext && (interventionContext.textContent = activeInterventionDraft === null
        ? "An agent is running. Intervene with exact guidance for Fable or Sol, or Stop the run separately."
        : "Intervention guidance is bound to this exact active run.");
      return;
    }
    if (presentation.kind === "pending") {
      interventionContext?.removeAttribute?.("role");
      interventionContext?.removeAttribute?.("tabindex");
      interventionContext && (interventionContext.textContent = "Intervention accepted; waiting for the exact source run to stop.");
      return;
    }
    if (presentation.kind === "resume") {
      interventionContext?.removeAttribute?.("role");
      interventionContext?.removeAttribute?.("tabindex");
      interventionContext && (interventionContext.textContent = "The interruption is ready to resume from its stored continuation.");
      return;
    }
    if (presentation.kind === "canceled") {
      interventionContext?.removeAttribute?.("role");
      interventionContext?.removeAttribute?.("tabindex");
      interventionContext && (interventionContext.textContent = "Intervention canceled by Stop.");
      return;
    }
    if (presentation.kind === "unknown") {
      if (
        acknowledgementDraft !== null
        && (acknowledgementDraft.interventionId !== presentation.interventionId
          || acknowledgementDraft.resumeGeneration !== presentation.resumeGeneration)
      ) {
        acknowledgementDraft = null;
      }
      if (interventionContext) {
        interventionContext.textContent = `Warning: ${presentation.warning}`;
        interventionContext.setAttribute("role", "alert");
        interventionContext.setAttribute("tabindex", "-1");
        const warningKey = interventionWarningKey({
          intervention_id: presentation.interventionId,
          resume_generation: presentation.resumeGeneration,
        });
        if (unknownWarningId !== warningKey) {
          unknownWarningId = warningKey;
          interventionContext.focus?.();
        }
      }
      interveneControl.hidden = true;
      const acknowledge = element(
        documentRoot,
        "button",
        "Acknowledge possible prior execution",
        "button button-danger",
      );
      acknowledge.id = "intervention-acknowledge-control";
      acknowledge.type = "button";
      acknowledge.addEventListener("click", () => {
        acknowledgementDraft ??= {
          interventionId: presentation.interventionId,
          resumeGeneration: presentation.resumeGeneration,
          acknowledgmentId: newControlId("acknowledgment"),
        };
        void submitUnknownAcknowledgement(presentation);
      });
      conversationContext?.append?.(acknowledge);
    }
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
        closeDrawer(documentRoot, projectDrawerId, toggle, isMobileDrawer);
        syncDrawerMode();
      }
    });
    renderTaskInspector(documentRoot, selectedTask(), {
      gate: state.gate,
      onAction: handleTaskAction,
    });
    renderPersistentInspector(documentRoot, selectedTask(), {
      gate: state.gate,
      onAction: handleTaskAction,
    });
    renderActivityAudit(documentRoot, selectedTask());
    renderPendingConversationCards(documentRoot, state.tasks, {
      onReply: (binding) => setComposerBinding(binding),
      onGrant: (binding) => {
        void (async () => {
          try {
            const request = exchangeGrantRequest(state, binding);
            await postJson(windowRoot.fetch.bind(windowRoot), request.path, request.payload, state.csrfToken);
            await controller?.refreshSelectedBootstrap();
          } catch (error) {
            showToast(documentRoot, String(error.message ?? error), true);
          }
        })();
      },
    }, state);
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

  async function submitInterventionResume(presentation) {
    try {
      await postJson(
        windowRoot.fetch.bind(windowRoot),
        presentation.path,
        presentation.payload,
        state.csrfToken,
      );
      conversationStatus && (conversationStatus.textContent = "Intervention resume accepted.");
      await controller?.refreshSelectedBootstrap();
    } catch (error) {
      showToast(documentRoot, String(error.message ?? error), true);
    }
  }

  async function submitUnknownAcknowledgement(presentation) {
    if (
      acknowledgementDraft?.interventionId !== presentation.interventionId
      || acknowledgementDraft?.resumeGeneration !== presentation.resumeGeneration
    ) {
      acknowledgementDraft = null;
      return;
    }
    try {
      const request = interventionRetryRequest(
        state,
        presentation,
        acknowledgementDraft.acknowledgmentId,
      );
      await postJson(
        windowRoot.fetch.bind(windowRoot), request.path, request.payload, state.csrfToken,
      );
      conversationStatus && (conversationStatus.textContent = "Possible prior execution acknowledged; retry accepted.");
      acknowledgementDraft = null;
      unknownWarningId = null;
      await controller?.refreshSelectedBootstrap();
    } catch (error) {
      if (error?.status === 409) {
        acknowledgementDraft = null;
      }
      showToast(documentRoot, String(error.message ?? error), true);
    }
  }

  interveneControl?.addEventListener("click", () => {
    const activeTask = activeTaskForControls(state);
    if (activeTask === null) {
      return;
    }
    const presentation = interventionPresentation(state, activeTask);
    if (presentation.kind === "new") {
      activeInterventionDraft ??= Object.freeze({
        taskId: taskIdentity(activeTask),
        revision: activeTask.revision,
        sourceGeneration: activeTask.continuation_generation,
        interventionId: newControlId("intervention"),
        addressedTo: "fable",
        message: null,
        submitted: false,
      });
      renderStatus();
      messageInput.focus?.();
      return;
    }
    if (presentation.kind === "resume") {
      void submitInterventionResume(presentation);
    }
  });

  stopControl?.addEventListener("click", () => {
    const activeTask = activeTaskForControls(state);
    if (activeTask === null) {
      return;
    }
    void (async () => {
      try {
        await sendTaskAction("stop", activeTask, null);
        activeInterventionDraft = null;
        acknowledgementDraft = null;
        conversationStatus && (conversationStatus.textContent = "Stop accepted; any pending intervention is canceled by the server.");
        await controller?.refreshSelectedBootstrap();
      } catch (error) {
        showToast(documentRoot, String(error.message ?? error), true);
      }
    })();
  });

  composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = String(messageInput.value ?? "");
    if (
      (!state.gate.canCompose && activeInterventionDraft === null)
      || !state.projectId
      || !state.sessionId
      || !text.trim()
      || (composerBinding === null && state.activeLease !== null && activeInterventionDraft === null)
    ) {
      return;
    }
    let request;
    try {
      if (activeInterventionDraft !== null) {
        const activeTask = activeTaskForControls(state);
        if (
          activeTask === null
          || activeInterventionDraft.taskId !== taskIdentity(activeTask)
          || activeInterventionDraft.revision !== activeTask.revision
          || activeInterventionDraft.sourceGeneration !== activeTask.continuation_generation
        ) {
          throw new Error("intervention source is no longer active");
        }
        const recipient = composerRecipient?.value ?? activeInterventionDraft.addressedTo;
        const candidate = activeInterventionDraft.message === null
          ? interventionDraft(activeTask, activeInterventionDraft.interventionId, recipient, text)
          : activeInterventionDraft;
        request = interventionRequest(
          state,
          activeTask,
          candidate.message,
          candidate.interventionId,
          candidate.addressedTo,
        );
        activeInterventionDraft = Object.freeze({...candidate, submitted: true});
      } else {
        request = composerRequest(state, composerBinding, text, composerRecipient?.value ?? "fable");
      }
    } catch (error) {
      showToast(documentRoot, String(error.message ?? error), true);
      return;
    }
    messageInput.disabled = true;
    composerSubmit.disabled = true;
    try {
      await postJson(
        windowRoot.fetch.bind(windowRoot),
        request.path,
        request.payload,
        state.csrfToken,
      );
      messageInput.value = "";
      if (activeInterventionDraft !== null) {
        conversationStatus && (conversationStatus.textContent = "Intervention accepted; waiting for server status.");
        void controller?.refreshSelectedBootstrap().catch(() => {});
      } else if (composerBinding !== null) {
        composerBinding = null;
        void controller?.refreshSelectedBootstrap().catch(() => {});
      }
    } catch (error) {
      if (activeInterventionDraft?.submitted) {
        activeInterventionDraft = Object.freeze({...activeInterventionDraft, submitted: false});
      }
      if (error?.status === 409) {
        activeInterventionDraft = null;
        messageInput.value = "";
      }
      showToast(documentRoot, String(error.message ?? error), true);
    } finally {
      renderStatus();
    }
  });

  composerRecipient?.addEventListener("change", renderStatus);
  messageInput.addEventListener("input", renderStatus);
  composerClearBinding?.addEventListener("click", () => setComposerBinding(null));

  usageCheckbox.addEventListener("change", () => {
    usageSubmit.disabled = !usageCheckbox.checked;
  });
  usageModal.addEventListener("cancel", (event) => event.preventDefault());
  documentRoot.querySelector("#activity-audit")?.addEventListener("toggle", renderWorkspace);
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

  const projectDrawerId = documentRoot.querySelector("#project-navigation")
    ? "#project-navigation"
    : "#task-list";
  const inspectorDrawerId = documentRoot.querySelector("#task-inspector-panel")
    ? "#task-inspector-panel"
    : "#task-inspector";

  function syncDrawerMode() {
    const conversationShell = documentRoot.querySelector("#conversation-shell")
      ?? documentRoot.querySelector("#conversation");
    const drawers = [
      [projectDrawerId, "#task-drawer-toggle"],
      [inspectorDrawerId, "#inspector-drawer-toggle"],
    ];
    const openPanel = drawers.find(([panelId]) => (
      documentRoot.querySelector(panelId)?.classList.contains("drawer-open")
    ))?.[0] ?? null;
    for (const [panelId, buttonId] of drawers) {
      const panel = documentRoot.querySelector(panelId);
      const button = documentRoot.querySelector(buttonId);
      if (!panel || !button) {
        continue;
      }
      if (!isMobileDrawer()) {
        panel.classList.remove("drawer-open");
        panel.inert = false;
        panel.removeAttribute?.("role");
        panel.removeAttribute?.("aria-modal");
        button.setAttribute("aria-expanded", "false");
      } else {
        panel.inert = panelId !== openPanel;
      }
    }
    if (conversationShell) {
      conversationShell.inert = isMobileDrawer() && openPanel !== null;
    }
  }

  wireDrawer(
    documentRoot,
    "#task-drawer-toggle",
    projectDrawerId,
    "#inspector-drawer-toggle",
    inspectorDrawerId,
    isMobileDrawer,
    syncDrawerMode,
  );
  wireDrawer(
    documentRoot,
    "#inspector-drawer-toggle",
    inspectorDrawerId,
    "#task-drawer-toggle",
    projectDrawerId,
    isMobileDrawer,
    syncDrawerMode,
  );

  syncDrawerMode();
  if (typeof drawerMedia.addEventListener === "function") {
    drawerMedia.addEventListener("change", syncDrawerMode);
  } else if (typeof drawerMedia.addListener === "function") {
    drawerMedia.addListener(syncDrawerMode);
  }

  documentRoot.addEventListener("keydown", (event) => {
    if (event.key === "Tab" && isMobileDrawer()) {
      const openPanel = [projectDrawerId, inspectorDrawerId]
        .map((panelId) => documentRoot.querySelector(panelId))
        .find((panel) => panel?.classList.contains("drawer-open"));
      const focusable = drawerFocusables(openPanel);
      if (focusable.length > 0) {
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (documentRoot.activeElement !== first && documentRoot.activeElement !== last) {
          event.preventDefault();
          (event.shiftKey ? last : first).focus?.();
          return;
        }
        if ((!event.shiftKey && documentRoot.activeElement === last)
          || (event.shiftKey && documentRoot.activeElement === first)) {
          event.preventDefault();
          (event.shiftKey ? last : first).focus?.();
          return;
        }
      }
    }
    if (event.key !== "Escape") {
      return;
    }
    for (const [panelId, buttonId] of [
      [projectDrawerId, "#task-drawer-toggle"],
      [inspectorDrawerId, "#inspector-drawer-toggle"],
    ]) {
      const panel = documentRoot.querySelector(panelId);
      const button = documentRoot.querySelector(buttonId);
      if (panel?.classList.contains("drawer-open") && button) {
        event.preventDefault();
        closeDrawer(documentRoot, panelId, button, isMobileDrawer, true);
        syncDrawerMode();
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
