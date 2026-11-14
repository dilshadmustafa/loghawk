<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute, useRouter, onBeforeRouteUpdate } from "vue-router";
import { ApiError, sendRcaMessage, confirmRcaCollector, retryRcaExplanation, regenerateRcaExplanation, getRcaProfiles, getRcaProgress, getCurrentUser, listInvestigations } from "../services/api";
import RcaLiveProgress from "../components/RcaLiveProgress.vue";
import type { Investigation, RcaMessage, RcaAnswer, RcaExecutionSummary, RcaMode, RcaProfile, RcaLiveProgress as Progress } from "../types/api";
type ChatEntry = RcaMessage & { requestId?: string; execution?: RcaExecutionSummary; explanation?: RcaAnswer["explanation"] };
type SavedRequest = { id: string; mode: RcaMode; startedAt: string; baseline: string; operationLabel: string; reviewArguments: boolean; redactEvidence: boolean; confirmationId?: string; explanationAttempts?: number; investigationId?: string };
const route = useRoute(), router = useRouter();
const investigationId = computed(() => typeof route.query.investigation === "string" ? route.query.investigation : "");
const directEntry = ref(!investigationId.value);
const chatStarted = ref(false);
const showInvestigationPicker = computed(() => directEntry.value && !chatStarted.value);
const pickerOpen = ref(false);
const investigationOptions = ref<Investigation[]>([]);
const selectedInvestigation = ref<Investigation | null>(null);
const investigationSearch = ref("");
const investigationListBusy = ref(false);
const investigationListError = ref("");
const investigationListOffset = ref(0);
const investigationListHasMore = ref(false);
let investigationSearchTimer: number | undefined;
let investigationListGeneration = 0;
const messages = ref<ChatEntry[]>([]);
const draft = ref("");
const busy = ref(false), ready = ref(false);
const error = ref("");
const pending = ref<RcaAnswer | null>(null);
const uncertain = ref(false);
const argumentDraft = ref("");
const reviewInputArgs = ref(false);
const redactEvidence = ref(true);
const newChatDialog = ref<HTMLDialogElement | null>(null);
const mode = ref<RcaMode>("performance");
const profiles = ref<RcaProfile[]>([]);
const selectedProfile = computed(() => profiles.value.find(p => p.mode === mode.value));
const activeId = ref("");
const requestMode = ref<RcaMode>("performance");
const requestInvestigationId = ref("");
const progress = ref<Progress | null>(null);
const progressAnchor = ref(-1);
const startedAt = ref("");
const checkedAt = ref<number | null>(null);
const connectionMessage = ref("");
const waitingForAction = ref(false);
const operationLabel = ref("Submitting request");
const conversation = computed(() => {
  const entries: { message?: ChatEntry; showProgress?: boolean }[] = messages.value.map(message => ({ message }));
  if (activeId.value) entries.splice(progressAnchor.value + 1, 0, { showProgress: true });
  return entries;
});
let storageKey = "", baseline = "", operation = 0, resolved = false, disposed = false, postInFlight = false;
let confirmationId = "", explanationAttempts: number | undefined;
let timer: number | undefined;
let pollController: AbortController | undefined;

function saveActive() {
  if (!storageKey || !activeId.value) return;
  const value: SavedRequest = { id: activeId.value, mode: mode.value, startedAt: startedAt.value,
    baseline, operationLabel: operationLabel.value, reviewArguments: reviewInputArgs.value, redactEvidence: redactEvidence.value,
    confirmationId, explanationAttempts, investigationId: requestInvestigationId.value };
  try { sessionStorage.setItem(storageKey, JSON.stringify(value)); } catch { /* Polling remains available in this page. */ }
}
function clearSaved() { try { if (storageKey) sessionStorage.removeItem(storageKey); } catch { /* Browser storage may be unavailable. */ } }
function stopPolling() {
  if (timer !== undefined) window.clearTimeout(timer);
  timer = undefined;
  pollController?.abort();
  pollController = undefined;
}
function isCurrent(current: number) { return !disposed && current === operation; }
function resetTracking() {
  stopPolling(); ++operation; resolved = true; clearSaved();
  activeId.value = ""; progress.value = null; checkedAt.value = null; connectionMessage.value = "";
  waitingForAction.value = false; busy.value = false; uncertain.value = false;
}
function requestNewChat() {
  if (busy.value || pending.value || uncertain.value) return;
  newChatDialog.value?.showModal();
}
function confirmNewChat() {
  if (busy.value || pending.value || uncertain.value) return;
  resetTracking(); messages.value = []; error.value = ""; argumentDraft.value = "";
  directEntry.value = true; chatStarted.value = false; pickerOpen.value = false;
  selectedInvestigation.value = null; investigationSearch.value = "";
  newChatDialog.value?.close();
  void router.replace({ path: "/rca/chat", query: {} });
  void loadInvestigationOptions(0);
}
onBeforeRouteUpdate(() => {
  if (busy.value || pending.value || uncertain.value) return false;
  resetTracking(); messages.value = []; error.value = ""; argumentDraft.value = "";
});
onBeforeUnmount(() => { disposed = true; ++operation; ++investigationListGeneration; if (investigationSearchTimer !== undefined) window.clearTimeout(investigationSearchTimer); stopPolling(); });
const blockedReason = computed(() => {
  if (!ready.value) return "Loading investigation modes and session…";
  if (busy.value) return "An investigation is running. Its recorded activity appears above.";
  if (uncertain.value) return "Waiting to reconcile the request with the server. Check RCA History for its recorded state.";
  if (pending.value) return "Choose Confirm & Execute or Cancel before sending another message.";
  if (!draft.value.trim()) return "Type a message to enable Send.";
  return "";
});
function formatInvestigationDate(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}
async function loadInvestigationOptions(offset = 0) {
  const generation = ++investigationListGeneration;
  investigationListBusy.value = true;
  investigationListError.value = "";
  try {
    const page = await listInvestigations(offset, { q: investigationSearch.value.trim() });
    if (disposed || generation !== investigationListGeneration) return;
    if (offset === 0) investigationOptions.value = page;
    else {
      const knownIds = new Set(investigationOptions.value.map(item => item.id));
      investigationOptions.value = [...investigationOptions.value, ...page.filter(item => !knownIds.has(item.id))];
    }
    investigationListOffset.value = offset;
    investigationListHasMore.value = page.length === 50;
  } catch (e) {
    if (!disposed && generation === investigationListGeneration) {
      investigationListError.value = e instanceof Error ? e.message : String(e);
    }
  } finally {
    if (generation === investigationListGeneration) investigationListBusy.value = false;
  }
}
function scheduleInvestigationSearch() {
  if (investigationSearchTimer !== undefined) window.clearTimeout(investigationSearchTimer);
  investigationSearchTimer = window.setTimeout(() => { void loadInvestigationOptions(0); }, 250);
}
function chooseInvestigation(item: Investigation | null) {
  selectedInvestigation.value = item;
  investigationSearch.value = "";
  pickerOpen.value = false;
  void loadInvestigationOptions(0);
}
function loadMoreInvestigations() {
  void loadInvestigationOptions(investigationListOffset.value + 50);
}
function formatEvidence(evidence: string): string {
  try { return JSON.stringify(JSON.parse(evidence), null, 2); } catch { return evidence; }
}
function receive(answer: RcaAnswer) {
  pending.value = answer.pending ? answer : null;
  argumentDraft.value = answer.pending ? JSON.stringify(answer.pending.arguments, null, 2) : "";
  if (answer.explanation && answer.request_id) {
    for (const entry of messages.value) {
      if (entry.requestId === answer.request_id && entry.explanation) {
        entry.explanation.can_retry = false; entry.explanation.can_regenerate = false;
      }
    }
  }
  if (answer.response || answer.execution) messages.value.push({
    role: "assistant", content: answer.response || "", requestId: answer.request_id,
    execution: answer.execution, explanation: answer.explanation,
  });
}
function applySnapshot(value: Progress) {
  progress.value = value; checkedAt.value = Date.now();
  const savedMode = value.settings?.mode;
  if (savedMode === "quality" || savedMode === "balanced" || savedMode === "performance") {
    requestMode.value = savedMode;
    if (busy.value || pending.value || uncertain.value) mode.value = savedMode;
  }
  requestInvestigationId.value = value.investigation_id || requestInvestigationId.value;
}
async function finishSnapshot(current: number) {
  try {
    const value = await getRcaProgress(activeId.value);
    if (!isCurrent(current)) return;
    applySnapshot(value); waitingForAction.value = false; connectionMessage.value = "";
  } catch {
    if (!isCurrent(current)) return;
    connectionMessage.value = "The response was received. Reconnecting to refresh its recorded activity…";
    timer = window.setTimeout(() => { void finishSnapshot(current); }, 2000);
  }
}
function settle(answer: RcaAnswer, current: number) {
  if (!isCurrent(current) || resolved) return;
  if (answer.duplicate || ["PROCESSING", "EXPLAINING"].includes(answer.status || "")) return;
  resolved = true; stopPolling(); busy.value = false; uncertain.value = false;
  waitingForAction.value = false; baseline = ""; confirmationId = ""; explanationAttempts = undefined;
  connectionMessage.value = ""; error.value = "";
  receive(answer);
  if (!answer.request_id) { clearSaved(); activeId.value = ""; progress.value = null; return; }
  if (answer.pending) saveActive(); else clearSaved();
  waitingForAction.value = true;
  operationLabel.value = answer.pending ? "Awaiting collector confirmation" : "Request finished; refreshing recorded activity";
  void finishSnapshot(current);
}
async function poll(current: number) {
  if (!isCurrent(current) || resolved) return;
  pollController = new AbortController();
  try {
    const value = await getRcaProgress(activeId.value, pollController.signal);
    if (!isCurrent(current) || resolved) return;
    applySnapshot(value);
    connectionMessage.value = "";
    const oldConfirmation = !!confirmationId && value.answer?.pending?.confirmation_id === confirmationId;
    const oldExplanation = explanationAttempts !== undefined && !!value.answer &&
      ((value.answer.explanation?.attempts ?? 0) <= explanationAttempts || value.answer.explanation?.status === "RUNNING");
    const changed = !value.snapshot_stale && (!baseline || value.revision !== baseline) && !oldConfirmation && !oldExplanation;
    waitingForAction.value = !changed;
    if (changed) {
      busy.value = true; uncertain.value = false;
      if (value.answer) { settle(value.answer, current); return; }
    } else if (!postInFlight && Date.now() - Date.parse(startedAt.value) > 15000) {
      busy.value = false; uncertain.value = true;
      connectionMessage.value = "The server has not recorded this action yet. Continuing to check this request; review RCA History before taking another action.";
    }
  } catch (e) {
    if (!isCurrent(current) || resolved || (e instanceof Error && e.name === "AbortError")) return;
    if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
      busy.value = false; uncertain.value = true;
      connectionMessage.value = "Progress is unavailable for this session. Sign in again to reconnect to this request.";
      return;
    }
    if (e instanceof ApiError && e.status === 404) {
      const stillSubmitting = postInFlight || Date.now() - Date.parse(startedAt.value) < 15000;
      connectionMessage.value = stillSubmitting ? "Submitting request; waiting for the server to record it."
        : "This request is not recorded yet. Continuing to check; its submission outcome is uncertain.";
      if (!stillSubmitting) { busy.value = false; uncertain.value = true; }
    } else {
      connectionMessage.value = "The progress connection was interrupted. Reconnecting to the same request…";
    }
  }
  if (isCurrent(current) && !resolved) timer = window.setTimeout(() => { void poll(current); }, 2000);
}
function beginTracking(id: string, label: string, previousRevision = "", restoredAt?: string) {
  stopPolling(); const current = ++operation;
  if (activeId.value !== id) { progress.value = null; checkedAt.value = null; }
  activeId.value = id; requestMode.value = mode.value; startedAt.value = restoredAt || new Date().toISOString(); baseline = previousRevision;
  operationLabel.value = label; waitingForAction.value = !!previousRevision || !progress.value;
  resolved = false; busy.value = true; uncertain.value = false; error.value = ""; connectionMessage.value = "";
  saveActive();
  timer = window.setTimeout(() => { void poll(current); }, 0);
  return current;
}
async function initialize() {
  ready.value = false; error.value = "";
  try {
    const [available, user] = await Promise.all([getRcaProfiles(), getCurrentUser()]);
    if (disposed) return;
    profiles.value = available.profiles;
    storageKey = `loghawk.rca.active.${encodeURIComponent(user.userid)}`;
    let saved: SavedRequest | null = null;
    try { saved = JSON.parse(sessionStorage.getItem(storageKey) || "null") as SavedRequest | null; } catch { clearSaved(); }
    if (saved && /^[a-f0-9]{32}$/i.test(saved.id) && ["quality", "balanced", "performance"].includes(saved.mode)) {
      chatStarted.value = true;
      mode.value = saved.mode; reviewInputArgs.value = saved.reviewArguments; redactEvidence.value = saved.redactEvidence;
      confirmationId = saved.confirmationId || ""; explanationAttempts = saved.explanationAttempts;
      requestInvestigationId.value = saved.investigationId || "";
      beginTracking(saved.id, saved.operationLabel || "Reconnecting to investigation", saved.baseline || "", saved.startedAt);
    }
    ready.value = true;
  } catch (e) { if (!disposed) error.value = e instanceof Error ? e.message : String(e); }
}
onMounted(() => { void initialize(); if (showInvestigationPicker.value) void loadInvestigationOptions(0); });

async function send() {
  if (!ready.value || busy.value || pending.value || uncertain.value || !draft.value.trim()) return;
  if (showInvestigationPicker.value) {
    const selectedId = selectedInvestigation.value?.id || "";
    if (selectedId !== investigationId.value) {
      try {
        await router.replace({ path: "/rca/chat", query: selectedId ? { investigation: selectedId } : {} });
      } catch (e) {
        error.value = e instanceof Error ? e.message : String(e);
        return;
      }
      if (selectedId !== investigationId.value) {
        error.value = "Could not set the selected investigation context. Please try again.";
        return;
      }
    }
    chatStarted.value = true;
  }
  const submitted = draft.value.trim();
  messages.value.push({ role: "user", content: submitted });
  progressAnchor.value = messages.value.length - 1;
  const history = messages.value.map(({ role, content }) => ({ role, content }));
  draft.value = "";
  const id = crypto.randomUUID().replaceAll("-", "");
  confirmationId = ""; explanationAttempts = undefined; requestInvestigationId.value = investigationId.value;
  const current = beginTracking(id, "Submitting request");
  postInFlight = true;
  try { settle(await sendRcaMessage(history, reviewInputArgs.value, redactEvidence.value, investigationId.value, mode.value, id), current); }
  catch (e) {
    if (!isCurrent(current) || resolved) return;
    if (e instanceof ApiError && e.executionNotStarted) {
      resetTracking(); messages.value.pop(); draft.value = submitted;
      error.value = e.message;
    } else {
      error.value = `The submission response could not be received: ${e instanceof Error ? e.message : String(e)}. Checking the saved request for its outcome.`;
    }
  } finally { if (isCurrent(current)) postInFlight = false; }
}
async function beginExisting(id: string, label: string, kind: "confirmation" | "explanation"): Promise<number | null> {
  busy.value = true; error.value = "";
  try {
    // Capture the previous revision before POST so a poll cannot consume an old completed answer.
    const value = await getRcaProgress(id);
    if (disposed) return null;
    if (value.snapshot_stale) {
      busy.value = false; error.value = "The request status is changing. Wait for its recorded activity to update before trying this action again.";
      return null;
    }
    requestInvestigationId.value = value.investigation_id || "";
    applySnapshot(value);
    if (!value.answer || (kind === "confirmation" &&
        value.answer.pending?.confirmation_id !== pending.value?.pending?.confirmation_id)) {
      confirmationId = ""; explanationAttempts = undefined;
      beginTracking(id, "Reconnecting to the current request state");
      return null;
    }
    confirmationId = kind === "confirmation" ? value.answer?.pending?.confirmation_id || "" : "";
    explanationAttempts = kind === "explanation" ? value.answer?.explanation?.attempts ?? 0 : undefined;
    const firstReply = messages.value.findIndex(message => message.requestId === id);
    progressAnchor.value = firstReply < 0 ? -1 : firstReply - 1;
    return beginTracking(id, label, value.revision);
  } catch (e) {
    if (!disposed) { busy.value = false; error.value = `Could not check the request before submitting: ${e instanceof Error ? e.message : String(e)}`; }
    return null;
  }
}
function rejectedAction(current: number, e: unknown) {
  if (!isCurrent(current) || resolved) return;
  error.value = e instanceof Error ? e.message : String(e);
  if (e instanceof ApiError && (e.status === 422 || e.executionNotStarted)) {
    resolved = true; stopPolling(); busy.value = false; uncertain.value = false; waitingForAction.value = false; baseline = ""; confirmationId = ""; explanationAttempts = undefined;
    if (pending.value) saveActive(); else clearSaved();
  } else {
    error.value += ". Checking the saved request for the action's outcome.";
  }
}
async function retryExplanation(message: ChatEntry) {
  if (busy.value || pending.value || uncertain.value || !message.requestId) return;
  const explanation = message.explanation;
  if (!explanation?.can_retry && !explanation?.can_regenerate) return;
  const current = await beginExisting(message.requestId, "Submitting explanation request", "explanation");
  if (current === null) return;
  postInFlight = true;
  try {
    const currentExplanation = progress.value?.answer?.explanation ?? explanation;
    const action = currentExplanation.can_regenerate ? regenerateRcaExplanation : retryRcaExplanation;
    settle(await action(message.requestId), current);
  } catch (e) { rejectedAction(current, e); }
  finally { if (isCurrent(current)) postInFlight = false; }
}
async function decide(approved: boolean) {
  const proposal = pending.value;
  if (busy.value || uncertain.value || !proposal?.pending || !proposal.request_id) return;
  error.value = "";
  let argumentsToRun: Record<string, unknown> | undefined;
  if (approved) {
    try {
      const parsed = JSON.parse(argumentDraft.value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Arguments must be a JSON object.");
      argumentsToRun = parsed;
    } catch (e) { error.value = e instanceof Error ? e.message : String(e); return; }
  }
  const current = await beginExisting(proposal.request_id, approved ? "Submitting collector confirmation" : "Cancelling collector request", "confirmation");
  if (current === null) return;
  postInFlight = true;
  try { settle(await confirmRcaCollector(proposal.request_id, proposal.pending.confirmation_id, approved, argumentsToRun), current); }
  catch (e) { rejectedAction(current, e); }
  finally { if (isCurrent(current)) postInFlight = false; }
}
</script>

<template>
  <section class="rca-chat">
    <h1>AI-Assisted RCA</h1>
    <section v-if="showInvestigationPicker" class="investigation-context-picker" aria-label="Investigation context">
      <label for="investigation-context-trigger">Investigation context</label>
      <button id="investigation-context-trigger" class="context-trigger" type="button"
        :aria-expanded="pickerOpen" aria-controls="investigation-context-options" @click="pickerOpen = !pickerOpen">
        <span class="context-choice">{{ selectedInvestigation?.problem || "General RCA chat" }}</span>
        <span v-if="selectedInvestigation" class="context-status" :class="{ open: selectedInvestigation.status === 'OPEN' }">
          {{ selectedInvestigation.status }}
        </span>
        <span class="context-chevron" aria-hidden="true">{{ pickerOpen ? "▴" : "▾" }}</span>
      </button>
      <p class="context-help">Choose an investigation before your first message. Once the chat starts, its context stays fixed until you start a New Chat.</p>
      <div v-if="pickerOpen" id="investigation-context-options" class="context-menu">
        <label class="context-search">Search by problem or owner
          <input v-model="investigationSearch" type="search" placeholder="Search investigations…" @input="scheduleInvestigationSearch" />
        </label>
        <div class="context-option-list" role="list">
          <button class="context-option" type="button" :aria-pressed="!selectedInvestigation"
            @click="chooseInvestigation(null)">
            <span><strong>General RCA chat</strong><small>No investigation context</small></span>
            <span v-if="!selectedInvestigation" aria-hidden="true">Selected</span>
          </button>
          <button v-for="item in investigationOptions" :key="item.id" class="context-option" type="button"
            :aria-pressed="selectedInvestigation?.id === item.id" @click="chooseInvestigation(item)">
            <span><strong>{{ item.problem }}</strong><small>Updated {{ formatInvestigationDate(item.updated_at) }}</small></span>
            <span class="context-status" :class="{ open: item.status === 'OPEN' }">{{ item.status }}</span>
          </button>
        </div>
        <p v-if="investigationListBusy" role="status" class="context-message">Searching investigations…</p>
        <p v-else-if="investigationListError" role="alert" class="context-message">{{ investigationListError }}</p>
        <p v-else-if="!investigationOptions.length" class="context-message">No matching investigations. Try another search or choose General RCA chat.</p>
        <button v-if="investigationListHasMore" class="context-load-more" type="button"
          :disabled="investigationListBusy" @click="loadMoreInvestigations">Load more investigations</button>
      </div>
      <RouterLink v-if="selectedInvestigation" class="context-detail-link"
        :to="'/rca/investigations/' + selectedInvestigation.id" target="_blank" rel="noopener">
        View investigation details ↗
      </RouterLink>
    </section>
    <p v-else-if="investigationId">Investigation:
      <RouterLink :to="'/rca/investigations/' + investigationId">{{ investigationId }}</RouterLink>.
      Each request captures its current incident and evidence selection.
      <span v-if="directEntry && chatStarted"> This chat is fixed to this investigation; use New Chat to choose another context.</span>
    </p>
    <p v-else-if="directEntry && chatStarted">General RCA chat · This chat has no investigation context. Use New Chat to select one.</p>
    <button :disabled="busy || !!pending || uncertain" @click="requestNewChat">
      New Chat
    </button>
    <dialog ref="newChatDialog" aria-labelledby="new-chat-title">
      <h2 id="new-chat-title">Start a new chat?</h2>
      <p>Clear the current conversation and RCA output?
         Saved history, audit records, and execution evidence will remain available.</p>
      <button @click="confirmNewChat">Yes</button>
      <button autofocus @click="newChatDialog?.close()">No</button>
    </dialog>
    <fieldset class="investigation-modes" :disabled="!ready || busy || !!pending || uncertain">
      <legend>Investigation mode</legend>
      <div class="mode-options">
        <label v-for="profile in profiles" :key="profile.mode" :class="{ selected: mode === profile.mode }">
          <input type="radio" name="rca-mode" :value="profile.mode" v-model="mode" />
          {{ profile.label }}<span v-if="profile.mode === 'performance'"> · default</span>
        </label>
      </div>
      <template v-if="selectedProfile">
        <p>{{ selectedProfile.description }}</p>
        <p class="mode-limits">Effective maximums: {{ selectedProfile.limits.MAX_MODEL_TURNS }} model turns ·
          {{ selectedProfile.limits.MAX_DISCOVERY_CALLS }} discovery requests ·
          {{ selectedProfile.limits.MAX_EXECUTION_REQUESTS }} execution requests ·
          {{ selectedProfile.limits.MAX_RAG_SEARCHES }} RAG searches</p>
        <p v-if="selectedProfile.limited_by_configuration">Administrator limits reduce this mode's preset.</p>
        <p v-if="!investigationId">RAG searches are available when this chat is linked to an investigation.</p>
      </template>
      <p v-else>Loading server limits…</p>
    </fieldset>
    <button v-if="!ready && error" @click="initialize">Reload investigation modes</button>
    <label>
      <input type="checkbox" v-model="reviewInputArgs" :disabled="busy || !!pending || uncertain" />
      Review Input Args
    </label>
    <p>Unchecked by default. When checked, review and edit collector arguments before execution.
       When unchecked, auto-approved collectors proceed automatically; their arguments appear with the result.
       AI-assisted collectors still require execution confirmation.</p>
    <label>
      <input type="checkbox" v-model="redactEvidence" :disabled="busy || !!pending || uncertain" />
      Redact sensitive values in collector evidence
    </label>
    <p>Checked by default. If unchecked, successful collector results are saved and sent to the configured model without sensitive-value redaction. Use this only when you are comfortable retaining and sharing those values.</p>
    <p>Collector tool calling must be enabled on the API server. Active requests reconnect when you return to this page in the same browser tab.</p>
    <div class="conversation">
      <template v-for="({ message, showProgress }, index) in conversation" :key="index">
      <article v-if="message" class="message">
        <strong>{{ message.role === 'user' ? 'You' : 'LogHawk' }}</strong>
        <section v-if="message.execution" class="execution-summary">
          <strong>Execution status · reported by LogHawk server</strong>
          <p>{{ message.execution.summary }}</p>
          <p>Discovery calls: {{ message.execution.discovery_calls }} · Request: {{ message.execution.request_id }}</p>
          <div v-if="message.execution.loop_progress">
            <strong>{{ message.execution.loop_progress.partial ? "Partial investigation" : "Investigation progress" }}</strong>
            <p>Model turns: {{ message.execution.loop_progress.model_turns }} / {{ message.execution.loop_progress.limits.MAX_MODEL_TURNS }}
              · Discovery requests: {{ message.execution.loop_progress.discovery_calls }} / {{ message.execution.loop_progress.limits.MAX_DISCOVERY_CALLS }}
              · Execution requests: {{ message.execution.loop_progress.execution_requests }} / {{ message.execution.loop_progress.limits.MAX_EXECUTION_REQUESTS }}
              · RAG searches: {{ message.execution.loop_progress.knowledge_searches }}<template v-if="message.execution.loop_progress.limits.MAX_RAG_SEARCHES != null"> / {{ message.execution.loop_progress.limits.MAX_RAG_SEARCHES }}</template></p>
            <p v-if="message.execution.loop_progress.stop_reason">{{ message.execution.loop_progress.stop_reason }}</p>
          </div>
          <details v-if="message.execution.assessment">
            <summary>Saved investigation assessment · validated evidence references</summary>
            <p>References are checked against saved evidence. Interpretations still require human judgment.</p>
            <h4>Observations</h4><ul><li v-for="(claim, n) in message.execution.assessment.observations" :key="n">{{ claim.text }} <code>{{ claim.references.join(", ") }}</code></li></ul>
            <h4>Hypotheses</h4><ul><li v-for="(claim, n) in message.execution.assessment.hypotheses" :key="n">{{ claim.text }} <code>{{ claim.references.join(", ") }}</code></li></ul>
            <h4>Contradictions</h4><ul><li v-for="(claim, n) in message.execution.assessment.contradictions" :key="n">{{ claim.text }} <code>{{ claim.references.join(", ") }}</code></li></ul>
            <h4>Missing evidence</h4><ul><li v-for="(item, n) in message.execution.assessment.missing_evidence" :key="n">{{ item }}</li></ul>
            <h4>Conclusion</h4><p>{{ message.execution.assessment.conclusion.text }} <code>{{ message.execution.assessment.conclusion.references.join(", ") }}</code></p>
            <h4>Next steps</h4><ul><li v-for="(item, n) in message.execution.assessment.next_steps" :key="n">{{ item }}</li></ul>
          </details>
          <ul v-if="message.execution.events.length">
            <li v-for="(event, eventIndex) in message.execution.events" :key="eventIndex">
              <span v-if="event.tool_type === 'knowledge_search'">RAG search · </span>
              {{ event.collector === 'unknown' ? 'Tool request' : event.collector }}
              <span v-if="event.version"> · Version {{ event.version }}</span> — {{ event.status }}
              <span v-if="event.run_id"> · Run {{ event.run_id }}</span>
              <div v-if="event.reason">{{ event.reason }}</div>
              <details v-if="event.arguments">
                <summary>Input arguments</summary>
                <pre>{{ JSON.stringify(event.arguments, null, 2) }}</pre>
              </details>
              <details v-if="event.evidence">
                <summary>{{ event.tool_type === "knowledge_search" ? "RAG excerpts · background knowledge" : event.tool_type === "assessment" ? "Assessment" : "Collector evidence" }}</summary>
                <p>
                  {{ event.tool_type === "knowledge_search" || message.execution.evidence_redaction_enabled
                    ? "Sensitive-value redaction was enabled for this request; output may also be truncated."
                    : "Sensitive-value redaction was disabled for this request; output may also be truncated." }}
                </p>
                <pre>{{ formatEvidence(event.evidence) }}</pre>
              </details>
            </li>
          </ul>
          <p>The assistant's narrative below is not execution confirmation.</p>
        </section>
        <div class="message-text">{{ message.content }}</div>
        <div v-if="message.explanation?.can_retry || message.explanation?.can_regenerate">
          <button :disabled="busy || !!pending || uncertain" @click="retryExplanation(message)">
            {{ message.explanation?.can_regenerate
              ? "Regenerate AI Explanation" : "Retry AI Explanation" }}
          </button>
          <p>Uses saved evidence and the API server's currently loaded model settings.
            Collectors will not run again.</p>
        </div>
      </article>
      <RcaLiveProgress v-if="showProgress" :request-id="activeId" :mode="requestMode" :progress="progress" :investigation-id="requestInvestigationId"
        :started-at="startedAt" :checked-at="checkedAt" :connection-message="connectionMessage"
        :waiting-for-action="waitingForAction" :operation-label="operationLabel">
        <section v-if="pending?.pending" class="confirmation">
          <h2>{{ pending.pending.review_arguments === false
            ? "Confirm collector execution" : "Review input arguments" }}</h2>
          <p>{{ pending.pending.collector }} · Version {{ pending.pending.version }}</p>
          <p v-if="pending.pending.review_arguments !== false">This reviews input arguments only. Collector code approval and enabled status are unchanged.</p>
          <p v-else>The AI-assisted execution policy requires confirmation. Input arguments are read-only.</p>
          <label for="collector-argument-review">Arguments (JSON)</label>
          <textarea id="collector-argument-review" v-model="argumentDraft" rows="8" :disabled="busy || uncertain" :readonly="pending.pending.review_arguments === false" spellcheck="false" />
          <p v-if="pending.pending.review_arguments !== false">Leave masked values unchanged to retain them. Replace them with actual input if editing them.</p>
          <button :disabled="busy || uncertain" @click="decide(true)">Confirm &amp; Execute</button>
          <button :disabled="busy || uncertain" @click="decide(false)">Cancel</button>
        </section>
      </RcaLiveProgress>
      </template>
    </div>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-if="uncertain">The request outcome is uncertain. Progress checks continue for the same request.
      <RouterLink :to="`/rca/history/${activeId}`">Review its saved history</RouterLink>.</p>

    <form @submit.prevent="send">
      <label for="rca-message">Message</label>
      <textarea id="rca-message" v-model="draft" rows="4" :disabled="!ready || busy || !!pending || uncertain"
        placeholder="Help me investigate Kubernetes pod health…" />
      <button type="submit" :disabled="!ready || busy || !!pending || uncertain || !draft.trim()" aria-describedby="rca-send-status">Send</button>
      <p id="rca-send-status" role="status">{{ blockedReason }}</p>
    </form>
  </section>
</template>

<style scoped>
.rca-chat { max-width: 1000px; }
.investigation-context-picker { position: relative; max-width: 680px; margin: .75rem 0; }
.investigation-context-picker > label, .context-search { display: block; font-weight: 700; margin-bottom: .35rem; }
.context-trigger { display: flex; align-items: center; gap: .65rem; width: 100%; text-align: left; padding: .7rem .85rem; border: 1px solid #51716f; border-radius: .5rem; color: #e8edf2; background: #17232d; }
.context-choice { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.context-chevron { color: #8fe5ce; }
.context-help, .context-option small, .context-message { color: #aebbc7; font-size: .9rem; }
.context-menu { position: relative; z-index: 2; margin-top: .35rem; padding: .75rem; border: 1px solid #51716f; border-radius: .5rem; background: #101a22; box-shadow: 0 8px 24px #0008; }
.context-search input { display: block; width: 100%; box-sizing: border-box; margin-top: .35rem; padding: .6rem; color: #e8edf2; background: #17232d; border: 1px solid #344553; border-radius: .4rem; font: inherit; }
.context-option-list { display: grid; gap: .35rem; max-height: 18rem; overflow-y: auto; margin-top: .65rem; }
.context-option { display: flex; align-items: center; justify-content: space-between; gap: .75rem; width: 100%; padding: .65rem; text-align: left; color: #e8edf2; background: #17232d; border: 1px solid #344553; border-radius: .4rem; }
.context-option:hover, .context-option[aria-pressed="true"] { border-color: #8fe5ce; background: #203d3b; }
.context-option strong, .context-option small { display: block; }
.context-status { flex: 0 0 auto; padding: .15rem .4rem; border-radius: 1rem; color: #d3dce5; background: #334451; font-size: .75rem; }
.context-status.open { color: #8fe5ce; background: #203d3b; }
.context-load-more { margin-top: .65rem; }
.context-detail-link { display: inline-block; margin-top: .45rem; }
.message, .confirmation { margin: 1rem 0; padding: 1rem; border: 1px solid #b8c3d1; border-radius: 8px; }
.investigation-modes { margin: 1rem 0; border: 1px solid #a9c9c5; border-radius: .65rem; padding: 1rem; }
.investigation-modes legend { font-weight: 700; }
.mode-options { display: flex; gap: .6rem; flex-wrap: wrap; }
.mode-options label { border: 1px solid #a9c9c5; border-radius: .4rem; padding: .6rem .8rem; cursor: pointer; }
.mode-options label.selected { border-color: #0f766e; outline: 1px solid #0f766e; }
.mode-options input { accent-color: #0f766e; }
.mode-limits { font-size: .9rem; }
.execution-summary { padding: .75rem; margin: .5rem 0; border-left: 4px solid #3684bd; overflow-wrap: anywhere; }
.message-text { white-space: pre-wrap; overflow-wrap: anywhere; margin-top: .5rem; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
label, textarea { display: block; }
textarea { width: 100%; box-sizing: border-box; margin: .5rem 0; }
button { margin-right: .5rem; }
</style>
