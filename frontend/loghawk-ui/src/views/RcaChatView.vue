<script setup lang="ts">
import { computed, ref } from "vue";
import { ApiError, sendRcaMessage, confirmRcaCollector, retryRcaExplanation } from "../services/api";
import type { RcaMessage, RcaAnswer, RcaExecutionSummary } from "../types/api";
type ChatEntry = RcaMessage & { requestId?: string; execution?: RcaExecutionSummary; explanation?: RcaAnswer["explanation"] };
const messages = ref<ChatEntry[]>([]);
const draft = ref("");
const busy = ref(false);
const error = ref("");
const pending = ref<RcaAnswer | null>(null);
const uncertain = ref(false);
const argumentDraft = ref("");
const reviewInputArgs = ref(true);
const blockedReason = computed(() => {
  if (busy.value) return "A request is processing. Please wait.";
  if (uncertain.value) return "Sending is paused because the previous request outcome is uncertain. Check the error above.";
  if (pending.value) return "Choose Confirm & Execute or Cancel before sending another message.";
  if (!draft.value.trim()) return "Type a message to enable Send.";
  return "";
});
function formatEvidence(evidence: string): string {
  try {
    return JSON.stringify(JSON.parse(evidence), null, 2);
  } catch {
    return evidence;
  }
}

function receive(answer: RcaAnswer) {
  pending.value = answer.pending ? answer : null;
  argumentDraft.value = answer.pending ? JSON.stringify(answer.pending.arguments, null, 2) : "";
  if (answer.explanation && answer.request_id) {
    // Keep historical execution snapshots; only the latest explanation can be retried.
    for (const entry of messages.value) {
      if (entry.requestId === answer.request_id && entry.explanation) entry.explanation.can_retry = false;
    }
  }
  if (answer.response || answer.execution) messages.value.push({
    role: "assistant", content: answer.response || "", requestId: answer.request_id,
    execution: answer.execution, explanation: answer.explanation,
  });
}
async function retryExplanation(message: ChatEntry) {
  if (busy.value || pending.value || uncertain.value || !message.requestId || !message.explanation?.can_retry) return;
  busy.value = true;
  error.value = "";
  try { receive(await retryRcaExplanation(message.requestId)); }
  catch (e) {
    error.value = `Could not receive the AI explanation: ${e instanceof Error ? e.message : String(e)}. This retry does not execute collectors.`;
  }
  finally { busy.value = false; }
}

async function send() {
  if (busy.value || pending.value || uncertain.value || !draft.value.trim()) return;
  const submitted = draft.value.trim();
  messages.value.push({ role: "user", content: submitted });
  draft.value = "";
  busy.value = true;
  error.value = "";
  try { receive(await sendRcaMessage(messages.value.map(({ role, content }) => ({ role, content })), reviewInputArgs.value)); }
  catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
    if (e instanceof ApiError && e.executionNotStarted) {
      // The API explicitly rejected this input before creating an execution request.
      messages.value.pop();
      draft.value = submitted;
      uncertain.value = false;
    } else {
      uncertain.value = true;
    }
  }
  finally { busy.value = false; }
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
  busy.value = true;
  try { receive(await confirmRcaCollector(proposal.request_id, proposal.pending.confirmation_id, approved, argumentsToRun)); }
  catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
    // 422 is request/argument validation: no confirmation was consumed.
    if (!(e instanceof ApiError && e.status === 422)) uncertain.value = true;
  }
  finally { busy.value = false; }
}

</script>

<template>
  <section class="rca-chat">
    <h1>AI-Assisted RCA</h1>
    <label>
      <input type="checkbox" v-model="reviewInputArgs" :disabled="busy || !!pending || uncertain" />
      Review Input Args
    </label>
    <p>Discuss an operational issue and gather evidence using approved collectors.
       When checked, review and edit collector arguments before execution.
       When unchecked, auto-approved collectors proceed automatically; their arguments appear with the result.
       AI-assisted collectors still require execution confirmation.</p>
    <p>Collector tool calling must be enabled on the API server. Chat is kept while this page is open.</p>
    <div aria-live="polite">
      <article v-for="(message, index) in messages" :key="index" class="message">
        <strong>{{ message.role === 'user' ? 'You' : 'LogHawk' }}</strong>
        <section v-if="message.execution" class="execution-summary">
          <strong>Execution status · reported by LogHawk server</strong>
          <p>{{ message.execution.summary }}</p>
          <p>Discovery calls: {{ message.execution.discovery_calls }} · Request: {{ message.execution.request_id }}</p>
          <ul v-if="message.execution.events.length">
            <li v-for="(event, eventIndex) in message.execution.events" :key="eventIndex">
              {{ event.collector === 'unknown' ? 'Tool request' : event.collector }}
              <span v-if="event.version"> · Version {{ event.version }}</span> — {{ event.status }}
              <span v-if="event.run_id"> · Run {{ event.run_id }}</span>
              <div v-if="event.reason">{{ event.reason }}</div>
              <details v-if="event.arguments">
                <summary>Input arguments</summary>
                <pre>{{ JSON.stringify(event.arguments, null, 2) }}</pre>
              </details>
              <details v-if="event.evidence">
                <summary>Collector evidence</summary>
                <p>Redacted collector results; output may be truncated.</p>
                <pre>{{ formatEvidence(event.evidence) }}</pre>
              </details>
            </li>
          </ul>
          <p>The assistant's narrative below is not execution confirmation.</p>
        </section>
        <div class="message-text">{{ message.content }}</div>
        <div v-if="message.explanation?.can_retry">
          <button :disabled="busy || !!pending || uncertain" @click="retryExplanation(message)">Retry AI Explanation</button>
          <p>Uses saved evidence. Collectors will not run again.</p>
        </div>
      </article>
    </div>
    <p v-if="busy" role="status">Processing request…</p>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-if="uncertain">The request could not be completed or its result could not be received.
      Check execution audit records before starting another request; execution may already have occurred.</p>
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
    <form @submit.prevent="send">
      <label for="rca-message">Message</label>
      <textarea id="rca-message" v-model="draft" rows="4" :disabled="busy || !!pending || uncertain"
        placeholder="Help me investigate Kubernetes pod health…" />
      <button type="submit" :disabled="busy || !!pending || uncertain || !draft.trim()" aria-describedby="rca-send-status">Send</button>
      <p id="rca-send-status" role="status">{{ blockedReason }}</p>
    </form>
  </section>
</template>

<style scoped>
.rca-chat { max-width: 1000px; }
.message, .confirmation { margin: 1rem 0; padding: 1rem; border: 1px solid #b8c3d1; border-radius: 8px; }
.execution-summary { padding: .75rem; margin: .5rem 0; border-left: 4px solid #3684bd; overflow-wrap: anywhere; }
.message-text { white-space: pre-wrap; overflow-wrap: anywhere; margin-top: .5rem; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
label, textarea { display: block; }
textarea { width: 100%; box-sizing: border-box; margin: .5rem 0; }
button { margin-right: .5rem; }
</style>
