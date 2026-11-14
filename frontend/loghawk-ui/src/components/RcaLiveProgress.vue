<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
import type { RcaLiveProgress, RcaLimits } from "../types/api";
const props = defineProps<{
  requestId: string;
  investigationId: string;
  mode: string;
  progress: RcaLiveProgress | null;
  startedAt: string;
  checkedAt: number | null;
  connectionMessage: string;
  waitingForAction: boolean;
  operationLabel: string;
}>();
const now = ref(Date.now());
const ticker = window.setInterval(() => { now.value = Date.now(); }, 1000);
onBeforeUnmount(() => window.clearInterval(ticker));
const live = computed(() => props.progress?.live_progress ?? {});
const terminal = computed(() => !!props.progress?.answer && !props.progress.answer.pending && !props.waitingForAction);
const serverNow = computed(() => now.value + (props.progress && props.checkedAt
  ? Date.parse(props.progress.server_time) - props.checkedAt : 0));
const duration = (seconds: number) => {
  const value = Math.max(0, Math.floor(seconds));
  return value < 60 ? `${value}s` : `${Math.floor(value / 60)}m ${value % 60}s`;
};
const age = (at: unknown, end = serverNow.value) => typeof at === "string" && Number.isFinite(Date.parse(at))
  ? duration((end - Date.parse(at)) / 1000) : "Not recorded";
const remaining = (at: unknown) => typeof at === "string" && Number.isFinite(Date.parse(at))
  ? duration((Date.parse(at) - serverNow.value) / 1000) : null;
const limits = computed(() => (live.value.limits ?? props.progress?.settings?.effective_limits ?? {}) as Partial<RcaLimits>);
const phaseLabels: Record<string, string> = {
  preparing_context: "Preparing investigation context",
  preparing_model_input: "Preparing model input",
  waiting_for_model: "Waiting for model response",
  discovering_collectors: "Discovering collectors",
  searching_knowledge: "Searching runbooks",
  executing_collector: "Running collector",
  awaiting_confirmation: "Awaiting collector confirmation",
  preparing_explanation: "Preparing explanation",
  waiting_for_explanation: "Waiting for explanation from the model",
  completed: "Investigation completed",
  failed: "Investigation ended with an error",
};
const label = computed(() => props.waitingForAction || !props.progress
  ? props.operationLabel : String(live.value.detail || phaseLabels[String(live.value.phase)] || props.progress.status.replaceAll("_", " ")));
const phaseStarted = computed(() => props.waitingForAction ? props.startedAt : live.value.phase_started_at);
const elapsedEnd = computed(() => terminal.value ? Date.parse(props.progress!.updated_at) : serverNow.value);
const counters = computed(() => [
  { label: "Model turns", used: live.value.model_turns, max: limits.value.MAX_MODEL_TURNS },
  { label: "Discovery", used: live.value.discovery_calls, max: limits.value.MAX_DISCOVERY_CALLS },
  { label: "Executions", used: live.value.execution_requests, max: limits.value.MAX_EXECUTION_REQUESTS },
  { label: "RAG searches", used: live.value.knowledge_searches, max: limits.value.MAX_RAG_SEARCHES },
]);
const lastActivity = computed(() => {
  const p = props.progress;
  if (!p) return null;
  return [p.updated_at, String(live.value.updated_at || p.created_at),
    ...p.events.flatMap(e => [e.started_at, e.completed_at || e.started_at]),
    ...p.invocations.flatMap(i => [i.started_at, i.completed_at || i.started_at])].sort().at(-1);
});
const modelWaiting = computed(() => !props.waitingForAction &&
  ["waiting_for_model", "waiting_for_explanation"].includes(String(live.value.phase)));
const slowModel = computed(() => modelWaiting.value && typeof phaseStarted.value === "string" &&
  serverNow.value - Date.parse(phaseStarted.value) >= 30000);
const stageLabel = (stage: string) => ({
  investigation_evaluation: "Assessing evidence and selecting the next action",
  initial_explanation: "Generating explanation",
  explanation_retry: "Retrying explanation",
  explanation_regeneration: "Regenerating explanation",
  narration_correction: "Correcting model response",
  empty_response_recovery: "Recovering an empty model response",
} as Record<string, string>)[stage] ?? stage.replaceAll("_", " ");
const timeline = computed(() => {
  const p = props.progress;
  if (!p) return [];
  return [
    ...p.events.filter(e => e.collector_name !== "request_settings").map(e => ({
      id: e.id, at: e.started_at, status: e.status,
      label: e.tool_type === "knowledge_search" ? "Runbook search" : e.collector_name === "list_collectors" ? "Collector discovery" : e.collector_name,
      summary: e.summary, arguments: e.arguments, runId: e.run_id,
      context: null as Record<string, unknown> | null, timeout: undefined as number | undefined,
      completed: e.completed_at,
    })),
    ...p.invocations.map(i => ({
      id: i.id, at: i.started_at, status: i.status, label: stageLabel(i.stage),
      summary: i.error_text || (i.status === "RUNNING" ? "Waiting for model response" : ""),
      arguments: null as Record<string, unknown> | null, runId: null as string | null,
      context: i.settings?.context_preparation ?? null, timeout: i.settings?.effective_timeout_seconds,
      completed: i.completed_at,
    })),
  ].sort((a, b) => a.at.localeCompare(b.at) || a.id.localeCompare(b.id));
});
const pretty = (value: unknown) => JSON.stringify(value, null, 2);
</script>

<template>
  <section class="live-progress" aria-label="Live investigation">
    <header>
      <h2>Live investigation</h2>
      <span class="mode">{{ mode }}</span>
      <RouterLink :to="`/rca/history/${requestId}`">View RCA History</RouterLink>
    </header>
    <p class="request-id">Request: <code>{{ requestId }}</code></p>
    <p v-if="investigationId" class="request-id">Request investigation:
      <RouterLink :to="`/rca/investigations/${investigationId}`">{{ investigationId }}</RouterLink></p>
    <p v-else class="request-id">This request has no linked investigation.</p>
    <div class="current-step" role="status" aria-live="polite">
      <span class="status-dot" :class="{ complete: terminal }" aria-hidden="true"></span>
      <strong>{{ label }}</strong>
    </div>
    <p class="timers">This step: {{ age(phaseStarted, elapsedEnd) }} · Total elapsed: {{ age(progress?.created_at ?? startedAt, elapsedEnd) }}</p>
    <div v-if="!waitingForAction && !terminal" class="budgets">
      <span v-if="live.processing_deadline_at">Investigation budget remaining: {{ remaining(live.processing_deadline_at) }}</span>
      <span v-else-if="live.phase === 'awaiting_confirmation'">Investigation budget paused for confirmation</span>
      <span v-if="live.explanation_deadline_at">Explanation budget remaining: {{ remaining(live.explanation_deadline_at) }}</span>
      <span v-if="modelWaiting && live.model_call_deadline_at">Model call timeout in: {{ remaining(live.model_call_deadline_at) }}</span>
    </div>
    <p v-if="slowModel" class="slow-notice">Still waiting for the model response ({{ age(phaseStarted) }}).
      <template v-if="live.model_call_deadline_at">This call's timeout is at {{ new Date(String(live.model_call_deadline_at)).toLocaleTimeString() }}.</template>
    </p>
    <dl class="counters">
      <div v-for="counter in counters" :key="counter.label">
        <dt>{{ counter.label }}</dt><dd>{{ counter.used ?? '—' }} / {{ counter.max ?? '—' }}</dd>
      </div>
    </dl>
    <p class="freshness">Status checked {{ checkedAt ? duration((now - checkedAt) / 1000) + ' ago' : '—' }}
      · Last recorded activity {{ lastActivity ? age(lastActivity) + ' ago' : '—' }}</p>
    <p v-if="connectionMessage" class="connection-notice" role="status">{{ connectionMessage }}</p>
    <slot />
    <details :open="!terminal">
      <summary>Activity · {{ timeline.length }} recorded steps</summary>
      <ol class="activity">
        <li v-for="entry in timeline" :key="entry.id">
          <details>
            <summary><strong>{{ entry.label }}</strong> · {{ entry.status.replaceAll('_', ' ') }}</summary>
            <p>{{ new Date(entry.at).toLocaleTimeString() }}<template v-if="entry.completed"> — {{ new Date(entry.completed).toLocaleTimeString() }}</template></p>
            <p v-if="entry.runId">Run: <code>{{ entry.runId }}</code></p>
            <p v-if="entry.timeout">Model call timeout: {{ entry.timeout }} seconds</p>
            <div v-if="entry.arguments"><strong>Recorded arguments</strong><pre>{{ pretty(entry.arguments) }}</pre></div>
            <div v-if="entry.context"><strong>Context prepared for model input</strong><pre>{{ pretty(entry.context) }}</pre></div>
            <RouterLink :to="`/rca/history/${requestId}`">View saved evidence and details</RouterLink>
          </details>
          <p v-if="entry.summary" class="activity-summary">{{ entry.summary }}</p>
        </li>
      </ol>
      <p v-if="!timeline.length">Waiting for the first recorded activity.</p>
    </details>
  </section>
</template>

<style scoped>
.live-progress { border: 1px solid #7cbcb4; border-left: 4px solid #0f766e; border-radius: .65rem; padding: 1rem; margin: 1rem 0; }
header { display: flex; align-items: center; flex-wrap: wrap; gap: .75rem; }
h2 { font-size: 1.1rem; margin: 0; } header a { margin-left: auto; }
.mode { border: 1px solid #7cbcb4; border-radius: 1rem; padding: .15rem .65rem; text-transform: capitalize; }
.request-id, .freshness, .timers { font-size: .9rem; }
.current-step { display: flex; gap: .6rem; align-items: center; margin-top: 1rem; }
.status-dot { width: .65rem; height: .65rem; background: #0f766e; border-radius: 50%; flex-shrink: 0; }
.status-dot.complete { background: #54756f; }
.budgets { display: flex; flex-wrap: wrap; gap: .5rem 1.25rem; font-size: .9rem; }
.counters { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .5rem; }
.counters > div { border: 1px solid #a9c9c5; border-radius: .4rem; padding: .6rem; }
dt { font-size: .85rem; } dd { margin: .35rem 0 0; font-weight: 700; }
.slow-notice, .connection-notice { border-left: 3px solid #b78c35; padding-left: .75rem; }
.activity { padding-left: 1.5rem; } .activity li { margin: .8rem 0; } .activity-summary { margin: .3rem 0; font-size: .9rem; }
summary { cursor: pointer; } pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 18rem; overflow: auto; }
code { overflow-wrap: anywhere; }
@media (max-width: 580px) { .counters { grid-template-columns: repeat(2, minmax(0, 1fr)); } header a { margin-left: 0; } }
</style>
