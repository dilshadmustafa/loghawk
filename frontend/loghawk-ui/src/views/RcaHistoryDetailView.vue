<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { getRcaHistoryDetail, retryRcaExplanation, regenerateRcaExplanation } from "../services/api";
import type { RcaHistoryDetail, RcaLimits } from "../types/api";
const route = useRoute();
const record = ref<RcaHistoryDetail | null>(null);
const busy = ref(false), error = ref(""), notice = ref("");
const explaining = ref(false);
const tabs = ["Overview", "Evidence & Inputs", "AI Explanations", "Timeline", "LLM Usage"];
const tab = ref("Overview");
const stageLabel = (stage: string) =>
  ({
    narration_correction: "Narration correction",
    empty_response_recovery: "Empty-response recovery",
    investigation_evaluation: "Investigation evaluation / next action",
    initial_explanation: "Initial explanation",
    explanation_retry: "Explanation retry",
    explanation_regeneration: "Explanation regeneration",
  } as Record<string, string>)[stage] ?? stage;
const pretty = (value: unknown) => value == null ? "Not recorded" : JSON.stringify(value, null, 2);
const evidence = (value: string | null) => { if (value == null) return "Not recorded"; try { return pretty(JSON.parse(value)); } catch { return value; } };
const time = (value: string | null) => value ? new Date(value).toLocaleString() : "Not recorded";
const savedMode = computed(() => record.value?.settings?.mode);
const effectiveLimits = computed(() => record.value?.settings?.effective_limits as Partial<RcaLimits> | undefined);
const explanations = computed(() => record.value?.events.filter(e => e.status.startsWith("EXPLANATION_")) ?? []);
const collectors = computed(() => record.value?.events.filter(e =>
  !e.status.startsWith("EXPLANATION_") && (e.tool_type ? ["collector", "knowledge_search"].includes(e.tool_type) : !["list_collectors", "request_settings", "stop_without_execution"].includes(e.collector_name))) ?? []);
const timeline = computed(() => {
  const r = record.value;
  if (!r) return [];
  return [
    { id: r.id, at: r.created_at, label: "Request created", detail: `${r.userid} · ${r.origin_software}` },
    ...r.events.map(e => ({ id: e.id, at: e.started_at, label: `${e.tool_type === "knowledge_search" ? "RAG search" : e.collector_name} · ${e.status}`, detail: e.run_id ? `Run: ${e.run_id}` : e.id })),
    ...r.invocations.map(i => ({ id: i.id, at: i.started_at, label: `${stageLabel(i.stage)} · ${i.status}`, detail: `${i.configured_model ?? 'Not recorded'} · Invocation ${i.invocation_number}` }))
  ].sort((a, b) => a.at.localeCompare(b.at) || a.id.localeCompare(b.id));
});
let generation = 0;
async function load() {
  const current = ++generation;
  busy.value = true; error.value = ""; record.value = null;
  try { const value = await getRcaHistoryDetail(String(route.params.id)); if (current === generation) record.value = value; }
  catch (e) { if (current === generation) error.value = e instanceof Error ? e.message : String(e); }
  finally { if (current === generation) busy.value = false; }
}
watch(() => route.params.id, () => { tab.value = "Overview"; notice.value = ""; void load(); }, { immediate: true });
async function explainAgain() {
  const current = record.value;
  const explanation = current?.explanation;
  if (!current || busy.value || explaining.value ||
      (!explanation?.can_retry && !explanation?.can_regenerate)) return;

  const requestId = current.id;
  const pageGeneration = generation;
  explaining.value = true;
  error.value = "";
  notice.value = "";
  try {
    const action = explanation.can_regenerate
      ? regenerateRcaExplanation : retryRcaExplanation;
    await action(requestId);
    if (pageGeneration !== generation) return;
    const refreshed = await getRcaHistoryDetail(requestId);
    if (pageGeneration === generation) {
      record.value = refreshed;
      notice.value = "Explanation attempt saved. Collectors were not rerun.";
    }
  } catch (e) {
    if (pageGeneration === generation) {
      error.value = `Could not receive the explanation: ${
        e instanceof Error ? e.message : String(e)
      }. Refresh history to check the attempt before trying again.`;
      if (record.value?.explanation) {
        record.value.explanation.can_retry = false;
        record.value.explanation.can_regenerate = false;
      }
    }
  } finally {
    explaining.value = false;
  }
}
async function copy(text: string) {
  try { await navigator.clipboard.writeText(text); notice.value = "Copied to clipboard."; }
  catch { notice.value = "Clipboard unavailable. Select the displayed text and copy it manually."; }
}
function download() {
  if (!record.value) return;
  const url = URL.createObjectURL(new Blob([pretty(record.value)], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url; link.download = `rca-${record.value.id}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
</script>
<template>
  <section>
    <h1>RCA Request History</h1>
    <RouterLink to="/rca/history">Back to History</RouterLink>
    <button :disabled="busy || explaining" @click="load">Refresh</button>
    <button v-if="record" @click="copy(pretty(record))">Copy JSON</button>
    <button v-if="record" @click="download">Download JSON</button>
    <p v-if="busy" role="status">Loading request…</p><p v-if="error" role="alert">{{ error }}</p>
    <p role="status">{{ notice }}</p>
    <template v-if="record">
      <p>Request ID: <code>{{ record.id }}</code> · {{ record.userid }} · {{ record.origin_software }}</p>
      <section v-if="record.investigation">
        <RouterLink :to="`/rca/investigations/${record.investigation.investigation_id}`">Open Investigation</RouterLink>
        <details><summary>Saved request context snapshot</summary><pre>{{ pretty(record.investigation.snapshot) }}</pre></details>
        <p>This snapshot preserves the inputs selected for this request. Regeneration does not add later attachments.</p>
      </section>
      <nav aria-label="History sections"><button v-for="name in tabs" :key="name" :aria-pressed="tab === name" @click="tab = name">{{ name }}</button></nav>
      <section v-if="tab === 'Overview'">
        <h2>Question</h2><p class="text">{{ record.question_text ?? 'Not recorded' }}</p>
        <p>Started: {{ time(record.created_at) }} · Updated: {{ time(record.updated_at) }} · Request status: {{ record.status }}</p>
        <section v-if="savedMode" class="saved-mode">
          <h2>Investigation mode: <span class="mode-name">{{ savedMode }}</span></h2>
          <p v-if="effectiveLimits">Saved effective maximums: {{ effectiveLimits.MAX_MODEL_TURNS ?? 'Not recorded' }} model turns ·
            {{ effectiveLimits.MAX_DISCOVERY_CALLS ?? 'Not recorded' }} discovery requests ·
            {{ effectiveLimits.MAX_EXECUTION_REQUESTS ?? 'Not recorded' }} execution requests ·
            {{ effectiveLimits.MAX_RAG_SEARCHES ?? 'Not recorded' }} RAG searches</p>
        </section>
        <h2>Execution status reported by server</h2><p>{{ record.execution.summary }}</p>
        <section v-if="record.execution.loop_progress">
          <h2>{{ record.execution.loop_progress.partial ? "Partial investigation" : "Investigation progress" }}</h2>
          <p>Model turns: {{ record.execution.loop_progress.model_turns }} / {{ record.execution.loop_progress.limits.MAX_MODEL_TURNS }}
            · Discovery requests: {{ record.execution.loop_progress.discovery_calls }} / {{ record.execution.loop_progress.limits.MAX_DISCOVERY_CALLS }}
            · Execution requests: {{ record.execution.loop_progress.execution_requests }} / {{ record.execution.loop_progress.limits.MAX_EXECUTION_REQUESTS }}
            · RAG searches: {{ record.execution.loop_progress.knowledge_searches }}<template v-if="record.execution.loop_progress.limits.MAX_RAG_SEARCHES != null"> / {{ record.execution.loop_progress.limits.MAX_RAG_SEARCHES }}</template></p>
          <p>{{ record.execution.loop_progress.stop_reason }}</p>
        </section>
        <section v-if="record.execution.assessment">
          <h2>Saved investigation assessment</h2>
          <p>Evidence references were validated against saved records. This checks provenance, not whether an AI interpretation is correct.</p>
          <h3>Observations</h3><ul><li v-for="(claim, n) in record.execution.assessment.observations" :key="n">{{ claim.text }} <code>{{ claim.references.join(", ") }}</code></li></ul>
          <h3>Hypotheses</h3><ul><li v-for="(claim, n) in record.execution.assessment.hypotheses" :key="n">{{ claim.text }} <code>{{ claim.references.join(", ") }}</code></li></ul>
          <h3>Contradictions</h3><ul><li v-for="(claim, n) in record.execution.assessment.contradictions" :key="n">{{ claim.text }} <code>{{ claim.references.join(", ") }}</code></li></ul>
          <h3>Missing evidence</h3><ul><li v-for="(item, n) in record.execution.assessment.missing_evidence" :key="n">{{ item }}</li></ul>
          <h3>Conclusion</h3><p>{{ record.execution.assessment.conclusion.text }} <code>{{ record.execution.assessment.conclusion.references.join(", ") }}</code></p>
          <h3>Next steps</h3><ul><li v-for="(item, n) in record.execution.assessment.next_steps" :key="n">{{ item }}</li></ul>
        </section>
        <h2>Latest saved response</h2><p class="text">{{ record.response_text ?? 'Not recorded; inspect saved explanation events below.' }}</p>
        <p>Hypotheses and conclusions are shown within the original explanation. They are AI interpretations, not independently verified findings.</p>
        <details><summary>Settings at request creation</summary><pre>{{ pretty(record.settings) }}</pre></details>
        <p>Older requests may lack questions, settings or model-usage metadata. Not recorded means unavailable, not zero.</p>
      </section>
      <section v-if="tab === 'Evidence & Inputs'">
        <p v-if="record.settings?.redact_evidence !== false">Sensitive-value redaction was enabled for this request. Evidence may also be compacted for model input or limited by its source/runner. Character counts are not token counts.</p>
        <p v-else>Sensitive-value redaction was disabled for this request. Collector results may contain sensitive values and were available to the configured model. Evidence may also be compacted for model input or limited by its source/runner.</p>
        <article v-for="event in collectors" :key="event.id">
          <h2>{{ event.tool_type === "knowledge_search" ? "RAG search" : event.collector_name }} · Version {{ event.collector_version }} · {{ event.status }}</h2>
          <p v-if="event.tool_type === 'knowledge_search'">Saved document excerpts are background guidance, not live observations. Source, content hash and excerpt truncation are recorded below.</p>
          <p>Event: {{ event.id }} · Run: {{ event.run_id ?? 'No run recorded' }}</p>
          <p>{{ time(event.started_at) }} — {{ time(event.completed_at) }}</p>
          <h3>{{ event.status === 'AWAITING_CONFIRMATION' ? 'Proposed arguments' : event.status === 'CONFIRMED' ? 'Confirmed arguments' : 'Recorded arguments' }}</h3>
          <pre>{{ pretty(event.arguments) }}</pre><button @click="copy(pretty(event.arguments))">Copy arguments</button>
          <details><summary>Saved evidence or event information</summary><pre>{{ evidence(event.result_text) }}</pre><button @click="copy(event.result_text ?? '')">Copy evidence</button></details>
          <p v-if="!record.evidence_views.some(v => v.tool_call_id === event.id)">Truncation metadata: Not recorded for this event.</p>
          <div v-for="view in record.evidence_views.filter(v => v.tool_call_id === event.id)" :key="view.id">
            <h3>{{ view.stage }}</h3><p>Invocation: {{ view.invocation_id ?? 'Prepared after collection; no model invocation yet' }}</p>
            <dl><dt>Saved evidence characters</dt><dd>{{ view.saved_evidence_chars ?? 'Not recorded' }}</dd>
              <dt>Input / output characters</dt><dd>{{ view.input_chars ?? 'Not recorded' }} / {{ view.output_chars }}</dd>
              <dt>Effective character limit</dt><dd>{{ view.effective_limit_chars ?? 'Not recorded' }}</dd></dl>
            <pre>{{ pretty(view.transformation) }}</pre>
            <details><summary>Prepared model evidence</summary><pre>{{ evidence(view.model_evidence_text) }}</pre><button @click="copy(view.model_evidence_text)">Copy model evidence</button></details>
          </div>
        </article>
        <p v-if="!collectors.length">No collector execution, RAG search or input review events recorded.</p>
      </section>
      <section v-if="tab === 'AI Explanations'">
        <p>Every recorded explanation attempt is retained. A failed explanation does not mean collection failed.</p>
        <button
          v-if="record.explanation?.can_retry || record.explanation?.can_regenerate"
          :disabled="busy || explaining"
          @click="explainAgain">
          {{ record.explanation?.can_regenerate
            ? "Regenerate AI Explanation" : "Retry AI Explanation" }}
        </button>
        <p v-if="explaining" role="status">Generating explanation from saved evidence…</p>
        <p>Uses the API server's currently loaded model settings.
          After changing model settings in .env, restart the API server.
          Collectors will not run again.</p>
        <article v-for="event in explanations" :key="event.id">
          <h2>{{ event.status }}</h2><p>{{ time(event.started_at) }} — {{ time(event.completed_at) }}</p>
          <h3>Attempt and model metadata</h3><pre>{{ pretty(event.arguments) }}</pre>
          <h3>Saved explanation or error</h3><p class="text">{{ event.result_text ?? 'Not recorded' }}</p>
          <button @click="copy(event.result_text ?? '')">Copy explanation</button>
        </article><p v-if="!explanations.length">No explanation attempts recorded.</p>
      </section>
      <section v-if="tab === 'Timeline'">
        <ol><li v-for="entry in timeline" :key="entry.id"><strong>{{ time(entry.at) }} · {{ entry.label }}</strong><p>{{ entry.detail }}</p></li></ol>
        <p>Events are ordered by recorded timestamp. View the other tabs for evidence and full attempt details.</p>
      </section>
      <section v-if="tab === 'LLM Usage'">
        <p>Historical settings captured for each invocation. Dispatch calls include discovery; invocation counts include correction turns and explanation retries. PREPARING means the API call may not have started. Token usage is shown only when reported by the provider.</p>
        <article v-for="invocation in record.invocations" :key="invocation.id">
          <h2>{{ invocation.invocation_number }} · {{ stageLabel(invocation.stage) }} · {{ invocation.status }}</h2>
          <p v-if="invocation.stage === 'narration_correction'">
            The previous model response contained narration instead of a required tool call.
            LogHawk requested a correction, reusing the existing conversation and discovery results.
          </p>
          <p v-if="invocation.stage === 'empty_response_recovery'">
            The previous model response contained neither text nor tool calls. LogHawk made one bounded recovery attempt using saved evidence. Previously completed collectors were not rerun.
          </p>
          <p v-if="invocation.stage === 'investigation_evaluation'">The model evaluates saved context and available evidence, then chooses discovery, collection, RAG retrieval or a finish assessment. A completed invocation does not mean a collector executed.</p>
          <p>Invocation: {{ invocation.id }} · Explanation attempt: {{ invocation.explanation_attempt_id ?? 'Not applicable' }}</p>
          <p>Provider: {{ invocation.provider ?? 'Not recorded' }} · Configured model: {{ invocation.configured_model ?? 'Not recorded' }} · Returned model: {{ invocation.returned_model ?? 'Not reported' }}</p>
          <p>{{ time(invocation.started_at) }} — {{ time(invocation.completed_at) }} · Duration including preparation: {{ invocation.duration_ms == null ? 'Not recorded' : (invocation.duration_ms / 1000).toFixed(2) + ' seconds' }}</p>
          <table><thead><tr><th>Setting</th><th>Recorded value</th></tr></thead><tbody><tr v-for="(value, key) in invocation.settings ?? {}" :key="key"><td>{{ key }}</td><td>{{ typeof value === "object" ? pretty(value) : value }}</td></tr></tbody></table>
          <h3>Provider token usage</h3><pre>{{ invocation.usage == null ? 'Not reported' : pretty(invocation.usage) }}</pre>
          <p v-if="invocation.error_text" class="text">{{ invocation.error_text }}</p>
          <details v-for="view in record.evidence_views.filter(v => v.invocation_id === invocation.id)" :key="view.id">
            <summary>{{ view.stage }} · {{ view.output_chars }} characters · Evidence {{ view.id }}</summary>
            <pre>{{ pretty(view.transformation) }}</pre><pre>{{ evidence(view.model_evidence_text) }}</pre>
          </details>
        </article><p v-if="!record.invocations.length">Per-invocation usage was not recorded for this request. Older explanation events may still identify their model/provider.</p>
      </section>
    </template>
  </section>
</template>
<style scoped>
.saved-mode { border-left: 3px solid #0f766e; padding: .25rem 1rem; margin: 1rem 0; } .mode-name { text-transform: capitalize; }
button { margin: .3rem .5rem .3rem 0; } nav { display: flex; flex-wrap: wrap; margin: 1rem 0; }
button[aria-pressed="true"] { outline: 2px solid #3684bd; }
article { border: 1px solid #64748b; border-radius: .5rem; padding: 1rem; margin: 1rem 0; }
pre, .text { white-space: pre-wrap; overflow-wrap: anywhere; } pre { max-height: 36rem; overflow-y: auto; }
code, td, dd, li p { overflow-wrap: anywhere; } table { width: 100%; } dt { font-weight: bold; } dd { margin-bottom: .5rem; }
</style>
