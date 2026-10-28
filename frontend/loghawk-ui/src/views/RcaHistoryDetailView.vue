<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { getRcaHistoryDetail } from "../services/api";
import type { RcaHistoryDetail } from "../types/api";
const route = useRoute();
const record = ref<RcaHistoryDetail | null>(null);
const busy = ref(false), error = ref(""), notice = ref("");
const tabs = ["Overview", "Evidence & Inputs", "AI Explanations", "Timeline", "LLM Usage"];
const tab = ref("Overview");
const pretty = (value: unknown) => value == null ? "Not recorded" : JSON.stringify(value, null, 2);
const evidence = (value: string | null) => { if (value == null) return "Not recorded"; try { return pretty(JSON.parse(value)); } catch { return value; } };
const time = (value: string | null) => value ? new Date(value).toLocaleString() : "Not recorded";
const explanations = computed(() => record.value?.events.filter(e => e.status.startsWith("EXPLANATION_")) ?? []);
const collectors = computed(() => record.value?.events.filter(e =>
  !e.status.startsWith("EXPLANATION_") && !["list_collectors", "request_settings", "stop_without_execution"].includes(e.collector_name)) ?? []);
const timeline = computed(() => {
  const r = record.value;
  if (!r) return [];
  return [
    { id: r.id, at: r.created_at, label: "Request created", detail: `${r.userid} · ${r.origin_software}` },
    ...r.events.map(e => ({ id: e.id, at: e.started_at, label: `${e.collector_name} · ${e.status}`, detail: e.run_id ? `Run: ${e.run_id}` : e.id })),
    ...r.invocations.map(i => ({ id: i.id, at: i.started_at, label: `${i.stage} · ${i.status}`, detail: `${i.configured_model ?? 'Not recorded'} · Invocation ${i.invocation_number}` }))
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
    <button :disabled="busy" @click="load">Refresh</button>
    <button v-if="record" @click="copy(pretty(record))">Copy JSON</button>
    <button v-if="record" @click="download">Download JSON</button>
    <p v-if="busy" role="status">Loading request…</p><p v-if="error" role="alert">{{ error }}</p>
    <p role="status">{{ notice }}</p>
    <template v-if="record">
      <p>Request ID: <code>{{ record.id }}</code> · {{ record.userid }} · {{ record.origin_software }}</p>
      <nav aria-label="History sections"><button v-for="name in tabs" :key="name" :aria-pressed="tab === name" @click="tab = name">{{ name }}</button></nav>
      <section v-if="tab === 'Overview'">
        <h2>Question</h2><p class="text">{{ record.question_text ?? 'Not recorded' }}</p>
        <p>Started: {{ time(record.created_at) }} · Updated: {{ time(record.updated_at) }} · Request status: {{ record.status }}</p>
        <h2>Execution status reported by server</h2><p>{{ record.execution.summary }}</p>
        <h2>Latest saved response</h2><p class="text">{{ record.response_text ?? 'Not recorded; inspect saved explanation events below.' }}</p>
        <p>Hypotheses and conclusions are shown within the original explanation. They are AI interpretations, not independently verified findings.</p>
        <details><summary>Settings at request creation</summary><pre>{{ pretty(record.settings) }}</pre></details>
        <p>Older requests may lack questions, settings or model-usage metadata. Not recorded means unavailable, not zero.</p>
      </section>
      <section v-if="tab === 'Evidence & Inputs'">
        <p>Saved evidence is redacted before model compaction. Source/runner limits still apply. Redaction masks sensitive values; compaction changes representation; truncation omits content. Character counts are not token counts.</p>
        <article v-for="event in collectors" :key="event.id">
          <h2>{{ event.collector_name }} · Version {{ event.collector_version }} · {{ event.status }}</h2>
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
        <p v-if="!collectors.length">No collector execution or input review events recorded.</p>
      </section>
      <section v-if="tab === 'AI Explanations'">
        <p>Every recorded explanation attempt is retained. A failed explanation does not mean collection failed.</p>
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
          <h2>{{ invocation.invocation_number }} · {{ invocation.stage }} · {{ invocation.status }}</h2>
          <p>Invocation: {{ invocation.id }} · Explanation attempt: {{ invocation.explanation_attempt_id ?? 'Not applicable' }}</p>
          <p>Provider: {{ invocation.provider ?? 'Not recorded' }} · Configured model: {{ invocation.configured_model ?? 'Not recorded' }} · Returned model: {{ invocation.returned_model ?? 'Not reported' }}</p>
          <p>{{ time(invocation.started_at) }} — {{ time(invocation.completed_at) }} · Duration including preparation: {{ invocation.duration_ms == null ? 'Not recorded' : (invocation.duration_ms / 1000).toFixed(2) + ' seconds' }}</p>
          <table><thead><tr><th>Setting</th><th>Recorded value</th></tr></thead><tbody><tr v-for="(value, key) in invocation.settings ?? {}" :key="key"><td>{{ key }}</td><td>{{ value }}</td></tr></tbody></table>
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
button { margin: .3rem .5rem .3rem 0; } nav { display: flex; flex-wrap: wrap; margin: 1rem 0; }
button[aria-pressed="true"] { outline: 2px solid #3684bd; }
article { border: 1px solid #64748b; border-radius: .5rem; padding: 1rem; margin: 1rem 0; }
pre, .text { white-space: pre-wrap; overflow-wrap: anywhere; } pre { max-height: 36rem; overflow-y: auto; }
code, td, dd, li p { overflow-wrap: anywhere; } table { width: 100%; } dt { font-weight: bold; } dd { margin-bottom: .5rem; }
</style>
