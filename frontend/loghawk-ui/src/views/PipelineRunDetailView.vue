<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { RouterLink, useRoute } from "vue-router";
import { getPipelineRun } from "../services/api";
import type { PipelineRunRecord } from "../types/api";

const route = useRoute(); const run = ref<PipelineRunRecord | null>(null); const error = ref("");
const workflowId = String(route.params.workflowId); let timer: number | undefined;
async function refresh() {
  try { run.value = await getPipelineRun(workflowId); error.value = ""; if (run.value.status !== "RUNNING" && timer) window.clearInterval(timer); }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not load run details"; }
}
onMounted(() => { void refresh(); timer = window.setInterval(() => void refresh(), 3000); });
onBeforeUnmount(() => { if (timer) window.clearInterval(timer); });
</script>

<template>
  <div class="view"><header class="view-header"><div><h1>Pipeline run</h1><p>Temporal workflow execution details.</p></div><RouterLink class="btn" to="/pipelines">Back to Pipelines</RouterLink></header>
    <p v-if="error" class="error">{{ error }}</p>
    <section v-if="run" class="card"><h2>{{ run.pipeline_name }}</h2><dl>
      <dt>Status</dt><dd><span class="pill">{{ run.status }}</span></dd>
      <dt>Workflow ID</dt><dd><code>{{ run.workflow_id }}</code></dd>
      <dt>Temporal Run ID</dt><dd><code>{{ run.temporal_run_id || "Pending" }}</code></dd>
      <dt>Config Set</dt><dd>{{ run.config_set_name }}</dd>
      <dt>Phases</dt><dd>{{ run.run_mode }}</dd>
      <dt>Started</dt><dd>{{ run.started_at }}</dd>
      <dt>Completed</dt><dd>{{ run.completed_at || "—" }}</dd>
      <dt v-if="run.error">Error</dt><dd v-if="run.error" class="error">{{ run.error }}</dd>
    </dl><a v-if="run.temporal_ui_url" :href="run.temporal_ui_url" target="_blank" rel="noreferrer">Open in Temporal Web UI</a></section>
    <p v-else class="muted">Loading run…</p>
  </div>
</template>

<style scoped>
dl { display: grid; grid-template-columns: minmax(130px, .35fr) 1fr; gap: .75rem 1rem; }
dt { color: #93a7b6; } dd { margin: 0; overflow-wrap: anywhere; }
code { color: #8fe5ce; }
@media (max-width: 520px) { dl { grid-template-columns: 1fr; gap: .25rem; } dd { margin-bottom: .5rem; } }
</style>
