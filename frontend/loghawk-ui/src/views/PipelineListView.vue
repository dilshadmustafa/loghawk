<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import { getPipelines, runPipeline } from "../services/api";
import type { Pipeline } from "../types/api";

const items = ref<Pipeline[]>([]); const error = ref(""); const busyId = ref("");
let timer: number | undefined;
async function refresh() {
  try { items.value = await getPipelines(); error.value = ""; }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not load Pipelines"; }
}
async function launch(item: Pipeline) {
  busyId.value = item.id; error.value = "";
  try { await runPipeline(item.id); await refresh(); }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not start Pipeline"; await refresh(); }
  finally { busyId.value = ""; }
}
onMounted(() => { void refresh(); timer = window.setInterval(() => void refresh(), 4000); });
onBeforeUnmount(() => { if (timer) window.clearInterval(timer); });
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>Pipelines</h1><p>Saved Temporal workflows and their latest run.</p></div><RouterLink class="btn primary" to="/pipelines/new">Create Pipeline</RouterLink></header>
    <p v-if="error" class="error">{{ error }}</p>
    <section class="card"><p v-if="!items.length" class="muted">No Pipelines have been created.</p><div v-else class="table-wrap"><table><thead><tr><th>Pipeline</th><th>Config Set</th><th>Workflow</th><th>Run status</th><th>Run ID</th><th>Actions</th></tr></thead>
      <tbody><tr v-for="item in items" :key="item.id"><td>{{ item.name }}</td><td>{{ item.config_set_name }}</td><td>{{ item.run_mode }}</td>
        <td><span class="pill">{{ item.latest_run?.status ?? "NOT RUN" }}</span></td>
        <td><RouterLink v-if="item.latest_run" :to="`/pipeline-runs/${encodeURIComponent(item.latest_run.workflow_id)}`">{{ item.latest_run.workflow_id }}</RouterLink><span v-else class="muted">—</span></td>
        <td><div class="actions"><RouterLink class="btn" :to="`/pipelines/${item.id}/edit`">View / Edit</RouterLink><button class="btn primary" :disabled="busyId === item.id || item.latest_run?.status === 'RUNNING'" @click="launch(item)">{{ busyId === item.id ? "Starting…" : "Run" }}</button></div></td>
      </tr></tbody>
    </table></div></section>
  </div>
</template>
