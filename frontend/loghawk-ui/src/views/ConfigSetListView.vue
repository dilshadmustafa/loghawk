<script setup lang="ts">
import { onMounted, ref } from "vue";
import { RouterLink } from "vue-router";
import { getConfigSets } from "../services/api";
import type { ConfigSet } from "../types/api";

const items = ref<ConfigSet[]>([]);
const error = ref("");
const loading = ref(true);
async function refresh() {
  loading.value = true;
  try { items.value = await getConfigSets(); error.value = ""; }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not load Config Sets"; }
  finally { loading.value = false; }
}
onMounted(() => void refresh());
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>Config Sets</h1><p>Save reusable input and output storage selections.</p></div><RouterLink class="btn primary" to="/config-sets/new">Create Config Set</RouterLink></header>
    <section class="card">
      <p v-if="error" class="error">{{ error }}</p><p v-else-if="loading" class="muted">Loading Config Sets…</p>
      <p v-else-if="!items.length" class="muted">No Config Sets yet.</p>
      <div v-else class="table-wrap"><table><thead><tr><th>Name</th><th>Mode</th><th>Inputs</th><th>Processing output</th><th></th></tr></thead>
        <tbody><tr v-for="item in items" :key="item.id"><td>{{ item.name }}</td><td><span class="pill">{{ item.external_data_use ? "External S3" : "Internal S3" }}</span></td>
          <td>{{ item.external_data_use ? `${item.train_sources.length} Train · ${item.detect_sources.length} Detect` : `${item.source_bucket}/${item.source_batch}` }}</td>
          <td>s3://{{ item.output_bucket }}/{{ item.output_batch }}/</td><td><RouterLink :to="`/config-sets/${item.id}/edit`">View / Edit</RouterLink></td></tr></tbody>
      </table></div>
    </section>
  </div>
</template>
