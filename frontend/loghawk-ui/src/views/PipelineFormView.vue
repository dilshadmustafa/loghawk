<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { getConfigSets, getPipeline, savePipeline } from "../services/api";
import type { ConfigSet, RunMode } from "../types/api";

const route = useRoute(); const router = useRouter();
const id = computed(() => typeof route.params.id === "string" ? route.params.id : undefined);
const name = ref(""); const configSetId = ref(""); const mode = ref<RunMode>("train-detect"); const configSets = ref<ConfigSet[]>([]);
const loading = ref(false); const saving = ref(false); const error = ref("");
async function load() {
  loading.value = true;
  try {
    configSets.value = await getConfigSets();
    if (id.value) { const item = await getPipeline(id.value); name.value = item.name; configSetId.value = item.config_set_id; mode.value = item.run_mode; }
  } catch (err) { error.value = err instanceof Error ? err.message : "Could not load Pipeline"; }
  finally { loading.value = false; }
}
async function submit() {
  saving.value = true; error.value = "";
  try { const item = await savePipeline({ name: name.value, config_set_id: configSetId.value, run_mode: mode.value }, id.value); await router.push(`/pipelines/${item.id}/edit`); }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not save Pipeline"; }
  finally { saving.value = false; }
}
onMounted(() => void load());
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>{{ id ? "View / Edit Pipeline" : "Create Pipeline" }}</h1><p>Select the saved configuration and phases to run.</p></div></header>
    <p v-if="error" class="error">{{ error }}</p><p v-if="loading" class="muted">Loading…</p>
    <template v-else><section class="card"><div class="form-grid">
      <label class="field">Pipeline name<input v-model="name" placeholder="e.g. Payment service daily run" /></label>
      <label class="field">Config Set<select v-model="configSetId"><option value="">Select a Config Set</option><option v-for="item in configSets" :key="item.id" :value="item.id">{{ item.name }}</option></select></label>
      <label class="field">Workflow phases<select v-model="mode"><option value="train">Train</option><option value="detect">Detect</option><option value="train-detect">Train &amp; Detect</option></select></label>
    </div></section>
    <p v-if="!configSets.length" class="muted">Create a Config Set before creating a Pipeline.</p>
    <div class="actions"><button class="btn primary" :disabled="saving || !name.trim() || !configSetId" @click="submit">{{ saving ? "Saving…" : "Save Pipeline" }}</button><RouterLink class="btn" to="/pipelines">Cancel</RouterLink></div></template>
  </div>
</template>
