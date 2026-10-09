<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import ExternalSourcePicker from "../components/ExternalSourcePicker.vue";
import { getBatches, getBuckets, getConfigSet, getConfigSetDefaults, saveConfigSet } from "../services/api";
import type { BatchInfo, ConfigSetDefaults, ExternalS3Source } from "../types/api";

const route = useRoute(); const router = useRouter();
const id = computed(() => typeof route.params.id === "string" ? route.params.id : undefined);
const name = ref(""); const external = ref(false); const sourceBucket = ref(""); const sourceBatch = ref("");
const outputBucket = ref(""); const outputBatch = ref(""); const trainSources = ref<ExternalS3Source[]>([]); const detectSources = ref<ExternalS3Source[]>([]);
const buckets = ref<string[]>([]); const batches = ref<BatchInfo[]>([]); const defaults = ref<ConfigSetDefaults | null>(null);
const loading = ref(false); const saving = ref(false); const error = ref("");
const availableBatches = computed(() => batches.value.map((item) => item.name));
const processingPath = computed(() => external.value ? `s3://${outputBucket.value || "<bucket>"}/${outputBatch.value || "<batch>"}/` : `s3://${sourceBucket.value || "<bucket>"}/${sourceBatch.value || "<batch>"}/`);

async function loadInputs(bucket: string) {
  sourceBucket.value = bucket; sourceBatch.value = ""; batches.value = [];
  if (!bucket) return;
  try { batches.value = await getBatches(bucket); }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not load RustFS folders"; }
}
function onSourceBucketChange(event: Event) {
  if (event.target instanceof HTMLSelectElement) void loadInputs(event.target.value);
}
function addSource(list: ExternalS3Source[], source: ExternalS3Source) {
  if (!list.some((item) => item.bucket === source.bucket && item.folder === source.folder && item.region === source.region)) list.push(source);
}
async function load() {
  loading.value = true;
  try {
    const [bucketList, defaultsValue] = await Promise.all([getBuckets(), getConfigSetDefaults()]);
    buckets.value = bucketList; defaults.value = defaultsValue;
    if (id.value) {
      const item = await getConfigSet(id.value);
      name.value = item.name; external.value = item.external_data_use; sourceBucket.value = item.source_bucket ?? ""; sourceBatch.value = item.source_batch ?? "";
      outputBucket.value = item.output_bucket; outputBatch.value = item.output_batch; trainSources.value = item.train_sources; detectSources.value = item.detect_sources;
      if (sourceBucket.value) { batches.value = await getBatches(sourceBucket.value); }
    } else {
      external.value = defaultsValue.external_data_use; outputBucket.value = defaultsValue.output_bucket; outputBatch.value = defaultsValue.output_batch;
      if (bucketList.length) { await loadInputs(bucketList.includes(defaultsValue.output_bucket) ? defaultsValue.output_bucket : bucketList[0]); }
      sourceBatch.value = defaultsValue.output_batch;
    }
  } catch (err) { error.value = err instanceof Error ? err.message : "Could not initialize Config Set form"; }
  finally { loading.value = false; }
}
async function submit() {
  saving.value = true; error.value = "";
  try {
    const result = await saveConfigSet({ name: name.value, external_data_use: external.value,
      source_bucket: external.value ? null : sourceBucket.value, source_batch: external.value ? null : sourceBatch.value,
      train_sources: external.value ? trainSources.value : [], detect_sources: external.value ? detectSources.value : [],
      output_bucket: external.value ? outputBucket.value : sourceBucket.value,
      output_batch: external.value ? outputBatch.value : sourceBatch.value }, id.value);
    await router.push(`/config-sets/${result.id}/edit`);
  } catch (err) { error.value = err instanceof Error ? err.message : "Could not save Config Set"; }
  finally { saving.value = false; }
}
onMounted(() => void load());
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>{{ id ? "View / Edit Config Set" : "Create Config Set" }}</h1><p>Choose input sources and the RustFS location for processing outputs.</p></div></header>
    <p v-if="error" class="error">{{ error }}</p><p v-if="loading" class="muted">Loading storage settings…</p>
    <template v-else>
      <section class="card"><div class="form-grid">
        <label class="field">Config Set name<input v-model="name" placeholder="e.g. Production logs" /></label>
        <label class="check-field"><input v-model="external" type="checkbox" /> Use External S3 Data</label>
      </div></section>
      <template v-if="external">
        <section class="card"><h2>Train sources</h2><ExternalSourcePicker phase="Train" :sources="trainSources" @add="addSource(trainSources, $event)" @remove="trainSources.splice($event, 1)" /></section>
        <section class="card"><h2>Detect sources</h2><ExternalSourcePicker phase="Detect" :sources="detectSources" @add="addSource(detectSources, $event)" @remove="detectSources.splice($event, 1)" /></section>
        <section class="card"><h2>Processing output · RustFS</h2><div class="form-grid">
          <label class="field">Output bucket<select v-model="outputBucket"><option value="">Select output bucket</option><option v-for="item in buckets" :key="item" :value="item">{{ item }}</option></select></label>
          <label class="field">Output batch folder<input v-model="outputBatch" placeholder="Batch folder" /></label>
        </div><p class="muted">Artifacts, features, models, anomalies, and incidents will be written under {{ processingPath }}</p></section>
      </template>
      <template v-else>
        <section class="card"><h2>Train and Detect inputs · Internal S3</h2><p class="muted">The selected batch uses its <code>train/</code> and <code>raw/</code> prefixes as input.</p><div class="form-grid">
          <label class="field">Input bucket<select :value="sourceBucket" @change="onSourceBucketChange"><option value="">Select bucket</option><option v-for="item in buckets" :key="item" :value="item">{{ item }}</option></select></label>
          <label class="field">Batch folder<input v-model="sourceBatch" list="internal-batches" placeholder="Select or enter a batch folder" /><datalist id="internal-batches"><option v-for="item in availableBatches" :key="item" :value="item" /></datalist></label>
        </div><p v-if="sourceBatch" class="muted">{{ batches.find((item) => item.name === sourceBatch)?.train_available ? "Train input available" : "Train input not found" }} · {{ batches.find((item) => item.name === sourceBatch)?.raw_available ? "Detect input available" : "Detect input not found" }}</p></section>
        <section class="card"><h2>Processing output</h2><p class="muted">Output is derived from the selected input bucket and batch.</p><code>{{ processingPath }}</code></section>
      </template>
      <p v-if="error" class="error">{{ error }}</p>
      <div class="actions"><button class="btn primary" :disabled="saving || !name.trim()" @click="submit">{{ saving ? "Saving…" : "Save Config Set" }}</button><RouterLink class="btn" to="/config-sets">Cancel</RouterLink></div>
    </template>
  </div>
</template>

<style scoped>
.check-field { display: flex; align-items: center; gap: .65rem; color: #e8edf2; }
.check-field input { width: 1.1rem; height: 1.1rem; accent-color: #6bd7bb; }
code { color: #8fe5ce; overflow-wrap: anywhere; }
</style>
