<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { getExternalBuckets, getExternalFolders, getExternalRegions } from "../services/api";
import type { ExternalS3Source } from "../types/api";

const props = defineProps<{ phase: "Train" | "Detect"; sources: ExternalS3Source[] }>();
const emit = defineEmits<{ add: [source: ExternalS3Source]; remove: [index: number] }>();
const region = ref("");
const bucket = ref("");
const folder = ref("");
const buckets = ref<string[]>([]);
const folders = ref<string[]>([]);
const regions = ref<string[]>([]);
const busy = ref(false);
const error = ref("");
const normalizedFolder = computed(() => folder.value.trim().replace(/^\/+|\/+$/g, ""));
const regionListId = computed(() => `aws-regions-${props.phase.toLowerCase()}`);

async function loadBuckets() {
  if (!region.value.trim()) { error.value = "Enter the bucket region first."; return; }
  busy.value = true; error.value = "";
  try { buckets.value = await getExternalBuckets(region.value.trim()); bucket.value = ""; folders.value = []; }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not list external buckets"; }
  finally { busy.value = false; }
}
async function browse() {
  if (!bucket.value || !region.value.trim()) { error.value = "Select a bucket and enter its region."; return; }
  busy.value = true; error.value = "";
  try { folders.value = await getExternalFolders(bucket.value, region.value.trim(), normalizedFolder.value); }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not list folders"; }
  finally { busy.value = false; }
}
function descend(prefix: string) { folder.value = prefix.replace(/^\/+|\/+$/g, ""); void browse(); }
function add() {
  if (!bucket.value || !region.value.trim()) { error.value = "Select a bucket and enter its region."; return; }
  emit("add", { bucket: bucket.value, folder: normalizedFolder.value, region: region.value.trim() });
  error.value = "";
}
onMounted(async () => {
  try { regions.value = await getExternalRegions(); }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not load AWS region suggestions"; }
});
</script>

<template>
  <div class="picker">
    <div class="form-grid">
      <label class="field">AWS region<input v-model="region" :list="regionListId" placeholder="Select or type a region, e.g. us-east-1" autocomplete="off" /><datalist :id="regionListId"><option v-for="item in regions" :key="item" :value="item" /></datalist></label>
      <label class="field">Bucket<select v-model="bucket" :disabled="busy || !buckets.length"><option value="">Select a bucket</option><option v-for="item in buckets" :key="item" :value="item">{{ item }}</option></select></label>
    </div>
    <div class="actions"><button class="btn" :disabled="busy || !region.trim()" @click="loadBuckets">{{ busy ? "Loading…" : "Load buckets" }}</button></div>
    <label class="field">Folder prefix<input v-model="folder" placeholder="Optional folder, e.g. logs/2026-10" /></label>
    <div class="actions"><button class="btn" :disabled="busy || !bucket" @click="browse">Browse folders</button><button class="btn primary" :disabled="busy || !bucket || !region.trim()" @click="add">Add to {{ phase }}</button></div>
    <div v-if="folders.length" class="folder-browser"><span class="muted">Folders under {{ normalizedFolder || "/" }}:</span><button v-for="item in folders" :key="item" class="folder-link" @click="descend(item)">{{ item }}</button></div>
    <p v-if="error" class="error">{{ error }}</p>
    <div v-if="sources.length" class="source-list"><div v-for="(source, index) in sources" :key="`${source.bucket}/${source.folder}/${source.region}`" class="source-item">
      <span>s3://{{ source.bucket }}/{{ source.folder ? `${source.folder}/` : "" }} <span class="muted">({{ source.region }})</span></span><button class="btn" @click="emit('remove', index)">Remove</button>
    </div></div>
  </div>
</template>

<style scoped>
.picker { display: grid; gap: .8rem; }
.folder-browser { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; }
.folder-link { color: #8fe5ce; background: transparent; border: 0; cursor: pointer; }
</style>
