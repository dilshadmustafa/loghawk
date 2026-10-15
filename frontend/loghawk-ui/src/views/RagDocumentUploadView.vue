<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import RagChatPanel from "../components/RagChatPanel.vue";
import { getRagIngestion, uploadRagDocuments } from "../services/api";
import type { RagIngestionJob } from "../types/api";

const fileInput = ref<HTMLInputElement | null>(null);
const folderInput = ref<HTMLInputElement | null>(null);
const selectedFiles = ref<File[]>([]);
const job = ref<RagIngestionJob | null>(null);
const busy = ref(false);
const error = ref("");
let pollTimer: number | undefined;

onMounted(() => folderInput.value?.setAttribute("webkitdirectory", ""));
onBeforeUnmount(() => window.clearTimeout(pollTimer));

function setFiles(event: Event) {
  selectedFiles.value = Array.from((event.target as HTMLInputElement).files ?? []);
  job.value = null;
  error.value = "";
}

async function poll(jobId: string) {
  try {
    job.value = await getRagIngestion(jobId);
    if (["QUEUED", "RUNNING"].includes(job.value.status)) {
      pollTimer = window.setTimeout(() => void poll(jobId), 1000);
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not read ingestion status";
  }
}

async function upload() {
  if (!selectedFiles.value.length || busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    job.value = await uploadRagDocuments(selectedFiles.value);
    void poll(job.value.id);
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Upload failed";
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>RAG · Document Upload</h1><p>Index local files or a folder into the configured LanceDB table.</p></div></header>
    <section class="card rag-upload">
      <h2>Select documents</h2>
      <p class="muted">Supported: PDF, TXT, Markdown, JSON/JSONL, CSV, HTML, XML, YAML, DOCX, and RTF. The total upload limit is configured by LH_RAG_MAX_UPLOAD_MB.</p>
      <input ref="fileInput" class="visually-hidden" type="file" multiple @change="setFiles" />
      <input ref="folderInput" class="visually-hidden" type="file" multiple @change="setFiles" />
      <div class="actions">
        <button class="btn" type="button" @click="fileInput?.click()">Browse files</button>
        <button class="btn" type="button" @click="folderInput?.click()">Browse folder</button>
        <button class="btn primary" :disabled="busy || !selectedFiles.length" @click="upload">{{ busy ? "Uploading…" : "Upload and index" }}</button>
      </div>
      <p v-if="selectedFiles.length" class="muted">{{ selectedFiles.length }} file(s) selected · {{ (selectedFiles.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024).toFixed(1) }} MB</p>
      <p v-if="error" class="error">{{ error }}</p>
      <div v-if="job" class="rag-job" aria-live="polite">
        <strong>{{ job.status }}</strong>
        <span v-if="job.status === 'QUEUED' || job.status === 'RUNNING'"> · Indexing documents…</span>
        <span v-else-if="job.status === 'COMPLETED'"> · Indexed {{ job.chunks_indexed }} chunks from {{ job.files_processed }} file(s).</span>
        <span v-else-if="job.status === 'PARTIAL'"> · Indexed {{ job.chunks_indexed }} chunks from {{ job.files_processed }} file(s), with some skipped content.</span>
        <span v-else-if="job.error"> · {{ job.error }}</span>
        <ul v-if="job.warnings?.length" class="rag-warnings"><li v-for="warning in job.warnings" :key="warning">{{ warning }}</li></ul>
      </div>
    </section>
    <RagChatPanel />
  </div>
</template>

<style scoped>
.rag-upload { display: grid; gap: .8rem; }
.rag-upload h2, .rag-upload p { margin-top: 0; }
.visually-hidden { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
.rag-job { padding: .8rem; background: #17252e; border-radius: .5rem; overflow-wrap: anywhere; }
.rag-warnings { margin-bottom: 0; padding-left: 1.2rem; color: #e6c98c; }
</style>
