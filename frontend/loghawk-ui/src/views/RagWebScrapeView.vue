<script setup lang="ts">
import { onBeforeUnmount, ref } from "vue";
import RagChatPanel from "../components/RagChatPanel.vue";
import { getRagIngestion, startRagWebCrawl } from "../services/api";
import type { RagIngestionJob } from "../types/api";

const url = ref("");
const job = ref<RagIngestionJob | null>(null);
const busy = ref(false);
const error = ref("");
let pollTimer: number | undefined;

onBeforeUnmount(() => window.clearTimeout(pollTimer));

async function poll(jobId: string) {
  try {
    job.value = await getRagIngestion(jobId);
    if (["QUEUED", "RUNNING"].includes(job.value.status)) {
      pollTimer = window.setTimeout(() => void poll(jobId), 1000);
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not read crawl status";
  }
}

async function scrape() {
  if (!url.value.trim() || busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    job.value = await startRagWebCrawl(url.value.trim());
    void poll(job.value.id);
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Website crawl failed";
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <div class="view">
    <header class="view-header"><div><h1>RAG · Web Scrape</h1><p>Crawl a website and index its pages and linked supported documents.</p></div></header>
    <section class="card rag-scrape">
      <h2>Website URL</h2>
      <p class="muted">The crawler stays on the same host, follows robots.txt, and uses configured page, depth, delay, and size limits. It does not render JavaScript.</p>
      <form class="scrape-form" @submit.prevent="scrape">
        <label class="field"><span>Starting URL</span><input v-model="url" type="url" required placeholder="https://example.com/docs" /></label>
        <div class="actions"><button class="btn primary" :disabled="busy || !url.trim()">{{ busy ? "Starting…" : "Scrape and index" }}</button></div>
      </form>
      <p v-if="error" class="error">{{ error }}</p>
      <div v-if="job" class="rag-job" aria-live="polite">
        <strong>{{ job.status }}</strong>
        <span v-if="job.status === 'QUEUED' || job.status === 'RUNNING'"> · Crawling and indexing…</span>
        <span v-else-if="job.status === 'COMPLETED'"> · Indexed {{ job.chunks_indexed }} chunks from {{ job.files_processed }} page(s) or linked document(s).</span>
        <span v-else-if="job.status === 'PARTIAL'"> · Indexed {{ job.chunks_indexed }} chunks from {{ job.files_processed }} page(s) or linked document(s), with some skipped content.</span>
        <span v-else-if="job.error"> · {{ job.error }}</span>
        <ul v-if="job.warnings?.length" class="rag-warnings"><li v-for="warning in job.warnings" :key="warning">{{ warning }}</li></ul>
      </div>
    </section>
    <RagChatPanel />
  </div>
</template>

<style scoped>
.rag-scrape { display: grid; gap: .8rem; }
.rag-scrape h2, .rag-scrape p { margin-top: 0; }
.scrape-form { display: grid; gap: .8rem; }
.rag-job { padding: .8rem; background: #17252e; border-radius: .5rem; overflow-wrap: anywhere; }
.rag-warnings { margin-bottom: 0; padding-left: 1.2rem; color: #e6c98c; }
</style>
