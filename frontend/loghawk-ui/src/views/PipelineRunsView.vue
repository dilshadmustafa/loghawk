<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import PipelineActions from "../components/PipelineActions.vue";
import StorageSelector from "../components/StorageSelector.vue";
import WorkflowStatus from "../components/WorkflowStatus.vue";
import { getBatches, getBuckets, getRunStatus, startRun } from "../services/api";
import type { BatchInfo, PipelineRunStatus, RunMode } from "../types/api";

const buckets = ref<string[]>([]);
const batches = ref<BatchInfo[]>([]);
const bucket = ref("");
const batch = ref("");
const currentRun = ref<PipelineRunStatus | null>(null);
const currentMode = ref("");
const error = ref("");
const loading = ref(false);
const busy = ref(false);
let pollTimer: number | undefined;

const selectedBatch = computed(() => batches.value.find((item) => item.name === batch.value));
const canRun = computed(() => Boolean(bucket.value && batch.value));

async function loadBuckets() {
  loading.value = true;
  error.value = "";
  try {
    buckets.value = await getBuckets();
    if (buckets.value.length === 1) await selectBucket(buckets.value[0]);
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not load buckets";
  } finally {
    loading.value = false;
  }
}

async function selectBucket(value: string) {
  bucket.value = value;
  batch.value = "";
  batches.value = [];
  if (!value) return;
  loading.value = true;
  error.value = "";
  try {
    batches.value = await getBatches(value);
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not load batches";
  } finally {
    loading.value = false;
  }
}

function selectBatch(value: string) {
  batch.value = value;
}

async function launch(mode: RunMode) {
  if (!canRun.value) return;

  busy.value = true;
  error.value = "";
  try {
    const started = await startRun(bucket.value, batch.value, mode);
    currentMode.value = mode;
    currentRun.value = { workflow_id: started.workflow_id, run_id: null, status: started.status };
    beginPolling(started.workflow_id);
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not start pipeline";
  } finally {
    busy.value = false;
  }
}

function beginPolling(workflowId: string) {
  if (pollTimer) window.clearInterval(pollTimer);
  const poll = async () => {
    try {
      currentRun.value = await getRunStatus(workflowId);
      error.value = "";
      if (["COMPLETED", "FAILED", "CANCELED", "TERMINATED", "TIMED_OUT"].includes(currentRun.value.status)) {
        window.clearInterval(pollTimer);
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : "Could not refresh workflow status";
    }
  };
  void poll();
  pollTimer = window.setInterval(() => void poll(), 3000);
}

onMounted(() => void loadBuckets());
onBeforeUnmount(() => {
  if (pollTimer) window.clearInterval(pollTimer);
});
</script>

<template>
  <main class="page">
    <header>
      <div class="mark">LH</div>
      <div><p class="eyebrow">AIOps control plane</p><h1>LogHawk</h1></div>
    </header>

    <section class="panel">
      <h2>Start a pipeline run</h2>
      <StorageSelector
        :buckets="buckets"
        :batches="batches"
        :bucket="bucket"
        :batch="batch"
        :loading="loading"
        @select-bucket="selectBucket"
        @select-batch="selectBatch"
      />
      <p v-if="selectedBatch" class="availability">
        Train input: {{ selectedBatch.train_available ? "available" : "not found" }}
        <span>·</span>
        Raw input: {{ selectedBatch.raw_available ? "available" : "not found" }}
      </p>
      <PipelineActions :disabled="!canRun" :busy="busy" @start="launch" />
    </section>

    <WorkflowStatus :run="currentRun" :mode="currentMode" :error="error" />
    <footer>LogHawk starts and monitors Temporal workflows. Processing runs in the worker.</footer>
  </main>
</template>

<style scoped>
.page { width: min(900px, calc(100% - 2rem)); margin: 3rem auto; display: grid; gap: 1.25rem; }
header { display: flex; align-items: center; gap: 1rem; }
.mark { display: grid; place-items: center; width: 3rem; height: 3rem; border-radius: .8rem; background: #6bd7bb; color: #101820; font-weight: 800; }
.eyebrow { margin: 0; color: #8fa4b5; text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; }
h1 { margin: .15rem 0; font-size: 1.8rem; }
.panel { display: grid; gap: 1.2rem; padding: 1.5rem; background: #131f28; border: 1px solid #2d404e; border-radius: .8rem; }
.panel h2 { margin: 0; font-size: 1.15rem; }
.availability, footer { color: #aebbc7; font-size: .88rem; }
.availability span { padding: 0 .35rem; }
footer { padding-bottom: 2rem; }
</style>
