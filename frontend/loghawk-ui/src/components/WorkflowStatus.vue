<script setup lang="ts">
import type { PipelineRunStatus } from "../types/api";

defineProps<{ run: PipelineRunStatus | null; mode: string; error: string }>();
</script>

<template>
  <section class="status" aria-live="polite">
    <h2>Pipeline status</h2>
    <p v-if="error" class="error">{{ error }}</p>
    <template v-else-if="run">
      <p><strong>{{ run.status }}</strong> · {{ mode }}</p>
      <code>{{ run.workflow_id }}</code>
    </template>
    <p v-else class="muted">No pipeline run started in this session.</p>
  </section>
</template>

<style scoped>
.status { padding: 1rem; background: #17232d; border: 1px solid #2d404e; border-radius: .65rem; }
h2 { margin: 0 0 .75rem; font-size: 1rem; }
.muted { color: #aebbc7; }
.error { color: #ff9f9f; }
code { overflow-wrap: anywhere; color: #8fe5ce; }
</style>
