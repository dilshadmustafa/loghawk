<script setup lang="ts">
import type { BatchInfo } from "../types/api";

defineProps<{
  buckets: string[];
  batches: BatchInfo[];
  bucket: string;
  batch: string;
  loading: boolean;
}>();

const emit = defineEmits<{
  selectBucket: [value: string];
  selectBatch: [value: string];
}>();

function changeBucket(event: Event) {
  if (event.target instanceof HTMLSelectElement) {
    emit("selectBucket", event.target.value);
  }
}

function changeBatch(event: Event) {
  if (event.target instanceof HTMLInputElement) {
    emit("selectBatch", event.target.value);
  }
}
</script>

<template>
  <div class="selectors">
    <label>
      Bucket
      <select :value="bucket" :disabled="loading" @change="changeBucket">
        <option value="" disabled>Select a bucket</option>
        <option v-for="item in buckets" :key="item" :value="item">{{ item }}</option>
      </select>
    </label>
    <label>
      Batch folder
      <input
        list="batch-options"
        :value="batch"
        :disabled="loading || !bucket"
        placeholder="Select or enter a batch folder"
        @input="changeBatch"
      />
      <datalist id="batch-options">
        <option v-for="item in batches" :key="item.name" :value="item.name" />
      </datalist>
    </label>
  </div>
</template>

<style scoped>
.selectors { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
label { display: grid; gap: .45rem; color: #aebbc7; font-size: .9rem; }
select, input { width: 100%; padding: .8rem; color: #e8edf2; background: #17232d; border: 1px solid #344553; border-radius: .5rem; }
</style>
