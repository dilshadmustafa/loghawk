<script setup lang="ts">
import type { RunMode } from "../types/api";

defineProps<{ disabled: boolean; busy: boolean }>();
const emit = defineEmits<{ start: [mode: RunMode] }>();
</script>

<template>
  <div class="actions">
    <button :disabled="disabled || busy" @click="emit('start', 'train')">Train</button>
    <button :disabled="disabled || busy" @click="emit('start', 'detect')">Detect</button>
    <button class="primary" :disabled="disabled || busy" @click="emit('start', 'train-detect')">
      Train &amp; Detect
    </button>
  </div>
</template>

<style scoped>
.actions { display: flex; flex-wrap: wrap; gap: .75rem; }
button { padding: .75rem 1rem; color: #e8edf2; background: #243442; border: 1px solid #405665; border-radius: .5rem; cursor: pointer; }
button.primary { color: #08131b; background: #6bd7bb; border-color: #6bd7bb; font-weight: 700; }
button:disabled { cursor: not-allowed; opacity: .45; }
</style>
