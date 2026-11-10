<script setup lang="ts">
withDefaults(defineProps<{
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
}>(), {
  confirmLabel: "Yes",
  cancelLabel: "No",
});
const emit = defineEmits<{confirm: []; cancel: []}>();
</script>

<template>
  <div class="confirm-backdrop" @click.self="emit('cancel')">
    <section class="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
      <h2 id="confirm-title">{{ title }}</h2>
      <p>{{ message }}</p>
      <div class="confirm-actions">
        <button type="button" @click="emit('cancel')">{{ cancelLabel }}</button>
        <button type="button" class="danger" @click="emit('confirm')">{{ confirmLabel }}</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.confirm-backdrop{position:fixed;inset:0;z-index:1000;display:grid;place-items:center;padding:1rem;background:rgba(0,0,0,.68)}
.confirm-dialog{width:min(100%,34rem);padding:1.25rem;background:#111c26;border:1px solid #078ed0;border-radius:.65rem;box-shadow:0 1rem 3rem rgba(0,0,0,.45);color:#e8edf2}
.confirm-dialog h2{margin:0 0 .75rem}.confirm-dialog p{line-height:1.5;white-space:pre-line}
.confirm-actions{display:flex;justify-content:flex-end;gap:.65rem;margin-top:1.25rem}
.confirm-actions button{padding:.55rem .85rem;border:1px solid #486071;border-radius:.4rem;color:inherit;background:#17232d;font:inherit;cursor:pointer}
.confirm-actions .danger{border-color:#b94a48;color:#fff;background:#a63835}
.confirm-actions button:focus-visible{outline:2px solid #078ed0;outline-offset:2px}
</style>
