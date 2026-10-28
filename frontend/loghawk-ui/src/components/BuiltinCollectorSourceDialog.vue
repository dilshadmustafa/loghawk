<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { getBuiltinCollectorSource } from "../services/api";
import type { BuiltinCollectorSource } from "../types/api";

const props = defineProps<{ name: string }>();
const emit = defineEmits<{ close: [] }>();
type Tab = "source" | "schema";
const tabs: Tab[] = ["source", "schema"];
const active = ref<Tab>("source");
const dialog = ref<HTMLDialogElement | null>(null);
const data = ref<BuiltinCollectorSource | null>(null);
const loading = ref(true);
const error = ref("");
const copyMessage = ref("");
const copying = ref(false);
let alive = true;
let opener: HTMLElement | null = null;
const content = computed(() => !data.value ? "" : active.value === "source"
  ? data.value.source : JSON.stringify(data.value.arguments_schema, null, 2));
watch(active, () => { copyMessage.value = ""; });

async function load() {
  loading.value = true;
  error.value = "";
  try {
    const result = await getBuiltinCollectorSource(props.name);
    if (alive) data.value = result;
  } catch (e) {
    if (alive) error.value = e instanceof Error ? e.message : "Could not load collector source.";
  } finally { if (alive) loading.value = false; }
}
function close() { dialog.value?.close(); }
async function moveTab(event: KeyboardEvent) {
  const index = tabs.indexOf(active.value);
  let next: Tab | undefined;
  if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
  if (event.key === "ArrowLeft") next = tabs[(index + tabs.length - 1) % tabs.length];
  if (event.key === "Home") next = "source";
  if (event.key === "End") next = "schema";
  if (!next) return;
  event.preventDefault();
  active.value = next;
  await nextTick();
  dialog.value?.querySelector<HTMLButtonElement>(`#collector-tab-${next}`)?.focus();
}
async function copy() {
  if (copying.value || !data.value) return;
  const tab = active.value;
  const text = content.value;
  copying.value = true;
  copyMessage.value = "";
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
    await navigator.clipboard.writeText(text);
    if (alive && active.value === tab) copyMessage.value = "Copied to clipboard.";
  } catch {
    if (alive && active.value === tab) copyMessage.value = "Could not access the clipboard. Select the text below and copy it manually.";
  } finally { if (alive) copying.value = false; }
}
onMounted(() => {
  opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  dialog.value?.showModal();
  void load();
});
onBeforeUnmount(() => {
  alive = false;
  dialog.value?.close();
  opener?.focus();
});
</script>

<template>
  <dialog ref="dialog" aria-labelledby="collector-source-title" @close="emit('close')" @cancel.prevent="close">
    <header>
      <div>
        <h2 id="collector-source-title">{{ data?.display_name || name }}</h2>
        <p v-if="data">Installed version {{ data.version }} · {{ data.filename }}</p>
      </div>
      <button type="button" class="btn" autofocus @click="close">Close</button>
    </header>
    <p class="muted">Read-only installed connector implementation and input definition. The source imports LogHawk helpers; it is not a standalone script.</p>
    <p v-if="loading" role="status">Loading collector source…</p>
    <div v-else-if="error">
      <p class="error" role="alert">{{ error }}</p>
      <button type="button" class="btn" @click="load">Retry</button>
    </div>
    <template v-else-if="data">
      <div role="tablist" aria-label="Collector details" class="tabs" @keydown="moveTab">
        <button v-for="tab in tabs" :id="`collector-tab-${tab}`" :key="tab" type="button"
          role="tab" :aria-selected="active === tab" :aria-controls="`collector-panel-${tab}`"
          :tabindex="active === tab ? 0 : -1" :class="['btn', { selected: active === tab }]"
          @click="active = tab">{{ tab === 'source' ? 'Source Code' : 'Input Schema' }}</button>
      </div>
      <section :id="`collector-panel-${active}`" role="tabpanel" :aria-labelledby="`collector-tab-${active}`">
        <div class="copy-row">
          <span v-if="active === 'schema'" class="muted">Built-in argument definition including defaults and limits.</span>
          <button type="button" class="btn" :disabled="copying" @click="copy">{{ copying ? 'Copying…' : 'Copy to Clipboard' }}</button>
        </div>
        <p role="status" aria-live="polite">{{ copyMessage }}</p>
        <pre tabindex="0" :aria-label="active === 'source' ? 'Collector source code' : 'Collector input schema'"><code>{{ content }}</code></pre>
      </section>
    </template>
  </dialog>
</template>

<style scoped>
dialog { width: min(64rem, calc(100vw - 2rem)); max-height: 90vh; overflow: auto; padding: 1.25rem; border: 1px solid #536578; border-radius: .6rem; background: #111c27; color: #edf3f8; }
dialog::backdrop { background: rgb(0 0 0 / 65%); }
header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
h2 { margin: 0; }
.tabs { display: flex; gap: .5rem; margin: 1rem 0; }
.selected { background: #17689b; color: white; border-color: #8ccdf5; }
.copy-row { display: flex; justify-content: flex-end; align-items: center; gap: 1rem; flex-wrap: wrap; }
pre { max-height: 52vh; overflow: auto; white-space: pre; padding: 1rem; background: #09121b; color: #edf3f8; border: 1px solid #536578; border-radius: .3rem; font-size: .9rem; user-select: text; }
button:focus-visible, pre:focus-visible { outline: 2px solid #8ccdf5; outline-offset: 3px; }
</style>
