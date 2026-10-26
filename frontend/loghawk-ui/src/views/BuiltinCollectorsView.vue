<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { getBuiltinCollectors, getCurrentUser, setBuiltinCollectorEnabled, setBuiltinCollectorPolicy, setBuiltinCollectorApproved } from "../services/api";
import type { BuiltinCollector, TaskExecutionPolicy } from "../types/api";

type Row = BuiltinCollector & { savedApproved: boolean; savedEnabled: boolean; savedPolicy: TaskExecutionPolicy };
const items = ref<Row[]>([]);
const loading = ref(true);
const busy = ref("");
const error = ref("");
const message = ref("");
const role = ref("");
const canSetPolicy = computed(() => ["Administrator", "Power User"].includes(role.value));
const canEnable = computed(() => ["Administrator", "Power User", "User"].includes(role.value));
const policies: TaskExecutionPolicy[] = ["manual", "ai-assisted", "auto-approved", "disabled"];
const dirty = (item: Row) => item.approved !== item.savedApproved || item.enabled !== item.savedEnabled || item.execution_policy !== item.savedPolicy;

async function load() {
  loading.value = true;
  error.value = "";
  try {
    const [user, collectors] = await Promise.all([getCurrentUser(), getBuiltinCollectors()]);
    role.value = user.role;
    items.value = collectors.map(item => ({ ...item, savedApproved: item.approved, savedEnabled: item.enabled, savedPolicy: item.execution_policy }));
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Could not load built-in collectors.";
  } finally { loading.value = false; }
}

async function save(item: Row) {
  if (busy.value || !dirty(item)) return;
  busy.value = item.name;
  error.value = "";
  message.value = "";
  let changed = false;
  try {
    if (canSetPolicy.value && item.approved !== item.savedApproved) {
      const result = await setBuiltinCollectorApproved(item.name, item.approved);
      item.savedApproved = result.approved;
      changed = true;
    }
    if (canSetPolicy.value && item.execution_policy !== item.savedPolicy) {
      const result = await setBuiltinCollectorPolicy(item.name, item.execution_policy);
      item.savedPolicy = result.execution_policy;
      changed = true;
    }
    if (canEnable.value && item.enabled !== item.savedEnabled) {
      const result = await setBuiltinCollectorEnabled(item.name, item.enabled);
      item.savedEnabled = result.enabled;
      changed = true;
    }
    message.value = `Saved settings for ${item.name}.`;
  } catch (e) {
    const detail = e instanceof Error ? e.message : "Could not save collector settings.";
    error.value = (changed ? "Some settings were saved; remaining changes are still unsaved. " : "") + detail;
  } finally { busy.value = ""; }
}
onMounted(() => void load());
</script>

<template>
  <div class="view">
    <header class="view-header">
      <div><h1>Built-in Collectors</h1><p>Manage collectors supplied with LogHawk. Custom collectors are listed under List Tasks.</p></div>
    </header>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p v-if="message" role="status">{{ message }}</p>
    <section class="card">
      <p class="muted">Administrators and Power Users can change approval and execution policies. All three roles can change enabled status. Click Save to apply changes.</p>
      <p class="muted">AI discovery requires approval, enabled status, and an AI-assisted or auto-approved policy. Manual collectors are available for human-initiated execution.</p>
      <p v-if="loading" class="muted">Loading built-in collectors…</p>
      <template v-else>
        <button v-if="error && !items.length" class="btn" @click="load">Retry</button>
        <p v-else-if="!items.length" class="muted">No built-in collectors are registered.</p>
        <div v-else class="table-wrap">
          <table>
            <thead><tr><th>Name</th><th>Description</th><th>Version</th><th>Approved</th><th>Enabled</th><th>Execution policy</th><th>Actions</th></tr></thead>
            <tbody>
              <tr v-for="item in items" :key="item.name">
                <td>{{ item.name }}</td><td>{{ item.description }}</td><td>{{ item.version }}</td>
                <td>
                  <input v-if="canSetPolicy" v-model="item.approved" type="checkbox" :disabled="!!busy" :aria-label="`Approve ${item.name}`" />
                  <span v-else>{{ item.approved ? "Yes" : "No" }}</span>
                </td>
                <td><input v-model="item.enabled" type="checkbox" :aria-label="`Enable ${item.name}`" :disabled="!canEnable || !!busy" /></td>
                <td>
                  <select v-if="canSetPolicy" v-model="item.execution_policy" :aria-label="`Execution policy for ${item.name}`" :disabled="!!busy">
                    <option v-for="policy in policies" :key="policy" :value="policy">{{ policy }}</option>
                  </select>
                  <span v-else>{{ item.execution_policy }}</span>
                </td>
                <td><button class="btn" :disabled="!!busy || !canEnable || !dirty(item)" @click="save(item)">{{ busy === item.name ? "Saving…" : "Save" }}</button></td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </section>
  </div>
</template>

<style scoped>
td { vertical-align: middle; }
input[type="checkbox"] { width: auto; }
</style>
