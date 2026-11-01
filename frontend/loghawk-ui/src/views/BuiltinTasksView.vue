<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import BuiltinTaskSourceDialog from "../components/BuiltinTaskSourceDialog.vue";
import { getBuiltinTasks, getCurrentUser, setBuiltinTaskEnabled, setBuiltinTaskPolicy, setBuiltinTaskApproved } from "../services/api";
import type { BuiltinTask, TaskExecutionPolicy, TaskKind } from "../types/api";

type Row = BuiltinTask & { savedApproved: boolean; savedEnabled: boolean; savedPolicy: TaskExecutionPolicy };
const items = ref<Row[]>([]);
const route = useRoute();
const kind = ref<string>(typeof route.query.kind === "string" ? route.query.kind : "");
const kinds: { value: TaskKind; label: string }[] = [
  { value: "general", label: "General Task" }, { value: "collector", label: "Collector" },
  { value: "remediation", label: "Remediation Task" }, { value: "rollback", label: "Rollback Task" },
];
const visibleItems = computed(() => items.value.filter(item => !kind.value || item.task_kind === kind.value));
const kindLabel = (value: TaskKind) => kinds.find(k => k.value === value)?.label || value;
const sourceTask = ref<Row | null>(null);
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
    const [user, tasks] = await Promise.all([getCurrentUser(), getBuiltinTasks()]);
    role.value = user.role;
    items.value = tasks.map(item => ({ ...item, savedApproved: item.approved, savedEnabled: item.enabled, savedPolicy: item.execution_policy }));
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Could not load built-in Tasks.";
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
      const result = await setBuiltinTaskApproved(item.name, item.approved);
      item.savedApproved = result.approved;
      changed = true;
    }
    if (canSetPolicy.value && item.execution_policy !== item.savedPolicy) {
      const result = await setBuiltinTaskPolicy(item.name, item.execution_policy);
      item.savedPolicy = result.execution_policy;
      changed = true;
    }
    if (canEnable.value && item.enabled !== item.savedEnabled) {
      const result = await setBuiltinTaskEnabled(item.name, item.enabled);
      item.savedEnabled = result.enabled;
      changed = true;
    }
    message.value = `Saved settings for ${item.name}.`;
  } catch (e) {
    const detail = e instanceof Error ? e.message : "Could not save Task settings.";
    error.value = (changed ? "Some settings were saved; remaining changes are still unsaved. " : "") + detail;
  } finally { busy.value = ""; }
}
onMounted(() => void load());
</script>

<template>
  <div class="view">
    <header class="view-header">
      <div><h1>Built-in Tasks</h1><p>Manage General, Collector, Remediation, and Rollback Tasks supplied with LogHawk. User-created Tasks remain in their respective lists.</p></div>
    </header>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p v-if="message" role="status">{{ message }}</p>
    <section class="card">
      <label>Task kind <select v-model="kind"><option value="">All kinds</option><option v-for="option in kinds" :key="option.value" :value="option.value">{{ option.label }}</option></select></label>
      <p class="muted">Administrators and Power Users can change approval and execution policies. All three roles can change enabled status. Click Save to apply changes.</p>
      <p class="muted">AI collector discovery includes only Collector tasks that are approved, enabled, and AI-assisted or auto-approved. Human API execution requires approval and enablement for every Task kind.</p>
      <p v-if="loading" class="muted">Loading built-in Tasks…</p>
      <template v-else>
        <button v-if="error && !visibleItems.length" class="btn" @click="load">Retry</button>
        <p v-else-if="!visibleItems.length" class="muted">No built-in Tasks match this filter.</p>
        <div v-else class="table-wrap">
          <table>
            <thead><tr><th>Name</th><th>Task kind</th><th>Type</th><th>Description</th><th>Version</th><th>Approved</th><th>Enabled</th><th>Execution policy</th><th>Actions</th></tr></thead>
            <tbody>
              <tr v-for="item in visibleItems" :key="item.name">
                <td>{{ item.name }}</td><td>{{ kindLabel(item.task_kind) }}</td><td>{{ item.artifact_type }}</td><td>{{ item.description }}</td><td>{{ item.version }}</td>
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
                <td><div class="row-actions">
                  <button class="btn" :disabled="!!busy || !canEnable || !dirty(item)" @click="save(item)">{{ busy === item.name ? "Saving…" : "Save" }}</button>
                  <button class="btn" :aria-label="`View source for ${item.name}`" @click="sourceTask = item">View Source</button>
                </div></td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </section>
    <BuiltinTaskSourceDialog v-if="sourceTask" :name="sourceTask.name" @close="sourceTask = null" />
  </div>
</template>

<style scoped>
td { vertical-align: middle; }
.row-actions { display: flex; justify-content: center; align-items: center; gap: .5rem; flex-wrap: wrap; }
.row-actions .btn { width: 8rem; }
input[type="checkbox"] { width: auto; }
</style>
