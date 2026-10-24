<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import { deleteTask, getTasks } from "../services/api";
import type { TaskRecord } from "../types/api";

const items = ref<TaskRecord[]>([]);
const error = ref("");
const loading = ref(true);
const deletingId = ref("");
const query = ref("");

async function refresh() {
  loading.value = true;
  try {
    items.value = await getTasks(query.value.trim());
    error.value = "";
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not load Tasks";
  } finally {
    loading.value = false;
  }
}

function typeLabel(item: TaskRecord) {
  if (item.artifact_type === "windows-cmd") return "Windows CMD";
  if (item.artifact_type === "bash" || item.artifact_type === "powershell") {
    return `${item.artifact_type} script`;
  }
  return item.artifact_type;
}

async function remove(item: TaskRecord) {
  if (!window.confirm(`Delete Task “${item.name}” and its saved conversation and artifacts?`)) return;
  deletingId.value = item.id;
  error.value = "";
  try { await deleteTask(item.id); await refresh(); }
  catch (err) { error.value = err instanceof Error ? err.message : "Could not delete Task"; }
  finally { deletingId.value = ""; }
}

watch(query, () => void refresh());
onMounted(() => void refresh());
</script>

<template>
  <div class="view">
    <header class="view-header">
      <div><h1>Tasks</h1><p>Draft Python, Bash, PowerShell, Windows CMD, and n8n artifacts with an LLM.</p></div>
      <RouterLink class="btn primary" to="/tasks/new">Create Task</RouterLink>
    </header>
    <p v-if="error" class="error">{{ error }}</p>
    <section class="card">
      <label class="field task-search">Search by name, description, category, or task kind<input v-model="query" placeholder="e.g. collect Prometheus metrics" /></label>
      <p v-if="loading" class="muted">Loading Tasks…</p>
      <p v-else-if="!items.length" class="muted">{{ query.trim() ? "No Tasks match this search." : "No Tasks have been created." }}</p>
      <div v-else class="table-wrap">
        <table>
          <thead><tr><th>Name</th><th>User ID</th><th>Category</th><th>Task kind</th><th>Execution policy</th><th>Type</th><th>Latest artifact</th><th>Updated</th><th>Actions</th></tr></thead>
          <tbody>
            <tr v-for="item in items" :key="item.id">
              <td>{{ item.name }}</td><td>{{ item.userid || "—" }}</td><td>{{ item.category }}</td><td>{{ item.task_kind }}</td><td>{{ item.execution_policy || "manual" }}</td><td><span class="pill">{{ typeLabel(item) }}</span></td>
              <td>{{ item.latest_artifact ? `Version ${item.latest_artifact.version} · ${item.latest_artifact.approved ? "Approved" : "Pending"} · ${item.latest_artifact.enabled ? "Enabled" : "Disabled"}` : "Not generated" }}</td>
              <td>{{ new Date(item.updated_at).toLocaleString() }}</td>
              <td><div class="actions"><RouterLink :to="`/tasks/${item.id}`">View / Continue</RouterLink><button class="btn" :disabled="deletingId === item.id" @click="remove(item)">{{ deletingId === item.id ? "Deleting…" : "Delete" }}</button></div></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.task-search { max-width: 36rem; margin-bottom: 1rem; }
</style>
