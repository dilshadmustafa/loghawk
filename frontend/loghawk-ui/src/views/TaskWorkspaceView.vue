<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import {
  createTask,
  approveTaskArtifact,
  deleteTaskArtifact,
  getTask,
  importTaskToN8n,
  saveTaskArtifact,
  sendTaskMessage,
  setTaskArtifactEnabled,
  getCurrentUser,
  setTaskExecutionPolicy,
  setTaskDescription,
} from "../services/api";
import type { TaskArtifactType, TaskExecutionPolicy, TaskKind, TaskRecord } from "../types/api";

const route = useRoute();
const router = useRouter();
const taskId = computed(() => typeof route.params.id === "string" ? route.params.id : "");
const task = ref<TaskRecord | null>(null);
const name = ref("");
const description = ref("");
const descriptionDraft = ref("");
const category = ref("");
const artifactType = ref<TaskArtifactType>("python");
const taskKind = ref<TaskKind>("general");
const executionPolicy = ref<TaskExecutionPolicy>("manual");
const currentUser = ref<{ userid: string; role: string } | null>(null);
const message = ref("");
const editorContent = ref("");
const savedBaseline = ref("");
const selectedVersionId = ref("");
const hasGeneratedDraft = ref(false);
const loading = ref(false);
const saving = ref(false);
const savingArtifact = ref(false);
const savingExecutionPolicy = ref(false);
const deletingArtifact = ref(false);
const sending = ref(false);
const importingToN8n = ref(false);
const importedWorkflow = ref<{ name: string; url: string } | null>(null);
const error = ref("");
const hasUnsavedChanges = computed(() =>
  hasGeneratedDraft.value || editorContent.value !== savedBaseline.value,
);
const hasArtifactToEdit = computed(() =>
  hasGeneratedDraft.value || Boolean(selectedVersionId.value),
);
const selectedVersion = computed(() =>
  task.value?.artifacts?.find((artifact) => artifact.id === selectedVersionId.value) ?? null,
);
const canManageTask = computed(() => Boolean(task.value && currentUser.value && (
  ["Administrator", "Power User"].includes(currentUser.value.role)
  || task.value.userid === currentUser.value.userid
)));
const canSetAutoApproved = computed(() =>
  ["Administrator", "Power User"].includes(currentUser.value?.role || ""),
);

watch(taskKind, (kind) => {
  if (kind === "collector" && artifactType.value === "n8n") artifactType.value = "python";
});

async function importSavedVersionToN8n() {
  if (
    !task.value
    || task.value.artifact_type !== "n8n"
    || !selectedVersionId.value
    || hasGeneratedDraft.value
    || hasUnsavedChanges.value
  ) return;

  importingToN8n.value = true;
  importedWorkflow.value = null;
  error.value = "";
  try {
    const result = await importTaskToN8n(task.value.id, selectedVersionId.value);
    importedWorkflow.value = { name: result.name, url: result.url };
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not import workflow into n8n";
  } finally {
    importingToN8n.value = false;
  }
}

async function loadTask() {
  try { currentUser.value = await getCurrentUser(); } catch { currentUser.value = null; }
  if (!taskId.value) { task.value = null; return; }
  loading.value = true;
  try {
    task.value = await getTask(taskId.value);
    executionPolicy.value = task.value.execution_policy || "manual";
    descriptionDraft.value = task.value.description || "";
    savedBaseline.value = task.value.latest_artifact?.content ?? "";
    editorContent.value = savedBaseline.value;
    selectedVersionId.value = task.value.latest_artifact?.id ?? "";
    hasGeneratedDraft.value = false;
    error.value = "";
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not load Task";
  } finally {
    loading.value = false;
  }
}

async function create() {
  saving.value = true;
  error.value = "";
  try {
    const created = await createTask({
      name: name.value,
      description: description.value,
      category: category.value,
      task_kind: taskKind.value,
      execution_policy: executionPolicy.value,
      artifact_type: artifactType.value,
    });
    await router.replace(`/tasks/${created.id}`);
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not create Task";
  } finally {
    saving.value = false;
  }
}

async function saveDescription() {
  if (!task.value || !descriptionDraft.value.trim()) return;
  saving.value = true;
  error.value = "";
  try {
    task.value = await setTaskDescription(task.value.id, descriptionDraft.value);
    descriptionDraft.value = task.value.description;
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not update Task description";
  } finally {
    saving.value = false;
  }
}

async function saveExecutionPolicy() {
  if (!task.value || executionPolicy.value === task.value.execution_policy) return;
  savingExecutionPolicy.value = true;
  error.value = "";
  try {
    task.value = await setTaskExecutionPolicy(task.value.id, {
      execution_policy: executionPolicy.value,
    });
    executionPolicy.value = task.value.execution_policy;
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not update execution policy";
  } finally {
    savingExecutionPolicy.value = false;
  }
}

async function approveSelectedVersion() {
  if (!task.value || !selectedVersion.value || hasUnsavedChanges.value) return;
  try {
    await approveTaskArtifact(task.value.id, selectedVersion.value.id);
    await loadTask();
  } catch (err) { error.value = err instanceof Error ? err.message : "Could not approve version"; }
}

async function toggleSelectedVersion() {
  if (!task.value || !selectedVersion.value || hasUnsavedChanges.value) return;
  try {
    await setTaskArtifactEnabled(task.value.id, selectedVersion.value.id, !selectedVersion.value.enabled);
    await loadTask();
  } catch (err) { error.value = err instanceof Error ? err.message : "Could not update version status"; }
}

async function send() {
  if (!task.value || !message.value.trim()) return;
  sending.value = true;
  error.value = "";
  try {
    task.value = await sendTaskMessage(task.value.id, { content: message.value });
    if (task.value.draft_artifact) {
      editorContent.value = task.value.draft_artifact;
      hasGeneratedDraft.value = true;
    }
    message.value = "";
  } catch (err) {
    const requestError = err instanceof Error
      ? err.message
      : "Task assistant request failed";
    await loadTask();
    error.value = requestError;
  } finally {
    sending.value = false;
  }
}

async function saveVersion() {
  if (!task.value || !hasUnsavedChanges.value || !editorContent.value.trim()) return;
  savingArtifact.value = true;
  error.value = "";
  try {
    task.value = await saveTaskArtifact(task.value.id, editorContent.value);
    const latest = task.value.latest_artifact;
    savedBaseline.value = latest?.content ?? "";
    editorContent.value = savedBaseline.value;
    selectedVersionId.value = latest?.id ?? "";
    hasGeneratedDraft.value = false;
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not save artifact version";
  } finally {
    savingArtifact.value = false;
  }
}

function cancelEdit() {
  editorContent.value = savedBaseline.value;
  hasGeneratedDraft.value = false;
}

function selectVersion(version: NonNullable<TaskRecord["latest_artifact"]>) {
  if (version.id === selectedVersionId.value && !hasGeneratedDraft.value && !hasUnsavedChanges.value) return;
  if (
    hasUnsavedChanges.value
    && !window.confirm("Discard the unsaved draft or edits and open this saved version?")
  ) return;
  selectedVersionId.value = version.id;
  savedBaseline.value = version.content;
  editorContent.value = version.content;
  hasGeneratedDraft.value = false;
}

async function deleteSelectedVersion() {
  if (!task.value || !selectedVersion.value || hasGeneratedDraft.value) return;
  const unsavedMessage = hasUnsavedChanges.value
    ? " Unsaved editor changes will also be discarded."
    : "";
  const confirmed = window.confirm(
    `Delete Version ${selectedVersion.value.version} from the visible version history? It will no longer be selectable or downloadable.${unsavedMessage}`,
  );
  if (!confirmed) return;

  deletingArtifact.value = true;
  error.value = "";
  try {
    task.value = await deleteTaskArtifact(task.value.id, selectedVersion.value.id);
    const latest = task.value.latest_artifact;
    selectedVersionId.value = latest?.id ?? "";
    savedBaseline.value = latest?.content ?? "";
    editorContent.value = savedBaseline.value;
    hasGeneratedDraft.value = false;
  } catch (err) {
    error.value = err instanceof Error ? err.message : "Could not delete artifact version";
  } finally {
    deletingArtifact.value = false;
  }
}

function downloadArtifact() {
  if (!task.value || !editorContent.value) return;
  const extension: Record<TaskArtifactType, string> = {
    n8n: "json",
    python: "py",
    bash: "sh",
    powershell: "ps1",
    "windows-cmd": "cmd",
  };
  const safeName = task.value.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "loghawk-task";
  const url = URL.createObjectURL(new Blob([editorContent.value], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${safeName}.${extension[task.value.artifact_type]}`;
  link.click();
  URL.revokeObjectURL(url);
}

watch(taskId, () => void loadTask());
watch(editorContent, () => { importedWorkflow.value = null; });
onMounted(() => void loadTask());
</script>

<template>
  <div class="view">
    <header class="view-header">
      <div><h1>{{ task ? task.name : "Create Task" }}</h1><p>Collaborate with the configured LLM to draft an artifact. Nothing is executed or activated.</p></div>
      <RouterLink class="btn" to="/tasks">Back to Tasks</RouterLink>
    </header>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="loading" class="muted">Loading Task…</p>
    <template v-else-if="!task">
      <section class="card">
        <div class="form-grid">
          <label class="field">Task name<input v-model="name" placeholder="e.g. Restart unhealthy service" /></label>
          <label class="field">Description · what this Task is for<textarea v-model="description" rows="3" placeholder="Describe the purpose, inputs, and outcome so users and the assistant can find the right Task." /></label>
          <label class="field">Category<input v-model="category" placeholder="e.g. Kubernetes" /></label>
          <label class="field">Artifact type
            <select v-model="artifactType"><option value="python">Python program</option><option value="bash">Bash</option><option value="powershell">PowerShell</option><option value="windows-cmd">Windows CMD</option><option value="n8n">n8n workflow JSON</option></select>
          </label>
          <label class="field">Task kind
            <select v-model="taskKind"><option value="general">General Task</option><option value="remediation">Remediation Task</option><option value="rollback">Rollback Task</option><option value="collector">Collector</option></select>
          </label>
          <label class="field">Execution policy
            <select v-model="executionPolicy">
              <option value="manual">Manual</option>
              <option value="ai-assisted">AI-assisted</option>
              <option value="auto-approved" :disabled="!canSetAutoApproved">Auto-approved</option>
              <option value="disabled">Disabled</option>
            </select>
          </label>
        </div>
      </section>
      <div class="actions"><button class="btn primary" :disabled="saving || !name.trim() || !description.trim() || !category.trim()" @click="create">{{ saving ? "Creating…" : "Create Task" }}</button><RouterLink class="btn" to="/tasks">Cancel</RouterLink></div>
    </template>
    <template v-else>
      <section class="card">
        <p><strong>Category:</strong> {{ task.category }}</p>
        <div class="field task-description-field">
          <label for="task-description">Description · what this Task is for</label>
          <textarea id="task-description" v-model="descriptionDraft" rows="3" />
          <button class="btn" :disabled="saving || !descriptionDraft.trim() || descriptionDraft === task.description" @click="saveDescription">{{ saving ? "Saving…" : "Save Description" }}</button>
        </div>
        <div class="form-grid execution-policy-setting">
          <label class="field">Execution policy
            <select v-model="executionPolicy">
              <option value="manual">Manual</option>
              <option value="ai-assisted">AI-assisted</option>
              <option value="auto-approved" :disabled="!canSetAutoApproved">Auto-approved</option>
              <option value="disabled">Disabled</option>
            </select>
          </label>
          <button class="btn" :disabled="savingExecutionPolicy || executionPolicy === task.execution_policy" @click="saveExecutionPolicy">
            {{ savingExecutionPolicy ? "Saving…" : "Save Execution Policy" }}
          </button>
          <p class="muted">AI-triggered runs are not connected yet. When available, AI may request only the existing approved and enabled version. Auto-approved removes per-run approval; it does not let AI change the Task code. Local development commands remain available for testing.</p>
        </div>
        <div class="task-type-notice">
          <strong>Artifact type:</strong>
          {{ task.artifact_type === "n8n" ? "n8n workflow JSON" : task.artifact_type === "python" ? "Python program" : task.artifact_type === "windows-cmd" ? "Windows CMD script" : `${task.artifact_type} script` }}.
          This task type is fixed. To draft a different kind of artifact, create a new task.
        </div>
        <h2>Chat</h2>
        <div class="messages" aria-live="polite">
          <p v-if="!task.messages?.length" class="muted">Describe what you want the {{ task.artifact_type === 'windows-cmd' ? 'Windows CMD script' : task.artifact_type === 'n8n' ? 'n8n workflow' : task.artifact_type === 'python' ? 'Python program' : `${task.artifact_type} script` }} to do.</p>
          <article v-for="item in task.messages" :key="item.id" class="message" :class="item.role">
            <strong>{{ item.role === "user" ? "You" : "Assistant" }}</strong><p>{{ item.content }}</p>
          </article>
          <p v-if="sending" class="muted">Generating response…</p>
        </div>
        <form class="composer" @submit.prevent="send">
          <label class="field">Message<textarea v-model="message" rows="4" :disabled="sending" placeholder="Describe requirements, constraints, and expected behavior…" /></label>
          <button class="btn primary" :disabled="sending || !message.trim()">{{ sending ? "Sending…" : "Send" }}</button>
        </form>
      </section>
      <section v-if="task.artifacts?.length" class="card">
        <h2>Version history</h2>
        <div class="version-list">
          <button
            v-for="version in task.artifacts"
            :key="version.id"
            type="button"
            class="version-button"
            :class="{ selected: version.id === selectedVersionId && !hasGeneratedDraft }"
            @click="selectVersion(version)"
          >
            <span>Version {{ version.version }}</span>
            <small>{{ new Date(version.created_at).toLocaleString() }} · {{ version.approved ? "Approved" : "Pending approval" }} · {{ version.enabled ? "Enabled" : "Disabled" }}</small>
          </button>
        </div>
      </section>
      <section v-if="hasArtifactToEdit" class="card">
        <div class="artifact-header">
          <h2>{{ hasGeneratedDraft ? "Generated draft · Unsaved" : `Version ${selectedVersion?.version} · Editable` }}</h2>
          <div class="artifact-actions">
            <button
              v-if="task.artifact_type === 'n8n'"
              class="btn primary"
              :disabled="importingToN8n || !selectedVersionId || hasGeneratedDraft || hasUnsavedChanges"
              @click="importSavedVersionToN8n"
            >{{ importingToN8n ? "Importing…" : "Import Into n8n" }}</button>
            <a
              v-if="importedWorkflow"
              class="btn"
              :href="importedWorkflow.url"
              target="_blank"
              rel="noopener noreferrer"
            >Open in n8n</a>
            <button class="btn" :disabled="!editorContent" @click="downloadArtifact">Download</button>
            <button
              v-if="selectedVersion && !hasGeneratedDraft"
              class="btn"
              :disabled="deletingArtifact || savingArtifact"
              @click="deleteSelectedVersion"
            >{{ deletingArtifact ? "Deleting…" : "Delete" }}</button>
          </div>
        </div>
        <label class="field artifact-field">Code<textarea v-model="editorContent" rows="20" spellcheck="false" /></label>
        <div v-if="selectedVersion && !hasGeneratedDraft" class="actions policy-actions">
          <button v-if="!selectedVersion.approved && ['Administrator', 'Power User'].includes(currentUser?.role || '')" class="btn" :disabled="hasUnsavedChanges" @click="approveSelectedVersion">Approve Version</button>
          <button v-if="canManageTask" class="btn" :disabled="hasUnsavedChanges" @click="toggleSelectedVersion">{{ selectedVersion.enabled ? "Disable Version" : "Enable Version" }}</button>
          <span>API execution requires an approved and enabled version.</span>
        </div>
        <div class="actions">
          <button class="btn primary" :disabled="savingArtifact || !hasUnsavedChanges || !editorContent.trim()" @click="saveVersion">{{ savingArtifact ? "Saving…" : "Save Version" }}</button>
          <button class="btn" :disabled="savingArtifact || !hasUnsavedChanges" @click="cancelEdit">Cancel</button>
        </div>
        <p v-if="task.artifact_type === 'n8n'" class="muted">
          Save this Task version before importing it. Import creates an inactive
          workflow; review it in n8n before activating it.
        </p>
        <p v-if="importedWorkflow" class="muted">
          Imported inactive workflow “{{ importedWorkflow.name }}”.
        </p>
      </section>
    </template>
  </div>
</template>

<style scoped>
.messages { display: grid; gap: .8rem; max-height: 50vh; overflow-y: auto; margin-bottom: 1rem; }
.message { padding: .8rem; border-radius: .5rem; background: #17232d; }
.message.user { border-left: 3px solid #6bd7bb; }
.message.assistant { border-left: 3px solid #809bb0; }
.message p { margin: .35rem 0 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.composer { display: grid; gap: .75rem; }
textarea { width: 100%; padding: .7rem .75rem; color: #e8edf2; background: #17232d; border: 1px solid #344553; border-radius: .45rem; resize: vertical; }
.task-description-field { display: grid; gap: .5rem; max-width: 60rem; }
.artifact-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.artifact-header h2 { margin-bottom: 0; }
.artifact-actions { display: flex; flex-wrap: wrap; align-items: center; gap: .65rem; }
.artifact-field textarea { min-height: 24rem; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .84rem; line-height: 1.5; white-space: pre; }
.version-list { display: flex; flex-wrap: wrap; gap: .6rem; }
.version-button { display: grid; gap: .25rem; padding: .6rem .75rem; color: #d8e1e7; background: #17232d; border: 1px solid #344553; border-radius: .45rem; text-align: left; cursor: pointer; }
.version-button.selected { border-color: #6bd7bb; box-shadow: 0 0 0 1px #6bd7bb; }
.version-button small { color: #aebbc7; }
.policy-actions { align-items: center; }
.policy-actions span { color: #aebbc7; font-size: .9rem; }
@media (max-width: 520px) { .artifact-header { align-items: flex-start; flex-direction: column; } }
</style>
