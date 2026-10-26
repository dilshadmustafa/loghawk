import { createRouter, createWebHistory } from "vue-router";
import BuiltinCollectorsView from "../views/BuiltinCollectorsView.vue";
import RcaChatView from "../views/RcaChatView.vue";
import PipelineRunsView from "../views/PipelineRunsView.vue";
import ConfigSetListView from "../views/ConfigSetListView.vue";
import ConfigSetFormView from "../views/ConfigSetFormView.vue";
import PipelineListView from "../views/PipelineListView.vue";
import PipelineFormView from "../views/PipelineFormView.vue";
import PipelineRunDetailView from "../views/PipelineRunDetailView.vue";
import TaskListView from "../views/TaskListView.vue";
import TaskWorkspaceView from "../views/TaskWorkspaceView.vue";
import RagDocumentUploadView from "../views/RagDocumentUploadView.vue";
import RagWebScrapeView from "../views/RagWebScrapeView.vue";
import LoginView from "../views/LoginView.vue";
import AccountSettingsView from "../views/AccountSettingsView.vue";
import UserManagementView from "../views/UserManagementView.vue";
import { clearAuthToken, getAuthStatus, getAuthToken, getCurrentUser } from "../services/api";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/builtin-collectors", name: "builtin-collectors", component: BuiltinCollectorsView, meta: { requiresAuth: true } },
    { path: "/rca/chat", name: "rca-chat", component: RcaChatView, meta: { requiresAuth: true } },
    { path: "/login", name: "login", component: LoginView },
    { path: "/account", name: "account", component: AccountSettingsView, meta: { requiresAuth: true } },
    { path: "/users", name: "users", component: UserManagementView, meta: { requiresAuth: true, requiresAdmin: true } },
    { path: "/", redirect: "/quick-run" },
    { path: "/quick-run", name: "quick-run", component: PipelineRunsView },
    { path: "/config-sets", name: "config-sets", component: ConfigSetListView },
    { path: "/config-sets/new", name: "config-set-create", component: ConfigSetFormView },
    { path: "/config-sets/:id/edit", name: "config-set-edit", component: ConfigSetFormView },
    { path: "/pipelines", name: "pipelines", component: PipelineListView },
    { path: "/pipelines/new", name: "pipeline-create", component: PipelineFormView },
    { path: "/pipelines/:id/edit", name: "pipeline-edit", component: PipelineFormView },
    { path: "/pipeline-runs/:workflowId", name: "pipeline-run-detail", component: PipelineRunDetailView },
    { path: "/tasks", name: "tasks", component: TaskListView },
    { path: "/tasks/new", name: "task-create", component: TaskWorkspaceView },
    { path: "/tasks/:id", name: "task-workspace", component: TaskWorkspaceView },
    { path: "/rag/documents", name: "rag-documents", component: RagDocumentUploadView },
    { path: "/rag/web-scrape", name: "rag-web-scrape", component: RagWebScrapeView },
  ],
});

router.beforeEach(async (to) => {
  const status = await getAuthStatus().catch(() => ({ webui_login: false, console_login: false }));
  if (!status.webui_login) {
    return ["login", "account", "users"].includes(String(to.name)) ? "/quick-run" : true;
  }
  if (to.name === "login") return getAuthToken() ? "/quick-run" : true;
  if (!getAuthToken()) return { name: "login", query: { redirect: to.fullPath } };
  try {
    const user = await getCurrentUser();
    if (to.meta.requiresAdmin && user.role !== "Administrator") return "/quick-run";
  } catch {
    clearAuthToken();
    return { name: "login", query: { redirect: to.fullPath } };
  }
  return true;
});

export default router;
