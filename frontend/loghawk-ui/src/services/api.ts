import type {
  BatchInfo,
  ConfigSet,
  ConfigSetDefaults,
  ConfigSetInput,
  ExternalS3Source,
  Pipeline,
  PipelineInput,
  PipelineRunRecord,
  PipelineRunStarted,
  PipelineRunStatus,
  RunMode,
  RagAnswer,
  RagIngestionJob,
  TaskInput,
  TaskArtifact,
  TaskExecutionPolicyInput,
  TaskMessageInput,
  TaskRecord,
} from "../types/api";

const API_ROOT = "/api/v1";
const AUTH_TOKEN_KEY = "loghawk.auth.token";

export const getAuthToken = () => sessionStorage.getItem(AUTH_TOKEN_KEY) || "";
export const setAuthToken = (token: string) => sessionStorage.setItem(AUTH_TOKEN_KEY, token);
export const clearAuthToken = () => sessionStorage.removeItem(AUTH_TOKEN_KEY);

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const token = getAuthToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

async function requestMultipart<T>(path: string, body: FormData): Promise<T> {
  const headers = new Headers();
  const token = getAuthToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_ROOT}${path}`, { method: "POST", body, headers });
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    throw new Error(result.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const getAuthStatus = () => request<{ webui_login: boolean; console_login: boolean }>("/auth/status");
export const login = (userid: string, password: string, client: "webui" | "console" = "webui") =>
  request<{ access_token: string; user: { userid: string; role: string } }>("/auth/login", {
    method: "POST", body: JSON.stringify({ userid, password, client }),
  });
export const getCurrentUser = () => request<{ userid: string; role: string }>("/auth/me");
export const logout = () => request<void>("/auth/logout", { method: "POST" });
export const changePassword = (current_password: string, new_password: string) =>
  request<{ status: string }>("/auth/change-password", {
    method: "POST", body: JSON.stringify({ current_password, new_password }),
  });
export type AuthUser = { userid: string; role: string; active: boolean; created_at: string; updated_at: string };
export const getUsers = () => request<AuthUser[]>("/auth/users");
export const createUser = (userid: string, password: string, role: "User" | "Power User" | "Administrator") =>
  request<AuthUser>("/auth/users", { method: "POST", body: JSON.stringify({ userid, password, role }) });
export const deleteUser = (userid: string) =>
  request<void>(`/auth/users/${encodeURIComponent(userid)}`, { method: "DELETE" });

export const uploadRagDocuments = (files: File[]) => {
  const body = new FormData();
  for (const file of files) {
    const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
    body.append("files", file, relativePath || file.name);
  }
  return requestMultipart<RagIngestionJob>("/rag/ingestions/upload", body);
};

export const startRagWebCrawl = (url: string) =>
  request<RagIngestionJob>("/rag/ingestions/web", {
    method: "POST",
    body: JSON.stringify({ url }),
  });

export const getRagIngestion = (jobId: string) =>
  request<RagIngestionJob>(`/rag/ingestions/${encodeURIComponent(jobId)}`);

export const askRagQuestion = (question: string) =>
  request<RagAnswer>("/rag/chat", {
    method: "POST",
    body: JSON.stringify({ question }),
  });

export const getBuckets = () => request<string[]>("/storage/buckets");

export const getBatches = (bucket: string) =>
  request<BatchInfo[]>(`/storage/batches?bucket=${encodeURIComponent(bucket)}`);

export const startRun = (bucket: string, batch: string, mode: RunMode) =>
  request<PipelineRunStarted>("/pipeline-runs", {
    method: "POST",
    body: JSON.stringify({ bucket, batch, mode }),
  });

export const getRunStatus = (workflowId: string) =>
  request<PipelineRunStatus>(
    `/pipeline-runs/${encodeURIComponent(workflowId)}`,
  );

export const getConfigSetDefaults = () =>
  request<ConfigSetDefaults>("/config-sets/defaults");
export const getConfigSets = () => request<ConfigSet[]>("/config-sets");
export const getConfigSet = (id: string) =>
  request<ConfigSet>(`/config-sets/${encodeURIComponent(id)}`);
export const saveConfigSet = (data: ConfigSetInput, id?: string) =>
  request<ConfigSet>(id ? `/config-sets/${encodeURIComponent(id)}` : "/config-sets", {
    method: id ? "PUT" : "POST",
    body: JSON.stringify(data),
  });
export const deleteConfigSet = (id: string) =>
  request<void>(`/config-sets/${encodeURIComponent(id)}`, { method: "DELETE" });
export const getPipelines = () => request<Pipeline[]>("/pipelines");
export const getPipeline = (id: string) =>
  request<Pipeline>(`/pipelines/${encodeURIComponent(id)}`);
export const savePipeline = (data: PipelineInput, id?: string) =>
  request<Pipeline>(id ? `/pipelines/${encodeURIComponent(id)}` : "/pipelines", {
    method: id ? "PUT" : "POST",
    body: JSON.stringify(data),
  });
export const runPipeline = (id: string) =>
  request<PipelineRunRecord>(`/pipelines/${encodeURIComponent(id)}/runs`, { method: "POST" });
export const deletePipeline = (id: string) =>
  request<void>(`/pipelines/${encodeURIComponent(id)}`, { method: "DELETE" });
export const getPipelineRun = (workflowId: string) =>
  request<PipelineRunRecord>(`/pipelines/runs/${encodeURIComponent(workflowId)}`);
export const getExternalBuckets = (region: string) =>
  request<string[]>(`/storage/external/buckets?region=${encodeURIComponent(region)}`);
export const getExternalRegions = () =>
  request<string[]>("/storage/external/regions");
export const getExternalFolders = (bucket: string, region: string, prefix = "") =>
  request<string[]>(`/storage/external/folders?bucket=${encodeURIComponent(bucket)}&region=${encodeURIComponent(region)}&prefix=${encodeURIComponent(prefix)}`);

export const getTasks = (query = "") => request<TaskRecord[]>(
  `/tasks${query ? `?q=${encodeURIComponent(query)}` : ""}`,
);
export const createTask = (data: TaskInput) =>
  request<TaskRecord>("/tasks", {
    method: "POST",
    body: JSON.stringify(data),
  });
export const getTask = (id: string) =>
  request<TaskRecord>(`/tasks/${encodeURIComponent(id)}`);
export const setTaskExecutionPolicy = (id: string, data: TaskExecutionPolicyInput) =>
  request<TaskRecord>(`/tasks/${encodeURIComponent(id)}/execution-policy`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
export const setTaskDescription = (id: string, description: string) =>
  request<TaskRecord>(`/tasks/${encodeURIComponent(id)}/description`, {
    method: "PUT",
    body: JSON.stringify({ description }),
  });
export const deleteTask = (id: string) =>
  request<void>(`/tasks/${encodeURIComponent(id)}`, { method: "DELETE" });
export const saveTaskArtifact = (id: string, content: string) =>
  request<TaskRecord>(`/tasks/${encodeURIComponent(id)}/artifacts`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
export const approveTaskArtifact = (taskId: string, artifactId: string) =>
  request<TaskArtifact>(`/tasks/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(artifactId)}/approve`, { method: "POST" });
export const setTaskArtifactEnabled = (taskId: string, artifactId: string, enabled: boolean) =>
  request<TaskArtifact>(`/tasks/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(artifactId)}/enabled`, {
    method: "PUT", body: JSON.stringify({ enabled }),
  });
export const deleteTaskArtifact = (taskId: string, artifactId: string) =>
  request<TaskRecord>(
    `/tasks/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(artifactId)}`,
    { method: "DELETE" },
  );
export const sendTaskMessage = (id: string, data: TaskMessageInput) =>
  request<TaskRecord>(`/tasks/${encodeURIComponent(id)}/messages`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const importTaskToN8n = (id: string, artifactId: string) =>
  request<{ workflow_id: string; name: string; url: string }>(
    `/tasks/${encodeURIComponent(id)}/import-to-n8n`,
    {
      method: "POST",
      body: JSON.stringify({ artifact_id: artifactId }),
    },
  );

export const externalSourceUrl = (source: ExternalS3Source) =>
  `s3://${source.bucket}/${source.folder ? `${source.folder.replace(/^\/+|\/+$/g, "")}/` : ""}`;
