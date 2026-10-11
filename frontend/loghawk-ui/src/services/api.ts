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
  TaskInput,
  TaskMessageInput,
  TaskRecord,
} from "../types/api";

const API_ROOT = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

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

export const getTasks = () => request<TaskRecord[]>("/tasks");
export const createTask = (data: TaskInput) =>
  request<TaskRecord>("/tasks", {
    method: "POST",
    body: JSON.stringify(data),
  });
export const getTask = (id: string) =>
  request<TaskRecord>(`/tasks/${encodeURIComponent(id)}`);
export const deleteTask = (id: string) =>
  request<void>(`/tasks/${encodeURIComponent(id)}`, { method: "DELETE" });
export const saveTaskArtifact = (id: string, content: string) =>
  request<TaskRecord>(`/tasks/${encodeURIComponent(id)}/artifacts`, {
    method: "POST",
    body: JSON.stringify({ content }),
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
