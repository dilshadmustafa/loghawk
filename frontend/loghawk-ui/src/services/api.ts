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
export const getPipelineRun = (workflowId: string) =>
  request<PipelineRunRecord>(`/pipelines/runs/${encodeURIComponent(workflowId)}`);
export const getExternalBuckets = (region: string) =>
  request<string[]>(`/storage/external/buckets?region=${encodeURIComponent(region)}`);
export const getExternalRegions = () =>
  request<string[]>("/storage/external/regions");
export const getExternalFolders = (bucket: string, region: string, prefix = "") =>
  request<string[]>(`/storage/external/folders?bucket=${encodeURIComponent(bucket)}&region=${encodeURIComponent(region)}&prefix=${encodeURIComponent(prefix)}`);

export const externalSourceUrl = (source: ExternalS3Source) =>
  `s3://${source.bucket}/${source.folder ? `${source.folder.replace(/^\/+|\/+$/g, "")}/` : ""}`;
