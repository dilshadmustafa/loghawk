import type {
  BatchInfo,
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
