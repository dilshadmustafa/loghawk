export type RunMode = "train" | "detect" | "train-detect";

export interface BatchInfo {
  name: string;
  train_available: boolean;
  raw_available: boolean;
}

export interface PipelineRunStarted {
  workflow_id: string;
  mode: RunMode;
  status: string;
}

export interface PipelineRunStatus {
  workflow_id: string;
  run_id: string | null;
  status: string;
}
