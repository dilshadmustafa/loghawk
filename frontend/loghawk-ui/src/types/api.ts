export type RunMode = "train" | "detect" | "train-detect";
export type PhaseMode = RunMode;

export interface ExternalS3Source {
  bucket: string;
  folder: string;
  region: string;
}

export interface ConfigSet {
  id: string;
  name: string;
  external_data_use: boolean;
  source_bucket: string | null;
  source_batch: string | null;
  train_sources: ExternalS3Source[];
  detect_sources: ExternalS3Source[];
  output_bucket: string;
  output_batch: string;
  created_at: string;
  updated_at: string;
}

export interface ConfigSetDefaults {
  external_data_use: boolean;
  output_bucket: string;
  output_batch: string;
}

export interface ConfigSetInput {
  name: string;
  external_data_use: boolean;
  source_bucket?: string | null;
  source_batch?: string | null;
  train_sources: ExternalS3Source[];
  detect_sources: ExternalS3Source[];
  output_bucket?: string | null;
  output_batch?: string | null;
}

export interface Pipeline {
  id: string;
  name: string;
  config_set_id: string;
  config_set_name: string;
  run_mode: RunMode;
  latest_run: PipelineRunRecord | null;
}

export interface PipelineInput {
  name: string;
  config_set_id: string;
  run_mode: RunMode;
}

export interface PipelineRunRecord {
  id?: string;
  pipeline_id?: string;
  workflow_id: string;
  temporal_run_id: string | null;
  status: string;
  started_at: string;
  completed_at: string | null;
  error: string | null;
  pipeline_name?: string;
  config_set_name?: string;
  run_mode?: RunMode;
  temporal_ui_url?: string;
}

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
