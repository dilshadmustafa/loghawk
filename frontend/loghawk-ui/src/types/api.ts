export type RunMode = "train" | "detect" | "train-detect";
export type PhaseMode = RunMode;
export type TaskArtifactType = "n8n" | "python" | "bash" | "powershell" | "windows-cmd";
export type TaskKind = "general" | "remediation" | "rollback" | "collector";
export type TaskExecutionPolicy = "manual" | "ai-assisted" | "auto-approved" | "disabled";

export interface TaskMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface TaskArtifact {
  arguments_schema?: Record<string, unknown> | null;
  id: string;
  version: number;
  content: string;
  created_at: string;
  approved: boolean;
  enabled: boolean;
  approved_by: string | null;
  approved_at: string | null;
}

export interface TaskRecord {
  id: string;
  name: string;
  description: string;
  userid: string | null;
  category: string;
  task_kind: TaskKind;
  execution_policy: TaskExecutionPolicy;
  artifact_type: TaskArtifactType;
  created_at: string;
  updated_at: string;
  messages?: TaskMessage[];
  latest_artifact?: TaskArtifact | null;
  artifacts?: TaskArtifact[];
  draft_artifact?: string | null;
}

export interface TaskInput {
  name: string;
  description: string;
  category: string;
  task_kind: TaskKind;
  execution_policy: TaskExecutionPolicy;
  artifact_type: TaskArtifactType;
}

export interface TaskDescriptionInput {
  description: string;
}

export interface TaskExecutionPolicyInput {
  execution_policy: TaskExecutionPolicy;
}

export interface TaskMessageInput {
  content: string;
}

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

export type RagJobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "PARTIAL" | "FAILED";

export interface RagIngestionJob {
  id: string;
  userid: string | null;
  kind: "UPLOAD" | "WEB";
  status: RagJobStatus;
  source: string;
  files_total: number;
  files_processed: number;
  chunks_indexed: number;
  error: string | null;
  warnings: string[];
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface RagAnswer {
  answer: string;
  sources: { reference: string; source: string }[];
}

export interface RcaMessage { role: "user" | "assistant"; content: string; }
export interface RcaExecutionSummary {
  request_id: string;
  request_status: string;
  discovery_calls: number;
  succeeded: number;
  failed: number;
  rejected: number;
  cancelled: number;
  running: number;
  awaiting_confirmation: boolean;
  summary: string;
  events: {
    collector: string;
    version: string;
    status: string;
    run_id: string | null;
    reason: string | null;
    arguments?: Record<string, unknown> | null;
    evidence?: string | null;
  }[];
}
export interface RcaAnswer {
  explanation?: {
    status: "RUNNING" | "COMPLETED" | "FAILED";
    can_retry: boolean;
    attempts: number;
  };
  execution?: RcaExecutionSummary;
  response: string;
  status?: string;
  request_id?: string;
  pending?: {
    collector: string;
    version: string | number;
    arguments: Record<string, unknown>;
    review_arguments?: boolean;
    confirmation_id: string;
  };
}

export interface BuiltinCollector {
  name: string;
  display_name: string;
  description: string;
  version: string;
  approved: boolean;
  enabled: boolean;
  execution_policy: TaskExecutionPolicy;
  arguments_schema: Record<string, unknown>;
  updated_at: string;
}

export interface BuiltinCollectorSource {
  name: string;
  display_name: string;
  version: string;
  filename: string;
  source: string;
  arguments_schema: Record<string, unknown>;
}
