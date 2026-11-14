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

export type RcaMode = "quality" | "balanced" | "performance";
export interface RcaLimits {
  MAX_MODEL_TURNS: number;
  MAX_DISCOVERY_CALLS: number;
  MAX_EXECUTION_REQUESTS: number;
  MAX_RAG_SEARCHES: number;
}
export interface RcaProfile {
  mode: RcaMode;
  label: string;
  description: string;
  limits: RcaLimits;
  preset_limits: RcaLimits;
  limited_by_configuration: boolean;
}
export interface RcaProfiles { default_mode: RcaMode; profiles: RcaProfile[]; }
export interface RcaProgressEvent {
  id: string; collector_name: string; collector_version: string; tool_type?: string;
  status: string; arguments: Record<string, unknown> | null; run_id: string | null;
  started_at: string; completed_at: string | null; summary: string;
}
export interface RcaProgressInvocation {
  id: string; stage: string; invocation_number: number; status: string;
  started_at: string; completed_at: string | null; duration_ms: number | null;
  error_text: string | null;
  settings: { effective_timeout_seconds?: number; context_preparation?: Record<string, unknown> } | null;
}
export interface RcaLiveProgress {
  snapshot_stale?: boolean;
  investigation_id?: string;
  request_id: string; status: string; revision: string;
  created_at: string; updated_at: string; server_time: string;
  settings: Record<string, unknown> | null;
  live_progress: Record<string, unknown> | null;
  events: RcaProgressEvent[]; invocations: RcaProgressInvocation[];
  answer?: RcaAnswer;
  pending?: RcaAnswer["pending"];
}
export interface RcaMessage { role: "user" | "assistant"; content: string; }
export interface RcaLoopProgress {
  model_turns: number;
  discovery_calls: number;
  execution_requests: number;
  knowledge_searches: number;
  limits: Record<string, number>;
  remaining_seconds: number;
  partial: boolean;
  stop_reason: string;
}
export interface RcaAssessment {
  observations: { text: string; references: string[] }[];
  hypotheses: { text: string; references: string[] }[];
  contradictions: { text: string; references: string[] }[];
  missing_evidence: string[];
  conclusion: { text: string; references: string[] };
  next_steps: string[];
}
export interface RcaExecutionSummary {
  loop_progress?: RcaLoopProgress | null;
  assessment?: RcaAssessment | null;
  request_id: string;
  request_status: string;
  discovery_calls: number;
  succeeded: number;
  failed: number;
  rejected: number;
  cancelled: number;
  running: number;
  awaiting_confirmation: boolean;
  evidence_redaction_enabled: boolean;
  summary: string;
  events: {
    tool_type?: string;
    call_id?: string;
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
  duplicate?: boolean;
  investigation_id?: string;
  explanation?: {
    status: "RUNNING" | "COMPLETED" | "FAILED";
    can_retry: boolean;
    can_regenerate: boolean;
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

export interface BuiltinTask {
  task_kind: TaskKind;
  artifact_type: string;
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

export interface BuiltinTaskSource {
  name: string;
  display_name: string;
  version: string;
  filename: string;
  source: string;
  arguments_schema: Record<string, unknown>;
}


export interface RcaHistoryItem {
  id: string; userid: string; origin_software: string; status: string;
  created_at: string; updated_at: string; question_text: string | null;
  execution_summary: string; explanation_status: string | null;
}
export interface RcaHistoryPage {
  items: RcaHistoryItem[]; total: number; limit: number; offset: number;
}
export interface RcaHistoryEvent {
  tool_type?: string;
  id: string; collector_name: string; collector_version: string; status: string;
  origin_userid: string; origin_software: string; origin: string;
  run_id: string | null; arguments: Record<string, unknown> | null;
  result_text: string | null; started_at: string; completed_at: string | null;
}
export interface RcaInvocation {
  id: string; stage: string; invocation_number: number; explanation_attempt_id: string | null;
  configured_model: string | null; returned_model: string | null; provider: string | null;
  settings: Record<string, unknown> | null; usage: Record<string, unknown> | null;
  started_at: string; completed_at: string | null; duration_ms: number | null;
  status: string; error_text: string | null;
}
export interface RcaEvidenceView {
  id: string; tool_call_id: string | null; invocation_id: string | null; stage: string; created_at: string;
  saved_evidence_chars: number | null; input_chars: number | null; output_chars: number;
  effective_limit_chars: number | null; transformation: Record<string, unknown> | null;
  model_evidence_text: string;
}
export interface RcaHistoryDetail {
  investigation?: {investigation_id: string; snapshot: Record<string, unknown>} | null;
  id: string; userid: string; origin_software: string; status: string;
  created_at: string; updated_at: string; question_text: string | null; response_text: string | null;
  settings: Record<string, unknown> | null; events: RcaHistoryEvent[];
  invocations: RcaInvocation[]; evidence_views: RcaEvidenceView[];
  execution: RcaExecutionSummary;
  explanation?: RcaAnswer["explanation"] | null;
}

export interface InvestigationSource {
  id: string; name: string; status: string; evidence_type: string; format: string;
  record_count: number; error?: string; sha256?: string; original_uri?: string;
  snapshot_uri?: string; mapping: Record<string, string>; tool_call_id?: string;
  is_directly_linked?: boolean | number;
}
export interface InvestigationRecord {
  id: string; record_number: number; source_line_number?: number | null; original_record: Record<string, unknown>;
}
export interface Investigation {
  id: string; userid: string; problem: string; status: string;
  created_at: string; updated_at: string; can_modify: boolean;
  reported_context: Record<string, unknown>;
  incidents: {id: string; source_incident_id: string; primary_details: Record<string, unknown>}[];
  sources: InvestigationSource[];
  evidence_links: Record<string, unknown>[]; incident_evidence: Record<string, unknown>[];
  findings: {id: string; content: string; origin: string; userid: string; created_at: string;
             request_id?: string; supersedes_id?: string; evidence: unknown[]}[];
  requests: {request_id: string; created_at: string; status: string | null; origin_software: string | null; evidence_removed: boolean}[];
}
export interface InvestigationPreview extends InvestigationSource {
  records: InvestigationRecord[]; next_offset: number | null;
}

export interface IncidentImportRecord extends InvestigationRecord {
  missing_incident_fields: string[];
}
export interface IncidentImportPreview extends InvestigationSource {
  records: IncidentImportRecord[]; next_offset: number | null;
}
export interface IncidentEvidenceSource extends InvestigationSource {
  is_attached: boolean;
  attachment_selection: number[] | null;
  attached_at?: string;
}
export interface Incident {
  id: string; userid: string; source_incident_id: string;
  primary_details: Record<string, unknown>; created_at: string; updated_at: string;
  can_modify: boolean;
  can_delete?: boolean;
  evidence_count?: number;
  investigations: {id: string; problem: string; status: string; created_at: string}[];
  sources: IncidentEvidenceSource[];
}
