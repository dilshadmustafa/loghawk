"""SQLAlchemy Core metadata for Console-local and Web UI persistence."""

from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    false,
    true,
)


CONSOLE_METADATA = MetaData()
SERVER_METADATA = MetaData()


def _task_tables(metadata: MetaData) -> dict[str, Table]:
    tasks = Table(
        "tasks", metadata,
        Column("id", String, primary_key=True),
        Column("name", String, nullable=False, unique=True),
        Column("description", Text, nullable=False, server_default=""),
        Column("userid", String, nullable=True),
        Column("category", String, nullable=False, server_default="Uncategorized"),
        Column("task_kind", String, nullable=False, server_default="general"),
        Column("execution_policy", String, nullable=False, server_default="manual"),
        Column("artifact_type", String, nullable=False),
        Column("created_at", String, nullable=False),
        Column("updated_at", String, nullable=False),
    )
    messages = Table(
        "task_messages", metadata,
        Column("id", String, primary_key=True),
        Column("task_id", String, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        Column("userid", String, nullable=True),
        Column("role", String, nullable=False),
        Column("content", Text, nullable=False),
        Column("created_at", String, nullable=False),
    )
    Index("task_messages_task_created", messages.c.task_id, messages.c.created_at, messages.c.id)
    artifacts = Table(
        "task_artifacts", metadata,
        Column("id", String, primary_key=True),
        Column("task_id", String, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        Column("userid", String, nullable=True),
        Column("version", Integer, nullable=False),
        Column("content", Text, nullable=False),
        Column("arguments_schema_json", Text, nullable=True),
        Column("created_at", String, nullable=False),
        Column("deleted_at", String, nullable=True),
        Column("approved", Boolean, nullable=False, server_default=false()),
        Column("enabled", Boolean, nullable=False, server_default=false()),
        Column("approved_by", String, nullable=True),
        Column("approved_at", String, nullable=True),
        UniqueConstraint("task_id", "version", name="uq_task_artifacts_task_version"),
    )
    return {"tasks": tasks, "task_messages": messages, "task_artifacts": artifacts}


CONSOLE_TABLES = _task_tables(CONSOLE_METADATA)
SERVER_TABLES = _task_tables(SERVER_METADATA)

console_rag_documents = Table(
    "rag_documents", CONSOLE_METADATA,
    Column("source_key", String, primary_key=True),
    Column("userid", String, nullable=True),
    Column("kind", String, nullable=False),
    Column("display_name", String, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("chunks_indexed", Integer, nullable=False, server_default="0"),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)
Index("console_rag_documents_display_name", console_rag_documents.c.display_name)
Index("console_rag_documents_content_sha256", console_rag_documents.c.content_sha256)

rag_ingestion_jobs = Table(
    "rag_ingestion_jobs", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("userid", String, nullable=True),
    Column("kind", String, nullable=False),
    Column("status", String, nullable=False),
    Column("source", Text, nullable=False),
    Column("payload_json", Text, nullable=False),
    Column("files_total", Integer, nullable=False, server_default="0"),
    Column("files_processed", Integer, nullable=False, server_default="0"),
    Column("chunks_indexed", Integer, nullable=False, server_default="0"),
    Column("error", Text),
    Column("warnings_json", Text, nullable=False, server_default="[]"),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("completed_at", String),
)
Index("rag_ingestion_jobs_created", rag_ingestion_jobs.c.created_at)

rag_documents = Table(
    "rag_documents", SERVER_METADATA,
    Column("source_key", String, primary_key=True),
    Column("userid", String, nullable=True),
    Column("kind", String, nullable=False),
    Column("display_name", String, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("chunks_indexed", Integer, nullable=False, server_default="0"),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)
Index("rag_documents_display_name", rag_documents.c.display_name)
Index("rag_documents_content_sha256", rag_documents.c.content_sha256)

auth_users = Table(
    "auth_users", SERVER_METADATA,
    Column("userid", String, primary_key=True),
    Column("password_hash", String, nullable=False),
    Column("role", String, nullable=False, server_default="User"),
    Column("active", Boolean, nullable=False, server_default=true()),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)
auth_sessions = Table(
    "auth_sessions", SERVER_METADATA,
    Column("token_hash", String(64), primary_key=True),
    Column("userid", String, ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
    Column("client", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("expires_at", String, nullable=False),
    Column("revoked_at", String),
)
Index("auth_sessions_userid", auth_sessions.c.userid)

collector_registry = Table(
    "collector_registry", SERVER_METADATA,
    Column("name", String, primary_key=True),
    Column("version", String, nullable=False),
    Column("display_name", String, nullable=False),
    Column("description", Text, nullable=False),
    Column("arguments_schema_json", Text, nullable=False),
    Column("approved", Boolean, nullable=False, server_default=true()),
    Column("enabled", Boolean, nullable=False, server_default=true()),
    Column("execution_policy", String, nullable=False, server_default="manual"),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)
collector_runs = Table(
    "collector_runs", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("collector_name", String, ForeignKey("collector_registry.name"), nullable=False),
    Column("collector_version", String, nullable=False),
    Column("userid", String, ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
    Column("status", String, nullable=False),
    Column("argument_keys_json", Text, nullable=False),
    Column("result_summary", Text),
    Column("error_type", String),
    Column("started_at", String, nullable=False),
    Column("completed_at", String),
)
Index("collector_runs_user_started", collector_runs.c.userid, collector_runs.c.started_at)
Index("collector_runs_name_started", collector_runs.c.collector_name, collector_runs.c.started_at)

llm_tool_requests = Table(
    "llm_tool_requests", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("userid", String, ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
    Column("origin_software", String, nullable=False, server_default="console"),
    Column("status", String, nullable=False),
    Column("state_json", Text, nullable=False),
    Column("expires_at", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)
llm_tool_calls = Table(
    "llm_tool_calls", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("request_id", String, ForeignKey("llm_tool_requests.id", ondelete="CASCADE"), nullable=False),
    Column("origin", String, nullable=False, server_default="AI"),
    Column("origin_userid", String, nullable=False),
    Column("collector_name", String, nullable=False),
    Column("collector_version", String, nullable=False),
    Column("artifact_id", String, nullable=True),
    Column("origin_software", String, nullable=False, server_default="console"),
    Column("status", String, nullable=False),
    Column("arguments_json", Text, nullable=False),
    Column("run_id", String, nullable=True),
    Column("result_text", Text, nullable=True),
    Column("started_at", String, nullable=False),
    Column("completed_at", String, nullable=True),
)

task_runs = Table(
    "task_runs", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("task_id", String, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
    Column("artifact_id", String, nullable=False),
    Column("userid", String, ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
    Column("status", String, nullable=False),
    Column("exit_code", Integer),
    Column("stdout", Text),
    Column("stderr", Text),
    Column("error_type", String),
    Column("started_at", String, nullable=False),
    Column("completed_at", String),
)
Index("task_runs_user_started", task_runs.c.userid, task_runs.c.started_at)
Index("task_runs_task_started", task_runs.c.task_id, task_runs.c.started_at)

console_llm_history = Table(
    "console_llm_history", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("userid", String, ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", String, nullable=False),
)
Index("console_llm_history_user_created", console_llm_history.c.userid, console_llm_history.c.created_at)

console_command_history = Table(
    "console_command_history", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("userid", String, ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
    Column("terminal_id", String, nullable=False),
    Column("command", Text, nullable=False),
    Column("created_at", String, nullable=False),
)
Index("console_command_history_user_terminal_created", console_command_history.c.userid, console_command_history.c.terminal_id, console_command_history.c.created_at)

Table(
    "cli_command_history", CONSOLE_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("userid", String, nullable=True),
    Column("terminal_id", String, nullable=False),
    Column("command", Text, nullable=False),
    Column("created_at", String, nullable=False, server_default="CURRENT_TIMESTAMP"),
)
Table(
    "llm_conversation_history", CONSOLE_METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("userid", String, nullable=True),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", String, nullable=False, server_default="CURRENT_TIMESTAMP"),
)

Table(
    "config_sets", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("external_data_use", Boolean, nullable=False),
    Column("source_bucket", String),
    Column("source_batch", String),
    Column("train_sources_json", Text, nullable=False),
    Column("detect_sources_json", Text, nullable=False),
    Column("output_bucket", String, nullable=False),
    Column("output_batch", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)
pipelines = Table(
    "pipelines", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False, unique=True),
    Column("config_set_id", String, ForeignKey("config_sets.id"), nullable=False),
    Column("run_mode", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)
runs = Table(
    "pipeline_runs", SERVER_METADATA,
    Column("id", String, primary_key=True),
    Column("pipeline_id", String, ForeignKey("pipelines.id"), nullable=False),
    Column("workflow_id", String, nullable=False, unique=True),
    Column("temporal_run_id", String),
    Column("status", String, nullable=False),
    Column("started_at", String, nullable=False),
    Column("completed_at", String),
    Column("error", Text),
)
Index("pipeline_runs_pipeline_started", runs.c.pipeline_id, runs.c.started_at)
# Partial indexes are supported by SQLite and PostgreSQL and prevent duplicate active runs.
Index(
    "one_active_run_per_pipeline",
    runs.c.pipeline_id,
    unique=True,
    sqlite_where=runs.c.status == "RUNNING",
    postgresql_where=runs.c.status == "RUNNING",
)


def metadata_for_role(role: str) -> MetaData:
    if role == "console":
        return CONSOLE_METADATA
    if role == "server":
        return SERVER_METADATA
    raise ValueError(f"Unknown database role: {role}")
