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
)


CONSOLE_METADATA = MetaData()
SERVER_METADATA = MetaData()


def _task_tables(metadata: MetaData) -> dict[str, Table]:
    tasks = Table(
        "tasks", metadata,
        Column("id", String, primary_key=True),
        Column("name", String, nullable=False, unique=True),
        Column("userid", String, nullable=True),
        Column("category", String, nullable=False, server_default="Uncategorized"),
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
        Column("created_at", String, nullable=False),
        Column("deleted_at", String, nullable=True),
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
