"""Add persistent Web UI RAG ingestion job status."""

from alembic import context, op
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, inspect


revision = "0002_rag_ingestion_jobs"
down_revision = "0001_database_abstraction"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    if "rag_ingestion_jobs" in inspect(bind).get_table_names():
        return
    metadata = MetaData()
    Table(
        "rag_ingestion_jobs",
        metadata,
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
    ).create(bind)
    op.create_index(
        "rag_ingestion_jobs_created", "rag_ingestion_jobs", ["created_at"]
    )


def downgrade():
    # Preserve ingestion history when rolling back application code.
    pass
