"""Add registry for documents indexed into the shared RAG corpus."""

from alembic import context, op
from sqlalchemy import Column, Index, Integer, MetaData, String, Table, inspect


revision = "0003_rag_documents"
down_revision = "0002_rag_ingestion_jobs"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    if "rag_documents" in inspect(bind).get_table_names():
        return
    metadata = MetaData()
    documents = Table(
        "rag_documents",
        metadata,
        Column("source_key", String, primary_key=True),
        Column("userid", String, nullable=True),
        Column("kind", String, nullable=False),
        Column("display_name", String, nullable=False),
        Column("content_sha256", String(64), nullable=False),
        Column("chunks_indexed", Integer, nullable=False, server_default="0"),
        Column("created_at", String, nullable=False),
        Column("updated_at", String, nullable=False),
    )
    documents.create(bind)
    Index("rag_documents_display_name", documents.c.display_name).create(bind)
    Index("rag_documents_content_sha256", documents.c.content_sha256).create(bind)


def downgrade():
    # Preserve the RAG corpus registry when rolling back application code.
    pass
