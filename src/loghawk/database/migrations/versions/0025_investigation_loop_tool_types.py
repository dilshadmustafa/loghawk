"""Distinguish collector execution, discovery, knowledge and assessment events."""
from alembic import context, op
import sqlalchemy as sa

revision = "0025_investigation_loop_tool_types"
down_revision = "0024_shared_evidence_source_membership"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    op.add_column("llm_tool_calls", sa.Column("tool_type", sa.String(), nullable=False, server_default="collector"))
    op.execute("""UPDATE llm_tool_calls SET tool_type = CASE
        WHEN collector_name = 'list_collectors' THEN 'discovery'
        WHEN collector_name IN ('request_settings', 'stop_without_execution') THEN 'control'
        WHEN collector_name = 'ai_explanation' THEN 'explanation'
        ELSE 'collector' END""")


def downgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    op.drop_column("llm_tool_calls", "tool_type")
