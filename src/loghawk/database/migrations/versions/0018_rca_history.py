"""Durable RCA history and per-invocation evidence metadata."""
from alembic import context, op
import sqlalchemy as sa
revision = "0018_rca_history"
down_revision = "0017_builtin_json_schema"
branch_labels = None
depends_on = None

def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    for name in ("question_text", "response_text", "settings_json"):
        op.add_column("llm_tool_requests", sa.Column(name, sa.Text(), nullable=True))
    op.create_table("rca_llm_invocations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("request_id", sa.String(), sa.ForeignKey("llm_tool_requests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("explanation_attempt_id", sa.String()),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("invocation_number", sa.Integer(), nullable=False),
        sa.Column("configured_model", sa.String()), sa.Column("returned_model", sa.String()),
        sa.Column("provider", sa.String()), sa.Column("settings_json", sa.Text()),
        sa.Column("started_at", sa.String(), nullable=False), sa.Column("completed_at", sa.String()),
        sa.Column("duration_ms", sa.Float()), sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_text", sa.Text()), sa.Column("usage_json", sa.Text()))
    op.create_table("rca_evidence_views",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("request_id", sa.String(), sa.ForeignKey("llm_tool_requests.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tool_call_id", sa.String(), sa.ForeignKey("llm_tool_calls.id", ondelete="CASCADE")),
        sa.Column("invocation_id", sa.String(), sa.ForeignKey("rca_llm_invocations.id", ondelete="CASCADE")),
        sa.Column("stage", sa.String(), nullable=False), sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("saved_evidence_chars", sa.Integer()), sa.Column("input_chars", sa.Integer()),
        sa.Column("output_chars", sa.Integer()), sa.Column("effective_limit_chars", sa.Integer()),
        sa.Column("transformation_json", sa.Text()), sa.Column("model_evidence_text", sa.Text()))
    op.create_index("ix_rca_invocation_request", "rca_llm_invocations", ["request_id", "started_at"])
    op.create_index("ix_rca_evidence_request", "rca_evidence_views", ["request_id"])
    op.create_index("ix_rca_history_user_date", "llm_tool_requests", ["userid", "created_at"])

def downgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    op.drop_table("rca_evidence_views")
    op.drop_table("rca_llm_invocations")
    op.drop_index("ix_rca_history_user_date", table_name="llm_tool_requests")
    with op.batch_alter_table("llm_tool_requests") as batch:
        for name in ("question_text", "response_text", "settings_json"):
            batch.drop_column(name)
