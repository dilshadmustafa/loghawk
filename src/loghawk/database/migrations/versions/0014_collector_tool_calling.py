"""Versioned collector contracts and server-side AI tool requests."""

from alembic import context, op
import sqlalchemy as sa

revision = "0014_collector_tool_calling"
down_revision = "0013_task_description"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    columns = {c["name"] for c in sa.inspect(bind).get_columns("task_artifacts")}
    if "arguments_schema_json" not in columns:
        op.add_column("task_artifacts", sa.Column("arguments_schema_json", sa.Text(), nullable=True))
    if context.config.attributes.get("database_role", "server") != "server":
        return
    columns = {c["name"] for c in sa.inspect(bind).get_columns("collector_registry")}
    if "execution_policy" not in columns:
        op.add_column("collector_registry", sa.Column("execution_policy", sa.String(), nullable=False, server_default="manual"))
    tables = set(sa.inspect(bind).get_table_names())
    if "llm_tool_requests" not in tables:
        op.create_table(
            "llm_tool_requests",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("userid", sa.String(), sa.ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("state_json", sa.Text(), nullable=False),
            sa.Column("expires_at", sa.String(), nullable=False),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("updated_at", sa.String(), nullable=False),
        )
    if "llm_tool_calls" not in tables:
        op.create_table(
            "llm_tool_calls",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("request_id", sa.String(), sa.ForeignKey("llm_tool_requests.id", ondelete="CASCADE"), nullable=False),
            sa.Column("origin", sa.String(), nullable=False, server_default="ai"),
            sa.Column("collector_name", sa.String(), nullable=False),
            sa.Column("collector_version", sa.String(), nullable=False),
            sa.Column("artifact_id", sa.String(), nullable=True),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("arguments_json", sa.Text(), nullable=False),
            sa.Column("run_id", sa.String(), nullable=True),
            sa.Column("result_text", sa.Text(), nullable=True),
            sa.Column("started_at", sa.String(), nullable=False),
            sa.Column("completed_at", sa.String(), nullable=True),
        )


def downgrade():
    if context.config.attributes.get("database_role", "server") == "server":
        op.drop_table("llm_tool_calls")
        op.drop_table("llm_tool_requests")
        with op.batch_alter_table("collector_registry") as batch:
            batch.drop_column("execution_policy")
    with op.batch_alter_table("task_artifacts") as batch:
        batch.drop_column("arguments_schema_json")
