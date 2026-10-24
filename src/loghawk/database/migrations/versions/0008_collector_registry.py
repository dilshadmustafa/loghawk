"""Add server-side read-only collector registry and run audit."""

from alembic import context, op
import sqlalchemy as sa


revision = "0008_collector_registry"
down_revision = "0007_console_api_history"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "collector_registry" not in tables:
        op.create_table(
            "collector_registry",
            sa.Column("name", sa.String(), primary_key=True),
            sa.Column("version", sa.String(), nullable=False),
            sa.Column("display_name", sa.String(), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("arguments_schema_json", sa.Text(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("updated_at", sa.String(), nullable=False),
        )
    if "collector_runs" not in tables:
        op.create_table(
            "collector_runs",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("collector_name", sa.String(), sa.ForeignKey("collector_registry.name"), nullable=False),
            sa.Column("collector_version", sa.String(), nullable=False),
            sa.Column("userid", sa.String(), sa.ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("argument_keys_json", sa.Text(), nullable=False),
            sa.Column("result_summary", sa.Text()),
            sa.Column("error_type", sa.String()),
            sa.Column("started_at", sa.String(), nullable=False),
            sa.Column("completed_at", sa.String()),
        )
        op.create_index("collector_runs_user_started", "collector_runs", ["userid", "started_at"])
        op.create_index("collector_runs_name_started", "collector_runs", ["collector_name", "started_at"])


def downgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "collector_runs" in tables:
        op.drop_index("collector_runs_name_started", table_name="collector_runs")
        op.drop_index("collector_runs_user_started", table_name="collector_runs")
        op.drop_table("collector_runs")
    if "collector_registry" in tables:
        op.drop_table("collector_registry")
