"""Add Task kinds, per-version execution policy, and API Task run history."""

from alembic import context, op
import sqlalchemy as sa


revision = "0009_task_execution_policy"
down_revision = "0008_collector_registry"
branch_labels = None
depends_on = None


def upgrade():
    role = context.config.attributes.get("database_role", "server")
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "tasks" in tables:
        columns = {column["name"] for column in sa.inspect(bind).get_columns("tasks")}
        if "task_kind" not in columns:
            with op.batch_alter_table("tasks") as batch:
                batch.add_column(
                    sa.Column("task_kind", sa.String(), nullable=False, server_default="general")
                )
    if "task_artifacts" in tables:
        columns = {column["name"] for column in sa.inspect(bind).get_columns("task_artifacts")}
        with op.batch_alter_table("task_artifacts") as batch:
            if "approved" not in columns:
                batch.add_column(
                    sa.Column("approved", sa.Boolean(), nullable=False, server_default=sa.false())
                )
            if "enabled" not in columns:
                batch.add_column(
                    sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false())
                )
            if "approved_by" not in columns:
                batch.add_column(sa.Column("approved_by", sa.String(), nullable=True))
            if "approved_at" not in columns:
                batch.add_column(sa.Column("approved_at", sa.String(), nullable=True))

    if role != "server" or "task_runs" in tables:
        return
    op.create_table(
        "task_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("task_id", sa.String(), sa.ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("userid", sa.String(), sa.ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("stdout", sa.Text(), nullable=True),
        sa.Column("stderr", sa.Text(), nullable=True),
        sa.Column("error_type", sa.String(), nullable=True),
        sa.Column("started_at", sa.String(), nullable=False),
        sa.Column("completed_at", sa.String(), nullable=True),
    )
    op.create_index("task_runs_user_started", "task_runs", ["userid", "started_at"])
    op.create_index("task_runs_task_started", "task_runs", ["task_id", "started_at"])


def downgrade():
    role = context.config.attributes.get("database_role", "server")
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if role == "server" and "task_runs" in tables:
        op.drop_index("task_runs_task_started", table_name="task_runs")
        op.drop_index("task_runs_user_started", table_name="task_runs")
        op.drop_table("task_runs")
    if "task_artifacts" in tables:
        columns = {column["name"] for column in sa.inspect(bind).get_columns("task_artifacts")}
        with op.batch_alter_table("task_artifacts") as batch:
            for name in ("approved_at", "approved_by", "enabled", "approved"):
                if name in columns:
                    batch.drop_column(name)
    if "tasks" in tables:
        columns = {column["name"] for column in sa.inspect(bind).get_columns("tasks")}
        if "task_kind" in columns:
            with op.batch_alter_table("tasks") as batch:
                batch.drop_column("task_kind")
