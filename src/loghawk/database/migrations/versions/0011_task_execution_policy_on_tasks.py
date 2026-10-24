"""Add task-level execution policy."""

from alembic import op
import sqlalchemy as sa


revision = "0011_task_execution_policy_on_tasks"
down_revision = "0010_collector_approval"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if "tasks" not in sa.inspect(bind).get_table_names():
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("tasks")}
    if "execution_policy" not in columns:
        with op.batch_alter_table("tasks") as batch:
            batch.add_column(
                sa.Column("execution_policy", sa.String(), nullable=False, server_default="manual")
            )


def downgrade():
    bind = op.get_bind()
    if "tasks" not in sa.inspect(bind).get_table_names():
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("tasks")}
    if "execution_policy" in columns:
        with op.batch_alter_table("tasks") as batch:
            batch.drop_column("execution_policy")
