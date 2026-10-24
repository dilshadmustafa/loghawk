"""Mark shipped built-in collectors as approved by default."""

from alembic import context, op
import sqlalchemy as sa


revision = "0010_collector_approval"
down_revision = "0009_task_execution_policy"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    if "collector_registry" not in sa.inspect(bind).get_table_names():
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("collector_registry")}
    if "approved" not in columns:
        with op.batch_alter_table("collector_registry") as batch:
            batch.add_column(
                sa.Column("approved", sa.Boolean(), nullable=False, server_default=sa.true())
            )


def downgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    if "collector_registry" not in sa.inspect(bind).get_table_names():
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("collector_registry")}
    if "approved" in columns:
        with op.batch_alter_table("collector_registry") as batch:
            batch.drop_column("approved")
