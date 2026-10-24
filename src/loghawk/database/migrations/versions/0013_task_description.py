"""Add purpose descriptions to Tasks."""

from alembic import op
import sqlalchemy as sa


revision = "0013_task_description"
down_revision = "0012_normalize_task_categories_uppercase"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if "tasks" not in sa.inspect(bind).get_table_names():
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("tasks")}
    if "description" not in columns:
        with op.batch_alter_table("tasks") as batch:
            batch.add_column(
                sa.Column("description", sa.Text(), nullable=False, server_default="")
            )


def downgrade():
    bind = op.get_bind()
    if "tasks" not in sa.inspect(bind).get_table_names():
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("tasks")}
    if "description" in columns:
        with op.batch_alter_table("tasks") as batch:
            batch.drop_column("description")
