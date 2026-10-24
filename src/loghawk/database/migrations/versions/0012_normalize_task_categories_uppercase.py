"""Normalize existing Task categories to uppercase."""

from alembic import op
import sqlalchemy as sa


revision = "0012_normalize_task_categories_uppercase"
down_revision = "0011_task_execution_policy_on_tasks"
branch_labels = None
depends_on = None


def upgrade():
    if "tasks" in sa.inspect(op.get_bind()).get_table_names():
        op.execute("UPDATE tasks SET category = UPPER(TRIM(category))")


def downgrade():
    pass
