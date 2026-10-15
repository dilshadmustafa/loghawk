"""Remove legacy shell tasks and their shell subtype column."""

from alembic import op
from sqlalchemy import Column, MetaData, String, Table, inspect


revision = "0005_distinct_task_artifact_types"
down_revision = "0004_console_rag_documents"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if "tasks" not in set(inspect(bind).get_table_names()):
        return

    # These task records and their dependent data are intentionally discarded.
    op.execute(
        "DELETE FROM task_messages WHERE task_id IN "
        "(SELECT id FROM tasks WHERE artifact_type = 'shell')"
    )
    op.execute(
        "DELETE FROM task_artifacts WHERE task_id IN "
        "(SELECT id FROM tasks WHERE artifact_type = 'shell')"
    )
    op.execute("DELETE FROM tasks WHERE artifact_type = 'shell'")

    columns = {column["name"] for column in inspect(bind).get_columns("tasks")}
    if "shell" in columns:
        if bind.dialect.name == "sqlite":
            # Rebuild without the legacy artifact_type CHECK constraint.
            legacy_tasks = Table(
                "tasks",
                MetaData(),
                Column("id", String, primary_key=True),
                Column("name", String, nullable=False, unique=True),
                Column("userid", String, nullable=True),
                Column(
                    "category",
                    String,
                    nullable=False,
                    server_default="Uncategorized",
                ),
                Column("artifact_type", String, nullable=False),
                Column("shell", String, nullable=True),
                Column("created_at", String, nullable=False),
                Column("updated_at", String, nullable=False),
            )
            with op.batch_alter_table(
                "tasks", copy_from=legacy_tasks
            ) as batch:
                batch.drop_column("shell")
        else:
            for constraint in inspect(bind).get_check_constraints("tasks"):
                sqltext = (constraint.get("sqltext") or "").lower()
                if "artifact_type" in sqltext and constraint.get("name"):
                    op.drop_constraint(
                        constraint["name"], "tasks", type_="check"
                    )
            with op.batch_alter_table("tasks") as batch:
                batch.drop_column("shell")


def downgrade():
    # Deleted task data cannot be restored.
    pass
