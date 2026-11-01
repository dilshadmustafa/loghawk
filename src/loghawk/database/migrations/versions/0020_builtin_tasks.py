"""Generalize built-in registry and runs without deleting existing records."""
from alembic import context, op
import sqlalchemy as sa

revision = "0020_builtin_tasks"
down_revision = "0019_creation_dates_retention"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    # Baseline 0001 creates current metadata on fresh installs. Historical
    # revisions also create the old names. Remove ONLY empty destination tables
    # before renaming, and refuse ambiguous populated destinations.
    pairs = (("collector_runs", "builtin_task_runs"), ("collector_registry", "builtin_task_registry"))
    for old, new in pairs:
        if old in tables and new in tables:
            table = sa.Table(new, sa.MetaData(), autoload_with=bind)
            if bind.execute(sa.select(sa.func.count()).select_from(table)).scalar():
                raise RuntimeError(f"Both {old} and populated {new} exist; migration requires review")
    for old, new in pairs:
        if old in tables and new in tables:
            op.drop_table(new)
    if "collector_registry" in tables:
        op.rename_table("collector_registry", "builtin_task_registry")
    if "collector_runs" in tables:
        op.rename_table("collector_runs", "builtin_task_runs")
    columns = {c["name"] for c in sa.inspect(bind).get_columns("builtin_task_registry")}
    if "task_kind" not in columns:
        op.add_column("builtin_task_registry", sa.Column("task_kind", sa.String(), nullable=False, server_default="collector"))
        op.add_column("builtin_task_registry", sa.Column("artifact_type", sa.String(), nullable=False, server_default="python"))
    columns = {c["name"] for c in sa.inspect(bind).get_columns("builtin_task_runs")}
    if "collector_name" in columns:
        # SQLite's native column rename preserves rows, keys and constraints.
        op.alter_column("builtin_task_runs", "collector_name", new_column_name="task_name", existing_type=sa.String())
        op.alter_column("builtin_task_runs", "collector_version", new_column_name="task_version", existing_type=sa.String())
    for column in ("origin", "origin_userid", "origin_software"):
        if column not in columns:
            op.add_column("builtin_task_runs", sa.Column(column, sa.String(), nullable=True))
    indexes = {i["name"] for i in sa.inspect(bind).get_indexes("builtin_task_runs")}
    for old, new, fields in (
        ("collector_runs_user_started", "builtin_task_runs_user_started", ["userid", "started_at"]),
        ("collector_runs_name_started", "builtin_task_runs_name_started", ["task_name", "started_at"]),
        ("ix_collector_runs_created_at", "builtin_task_runs_created_at", ["created_at"]),
    ):
        if old in indexes:
            op.drop_index(old, table_name="builtin_task_runs")
        if new not in indexes:
            op.create_index(new, "builtin_task_runs", fields)


def downgrade():
    if context.config.attributes.get("database_role", "server") == "server":
        raise RuntimeError("Built-in Task migration cannot be downgraded without losing Task kind and provenance")
