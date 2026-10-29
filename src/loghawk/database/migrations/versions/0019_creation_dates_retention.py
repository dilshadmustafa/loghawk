"""Creation dates and date-leading indexes; no record deletion."""

import logging
from datetime import UTC, datetime

from alembic import context, op
import sqlalchemy as sa

revision = "0019_creation_dates_retention"
down_revision = "0018_rca_history"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")
_NEW_DATES = (
    "pipeline_runs", "task_runs", "collector_runs", "llm_tool_calls",
    "rca_llm_invocations",
)
_INDEX_TABLES = {
    "server": _NEW_DATES + (
        "llm_tool_requests", "rca_evidence_views", "rag_ingestion_jobs",
        "console_llm_history", "console_command_history", "auth_sessions",
    ),
    "console": ("cli_command_history", "llm_conversation_history"),
}


def _original_date(value):
    """Only preserve known, timezone-qualified instants from original records."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC).isoformat(timespec="microseconds")
    except (ValueError, OverflowError):
        return None


def _backfill(bind, name):
    table = sa.table(name, sa.column("id", sa.String()),
                     sa.column("started_at", sa.String()),
                     sa.column("created_at", sa.String()))
    last_id = None
    skipped = 0
    while True:
        query = sa.select(table.c.id, table.c.started_at).where(table.c.created_at.is_(None))
        if last_id is not None:
            query = query.where(table.c.id > last_id)
        rows = bind.execute(query.order_by(table.c.id).limit(500)).mappings().all()
        if not rows:
            break
        updates = []
        for row in rows:
            value = _original_date(row["started_at"])
            if value is None:
                skipped += 1
            else:
                updates.append({"row_id": row["id"], "original_date": value})
        if updates:
            bind.execute(table.update().where(
                table.c.id == sa.bindparam("row_id"), table.c.created_at.is_(None)
            ).values(created_at=sa.bindparam("original_date")), updates)
        last_id = rows[-1]["id"]
    if skipped:
        logger.warning("%s: left %d creation dates unset because original dates were unknown or invalid",
                       name, skipped)


def upgrade():
    role = context.config.attributes.get("database_role", "server")
    if role not in _INDEX_TABLES:
        raise ValueError(f"Unknown database role: {role}")
    bind = op.get_bind()
    if role == "server":
        for name in _NEW_DATES:
            columns = {column["name"] for column in sa.inspect(bind).get_columns(name)}
            if "created_at" not in columns:
                op.add_column(name, sa.Column("created_at", sa.String(), nullable=True))
            _backfill(bind, name)
    for name in _INDEX_TABLES[role]:
        indexes = sa.inspect(bind).get_indexes(name)
        if not any(index["column_names"] and index["column_names"][0] == "created_at"
                   for index in indexes):
            op.create_index(f"ix_{name}_created_at", name, ["created_at"])


def downgrade():
    # Removing creation dates would discard audit information. Roll back application
    # code without removing these additive, backward-compatible schema fields.
    raise RuntimeError("Creation-date migration is forward-only to preserve audit dates.")
