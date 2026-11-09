"""Add recoverable soft deletion for standalone incidents."""
from alembic import context, op
import sqlalchemy as sa

revision = "0023_incident_soft_delete"
down_revision = "0022_standalone_incidents"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") == "server":
        with op.batch_alter_table("rca_incidents") as batch:
            batch.add_column(sa.Column("deleted_at", sa.String(), nullable=True))
            batch.add_column(sa.Column("deleted_by", sa.String(), nullable=True))


def downgrade():
    raise RuntimeError("Incident deletion history must be preserved; this migration is forward-only.")
