"""Allow evidence sources to belong directly to standalone incidents."""

from alembic import context, op
import sqlalchemy as sa

revision = "0022_standalone_incidents"
down_revision = "0021_rca_investigations"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") == "server":
        with op.batch_alter_table("rca_evidence_sources") as batch:
            batch.alter_column(
                "investigation_id",
                existing_type=sa.String(),
                existing_nullable=False,
                nullable=True,
            )


def downgrade():
    raise RuntimeError("Standalone incident evidence must be preserved; this migration is forward-only.")
