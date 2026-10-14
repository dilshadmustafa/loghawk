"""Initialize or upgrade LogHawk's role-specific persistence schema."""

from alembic import context, op

from loghawk.database.database_layer import migrate_existing_schema


revision = "0001_database_abstraction"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    role = context.config.attributes.get("database_role", "server")
    migrate_existing_schema(op.get_bind(), role)


def downgrade():
    # Keep application data intact; this baseline migration is intentionally irreversible.
    pass
