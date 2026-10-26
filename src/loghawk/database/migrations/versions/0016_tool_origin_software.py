"""Attribute AI requests and audit events to their originating interface."""
from alembic import context, op
import sqlalchemy as sa
revision = "0016_tool_origin_software"
down_revision = "0015_tool_origin_userid"
branch_labels = None
depends_on = None

def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    for table in ("llm_tool_requests", "llm_tool_calls"):
        columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}
        if "origin_software" not in columns:
            op.add_column(table, sa.Column("origin_software", sa.String(), nullable=False,
                                          server_default="console"))

def downgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    for table in ("llm_tool_calls", "llm_tool_requests"):
        with op.batch_alter_table(table) as batch:
            batch.drop_column("origin_software")
