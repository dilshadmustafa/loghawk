"""Record the originating authenticated login directly on AI audit rows."""

from alembic import context, op
import sqlalchemy as sa

revision = "0015_tool_origin_userid"
down_revision = "0014_collector_tool_calling"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    # Validate attribution before altering any existing data.
    orphan = bind.execute(sa.text(
        "SELECT c.id FROM llm_tool_calls c LEFT JOIN llm_tool_requests r "
        "ON r.id = c.request_id WHERE r.userid IS NULL LIMIT 1"
    )).first()
    if orphan:
        raise RuntimeError("Cannot attribute an orphan tool audit record; migration stopped.")
    columns = {c["name"] for c in sa.inspect(bind).get_columns("llm_tool_calls")}
    if "origin_userid" not in columns:
        op.add_column("llm_tool_calls", sa.Column("origin_userid", sa.String(), nullable=True))
    bind.execute(sa.text(
        "UPDATE llm_tool_calls SET origin_userid = "
        "(SELECT userid FROM llm_tool_requests WHERE llm_tool_requests.id = llm_tool_calls.request_id) "
        "WHERE origin_userid IS NULL"
    ))
    bind.execute(sa.text("UPDATE llm_tool_calls SET origin = 'AI' WHERE origin = 'ai'"))
    with op.batch_alter_table("llm_tool_calls") as batch:
        batch.alter_column("origin_userid", existing_type=sa.String(), nullable=False)
        batch.alter_column("origin", existing_type=sa.String(), server_default="AI")


def downgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    op.execute("UPDATE llm_tool_calls SET origin = 'ai' WHERE origin = 'AI'")
    with op.batch_alter_table("llm_tool_calls") as batch:
        batch.drop_column("origin_userid")
        batch.alter_column("origin", existing_type=sa.String(), server_default="ai")
