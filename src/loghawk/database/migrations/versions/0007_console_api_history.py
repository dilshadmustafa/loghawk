"""Store signed-in Hawk Console chat and command history on the server."""

from alembic import context, op
import sqlalchemy as sa


revision = "0007_console_api_history"
down_revision = "0006_authentication"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "console_llm_history" not in tables:
        op.create_table(
            "console_llm_history",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("userid", sa.String(), sa.ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
            sa.Column("role", sa.String(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("created_at", sa.String(), nullable=False),
        )
        op.create_index("console_llm_history_user_created", "console_llm_history", ["userid", "created_at"])
    if "console_command_history" not in tables:
        op.create_table(
            "console_command_history",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("userid", sa.String(), sa.ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
            sa.Column("terminal_id", sa.String(), nullable=False),
            sa.Column("command", sa.Text(), nullable=False),
            sa.Column("created_at", sa.String(), nullable=False),
        )
        op.create_index(
            "console_command_history_user_terminal_created",
            "console_command_history", ["userid", "terminal_id", "created_at"],
        )


def downgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "console_command_history" in tables:
        op.drop_index("console_command_history_user_terminal_created", table_name="console_command_history")
        op.drop_table("console_command_history")
    if "console_llm_history" in tables:
        op.drop_index("console_llm_history_user_created", table_name="console_llm_history")
        op.drop_table("console_llm_history")
