"""Add server-side users and login sessions."""

from alembic import context, op
import sqlalchemy as sa


revision = "0006_authentication"
down_revision = "0005_distinct_task_artifact_types"
branch_labels = None
depends_on = None


def upgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "auth_users" not in tables:
        op.create_table(
            "auth_users",
            sa.Column("userid", sa.String(), primary_key=True),
            sa.Column("password_hash", sa.String(), nullable=False),
            sa.Column("role", sa.String(), nullable=False, server_default="User"),
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("updated_at", sa.String(), nullable=False),
        )
    if "auth_sessions" not in tables:
        op.create_table(
            "auth_sessions",
            sa.Column("token_hash", sa.String(length=64), primary_key=True),
            sa.Column("userid", sa.String(), sa.ForeignKey("auth_users.userid", ondelete="CASCADE"), nullable=False),
            sa.Column("client", sa.String(), nullable=False),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("expires_at", sa.String(), nullable=False),
            sa.Column("revoked_at", sa.String(), nullable=True),
        )
        op.create_index("auth_sessions_userid", "auth_sessions", ["userid"])


def downgrade():
    if context.config.attributes.get("database_role", "server") != "server":
        return
    op.drop_index("auth_sessions_userid", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("auth_users")
