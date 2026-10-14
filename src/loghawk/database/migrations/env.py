from alembic import context

from loghawk.database.models import metadata_for_role


config = context.config
role = config.attributes.get("database_role", "server")
target_metadata = metadata_for_role(role)


def run_migrations_offline():
    raise RuntimeError("LogHawk database migrations require an online connection")


def run_migrations_online():
    connection = config.attributes.get("connection")
    if connection is None:
        raise RuntimeError("A SQLAlchemy connection is required for migrations")
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations_online()
