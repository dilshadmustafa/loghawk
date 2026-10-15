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

    sqlite = connection.dialect.name == "sqlite"
    foreign_keys_enabled = False
    if sqlite:
        foreign_keys_enabled = bool(
            connection.exec_driver_sql("PRAGMA foreign_keys").scalar()
        )
        connection.commit()
        if foreign_keys_enabled:
            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            connection.commit()

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    try:
        with context.begin_transaction():
            context.run_migrations()
    finally:
        if sqlite and foreign_keys_enabled:
            connection.rollback()
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            connection.commit()

    if sqlite and foreign_keys_enabled:
        violations = connection.exec_driver_sql(
            "PRAGMA foreign_key_check"
        ).fetchall()
        connection.commit()
        if violations:
            raise RuntimeError(
                f"Database migration left foreign-key violations: {violations}"
            )


run_migrations_online()
