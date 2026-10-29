"""Connection, URL resolution, migrations, and database-neutral errors."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from threading import RLock

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError


class DatabaseIntegrityError(Exception):
    """An integrity constraint failed, independent of the selected SQL backend."""

    def __init__(self, message: str, original: Exception):
        super().__init__(message)
        self.original = original


_ENGINES: dict[str, Engine] = {}
_MIGRATED: set[str] = set()
_LOCK = RLock()


class _Result:
    def __init__(self, result):
        self._result = result
        self.rowcount = result.rowcount

    def fetchone(self):
        return self._result.mappings().fetchone()

    def fetchall(self):
        return self._result.mappings().fetchall()


class _Connection:
    """Small DB-API-like facade backed by SQLAlchemy Core connections."""

    def __init__(self, role: str):
        initialize_database(role)
        self._connection = get_engine(role).connect()

    def execute(self, statement: str, parameters=()):
        normalized = statement.strip()
        if normalized.upper() == "BEGIN IMMEDIATE":
            if self._connection.dialect.name == "sqlite":
                self._connection.exec_driver_sql("BEGIN IMMEDIATE")
            return _Result(_EmptyResult())
        normalized = normalized.replace("p.name COLLATE NOCASE", "LOWER(p.name)")
        normalized = normalized.replace("name COLLATE NOCASE", "LOWER(name)")
        if isinstance(parameters, dict):
            bound = parameters
        else:
            values = tuple(parameters or ())
            index = 0

            def bind(_match):
                nonlocal index
                key = f"p{index}"
                index += 1
                return f":{key}"

            normalized = re.sub(r"\?", bind, normalized)
            bound = {f"p{i}": value for i, value in enumerate(values)}
        try:
            return _Result(self._connection.execute(text(normalized), bound))
        except IntegrityError as exc:
            raise DatabaseIntegrityError(str(exc.orig), exc) from exc

    def commit(self):
        try:
            self._connection.commit()
        except IntegrityError as exc:
            raise DatabaseIntegrityError(str(exc.orig), exc) from exc

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                try:
                    self.commit()
                except DatabaseIntegrityError:
                    raise
            else:
                self.rollback()
        finally:
            self.close()
        return False


class _EmptyResult:
    rowcount = 0

    def mappings(self):
        return self

    def fetchone(self):
        return None

    def fetchall(self):
        return []


def _url_for_role(role: str):
    import loghawk.config as config

    setting = {
        "console": config.LH_CONSOLE_DATABASE_URL,
        "server": config.LH_SERVER_DATABASE_URL,
    }.get(role)
    if setting is None:
        raise ValueError(f"Unknown database role: {role}")
    url = make_url(setting)
    if url.get_backend_name() == "sqlite" and url.database not in {None, ":memory:"}:
        database = Path(url.database)
        if not database.is_absolute():
            database = config.PROJECT_ROOT / database
        database.parent.mkdir(parents=True, exist_ok=True)
        url = url.set(database=str(database))
    return url


def get_engine(role: str) -> Engine:
    """Return the cached SQLAlchemy engine for the Console or server database."""
    with _LOCK:
        engine = _ENGINES.get(role)
        if engine is not None:
            return engine
        url = _url_for_role(role)
        kwargs = {"pool_pre_ping": True}
        if url.get_backend_name() == "sqlite":
            kwargs["connect_args"] = {"timeout": 15, "check_same_thread": False}
        engine = create_engine(url, **kwargs)
        if url.get_backend_name() == "sqlite":
            @event.listens_for(engine, "connect")
            def _sqlite_foreign_keys(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        _ENGINES[role] = engine
        return engine


def initialize_database(role: str) -> None:
    """Run versioned Alembic migrations while retaining existing records."""
    import loghawk.config as config

    alembic_path = config.PROJECT_ROOT / "alembic.ini"
    with _LOCK:
        if role in _MIGRATED:
            return
        if not alembic_path.exists():
            raise FileNotFoundError(f"Alembic configuration not found: {alembic_path}")
        alembic_config = Config(str(alembic_path))
        alembic_config.set_main_option(
            "script_location",
            str(config.PROJECT_ROOT / "src" / "loghawk" / "database" / "migrations"),
        )
        alembic_config.attributes["connection"] = get_engine(role).connect()
        alembic_config.attributes["database_role"] = role
        connection = alembic_config.attributes["connection"]
        try:
            command.upgrade(alembic_config, "head")
        finally:
            connection.close()
        _MIGRATED.add(role)


def connect(role: str):
    """Open a DB-API-like connection facade for a database role."""
    return _Connection(role)


def migrate_existing_schema(connection, role: str) -> None:
    """Create missing tables and add columns introduced after the first schema."""
    from loghawk.database.models import metadata_for_role

    metadata = metadata_for_role(role)
    metadata.create_all(connection)
    inspector = inspect(connection)
    additions = {
        "tasks": {
            "userid": "VARCHAR", "category": "VARCHAR DEFAULT 'Uncategorized'",
            "task_kind": "VARCHAR DEFAULT 'general'",
        },
        "task_messages": {"userid": "VARCHAR"},
        "task_artifacts": {
            "userid": "VARCHAR", "deleted_at": "VARCHAR",
            "approved": "BOOLEAN DEFAULT FALSE", "enabled": "BOOLEAN DEFAULT FALSE",
            "approved_by": "VARCHAR", "approved_at": "VARCHAR",
        },
    }
    if role == "console":
        additions.update({
            "cli_command_history": {"userid": "VARCHAR"},
            "llm_conversation_history": {"userid": "VARCHAR"},
        })
    for table, columns in additions.items():
        if table not in inspector.get_table_names():
            continue
        present = {column["name"] for column in inspect(connection).get_columns(table)}
        for name, sql_type in columns.items():
            if name not in present:
                connection.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}'))
    from loghawk.utils.user_identity import get_userid

    userid = get_userid()
    for table in additions:
        if table in inspect(connection).get_table_names():
            connection.execute(
                text(f'UPDATE "{table}" SET userid = :userid WHERE userid IS NULL OR TRIM(userid) = :blank'),
                {"userid": userid, "blank": ""},
            )
    if role == "console":
        # Console deliberately excludes server-owned config and pipeline tables.
        for table in ("pipeline_runs", "pipelines", "config_sets"):
            if table in inspect(connection).get_table_names():
                connection.execute(text(f'DROP TABLE "{table}"'))


@contextmanager
def maintenance_connection(role: str, writable: bool = False):
    """Yield a SQLAlchemy connection without creating files or running migrations.

    Preview uses a read-only snapshot. SQLite writes acquire BEGIN IMMEDIATE;
    callers must additionally lock relevant tables for PostgreSQL maintenance.
    Normal application configuration is intentionally imported only by normal
    connection functions: its import can create application directories.
    """
    import os
    import sqlite3
    from dotenv import load_dotenv

    defaults = {
        "server": "sqlite:///data/db/loghawk_server.sqlite3",
        "console": "sqlite:///data/db/loghawk_local_hawkconsole.sqlite3",
    }
    if role not in defaults:
        raise ValueError("Database role must be server or console.")
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")
    setting = os.getenv(f"LH_{role.upper()}_DATABASE_URL", defaults[role]).strip()
    if not setting:
        raise ValueError("The selected database URL is blank.")
    url = make_url(setting)
    backend = url.get_backend_name()
    if backend == "sqlite":
        if not url.database or url.database == ":memory:" or url.database.startswith("file:"):
            raise ValueError("Maintenance requires an existing SQLite file path.")
        database = Path(url.database)
        if not database.is_absolute():
            database = root / database
        database = database.resolve()
        if not database.is_file():
            raise FileNotFoundError("The selected SQLite database does not exist; nothing was created.")
        uri = database.as_uri() + ("?mode=rw" if writable else "?mode=ro")
        engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(uri, uri=True, timeout=15))
        location = {"backend": backend, "database": str(database)}
    elif backend == "postgresql":
        engine = create_engine(url, hide_parameters=True)
        location = {"backend": backend, "host": url.host, "port": url.port,
                    "database": url.database}
    else:
        raise ValueError("Retention currently supports SQLite and PostgreSQL only.")
    try:
        with engine.connect() as connection:
            if backend == "sqlite":
                connection.exec_driver_sql("PRAGMA foreign_keys=ON")
                if not writable:
                    connection.exec_driver_sql("PRAGMA query_only=ON")
                connection.exec_driver_sql("BEGIN IMMEDIATE" if writable else "BEGIN")
            else:
                connection.exec_driver_sql("BEGIN" if writable else
                                           "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
                connection.exec_driver_sql("SET LOCAL lock_timeout = '15s'")
            connection.info["maintenance_location"] = location
            try:
                yield connection
                if writable:
                    connection.commit()
                else:
                    connection.rollback()
            except BaseException:
                connection.rollback()
                raise
    finally:
        engine.dispose()
