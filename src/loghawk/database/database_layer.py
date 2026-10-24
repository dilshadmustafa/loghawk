"""Connection, URL resolution, migrations, and database-neutral errors."""

from __future__ import annotations

from pathlib import Path
import re
from threading import RLock

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

import loghawk.config as config


class DatabaseIntegrityError(Exception):
    """An integrity constraint failed, independent of the selected SQL backend."""

    def __init__(self, message: str, original: Exception):
        super().__init__(message)
        self.original = original


_ENGINES: dict[str, Engine] = {}
_MIGRATED: set[str] = set()
_LOCK = RLock()
_ALEMBIC_CONFIG = config.PROJECT_ROOT / "alembic.ini"


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
    with _LOCK:
        if role in _MIGRATED:
            return
        if not _ALEMBIC_CONFIG.exists():
            raise FileNotFoundError(f"Alembic configuration not found: {_ALEMBIC_CONFIG}")
        alembic_config = Config(str(_ALEMBIC_CONFIG))
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
