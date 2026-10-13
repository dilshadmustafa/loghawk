"""Initialize Hawk Console's independent SQLite database."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import loghawk.config as config
from loghawk.api.services import task_store
from loghawk.utils.user_identity import get_userid


LOCAL_DB_PATH = Path(config.LH_DATA_DIR) / "db" / "loghawk_local_hawkconsole.sqlite3"
WEBUI_DB_PATH = Path(config.LH_DATA_DIR) / "db" / "loghawk_webui.sqlite3"
_INITIALIZED = False
# _CHAT_HISTORY_SCHEMA_VERSION = 2


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _migrate_console_schema(connection: sqlite3.Connection) -> None:
    """Keep Console-only tables, stamp rows with the OS login, and import DuckDB chat."""
    userid = get_userid()
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("DROP TABLE IF EXISTS pipeline_runs")
    connection.execute("DROP TABLE IF EXISTS pipelines")
    connection.execute("DROP TABLE IF EXISTS config_sets")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS cli_command_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            terminal_id TEXT NOT NULL
                CHECK (terminal_id IN ('command-one', 'command-two')),
            command TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS llm_conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userid TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )

    table_names = [
        row[0]
        for row in connection.execute(
            """SELECT name FROM sqlite_master
               WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"""
        ).fetchall()
    ]
    for table_name in table_names:
        table = _quote_identifier(table_name)
        columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if "userid" not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN userid TEXT")
        connection.execute(
            f"UPDATE {table} SET userid = ? "
            "WHERE userid IS NULL OR TRIM(userid) = ''",
            (userid,),
        )

    # Legacy DuckDB conversation-history migration. Conversation history has
    # already been migrated to SQLite, so this is disabled for now.
    # migration_version = connection.execute("PRAGMA user_version").fetchone()[0]
    # if migration_version < 1:
    #     connection.execute("PRAGMA user_version = 1")
    #
    # if (
    #     migration_version < _CHAT_HISTORY_SCHEMA_VERSION
    #     and config.LH_DUCKDB_FILE_PATH.exists()
    # ):
    #     import duckdb
    #
    #     table = _quote_identifier(config.LH_DUCKDB_TABLE_NAME)
    #     duckdb_connection = duckdb.connect(
    #         str(config.LH_DUCKDB_FILE_PATH), read_only=True
    #     )
    #     try:
    #         table_exists = duckdb_connection.execute(
    #             """SELECT 1 FROM information_schema.tables
    #                WHERE table_schema = 'main' AND table_name = ?""",
    #             [config.LH_DUCKDB_TABLE_NAME],
    #         ).fetchone()
    #         if table_exists:
    #             rows = duckdb_connection.execute(
    #                 f"SELECT role, content, creation_date FROM {table} "
    #                 "ORDER BY creation_date"
    #             ).fetchall()
    #             connection.executemany(
    #                 """INSERT INTO llm_conversation_history
    #                    (userid, role, content, created_at) VALUES (?, ?, ?, ?)""",
    #                 [
    #                     (userid, role, content, str(created_at))
    #                     for role, content, created_at in rows
    #                     if role in {"user", "assistant"}
    #                 ],
    #             )
    #             connection.execute(
    #                 f"PRAGMA user_version = {_CHAT_HISTORY_SCHEMA_VERSION}"
    #             )
    #     finally:
    #         duckdb_connection.close()


def initialize_console_database() -> Path:
    """Copy the Web UI database once, then use the Console-local copy."""
    global _INITIALIZED
    if _INITIALIZED:
        return LOCAL_DB_PATH

    LOCAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOCAL_DB_PATH.exists():
        if WEBUI_DB_PATH.exists():
            with tempfile.NamedTemporaryFile(
                prefix="loghawk_local_hawkconsole_",
                suffix=".sqlite3",
                dir=LOCAL_DB_PATH.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            try:
                source = sqlite3.connect(WEBUI_DB_PATH, timeout=15)
                destination = sqlite3.connect(temporary_path, timeout=15)
                try:
                    source.backup(destination)
                finally:
                    destination.close()
                    source.close()
                temporary_path.replace(LOCAL_DB_PATH)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        else:
            with sqlite3.connect(LOCAL_DB_PATH, timeout=15):
                pass

    with sqlite3.connect(LOCAL_DB_PATH, timeout=15) as connection:
        _migrate_console_schema(connection)

    task_store.configure_database_path(LOCAL_DB_PATH)
    _INITIALIZED = True
    return LOCAL_DB_PATH


def load_conversation_history() -> list[dict[str, str]]:
    initialize_console_database()
    with sqlite3.connect(LOCAL_DB_PATH, timeout=15) as connection:
        rows = connection.execute(
            """SELECT role, content FROM llm_conversation_history
               WHERE userid = ? ORDER BY id""",
            (get_userid(),),
        ).fetchall()
    return [
        {"role": str(role), "content": str(content)}
        for role, content in rows
        if role in {"user", "assistant"}
    ]


def save_conversation_message(role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported conversation role: {role}")
    initialize_console_database()
    with sqlite3.connect(LOCAL_DB_PATH, timeout=15) as connection:
        connection.execute(
            """INSERT INTO llm_conversation_history
               (userid, role, content, created_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
            (get_userid(), role, content),
        )
