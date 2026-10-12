"""Persistent command history for Hawk CLI command terminals."""

from __future__ import annotations

import sqlite3

from loghawk.cli.local_store import LOCAL_DB_PATH
from loghawk.utils.user_identity import get_userid


MAX_HISTORY_ENTRIES = 500
_TERMINALS = {"command-one", "command-two"}
_DB_PATH = LOCAL_DB_PATH


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_DB_PATH, timeout=15)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS cli_command_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userid TEXT NOT NULL,
            terminal_id TEXT NOT NULL
                CHECK (terminal_id IN ('command-one', 'command-two')),
            command TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(cli_command_history)")
    }
    if "userid" not in columns:
        connection.execute("ALTER TABLE cli_command_history ADD COLUMN userid TEXT")
    connection.execute(
        "UPDATE cli_command_history SET userid = ? "
        "WHERE userid IS NULL OR TRIM(userid) = ''",
        (get_userid(),),
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS cli_command_history_terminal_user_id
           ON cli_command_history (terminal_id, userid, id)"""
    )
    connection.commit()
    return connection


def load_history(terminal_id: str) -> list[str]:
    if terminal_id not in _TERMINALS:
        raise ValueError(f"Unknown command terminal: {terminal_id}")
    connection = _connect()
    try:
        rows = connection.execute(
            """SELECT command FROM cli_command_history
               WHERE terminal_id = ? AND userid = ?
               ORDER BY id DESC LIMIT ?""",
            (terminal_id, get_userid(), MAX_HISTORY_ENTRIES),
        ).fetchall()
        return [row[0] for row in reversed(rows)]
    finally:
        connection.close()


def save_command(terminal_id: str, command: str) -> None:
    if terminal_id not in _TERMINALS:
        raise ValueError(f"Unknown command terminal: {terminal_id}")
    if not command.strip():
        return
    connection = _connect()
    try:
        connection.execute(
            """INSERT INTO cli_command_history (userid, terminal_id, command)
               VALUES (?, ?, ?)""",
            (get_userid(), terminal_id, command),
        )
        connection.execute(
            """DELETE FROM cli_command_history
               WHERE terminal_id = ? AND userid = ? AND id NOT IN (
                   SELECT id FROM cli_command_history
                   WHERE terminal_id = ? AND userid = ? ORDER BY id DESC LIMIT ?
               )""",
            (terminal_id, get_userid(), terminal_id, get_userid(), MAX_HISTORY_ENTRIES),
        )
        connection.commit()
    finally:
        connection.close()
