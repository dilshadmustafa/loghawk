"""Persistent command history for Hawk CLI command terminals."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import loghawk.config as config


MAX_HISTORY_ENTRIES = 500
_TERMINALS = {"command-one", "command-two"}
_DB_PATH = Path(config.LH_DATA_DIR) / "db" / "loghawk_webui.sqlite3"


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_DB_PATH, timeout=15)
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
        """CREATE INDEX IF NOT EXISTS cli_command_history_terminal_id
           ON cli_command_history (terminal_id, id)"""
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
               WHERE terminal_id = ? ORDER BY id DESC LIMIT ?""",
            (terminal_id, MAX_HISTORY_ENTRIES),
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
            "INSERT INTO cli_command_history (terminal_id, command) VALUES (?, ?)",
            (terminal_id, command),
        )
        connection.execute(
            """DELETE FROM cli_command_history
               WHERE terminal_id = ? AND id NOT IN (
                   SELECT id FROM cli_command_history
                   WHERE terminal_id = ? ORDER BY id DESC LIMIT ?
               )""",
            (terminal_id, terminal_id, MAX_HISTORY_ENTRIES),
        )
        connection.commit()
    finally:
        connection.close()
