"""Persistent command history through the shared database abstraction."""

from __future__ import annotations

from loghawk.database.timestamps import utc_now
from loghawk.database.database_layer import connect
from loghawk.utils.user_identity import get_userid


MAX_HISTORY_ENTRIES = 500
_TERMINALS = {"command-one", "command-two"}


def load_history(terminal_id: str) -> list[str]:
    if terminal_id not in _TERMINALS:
        raise ValueError(f"Unknown command terminal: {terminal_id}")
    with connect("console") as connection:
        rows = connection.execute(
            """SELECT command FROM cli_command_history
               WHERE terminal_id = ? AND userid = ?
               ORDER BY id DESC LIMIT ?""",
            (terminal_id, get_userid(), MAX_HISTORY_ENTRIES),
        ).fetchall()
    return [row["command"] for row in reversed(rows)]


def save_command(terminal_id: str, command: str) -> None:
    if terminal_id not in _TERMINALS:
        raise ValueError(f"Unknown command terminal: {terminal_id}")
    if not command.strip():
        return
    userid = get_userid()
    with connect("console") as connection:
        connection.execute(
            """INSERT INTO cli_command_history (userid, terminal_id, command, created_at)
               VALUES (?, ?, ?, ?)""",
            (userid, terminal_id, command, utc_now()),
        )
        connection.execute(
            """DELETE FROM cli_command_history
               WHERE terminal_id = ? AND userid = ? AND id NOT IN (
                   SELECT id FROM cli_command_history
                   WHERE terminal_id = ? AND userid = ? ORDER BY id DESC LIMIT ?
               )""",
            (terminal_id, userid, terminal_id, userid, MAX_HISTORY_ENTRIES),
        )
