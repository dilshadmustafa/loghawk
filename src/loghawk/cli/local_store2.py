"""Hawk Console persistence using the shared database abstraction."""

from __future__ import annotations

import loghawk.config as config
from loghawk.database.timestamps import utc_now
from loghawk.database.database_layer import connect, initialize_database
from loghawk.utils.user_identity import get_userid


def initialize_console_database() -> str:
    """Apply Console schema migrations and return the configured database URL."""
    initialize_database("console")
    return config.LH_CONSOLE_DATABASE_URL


def load_conversation_history() -> list[dict[str, str]]:
    initialize_database("console")
    with connect("console") as connection:
        rows = connection.execute(
            """SELECT role, content FROM llm_conversation_history
               WHERE userid = ? ORDER BY id""",
            (get_userid(),),
        ).fetchall()
    return [
        {"role": str(row["role"]), "content": str(row["content"])}
        for row in rows if row["role"] in {"user", "assistant"}
    ]


def save_conversation_message(role: str, content: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported conversation role: {role}")
    with connect("console") as connection:
        connection.execute(
            """INSERT INTO llm_conversation_history
               (userid, role, content, created_at)
               VALUES (?, ?, ?, ?)""",
            (get_userid(), role, content, utc_now()),
        )
