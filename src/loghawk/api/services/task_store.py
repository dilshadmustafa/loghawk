"""SQLite persistence for AI-assisted Tasks, chat messages, and artifacts."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import loghawk.config as config


DB_PATH = Path(config.LH_DATA_DIR) / "db" / "loghawk_webui.sqlite3"
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH, timeout=15) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    artifact_type TEXT NOT NULL CHECK (
                        artifact_type IN ('n8n', 'python', 'shell')
                    ),
                    shell TEXT CHECK (shell IN ('bash', 'powershell') OR shell IS NULL),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_messages (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_messages_task_created
                    ON task_messages(task_id, created_at, id);
                CREATE TABLE IF NOT EXISTS task_artifacts (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT,
                    UNIQUE(task_id, version)
                );
                """
            )
            artifact_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(task_artifacts)")
            }
            if "deleted_at" not in artifact_columns:
                connection.execute(
                    "ALTER TABLE task_artifacts ADD COLUMN deleted_at TEXT"
                )
        _SCHEMA_READY = True


def _connect() -> sqlite3.Connection:
    _ensure_schema()
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _artifact(connection: sqlite3.Connection, task_id: str) -> dict | None:
    row = connection.execute(
        """SELECT id, version, content, created_at FROM task_artifacts
           WHERE task_id = ? AND deleted_at IS NULL
           ORDER BY version DESC LIMIT 1""",
        (task_id,),
    ).fetchone()
    return dict(row) if row else None


def _artifacts(connection: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = connection.execute(
        """SELECT id, version, content, created_at FROM task_artifacts
           WHERE task_id = ? AND deleted_at IS NULL ORDER BY version DESC""",
        (task_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _task(connection: sqlite3.Connection, row: sqlite3.Row, *, include_messages: bool) -> dict:
    result = dict(row)
    result["latest_artifact"] = _artifact(connection, result["id"])
    if include_messages:
        result["artifacts"] = _artifacts(connection, result["id"])
        result["messages"] = [
            dict(message)
            for message in connection.execute(
                """SELECT id, role, content, created_at FROM task_messages
                   WHERE task_id = ? ORDER BY created_at, id""",
                (result["id"],),
            ).fetchall()
        ]
    return result


def list_tasks() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM tasks ORDER BY updated_at DESC, name COLLATE NOCASE"
        ).fetchall()
        return [_task(connection, row, include_messages=False) for row in rows]


def get_task(task_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _task(connection, row, include_messages=True) if row else None


def get_artifact_version(task_id: str, artifact_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            """SELECT id, version, content, created_at FROM task_artifacts
               WHERE task_id = ? AND id = ? AND deleted_at IS NULL""",
            (task_id, artifact_id),
        ).fetchone()
    return dict(row) if row else None


def delete_task(task_id: str) -> bool:
    """Delete a Task; its messages and artifacts cascade through SQLite FKs."""
    with _connect() as connection:
        cursor = connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cursor.rowcount > 0


def delete_artifact_version(task_id: str, artifact_id: str) -> bool:
    """Hide a saved version while retaining its number to prevent reuse."""
    with _connect() as connection:
        cursor = connection.execute(
            """UPDATE task_artifacts SET deleted_at = ?
               WHERE task_id = ? AND id = ? AND deleted_at IS NULL""",
            (_now(), task_id, artifact_id),
        )
        return cursor.rowcount > 0


def create_task(data: dict) -> dict:
    task_id = uuid4().hex
    now = _now()
    with _connect() as connection:
        connection.execute(
            """INSERT INTO tasks
               (id, name, artifact_type, shell, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                data["name"],
                data["artifact_type"],
                data.get("shell"),
                now,
                now,
            ),
        )
    task = get_task(task_id)
    assert task is not None
    return task


def add_message(task_id: str, role: str, content: str) -> dict | None:
    message_id = uuid4().hex
    now = _now()
    with _connect() as connection:
        row = connection.execute(
            "SELECT id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        connection.execute(
            """INSERT INTO task_messages (id, task_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (message_id, task_id, role, content, now),
        )
        connection.execute(
            "UPDATE tasks SET updated_at = ? WHERE id = ?", (now, task_id)
        )
        saved = connection.execute(
            """SELECT id, role, content, created_at FROM task_messages
               WHERE id = ?""",
            (message_id,),
        ).fetchone()
    return dict(saved)


def get_messages(task_id: str) -> list[dict[str, str]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT role, content FROM task_messages WHERE task_id = ? ORDER BY created_at, id",
            (task_id,),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def save_artifact(task_id: str, content: str) -> dict:
    now = _now()
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS version FROM task_artifacts WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        version = int(row["version"])
        artifact_id = uuid4().hex
        connection.execute(
            """INSERT INTO task_artifacts
               (id, task_id, version, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (artifact_id, task_id, version, content, now),
        )
        connection.execute(
            "UPDATE tasks SET updated_at = ? WHERE id = ?", (now, task_id)
        )
        saved = connection.execute(
            """SELECT id, version, content, created_at FROM task_artifacts
               WHERE id = ?""",
            (artifact_id,),
        ).fetchone()
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return dict(saved)
