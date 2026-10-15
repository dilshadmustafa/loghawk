"""Task persistence through the database abstraction."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from loghawk.database.database_layer import connect, initialize_database
from loghawk.utils.user_identity import get_userid


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(database_role: str = "server"):
    initialize_database(database_role)
    return connect(database_role)


def _artifact(connection, task_id: str) -> dict | None:
    row = connection.execute(
        """SELECT id, version, content, created_at FROM task_artifacts
           WHERE task_id = ? AND deleted_at IS NULL
           ORDER BY version DESC LIMIT 1""",
        (task_id,),
    ).fetchone()
    return dict(row) if row else None


def _artifacts(connection, task_id: str) -> list[dict]:
    rows = connection.execute(
        """SELECT id, version, content, created_at FROM task_artifacts
           WHERE task_id = ? AND deleted_at IS NULL ORDER BY version DESC""",
        (task_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _task(connection, row, *, include_messages: bool) -> dict:
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


def list_tasks(database_role: str = "server") -> list[dict]:
    with _connect(database_role) as connection:
        rows = connection.execute(
            "SELECT * FROM tasks ORDER BY updated_at DESC, name COLLATE NOCASE"
        ).fetchall()
        return [_task(connection, row, include_messages=False) for row in rows]


def get_task(task_id: str, database_role: str = "server") -> dict | None:
    with _connect(database_role) as connection:
        row = connection.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _task(connection, row, include_messages=True) if row else None


def get_artifact_version(task_id: str, artifact_id: str, database_role: str = "server") -> dict | None:
    with _connect(database_role) as connection:
        row = connection.execute(
            """SELECT id, version, content, created_at FROM task_artifacts
               WHERE task_id = ? AND id = ? AND deleted_at IS NULL""",
            (task_id, artifact_id),
        ).fetchone()
    return dict(row) if row else None


def delete_task(task_id: str, database_role: str = "server") -> bool:
    """Delete a Task; its messages and artifacts cascade through foreign keys."""
    with _connect(database_role) as connection:
        cursor = connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cursor.rowcount > 0


def delete_artifact_version(task_id: str, artifact_id: str, database_role: str = "server") -> bool:
    """Hide a saved version while retaining its number to prevent reuse."""
    with _connect(database_role) as connection:
        cursor = connection.execute(
            """UPDATE task_artifacts SET deleted_at = ?
               WHERE task_id = ? AND id = ? AND deleted_at IS NULL""",
            (_now(), task_id, artifact_id),
        )
        return cursor.rowcount > 0


def create_task(data: dict, database_role: str = "server") -> dict:
    task_id = uuid4().hex
    now = _now()
    with _connect(database_role) as connection:
        connection.execute(
            """INSERT INTO tasks
               (id, name, category, artifact_type, created_at, updated_at, userid)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                data["name"],
                data.get("category") or "Uncategorized",
                data["artifact_type"],
                now,
                now,
                get_userid(),
            ),
        )
    task = get_task(task_id, database_role)
    assert task is not None
    return task


def add_message(task_id: str, role: str, content: str, database_role: str = "server") -> dict | None:
    message_id = uuid4().hex
    now = _now()
    with _connect(database_role) as connection:
        row = connection.execute(
            "SELECT id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        connection.execute(
            """INSERT INTO task_messages
               (id, task_id, role, content, created_at, userid)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (message_id, task_id, role, content, now, get_userid()),
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


def get_messages(task_id: str, database_role: str = "server") -> list[dict[str, str]]:
    with _connect(database_role) as connection:
        rows = connection.execute(
            "SELECT role, content FROM task_messages WHERE task_id = ? ORDER BY created_at, id",
            (task_id,),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def save_artifact(task_id: str, content: str, database_role: str = "server") -> dict:
    now = _now()
    connection = _connect(database_role)
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
               (id, task_id, version, content, created_at, userid)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (artifact_id, task_id, version, content, now, get_userid()),
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
