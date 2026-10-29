"""Task persistence through the database abstraction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from loghawk.database.timestamps import utc_now
from loghawk.database.database_layer import connect, initialize_database
from loghawk.utils.user_identity import get_userid


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _connect(database_role: str = "server"):
    initialize_database(database_role)
    return connect(database_role)


def _decode_artifact(row) -> dict:
    result = dict(row)
    raw = result.pop("arguments_schema_json", None)
    result["arguments_schema"] = json.loads(raw) if raw else None
    return result


def _artifact(connection, task_id: str) -> dict | None:
    row = connection.execute(
        """SELECT id, version, content, created_at, approved, enabled,
                  approved_by, approved_at, arguments_schema_json FROM task_artifacts
           WHERE task_id = ? AND deleted_at IS NULL
           ORDER BY version DESC LIMIT 1""",
        (task_id,),
    ).fetchone()
    return _decode_artifact(row) if row else None


def _artifacts(connection, task_id: str) -> list[dict]:
    rows = connection.execute(
        """SELECT id, version, content, created_at, approved, enabled,
                  approved_by, approved_at, arguments_schema_json FROM task_artifacts
           WHERE task_id = ? AND deleted_at IS NULL ORDER BY version DESC""",
        (task_id,),
    ).fetchall()
    return [_decode_artifact(row) for row in rows]


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


def list_tasks(database_role: str = "server", query: str = "") -> list[dict]:
    with _connect(database_role) as connection:
        rows = connection.execute(
            "SELECT * FROM tasks ORDER BY updated_at DESC, name COLLATE NOCASE"
        ).fetchall()
        tasks = [_task(connection, row, include_messages=False) for row in rows]
    if not query.strip():
        return tasks
    terms = [term.casefold() for term in query.split() if len(term) > 2]
    if not terms:
        terms = [query.casefold()]
    scored = []
    for task in tasks:
        fields = {
            "name": str(task.get("name") or "").casefold(),
            "description": str(task.get("description") or "").casefold(),
            "category": str(task.get("category") or "").casefold(),
            "task_kind": str(task.get("task_kind") or "").casefold(),
        }
        score = sum(
            (4 if term in fields["name"] else 0)
            + (3 if term in fields["description"] else 0)
            + (2 if term in fields["category"] else 0)
            + (1 if term in fields["task_kind"] else 0)
            for term in terms
        )
        if score:
            scored.append((score, task))
    scored.sort(key=lambda item: (-item[0], item[1]["name"].casefold()))
    return [task for _, task in scored]


def get_task(task_id: str, database_role: str = "server") -> dict | None:
    with _connect(database_role) as connection:
        row = connection.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _task(connection, row, include_messages=True) if row else None


def get_artifact_version(task_id: str, artifact_id: str, database_role: str = "server") -> dict | None:
    with _connect(database_role) as connection:
        row = connection.execute(
            """SELECT id, version, content, created_at, approved, enabled,
                      approved_by, approved_at, arguments_schema_json FROM task_artifacts
               WHERE task_id = ? AND id = ? AND deleted_at IS NULL""",
            (task_id, artifact_id),
        ).fetchone()
    return _decode_artifact(row) if row else None


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
               (id, name, category, task_kind, execution_policy, artifact_type,
                description, created_at, updated_at, userid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                data["name"],
                (data.get("category") or "Uncategorized").strip().upper(),
                data.get("task_kind") or "general",
                data.get("execution_policy") or "manual",
                data["artifact_type"],
                data.get("description", "").strip(),
                now,
                now,
                get_userid(),
            ),
        )
    task = get_task(task_id, database_role)
    assert task is not None
    return task


def set_task_description(
    task_id: str, description: str, database_role: str = "server"
) -> dict | None:
    with _connect(database_role) as connection:
        connection.execute(
            "UPDATE tasks SET description = ?, updated_at = ? WHERE id = ?",
            (description.strip(), _now(), task_id),
        )
    return get_task(task_id, database_role)


def relevant_tasks(query: str, database_role: str = "server", limit: int = 5) -> list[dict]:
    return list_tasks(database_role, query)[:limit]


def set_task_execution_policy(
    task_id: str, execution_policy: str, database_role: str = "server"
) -> dict | None:
    with _connect(database_role) as connection:
        connection.execute(
            "UPDATE tasks SET execution_policy = ?, updated_at = ? WHERE id = ?",
            (execution_policy, _now(), task_id),
        )
    return get_task(task_id, database_role)


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


def save_artifact(task_id: str, content: str, database_role: str = "server", *, arguments_schema: dict | None = None) -> dict:
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
               (id, task_id, version, content, created_at, userid, approved, enabled, arguments_schema_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (artifact_id, task_id, version, content, now, get_userid(), False, False,
             json.dumps(arguments_schema, sort_keys=True) if arguments_schema is not None else None),
        )
        connection.execute(
            "UPDATE tasks SET updated_at = ? WHERE id = ?", (now, task_id)
        )
        saved = connection.execute(
            """SELECT id, version, content, created_at, approved, enabled,
                      approved_by, approved_at, arguments_schema_json FROM task_artifacts
               WHERE id = ?""",
            (artifact_id,),
        ).fetchone()
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return _decode_artifact(saved)


def set_artifact_policy(
    task_id: str, artifact_id: str, *, approved: bool | None = None,
    enabled: bool | None = None, approved_by: str | None = None,
    database_role: str = "server",
) -> dict | None:
    updates = []
    values = []
    if approved is not None:
        updates.extend(("approved = ?", "approved_by = ?", "approved_at = ?"))
        values.extend((approved, approved_by if approved else None, _now() if approved else None))
    if enabled is not None:
        updates.append("enabled = ?")
        values.append(enabled)
    if not updates:
        return get_artifact_version(task_id, artifact_id, database_role)
    values.extend((task_id, artifact_id))
    with _connect(database_role) as connection:
        connection.execute(
            f"UPDATE task_artifacts SET {', '.join(updates)} "
            "WHERE task_id = ? AND id = ? AND deleted_at IS NULL", values
        )
    return get_artifact_version(task_id, artifact_id, database_role)


def start_task_run(task_id: str, artifact_id: str, userid: str) -> tuple[str, str]:
    run_id, started_at = uuid4().hex, utc_now()
    with _connect("server") as connection:
        connection.execute(
            """INSERT INTO task_runs (id, task_id, artifact_id, userid, status, started_at, created_at)
               VALUES (?, ?, ?, ?, 'RUNNING', ?, ?)""",
            (run_id, task_id, artifact_id, userid, started_at, started_at),
        )
    return run_id, started_at


def finish_task_run(run_id: str, result: dict) -> None:
    with _connect("server") as connection:
        connection.execute(
            """UPDATE task_runs SET status = ?, exit_code = ?, stdout = ?, stderr = ?,
                      error_type = ?, completed_at = ? WHERE id = ?""",
            (result["status"], result.get("exit_code"), result.get("stdout"),
             result.get("stderr"), result.get("error_type"), _now(), run_id),
        )
