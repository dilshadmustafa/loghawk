"""Manifest-backed registry and audited execution for all built-in Task kinds."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

from loghawk.database.timestamps import utc_now
from loghawk.database.database_layer import connect
from loghawk.builtin_tasks.loader import load_definitions, validate_arguments as _validate_arguments



logger = logging.getLogger(__name__)

def _now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_registered() -> None:
    now = _now()
    with connect("server") as db:
        for name, definition in load_definitions().items():
            db.execute(
                """INSERT INTO builtin_task_registry
                   (name, version, display_name, description, arguments_schema_json, approved,
                    enabled, execution_policy, created_at, updated_at, task_kind, artifact_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (name) DO NOTHING""",
                (
                    name, definition["version"], definition["display_name"],
                    definition["description"],
                    json.dumps(definition["arguments_schema"], sort_keys=True),
                    definition["approved"], definition["enabled"], definition["execution_policy"], now, now,
                    definition["task_kind"], definition["artifact_type"],
                ),
            )


def list_tasks(task_kind: str | None = None) -> list[dict]:
    ensure_registered()
    with connect("server") as db:
        rows = db.execute(
            """SELECT name, version, display_name, description,
                      arguments_schema_json, approved, enabled, execution_policy, updated_at, task_kind, artifact_type
               FROM builtin_task_registry ORDER BY name"""
        ).fetchall()
    return [_deserialize_definition(row) for row in rows
            if task_kind is None or row["task_kind"] == task_kind]


def get_task(name: str) -> dict | None:
    ensure_registered()
    with connect("server") as db:
        row = db.execute(
            """SELECT name, version, display_name, description,
                      arguments_schema_json, approved, enabled, execution_policy, updated_at, task_kind, artifact_type
               FROM builtin_task_registry WHERE name = ?""",
            (name,),
        ).fetchone()
        if row is None:
            return None
        task = _deserialize_definition(row)
        task["recent_runs"] = [dict(run) for run in db.execute(
            """SELECT id, userid, origin, origin_userid, origin_software, status, result_summary, error_type, started_at, completed_at
               FROM builtin_task_runs WHERE task_name = ?
               ORDER BY started_at DESC LIMIT 10""",
            (name,),
        ).fetchall()]
    return task


def _deserialize_definition(row) -> dict:
    result = dict(row)
    result["arguments_schema"] = json.loads(result.pop("arguments_schema_json"))
    # Refresh documentation for the installed version without altering stored permissions.
    definition = load_definitions().get(result["name"])
    if definition and str(definition["version"]) == str(result["version"]):
        result["description"] = definition["description"]
        properties = result["arguments_schema"].get("properties", {})
        for name, spec in definition["arguments_schema"].get("properties", {}).items():
            if name in properties and "description" in spec:
                properties[name]["description"] = spec["description"]
    result["enabled"] = bool(result["enabled"])
    result["approved"] = bool(result["approved"])
    return result


def set_enabled(name: str, enabled: bool) -> bool:
    ensure_registered()
    with connect("server") as db:
        result = db.execute(
            "UPDATE builtin_task_registry SET enabled = ?, updated_at = ? WHERE name = ?",
            (enabled, _now(), name),
        )
    return result.rowcount > 0


def set_approved(name: str, approved: bool) -> bool:
    ensure_registered()
    with connect("server") as db:
        result = db.execute(
            "UPDATE builtin_task_registry SET approved = ?, updated_at = ? WHERE name = ?",
            (approved, _now(), name),
        )
    return result.rowcount > 0


def set_execution_policy(name: str, policy: str) -> bool:
    if policy not in {"manual", "ai-assisted", "auto-approved", "disabled"}:
        raise ValueError("Unknown execution policy")
    ensure_registered()
    with connect("server") as db:
        result = db.execute(
            "UPDATE builtin_task_registry SET execution_policy = ?, updated_at = ? WHERE name = ?",
            (policy, _now(), name),
        )
    return result.rowcount > 0


def run_task(name: str, arguments: dict, userid: str, *, timeout: float = 60,
                  expected_version: str | None = None, origin: str = "Human",
                  origin_software: str = "console", request_id: str | None = None,
                  confirmed: bool = False) -> dict:
    if origin not in {"Human", "AI"} or origin_software not in {"web", "console", "cli"}:
        raise ValueError("Invalid execution origin")
    if origin == "AI":
        # Resolve provenance from the server-owned RCA request, never model arguments.
        with connect("server") as db:
            request = db.execute("SELECT userid, origin_software FROM llm_tool_requests WHERE id = ?", (request_id,)).fetchone()
        if request is None or request["userid"] != userid:
            raise PermissionError("AI execution requires its authenticated RCA request")
        origin_software = request["origin_software"]
    task = get_task(name)
    if task is None:
        raise KeyError(name)
    if task.get("execution_policy") == "disabled":
        raise PermissionError("Task execution policy is disabled.")
    installed = load_definitions().get(name)
    if not installed or any(str(installed[key]) != str(task[key]) for key in ("version", "task_kind", "artifact_type")):
        raise PermissionError("Installed Task does not match the registered version")
    if installed["arguments_schema"] != task["arguments_schema"]:
        raise PermissionError("Installed Task schema does not match its registration")
    if origin == "AI" and (
        task["task_kind"] != "collector"
        or task["execution_policy"] not in {"ai-assisted", "auto-approved"}
        or (task["execution_policy"] == "ai-assisted" and not confirmed)
    ):
        raise PermissionError("Current policy prohibits AI execution or requires confirmation")
    if expected_version is not None and (
        str(task["version"]) != expected_version
        or str(load_definitions().get(name, {}).get("version")) != expected_version
    ):
        raise PermissionError("Task version has changed.")
    if not task["enabled"]:
        raise PermissionError("This Task is disabled by an Administrator.")
    if not task["approved"]:
        raise PermissionError("This Task has not been approved for API execution.")
    normalized = _validate_arguments(name, arguments)
    run_id = uuid4().hex
    started_at = utc_now()
    with connect("server") as db:
        db.execute(
            """INSERT INTO builtin_task_runs
               (id, task_name, task_version, userid, status,
                argument_keys_json, started_at, created_at, origin, origin_userid, origin_software)
               VALUES (?, ?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?)""",
            (run_id, name, task["version"], userid,
             json.dumps(sorted(normalized)), started_at, started_at, origin, userid, origin_software),
        )
    try:
        from loghawk.lhrunner import run_builtin_task

        result = run_builtin_task(name, normalized, timeout=timeout)
    except Exception as exc:
        logger.exception("Built-in Task %s failed", name)
        from loghawk.lhrunner import normalize_output
        result = normalize_output({
            "status": "FAILED", "exit_code": None, "stdout": "",
            "stderr": str(exc), "error_type": type(exc).__name__,
        })
    summary = str(result.get("summary", "Completed"))[:500]
    completed_at = _now()
    with connect("server") as db:
        db.execute(
            """UPDATE builtin_task_runs SET status = ?, result_summary = ?, error_type = ?, completed_at = ?
               WHERE id = ?""",
            (result["status"], summary, result.get("error_type"), completed_at, run_id),
        )
    return {
        "run_id": run_id,
        "task_name": name,
        "task_kind": task["task_kind"],
        **({"collector": name} if task["task_kind"] == "collector" else {}),
        "version": task["version"],
        "status": result["status"],
        "started_at": started_at,
        "completed_at": completed_at,
        **result,
    }
