"""Versioned registry and execution service for built-in read-only collectors."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

from loghawk.database.database_layer import connect
from loghawk.collectors.builtins import (
    DEFINITIONS as _DEFINITIONS,
    run_builtin as _run_builtin,
    validate_arguments as _validate_arguments,
)


logger = logging.getLogger(__name__)

def _now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_registered() -> None:
    now = _now()
    with connect("server") as db:
        for name, definition in _DEFINITIONS.items():
            db.execute(
                """INSERT INTO collector_registry
                   (name, version, display_name, description, arguments_schema_json, approved,
                    enabled, execution_policy, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (name) DO NOTHING""",
                (
                    name, definition["version"], definition["display_name"],
                    definition["description"],
                    json.dumps(definition["arguments_schema"], sort_keys=True),
                    True, True, "ai-assisted", now, now,
                ),
            )


def list_collectors() -> list[dict]:
    ensure_registered()
    with connect("server") as db:
        rows = db.execute(
            """SELECT name, version, display_name, description,
                      arguments_schema_json, approved, enabled, execution_policy, updated_at
               FROM collector_registry ORDER BY name"""
        ).fetchall()
    return [_deserialize_definition(row) for row in rows]


def get_collector(name: str) -> dict | None:
    ensure_registered()
    with connect("server") as db:
        row = db.execute(
            """SELECT name, version, display_name, description,
                      arguments_schema_json, approved, enabled, execution_policy, updated_at
               FROM collector_registry WHERE name = ?""",
            (name,),
        ).fetchone()
        if row is None:
            return None
        collector = _deserialize_definition(row)
        collector["recent_runs"] = [dict(run) for run in db.execute(
            """SELECT id, userid, status, result_summary, error_type, started_at, completed_at
               FROM collector_runs WHERE collector_name = ?
               ORDER BY started_at DESC LIMIT 10""",
            (name,),
        ).fetchall()]
    return collector


def _deserialize_definition(row) -> dict:
    result = dict(row)
    result["arguments_schema"] = json.loads(result.pop("arguments_schema_json"))
    # Refresh documentation for the installed version without altering stored permissions.
    definition = _DEFINITIONS.get(result["name"])
    if definition and str(definition["version"]) == str(result["version"]):
        result["description"] = definition["description"]
        properties = result["arguments_schema"]["properties"]
        for name, spec in definition["arguments_schema"]["properties"].items():
            if name in properties and "description" in spec:
                properties[name]["description"] = spec["description"]
    result["enabled"] = bool(result["enabled"])
    result["approved"] = bool(result["approved"])
    return result


def set_enabled(name: str, enabled: bool) -> bool:
    ensure_registered()
    with connect("server") as db:
        result = db.execute(
            "UPDATE collector_registry SET enabled = ?, updated_at = ? WHERE name = ?",
            (enabled, _now(), name),
        )
    return result.rowcount > 0


def set_approved(name: str, approved: bool) -> bool:
    ensure_registered()
    with connect("server") as db:
        result = db.execute(
            "UPDATE collector_registry SET approved = ?, updated_at = ? WHERE name = ?",
            (approved, _now(), name),
        )
    return result.rowcount > 0


def set_execution_policy(name: str, policy: str) -> bool:
    if policy not in {"manual", "ai-assisted", "auto-approved", "disabled"}:
        raise ValueError("Unknown execution policy")
    ensure_registered()
    with connect("server") as db:
        result = db.execute(
            "UPDATE collector_registry SET execution_policy = ?, updated_at = ? WHERE name = ?",
            (policy, _now(), name),
        )
    return result.rowcount > 0


def run_collector(name: str, arguments: dict, userid: str, *, timeout: float = 60,
                  expected_version: str | None = None) -> dict:
    collector = get_collector(name)
    if collector is None:
        raise KeyError(name)
    if collector.get("execution_policy") == "disabled":
        raise PermissionError("Collector execution policy is disabled.")
    if expected_version is not None and (
        str(collector["version"]) != expected_version
        or str(_DEFINITIONS.get(name, {}).get("version")) != expected_version
    ):
        raise PermissionError("Collector version has changed.")
    if not collector["enabled"]:
        raise PermissionError("This collector is disabled by an Administrator.")
    if not collector["approved"]:
        raise PermissionError("This collector has not been approved for API execution.")
    normalized = _validate_arguments(name, arguments)
    run_id = uuid4().hex
    started_at = _now()
    with connect("server") as db:
        db.execute(
            """INSERT INTO collector_runs
               (id, collector_name, collector_version, userid, status,
                argument_keys_json, started_at)
               VALUES (?, ?, ?, ?, 'RUNNING', ?, ?)""",
            (run_id, name, collector["version"], userid,
             json.dumps(sorted(normalized)), started_at),
        )
    try:
        from loghawk.lhrunner import run_builtin_collector

        result = run_builtin_collector(name, normalized, timeout=timeout)
    except Exception as exc:
        logger.exception("Read-only collector %s failed", name)
        with connect("server") as db:
            db.execute(
                """UPDATE collector_runs SET status = 'FAILED', error_type = ?, completed_at = ?
                   WHERE id = ?""",
                (type(exc).__name__, _now(), run_id),
            )
        raise
    summary = str(result.get("summary", "Completed"))[:500]
    completed_at = _now()
    with connect("server") as db:
        db.execute(
            """UPDATE collector_runs SET status = 'COMPLETED', result_summary = ?, completed_at = ?
               WHERE id = ?""",
            (summary, completed_at, run_id),
        )
    return {
        "run_id": run_id,
        "collector": name,
        "version": collector["version"],
        "status": "COMPLETED",
        "started_at": started_at,
        "completed_at": completed_at,
        **result,
    }
