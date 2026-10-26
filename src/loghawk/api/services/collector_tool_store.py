"""Server-owned, single-use collector confirmations and AI execution audit."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from loghawk.database.database_layer import connect


def now() -> str:
    return datetime.now(UTC).isoformat()


def create(userid: str, origin_software: str = "console") -> str:
    if origin_software not in {"web", "console", "cli"}:
        raise ValueError("Unsupported origin software.")
    request_id = uuid4().hex
    timestamp = now()
    with connect("server") as db:
        # Expired proposals must never retain runnable state indefinitely.
        db.execute("UPDATE llm_tool_requests SET state_json = '{}', status = 'EXPIRED' "
                   "WHERE status = 'PENDING' AND expires_at <= ?", (timestamp,))
        db.execute("""INSERT INTO llm_tool_requests
                   (id, userid, origin_software, status, state_json, expires_at, created_at, updated_at)
                   VALUES (?, ?, ?, 'PROCESSING', '{}', ?, ?, ?)""",
                   (request_id, userid, origin_software, timestamp, timestamp, timestamp))
    return request_id


def save(request_id: str, state: dict, status: str) -> None:
    timestamp = now()
    expiry = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
    with connect("server") as db:
        db.execute("""UPDATE llm_tool_requests SET state_json = ?, status = ?,
                   expires_at = ?, updated_at = ? WHERE id = ?""",
                   (json.dumps(state) if status == "PENDING" else "{}", status,
                    expiry, timestamp, request_id))


def claim(request_id: str, userid: str, confirmation_id: str, origin_software: str = "console", *, validate=None) -> dict:
    with connect("server") as db:
        row = db.execute("SELECT * FROM llm_tool_requests WHERE id = ? AND userid = ?",
                         (request_id, userid)).fetchone()
        if not row or row["origin_software"] != origin_software or row["status"] != "PENDING" or row["expires_at"] <= now():
            raise ValueError("Confirmation expired, already handled, or not owned by this user.")
        state = json.loads(row["state_json"])
        if state["pending"]["confirmation_id"] != confirmation_id:
            raise ValueError("This confirmation does not match the current proposal.")
        # Validation is performed before the one-use claim. Invalid input leaves it pending.
        if validate is not None:
            validate(state)
        result = db.execute("""UPDATE llm_tool_requests SET status = 'PROCESSING', updated_at = ?
                            WHERE id = ? AND userid = ? AND status = 'PENDING'
                            AND state_json = ? AND expires_at > ?""",
                            (now(), request_id, userid, row["state_json"], now()))
        if result.rowcount != 1:
            raise ValueError("This confirmation was already handled.")
    return state


def record(request_id: str, target: dict, arguments: dict, status: str,
           evidence: str, run_id: str | None = None, call_id: str | None = None) -> str:
    """Arguments/evidence must already be redacted by the service."""
    timestamp = now()
    with connect("server") as db:
        if call_id:
            db.execute("""UPDATE llm_tool_calls SET status = ?, result_text = ?,
                       run_id = ?, completed_at = ? WHERE id = ?""",
                       (status, evidence, run_id, timestamp, call_id))
            return call_id
        request = db.execute(
            "SELECT userid, origin_software FROM llm_tool_requests WHERE id = ?", (request_id,),
        ).fetchone()
        if request is None:
            raise ValueError("Tool request not found.")
        origin_userid = request["userid"]
        call_id = uuid4().hex
        db.execute("""INSERT INTO llm_tool_calls
                   (id, request_id, origin, origin_userid, origin_software, collector_name, collector_version, artifact_id,
                    status, arguments_json, result_text, run_id, started_at, completed_at)
                   VALUES (?, ?, 'AI', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (call_id, request_id, origin_userid, request["origin_software"], target.get("name", "unknown"), str(target.get("version", "")),
                    target.get("artifact_id"), status, json.dumps(arguments), evidence, run_id,
                    timestamp, None if status == "RUNNING" else timestamp))
    return call_id


def execution_summary(request_id: str) -> dict:
    """Summarize recorded outcomes, never model assertions or discovery success."""
    with connect("server") as db:
        request = db.execute("SELECT status FROM llm_tool_requests WHERE id = ?", (request_id,)).fetchone()
        if request is None:
            raise ValueError("Tool request not found.")
        rows = db.execute(
            "SELECT collector_name, collector_version, status, run_id, result_text, arguments_json "
            "FROM llm_tool_calls WHERE request_id = ? ORDER BY started_at, id",
            (request_id,),
        ).fetchall()
    events = []
    counts = {"succeeded": 0, "failed": 0, "rejected": 0, "cancelled": 0, "running": 0}
    discovery_calls = 0
    for row in rows:
        if row["collector_name"] == "list_collectors" and row["status"] == "COMPLETED":
            discovery_calls += 1
            continue
        status = row["status"]
        if status in {"CONFIRMED", "AWAITING_CONFIRMATION", "REQUEST_SETTINGS"}:
            continue
        bucket = {"COMPLETED": "succeeded", "FAILED": "failed", "REJECTED": "rejected",
                  "CANCELLED": "cancelled", "RUNNING": "running"}.get(status)
        if bucket:
            counts[bucket] += 1
        events.append({"collector": row["collector_name"], "version": row["collector_version"],
                       "arguments": json.loads(row["arguments_json"])
                           if status in {"RUNNING", "COMPLETED", "FAILED"} else None,
                       "status": status, "run_id": row["run_id"],
                       "evidence": row["result_text"] if status in {"COMPLETED", "FAILED"} else None,
                       "reason": row["result_text"] if status in {"REJECTED", "CANCELLED"} else None})
    waiting = request["status"] == "PENDING"
    parts = [f"{value} {key}" for key, value in counts.items() if value]
    if waiting:
        parts.append("awaiting human confirmation")
    if not parts:
        parts.append("No collector execution was recorded")
    return {"request_id": request_id, "request_status": request["status"],
            "discovery_calls": discovery_calls, **counts, "awaiting_confirmation": waiting,
            "summary": "; ".join(parts) + ".", "events": events}
