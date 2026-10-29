"""Server-owned, single-use collector confirmations and AI execution audit."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from loghawk.database.timestamps import utc_now
from loghawk.database.database_layer import connect


def now() -> str:
    return utc_now()


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
                    status, arguments_json, result_text, run_id, started_at, completed_at, created_at)
                   VALUES (?, ?, 'AI', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (call_id, request_id, origin_userid, request["origin_software"], target.get("name", "unknown"), str(target.get("version", "")),
                    target.get("artifact_id"), status, json.dumps(arguments), evidence, run_id,
                    timestamp, None if status in {"RUNNING", "EXPLANATION_RUNNING"} else timestamp, timestamp))
    return call_id


def execution_summary(request_id: str) -> dict:
    """Summarize recorded outcomes, never model assertions or discovery success."""
    with connect("server") as db:
        request = db.execute("SELECT status FROM llm_tool_requests WHERE id = ?", (request_id,)).fetchone()
        if request is None:
            raise ValueError("Tool request not found.")
        rows = db.execute(
            "SELECT id, collector_name, collector_version, status, run_id, result_text, arguments_json "
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
        if status.startswith("EXPLANATION_") or status in {"CONFIRMED", "AWAITING_CONFIRMATION", "REQUEST_SETTINGS"}:
            continue
        bucket = {"COMPLETED": "succeeded", "FAILED": "failed", "REJECTED": "rejected",
                  "CANCELLED": "cancelled", "RUNNING": "running"}.get(status)
        if bucket:
            counts[bucket] += 1
        events.append({"call_id": row["id"], "collector": row["collector_name"], "version": row["collector_version"],
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


def begin_explanation(request_id: str, userid: str, origin_software: str,
                      timeout: int, context: dict | None = None) -> dict:
    """Atomically claim a text-only explanation; retain no runnable collector state."""
    timestamp = now()
    with connect("server") as db:
        row = db.execute("SELECT * FROM llm_tool_requests WHERE id = ? AND userid = ?",
                         (request_id, userid)).fetchone()
        if not row or row["origin_software"] != origin_software:
            raise PermissionError("Explanation request not found or not owned by this user.")
        previous = json.loads(row["state_json"]).get("explanation")
        if context is None:
            expired = (previous and previous.get("status") == "RUNNING"
                       and row["expires_at"] <= timestamp)
            if not previous or (previous.get("status") != "FAILED" and not expired):
                raise ValueError("Explanation is not retryable or another attempt is still running.")
            context = previous["context"]
        elif row["status"] != "PROCESSING" or previous:
            raise ValueError("This request has already entered its explanation stage.")
        explanation = {"status": "RUNNING", "attempt_id": uuid4().hex,
                       "attempts": (previous or {}).get("attempts", 0) + 1,
                       "context": context, "response": "", "error": None}
        lease = (datetime.now(UTC) + timedelta(seconds=timeout + 60)).isoformat()
        result = db.execute(
            "UPDATE llm_tool_requests SET state_json = ?, status = 'EXPLAINING', "
            "expires_at = ?, updated_at = ? WHERE id = ? AND userid = ? "
            "AND status = ? AND state_json = ?",
            (json.dumps({"explanation": explanation}), lease, timestamp,
             request_id, userid, row["status"], row["state_json"]),
        )
        if result.rowcount != 1:
            raise ValueError("Another explanation attempt already claimed this request.")
    return explanation


def finish_explanation(request_id: str, attempt_id: str, response: str, error: str | None) -> None:
    with connect("server") as db:
        row = db.execute("SELECT state_json FROM llm_tool_requests WHERE id = ?",
                         (request_id,)).fetchone()
        state = json.loads(row["state_json"]) if row else {}
        explanation = state.get("explanation", {})
        if explanation.get("attempt_id") != attempt_id or explanation.get("status") != "RUNNING":
            raise ValueError("This explanation attempt was superseded; no collector was rerun.")
        explanation.update(status="FAILED" if error else "COMPLETED", response=response, error=error)
        result = db.execute(
            "UPDATE llm_tool_requests SET state_json = ?, status = ?, updated_at = ? "
            "WHERE id = ? AND state_json = ?",
            (json.dumps(state), "FAILED" if error else "COMPLETED", now(), request_id, row["state_json"]),
        )
        if result.rowcount != 1:
            raise ValueError("Explanation state changed; no collector was rerun.")


def explanation_info(request_id: str) -> dict | None:
    with connect("server") as db:
        row = db.execute("SELECT state_json, expires_at FROM llm_tool_requests WHERE id = ?",
                         (request_id,)).fetchone()
    explanation = json.loads(row["state_json"]).get("explanation") if row else None
    if not explanation:
        return None
    retryable = explanation["status"] == "FAILED" or (
        explanation["status"] == "RUNNING" and row["expires_at"] <= now()
    )
    return {"status": explanation["status"], "can_retry": retryable,
            "attempts": explanation["attempts"]}


# History reads never expose runnable confirmation state or secret-bearing context.
def settings_snapshot() -> dict:
    from loghawk import config
    names = ("MODEL", "CONTEXT_LENGTH", "MAX_TOKENS", "MAX_CALLS", "TIMEOUT_SECONDS",
             "EXPLANATION_TIMEOUT_SECONDS", "MAX_RESULT_CHARS")
    return {"LH_LLM_TOOL_" + name: getattr(config, "LH_LLM_TOOL_" + name) for name in names}


def initialize_history(request_id: str, question: str, settings: dict) -> None:
    with connect("server") as db:
        db.execute("UPDATE llm_tool_requests SET question_text = ?, settings_json = ? WHERE id = ?",
                   (question, json.dumps(settings), request_id))


def save_response(request_id: str, response: str) -> None:
    with connect("server") as db:
        db.execute("UPDATE llm_tool_requests SET response_text = ? WHERE id = ?", (response, request_id))


def begin_invocation(request_id: str, stage: str, settings: dict, provider: str,
                     attempt_id: str | None = None) -> str:
    invocation_id = uuid4().hex
    created_at = utc_now()
    with connect("server") as db:
        number = db.execute("SELECT COUNT(*) AS n FROM rca_llm_invocations WHERE request_id = ?",
                            (request_id,)).fetchone()["n"] + 1
        db.execute("""INSERT INTO rca_llm_invocations
            (id, request_id, explanation_attempt_id, stage, invocation_number, configured_model,
             provider, settings_json, started_at, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PREPARING')""",
            (invocation_id, request_id, attempt_id, stage, number, settings.get("LH_LLM_TOOL_MODEL"),
             provider, json.dumps(settings), created_at, created_at))
    return invocation_id


def invocation_dispatch(invocation_id: str, timeout: float) -> None:
    with connect("server") as db:
        row = db.execute("SELECT settings_json FROM rca_llm_invocations WHERE id = ?", (invocation_id,)).fetchone()
        settings = json.loads(row["settings_json"])
        settings["effective_timeout_seconds"] = timeout
        db.execute("UPDATE rca_llm_invocations SET settings_json = ?, status = 'RUNNING' WHERE id = ?",
                   (json.dumps(settings), invocation_id))


def finish_invocation(invocation_id: str, duration_ms: float, error: str | None, metadata: dict) -> None:
    with connect("server") as db:
        db.execute("""UPDATE rca_llm_invocations SET completed_at = ?, duration_ms = ?, status = ?,
            error_text = ?, returned_model = ?, usage_json = ? WHERE id = ?""",
            (now(), duration_ms, "FAILED" if error else "COMPLETED", error,
             metadata.get("returned_model"), json.dumps(metadata["usage"]) if metadata.get("usage") is not None else None,
             invocation_id))


def record_evidence_view(request_id, call_id, invocation_id, stage, original, prepared, limit, metadata):
    with connect("server") as db:
        db.execute("""INSERT INTO rca_evidence_views
            (id, request_id, tool_call_id, invocation_id, stage, created_at, saved_evidence_chars,
             input_chars, output_chars, effective_limit_chars, transformation_json, model_evidence_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (uuid4().hex, request_id, call_id, invocation_id, stage, now(), len(original) if original is not None else None,
             len(original) if original is not None else None, len(prepared), limit, json.dumps(metadata), prepared))


def _json_fields(row, fields):
    result = dict(row)
    for name in fields:
        value = result.get(name)
        try:
            result[name.removesuffix("_json")] = json.loads(value) if value is not None else None
        except (ValueError, TypeError):
            result[name.removesuffix("_json")] = None
        result.pop(name, None)
    return result


def history_list(user, *, search="", userid="", origin="", collector="", status="",
                 date_from="", date_to="", limit=25, offset=0):
    conditions, args = [], []
    if user["role"] not in {"Administrator", "Power User"}:
        conditions.append("r.userid = ?"); args.append(user["userid"])
    elif userid:
        conditions.append("r.userid = ?"); args.append(userid)
    for column, value in (("origin_software", origin), ("status", status)):
        if value:
            conditions.append(f"r.{column} = ?"); args.append(value)
    if date_from:
        conditions.append("r.created_at >= ?"); args.append(date_from)
    if date_to:
        conditions.append("r.created_at < ?"); args.append(date_to)
    if collector:
        conditions.append("EXISTS (SELECT 1 FROM llm_tool_calls c WHERE c.request_id = r.id AND c.collector_name = ?)")
        args.append(collector)
    if search:
        conditions.append("(LOWER(COALESCE(r.question_text, '')) LIKE ? OR r.id LIKE ? OR EXISTS "
                          "(SELECT 1 FROM llm_tool_calls c WHERE c.request_id = r.id AND c.run_id LIKE ?))")
        args.extend(["%" + search.lower() + "%", "%" + search + "%", "%" + search + "%"])
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    with connect("server") as db:
        total = db.execute("SELECT COUNT(*) AS n FROM llm_tool_requests r" + where, args).fetchone()["n"]
        rows = db.execute("SELECT r.id, r.userid, r.origin_software, r.status, r.created_at, r.updated_at, "
                          "r.question_text FROM llm_tool_requests r" + where +
                          " ORDER BY r.created_at DESC, r.id DESC LIMIT ? OFFSET ?", [*args, limit, offset]).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        with connect("server") as db:
            statuses = db.execute("SELECT status FROM llm_tool_calls WHERE request_id = ? "
                                  "AND collector_name <> 'list_collectors' ORDER BY started_at, id",
                                  (item["id"],)).fetchall()
        labels = {"COMPLETED": "succeeded", "FAILED": "failed", "REJECTED": "rejected",
                  "CANCELLED": "cancelled", "RUNNING": "running"}
        counts = {label: sum(r["status"] == key for r in statuses) for key, label in labels.items()}
        parts = [f"{count} {label}" for label, count in counts.items() if count]
        if item["status"] == "PENDING":
            parts.append("awaiting human confirmation")
        item["execution_summary"] = "; ".join(parts) + "." if parts else "No collector execution was recorded."
        attempts = [r["status"].removeprefix("EXPLANATION_") for r in statuses
                    if r["status"].startswith("EXPLANATION_")]
        item["explanation_status"] = attempts[-1] if attempts else None
        items.append(item)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def history_detail(request_id, user):
    with connect("server") as db:
        row = db.execute("SELECT id, userid, origin_software, status, created_at, updated_at, "
                         "question_text, response_text, settings_json FROM llm_tool_requests WHERE id = ?",
                         (request_id,)).fetchone()
        if not row or (user["role"] not in {"Administrator", "Power User"} and row["userid"] != user["userid"]):
            return None
        events = db.execute("SELECT * FROM llm_tool_calls WHERE request_id = ? ORDER BY started_at, id", (request_id,)).fetchall()
        invocations = db.execute("SELECT * FROM rca_llm_invocations WHERE request_id = ? ORDER BY invocation_number, id", (request_id,)).fetchall()
        views = db.execute("SELECT * FROM rca_evidence_views WHERE request_id = ? ORDER BY created_at, id", (request_id,)).fetchall()
    detail = _json_fields(row, ["settings_json"])
    detail["events"] = [_json_fields(r, ["arguments_json"]) for r in events]
    detail["invocations"] = [_json_fields(r, ["settings_json", "usage_json"]) for r in invocations]
    detail["evidence_views"] = [_json_fields(r, ["transformation_json"]) for r in views]
    detail["execution"] = execution_summary(request_id)
    return detail


def latest_execution_event(request_id):
    with connect("server") as db:
        row = db.execute("SELECT id FROM llm_tool_calls WHERE request_id = ? "
                         "AND collector_name <> 'list_collectors' AND status IN ('COMPLETED', 'FAILED') "
                         "ORDER BY started_at DESC, id DESC LIMIT 1", (request_id,)).fetchone()
    return row["id"] if row else None


def saved_execution_evidence(request_id, call_id):
    with connect("server") as db:
        row = db.execute("SELECT result_text FROM llm_tool_calls WHERE request_id = ? AND id = ?",
                         (request_id, call_id)).fetchone()
    return row["result_text"] if row else None
