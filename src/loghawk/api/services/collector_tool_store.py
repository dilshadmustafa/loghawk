"""Server-owned, single-use collector confirmations and AI execution audit."""

import json
import re
from hashlib import sha256
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from loghawk.database.timestamps import utc_now
from loghawk.database.database_layer import DatabaseIntegrityError, connect


def now() -> str:
    return utc_now()


class ExistingRequest(Exception):
    """The same authenticated submission already owns this tracking ID."""

    def __init__(self, request_id: str):
        super().__init__("This request has already been submitted.")
        self.request_id = request_id


class RequestConflict(ValueError):
    """A tracking ID cannot be reused for a different submission."""


def create(userid: str, origin_software: str = "console", *,
           request_id: str | None = None, submission_hash: str | None = None) -> str:
    if origin_software not in {"web", "console", "cli"}:
        raise ValueError("Unsupported origin software.")
    if request_id is not None and not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise ValueError("Request ID must contain 32 lowercase hexadecimal characters.")
    request_id = request_id or uuid4().hex
    timestamp = now()
    settings = {"submission_hash": submission_hash} if submission_hash else {}
    try:
        with connect("server") as db:
            # The primary key claims the ID before any model or collector execution.
            db.execute("""INSERT INTO llm_tool_requests
                       (id, userid, origin_software, status, state_json, settings_json,
                        expires_at, created_at, updated_at)
                       VALUES (?, ?, ?, 'PROCESSING', '{}', ?, ?, ?, ?)""",
                       (request_id, userid, origin_software, json.dumps(settings),
                        timestamp, timestamp, timestamp))
            db.execute("UPDATE llm_tool_requests SET state_json = '{}', status = 'EXPIRED' "
                       "WHERE status = 'PENDING' AND expires_at <= ?", (timestamp,))
    except DatabaseIntegrityError:
        # Read after the failed transaction has rolled back (also required by PostgreSQL).
        with connect("server") as db:
            row = db.execute("SELECT userid, origin_software, settings_json FROM llm_tool_requests "
                             "WHERE id = ?", (request_id,)).fetchone()
        if row is None:
            raise
        previous = json.loads(row["settings_json"] or "{}")
        if (submission_hash and row["userid"] == userid and row["origin_software"] == origin_software
                and previous.get("submission_hash") == submission_hash):
            raise ExistingRequest(request_id) from None
        raise RequestConflict("Request ID is already in use. Submit a new tracking ID.") from None
    return request_id


def save(request_id: str, state: dict, status: str) -> None:
    timestamp = now()
    expiry = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
    with connect("server") as db:
        db.execute("""UPDATE llm_tool_requests SET state_json = ?, status = ?, response_text = NULL,
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
           evidence: str, run_id: str | None = None, call_id: str | None = None, *, tool_type: str | None = None) -> str:
    """Arguments/evidence must already be redacted by the service."""
    tool_type = tool_type or ("collector" if target.get("kind") in {"builtin", "custom"} else {
        "list_collectors": "discovery", "request_settings": "control",
        "stop_without_execution": "control", "search_knowledge": "knowledge_search",
        "finish_investigation": "assessment", "investigation_progress": "control",
        "ai_explanation": "explanation",
    }.get(target.get("name"), "collector"))
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
                    status, arguments_json, result_text, run_id, started_at, completed_at, created_at, tool_type)
                   VALUES (?, ?, 'AI', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (call_id, request_id, origin_userid, request["origin_software"], target.get("name", "unknown"), str(target.get("version", "")),
                    target.get("artifact_id"), status, json.dumps(arguments), evidence, run_id,
                    timestamp, None if status in {"RUNNING", "EXPLANATION_RUNNING"} else timestamp, timestamp, tool_type))
    return call_id


def execution_summary(request_id: str) -> dict:
    """Summarize recorded outcomes, never model assertions or discovery success."""
    with connect("server") as db:
        request = db.execute("SELECT status, settings_json, state_json FROM llm_tool_requests WHERE id = ?", (request_id,)).fetchone()
        if request is None:
            raise ValueError("Tool request not found.")
        rows = db.execute(
            "SELECT id, collector_name, collector_version, status, run_id, result_text, arguments_json, tool_type "
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
        if bucket and row["tool_type"] == "collector":
            counts[bucket] += 1
        events.append({"tool_type": row["tool_type"], "call_id": row["id"], "collector": row["collector_name"], "version": row["collector_version"],
                       "arguments": json.loads(row["arguments_json"])
                           if status in {"RUNNING", "COMPLETED", "FAILED"} else None,
                       "status": status, "run_id": row["run_id"],
                       "evidence": row["result_text"] if status in {"COMPLETED", "FAILED"} else None,
                       "reason": row["result_text"] if status in {"REJECTED", "CANCELLED"} else None})
    try:
        request_settings = json.loads(request["settings_json"] or "{}")
    except (TypeError, ValueError):
        request_settings = {}
    evidence_redaction_enabled = request_settings.get("redact_evidence", True) is not False
    saved_state = json.loads(request["state_json"] or "{}")
    loop = saved_state.get("loop_progress") or saved_state.get("explanation", {}).get("context", {}).get("loop_progress")
    assessment = saved_state.get("assessment") or saved_state.get("explanation", {}).get("context", {}).get("assessment")
    waiting = request["status"] == "PENDING"
    parts = [f"{value} {key}" for key, value in counts.items() if value]
    if waiting:
        parts.append("awaiting human confirmation")
    if not parts:
        parts.append("No collector execution was recorded")
    return {"request_id": request_id, "request_status": request["status"],
            "discovery_calls": discovery_calls, **counts, "awaiting_confirmation": waiting,
            "evidence_redaction_enabled": evidence_redaction_enabled,
            "loop_progress": loop, "assessment": assessment,
            "summary": "; ".join(parts) + ".", "events": events}


def begin_explanation(request_id: str, userid: str, origin_software: str,
                      timeout: int, context: dict | None = None,
                      *, regenerate: bool = False) -> dict:
    """Atomically claim a text-only explanation; retain evidence and progress, never runnable state."""
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
            eligible = (
                previous and previous.get("status") == "COMPLETED"
                if regenerate
                else previous and (previous.get("status") == "FAILED" or expired)
            )
            if not eligible:
                raise ValueError("Explanation is not eligible for this action or another attempt is still running.")
            context = previous["context"]
        elif row["status"] != "PROCESSING" or previous:
            raise ValueError("This request has already entered its explanation stage.")
        explanation = {"status": "RUNNING", "attempt_id": uuid4().hex,
                       "attempts": (previous or {}).get("attempts", 0) + 1,
                       "stage": (
                           "explanation_regeneration" if regenerate
                           else "explanation_retry" if previous
                           else "initial_explanation"
                       ),
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


def complete_assessment(request_id: str, context: dict, response: str) -> None:
    """Keep the validated assessment as the final explanation without another model call."""
    explanation = {"status": "COMPLETED", "attempts": 0, "stage": "investigation_assessment",
                   "context": context, "response": response, "error": None}
    with connect("server") as db:
        result = db.execute(
            "UPDATE llm_tool_requests SET state_json = ?, status = 'COMPLETED', "
            "response_text = ?, updated_at = ? WHERE id = ? AND status = 'PROCESSING'",
            (json.dumps({"explanation": explanation}), response, now(), request_id),
        )
        if result.rowcount != 1:
            raise ValueError("This request has already entered its explanation stage.")
    record(request_id, {"name": "ai_explanation"}, {"stage": "investigation_assessment", "attempts": 0},
           "EXPLANATION_COMPLETED", response, tool_type="explanation")


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
            "UPDATE llm_tool_requests SET state_json = ?, status = ?, response_text = ?, updated_at = ? "
            "WHERE id = ? AND state_json = ?",
            (json.dumps(state), "FAILED" if error else "COMPLETED", response, now(), request_id, row["state_json"]),
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
            "can_regenerate": explanation["status"] == "COMPLETED",
            "attempts": explanation["attempts"]}


# History reads never expose runnable confirmation state or secret-bearing context.
def settings_snapshot() -> dict:
    from loghawk import config
    names = ("MODEL", "CONTEXT_LENGTH", "MAX_TOKENS", "MAX_DISCOVERY_CALLS",
             "MAX_EXECUTION_REQUESTS", "MAX_MODEL_TURNS", "MAX_RAG_SEARCHES", "TIMEOUT_SECONDS",
             "MODEL_CALL_TIMEOUT_SECONDS", "EXPLANATION_TIMEOUT_SECONDS", "MAX_RESULT_CHARS")
    return {"LH_LLM_TOOL_" + name: getattr(config, "LH_LLM_TOOL_" + name) for name in names}


def initialize_history(request_id: str, question: str, settings: dict) -> None:
    with connect("server") as db:
        row = db.execute("SELECT settings_json FROM llm_tool_requests WHERE id = ?", (request_id,)).fetchone()
        previous = json.loads(row["settings_json"] or "{}") if row else {}
        merged = {**previous, **settings}
        if "submission_hash" in previous:
            merged["submission_hash"] = previous["submission_hash"]
        db.execute("UPDATE llm_tool_requests SET question_text = ?, settings_json = ? WHERE id = ?",
                   (question, json.dumps(merged), request_id))


def update_progress(request_id: str, progress: dict) -> None:
    """Persist small, public activity metadata independently of runnable state."""
    for _ in range(5):
        with connect("server") as db:
            row = db.execute("SELECT settings_json FROM llm_tool_requests WHERE id = ?", (request_id,)).fetchone()
            if row is None:
                raise ValueError("Tool request not found.")
            settings = json.loads(row["settings_json"] or "{}")
            timestamp = now()
            settings["live_progress"] = {**settings.get("live_progress", {}), **progress,
                                         "updated_at": timestamp}
            result = db.execute(
                "UPDATE llm_tool_requests SET settings_json = ?, updated_at = ? "
                "WHERE id = ? AND COALESCE(settings_json, '{}') = ?",
                (json.dumps(settings), timestamp, request_id, row["settings_json"] or "{}"),
            )
            if result.rowcount == 1:
                return
    raise ValueError("Request progress changed concurrently; update was not saved.")


def save_response(request_id: str, response: str) -> None:
    with connect("server") as db:
        db.execute("UPDATE llm_tool_requests SET response_text = ?, updated_at = ? WHERE id = ?",
                   (response, now(), request_id))


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
        row = db.execute("SELECT settings_json FROM rca_llm_invocations WHERE id = ?", (invocation_id,)).fetchone()
        settings = json.loads(row["settings_json"] or "{}") if row else {}
        for key in ("response_diagnostics", "context_preparation"):
            if isinstance(metadata.get(key), dict):
                settings[key] = metadata[key]
        db.execute("""UPDATE rca_llm_invocations SET completed_at = ?, duration_ms = ?, status = ?,
            error_text = ?, returned_model = ?, usage_json = ?, settings_json = ? WHERE id = ?""",
            (now(), duration_ms, "FAILED" if error else "COMPLETED", error,
             metadata.get("returned_model"), json.dumps(metadata["usage"]) if metadata.get("usage") is not None else None,
             json.dumps(settings), invocation_id))


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


def _activity_summary(event: dict) -> str:
    """Summarize bounded evidence metadata; never return collector payload fragments."""
    status, kind = event["status"], event["tool_type"]
    if status == "AWAITING_CONFIRMATION":
        return "Awaiting confirmation."
    if status in {"RUNNING", "EXPLANATION_RUNNING"}:
        return {"explanation": "Waiting for model response.",
                "knowledge_search": "Searching indexed runbooks.",
                "discovery": "Discovering collectors."}.get(kind, "Collector running.")
    if status == "COMPLETED":
        try:
            value = json.loads(event.get("result_preview") or "{}")
        except (TypeError, ValueError):
            value = {}
        if kind == "knowledge_search" and isinstance(value, dict):
            chunks = value.get("chunks")
            if isinstance(chunks, list):
                return f"{len(chunks)} runbook excerpts retrieved."
        if kind == "discovery":
            collectors = value.get("collectors") if isinstance(value, dict) else value
            if isinstance(collectors, list):
                return f"{len(collectors)} eligible collectors found."
            return "Collector discovery completed."
        if kind == "collector":
            result = value.get("result", {}) if isinstance(value, dict) else {}
            data = result.get("data", {}) if isinstance(result, dict) else {}
            if isinstance(data, list):
                return f"Execution succeeded; {len(data)} records returned."
            if isinstance(data, dict):
                for key, noun in (("total_series", "series"), ("total_pods", "pods"),
                                  ("row_count", "rows"), ("record_count", "records")):
                    count = data.get(key)
                    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        return (f"Execution succeeded; no matching {noun} returned." if count == 0
                                else f"Execution succeeded; {count} {noun} returned.")
                # Only a complete, untruncated array can establish an empty result.
                if not data.get("truncated"):
                    for key, noun in (("pods", "pods"), ("items", "items"), ("records", "records"),
                                      ("rows", "rows"), ("result", "series")):
                        if isinstance(data.get(key), list):
                            count = len(data[key])
                            return (f"Execution succeeded; no matching {noun} returned." if count == 0
                                    else f"Execution succeeded; {count} {noun} returned.")
            return "Execution succeeded; evidence saved."
        return "Assessment completed." if kind == "assessment" else "Completed."
    return {"FAILED": "Execution failed; details are available in RCA History.",
            "REJECTED": "Request rejected; details are available in RCA History.",
            "CANCELLED": "Cancelled by the user.", "CONFIRMED": "Execution confirmed.",
            "EXPLANATION_COMPLETED": "Explanation completed.",
            "EXPLANATION_FAILED": "Explanation failed; saved evidence is available.",
            "REQUEST_SETTINGS": "Request settings saved.",
            "PARTIAL": "Partial assessment saved.", "NOT_EXECUTED": "No collector was executed."}.get(
                status, status.replace("_", " ").capitalize() + ".")


def progress_detail(request_id: str, userid: str, *, origin_software: str = "web") -> dict | None:
    """Owner-only live activity. Active requests never expose runnable state or full evidence."""
    with connect("server") as db:
        row = db.execute(
            "SELECT id, status, created_at, updated_at, expires_at, response_text, settings_json, "
            "CASE WHEN status IN ('PENDING', 'COMPLETED', 'FAILED') THEN state_json ELSE '{}' END AS state_json "
            "FROM llm_tool_requests WHERE id = ? AND userid = ? AND origin_software = ?",
            (request_id, userid, origin_software),
        ).fetchone()
        if row is None:
            return None
        events = db.execute(
            "SELECT id, collector_name, collector_version, tool_type, status, arguments_json, run_id, "
            "started_at, completed_at, SUBSTR(result_text, 1, 6000) AS result_preview "
            "FROM llm_tool_calls WHERE request_id = ? ORDER BY started_at, id", (request_id,),
        ).fetchall()
        invocations = db.execute(
            "SELECT id, stage, invocation_number, status, started_at, completed_at, duration_ms, "
            "error_text, settings_json FROM rca_llm_invocations WHERE request_id = ? "
            "ORDER BY invocation_number, id", (request_id,),
        ).fetchall()
        investigation = db.execute(
            "SELECT investigation_id FROM rca_investigation_requests WHERE request_id = ?", (request_id,),
        ).fetchone()
    settings = json.loads(row["settings_json"] or "{}")
    settings.pop("submission_hash", None)
    state = json.loads(row["state_json"] or "{}")
    timestamp = now()
    status = "EXPIRED" if row["status"] == "PENDING" and row["expires_at"] <= timestamp else row["status"]
    public_events = []
    for item in events:
        event = _json_fields(item, ["arguments_json"])
        event["summary"] = _activity_summary(event)
        event.pop("result_preview", None)
        public_events.append(event)
    public_invocations = []
    for item in invocations:
        invocation = _json_fields(item, ["settings_json"])
        invocation["settings"] = {key: value for key, value in (invocation.get("settings") or {}).items()
                                  if key in {"effective_timeout_seconds", "context_preparation"}}
        public_invocations.append(invocation)
    revision = sha256(json.dumps(
        [status, row["updated_at"], [(e["id"], e["status"], e["completed_at"]) for e in events],
         [(i["id"], i["status"], i["completed_at"]) for i in invocations]],
        separators=(",", ":"),
    ).encode()).hexdigest()
    result = {"id": request_id, "request_id": request_id, "status": status,
              "created_at": row["created_at"], "updated_at": row["updated_at"], "server_time": timestamp,
              "revision": revision, "settings": settings, "live_progress": settings.get("live_progress", {}),
              "events": public_events, "invocations": public_invocations}
    if investigation:
        result["investigation_id"] = investigation["investigation_id"]
    if status in {"PENDING", "COMPLETED", "FAILED", "CANCELLED", "EXPIRED"}:
        explanation = state.get("explanation", {})
        response = row["response_text"] or explanation.get("response") or ""
        if status == "PENDING":
            from loghawk.api.services.collector_tool_service import redact
            pending = state.get("pending") or {}
            target = pending.get("target") or {}
            result["pending"] = {"collector": target.get("name", ""), "version": target.get("version", ""),
                                 "review_arguments": state.get("review_arguments", True),
                                 "arguments": redact(pending.get("arguments", {})),
                                 "confirmation_id": pending.get("confirmation_id", "")}
            response = ("Review the input arguments, then confirm execution or cancel."
                        if state.get("review_arguments", True)
                        else "AI-assisted policy requires execution confirmation.")
        elif status == "EXPIRED":
            response = "Collector confirmation expired. Start a new request to continue."
        execution = execution_summary(request_id)
        execution["request_status"] = status
        execution["awaiting_confirmation"] = status == "PENDING"
        if status == "EXPIRED":
            execution["summary"] = execution["summary"].replace("awaiting human confirmation", "confirmation expired")
        answer = {"request_id": request_id, "status": status, "response": response, "execution": execution}
        if result.get("pending"):
            answer["pending"] = result["pending"]
        if investigation:
            answer["investigation_id"] = investigation["investigation_id"]
        explanation_details = explanation_info(request_id)
        if explanation_details:
            answer["explanation"] = explanation_details
        # save() clears the old response before the new reply is persisted. Do not
        # let a poll complete an action with empty or previous confirmation text.
        if status not in {"COMPLETED", "FAILED", "CANCELLED"} or response:
            result["answer"] = answer
    with connect("server") as db:
        latest = db.execute(
            "SELECT status, updated_at FROM llm_tool_requests "
            "WHERE id = ? AND userid = ? AND origin_software = ?",
            (request_id, userid, origin_software),
        ).fetchone()
    if latest is None:
        return None
    if latest["status"] != row["status"] or latest["updated_at"] != row["updated_at"]:
        # Confirmation/retry may have changed the request while its timeline was
        # being read. The next poll obtains a fresh projection; no work is run here.
        result.pop("answer", None)
        result.pop("pending", None)
        result["snapshot_stale"] = True
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
                          "r.question_text, r.settings_json FROM llm_tool_requests r" + where +
                          " ORDER BY r.created_at DESC, r.id DESC LIMIT ? OFFSET ?", [*args, limit, offset]).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        settings = json.loads(item.pop("settings_json") or "{}")
        item["mode"] = settings.get("mode")
        item["effective_limits"] = settings.get("effective_limits")
        with connect("server") as db:
            statuses = db.execute("SELECT status, tool_type FROM llm_tool_calls WHERE request_id = ? "
                                  "AND collector_name <> 'list_collectors' ORDER BY started_at, id",
                                  (item["id"],)).fetchall()
        labels = {"COMPLETED": "succeeded", "FAILED": "failed", "REJECTED": "rejected",
                  "CANCELLED": "cancelled", "RUNNING": "running"}
        counts = {label: sum(r["status"] == key and r["tool_type"] == "collector" for r in statuses) for key, label in labels.items()}
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
    if isinstance(detail.get("settings"), dict):
        detail["settings"].pop("submission_hash", None)
    detail["events"] = [_json_fields(r, ["arguments_json"]) for r in events]
    detail["invocations"] = [_json_fields(r, ["settings_json", "usage_json"]) for r in invocations]
    detail["evidence_views"] = [_json_fields(r, ["transformation_json"]) for r in views]
    from loghawk.api.services.investigation_context import saved
    detail["investigation"] = saved(request_id)
    detail["execution"] = execution_summary(request_id)
    detail["explanation"] = explanation_info(request_id)
    if detail["explanation"]:
        own_web_request = (
            row["userid"] == user["userid"]
            and row["origin_software"] == "web"
        )
        detail["explanation"]["can_retry"] &= own_web_request
        detail["explanation"]["can_regenerate"] &= own_web_request
    return detail


def latest_execution_event(request_id):
    with connect("server") as db:
        row = db.execute("SELECT id FROM llm_tool_calls WHERE request_id = ? "
                         "AND tool_type = 'collector' AND status IN ('COMPLETED', 'FAILED') "
                         "ORDER BY started_at DESC, id DESC LIMIT 1", (request_id,)).fetchone()
    return row["id"] if row else None


def saved_execution_evidence(request_id, call_id):
    with connect("server") as db:
        row = db.execute("SELECT result_text FROM llm_tool_calls WHERE request_id = ? AND id = ?",
                         (request_id, call_id)).fetchone()
    return row["result_text"] if row else None


def assessment_references(request_id, snapshot):
    """Resolve references against saved request evidence, including immutable imports."""
    from loghawk.api.services.investigation_assessment import context_references
    refs = context_references(snapshot)
    with connect("server") as db:
        rows = db.execute(
            "SELECT id, tool_type, result_text FROM llm_tool_calls "
            "WHERE request_id = ? AND status = 'COMPLETED' "
            "AND tool_type IN ('collector', 'knowledge_search')", (request_id,),
        ).fetchall()
    for row in rows:
        if row["tool_type"] == "collector":
            refs["collector:" + row["id"]] = "collector result"
        else:
            for index, chunk in enumerate(json.loads(row["result_text"]).get("chunks", []), 1):
                expected = f"rag:{row['id']}:{index}"
                if chunk.get("evidence_ref") == expected:
                    refs[expected] = "background knowledge"
    return refs
