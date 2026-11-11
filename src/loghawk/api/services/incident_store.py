"""Standalone incidents and evidence attachments owned by authenticated users."""
import json
from datetime import date, timedelta
from uuid import uuid4

from loghawk.api.services import investigation_store as evidence_store
from loghawk.database.database_layer import connect
from loghawk.database.timestamps import utc_now

MANAGERS = {"Administrator", "Power User"}


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


def _details(row):
    item = dict(row)
    item["primary_details"] = json.loads(item.pop("primary_details_json"))
    return item


def _incident(db, incident_id, user, *, write=False, include_deleted=False):
    row = db.execute("SELECT * FROM rca_incidents WHERE id=?", (incident_id,)).fetchone()
    if not row or (row["userid"] != user["userid"] and user.get("role") not in MANAGERS):
        raise PermissionError("Incident not found or access denied.")
    if row["deleted_at"] and not include_deleted:
        raise ValueError("This incident has been permanently deleted or is no longer available.")
    if write and row["userid"] != user["userid"]:
        raise PermissionError("Only the incident owner can modify it.")
    return row


def create(user, source_incident_id, summary, start_time, end_time, severity, primary_service, services, details):
    source_incident_id = str(source_incident_id).strip()
    summary = str(summary).strip()
    start_time, end_time = str(start_time).strip(), str(end_time).strip()
    if not source_incident_id or not start_time or not end_time:
        raise ValueError("Incident ID, start time, and end time are required.")
    if summary:
        details = dict(details or {})
        details["summary"] = summary
    payload = {**(details or {}), "incident_id": source_incident_id,
               "start_time": start_time, "end_time": end_time,
               "max_severity": str(severity or "").strip(),
               "primary_service": str(primary_service or "").strip(),
               "services": services or []}
    key, now = uuid4().hex, utc_now()
    with connect("server") as db:
        db.execute(
            "INSERT INTO rca_incidents (id,userid,source_incident_id,primary_details_json,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (key, user["userid"], source_incident_id, _encode(payload), now, now),
        )
    return detail(key, user)


def listing(user, offset=0, limit=50, query="", severity=None, service=None, created_from=None, created_to=None):
    clauses, values = ["deleted_at IS NULL"], []
    if user.get("role") not in MANAGERS:
        clauses.append("userid=?")
        values.append(user["userid"])
    term = str(query or "").strip().lower()
    if term:
        clauses.append("(INSTR(LOWER(source_incident_id),?)>0 OR INSTR(LOWER(primary_details_json),?)>0 OR INSTR(LOWER(userid),?)>0)")
        values.extend((term, term, term))
    if severity:
        clauses.append("INSTR(LOWER(primary_details_json),?)>0")
        values.append(str(severity).lower())
    if service:
        clauses.append("INSTR(LOWER(primary_details_json),?)>0")
        values.append(str(service).lower())
    if created_from:
        clauses.append("created_at>=?")
        values.append(f"{created_from.isoformat()}T00:00:00")
    if created_to:
        clauses.append("created_at<?")
        values.append(f"{(created_to + timedelta(days=1)).isoformat()}T00:00:00")
    where = " WHERE " + " AND ".join(clauses)
    with connect("server") as db:
        rows = db.execute(
            f"SELECT i.*, (SELECT COUNT(*) FROM rca_incident_evidence e WHERE e.incident_id=i.id) AS evidence_count "
            f"FROM rca_incidents i{where} ORDER BY i.created_at DESC,i.id LIMIT ? OFFSET ?",
            (*values, limit, offset),
        ).fetchall()
    result = []
    for row in rows:
        item = _details(row)
        item["can_delete"] = row["userid"] == user["userid"] or user.get("role") == "Administrator"
        result.append(item)
    return result

def detail(incident_id, user):
    with connect("server") as db:
        row = _incident(db, incident_id, user)
        result = _details(row)
        result["can_modify"] = row["userid"] == user["userid"]
        result["can_delete"] = row["userid"] == user["userid"] or user.get("role") == "Administrator"
        result["investigations"] = [dict(item) for item in db.execute(
            "SELECT i.id,i.problem,i.status,i.created_at FROM rca_investigations i "
            "JOIN rca_investigation_incidents l ON l.investigation_id=i.id "
            "WHERE l.incident_id=? ORDER BY i.created_at DESC", (incident_id,)
        ).fetchall()]
        sources = db.execute(
            "SELECT s.*,e.selection_json AS attachment_selection_json,e.created_at AS attached_at "
            "FROM rca_incident_evidence e JOIN rca_evidence_sources s ON s.id=e.source_id "
            "WHERE e.incident_id=? ORDER BY e.created_at,s.name", (incident_id,)
        ).fetchall()
        result["sources"] = []
        for source in sources:
            item = dict(source)
            item["attachment_selection"] = json.loads(item.pop("attachment_selection_json"))
            result["sources"].append(item)
    return result


def permanently_delete(incident_id, user):
    """Permanently remove an incident and its dependent links, retaining evidence and RCA snapshots."""
    with connect("server") as db:
        row = _incident(db, incident_id, user, include_deleted=True)
        if row["userid"] != user["userid"] and user.get("role") != "Administrator":
            raise PermissionError("Only the incident owner or an Administrator can permanently delete it.")
        db.execute("DELETE FROM rca_investigation_incidents WHERE incident_id=?", (incident_id,))
        db.execute("DELETE FROM rca_incident_evidence WHERE incident_id=?", (incident_id,))
        db.execute("DELETE FROM rca_incidents WHERE id=?", (incident_id,))
    return {"incident_id": incident_id, "permanently_deleted": True}


def attach(incident_id, user, source_id, selection=None):
    with connect("server") as db:
        incident = _incident(db, incident_id, user, write=True)
        source = evidence_store.owned(db, "rca_evidence_sources", source_id, incident["userid"])
        selected = evidence_store.select_records(db, source, selection)
        existing = db.execute("SELECT id FROM rca_incident_evidence WHERE incident_id=? AND source_id=?",
                              (incident_id, source_id)).fetchone()
        if existing:
            db.execute("UPDATE rca_incident_evidence SET selection_json=? WHERE id=?",
                       (_encode(selected), existing["id"]))
        else:
            db.execute("INSERT INTO rca_incident_evidence (id,incident_id,source_id,selection_json,created_at) VALUES (?,?,?,?,?)",
                       (uuid4().hex, incident_id, source_id, _encode(selected), utc_now()))
    return detail(incident_id, user)


def add_collector_evidence(incident_id, user, tool_call_id):
    """Attach an owned, saved collector execution to an incident without rerunning it."""
    tool_call_id = str(tool_call_id or "").strip()
    if not tool_call_id:
        raise ValueError("Provide a collector tool-call event ID.")

    with connect("server") as db:
        incident = _incident(db, incident_id, user, write=True)
        event = db.execute(
            "SELECT c.* FROM llm_tool_calls c JOIN llm_tool_requests r ON c.request_id=r.id "
            "WHERE c.id=? AND r.userid=?",
            (tool_call_id, incident["userid"]),
        ).fetchone()
        if not event or event["status"] not in {"COMPLETED", "FAILED"} or not event["run_id"]:
            raise ValueError("Select an owned, completed collector execution with a run ID.")

        source = db.execute(
            "SELECT * FROM rca_evidence_sources WHERE userid=? AND investigation_id IS NULL AND tool_call_id=?",
            (incident["userid"], tool_call_id),
        ).fetchone()
        now = utc_now()
        if source:
            source_id = source["id"]
            record = db.execute(
                "SELECT id FROM rca_evidence_records WHERE source_id=? AND record_number=1",
                (source_id,),
            ).fetchone()
            if not record:
                result = {
                    "collector": event["collector_name"],
                    "version": event["collector_version"],
                    "status": event["status"],
                    "run_id": event["run_id"],
                    "arguments": json.loads(event["arguments_json"] or "{}"),
                    "result_text": event["result_text"] or "",
                }
                db.execute(
                    "INSERT INTO rca_evidence_records (id,source_id,record_number,source_line_number,original_record_json,created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (uuid4().hex, source_id, 1, None, _encode(result), now),
                )
                db.execute(
                    "UPDATE rca_evidence_sources SET status='READY',format='collector',record_count=1,updated_at=? WHERE id=?",
                    (now, source_id),
                )
        else:
            source_id = uuid4().hex
            result = {
                "collector": event["collector_name"],
                "version": event["collector_version"],
                "status": event["status"],
                "run_id": event["run_id"],
                "arguments": json.loads(event["arguments_json"] or "{}"),
                "result_text": event["result_text"] or "",
            }
            db.execute(
                "INSERT INTO rca_evidence_sources "
                "(id,userid,investigation_id,name,original_uri,snapshot_uri,sha256,size_bytes,evidence_type,format,status,error,mapping_json,record_path,record_count,tool_call_id,created_at,updated_at) "
                "VALUES (?,?,NULL,?,?,NULL,NULL,?,'other','collector','READY',NULL,'{}',NULL,1,?,?,?)",
                (source_id, incident["userid"], event["collector_name"],
                 f"collector://{tool_call_id}", len((event["result_text"] or "").encode("utf-8")),
                 tool_call_id, now, now),
            )
            db.execute(
                "INSERT INTO rca_evidence_records (id,source_id,record_number,source_line_number,original_record_json,created_at) "
                "VALUES (?,?,?,?,?,?)",
                (uuid4().hex, source_id, 1, None, _encode(result), now),
            )

        linked = db.execute(
            "SELECT id FROM rca_incident_evidence WHERE incident_id=? AND source_id=?",
            (incident_id, source_id),
        ).fetchone()
        if not linked:
            db.execute(
                "INSERT INTO rca_incident_evidence (id,incident_id,source_id,selection_json,created_at) VALUES (?,?,?,?,?)",
                (uuid4().hex, incident_id, source_id, _encode([1]), now),
            )
    return detail(incident_id, user)


def unlink(incident_id, user, source_id):
    with connect("server") as db:
        _incident(db, incident_id, user, write=True)
        row = db.execute("SELECT id,selection_json FROM rca_incident_evidence WHERE incident_id=? AND source_id=?",
                         (incident_id, source_id)).fetchone()
        if not row:
            raise ValueError("This evidence is not attached to the incident.")
        db.execute("DELETE FROM rca_incident_evidence WHERE id=?", (row["id"],))
        selection = json.loads(row["selection_json"])
    return {"incident_id": incident_id, "source_id": source_id, "records": selection}


def preview(incident_id, source_id, user, offset=0, limit=100):
    with connect("server") as db:
        incident = _incident(db, incident_id, user)
        source = evidence_store.owned(db, "rca_evidence_sources", source_id, incident["userid"])
        linked = db.execute("SELECT id FROM rca_incident_evidence WHERE incident_id=? AND source_id=?",
                            (incident_id, source_id)).fetchone()
        if not linked:
            raise PermissionError("Evidence source is not imported for this incident.")
        rows = db.execute("SELECT * FROM rca_evidence_records WHERE source_id=? ORDER BY record_number LIMIT ? OFFSET ?",
                          (source_id, limit, offset)).fetchall()
        result = dict(source)
        result["records"] = [evidence_store.decoded(row) for row in rows]
        result["next_offset"] = offset + len(rows) if offset + len(rows) < source["record_count"] else None
        return result


def list_import_sources(user, limit=100):
    """List this user's standalone Stage C imports for the Create Incident workflow."""
    with connect("server") as db:
        rows = db.execute(
            "SELECT * FROM rca_evidence_sources WHERE userid=? AND investigation_id IS NULL "
            "AND evidence_type='stage-c' ORDER BY created_at DESC,id LIMIT ?",
            (user["userid"], limit),
        ).fetchall()
    return [evidence_store.decoded(row) for row in rows]


def _stage_c_details(source, original):
    mapping = json.loads(source["mapping_json"] or "{}")
    fields = ("incident_id", "start_time", "end_time", "entity_id", "max_severity",
              "primary_service", "services", "duration_seconds", "anomaly_count",
              "service_count", "correlation_score")
    return {field: original[mapping.get(field, field)] for field in fields
            if mapping.get(field, field) in original}


def _missing_incident_fields(details):
    return [field for field in ("incident_id", "start_time", "end_time")
            if details.get(field) is None or not str(details[field]).strip()]


def preview_import_source(source_id, user, offset=0, limit=100):
    with connect("server") as db:
        source = evidence_store.owned(db, "rca_evidence_sources", source_id, user["userid"])
        if source["investigation_id"] is not None or source["evidence_type"] != "stage-c":
            raise ValueError("Choose a standalone Stage C incident file import.")
        result = evidence_store.decoded(source)
        rows = db.execute(
            "SELECT * FROM rca_evidence_records WHERE source_id=? ORDER BY record_number LIMIT ? OFFSET ?",
            (source_id, limit, offset),
        ).fetchall()
        result["records"] = []
        for row in rows:
            item = evidence_store.decoded(row)
            details = _stage_c_details(source, item["original_record"])
            item["missing_incident_fields"] = _missing_incident_fields(details)
            result["records"].append(item)
        result["next_offset"] = offset + len(rows) if offset + len(rows) < source["record_count"] else None
        return result


def create_from_import(user, source_id, record_numbers):
    """Create incidents from selected, validated Stage C source records."""
    if len(record_numbers) > 500:
        raise ValueError("Select no more than 500 incident records at a time.")
    if any(isinstance(number, bool) or not isinstance(number, int) or number < 1 for number in record_numbers):
        raise ValueError("Incident record numbers must be positive integers.")
    numbers = sorted(set(record_numbers))
    if not numbers:
        raise ValueError("Select at least one incident record.")
    with connect("server") as db:
        source = evidence_store.owned(db, "rca_evidence_sources", source_id, user["userid"])
        if source["investigation_id"] is not None or source["status"] != "READY" or source["evidence_type"] != "stage-c":
            raise ValueError("Choose a ready standalone Stage C incident file import.")
        slots = ",".join("?" for _ in numbers)
        rows = db.execute(
            f"SELECT * FROM rca_evidence_records WHERE source_id=? AND record_number IN ({slots})",
            (source_id, *numbers),
        ).fetchall()
        by_number = {row["record_number"]: row for row in rows}
        if set(by_number) != set(numbers):
            raise ValueError("The selection contains records that are not in this file. Refresh and select again.")
        validated = []
        problems = []
        for number in numbers:
            row = by_number[number]
            original = json.loads(row["original_record_json"])
            details = _stage_c_details(source, original)
            missing = _missing_incident_fields(details)
            if missing:
                problems.append(f"Record {number}: missing {', '.join(missing)}")
            else:
                details["source_record_id"] = row["id"]
                validated.append((number, row, details))
        if problems:
            raise ValueError("Cannot create incidents. Deselect incomplete records: " + "; ".join(problems[:20]))

        incident_ids = []
        for number, evidence_row, details in validated:
            existing = None
            candidates = db.execute(
                "SELECT id,primary_details_json FROM rca_incidents WHERE userid=? AND deleted_at IS NULL",
                (user["userid"],),
            ).fetchall()
            for candidate in candidates:
                if json.loads(candidate["primary_details_json"]).get("source_record_id") == evidence_row["id"]:
                    existing = candidate["id"]
                    break
            now = utc_now()
            if existing:
                incident_id = existing
            else:
                incident_id = uuid4().hex
                db.execute(
                    "INSERT INTO rca_incidents (id,userid,source_incident_id,primary_details_json,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                    (incident_id,user["userid"],str(details["incident_id"]),_encode(details),now,now),
                )
            linked = db.execute(
                "SELECT id FROM rca_incident_evidence WHERE incident_id=? AND source_id=?",
                (incident_id,source_id),
            ).fetchone()
            if linked:
                selected = json.loads(db.execute(
                    "SELECT selection_json FROM rca_incident_evidence WHERE id=?", (linked["id"],)
                ).fetchone()["selection_json"])
                if number not in selected:
                    selected.append(number)
                    db.execute("UPDATE rca_incident_evidence SET selection_json=? WHERE id=?",
                               (_encode(sorted(selected)),linked["id"]))
            else:
                db.execute(
                    "INSERT INTO rca_incident_evidence (id,incident_id,source_id,selection_json,created_at) VALUES (?,?,?,?,?)",
                    (uuid4().hex,incident_id,source_id,_encode([number]),now),
                )
            incident_ids.append(incident_id)
    return {"incident_ids": incident_ids, "created_or_linked": len(incident_ids)}


def authorize(incident_id, user, *, write=False):
    with connect("server") as db:
        _incident(db, incident_id, user, write=write)
