"""Server-owned investigations and shared evidence. No model execution here."""
import json
from datetime import date, timedelta
from uuid import uuid4
from loghawk.database.database_layer import connect
from loghawk.database.timestamps import utc_now

MANAGERS = {"Administrator", "Power User"}

def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)

def decoded(row):
    result = dict(row)
    for key in list(result):
        if key.endswith("_json"):
            result[key[:-5]] = json.loads(result.pop(key))
    return result

def require(db, investigation_id, user, *, write=False):
    row = db.execute("SELECT * FROM rca_investigations WHERE id = ?", (investigation_id,)).fetchone()
    if not row or (row["userid"] != user["userid"] and (write or user.get("role") not in MANAGERS)):
        raise PermissionError("Investigation not found or access denied.")
    if write and row["status"] != "OPEN":
        raise ValueError("Investigation is closed.")
    if write:
        # Serialize edits, closure and snapshot capture on the investigation row.
        db.execute("UPDATE rca_investigations SET updated_at = ? WHERE id = ?", (utc_now(), investigation_id))
        row = db.execute("SELECT * FROM rca_investigations WHERE id = ?", (investigation_id,)).fetchone()
        if row["status"] != "OPEN":
            raise ValueError("Investigation is closed.")
    return row

def owned(db, table, key, userid):
    if table not in {"rca_incidents", "rca_evidence_sources"}:
        raise ValueError("Unsupported evidence table")
    row = db.execute(f"SELECT * FROM {table} WHERE id = ? AND userid = ?", (key, userid)).fetchone()
    if not row:
        raise PermissionError("Incident or evidence not owned by this investigation's owner.")
    if table == "rca_incidents" and row["deleted_at"]:
        raise ValueError("This incident is deleted. Restore it before linking it to an investigation.")
    return row

def create(user, problem, reported_context):
    key, now = uuid4().hex, utc_now()
    with connect("server") as db:
        db.execute("INSERT INTO rca_investigations (id,userid,problem,reported_context_json,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                   (key,user["userid"],problem,encode(reported_context),"OPEN",now,now))
    return detail(key,user)

def listing(user, offset=0, limit=50, query="", status=None, created_from=None, created_to=None):
    clauses=[]
    values=[]
    if user.get("role") not in MANAGERS:
        clauses.append("userid=?")
        values.append(user["userid"])
    term=str(query or "").strip().lower()
    if term:
        clauses.append("(INSTR(LOWER(problem),?)>0 OR INSTR(LOWER(userid),?)>0)")
        values.extend((term,term))
    if status:
        clauses.append("status=?")
        values.append(status)
    if created_from:
        clauses.append("created_at>=?")
        values.append(f"{created_from.isoformat()}T00:00:00")
    if created_to:
        clauses.append("created_at<?")
        values.append(f"{(created_to+timedelta(days=1)).isoformat()}T00:00:00")
    where=" WHERE "+" AND ".join(clauses) if clauses else ""
    with connect("server") as db:
        rows=db.execute(f"SELECT * FROM rca_investigations{where} ORDER BY created_at DESC,id LIMIT ? OFFSET ?",(*values,limit,offset)).fetchall()
    return [decoded(r) for r in rows]

def detail(key,user):
    with connect("server") as db:
        row=require(db,key,user)
        result=decoded(row)
        result["can_modify"]=row["userid"]==user["userid"] and row["status"]=="OPEN"
        result["incidents"]=[decoded(r) for r in db.execute("SELECT i.* FROM rca_incidents i JOIN rca_investigation_incidents l ON i.id=l.incident_id WHERE l.investigation_id=? AND i.deleted_at IS NULL ORDER BY l.created_at",(key,)).fetchall()]
        result["sources"]=[decoded(r) for r in db.execute("SELECT DISTINCT s.* FROM rca_evidence_sources s WHERE s.investigation_id=? OR s.id IN (SELECT source_id FROM rca_investigation_evidence WHERE investigation_id=?) OR s.id IN (SELECT e.source_id FROM rca_incident_evidence e JOIN rca_investigation_incidents i ON i.incident_id=e.incident_id WHERE i.investigation_id=?) OR s.id IN (SELECT e.source_id FROM rca_finding_evidence e JOIN rca_investigation_findings f ON f.id=e.finding_id WHERE f.investigation_id=?) ORDER BY s.created_at",(key,key,key,key)).fetchall()]
        result["sources"] = [source for source in result["sources"] if source["status"] not in {"DISMISSED", "REMOVED"}]
        result["evidence_links"]=[decoded(r) for r in db.execute("SELECT * FROM rca_investigation_evidence WHERE investigation_id=?",(key,)).fetchall()]
        result["incident_evidence"]=[decoded(r) for r in db.execute("SELECT e.* FROM rca_incident_evidence e JOIN rca_investigation_incidents i ON i.incident_id=e.incident_id JOIN rca_incidents incident ON incident.id=e.incident_id WHERE i.investigation_id=? AND incident.deleted_at IS NULL",(key,)).fetchall()]
        result["findings"]=[decoded(r) for r in db.execute("SELECT * FROM rca_investigation_findings WHERE investigation_id=? ORDER BY created_at,id",(key,)).fetchall()]
        for finding in result["findings"]:
            finding["evidence"]=[decoded(r) for r in db.execute("SELECT * FROM rca_finding_evidence WHERE finding_id=?",(finding["id"],)).fetchall()]
        result["requests"]=[dict(r) for r in db.execute("SELECT l.id,l.request_id,l.created_at,r.status,r.origin_software FROM rca_investigation_requests l LEFT JOIN llm_tool_requests r ON r.id=l.request_id WHERE l.investigation_id=? ORDER BY l.created_at DESC",(key,)).fetchall()]
        for request in result["requests"]:
            request["evidence_removed"]=request["status"] is None
    return result

def source_access(db,key,source_id,user,write=False):
    parent=require(db,key,user,write=write)
    source=owned(db,"rca_evidence_sources",source_id,parent["userid"])
    if source["investigation_id"] != key:
        linked=db.execute("SELECT id FROM rca_investigation_evidence WHERE investigation_id=? AND source_id=?",(key,source_id)).fetchone()
        indirect=db.execute("SELECT e.id FROM rca_incident_evidence e JOIN rca_investigation_incidents i ON i.incident_id=e.incident_id WHERE i.investigation_id=? AND e.source_id=?",(key,source_id)).fetchone()
        finding=db.execute("SELECT e.id FROM rca_finding_evidence e JOIN rca_investigation_findings f ON f.id=e.finding_id WHERE f.investigation_id=? AND e.source_id=?",(key,source_id)).fetchone()
        if not linked and not indirect and not finding and not write:
            raise PermissionError("Source is not attached to this investigation.")
    return source

def preview(key,source_id,user,offset=0,limit=100):
    with connect("server") as db:
        source=source_access(db,key,source_id,user)
        rows=db.execute("SELECT * FROM rca_evidence_records WHERE source_id=? ORDER BY record_number LIMIT ? OFFSET ?",(source_id,limit,offset)).fetchall()
        result=decoded(source)
        result["records"]=[decoded(r) for r in rows]
        result["next_offset"]=offset+len(rows) if offset+len(rows)<source["record_count"] else None
        return result

def select_records(db,source,selection):
    if source["status"] != "READY": raise ValueError("Only successfully parsed evidence can be attached.")
    if selection is None: return None
    numbers=sorted(set(selection))
    if not numbers: raise ValueError("Select at least one record, or use all records.")
    if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in numbers): raise ValueError("Invalid record number.")
    existing={r["record_number"] for r in db.execute("SELECT record_number FROM rca_evidence_records WHERE source_id=?",(source["id"],)).fetchall()}
    if not set(numbers)<=existing: raise ValueError("Selection contains unknown record numbers.")
    return numbers

def attach(key,user,source_id,selection=None,incident_id=None):
    with connect("server") as db:
        parent=require(db,key,user,write=True)
        source=owned(db,"rca_evidence_sources",source_id,parent["userid"])
        selected=select_records(db,source,selection)
        if incident_id:
            owned(db,"rca_incidents",incident_id,parent["userid"])
            if not db.execute("SELECT id FROM rca_investigation_incidents WHERE investigation_id=? AND incident_id=?",(key,incident_id)).fetchone(): raise ValueError("Attach the incident to this investigation first.")
            table,column,target="rca_incident_evidence","incident_id",incident_id
        else: table,column,target="rca_investigation_evidence","investigation_id",key
        payload=encode(selected)
        existing=db.execute(f"SELECT id FROM {table} WHERE {column}=? AND source_id=?",(target,source_id)).fetchone()
        if existing:
            # Selection changes affect future requests only; saved snapshots are immutable.
            db.execute(f"UPDATE {table} SET selection_json=? WHERE {column}=? AND source_id=?",(payload,target,source_id))
        else:
            db.execute(f"INSERT INTO {table} (id,{column},source_id,selection_json,created_at) VALUES (?,?,?,?,?)",(uuid4().hex,target,source_id,payload,utc_now()))
    return detail(key,user)

def add_incident(key,user,source_id,record_number):
    with connect("server") as db:
        parent=require(db,key,user,write=True)
        source=owned(db,"rca_evidence_sources",source_id,parent["userid"])
        select_records(db,source,[record_number])
        row=db.execute("SELECT * FROM rca_evidence_records WHERE source_id=? AND record_number=?",(source_id,record_number)).fetchone()
        original=json.loads(row["original_record_json"]); mapping=json.loads(source["mapping_json"])
        fields=("incident_id","start_time","end_time","entity_id","max_severity","primary_service","services","duration_seconds","anomaly_count","service_count","correlation_score")
        details={field:original[mapping.get(field,field)] for field in fields if mapping.get(field,field) in original}
        details["source_record_id"]=row["id"]
        # Repeated import of the exact source record reuses its incident. Matching
        # external incident IDs in different sources are not automatically merged.
        existing=None
        for candidate in db.execute("SELECT * FROM rca_incidents WHERE userid=? AND deleted_at IS NULL",(parent["userid"],)).fetchall():
            if json.loads(candidate["primary_details_json"]).get("source_record_id")==row["id"]:
                existing=candidate["id"];break
        if existing:
            if not db.execute("SELECT id FROM rca_investigation_incidents WHERE investigation_id=? AND incident_id=?",(key,existing)).fetchone():
                db.execute("INSERT INTO rca_investigation_incidents (id,investigation_id,incident_id,created_at) VALUES (?,?,?,?)",(uuid4().hex,key,existing,utc_now()))
            incident=existing
        else:
            incident,now=uuid4().hex,utc_now()
            db.execute("INSERT INTO rca_incidents (id,userid,source_incident_id,primary_details_json,created_at,updated_at) VALUES (?,?,?,?,?,?)",(incident,parent["userid"],str(details.get("incident_id") or ""),encode(details),now,now))
            db.execute("INSERT INTO rca_investigation_incidents (id,investigation_id,incident_id,created_at) VALUES (?,?,?,?)",(uuid4().hex,key,incident,now))
            db.execute("INSERT INTO rca_incident_evidence (id,incident_id,source_id,selection_json,created_at) VALUES (?,?,?,?,?)",(uuid4().hex,incident,source_id,encode([record_number]),now))
    return detail(key,user)

def add_incidents(key, user, source_id, record_numbers):
    """Atomically create/link a batch after validating every Stage C row."""
    if len(record_numbers) > 500:
        raise ValueError("Select no more than 500 incident records at a time.")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in record_numbers):
        raise ValueError("Incident record numbers must be positive integers.")
    numbers = sorted(set(record_numbers))
    if not numbers:
        raise ValueError("Select at least one incident record.")
    with connect("server") as db:
        parent = require(db, key, user, write=True)
        source = owned(db, "rca_evidence_sources", source_id, parent["userid"])
        if source["investigation_id"] != key or source["status"] != "READY" or source["evidence_type"] != "stage-c":
            raise ValueError("Choose a ready Stage C incident source imported into this investigation.")
        slots = ",".join("?" for _ in numbers)
        rows = db.execute(f"SELECT * FROM rca_evidence_records WHERE source_id=? AND record_number IN ({slots})", (source_id, *numbers)).fetchall()
        by_number = {row["record_number"]: row for row in rows}
        if set(by_number) != set(numbers):
            raise ValueError("The selection contains records that are not in this source. Refresh and select again.")
        mapping = json.loads(source["mapping_json"])
        validated = []
        problems = []
        fields = ("incident_id", "start_time", "end_time", "entity_id", "max_severity", "primary_service", "services", "duration_seconds", "anomaly_count", "service_count", "correlation_score")
        for number in numbers:
            row = by_number[number]
            original = json.loads(row["original_record_json"])
            details = {field: original[mapping.get(field, field)] for field in fields if mapping.get(field, field) in original}
            missing = [field for field in ("incident_id", "start_time", "end_time") if details.get(field) is None or not str(details[field]).strip()]
            if missing:
                problems.append(f"Record {number}: missing {', '.join(missing)}")
            else:
                details["source_record_id"] = row["id"]
                validated.append((number, row, details))
        if problems:
            raise ValueError("Cannot create incidents. Fix or deselect incomplete records: " + "; ".join(problems[:20]))
        incident_ids = []
        for number, evidence_row, details in validated:
            existing = None
            candidates = db.execute("SELECT id,primary_details_json FROM rca_incidents WHERE userid=? AND deleted_at IS NULL", (parent["userid"],)).fetchall()
            for candidate in candidates:
                if json.loads(candidate["primary_details_json"]).get("source_record_id") == evidence_row["id"]:
                    existing = candidate["id"]
                    break
            now = utc_now()
            if existing:
                if not db.execute("SELECT id FROM rca_investigation_incidents WHERE investigation_id=? AND incident_id=?", (key, existing)).fetchone():
                    db.execute("INSERT INTO rca_investigation_incidents (id,investigation_id,incident_id,created_at) VALUES (?,?,?,?)", (uuid4().hex, key, existing, now))
                incident_id = existing
            else:
                incident_id = uuid4().hex
                db.execute("INSERT INTO rca_incidents (id,userid,source_incident_id,primary_details_json,created_at,updated_at) VALUES (?,?,?,?,?,?)", (incident_id, parent["userid"], str(details["incident_id"]), encode(details), now, now))
                db.execute("INSERT INTO rca_investigation_incidents (id,investigation_id,incident_id,created_at) VALUES (?,?,?,?)", (uuid4().hex, key, incident_id, now))
                db.execute("INSERT INTO rca_incident_evidence (id,incident_id,source_id,selection_json,created_at) VALUES (?,?,?,?,?)", (uuid4().hex, incident_id, source_id, encode([number]), now))
            incident_ids.append(incident_id)
    return {"incident_ids": incident_ids, "created_or_linked": len(incident_ids)}


def link_incident(key,user,incident_id):
    with connect("server") as db:
        parent=require(db,key,user,write=True);owned(db,"rca_incidents",incident_id,parent["userid"])
        if not db.execute("SELECT id FROM rca_investigation_incidents WHERE investigation_id=? AND incident_id=?",(key,incident_id)).fetchone():
            db.execute("INSERT INTO rca_investigation_incidents (id,investigation_id,incident_id,created_at) VALUES (?,?,?,?)",(uuid4().hex,key,incident_id,utc_now()))
    return detail(key,user)

def set_source_type(key, user, source_id, evidence_type):
    if evidence_type not in {"stage-a", "stage-b", "stage-c", "raw-logs", "other"}:
        raise ValueError("Invalid evidence type.")
    with connect("server") as db:
        require(db, key, user, write=True)
        source = source_access(db, key, source_id, user)
        if source["status"] not in {"READY", "FAILED"}:
            raise ValueError("Only ready or failed sources can be reclassified.")
        previous = source["evidence_type"]
        db.execute(
            "UPDATE rca_evidence_sources SET evidence_type=?,updated_at=? WHERE id=?",
            (evidence_type, utc_now(), source_id),
        )
    return {"source_id": source_id, "previous_type": previous, "evidence_type": evidence_type}


def _attachment_target(db, key, user, source_id, incident_id):
    parent = require(db, key, user, write=True)
    source = owned(db, "rca_evidence_sources", source_id, parent["userid"])
    if incident_id:
        owned(db, "rca_incidents", incident_id, parent["userid"])
        if not db.execute(
            "SELECT id FROM rca_investigation_incidents WHERE investigation_id=? AND incident_id=?",
            (key, incident_id),
        ).fetchone():
            raise ValueError("Link the incident to this investigation first.")
        return source, "rca_incident_evidence", "incident_id", incident_id
    return source, "rca_investigation_evidence", "investigation_id", key


def unlink_evidence(key, user, source_id, incident_id=None):
    with connect("server") as db:
        _, table, column, target = _attachment_target(db, key, user, source_id, incident_id)
        link = db.execute(
            f"SELECT * FROM {table} WHERE {column}=? AND source_id=?", (target, source_id)
        ).fetchone()
        if not link:
            raise ValueError("This attachment no longer exists. Refresh the investigation.")
        selection = json.loads(link["selection_json"])
        db.execute(f"DELETE FROM {table} WHERE id=?", (link["id"],))
    return {"source_id": source_id, "incident_id": incident_id, "records": selection}


def restore_evidence(key, user, source_id, selection, incident_id=None):
    with connect("server") as db:
        source, table, column, target = _attachment_target(db, key, user, source_id, incident_id)
        selected = select_records(db, source, selection)
        if db.execute(
            f"SELECT id FROM {table} WHERE {column}=? AND source_id=?", (target, source_id)
        ).fetchone():
            raise ValueError("An attachment already exists for this target. Undo will not overwrite its record selection.")
        db.execute(
            f"INSERT INTO {table} (id,{column},source_id,selection_json,created_at) VALUES (?,?,?,?,?)",
            (uuid4().hex, target, source_id, encode(selected), utc_now()),
        )
    return {"source_id": source_id, "incident_id": incident_id, "records": selected}


def unlink_incident(key, user, incident_id):
    with connect("server") as db:
        parent = require(db, key, user, write=True)
        owned(db, "rca_incidents", incident_id, parent["userid"])
        link = db.execute(
            "SELECT id FROM rca_investigation_incidents WHERE investigation_id=? AND incident_id=?",
            (key, incident_id),
        ).fetchone()
        if not link:
            raise ValueError("This incident is no longer linked. Refresh the investigation.")
        db.execute("DELETE FROM rca_investigation_incidents WHERE id=?", (link["id"],))
    return {"incident_id": incident_id}


def set_mapping(key,source_id,user,mapping):
    with connect("server") as db:
        source=source_access(db,key,source_id,user,write=True)
        if source["status"]!="READY":raise ValueError("Parse the source before mapping columns.")
        if db.execute("SELECT id FROM rca_incident_evidence WHERE source_id=?",(source_id,)).fetchone():raise ValueError("Mapping is fixed after creating/linking incident evidence. Import a new source for a different interpretation.")
        columns=set()
        for row in db.execute("SELECT original_record_json FROM rca_evidence_records WHERE source_id=?",(source_id,)).fetchall():columns.update(json.loads(row["original_record_json"]))
        if not set(mapping.values())<=columns:raise ValueError("Mapping references nonexistent columns.")
        db.execute("UPDATE rca_evidence_sources SET mapping_json=?,updated_at=? WHERE id=?",(encode(mapping),utc_now(),source_id))
    return preview(key,source_id,user)

def add_finding(key,user,content,evidence,supersedes_id=None,request_id=None):
    with connect("server") as db:
        parent=require(db,key,user,write=True)
        if supersedes_id and not db.execute("SELECT id FROM rca_investigation_findings WHERE id=? AND investigation_id=?",(supersedes_id,key)).fetchone():raise ValueError("Unknown finding revision.")
        origin="Human"
        if request_id:
            row=db.execute("SELECT r.response_text FROM llm_tool_requests r JOIN rca_investigation_requests l ON r.id=l.request_id WHERE r.id=? AND l.investigation_id=?",(request_id,key)).fetchone()
            if not row or not row["response_text"]:raise ValueError("No saved explanation for this investigation request.")
            content=row["response_text"];origin="AI"
        finding,now=uuid4().hex,utc_now()
        db.execute("INSERT INTO rca_investigation_findings (id,investigation_id,userid,origin,content,request_id,supersedes_id,created_at) VALUES (?,?,?,?,?,?,?,?)",(finding,key,user["userid"],origin,content,request_id,supersedes_id,now))
        for item in evidence:
            source=owned(db,"rca_evidence_sources",item["source_id"],parent["userid"])
            selected=select_records(db,source,item.get("records"))
            if item["relationship"] not in {"supports","contradicts"}:raise ValueError("Evidence must support or contradict the finding.")
            db.execute("INSERT INTO rca_finding_evidence (id,finding_id,source_id,relationship,selection_json,created_at) VALUES (?,?,?,?,?,?)",(uuid4().hex,finding,source["id"],item["relationship"],encode(selected),now))
    return detail(key,user)

def remove_ready_source(key, user, source_id):
    """Hide an unused successful import without deleting its records or snapshot."""
    with connect("server") as db:
        require(db, key, user, write=True)
        source = owned(db, "rca_evidence_sources", source_id, user["userid"])
        if source["investigation_id"] != key or source["status"] != "READY":
            raise ValueError("Only ready sources imported into this investigation can be removed.")
        referenced = any(
            db.execute(f"SELECT id FROM {table} WHERE source_id=?", (source_id,)).fetchone()
            for table in ("rca_investigation_evidence", "rca_incident_evidence", "rca_finding_evidence")
        )
        # Incident primary records remain references even after their evidence link is removed.
        record_ids = {r["id"] for r in db.execute(
            "SELECT id FROM rca_evidence_records WHERE source_id=?", (source_id,)).fetchall()}
        incident_reference = any(
            json.loads(r["primary_details_json"]).get("source_record_id") in record_ids
            for r in db.execute("SELECT primary_details_json FROM rca_incidents WHERE userid=?",
                                (user["userid"],)).fetchall()
        )
        if referenced or incident_reference:
            reasons = []
            if referenced:
                reasons.append("it is linked by an attachment or finding")
            if incident_reference:
                reasons.append(
                    "one of its records is used as an incident's primary evidence"
                )
            raise ValueError(
                "This source cannot be removed because "
                + " and ".join(reasons)
                + ". Remove the related attachment or revise the finding first. "
                "Incident primary evidence must remain available."
            )
        db.execute("UPDATE rca_evidence_sources SET status='REMOVED',updated_at=? WHERE id=?",
                   (utc_now(), source_id))
    return {"source_id": source_id}


def restore_ready_source(key, user, source_id):
    """Undo only successful-import removal, never promote a failed import to READY."""
    with connect("server") as db:
        require(db, key, user, write=True)
        source = owned(db, "rca_evidence_sources", source_id, user["userid"])
        if source["investigation_id"] != key or source["status"] != "REMOVED":
            raise ValueError("This removed source cannot be restored. Refresh the investigation.")
        db.execute("UPDATE rca_evidence_sources SET status='READY',updated_at=? WHERE id=?",
                   (utc_now(), source_id))
    return {"source_id": source_id}


def clear_failed_sources(key, user, source_id=None):
    """Dismiss unreferenced failed imports; retain snapshots and failure details."""
    with connect("server") as db:
        require(db, key, user, write=True)
        rows = db.execute(
            "SELECT id FROM rca_evidence_sources "
            "WHERE investigation_id=? AND userid=? AND status='FAILED'",
            (key, user["userid"]),
        ).fetchall()
        cleared, skipped = [], []
        for row in rows:
            sid = row["id"]
            if source_id is not None and sid != source_id:
                continue
            referenced = any(
                db.execute(
                    f"SELECT id FROM {table} WHERE source_id=?", (sid,)
                ).fetchone()
                for table in (
                    "rca_investigation_evidence",
                    "rca_incident_evidence",
                    "rca_finding_evidence",
                )
            )
            if referenced:
                skipped.append(sid)
                continue
            db.execute(
                "UPDATE rca_evidence_sources "
                "SET status='DISMISSED',updated_at=? WHERE id=?",
                (utc_now(), sid),
            )
            cleared.append(sid)
    return {"cleared": cleared, "skipped": skipped}


def close(key,user):
    with connect("server") as db:
        require(db,key,user,write=True)
        active=db.execute("SELECT r.id FROM llm_tool_requests r JOIN rca_investigation_requests l ON r.id=l.request_id WHERE l.investigation_id=? AND r.status IN ('PROCESSING','PENDING','EXPLAINING')",(key,)).fetchone()
        importing=db.execute("SELECT id FROM rca_evidence_sources WHERE investigation_id=? AND status='IMPORTING'",(key,)).fetchone()
        if active or importing:raise ValueError("Finish or cancel active requests/imports before closing this investigation.")
        db.execute("UPDATE rca_investigations SET status='CLOSED',updated_at=? WHERE id=?",(utc_now(),key))
    return detail(key,user)
