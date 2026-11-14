"""Capture immutable, redacted request inputs and prepare bounded model context."""
import json
from datetime import datetime
from uuid import uuid4
from loghawk.api.services import investigation_store as store, investigation_assessment as assessment
from loghawk.database.database_layer import connect
from loghawk.database.timestamps import utc_now
from loghawk.prompts.loader import prompt_snapshot
from loghawk.api.services.collector_evidence import bounded_json, prepare_model_evidence

def capture(key,userid,request_id,question,redact,conversation=None):
    with connect("server") as db:
        inv=store.require(db,key,{"userid":userid},write=True)
        context={"investigation_id":key,"captured_at":utc_now(),"question":question,"conversation":conversation or [],
                 "problem":inv["problem"],"reported_context":json.loads(inv["reported_context_json"]),
                 "incidents":[],"evidence":[],"findings":[],"prompts":[
                     prompt_snapshot("investigation/incident_context.txt"),
                     prompt_snapshot("investigation/evidence_explanation.txt"),
                     prompt_snapshot("investigation/evidence_driven_loop.txt")]}
        links=list(db.execute("SELECT source_id,selection_json FROM rca_investigation_evidence WHERE investigation_id=?",(key,)).fetchall())
        for row in db.execute("SELECT i.* FROM rca_incidents i JOIN rca_investigation_incidents l ON i.id=l.incident_id WHERE l.investigation_id=? AND i.deleted_at IS NULL ORDER BY i.id",(key,)).fetchall():
            context["incidents"].append(store.decoded(row))
            links.extend(db.execute("SELECT source_id,selection_json FROM rca_incident_evidence WHERE incident_id=?",(row["id"],)).fetchall())
        findings=db.execute("SELECT f.* FROM rca_investigation_findings f WHERE investigation_id=? AND NOT EXISTS (SELECT 1 FROM rca_investigation_findings newer WHERE newer.supersedes_id=f.id) ORDER BY created_at,id",(key,)).fetchall()
        for row in findings:
            finding=store.decoded(row)
            refs=db.execute("SELECT * FROM rca_finding_evidence WHERE finding_id=?",(row["id"],)).fetchall()
            finding["evidence"]=[store.decoded(r) for r in refs]
            links.extend(refs);context["findings"].append(finding)
        selections={}
        for link in links:
            source=link["source_id"];selection=json.loads(link["selection_json"])
            if source not in selections:selections[source]=selection
            elif selection is None or selections[source] is None:selections[source]=None
            else:selections[source]=sorted(set(selection)|set(selections[source]))
        for source_id,selection in sorted(selections.items()):
            source=store.owned(db,"rca_evidence_sources",source_id,userid)
            item={"source_id":source_id,"name":source["name"],"sha256":source["sha256"],
                  "original_uri":source["original_uri"],"evidence_type":source["evidence_type"],
                  "mapping":json.loads(source["mapping_json"]),"selected_record_numbers":selection,
                  "records":[],"collected_or_imported_at":source["created_at"]}
            if source["tool_call_id"]:
                evidence=db.execute("SELECT result_text,run_id,collector_name,collector_version FROM llm_tool_calls WHERE id=?",(source["tool_call_id"],)).fetchone()
                item["collector_evidence"]=dict(evidence) if evidence else {"unavailable":"Evidence removed by retention"}
            else:
                for row in db.execute("SELECT * FROM rca_evidence_records WHERE source_id=? ORDER BY record_number",(source_id,)).fetchall():
                    if selection is None or row["record_number"] in selection:item["records"].append(store.decoded(row))
            context["evidence"].append(item)
        context=redact(context)
        encoded=store.encode(context)
        if len(encoded.encode("utf-8"))>8*1024*1024:
            raise ValueError("Selected investigation context exceeds 8 MiB. Select fewer source records before asking a question.")
        db.execute("INSERT INTO rca_investigation_requests (id,investigation_id,request_id,snapshot_json,created_at) VALUES (?,?,?,?,?)",(uuid4().hex,key,request_id,encoded,utc_now()))
    return context

def _timestamp(value):
    """Parse an observation timestamp without inventing a missing timezone."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _incident_intervals(context):
    intervals = []
    for incident in context.get("incidents", []):
        details = incident.get("primary_details", {})
        start, end = _timestamp(details.get("start_time")), _timestamp(details.get("end_time"))
        if start is not None and end is not None and (start.tzinfo is None) == (end.tzinfo is None):
            if start <= end:
                intervals.append((start, end))
    return intervals


def _record_relation(record, mapping, intervals):
    fields = record.get("original_record", {})
    timestamp_key = mapping.get("timestamp", "timestamp")
    point = _timestamp(fields.get(timestamp_key)) or _timestamp(fields.get("start_time"))
    if point is None or not intervals:
        return "unknown"
    comparable = [(start, end) for start, end in intervals
                  if (point.tzinfo is None) == (start.tzinfo is None)]
    if not comparable:
        return "timezone_unresolved"
    inside = any(start <= point <= end for start, end in comparable)
    suffix = "_naive_interval" if point.tzinfo is None else "_incident_interval"
    return ("within" if inside else "outside") + suffix


def _anomaly_class(fields):
    flag = fields.get("is_anomaly")
    severity = str(fields.get("severity", fields.get("max_severity", ""))).upper()
    if flag is True or severity in {"CRITICAL", "HIGH", "WARNING", "WARN", "ERROR"}:
        return "anomalous"
    if flag is False or severity in {"NORMAL", "LOW", "INFO", "HEALTHY"}:
        return "normal"
    return "unknown"


def _ordered_records(source, intervals):
    """Interleave anomaly and baseline records, with incident-window records first."""
    groups = {}
    for record in source.get("records", []):
        relation = _record_relation(record, source.get("mapping", {}), intervals)
        rank = 0 if relation.startswith("within") else 2 if relation.startswith("outside") else 1
        kind = _anomaly_class(record.get("original_record", {}))
        groups.setdefault((rank, kind), []).append((record, relation))
    ordered = []
    for rank in range(3):
        buckets = [groups.get((rank, kind), []) for kind in ("anomalous", "normal", "unknown")]
        for index in range(max((len(bucket) for bucket in buckets), default=0)):
            ordered.extend(bucket[index] for bucket in buckets if index < len(bucket))
    return ordered


def _record_projection(source, record, relation, maximum_fields):
    fields = record.get("original_record", {})
    if not isinstance(fields, dict):
        fields = {"value": fields}
    # Prefer operational fields over database/import metadata. Keep the original field names.
    priority = ["timestamp", "start_time", "end_time", "incident_id", "service", "entity_id",
                "is_anomaly", "severity", "max_severity", "error_count", "http_5xx_count",
                "timeout_count", "error_rate", "timeout_rate", "anomaly_score", "reason",
                "namespace", "summary", "message"]
    mapping = source.get("mapping", {})
    keys = []
    for key in priority:
        actual = mapping.get(key, key)
        if not isinstance(actual, str) or actual not in fields or actual in keys:
            continue
        if key == "entity_id" and "service" in keys and fields.get(actual) == fields.get("service"):
            continue
        keys.append(actual)
    keys.extend(key for key in fields if key not in keys)
    selected = {key: fields[key] for key in keys[:maximum_fields]}
    return {"evidence_ref": f"source:{source['source_id']}:record:{record['record_number']}",
            "record_number": record["record_number"], "time_relation": relation,
            "original_record": selected, "omitted_fields": len(fields) - len(selected)}


def prepared(context, limit, aliases=None):
    """Build a source-balanced model view; the immutable snapshot remains complete.

    Counts refer only to records selected in the saved request. Size is checked after
    alias replacement, using the same compact JSON serialization as the caller.
    """
    limit = int(limit)
    if limit < 1:
        raise ValueError("Investigation context limit must be positive.")

    def encoded_size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))

    def model(value):
        return assessment.model_context(value, aliases, hide_source_ids=True) if aliases is not None else value

    sources = context.get("evidence", [])
    total_records = sum(len(source.get("records", [])) for source in sources)
    result = {"problem": "", "incidents": [], "findings": [], "evidence": [],
              "omitted_incidents": len(context.get("incidents", [])),
              "omitted_findings": len(context.get("findings", [])),
              "omitted_records": total_records, "omitted_sources": len(sources),
              "omitted_collector_sources": sum("collector_evidence" in source for source in sources),
              "notice": "Budget omissions are not absent evidence. Naive timestamp comparisons do not establish timezone alignment."}

    def fits():
        return encoded_size(result) <= limit

    # Reserve a small share for the question's context; avoid allowing long prose to
    # consume the budget intended for selected records.
    problem = str(context.get("problem", ""))
    result["problem"] = problem[:min(240, max(0, limit // 12))]
    if result["problem"] != problem:
        result["problem_omitted_chars"] = len(problem) - len(result["problem"])
    reported = context.get("reported_context", {})
    if reported:
        # Reported prose must leave room for selected observations.
        share = min(768, max(128, limit // 8))
        result["reported_context"] = json.loads(bounded_json(reported, share))
        if not fits():
            result.pop("reported_context")
            result["reported_context_omitted"] = True

    for incident in context.get("incidents", []):
        details = incident.get("primary_details", {})
        useful = {key: details[key] for key in (
            "incident_id", "start_time", "end_time", "primary_service", "services",
            "max_severity", "anomaly_count", "correlation_score") if key in details}
        item = model({"evidence_ref": "incident:" + incident["id"], "primary_details": useful,
                      "omitted_fields": len(details) - len(useful)})
        result["incidents"].append(item)
        result["omitted_incidents"] -= 1
        if not fits() or encoded_size(result["incidents"]) > max(400, limit // 3):
            result["incidents"].pop()
            result["omitted_incidents"] += 1

    intervals = _incident_intervals(context)
    incident_services = set()
    for incident in context.get("incidents", []):
        details = incident.get("primary_details", {})
        for value in (details.get("primary_service"), details.get("services")):
            if isinstance(value, str):
                incident_services.update(part.strip() for part in value.split(",") if part.strip())
            elif isinstance(value, list):
                incident_services.update(str(part) for part in value)

    def source_priority(source):
        relevant = any(str(record.get("original_record", {}).get("service", "")) in incident_services
                       or str(record.get("original_record", {}).get("entity_id", "")) in incident_services
                       for record in source.get("records", []))
        return (not relevant, str(source.get("evidence_type", "")),
                str(source.get("name", "")), str(source.get("original_uri", "")))

    included = []
    # Reserve descriptors for all sources before spending any budget on record rows.
    # Source names/locations and incident relevance determine order, never UUIDs.
    for source in sorted(sources, key=source_priority):
        item = {"name": source.get("name", ""), "evidence_type": source.get("evidence_type", "other"),
                "original_uri": source.get("original_uri", ""),
                "selected_record_count": len(source.get("records", [])), "records": [],
                "omitted_records": len(source.get("records", []))}
        if len(str(item["original_uri"])) > min(256, max(64, limit // (6 * max(1, len(sources))))):
            item.pop("original_uri", None)
            item["source_location_omitted"] = True
        if "collector_evidence" in source:
            item["collector_evidence_omitted"] = True
        result["evidence"].append(item)
        result["omitted_sources"] -= 1
        if not fits():
            item.pop("original_uri", None)
            item["source_location_omitted"] = True
        if not fits():
            result["evidence"].pop()
            result["omitted_sources"] += 1
            continue
        included.append((source, item, _ordered_records(source, intervals)))

    # A first compact row from each source gets a fair share; later rounds add
    # anomaly/baseline coverage before any existing row is expanded.
    rounds = max((len(rows) + ("collector_evidence" in source) for source, _, rows in included), default=0)
    for index in range(rounds):
        for source, item, rows in included:
            if index == 0 and "collector_evidence" in source:
                evidence = source["collector_evidence"]
                raw = evidence.get("result_text", evidence) if isinstance(evidence, dict) else evidence
                available = max(0, limit - encoded_size(result))
                share = min(1500, available // max(1, len(included)))
                if share >= 256:
                    try:
                        payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    except (ValueError, TypeError):
                        payload = {"text": raw}
                    if not isinstance(payload, dict):
                        payload = {"data": payload}
                    # The attached-source alias is the valid reference in this request.
                    payload["evidence_ref"] = "source:" + source["source_id"]
                    compacted = json.loads(prepare_model_evidence(payload, share, aliases))
                    if not compacted.get("evidence_omitted"):
                        item["collector_evidence"] = compacted
                        item["collector_evidence_omitted"] = False
                        result["omitted_collector_sources"] -= 1
                        if not fits():
                            item.pop("collector_evidence")
                            item["collector_evidence_omitted"] = True
                            result["omitted_collector_sources"] += 1
            row_index = index - int("collector_evidence" in source)
            if row_index < 0 or row_index >= len(rows):
                continue
            record, relation = rows[row_index]
            # Reserve capacity for the other sources in the first round. A large
            # first row must not starve the remaining sources' compact samples.
            remaining_sources = sum(not other["records"] and bool(other_rows)
                                    for _, other, other_rows in included)
            row_share = (max(0, limit - encoded_size(result)) // max(1, remaining_sources)
                         if index == 0 else limit)
            for field_count in (10, 7, 4):
                candidate = model(_record_projection(source, record, relation, field_count))
                if index == 0 and encoded_size(candidate) > row_share:
                    continue
                item["records"].append(candidate)
                item["omitted_records"] -= 1
                result["omitted_records"] -= 1
                if fits():
                    break
                item["records"].pop()
                item["omitted_records"] += 1
                result["omitted_records"] += 1

    # Findings are explicitly marked as user assertions and are considered only
    # after each selected evidence source has had an opportunity to contribute.
    for finding in context.get("findings", []):
        item = model({"evidence_ref": "finding:" + finding["id"], "origin": finding.get("origin", "user"),
                      "content": finding.get("content", "")})
        result["findings"].append(item)
        result["omitted_findings"] -= 1
        if not fits():
            result["findings"].pop()
            result["omitted_findings"] += 1

    if not fits():
        result = {"notice": "All selected context omitted because the model budget is too small.",
                  "omitted_incidents": len(context.get("incidents", [])),
                  "omitted_findings": len(context.get("findings", [])),
                  "omitted_sources": len(sources), "omitted_records": total_records}
    if encoded_size(result) > limit:
        raise ValueError("Investigation context budget is too small for an explicit omission report.")
    return result

def saved(request_id):
    with connect("server") as db:
        row=db.execute("SELECT investigation_id,snapshot_json FROM rca_investigation_requests WHERE request_id=?",(request_id,)).fetchone()
        return {"investigation_id":row["investigation_id"],"snapshot":json.loads(row["snapshot_json"])} if row else None
