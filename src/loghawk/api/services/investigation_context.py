"""Capture immutable, redacted request inputs and prepare bounded model context."""
import json
from uuid import uuid4
from loghawk.api.services import investigation_store as store
from loghawk.database.database_layer import connect
from loghawk.database.timestamps import utc_now
from loghawk.prompts.loader import prompt_snapshot

def capture(key,userid,request_id,question,redact,conversation=None):
    with connect("server") as db:
        inv=store.require(db,key,{"userid":userid},write=True)
        context={"investigation_id":key,"captured_at":utc_now(),"question":question,"conversation":conversation or [],
                 "problem":inv["problem"],"reported_context":json.loads(inv["reported_context_json"]),
                 "incidents":[],"evidence":[],"findings":[],"prompts":[
                     prompt_snapshot("investigation/incident_context.txt"),
                     prompt_snapshot("investigation/evidence_explanation.txt")]}
        links=list(db.execute("SELECT source_id,selection_json FROM rca_investigation_evidence WHERE investigation_id=?",(key,)).fetchall())
        for row in db.execute("SELECT i.* FROM rca_incidents i JOIN rca_investigation_incidents l ON i.id=l.incident_id WHERE l.investigation_id=? ORDER BY i.id",(key,)).fetchall():
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

def prepared(context,limit):
    """Keep complete selected records where possible; never silently truncate JSON."""
    def size(value):return len(store.encode(value))
    result={"investigation_id":context["investigation_id"],"problem":context["problem"],
            "reported_context":context["reported_context"],"incidents":[],"findings":[],"evidence":[],
            "omitted_incidents":0,"omitted_findings":0,"omitted_records":0,"omitted_sources":0}
    for field in ("incidents","findings"):
        for item in context[field]:
            result[field].append(item)
            if size(result)>limit:
                result[field].pop();result["omitted_"+field]+=1
    for source in context["evidence"]:
        item={k:v for k,v in source.items() if k not in {"records","collector_evidence","selected_record_numbers"}}
        item.update(records=[],selected_record_count=len(source["records"]),omitted_records=0)
        if "collector_evidence" in source:item["collector_evidence"]=source["collector_evidence"]
        result["evidence"].append(item)
        if size(result)>limit:
            result["evidence"].pop();result["omitted_sources"]+=1;result["omitted_records"]+=len(source["records"]);continue
        for row in source["records"]:
            item["records"].append(row)
            if size(result)>limit:
                item["records"].pop();item["omitted_records"]+=1;result["omitted_records"]+=1
    if size(result)>limit:
        result={"investigation_id":context["investigation_id"],"notice":"Context exceeds this model budget; source details omitted.",
                "omitted_incidents":len(context["incidents"]),"omitted_findings":len(context["findings"]),
                "omitted_sources":len(context["evidence"]),"omitted_records":sum(len(s["records"]) for s in context["evidence"])}
    return result

def saved(request_id):
    with connect("server") as db:
        row=db.execute("SELECT investigation_id,snapshot_json FROM rca_investigation_requests WHERE request_id=?",(request_id,)).fetchone()
        return {"investigation_id":row["investigation_id"],"snapshot":json.loads(row["snapshot_json"])} if row else None
