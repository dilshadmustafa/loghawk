"""Authenticated investigation and evidence APIs shared by both clients."""
from typing import Literal
from uuid import uuid4
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from loghawk import config
from loghawk.api.routers.auth import require_authenticated_user
from loghawk.api.services import investigation_store as store, evidence_import_service as imports
from loghawk.database.database_layer import connect
from loghawk.database.timestamps import utc_now

router=APIRouter()
class Create(BaseModel):
    problem: str=Field(min_length=1,max_length=10000)
    reported_context: dict=Field(default_factory=dict)
class Selection(BaseModel):
    source_id: str
    records: list[int] | None=None
    incident_id: str | None=None
class UnlinkEvidence(BaseModel):
    source_id: str
    incident_id: str | None = None
class SourceType(BaseModel):
    evidence_type: Literal["stage-a", "stage-b", "stage-c", "raw-logs", "other"]
class Incident(BaseModel):
    source_id: str
    record_number: int=Field(ge=1)
class LinkIncident(BaseModel):
    incident_id: str
class Mapping(BaseModel):
    mapping: dict[str,str]
class Reparse(BaseModel):
    record_path: str | None=None
    format: Literal["csv","parquet","json","jsonl","log","txt"] | None=None
class FindingReference(BaseModel):
    source_id: str
    records: list[int] | None=None
    relationship: Literal["supports","contradicts"]
class Finding(BaseModel):
    content: str=Field(default="",max_length=20000)
    evidence: list[FindingReference]=Field(default_factory=list,max_length=100)
    supersedes_id: str | None=None
    request_id: str | None=None
class S3Object(BaseModel):
    uri: str
    etag: str | None=None
class S3Import(BaseModel):
    objects: list[S3Object]
    evidence_type: Literal["stage-a","stage-b","stage-c","raw-logs","other"]="other"
    record_path: str | None=None
class S3Selection(BaseModel):
    uris: list[str]=Field(min_length=1,max_length=100)
class ClearFailedSources(BaseModel):
    source_id: str | None = None
class CollectorEvidence(BaseModel):
    tool_call_id: str

def call(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except PermissionError as exc:raise HTTPException(403,str(exc)) from exc
    except (ValueError,KeyError) as exc:raise HTTPException(422,str(exc)) from exc

def allowed(key,user):
    with connect("server") as db:store.require(db,key,user,write=True)

@router.get("")
def listing(offset:int=Query(0,ge=0),limit:int=Query(50,ge=1,le=100),user=Depends(require_authenticated_user)):
    return call(store.listing,user,offset,limit)
@router.post("",status_code=201)
def create(body:Create,user=Depends(require_authenticated_user)):
    if not body.problem.strip():raise HTTPException(422,"Describe the investigation problem.")
    return call(store.create,user,body.problem.strip(),body.reported_context)
@router.get("/{key}")
def detail(key:str,user=Depends(require_authenticated_user)):return call(store.detail,key,user)
@router.post("/{key}/close")
def close(key:str,user=Depends(require_authenticated_user)):return call(store.close,key,user)
@router.get("/{key}/sources/{source_id}")
def preview(key:str,source_id:str,offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=100),user=Depends(require_authenticated_user)):
    return call(store.preview,key,source_id,user,offset,limit)
@router.post("/{key}/sources/{source_id}/remove")
def remove_ready_source(key: str, source_id: str, user=Depends(require_authenticated_user)):
    return call(store.remove_ready_source, key, user, source_id)

@router.post("/{key}/sources/{source_id}/restore")
def restore_ready_source(key: str, source_id: str, user=Depends(require_authenticated_user)):
    return call(store.restore_ready_source, key, user, source_id)

@router.put("/{key}/sources/{source_id}/type")
def source_type(key: str, source_id: str, body: SourceType, user=Depends(require_authenticated_user)):
    return call(store.set_source_type, key, user, source_id, body.evidence_type)

@router.post("/{key}/evidence/unlink")
def unlink_evidence(key: str, body: UnlinkEvidence, user=Depends(require_authenticated_user)):
    return call(store.unlink_evidence, key, user, body.source_id, body.incident_id)

@router.post("/{key}/evidence/restore")
def restore_evidence(key: str, body: Selection, user=Depends(require_authenticated_user)):
    return call(store.restore_evidence, key, user, body.source_id, body.records, body.incident_id)

@router.post("/{key}/incident-links/unlink")
def unlink_incident(key: str, body: LinkIncident, user=Depends(require_authenticated_user)):
    return call(store.unlink_incident, key, user, body.incident_id)

@router.put("/{key}/sources/{source_id}/mapping")
def mapping(key:str,source_id:str,body:Mapping,user=Depends(require_authenticated_user)):
    return call(store.set_mapping,key,source_id,user,body.mapping)
@router.post("/{key}/sources/{source_id}/reparse")
def reparse(key:str,source_id:str,body:Reparse,user=Depends(require_authenticated_user)):
    return call(imports.reparse,key,source_id,user,body.record_path,body.format)
@router.post("/{key}/evidence")
def attach(key:str,body:Selection,user=Depends(require_authenticated_user)):
    return call(store.attach,key,user,body.source_id,body.records,body.incident_id)
@router.post("/{key}/incidents")
def incident(key:str,body:Incident,user=Depends(require_authenticated_user)):
    return call(store.add_incident,key,user,body.source_id,body.record_number)
@router.post("/{key}/incident-links")
def incident_link(key:str,body:LinkIncident,user=Depends(require_authenticated_user)):
    return call(store.link_incident,key,user,body.incident_id)
@router.post("/{key}/findings")
def finding(key:str,body:Finding,user=Depends(require_authenticated_user)):
    if not body.content.strip() and not body.request_id:raise HTTPException(422,"Enter a finding or select a saved AI request.")
    return call(store.add_finding,key,user,body.content,[r.model_dump() for r in body.evidence],body.supersedes_id,body.request_id)
@router.post("/{key}/sources/clear-failed")
def clear_failed_sources(
    key: str,
    body: ClearFailedSources,
    user=Depends(require_authenticated_user),
):
    return call(store.clear_failed_sources, key, user, body.source_id)


@router.post("/{key}/sources/upload")
def upload(key:str,files:list[UploadFile]=File(...),evidence_type:str=Form("other"),record_path:str=Form(""),user=Depends(require_authenticated_user)):
    call(allowed,key,user)
    if not 1<=len(files)<=config.LH_INVESTIGATION_MAX_IMPORT_FILES:raise HTTPException(422,"Too many files in this import.")
    for file in files:call(imports.validate_source,file.filename or "unknown",evidence_type)
    staged=[];remaining=imports.max_bytes()
    try:
        for file in files:
            temp,size,digest=call(imports.stage,file.file,remaining)
            remaining-=size;staged.append((file.filename or "unknown",temp,size,digest))
        return {"sources":[call(imports.import_staged,key,user,name,temp,size,digest,evidence_type,record_path or None) for name,temp,size,digest in staged]}
    finally:
        for _,temp,_,_ in staged:temp.close()
        for file in files:file.file.close()
@router.get("/{key}/s3")
def list_s3(key:str,uri:str,token:str|None=None,browse:bool=False,user=Depends(require_authenticated_user)):
    call(allowed,key,user)
    try:return call(imports.list_s3,uri,token,browse)
    except HTTPException:raise
    except Exception as exc:raise HTTPException(502,"Could not list internal S3 objects. Check the URI and server storage configuration.") from exc
@router.post("/{key}/s3/resolve")
def resolve_s3(key:str,body:S3Selection,user=Depends(require_authenticated_user)):
    call(allowed,key,user)
    try:return call(imports.resolve_s3_selection,body.uris)
    except HTTPException:raise
    except Exception as exc:raise HTTPException(502,"Could not review selected S3 files. Refresh the browser and check storage connectivity.") from exc

@router.post("/{key}/sources/s3")
def import_s3(key:str,body:S3Import,user=Depends(require_authenticated_user)):
    call(allowed,key,user)
    try:return {"sources":call(imports.import_s3,key,user,[o.model_dump() for o in body.objects],body.evidence_type,body.record_path)}
    except HTTPException:raise
    except Exception as exc:raise HTTPException(502,"Could not import selected S3 objects. Refresh the listing and check storage connectivity.") from exc
@router.post("/{key}/sources/collector")
def collector_source(key:str,body:CollectorEvidence,user=Depends(require_authenticated_user)):
    def save():
        with connect("server") as db:
            parent=store.require(db,key,user,write=True)
            event=db.execute("SELECT c.* FROM llm_tool_calls c JOIN llm_tool_requests r ON c.request_id=r.id WHERE c.id=? AND r.userid=?",(body.tool_call_id,parent["userid"])).fetchone()
            if not event or event["status"] not in {"COMPLETED","FAILED"} or not event["run_id"]:raise ValueError("Select an owned collector execution event with a run ID.")
            existing=db.execute("SELECT id FROM rca_evidence_sources WHERE investigation_id=? AND tool_call_id=?",(key,body.tool_call_id)).fetchone()
            if existing:return {"id":existing["id"],"status":"DUPLICATE"}
            source,now=uuid4().hex,utc_now()
            db.execute("INSERT INTO rca_evidence_sources (id,userid,investigation_id,name,evidence_type,format,status,mapping_json,record_count,tool_call_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(source,user["userid"],key,event["collector_name"],"other","collector","READY","{}",0,body.tool_call_id,now,now))
        return {"id":source,"status":"READY"}
    return call(save)
