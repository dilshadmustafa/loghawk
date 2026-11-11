"""Authenticated standalone incident and incident-evidence APIs."""
from datetime import date, datetime, timezone
from typing import Literal
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from loghawk import config
from loghawk.api.routers.auth import require_authenticated_user
from loghawk.api.services import evidence_import_service as imports, incident_store as store

router = APIRouter()


class IncidentCreate(BaseModel):
    incident_id: str = Field(min_length=1, max_length=300)
    summary: str = Field(default="", max_length=10000)
    start_time: str = Field(min_length=1, max_length=100)
    end_time: str = Field(min_length=1, max_length=100)
    severity: str = Field(default="", max_length=100)
    primary_service: str = Field(default="", max_length=300)
    services: list[str] = Field(default_factory=list, max_length=100)
    details: dict = Field(default_factory=dict)


class EvidenceSelection(BaseModel):
    source_id: str
    records: list[int] | None = None


class S3Object(BaseModel):
    uri: str
    etag: str | None = None


class S3Selection(BaseModel):
    uris: list[str] = Field(min_length=1, max_length=100)


class S3Import(BaseModel):
    objects: list[S3Object] = Field(min_length=1, max_length=100)
    evidence_type: Literal["stage-a", "stage-b", "stage-c", "raw-logs", "other"] = "other"
    record_path: str | None = None


class IncidentImportS3(BaseModel):
    objects: list[S3Object] = Field(min_length=1, max_length=100)
    record_path: str | None = None


class CollectorEvidence(BaseModel):
    tool_call_id: str = Field(min_length=1, max_length=128)


class CreateIncidentsFromImport(BaseModel):
    record_numbers: list[int] = Field(min_length=1, max_length=500)


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("")
def listing(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
            q: str = Query("", max_length=200), severity: str | None = Query(None, max_length=100),
            service: str | None = Query(None, max_length=300), created_from: date | None = None,
            created_to: date | None = None, user=Depends(require_authenticated_user)):
    if created_from and created_to and created_from > created_to:
        raise HTTPException(422, "Created-from date must be on or before created-to date.")
    return call(store.listing, user, offset, limit, q, severity, service, created_from, created_to)


@router.post("", status_code=201)
def create(body: IncidentCreate, user=Depends(require_authenticated_user)):
    try:
        start_time = datetime.fromisoformat(body.start_time.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(body.end_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, "Incident times must be valid ISO-8601 timestamps.") from exc
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    if end_time < start_time:
        raise HTTPException(422, "Incident end time must be at or after its start time.")
    return call(store.create, user, body.incident_id, body.summary, body.start_time, body.end_time,
                body.severity, body.primary_service, body.services, body.details)


@router.get("/import/sources")
def list_import_sources(user=Depends(require_authenticated_user)):
    return call(store.list_import_sources, user)


@router.get("/import/sources/{source_id}")
def preview_import_source(source_id: str, offset: int = Query(0, ge=0),
                         limit: int = Query(100, ge=1, le=100),
                         user=Depends(require_authenticated_user)):
    return call(store.preview_import_source, source_id, user, offset, limit)


@router.post("/import/sources/{source_id}/create")
def create_from_import(source_id: str, body: CreateIncidentsFromImport,
                       user=Depends(require_authenticated_user)):
    return call(store.create_from_import, user, source_id, body.record_numbers)


@router.post("/import/upload")
def upload_incident_file(files: list[UploadFile] = File(...),
                         record_path: str = Form(""),
                         user=Depends(require_authenticated_user)):
    if not 1 <= len(files) <= config.LH_INVESTIGATION_MAX_IMPORT_FILES:
        raise HTTPException(422, "Too many files in this import.")
    for file in files:
        call(imports.validate_source, file.filename or "unknown", "stage-c")
    staged, remaining = [], imports.max_bytes()
    try:
        for file in files:
            temp, size, digest = call(imports.stage, file.file, remaining)
            remaining -= size
            staged.append((file.filename or "unknown", temp, size, digest))
        return {"sources": [call(imports.import_staged, None, user, name, temp, size, digest,
                                  "stage-c", record_path or None)
                           for name, temp, size, digest in staged]}
    finally:
        for _, temp, _, _ in staged:
            temp.close()


@router.get("/import/s3")
def list_import_s3(uri: str = Query(...), token: str | None = None,
                   browse: bool = True, user=Depends(require_authenticated_user)):
    return call(imports.list_s3, uri, token, browse)


@router.post("/import/s3/resolve")
def resolve_import_s3(body: S3Selection, user=Depends(require_authenticated_user)):
    return call(imports.resolve_s3_selection, body.uris)


@router.post("/import/s3")
def import_incident_s3(body: IncidentImportS3, user=Depends(require_authenticated_user)):
    objects = [item.model_dump() for item in body.objects]
    return {"sources": call(imports.import_s3, None, user, objects, "stage-c", body.record_path)}


@router.delete("/{incident_id}", status_code=204)
def delete(incident_id: str, user=Depends(require_authenticated_user)) -> None:
    call(store.permanently_delete, incident_id, user)


@router.get("/{incident_id}")
def detail(incident_id: str, user=Depends(require_authenticated_user)):
    return call(store.detail, incident_id, user)


@router.post("/{incident_id}/evidence")
def attach(incident_id: str, body: EvidenceSelection, user=Depends(require_authenticated_user)):
    return call(store.attach, incident_id, user, body.source_id, body.records)


@router.post("/{incident_id}/evidence/unlink")
def unlink(incident_id: str, body: EvidenceSelection, user=Depends(require_authenticated_user)):
    return call(store.unlink, incident_id, user, body.source_id)

@router.post("/{incident_id}/sources/{source_id}/remove")
def remove_source(incident_id: str, source_id: str, user=Depends(require_authenticated_user)):
    return call(store.remove_source, incident_id, user, source_id)

@router.post("/{incident_id}/sources/{source_id}/restore")
def restore_source(incident_id: str, source_id: str, user=Depends(require_authenticated_user)):
    return call(store.restore_source, incident_id, user, source_id)


@router.get("/{incident_id}/sources/{source_id}")
def preview(incident_id: str, source_id: str, offset: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=100), user=Depends(require_authenticated_user)):
    return call(store.preview, incident_id, source_id, user, offset, limit)


@router.post("/{incident_id}/sources/upload")
def upload(incident_id: str, files: list[UploadFile] = File(...),
           evidence_type: str = Form("other"), record_path: str = Form(""),
           user=Depends(require_authenticated_user)):
    call(store.authorize, incident_id, user, write=True)
    if not 1 <= len(files) <= config.LH_INVESTIGATION_MAX_IMPORT_FILES:
        raise HTTPException(422, "Too many files in this import.")
    for file in files:
        call(imports.validate_source, file.filename or "unknown", evidence_type)
    staged, remaining = [], imports.max_bytes()
    try:
        for file in files:
            temp, size, digest = call(imports.stage, file.file, remaining)
            remaining -= size
            staged.append((file.filename or "unknown", temp, size, digest))
        return {"sources": [call(imports.import_staged, None, user, name, temp, size, digest,
                                  evidence_type, record_path or None, None, incident_id)
                           for name, temp, size, digest in staged]}
    finally:
        for _, temp, _, _ in staged:
            temp.close()


@router.get("/{incident_id}/s3")
def list_s3(incident_id: str, uri: str = Query(...), user=Depends(require_authenticated_user)):
    call(store.authorize, incident_id, user)
    return call(imports.list_s3, uri, browse=True)


@router.post("/{incident_id}/s3/resolve")
def resolve_s3(incident_id: str, body: S3Selection, user=Depends(require_authenticated_user)):
    call(store.authorize, incident_id, user)
    return call(imports.resolve_s3_selection, body.uris)


@router.post("/{incident_id}/sources/s3")
def import_s3(incident_id: str, body: S3Import, user=Depends(require_authenticated_user)):
    call(store.authorize, incident_id, user, write=True)
    return {"sources": call(imports.import_s3, None, user, [item.model_dump() for item in body.objects],
                              body.evidence_type, body.record_path, incident_id)}


@router.post("/{incident_id}/sources/collector")
def add_collector_evidence(incident_id: str, body: CollectorEvidence,
                           user=Depends(require_authenticated_user)):
    return call(store.add_collector_evidence, incident_id, user, body.tool_call_id)
