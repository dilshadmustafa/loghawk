"""Web UI endpoints for shared document ingestion and RAG chat."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil
import tempfile

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, field_validator

import loghawk.config as config
from loghawk.api.services import rag_service, rag_store


router = APIRouter()


class WebCrawlRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class RagQuestion(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty")
        return value


def _public_job(job: dict) -> dict:
    job.pop("payload", None)
    return job


def _safe_relative_name(filename: str) -> Path:
    normalized = filename.replace("\\", "/")
    parts = [
        part
        for part in PurePosixPath(normalized).parts
        if part not in {"", ".", "..", "/"}
    ]
    if parts and parts[0].endswith(":"):
        parts = parts[1:]
    return Path(*parts) if parts else Path("upload.bin")


@router.post("/ingestions/upload", status_code=202)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> dict:
    if not files:
        raise HTTPException(status_code=422, detail="Select at least one file or folder.")

    max_bytes = config.LH_RAG_MAX_UPLOAD_MB * 1024 * 1024
    config.LH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    upload_root = Path(tempfile.mkdtemp(prefix="rag-upload-", dir=config.LH_DATA_DIR))
    saved_files: list[tuple[str, str]] = []
    total_bytes = 0
    used_names: set[str] = set()
    try:
        for upload in files:
            source_name = upload.filename or "upload.bin"
            relative = _safe_relative_name(source_name)
            candidate = relative
            index = 1
            while candidate.as_posix().lower() in used_names:
                candidate = relative.with_name(f"{relative.stem}-{index}{relative.suffix}")
                index += 1
            used_names.add(candidate.as_posix().lower())
            target = upload_root / candidate
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds the {config.LH_RAG_MAX_UPLOAD_MB} MB limit.",
                        )
                    output.write(chunk)
            saved_files.append((str(target), candidate.as_posix()))
            await upload.close()

        if not saved_files:
            raise HTTPException(status_code=422, detail="No uploaded files were received.")
        job = rag_service.create_upload_job(saved_files, str(upload_root))
    except HTTPException:
        shutil.rmtree(upload_root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(upload_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not stage upload: {exc}") from exc

    background_tasks.add_task(rag_service.run_job, job["id"])
    return _public_job(job)


@router.post("/ingestions/web", status_code=202)
def scrape_website(body: WebCrawlRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        job = rag_service.create_web_job(body.url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not create crawl job: {exc}") from exc
    background_tasks.add_task(rag_service.run_job, job["id"])
    return _public_job(job)


@router.get("/ingestions/{job_id}")
def get_ingestion(job_id: str) -> dict:
    job = rag_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="RAG ingestion job not found")
    return _public_job(job)


@router.post("/chat")
def rag_chat(body: RagQuestion) -> dict:
    try:
        return rag_service.answer_question(body.question)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG chat request failed: {exc}") from exc
