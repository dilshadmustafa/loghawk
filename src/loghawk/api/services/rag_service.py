"""Web API orchestration for reusable LogHawk RAG services."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import shutil
from pathlib import Path

import loghawk.config as config
from loghawk.api.services import rag_store
from loghawk.rag import ingestion
from loghawk.rag import retrieval


def create_upload_job(files: list[tuple[str, str]], upload_dir: str) -> dict:
    # (saved path, display/source name) pairs are kept in the job payload.
    return rag_store.create_job(
        "UPLOAD",
        f"{len(files)} uploaded file(s)",
        {
            "upload_dir": upload_dir,
            "files": [{"path": path, "source": name} for path, name in files],
        },
        files_total=len(files),
    )


def create_web_job(url: str) -> dict:
    return rag_store.create_job("WEB", url, {"url": url})


def run_job(job_id: str) -> None:
    job = rag_store.get_job(job_id)
    if not job or job["status"] not in {"QUEUED", "RUNNING"}:
        return
    print(f"[RAG] Started job={job_id} kind={job['kind']}", flush=True)
    rag_store.update_job(job_id, status="RUNNING", error=None)
    payload = job["payload"]
    try:
        if job["kind"] == "UPLOAD":
            paths = [(item["path"], item["source"]) for item in payload["files"]]
            chunks, processed, warnings = ingestion.ingest_file_paths(
                paths,
                rag_store,
                lancedb_path=config.LH_SERVER_LANCEDB_FILE_PATH,
                table_name=config.LH_SERVER_LANCEDB_TABLE_NAME,
            )
        else:
            chunks, processed, warnings = ingestion.ingest_website(
                payload["url"],
                rag_store,
                lancedb_path=config.LH_SERVER_LANCEDB_FILE_PATH,
                table_name=config.LH_SERVER_LANCEDB_TABLE_NAME,
            )

        duplicate_prefixes = (
            "Already indexed;",
            "Duplicate document ",
        )
        duplicate_only = (
            chunks == 0
            and processed > 0
            and bool(warnings)
            and all(message.startswith(duplicate_prefixes) for message in warnings)
        )
        if chunks == 0 and not duplicate_only:
            rag_store.update_job(
                job_id,
                status="FAILED",
                files_processed=processed,
                warnings_json=json.dumps(warnings),
                error="No supported, readable document content was found to index.",
                completed_at=datetime.now(UTC).isoformat(),
            )
            print(
                f"[RAG] Failed job={job_id}: no readable content; "
                f"items={processed}, warnings={len(warnings)}",
                flush=True,
            )
            return
        informational_prefixes = (
            "Already indexed;",
            "Duplicate document ",
            "A document named ",
            "Content changed;",
        )
        status = (
            "PARTIAL"
            if any(not message.startswith(informational_prefixes) for message in warnings)
            else "COMPLETED"
        )
        rag_store.update_job(
            job_id,
            status=status,
            files_processed=processed,
            chunks_indexed=chunks,
            warnings_json=json.dumps(warnings),
            completed_at=datetime.now(UTC).isoformat(),
        )
        print(
            f"[RAG] {status} job={job_id}: items={processed}, "
            f"chunks={chunks}, warnings={len(warnings)}",
            flush=True,
        )
    except Exception as exc:
        rag_store.update_job(
            job_id,
            status="FAILED",
            error=str(exc),
            completed_at=datetime.now(UTC).isoformat(),
        )
        print(
            f"[RAG] Failed job={job_id}: {type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        if job["kind"] == "UPLOAD":
            upload_dir = payload.get("upload_dir")
            if upload_dir:
                try:
                    shutil.rmtree(Path(upload_dir))
                except OSError:
                    pass


def answer_question(question: str, limit: int = 5) -> dict:
    return retrieval.answer_question(question, limit=limit)
