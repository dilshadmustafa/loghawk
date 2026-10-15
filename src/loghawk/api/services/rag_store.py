"""Persistent RAG ingestion-job records using LogHawk's database abstraction."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from uuid import uuid4

from loghawk.database.database_layer import connect
from loghawk.utils.user_identity import get_userid


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_job(kind: str, source: str, payload: dict, files_total: int = 0) -> dict:
    job_id = uuid4().hex
    now = _now()
    with connect("server") as db:
        db.execute(
            """INSERT INTO rag_ingestion_jobs
               (id, userid, kind, status, source, payload_json, files_total,
                files_processed, chunks_indexed, error, warnings_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, '[]', ?, ?)""",
            (
                job_id, get_userid(), kind, "QUEUED", source,
                json.dumps(payload), files_total, now, now,
            ),
        )
    return get_job(job_id)


def get_document(source_key: str) -> dict | None:
    with connect("server") as db:
        row = db.execute(
            """SELECT source_key, userid, kind, display_name, content_sha256,
                      chunks_indexed, created_at, updated_at
               FROM rag_documents WHERE source_key = ?""",
            (source_key,),
        ).fetchone()
    return dict(row) if row is not None else None


def find_document_by_name(display_name: str) -> list[dict]:
    with connect("server") as db:
        rows = db.execute(
            """SELECT source_key, userid, kind, display_name, content_sha256,
                      chunks_indexed, created_at, updated_at
               FROM rag_documents WHERE LOWER(display_name) = LOWER(?)""",
            (display_name,),
        ).fetchall()
    return [dict(row) for row in rows]


def record_document(
    *,
    source_key: str,
    kind: str,
    display_name: str,
    content_sha256: str,
    chunks_indexed: int,
) -> None:
    now = _now()
    with connect("server") as db:
        existing = db.execute(
            "SELECT created_at FROM rag_documents WHERE source_key = ?",
            (source_key,),
        ).fetchone()
        if existing:
            db.execute(
                """UPDATE rag_documents SET userid = ?, kind = ?, display_name = ?,
                          content_sha256 = ?, chunks_indexed = ?, updated_at = ?
                   WHERE source_key = ?""",
                (
                    get_userid(), kind, display_name, content_sha256,
                    chunks_indexed, now, source_key,
                ),
            )
        else:
            db.execute(
                """INSERT INTO rag_documents
                   (source_key, userid, kind, display_name, content_sha256,
                    chunks_indexed, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source_key, get_userid(), kind, display_name, content_sha256,
                    chunks_indexed, now, now,
                ),
            )


def fail_interrupted_jobs() -> None:
    now = _now()
    with connect("server") as db:
        db.execute(
            """UPDATE rag_ingestion_jobs
               SET status = ?, error = ?, completed_at = ?, updated_at = ?
               WHERE status IN (?, ?)""",
            (
                "FAILED",
                "The API server restarted before this ingestion finished. Please retry it.",
                now,
                now,
                "QUEUED",
                "RUNNING",
            ),
        )


def update_job(job_id: str, **fields) -> None:
    allowed = {
        "status", "files_processed", "chunks_indexed", "error",
        "warnings_json", "completed_at",
    }
    values = {key: value for key, value in fields.items() if key in allowed}
    if not values:
        return
    values["updated_at"] = _now()
    assignments = ", ".join(f"{key} = ?" for key in values)
    with connect("server") as db:
        db.execute(
            f"UPDATE rag_ingestion_jobs SET {assignments} WHERE id = ?",
            (*values.values(), job_id),
        )


def get_job(job_id: str) -> dict | None:
    with connect("server") as db:
        row = db.execute(
            """SELECT id, userid, kind, status, source, files_total,
                      files_processed, chunks_indexed, error, warnings_json,
                      created_at, updated_at, completed_at, payload_json
               FROM rag_ingestion_jobs WHERE id = ?""",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["warnings"] = json.loads(item.pop("warnings_json") or "[]")
    item["payload"] = json.loads(item.pop("payload_json") or "{}")
    return item
