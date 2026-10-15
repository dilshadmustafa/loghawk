"""Hawk Console's local registry for documents indexed into its RAG corpus."""

from __future__ import annotations

from datetime import UTC, datetime

from loghawk.database.database_layer import connect
from loghawk.utils.user_identity import get_userid


def _now() -> str:
    return datetime.now(UTC).isoformat()


def get_document(source_key: str) -> dict | None:
    with connect("console") as db:
        row = db.execute(
            """SELECT source_key, userid, kind, display_name, content_sha256,
                      chunks_indexed, created_at, updated_at
               FROM rag_documents WHERE source_key = ?""",
            (source_key,),
        ).fetchone()
    return dict(row) if row is not None else None


def find_document_by_name(display_name: str) -> list[dict]:
    with connect("console") as db:
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
    with connect("console") as db:
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
