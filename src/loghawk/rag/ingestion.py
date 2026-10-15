"""Shared RAG ingestion functions, independent of Web UI and Hawk Console."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Protocol
import zipfile

from loghawk.rag.crawling import crawl_site
from loghawk.utils import lancedbutils3


class DocumentRegistry(Protocol):
    """Persistence operations needed to make ingestion idempotent."""

    def get_document(self, source_key: str) -> dict | None: ...

    def find_document_by_name(self, display_name: str) -> list[dict]: ...

    def record_document(
        self,
        *,
        source_key: str,
        kind: str,
        display_name: str,
        content_sha256: str,
        chunks_indexed: int,
    ) -> None: ...


def ingest_file_paths(
    paths: list[tuple[str, str]],
    registry: DocumentRegistry,
    *,
    lancedb_path: str | Path,
    table_name: str,
) -> tuple[int, int, list[str]]:
    """Index files, skipping unchanged registry entries and replacing changed ones."""
    groups: dict[tuple[str, str], list] = defaultdict(list)
    skipped: list[str] = []
    processed = 0
    for path, source in paths:
        try:
            content = Path(path).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            parsed = lancedbutils3.load_document_bytes(Path(path).name, content, source=source)
            for document in parsed:
                document.metadata["content_sha256"] = digest
            groups[(source, digest)].extend(parsed)
            processed += 1
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            skipped.append(f"{source}: {exc}")
    return (
        _index_groups(groups, "UPLOAD", skipped, registry, lancedb_path, table_name),
        processed,
        skipped,
    )


def ingest_website(
    url: str,
    registry: DocumentRegistry,
    *,
    lancedb_path: str | Path,
    table_name: str,
) -> tuple[int, int, list[str]]:
    documents, pages, skipped = crawl_site(url)
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for document in documents:
        source = str(document.metadata.get("source", ""))
        digest = str(document.metadata.get("content_sha256", ""))
        groups[(source, digest)].append(document)
    return (
        _index_groups(groups, "WEB", skipped, registry, lancedb_path, table_name),
        pages,
        skipped,
    )


def _index_groups(
    groups: dict[tuple[str, str], list],
    kind: str,
    warnings: list[str],
    registry: DocumentRegistry,
    lancedb_path: str | Path,
    table_name: str,
) -> int:
    if not groups:
        return 0
    db = lancedbutils3.connect_database(lancedb_path)
    chunks_indexed = 0
    for (source, digest), documents in groups.items():
        if not digest:
            warnings.append(f"Could not calculate a content hash for {source}; skipped.")
            continue

        existing = registry.get_document(source)
        if existing and existing["content_sha256"] == digest:
            warnings.append(f"Already indexed; skipped unchanged source: {source}")
            continue

        display_name = Path(source).name if kind == "UPLOAD" else source
        if kind == "UPLOAD":
            name_matches = registry.find_document_by_name(display_name)
            identical_name_match = next(
                (item for item in name_matches if item["content_sha256"] == digest),
                None,
            )
            if identical_name_match and not existing:
                warnings.append(
                    f"Duplicate document {display_name!r}; content matches the indexed copy "
                    f"from {identical_name_match['source_key']}; skipped."
                )
                continue
            different_content = [item for item in name_matches if item["content_sha256"] != digest]
            if different_content and not existing:
                warnings.append(
                    f"A document named {display_name!r} is already indexed with different content; "
                    "this copy was indexed separately."
                )

        indexed = lancedbutils3.replace_source_documents(
            db, table_name, source, documents
        )
        if indexed == 0:
            warnings.append(
                f"No text was extracted from {source}; previously indexed content was kept."
            )
            continue
        registry.record_document(
            source_key=source,
            kind=kind,
            display_name=display_name,
            content_sha256=digest,
            chunks_indexed=indexed,
        )
        chunks_indexed += indexed
        if existing:
            warnings.append(f"Content changed; replaced the previously indexed copy: {source}")
    return chunks_indexed
