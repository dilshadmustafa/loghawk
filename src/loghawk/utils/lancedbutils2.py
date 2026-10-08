"""LanceDB helpers that store explicit LiteLLM-generated vectors."""

from pathlib import Path
from typing import Any

import lancedb
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import loghawk.config as config
from loghawk.llm import get_llm_client


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv"}


def connect_database(db_path: str | Path | None = None):
    path = Path(db_path or config.LH_LANCEDB_FILE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(path))


def open_table(db, table_name: str):
    if table_name not in db.table_names():
        return None
    return db.open_table(table_name)


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def load_documents(directory: str | Path) -> list[Document]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Document source directory does not exist: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(PDFPlumberLoader(str(path)).load())
        elif suffix in SUPPORTED_TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": str(path)}))
    return documents


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_llm_client()
    vectors: list[list[float]] = []
    batch_size = config.LH_EMBEDDING_BATCH_SIZE
    for offset in range(0, len(texts), batch_size):
        vectors.extend(client.embed(texts[offset:offset + batch_size]))
    return vectors


def make_rows(
    texts: list[str],
    sources: list[str],
    vectors: list[list[float]],
) -> list[dict[str, Any]]:
    if not (len(texts) == len(sources) == len(vectors)):
        raise ValueError("Text, source, and vector counts must match.")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) > 1:
        raise ValueError("Cannot write vectors with differing dimensions.")
    return [
        {"text": text, "source": source, "vector": vector}
        for text, source, vector in zip(texts, sources, vectors)
    ]


def add_documents(db, table_name: str, documents: list[Document]) -> int:
    chunks = chunk_documents(documents)
    if not chunks:
        return 0
    texts = [chunk.page_content for chunk in chunks]
    sources = [str(chunk.metadata.get("source", "")) for chunk in chunks]
    rows = make_rows(texts, sources, embed_texts(texts))
    if table_name in db.table_names():
        db.open_table(table_name).add(rows)
    else:
        db.create_table(table_name, data=rows)
    return len(rows)


def search_text(db, table_name: str, text: str, limit: int = 5) -> list[dict[str, Any]]:
    if table_name not in db.table_names():
        return []
    query_vector = embed_texts([text])[0]
    return db.open_table(table_name).search(query_vector).limit(limit).to_list()


def rebuild_table(
    db,
    table_name: str,
    documents: list[Document],
) -> int:
    """Build a replacement table first, then replace only the named table."""
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No supported documents were found; existing table was kept.")

    texts = [chunk.page_content for chunk in chunks]
    sources = [str(chunk.metadata.get("source", "")) for chunk in chunks]
    rows = make_rows(texts, sources, embed_texts(texts))
    staging_name = f"{table_name}__litellm_rebuild_staging"

    if staging_name in db.table_names():
        db.drop_table(staging_name)
    staging = db.create_table(staging_name, data=rows)
    if staging.count_rows() != len(rows):
        raise RuntimeError("Staging table row count did not match the rebuilt corpus.")

    if table_name in db.table_names():
        db.drop_table(table_name)
    db.create_table(table_name, data=rows)
    db.drop_table(staging_name)
    return len(rows)
