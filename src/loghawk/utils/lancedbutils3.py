"""Reusable LanceDB document helpers backed by LiteLLM embeddings.

This version adds an in-memory byte loader for files received by the Web UI or
other LogHawk clients. It has no FastAPI or UI dependencies.
"""

from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
import zipfile

import lancedb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import loghawk.config as config
from loghawk.llm import get_llm_client


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".csv",
    ".html",
    ".htm",
    ".xml",
    ".yaml",
    ".yml",
    ".docx",
    ".rtf",
}
SUPPORTED_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".csv",
    ".xml", ".yaml", ".yml",
}


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text while skipping non-content HTML elements."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, _attrs) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag.lower() in {"br", "p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def _suffix(filename: str | Path) -> str:
    # URLs may include query strings, so determine the extension from their path.
    return Path(urlsplit(str(filename)).path).suffix.lower()


def _html_text(content: bytes) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(content.decode("utf-8", errors="replace"))
    return " ".join(" ".join(parser.parts).split())


def _docx_text(content: bytes) -> str:
    with zipfile.ZipFile(BytesIO(content)) as archive:
        xml_content = archive.read("word/document.xml")
    root = ET.fromstring(xml_content)
    paragraphs: list[str] = []
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        text = "".join(
            node.text or ""
            for node in paragraph.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
        )
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _rtf_text(content: bytes) -> str:
    # Decode common RTF character escapes and remove formatting control words.
    value = content.decode("latin-1", errors="replace")
    value = re.sub(r"\\'([0-9a-fA-F]{2})", lambda match: bytes.fromhex(match.group(1)).decode("cp1252", errors="replace"), value)
    value = re.sub(r"\\(par|line)\b ?", "\n", value)
    value = re.sub(r"\\'[0-9a-fA-F]{2}", "", value)
    value = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", value)
    value = re.sub(r"\\([{}\\])", r"\1", value)
    value = re.sub(r"\\[^\s]", "", value)
    return re.sub(r"[{}]", "", value).strip()


def _extract_documents(filename: str, content: bytes, source: str) -> list[Document]:
    suffix = _suffix(filename)
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported document format {suffix or '(no extension)'!r}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    if suffix == ".pdf":
        import pdfplumber

        documents: list[Document] = []
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": source, "page": page_number},
                        )
                    )
        return documents

    if suffix in {".html", ".htm"}:
        text = _html_text(content)
    elif suffix == ".docx":
        text = _docx_text(content)
    elif suffix == ".rtf":
        text = _rtf_text(content)
    else:
        text = content.decode("utf-8", errors="replace")

    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": source})]


def connect_database(db_path: str | Path | None = None):
    path = Path(db_path or config.LH_SERVER_LANCEDB_FILE_PATH)
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


def load_document_bytes(
    filename: str,
    content: bytes,
    *,
    source: str | None = None,
) -> list[Document]:
    """Parse one supported file from memory, retaining its path or URL as source."""
    return _extract_documents(filename, content, source or filename)


def load_documents(directory: str | Path) -> list[Document]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Document source directory does not exist: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        documents.extend(
            load_document_bytes(path.name, path.read_bytes(), source=str(path))
        )
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


def replace_source_documents(db, table_name: str, source: str, documents: list[Document]) -> int:
    """Replace all indexed chunks for a source after preparing its new vectors."""
    chunks = chunk_documents(documents)
    if not chunks:
        return 0
    texts = [chunk.page_content for chunk in chunks]
    sources = [str(chunk.metadata.get("source", source)) for chunk in chunks]
    rows = make_rows(texts, sources, embed_texts(texts))

    if table_name in db.table_names():
        escaped_source = source.replace("'", "''")
        table = db.open_table(table_name)
        table.delete(f"source = '{escaped_source}'")
        table.add(rows)
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
