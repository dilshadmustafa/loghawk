"""Ingest local documents into LanceDB using the configured LiteLLM embedder."""

import loghawk.config as config
from loghawk.utils import lancedbutils2


def ingest_directory(directory: str | None = None) -> int:
    source_dir = directory or str(config.LH_DOCS_STORAGE_DIR_PATH)
    documents = lancedbutils2.load_documents(source_dir)
    if not documents:
        print(f"No supported source documents found under {source_dir}")
        return 0

    db = lancedbutils2.connect_database()
    inserted = lancedbutils2.add_documents(
        db,
        config.LH_LANCEDB_TABLE_NAME,
        documents,
    )
    print(
        f"Indexed {inserted} chunks using {config.LH_EMBEDDING_MODEL} "
        f"into LanceDB table {config.LH_LANCEDB_TABLE_NAME!r}."
    )
    return inserted


if __name__ == "__main__":
    ingest_directory()
