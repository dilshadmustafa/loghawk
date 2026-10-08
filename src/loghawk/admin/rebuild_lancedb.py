"""Re-embed the configured document corpus and replace one LanceDB table."""

import loghawk.config as config
from loghawk.utils import lancedbutils2


def main() -> None:
    print("=" * 70)
    print("LogHawk - Rebuild LanceDB with LiteLLM embeddings")
    print("=" * 70)
    print(f"Database: {config.LH_LANCEDB_FILE_PATH}")
    print(f"Table   : {config.LH_LANCEDB_TABLE_NAME}")
    print(f"Corpus  : {config.LH_DOCS_STORAGE_DIR_PATH}")
    print(f"Embedding model: {config.LH_EMBEDDING_MODEL}")

    documents = lancedbutils2.load_documents(config.LH_DOCS_STORAGE_DIR_PATH)
    print(f"Documents loaded: {len(documents)}")
    db = lancedbutils2.connect_database()
    rows = lancedbutils2.rebuild_table(
        db,
        config.LH_LANCEDB_TABLE_NAME,
        documents,
    )
    print(f"Rebuilt {config.LH_LANCEDB_TABLE_NAME!r} with {rows} embedded chunks.")


if __name__ == "__main__":
    main()
