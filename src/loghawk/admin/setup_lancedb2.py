"""Initialize the LanceDB directory without changing indexed user data."""

import loghawk.config as config
from loghawk.utils import lancedbutils2


def main() -> None:
    print("=" * 70)
    print("LogHawk LanceDB Setup - LiteLLM Embeddings")
    print("=" * 70)
    print(f"Database path   : {config.LH_LANCEDB_FILE_PATH}")
    print(f"Table name      : {config.LH_LANCEDB_TABLE_NAME}")
    print(f"Embedding model : {config.LH_EMBEDDING_MODEL}")

    db = lancedbutils2.connect_database()
    if config.LH_LANCEDB_TABLE_NAME in db.table_names():
        print("Existing table found; setup left its rows and schema unchanged.")
        print("Run 'python -m loghawk.admin.rebuild_lancedb' to re-embed it.")
    else:
        print("Database is ready; the table will be created on first ingestion.")


if __name__ == "__main__":
    main()
