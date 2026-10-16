import os
from pathlib import Path
import lancedb
from sentence_transformers import SentenceTransformer
import loghawk.config as config


def embed_text(text: str):
    model = SentenceTransformer(config.LH_EMBEDDING_MODEL)
    return model.encode(text).tolist()


def ingest_directory(directory: str):
    db = lancedb.connect(str(config.LH_LANCEDB_FILE_PATH))

    # create/open table
    try:
        table = db.open_table(config.LH_LANCEDB_TABLE_NAME)
    except Exception:
        table = None

    docs = []
    for path in Path(directory).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".json", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({
                "text": text,
                "source": str(path)
            })

    rows = []
    for doc in docs:
        rows.append({
            "text": doc["text"],
            "source": doc["source"],
            "vector": embed_text(doc["text"])
        })

    if table is None:
        table = db.create_table(config.LH_LANCEDB_TABLE_NAME, data=rows)
    else:
        table.add(rows)

    print(f"Inserted {len(rows)} rows into LanceDB table: {config.LH_LANCEDB_TABLE_NAME}")


if __name__ == "__main__":
    ingest_directory(str(config.LH_DOCS_STORAGE_DIR_PATH))