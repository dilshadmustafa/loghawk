
import os
import duckdb
import ollama
import lancedb
from loghawk.utils import genutils
from loghawk.utils import duckdbutils
import loghawk.config as config

print("DUCKDB FILE PATH : ", config.LH_DUCKDB_FILE_PATH)
print("DUCKDB TABLE NAME : ", config.LH_DUCKDB_TABLE_NAME)
print("LANCEDB FILE PATH : ", config.LH_LANCEDB_FILE_PATH)
print("LANCEDB TABLE NAME : ", config.LH_LANCEDB_TABLE_NAME)

convo = []
con = duckdb.connect(database=str(config.LH_DUCKDB_FILE_PATH))

lance_db = None
lance_table = None

try:
    lance_db = lancedb.connect(str(config.LH_LANCEDB_FILE_PATH))
    lance_table = lance_db.open_table(config.LH_LANCEDB_TABLE_NAME)
except Exception as e:
    print(f"Could not open LanceDB table '{config.LH_LANCEDB_TABLE_NAME}': {e}")
    lance_table = None


def embed_text(text: str):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.LH_EMBEDDING_MODEL)
        return model.encode(text).tolist()
    except Exception:
        try:
            return ollama.embeddings(model=config.LH_EMBEDDING_MODEL, prompt=text)["embedding"]
        except Exception as e:
            print(f"Embedding failed: {e}")
            return None


def retrieve_rag_context(prompt: str, top_k: int = 5) -> str:
    if lance_table is None:
        return ""

    vector = embed_text(prompt)
    if not vector:
        return ""

    try:
        results = lance_table.search(vector).limit(top_k)

        try:
            items = results.to_list()
        except Exception:
            try:
                items = results.to_pandas().to_dict("records")
            except Exception:
                items = list(results)
    except Exception as e:
        print(f"LanceDB search failed: {e}")
        return ""

    context_chunks = []
    for row in items:
        if isinstance(row, dict):
            chunk = (
                row.get("text")
                or row.get("content")
                or row.get("document")
                or row.get("chunk")
                or row.get("value")
                or str(row)
            )
        else:
            chunk = str(row)

        if chunk and str(chunk).strip():
            context_chunks.append(str(chunk).strip())

    return "\n\n".join(context_chunks[:top_k])


def stream_response(prompt):
    rag_context = retrieve_rag_context(prompt)
    user_message = prompt

    if rag_context:
        user_message = (
            "Use the following retrieved context to answer the user's question.\n\n"
            f"Context:\n{rag_context}\n\n"
            f"Question:\n{prompt}"
        )

    convo.append({"role": "user", "content": user_message})

    if not genutils.contains_mostly_numbers(prompt):
        duckdbutils.db_insert(
            con,
            config.LH_DUCKDB_TABLE_NAME,
            "anon",
            "anon",
            "anon@anon.com",
            "user",
            content=prompt
        )

    response = ""
    stream = ollama.chat(model=config.LH_LLM_MODEL, messages=convo, stream=True)
    print("ASSISTANT: ")
    for chunk in stream:
        content = chunk["message"]["content"]
        response += content
        print(content, end="", flush=True)
    print("\n")
    print("End of Assistant Response\n")

    convo.append({"role": "assistant", "content": response})
    duckdbutils.db_insert(
        con,
        config.LH_DUCKDB_TABLE_NAME,
        "anon",
        "anon",
        "anon@anon.com",
        "assistant",
        content=response
    )


duckdbutils.populate_convo_from_db(con, config.LH_DUCKDB_TABLE_NAME, convo)

while True:
    prompt = genutils.get_multiline_input()
    if len(prompt.strip()) == 0:
        print("You entered all blank lines. Pls enter again.")
        continue
    if prompt.strip() == "exit":
        break
    stream_response(prompt=prompt)

con.close()
# No explicit LanceDB close() here; this connection object does not support it.