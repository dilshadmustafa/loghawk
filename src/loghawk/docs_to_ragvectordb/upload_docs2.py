"""Streamlit document assistant using LiteLLM chat and embeddings."""

from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document

import loghawk.config as config
from loghawk.llm import get_llm_client
from loghawk.utils import lancedbutils2


PROMPT_TEMPLATE = (
    "You are an expert research assistant. Use the provided document context "
    "to answer the query. If unsure, say you don't know. Be concise and factual.\n\n"
    "Previous conversation:\n{history}\n\n"
    "Query: {query}\n\nContext:\n{context}"
)

llm_client = get_llm_client()
db = lancedbutils2.connect_database()


def save_uploaded_file(uploaded_file) -> Path:
    config.LH_DOCS_STORAGE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    file_path = config.LH_DOCS_STORAGE_DIR_PATH / Path(uploaded_file.name).name
    with file_path.open("wb") as output_file:
        output_file.write(uploaded_file.getbuffer())
    return file_path


def load_pdf_documents(file_path: str | Path) -> list[Document]:
    return PDFPlumberLoader(str(file_path)).load()


def index_documents(documents: list[Document]) -> int:
    return lancedbutils2.add_documents(
        db,
        config.LH_LANCEDB_TABLE_NAME,
        documents,
    )


def find_related_documents(query: str, top_k: int = 5) -> list[str]:
    rows = lancedbutils2.search_text(
        db,
        config.LH_LANCEDB_TABLE_NAME,
        query,
        limit=top_k,
    )
    return [row.get("text", "") for row in rows if row.get("text")]


def generate_answer(query: str, context_documents: list[str], history: str = "") -> str:
    context = "\n\n".join(context_documents)
    prompt = PROMPT_TEMPLATE.format(
        history=history,
        query=query,
        context=context,
    )
    return llm_client.complete([
        {"role": "user", "content": prompt},
    ])


st.title("📘 LogHawk Document Assistant")
st.caption(
    f"Chat provider: {config.LH_LLM_PROVIDER} · "
    f"Embedding model: {config.LH_EMBEDDING_MODEL}"
)

uploaded_pdf = st.file_uploader(
    "Upload a PDF document",
    type="pdf",
    accept_multiple_files=False,
)

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""
if "queriesQA" not in st.session_state:
    st.session_state.queriesQA = ""

if uploaded_pdf and st.session_state.uploaded_file_name != uploaded_pdf.name:
    with st.spinner("Reading and indexing the document…"):
        saved_path = save_uploaded_file(uploaded_pdf)
        docs = load_pdf_documents(saved_path)
        indexed = index_documents(docs)
    st.session_state.uploaded_file_name = uploaded_pdf.name
    st.success(f"Document indexed in {indexed} chunks.")

user_input = st.chat_input("Ask a question about the indexed documents…")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    with st.spinner("Searching and generating an answer…"):
        relevant_docs = find_related_documents(user_input)
        answer = generate_answer(
            user_input,
            relevant_docs,
            st.session_state.queriesQA,
        )
    with st.chat_message("assistant", avatar="🤖"):
        st.write(answer)
    st.session_state.queriesQA += f"User: {user_input}\nAssistant: {answer}\n"
