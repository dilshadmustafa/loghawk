#
# First set PYTHONPATH environment variable using below command based on your OS
# Replace all folder path to /path/to/your/loghawkproject/
# Linux:
# export PYTHONPATH="${PYTHONPATH}:/path/to/your/loghawkproject/"
# Windows cmd:
# set PYTHONPATH=%PYTHONPATH%;C:\aiopsmain\loghawk\
# Windows PowerShell:
# $Env:PYTHONPATH = "%PYTHONPATH%;C:\aiopsmain\loghawk\"
# Run this program using command 'streamlit run .\upload_security_doc.py'
# This will open the upload web page in your Browser
#
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from torch.cuda import device
import sys
sys.path.append("C:\\aiopsmain\\loghawk\\")
from loghawk_utils.lancedbutils import init_database

# --- Configuration ---
DB_FILE = "C:\\aiopsmain\\my_work\\mydb\\mylancedb"
TABLE_NAME = "loghawk"
PDF_STORAGE_PATH = './security_docs/'

# EmbeddingGemma (requires Hugging Face access request)
# Visit: https://huggingface.co/google/embeddinggemma-300m
# Run: huggingface-cli login
EMBEDDING_MODEL = 'google/embeddinggemma-300m'  # Google's new EmbeddingGemma model
EMBEDDING_DIMS = 256  # Truncated from 768 for 3x faster processing (Matryoshka learning)

# More efficient and powerful than llama3
LLM_MODEL = 'deepseek-r1:1.5b' #'qwen3:4b'  # 2.5GB, 256K context, rivals much larger models

# Global model instance to avoid reloading
EMBEDDING_MODEL_INSTANCE = None

# Get a sentence-transformer function
func = get_registry().get("sentence-transformers").create(name=EMBEDDING_MODEL)

class MySchema(LanceModel):
    # Embed the 'text' field automatically
    text: str = func.SourceField()
    # Store the embeddings in the 'vector' field
    vector: Vector(func.ndims()) = func.VectorField()

# Create a LanceDB table with the schema
import lancedb
#db = lancedb.connect("C:\\aiopsmain\\my_work\\mydb\\mylancedb")
#table = db.open_table("mytable")
db, table = init_database(DB_FILE, TABLE_NAME)

#queriesQA = ""

def appendQA(query):
    global queriesQA
    queriesQA = queriesQA + query + "\n"

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    print("------------------DOCMENTCHUNK---------------")
    print(document_chunks)
    for chunk in document_chunks:
        table.add([
            {"text": chunk.page_content }
        ])

def find_related_documents(query):
    #results = table.search(query).limit(1000).to_pandas()
    results = table.search(query).limit(1).to_pandas()
    return results.text.tolist()

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

if 'queriesQA' not in st.session_state:
    st.session_state.queriesQA = ""
queriesQA = st.session_state.queriesQA


if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""


# UI Configuration


st.title("📘 Upload Pdf/CSV/Json/Text files and Enter your query when prompted")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf and st.session_state.uploaded_file_name != uploaded_pdf.name:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    print("-----------CALLING INDEX----------")
    index_documents(processed_chunks)

    st.success("✅ Document processed successfully! Ask your questions below.")
    st.session_state.uploaded_file_name = uploaded_pdf.name

print("queriesQA:" + queriesQA)

user_input = st.chat_input("Enter your question about the document...")
print("user_input: " + str(user_input))

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Analyzing document..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(queriesQA + "\n" + user_input, relevant_docs)
        appendQA(user_input)
        appendQA(ai_response)
    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)

st.session_state.queriesQA = queriesQA



