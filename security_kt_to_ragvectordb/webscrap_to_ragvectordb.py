import ollama
from sentence_transformers import SentenceTransformer
import time
import os
import glob
import requests
import re
from bs4 import BeautifulSoup

from typing import List
import struct

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from huggingface_hub import login

from loghawk_utils import logutils
from loghawk_utils.lancedbutils import init_database

# --- Configuration ---
DRY_RUN = False # Set to True to test logic without running the model
DB_FILE = "C:\\aiopsmain\\my_work\\mydb\\mylancedb"
TABLE_NAME = "loghawk"

# EmbeddingGemma (requires Hugging Face access request)
# Visit: https://huggingface.co/google/embeddinggemma-300m
# Run: huggingface-cli login
EMBEDDING_MODEL = 'google/embeddinggemma-300m'  # Google's new EmbeddingGemma model
EMBEDDING_DIMS = 256  # Truncated from 768 for 3x faster processing (Matryoshka learning)

# More efficient and powerful than llama3
LLM_MODEL = 'deepseek-r1:1.5b' #'qwen3:4b'  # 2.5GB, 256K context, rivals much larger models
DOCS_DIR = 'docs_from_webscrap/'  # Directory containing scraped documentation

# Global model instance to avoid reloading
EMBEDDING_MODEL_INSTANCE = None

# Documentation URLs to scrape
DOCUMENTATION_URLS = [
    { 'name': 'cve', 'url': 'https://www.cve.org/', 'enabled': True },
    { 'name': 'nvd', 'url': 'https://nvd.nist.gov/vuln', 'enabled': True },
    { 'name': 'qwen3_ollama', 'url': 'https://dilshadmustafa.bss.design', 'enabled': True }
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Get a sentence-transformer function
embedding_function = get_registry().get("sentence-transformers").create(name=EMBEDDING_MODEL)

class MyTableSchema(LanceModel):
    text: str = embedding_function.SourceField()
    vector: Vector(embedding_function.ndims()) = embedding_function.VectorField()  # Automatically gets the embedding dimension

def get_embedding_model():
    """Get or create the global embedding model instance."""
    global EMBEDDING_MODEL_INSTANCE
    if EMBEDDING_MODEL_INSTANCE is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)
        logutils.log_memory("Model Load", f"({EMBEDDING_MODEL})")
    return EMBEDDING_MODEL_INSTANCE

def scrape_docs():
    """Simple function to scrape documentation from URLs and save to docs folder."""
    print("📥 Scraping Documentation")
    print("=" * 40)
    
    # Create docs directory if it doesn't exist
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    successful = 0
    total = len(DOCUMENTATION_URLS)
    
    #for name, url in DOCUMENTATION_URLS.items():
    for item in DOCUMENTATION_URLS:
        name = item['name']
        url = item['url']
        enabled = item['enabled']
        if not enabled:
            continue
        print(f"📄 Fetching: {name}")
        print(f"   URL: {url}")
        
        try:
            # Fetch the page
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            # Parse with Beautiful Soup and extract text
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Get text content
            text_content = soup.get_text(separator='\n', strip=True)
            
            if not text_content.strip():
                print(f"   ⚠️  No content extracted from {url}")
                continue
            
            # Save to file
            filename = f"{name}.txt"
            filepath = os.path.join(DOCS_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {url}\n")
                f.write(f"Fetched: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(text_content)
            
            print(f"   ✅ Saved to: {filename} ({len(text_content)} chars)")
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Error fetching {url}: {e}")
        
        # Be respectful - add delay between requests
        time.sleep(1)
    
    print(f"\n📊 Results: {successful}/{total} documents scraped")
    return successful > 0

def token_based_chunking(text, tokenizer, max_tokens=2048, overlap_tokens=100):
    """
    Token-based chunking using the actual embedding model's tokenizer.
    Much more accurate than word-based chunking for demo purposes.
    """
    # Tokenize the entire text
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]  # No need to chunk
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Get chunk tokens
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        
        # Move start position with overlap
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    
    return chunks

def chunk_text(text, model, max_tokens=2048, overlap_tokens=100):
    """Use token-based chunking with the embedding model's tokenizer."""
    return token_based_chunking(text, model.tokenizer, max_tokens, overlap_tokens)

def ingest_docs():
    """Reads documents from docs directory and ingests them into the vector store."""
    logutils.log_memory("Demo Start", "")
    
    # If needed fresh start for ingestion for demo purpose
    # if os.path.exists(DB_FILE):
    #     print("🗑️  Removing existing database for fresh demo run...")
    #     os.remove(DB_FILE)
    
    # Remove docs folder for completely fresh scraping
    if os.path.exists(DOCS_DIR):
        print("🗑️  Removing existing docs folder for fresh scraping...")
        import shutil
        shutil.rmtree(DOCS_DIR)
    
    logutils.log_memory("After Cleanup", "")
        
    print("--- Starting Document Ingestion ---")
    
    # Always scrape since we removed the docs folder
    print("🌐 Scraping fresh documentation...")
    if not scrape_docs():
        print("❌ Failed to scrape documentation.")
        return
    
    logutils.log_memory("After Scraping", "")
        
    # Check if docs were scraped successfully
    doc_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not doc_files:
        print("❌ No documentation files found after scraping.")
        return
    
    # 2. Initialize embedding model
    model = get_embedding_model()
    
    # 3. Initialize database
    db, table = init_database(DB_FILE, TABLE_NAME)

    print(f"📁 Found {len(doc_files)} documentation files:")
    for file in doc_files:
        print(f"   • {os.path.basename(file)}")
    
    all_chunks = []
    chunk_sources = []
    
    # 4. Process each document file
    for doc_file in doc_files:
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use token-based chunking with the embedding model's tokenizer
            chunks = chunk_text(content, model, max_tokens=2048, overlap_tokens=100)
            source_name = os.path.basename(doc_file).replace('.txt', '')
            
            all_chunks.extend(chunks)
            chunk_sources.extend([source_name] * len(chunks))
            
            print(f"📄 {source_name}: {len(chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error reading {doc_file}: {e}")
            continue

    if not all_chunks:
        print("❌ No content found to ingest.")
        return

    print(f"📊 Total chunks to process: {len(all_chunks)}")
    logutils.log_memory("After Chunking", f"({len(all_chunks)} chunks)")
    
    # 5. Generate embeddings and insert documents
    start_time = time.time()
    batch_size = 10  # Process in batches for better progress tracking
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_sources = chunk_sources[i:i + batch_size]
        
        for j, (chunk, source) in enumerate(zip(batch_chunks, batch_sources)):
            # Generate embedding using proper document prompt and dimension truncation
            # EmbeddingGemma uses specific prompts for optimal performance
            embedding = model.encode_document(chunk, truncate_dim=256)
            
            table.add([
                {
                    "text": chunk
                }
            ])
        # Progress indicator
        processed = min(i + batch_size, len(all_chunks))
        print(f"🔄 Processed {processed}/{len(all_chunks)} chunks...")

    #close db
    end_time = time.time()
    
    print(f"✅ Ingestion complete in {end_time - start_time:.2f} seconds.")
    print(f"📈 Average: {len(all_chunks)/(end_time - start_time):.1f} chunks/second")
    print("--- Ingestion Finished ---")
    logutils.log_memory("After Storage", "(data saved to SQLite)")
    print("✅ Document ingestion complete!")


def semantic_search_and_query(query_text, top_k=3):
    """Performs semantic search and generates response using local LLM."""
    logutils.log_memory("Query Start", f"('{query_text[:30]}...')")
    
    # 1. Get embedding model (reuse existing instance)
    model = get_embedding_model()

    # 2. Connect to database
    db, table = init_database(DB_FILE, TABLE_NAME)

    # 3. Generate query embedding using proper query prompt and dimension truncation
    # EmbeddingGemma uses specific prompts for optimal performance
    query_embedding = model.encode_query(query_text, truncate_dim=256)

    # 4. Find similar documents using sqlite-vec
    start_time = time.time()

    results = table.search(query_text).limit(10).to_pandas()
    print(results)
    print(results.text.tolist())
    #results = cursor.fetchall()
    end_time = time.time()
    
    if results.empty:
        print("❌ No relevant documents found.")
        #close db
        logutils.log_memory("After Vector Search", "0 results")
        return "No relevant documents found."

    print(f"✅ Found {len(results)} relevant chunks in {end_time - start_time:.3f} seconds")
    logutils.log_memory("After Vector Search", f"({len(results)} results)")
    
    # 5. Combine top results for context
    contexts = []
    sources = []
    for item in results.text:
        contexts.append(item)
        #sources.append(item)
        #sources.append(f"{source} (distance: {distance:.3f})")
        #print(f"📄 Source: {source} | Distance: {distance:.4f}")
    
    combined_context = "\n\n".join(contexts)
    unique_sources = list(set([s.split(' (')[0] for s in sources]))
    
    # 6. Build the prompt with multiple contexts
    prompt = f"""Use the following contexts to answer the question comprehensively.
If you don't know the answer based on the provided contexts, just say that you don't know.

Contexts:
{combined_context}

Question: {query_text}

Answer:"""

    # 7. Get streaming response from LLM
    print(f"\n💡 Answer (sources: {', '.join(unique_sources)}):")
    print("=" * 60)

    if DRY_RUN:
        response_content = "This is a DRY RUN response based on the found contexts."
        print(response_content)
    else:
        print(f"🤖 {LLM_MODEL} is thinking and responding...")
        print()
        
        start_time = time.time()
        
        # Stream the response in real-time
        try:
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,  # Enable streaming!
            )
            
            full_response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)  # Print immediately
                    full_response += content
            
            end_time = time.time()
            print(f"\n\n⚡ Response completed in {end_time - start_time:.2f} seconds.")
            response_text = full_response
            
        except Exception as e:
            print(f"❌ Error during streaming: {e}")
            response_text = "Error during LLM response."

    print("=" * 60)
    
    #close db
    logutils.log_memory("After LLM Response", "")
    
    # Print memory summary after each query
    logutils.print_memory_summary()
    
    return response_text

def main():
    """Main function for web scrapping."""
    print("🚀 Web Scrap To RAG Vector DB")
    print("=" * 60)
    print("🔒 100% Private | 💰 Zero Cost | 📱 Local")
    print("📚 Using official docs from NIST, MITRE ATTCK, CVEs, etc.")
    print()
    print("📥 Login to Hugging Face Hub using command 'huggingface-cli login' and provide your free token. This is needed to access Google's EmbeddingGemma-300M model")
    print("📥 You will be prompted to enter your Hugging Face token")
    #login()  # You will be prompted to enter your Hugging Face token

    # Ingest all documentation
    ingest_docs()
    
    # Run demo queries
    run_demo_queries()
    
    # Demo complete - memory summary already printed after last query
    print("\n🏁 Demo Complete!")

def run_demo_queries():
    """Run a series of demo queries to showcase the RAG system."""
    demo_queries = [
        "What CVEs are related to vulnerabilities in vsftpd?",
        "Based on the log data I gave above, check for vulnerability?"
    ]
    
    print("🎯 Running demo queries to showcase semantic search capabilities:")
    print()
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*20} Demo Query {i}/{len(demo_queries)} {'='*20}")
        print()
        print("--- Starting Query ---")
        print(f"Query: {query}")
        
        response = semantic_search_and_query(query)
        
        if response:
            print(f"\n--- Response ---")
            print(response)

if __name__ == "__main__":
    main()
