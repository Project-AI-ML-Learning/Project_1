# ─────────────────────────────────────────────
# api/main.py
# FastAPI backend for the RAG pipeline.
#
# Endpoints:
#   GET    /health              → is API alive?
#   GET    /docs-list           → list loaded PDFs
#   POST   /upload              → upload + embed PDF
#   POST   /query               → ask a question
#   DELETE /document/{filename} → remove a PDF
#
# Run locally:
#   uvicorn api.main:app --reload --port 8000
#
# On EC2:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────

import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI

# ── RAG core modules ───────────────────────────
from core.loader    import load_all_pdfs
from core.chunker   import chunk_documents
from core.embedder  import (embed_and_store,
                             delete_file_vectors,
                             get_vectorstore)
from core.registry  import (has_file_changed,
                             update_registry,
                             remove_from_registry)
from core.cache     import (get_cached_answer,
                             store_answer,
                             clear_all_cache)
from core.retriever import (route_and_retrieve,
                             build_prompt,
                             is_casual_conversation,
                             get_casual_reply)

# ── Schemas ────────────────────────────────────
from api.schemas import (QueryRequest, QueryResponse,
                          SourceModel, DocumentModel,
                          DocumentListResponse, UploadResponse,
                          DeleteResponse, HealthResponse)

load_dotenv()

# ── Constants ──────────────────────────────────
# Use absolute path so it works from any directory
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_FOLDER = os.path.join(BASE_DIR, "documents")
API_VERSION = "1.0.0"

os.makedirs(DOCS_FOLDER, exist_ok=True)

# ── LLM — one instance reused across requests ──
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0
)


# ═══════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════

app = FastAPI(
    title="RAG Document Assistant API",
    description="Production RAG pipeline with FastAPI",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS ───────────────────────────────────────
# Allows frontend on different port to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # lock to domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════

def get_chunk_count(filename):
    """Count chunks stored in ChromaDB for a file."""
    try:
        vectorstore = get_vectorstore()
        existing    = vectorstore.get()
        return sum(
            1 for cid in existing["ids"]
            if cid.startswith(filename)
        )
    except:
        return 0


def extract_sources(chunks):
    """
    Pull unique source + page from chunks.
    Returns list of SourceModel objects.
    Deduplicates so same page not listed twice.
    """
    sources = []
    seen    = set()
    for chunk in chunks:
        key = (
            chunk.metadata.get("source_file", "?"),
            chunk.metadata.get("page", 0)
        )
        if key not in seen:
            seen.add(key)
            sources.append(SourceModel(
                file=str(key[0]),
                page=int(key[1]) if str(key[1]).isdigit() else 0
            ))
    return sources


# ═══════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════

# ── 1. Health Check ────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    GET /health
    Confirms API is alive.
    Used by monitoring tools and load balancers.
    Always test this first after deployment.
    """
    docs_count = len([
        f for f in os.listdir(DOCS_FOLDER)
        if f.endswith(".pdf")
    ])

    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        docs_loaded=docs_count
    )


# ── 2. List Documents ──────────────────────────
@app.get("/docs-list", response_model=DocumentListResponse)
async def list_documents():
    """
    GET /docs-list
    Returns all loaded PDFs with chunk counts.
    Client uses this to show document list in UI.
    """
    pdf_files = [
        f for f in os.listdir(DOCS_FOLDER)
        if f.endswith(".pdf")
    ]

    documents = [
        DocumentModel(
            filename=f,
            chunks=get_chunk_count(f)
        )
        for f in pdf_files
    ]

    return DocumentListResponse(
        documents=documents,
        total=len(documents)
    )


# ── 3. Upload Document ─────────────────────────
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    POST /upload
    Accepts PDF file upload.
    Saves → checks registry → embeds if needed.

    Three outcomes:
      new       → embed and store
      updated   → delete old vectors + re-embed
      unchanged → skip, already in system
    """

    # Validate — only PDFs accepted
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    filepath = os.path.join(DOCS_FOLDER, file.filename)

    # Save uploaded file to disk
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Check status via registry
    status = has_file_changed(file.filename, filepath)

    if status == "new":
        # New file → embed and store
        pages  = load_all_pdfs(DOCS_FOLDER)
        chunks = chunk_documents(pages)
        embed_and_store(chunks)
        update_registry(file.filename, filepath)
        chunk_count = get_chunk_count(file.filename)

        return UploadResponse(
            filename=file.filename,
            status="new",
            chunks=chunk_count,
            message=f"Successfully embedded {chunk_count} chunks"
        )

    elif status == "edited":
        # Edited → delete old + re-embed
        delete_file_vectors(file.filename)
        clear_all_cache()
        pages  = load_all_pdfs(DOCS_FOLDER)
        chunks = chunk_documents(pages)
        embed_and_store(chunks)
        update_registry(file.filename, filepath)
        chunk_count = get_chunk_count(file.filename)

        return UploadResponse(
            filename=file.filename,
            status="updated",
            chunks=chunk_count,
            message=f"Updated — re-embedded {chunk_count} chunks"
        )

    else:
        # Unchanged — already in system
        chunk_count = get_chunk_count(file.filename)

        return UploadResponse(
            filename=file.filename,
            status="unchanged",
            chunks=chunk_count,
            message="File already loaded — no changes detected"
        )


# ── 4. Query ───────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    POST /query
    Main RAG endpoint with three layer check.

    Layer 1 → Casual conversation → warm reply
    Layer 2 → No docs loaded     → friendly nudge
    Layer 3 → Full RAG pipeline:
                cache check
                → intent routing
                → retrieval
                → Gemini
                → answer + sources

    Request body:
    {
      "question": "...",
      "chat_history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
    """

    question     = request.question.strip()
    chat_history = request.chat_history or []

    # Guard: empty question
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # ── Layer 1: Casual conversation ───────────
    # Greetings + chit-chat handled here
    # No ChromaDB. No Gemini. Free and instant.
    if is_casual_conversation(question):
        answer = get_casual_reply(question)
        return QueryResponse(
            question=question,
            answer=answer,
            sources=[],
            retriever_type="conversation",
            from_cache=False
        )

    # ── Layer 2: Documents loaded? ─────────────
    # Only check after confirming real question
    docs_exist = any(
        f.endswith(".pdf")
        for f in os.listdir(DOCS_FOLDER)
    )
    if not docs_exist:
        return QueryResponse(
            question=question,
            answer=(
                "No documents loaded. "
                "Please upload a PDF via POST /upload first."
            ),
            sources=[],
            retriever_type="none",
            from_cache=False
        )

    # ── Layer 3: Cache check ───────────────────
    # Same question before? Return instantly.
    # Zero ChromaDB search. Zero Gemini call.
    cached = get_cached_answer(question)
    if cached:
        return QueryResponse(
            question=question,
            answer=cached,
            sources=[],
            retriever_type="cache",
            from_cache=True
        )

    # ── Layer 4: Full RAG pipeline ─────────────
    # Step 1: Rewrite query + route + retrieve
    chunks, retriever_type = route_and_retrieve(
        question,
        chat_history=chat_history        # ← memory passed here
    )

    # Step 2: Build smart prompt with memory
    prompt = build_prompt(
        question,
        chunks,
        retriever_type,
        chat_history=chat_history        # ← memory passed here
    )

    # Step 3: Send to Gemini
    response = llm.invoke(prompt)
    answer   = response.content

    # Step 4: Extract source citations
    sources = extract_sources(chunks)

    # Step 5: Store in cache for next time
    store_answer(question, answer)

    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        retriever_type=retriever_type,
        from_cache=False
    )


# ── 5. Delete Document ─────────────────────────
@app.delete("/document/{filename}",
            response_model=DeleteResponse)
async def delete_document(filename: str):
    """
    DELETE /document/{filename}
    Removes document completely:
      → vectors from ChromaDB
      → query cache
      → physical file from disk
      → entry from file registry
    """

    filepath = os.path.join(DOCS_FOLDER, filename)

    # Check file exists
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found in system"
        )

    # Step 1: Remove vectors from ChromaDB
    delete_file_vectors(filename)

    # Step 2: Clear query cache
    clear_all_cache()

    # Step 3: Delete physical file
    os.remove(filepath)

    # Step 4: Remove from registry
    remove_from_registry(filename)

    return DeleteResponse(
        filename=filename,
        message=f"{filename} fully removed from system"
    )