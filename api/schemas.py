# ─────────────────────────────────────────────
# api/schemas.py
# Pydantic models define what data goes IN
# and what data comes OUT of each endpoint.
#
# FastAPI uses these to:
#   - Validate incoming requests automatically
#   - Generate API docs at /docs
#   - Show clear error messages if wrong data sent
# ─────────────────────────────────────────────

from pydantic import BaseModel
from typing import List, Optional


# ── Request Models (what client sends) ────────

class QueryRequest(BaseModel):
    """
    Body for POST /query
    question     → the user's question
    chat_history → list of previous messages
                   for conversation memory
    """
    question:     str
    chat_history: Optional[List[dict]] = []   # default empty list


# ── Response Models (what server returns) ─────

class SourceModel(BaseModel):
    """
    One source citation.
    file → which PDF the chunk came from
    page → which page number
    """
    file: str
    page: int


class QueryResponse(BaseModel):
    """
    Response for POST /query
    question       → echoed back for clarity
    answer         → Gemini's grounded answer
    sources        → list of source citations
    retriever_type → which path ran (normal/multiquery)
    from_cache     → was this served from cache?
    """
    question:       str
    answer:         str
    sources:        List[SourceModel]
    retriever_type: str
    from_cache:     bool


class DocumentModel(BaseModel):
    """
    One document entry.
    filename → name of the PDF
    chunks   → how many chunks stored in ChromaDB
    """
    filename: str
    chunks:   int


class DocumentListResponse(BaseModel):
    """
    Response for GET /docs
    documents → list of all loaded documents
    total     → total count
    """
    documents: List[DocumentModel]
    total:     int


class UploadResponse(BaseModel):
    """
    Response for POST /upload
    filename → which file was processed
    status   → new / updated / unchanged
    chunks   → how many chunks stored
    message  → human readable result
    """
    filename: str
    status:   str
    chunks:   int
    message:  str


class DeleteResponse(BaseModel):
    """
    Response for DELETE /docs/{filename}
    filename → which file was deleted
    message  → confirmation message
    """
    filename: str
    message:  str


class HealthResponse(BaseModel):
    """
    Response for GET /health
    Used by monitoring tools to check if
    the API is alive and responding.
    """
    status:   str
    version:  str
    docs_loaded: int