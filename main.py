# ─────────────────────────────────────────────
# main.py
# Terminal testing tool for the RAG pipeline.
# Not for production — use app.py for that.
#
# Fixes applied:
#   1. chat_history=[] passed to ask()
#   2. Three layer check added (casual/no docs/RAG)
#   3. Source citations printed in terminal
#   4. Consistent with app.py behaviour
# ─────────────────────────────────────────────

import os
from dotenv import load_dotenv

from core.loader    import load_all_pdfs
from core.chunker   import chunk_documents
from core.registry  import (has_file_changed, update_registry,
                             remove_from_registry, load_registry)
from core.cache     import get_cached_answer, store_answer, clear_all_cache
from core.embedder  import embed_and_store, delete_file_vectors
from core.retriever import (ask,
                             is_casual_conversation,
                             get_casual_reply,
                             route_and_retrieve,
                             build_prompt)

load_dotenv()

DOCS_FOLDER = "documents"

# Chat history stored in memory for terminal session
# Resets every time you run main.py
# Same behaviour as clearing chat in app.py
chat_history = []


# ═══════════════════════════════════════════════
# SYNC DOCUMENTS
# ═══════════════════════════════════════════════

def sync_documents():
    """
    Scans documents folder.
    Handles new / edited / deleted files.
    Updates vectors and cache accordingly.
    Run this once at startup.
    """
    print("=== Syncing documents ===")

    current_files = {
        f for f in os.listdir(DOCS_FOLDER)
        if f.endswith(".pdf")
    }

    for filename in current_files:
        filepath = os.path.join(DOCS_FOLDER, filename)
        status   = has_file_changed(filename, filepath)

        if status == "new":
            print(f"NEW: {filename}")
            pages  = load_all_pdfs(DOCS_FOLDER)
            chunks = chunk_documents(pages)
            embed_and_store(chunks)
            update_registry(filename, filepath)

        elif status == "edited":
            print(f"EDITED: {filename}")
            delete_file_vectors(filename)
            clear_all_cache()
            pages  = load_all_pdfs(DOCS_FOLDER)
            chunks = chunk_documents(pages)
            embed_and_store(chunks)
            update_registry(filename, filepath)

        else:
            print(f"NO CHANGE: {filename} — cache valid")

    # Handle deleted files
    registry = load_registry()
    for filename in list(registry.keys()):
        if filename not in current_files:
            print(f"DELETED: {filename}")
            delete_file_vectors(filename)
            clear_all_cache()
            remove_from_registry(filename)

    print("=== Sync complete ===\n")


# ═══════════════════════════════════════════════
# QUERY FUNCTION
# ═══════════════════════════════════════════════

def query(user_question):
    """
    Complete query flow — matches app.py behaviour.

    Layer 1 → Casual conversation → warm reply
    Layer 2 → No docs loaded     → friendly nudge
    Layer 3 → Cache check        → instant if hit
    Layer 4 → Full RAG pipeline  → Gemini answer
    """

    print(f"\n{'='*40}")
    print(f"Query: {user_question}")
    print(f"{'='*40}")

    # ── Layer 1: Casual conversation ───────────
    # Same check as app.py
    # No ChromaDB. No Gemini. Free and instant.
    if is_casual_conversation(user_question):
        answer = get_casual_reply(user_question)
        print(f"💬 Conversation reply: {answer}")
        # Add to history but skip cache
        chat_history.append({"role": "user",    "content": user_question})
        chat_history.append({"role": "assistant","content": answer})
        return answer

    # ── Layer 2: Documents loaded? ─────────────
    docs_exist = any(
        f.endswith(".pdf")
        for f in os.listdir(DOCS_FOLDER)
    )
    if not docs_exist:
        answer = (
            "📂 No documents loaded.\n"
            "Add a PDF to the documents/ folder and restart."
        )
        print(answer)
        return answer

    # ── Layer 3: Cache check ───────────────────
    cached = get_cached_answer(user_question)
    if cached:
        print("⚡ Served from cache!")
        print(f"\nAnswer: {cached}")
        # Add to history
        chat_history.append({"role": "user",    "content": user_question})
        chat_history.append({"role": "assistant","content": cached})
        return cached

    # ── Layer 4: Full RAG pipeline ─────────────
    # Pass chat_history for conversation memory
    # Follow-up questions work correctly now
    chunks, retriever_type = route_and_retrieve(
        user_question,
        chat_history=chat_history        # ← FIX 1: memory added
    )

    # Build smart prompt with memory
    prompt = build_prompt(
        user_question,
        chunks,
        retriever_type,
        chat_history=chat_history        # ← FIX 1: memory added
    )

    # Send to Gemini via ask()
    answer = ask(
        user_question,
        chat_history=chat_history        # ← FIX 1: memory added
    )

    # Store in cache
    store_answer(user_question, answer)

    # FIX 2: Print source citations in terminal
    print(f"\nRetriever: {retriever_type}")
    print("\n📎 Sources:")
    seen = set()
    for chunk in chunks:
        key  = (
            chunk.metadata.get("source_file", "?"),
            chunk.metadata.get("page", "?")
        )
        if key not in seen:
            seen.add(key)
            print(f"  - {key[0]} → page {key[1]}")

    print(f"\nAnswer:\n{answer}")

    # FIX 3: Update chat history for next question
    chat_history.append({"role": "user",    "content": user_question})
    chat_history.append({"role": "assistant","content": answer})

    return answer


# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════

if __name__ == "__main__":

    # Step 1: Sync documents folder
    sync_documents()

    # Step 2: Test casual conversation
    # Should NOT hit ChromaDB or Gemini
    query("Hi")

    # Step 3: Test specific question
    # Cache miss → full RAG → answer + sources
    query("What is the probation period?")

    # Step 4: Same question again
    # Cache hit → instant answer
    query("What is the probation period?")

    # Step 5: Follow-up question
    # Memory working → "that" = probation period
    query("What happens after that?")

    # Step 6: Vague question
    # MultiQuery routing triggered
    query("Give me an overview of this document")


# ═══════════════════════════════════════════════
# RAG SYSTEM OVERVIEW
# ═══════════════════════════════════════════════

"""
Embedding         → Text to vector              (embedder.py)
ChromaDB          → Stores vectors on disk      (embedder.py)
Cosine similarity → Finds nearest chunks        (ChromaDB internal)
Chunk ID          → Unique label per chunk      (embedder.py)
MD5 hash          → Content fingerprint         (registry.py)
File registry     → Detects new/edit/delete     (registry.py)
Set math          → Finds what changed          (main.py)
Source metadata   → Tracks document source      (loader.py)
Query cache       → Saves answers on disk       (cache.py)
Cache clear       → Triggered on doc change     (cache.py)
Safe delete       → File + vector delete        (main.py)
Casual layer      → No API call for greetings   (retriever.py)
Memory            → Rewrites follow-up queries  (retriever.py)
Intent routing    → Vague vs specific           (retriever.py)
MultiQuery        → Multiple search angles      (retriever.py)
Source citations  → Which file + page           (main.py)
Separate files    → One job per file            (all files)
"""