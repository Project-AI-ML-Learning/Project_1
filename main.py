# main.py
# The ONLY file you run.
# Think of it like a TV remote —
# one button, everything happens behind the scenes.
#
# WHAT CHANGED FROM PREVIOUS VERSION:
# → Added: from core.retriever import ask
# → query() now calls ask() instead of placeholder chunk preview
# → Everything else is identical

import os
from core.loader    import load_all_pdfs
from core.chunker   import chunk_documents
from core.registry  import (has_file_changed, update_registry,
                             remove_from_registry, load_registry)
from core.cache     import get_cached_answer, store_answer, clear_all_cache
from core.embedder  import embed_and_store, delete_file_vectors, get_vectorstore
from core.retriever import ask                        # ← ONLY NEW IMPORT

DOCS_FOLDER = "documents"


def sync_documents():
    """
    Scans documents folder.
    Handles new / edited / deleted files.
    Updates vectors and cache accordingly.
    NO CHANGES from previous version.
    """
    print("=== Syncing documents ===")

    current_files = {
        f for f in os.listdir(DOCS_FOLDER)
        if f.endswith(".pdf")
    }

    for filename in current_files:
        filepath = os.path.join(DOCS_FOLDER, filename)
        status   = has_file_changed(filename, filepath)   # registry.py

        if status == "new":
            print(f"NEW: {filename}")
            pages  = load_all_pdfs(DOCS_FOLDER)           # loader.py
            chunks = chunk_documents(pages)                # chunker.py
            embed_and_store(chunks)                        # embedder.py
            update_registry(filename, filepath)            # registry.py

        elif status == "edited":
            print(f"EDITED: {filename}")
            delete_file_vectors(filename)                  # embedder.py
            clear_all_cache()                              # cache.py
            pages  = load_all_pdfs(DOCS_FOLDER)           # loader.py
            chunks = chunk_documents(pages)                # chunker.py
            embed_and_store(chunks)                        # embedder.py
            update_registry(filename, filepath)            # registry.py

        else:
            print(f"NO CHANGE: {filename} — cache valid")

    # Handle deleted files
    registry = load_registry()
    for filename in list(registry.keys()):
        if filename not in current_files:
            print(f"DELETED: {filename}")
            delete_file_vectors(filename)                  # embedder.py
            clear_all_cache()                              # cache.py
            remove_from_registry(filename)                 # registry.py


def query(user_question):
    """
    Complete query flow.

    PREVIOUS VERSION:
      → searched ChromaDB manually
      → returned raw chunk text as placeholder

    CURRENT VERSION:
      → cache check first (same as before)
      → on miss: calls ask() from retriever.py
        which does search + prompt + Gemini
      → stores real Gemini answer in cache
    """

    print(f"\n{'='*40}")
    print(f"Query: {user_question}")
    print(f"{'='*40}")

    # Step 1: Check cache first — no change from before
    # Same question asked before? Return instantly.
    # Zero ChromaDB search. Zero Gemini API call.
    cached = get_cached_answer(user_question)              # cache.py
    if cached:
        print("Served from cache!")
        return cached

    # Step 2: Cache miss — THIS IS WHAT CHANGED
    # Before → manual search + raw chunk text returned
    # Now    → ask() handles everything cleanly:
    #           retrieve chunks → build prompt → Gemini → answer
    answer = ask(user_question)                            # retriever.py

    # Step 3: Store real answer in cache — no change
    store_answer(user_question, answer)                    # cache.py

    return answer


if __name__ == "__main__":

    # Sync documents folder with vector store
    sync_documents()

    # Test 1: First time → cache miss → full RAG → Gemini answers
    print(query("What is this document about?"))

    # Test 2: Same question → cache hit → instant return
    print(query("What is this document about?"))

    # Test 3: New question → cache miss → full RAG again
    print(query("Who is the president of india"))


"""
# RAG SYSTEM OVERVIEW

# Embedding         → Text to vector              (embedder.py)
# ChromaDB          → Stores vectors on disk      (embedder.py)
# Cosine similarity → Finds nearest chunks        (ChromaDB internal)
# Chunk ID          → Unique label per chunk      (embedder.py)
# MD5 hash          → Content fingerprint         (registry.py)
# File registry     → Detects new/edit/delete     (registry.py)
# Set math          → Finds what changed          (main.py)
# Source metadata   → Tracks document source      (loader.py)
# Query cache       → Saves answers on disk       (cache.py)
# Cache clear       → Triggered on doc change     (cache.py)
# Safe delete       → File + vector delete        (main.py)
# Separate files    → One job per file            (all files)

"""
