# ─────────────────────────────────────────────
# app.py — Streamlit UI
# The complete web interface for the RAG pipeline.
#
# Run with: streamlit run app.py
#
# Three layer check on every user message:
#   Layer 1 → Casual chat?    → reply warmly
#   Layer 2 → Docs loaded?    → friendly nudge
#   Layer 3 → Document query  → full RAG pipeline
# ─────────────────────────────────────────────

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from core.loader    import load_all_pdfs
from core.chunker   import chunk_documents
from core.embedder  import embed_and_store, delete_file_vectors
from core.registry  import (has_file_changed, update_registry,
                             remove_from_registry)
from core.cache     import get_cached_answer, store_answer, clear_all_cache
from core.retriever import (route_and_retrieve, build_prompt,
                             is_casual_conversation, get_casual_reply)

load_dotenv()

DOCS_FOLDER = "documents"
os.makedirs(DOCS_FOLDER, exist_ok=True)


# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)


# ═══════════════════════════════════════════════
# LLM — initialised once, reused across queries
# ═══════════════════════════════════════════════

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# ═══════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════

# Chat history persists across Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = []


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def process_uploaded_file(uploaded_file):
    """
    Save uploaded PDF to disk and embed it.
    Handles new, edited, and unchanged files.
    Returns a status message for the UI.
    """
    filepath = os.path.join(DOCS_FOLDER, uploaded_file.name)

    # Save raw bytes to disk
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Check file status via registry
    status = has_file_changed(uploaded_file.name, filepath)

    if status == "new":
        with st.spinner(f"Embedding {uploaded_file.name}..."):
            pages  = load_all_pdfs(DOCS_FOLDER)
            chunks = chunk_documents(pages)
            embed_and_store(chunks)
            update_registry(uploaded_file.name, filepath)
        return f"✅ {uploaded_file.name} — {len(chunks)} chunks stored"

    elif status == "edited":
        with st.spinner(f"Re-embedding {uploaded_file.name}..."):
            delete_file_vectors(uploaded_file.name)
            clear_all_cache()
            pages  = load_all_pdfs(DOCS_FOLDER)
            chunks = chunk_documents(pages)
            embed_and_store(chunks)
            update_registry(uploaded_file.name, filepath)
        return f"🔄 {uploaded_file.name} — updated and re-embedded"

    else:
        return f"ℹ️ {uploaded_file.name} — already loaded, no changes"


def delete_document(filename):
    """
    Remove document completely:
    vectors + cache + file + registry.
    """
    delete_file_vectors(filename)
    clear_all_cache()

    filepath = os.path.join(DOCS_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    remove_from_registry(filename)
    return f"🗑️ {filename} removed from system"


def extract_sources(chunks):
    """
    Pull unique source + page from retrieved chunks.
    Used for citation display in the UI.
    """
    sources = []
    seen    = set()
    for chunk in chunks:
        key = (
            chunk.metadata.get("source_file", "?"),
            chunk.metadata.get("page", "?")
        )
        if key not in seen:
            seen.add(key)
            sources.append({"file": key[0], "page": key[1]})
    return sources


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

with st.sidebar:
    st.title("📚 Document Manager")
    st.divider()

    # ── Upload ─────────────────────────────────
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for f in uploaded_files:
            result = process_uploaded_file(f)
            st.success(result)

    st.divider()

    # ── Loaded documents ───────────────────────
    st.subheader("Loaded Documents")
    current_docs = [
        f for f in os.listdir(DOCS_FOLDER)
        if f.endswith(".pdf")
    ]

    if current_docs:
        for doc in current_docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"📄 {doc}")
            with col2:
                if st.button("🗑️", key=f"del_{doc}"):
                    msg = delete_document(doc)
                    st.toast(msg)
                    st.rerun()
    else:
        st.info("No documents loaded.\nUpload a PDF to start.")

    st.divider()

    # ── Controls ───────────────────────────────
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Clear Answer Cache"):
        clear_all_cache()
        st.toast("Cache cleared!")


# ═══════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════

st.title("🤖 RAG Document Assistant")
st.caption("Upload PDFs in the sidebar → Ask questions below")

# ── Render chat history ────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources only for assistant messages
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"- **{src['file']}** — Page {src['page']}"
                    )

# ── Chat input ─────────────────────────────────
user_input = st.chat_input("Ask a question or say hi...")

if user_input:

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role"   : "user",
        "content": user_input
    })

    # ── Generate response ──────────────────────
    with st.chat_message("assistant"):

        sources = []     # default empty

        # ── LAYER 1: Casual conversation ───────
        # Greetings, chit-chat, capability questions
        # No ChromaDB. No Gemini. Free and instant.
        if is_casual_conversation(user_input):
            answer = get_casual_reply(user_input)
            st.markdown(answer)
            st.caption("💬 Conversation")

        else:
            # ── LAYER 2: Documents loaded? ──────
            # Only check docs if it's a real question
            docs_exist = any(
                f.endswith(".pdf")
                for f in os.listdir(DOCS_FOLDER)
            )

            if not docs_exist:
                answer = (
                    "📂 No documents loaded yet!\n\n"
                    "Upload a PDF in the left sidebar and "
                    "I'll be ready to answer your questions."
                )
                st.markdown(answer)

            else:
                # ── LAYER 3: Full RAG pipeline ──
                with st.spinner("Searching documents..."):

                    # Check cache first
                    cached = get_cached_answer(user_input)

                    if cached:
                        # Cache hit — instant answer
                        answer = cached
                        st.markdown(answer)
                        st.caption("⚡ Answered from cache")

                    else:
                        # Cache miss — full RAG chain
                        # Intent check → right retriever
                        chunks, retriever_type = route_and_retrieve(
                            user_input
                        )

                        # Build smart prompt
                        prompt   = build_prompt(
                            user_input, chunks, retriever_type
                        )

                        # Send to Gemini
                        response = llm.invoke(prompt)
                        answer   = response.content

                        # Extract sources for citations
                        sources = extract_sources(chunks)

                        # Save to cache
                        store_answer(user_input, answer)

                        # Show answer
                        st.markdown(answer)

                        # Show retriever badge
                        badge = (
                            "🔍 MultiQuery"
                            if retriever_type == "multiquery"
                            else "⚡ Normal"
                        )
                        st.caption(f"{badge} retrieval")

                        # Show source citations
                        if sources:
                            with st.expander("📎 Sources"):
                                for src in sources:
                                    st.markdown(
                                        f"- **{src['file']}** "
                                        f"— Page {src['page']}"
                                    )

    # Save assistant message to history
    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "sources": sources
    })