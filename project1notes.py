# ═══════════════════════════════════════════════
# APP.PY — COMPLETE FLOW OVERVIEW
# ═══════════════════════════════════════════════

"""
─────────────────────────────────────────────────
ARCHITECTURE
─────────────────────────────────────────────────
app.py              → Streamlit UI (customer face)
core/loader.py      → PDF reading (postman)
core/chunker.py     → Text splitting (scissors)
core/embedder.py    → Vectors + ChromaDB (translator + filing cabinet)
core/registry.py    → File hash tracking (notebook)
core/cache.py       → Answer caching (sticky notes)
core/retriever.py   → All retrieval logic (brain)
api/main.py         → FastAPI REST backend (production API)

─────────────────────────────────────────────────
PAGE SETUP
─────────────────────────────────────────────────
st.set_page_config  → Sets browser tab title, icon, layout
layout="wide"       → Uses full screen width
Must be FIRST       → Streamlit rule — before any other st. call

─────────────────────────────────────────────────
LLM INITIALISATION
─────────────────────────────────────────────────
ChatGoogleGenerativeAI  → LangChain wrapper for Gemini
temperature=0           → Deterministic answers, no creativity
                          Facts only — good for document Q&A
Initialised ONCE        → Reused across all queries
                          Saves connection overhead

─────────────────────────────────────────────────
SESSION STATE
─────────────────────────────────────────────────
st.session_state        → Persists data across Streamlit reruns
                          Streamlit reruns entire script on every
                          user interaction — session_state keeps
                          data alive between reruns
messages list           → Stores all chat turns
                          Format: {role, content, sources}
                          role    = "user" or "assistant"
                          content = message text
                          sources = list of {file, page}

─────────────────────────────────────────────────
PROCESS_UPLOADED_FILE()
─────────────────────────────────────────────────
uploaded_file.getbuffer() → raw bytes of PDF
open(filepath, "wb")      → write bytes to disk
has_file_changed()        → registry.py checks MD5 hash
                            new     → embed and store
                            edited  → delete old vectors
                                      clear cache
                                      re-embed fresh
                            same    → skip, already loaded
load_all_pdfs()           → loader.py reads all PDFs in folder
chunk_documents()         → chunker.py splits into 1000 char chunks
embed_and_store()         → embedder.py converts to vectors
                            stores in ChromaDB on disk
update_registry()         → registry.py saves new file hash

─────────────────────────────────────────────────
DELETE_DOCUMENT()
─────────────────────────────────────────────────
delete_file_vectors()   → embedder.py removes chunks from ChromaDB
                          finds all IDs starting with filename
                          deletes them surgically
clear_all_cache()       → cache.py deletes query_cache.json
                          all cached answers wiped
                          old answers may reference deleted doc
os.remove(filepath)     → physical PDF deleted from disk
remove_from_registry()  → registry.py removes file hash entry
                          next sync won't try to process it

─────────────────────────────────────────────────
EXTRACT_SOURCES()
─────────────────────────────────────────────────
chunk.metadata          → each chunk carries source_file + page
                          stamped in loader.py when PDF loaded
seen set                → deduplicates sources
                          same page referenced by multiple chunks
                          only shown once in citations
returns list            → [{file: "policy.pdf", page: 3}, ...]
                          displayed in expandable Sources section

─────────────────────────────────────────────────
SIDEBAR
─────────────────────────────────────────────────
st.file_uploader        → accepts multiple PDFs at once
                          type=["pdf"] restricts to PDFs only
                          accept_multiple_files=True
                          each file processed independently

st.columns([3,1])       → splits row into two columns
                          col1 → filename display (3/4 width)
                          col2 → delete button (1/4 width)

st.button("🗑️")        → unique key per document
                          key=f"del_{doc}" prevents conflicts
                          when multiple docs loaded

st.toast()              → popup notification at bottom right
st.rerun()              → refreshes entire Streamlit app
                          sidebar updates immediately after delete

─────────────────────────────────────────────────
CHAT HISTORY RENDERING
─────────────────────────────────────────────────
for message in messages → loops all previous turns
st.chat_message(role)   → renders user/assistant bubble
                          user      → right aligned
                          assistant → left aligned
message.get("sources")  → only assistant messages have sources
st.expander()           → collapsible Sources section
                          keeps UI clean
                          user opens only if they want citations

─────────────────────────────────────────────────
CHAT INPUT
─────────────────────────────────────────────────
st.chat_input()         → text box fixed at bottom of screen
                          returns None until user submits
                          triggers rerun when submitted

─────────────────────────────────────────────────
THREE LAYER CHECK
─────────────────────────────────────────────────
Layer 1 → CASUAL CONVERSATION
  is_casual_conversation()  → keyword check on question
                               matches greetings, thanks, bye
                               short messages (2 words or less)
  get_casual_reply()        → pattern match → warm response
                               NO ChromaDB search
                               NO Gemini API call
                               completely FREE and instant
  st.caption("💬 Conversation") → badge shows conversation mode

Layer 2 → DOCUMENTS LOADED CHECK
  os.listdir(DOCS_FOLDER)   → scans documents/ folder
  any(.endswith(".pdf"))    → checks at least one PDF exists
  if not docs_exist         → friendly nudge to upload
                               not an error message
                               guides user naturally

Layer 3 → FULL RAG PIPELINE
  get_cached_answer()       → cache.py checks query_cache.json
                               MD5 hash of question = cache key
                               HIT  → instant answer, zero cost
                               MISS → continue to retrieval

  route_and_retrieve()      → retriever.py
                               rewrites question using memory
                               detects intent (vague/specific)
                               VAGUE    → MultiQuery retriever
                                          LLM generates 3 variations
                                          3 ChromaDB searches
                                          results merged + deduped
                               SPECIFIC → Normal retriever
                                          1 ChromaDB search
                                          top 3 chunks returned

  build_prompt()            → retriever.py
                               formats chunks as context
                               adds conversation history
                               MULTIQUERY → synthesis instruction
                               NORMAL     → strict factual instruction

  llm.invoke(prompt)        → sends to Gemini
                               gets grounded answer back
                               answer based ONLY on chunks
                               not Gemini's general knowledge

  extract_sources()         → pulls unique file + page from chunks
                               shown in expandable Sources section

  store_answer()            → cache.py saves to query_cache.json
                               next same question = cache hit

  badge                     → "🔍 MultiQuery" or "⚡ Normal"
                               shown below answer
                               tells user which path ran

─────────────────────────────────────────────────
CHAT HISTORY UPDATE
─────────────────────────────────────────────────
messages.append()       → saves every turn to session_state
                          user message    → saved before processing
                          assistant reply → saved after processing
                          sources         → saved with assistant msg
                          history passed to route_and_retrieve()
                          and build_prompt() on next question
                          enables follow-up questions to work

─────────────────────────────────────────────────
MEMORY — HOW IT WORKS
─────────────────────────────────────────────────
chat_history passed to route_and_retrieve()
  → build_context_aware_query() in retriever.py
  → takes last 3 turns (6 messages)
  → asks Gemini to rewrite follow-up as standalone
  → "What happens after that?"
    becomes
    "What happens after the 6 month probation period?"
  → ChromaDB searches the rewritten query
  → finds correct chunks

chat_history passed to build_prompt()
  → included in prompt as CONVERSATION HISTORY section
  → Gemini sees full context when answering
  → pronouns and references resolved correctly

─────────────────────────────────────────────────
COST PROFILE
─────────────────────────────────────────────────
Casual message          → $0.00 (no API)
Cache hit               → $0.00 (no API)
Specific question miss  → 1 Gemini call
Vague question miss     → 2 Gemini calls (variations + answer)
Follow-up question      → 1 extra Gemini call (rewrite)
Upload new PDF          → N embedding calls (N = chunk count)
Upload unchanged PDF    → $0.00 (skipped)
Delete document         → $0.00 (local operation)

─────────────────────────────────────────────────
WHAT MAKES THIS PRODUCTION GRADE
─────────────────────────────────────────────────
Incremental embedding   → only new/changed chunks embedded
Content hashing         → edits auto detected via MD5
Atomic delete           → vectors + cache + file + registry
Query cache             → repeated questions cost nothing
Conversation memory     → follow-up questions work correctly
Intent routing          → right retriever for right question
Source citations        → every answer traceable to page
Three layer check       → graceful handling of all edge cases
Separate responsibilities → one job per file
"""