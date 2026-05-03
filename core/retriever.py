# ─────────────────────────────────────────────
# core/retriever.py
# Handles all retrieval logic:
#   - Casual conversation detection
#   - Conversation memory (context aware query)
#   - Intent routing (vague vs specific)
#   - Normal retriever
#   - Manual MultiQuery retriever
#   - Prompt building with memory
#   - Main ask() function
# ─────────────────────────────────────────────

from dotenv import load_dotenv
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from core.embedder import get_vectorstore

load_dotenv()

# ── LLM setup ─────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0
)


# ═══════════════════════════════════════════════
# SECTION 1: CASUAL CONVERSATION
# ═══════════════════════════════════════════════

CASUAL_TRIGGERS = [
    "hi", "hello", "hey", "good morning",
    "good afternoon", "good evening", "howdy",
    "what's up", "whats up", "how are you",
    "who are you", "what are you", "what can you do",
    "help", "thanks", "thank you", "bye", "goodbye",
    "ok", "okay", "cool", "got it", "nice", "great"
]

def is_casual_conversation(question):
    """
    Check if user is chatting casually.
    Two checks:
    1. Matches casual trigger word
    2. Message is 2 words or less
    Returns True if casual, False if document query.
    """
    q_clean = question.strip().lower()

    for trigger in CASUAL_TRIGGERS:
        if q_clean == trigger or q_clean.startswith(trigger):
            return True

    # Very short message — likely casual
    if len(q_clean.split()) <= 2:
        return True

    return False


def get_casual_reply(question):
    """
    Return warm friendly reply for casual messages.
    No ChromaDB. No Gemini. Free and instant.
    """
    q = question.strip().lower()

    if any(g in q for g in ["hi", "hello", "hey", "howdy"]):
        return (
            "Hello! 👋 I'm your document assistant. "
            "Upload a PDF and ask me anything about it!"
        )

    if any(g in q for g in ["good morning", "good afternoon", "good evening"]):
        return "Good day! 😊 Ready to help. What would you like to know?"

    if "how are you" in q:
        return (
            "I'm doing great and ready to help! 🚀 "
            "Upload a document and fire away your questions."
        )

    if any(g in q for g in ["what are you", "who are you", "what can you do"]):
        return (
            "I'm a RAG-powered document assistant! 🤖\n\n"
            "Here's what I can do:\n"
            "- 📄 Read and understand your PDF documents\n"
            "- 🔍 Answer specific questions with page citations\n"
            "- 📋 Summarise entire documents\n"
            "- 🧠 Remember our conversation for follow-up questions\n"
            "- ⚡ Cache answers so repeated questions are instant\n"
            "- 🔀 Auto-detect vague vs specific questions\n\n"
            "Upload a PDF to get started!"
        )

    if "help" in q:
        return (
            "Here's how to use me:\n\n"
            "1️⃣ Upload PDFs via POST /upload\n"
            "2️⃣ Ask any question via POST /query\n"
            "3️⃣ Ask follow-up questions — I remember context!\n"
            "4️⃣ Sources show exactly which page I used"
        )

    if any(g in q for g in ["thank", "thanks"]):
        return "You're welcome! 😊 Ask me anything else."

    if any(g in q for g in ["bye", "goodbye"]):
        return "Goodbye! 👋 Come back anytime."

    if any(g in q for g in ["ok", "okay", "cool", "got it", "nice", "great"]):
        return "Got it! Feel free to ask anything. 😊"

    return (
        "I'm here to help with your documents! 📚 "
        "Try asking a question about an uploaded PDF."
    )


# ═══════════════════════════════════════════════
# SECTION 2: CONVERSATION MEMORY
# ═══════════════════════════════════════════════

def build_context_aware_query(question, chat_history):
    """
    THE MEMORY FUNCTION.

    Problem:
      User asks "What happens after that?"
      ChromaDB has no idea what "that" means.

    Solution:
      Take last 3 turns + current question
      Ask Gemini to rewrite as standalone question.

    Example:
      History : "What is probation period?" → "6 months"
      Current : "What happens after that?"
      Rewritten: "What happens after the 6 month probation?"

    If no history → return question unchanged.
    """

    # No history yet — first question
    if not chat_history:
        return question

    # Take only last 6 messages (3 turns)
    recent_history = chat_history[-6:]

    # Format as readable conversation
    history_text = ""
    for msg in recent_history:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]    # trim long answers
        history_text += f"{role}: {content}\n"

    # Ask Gemini to rewrite
    rewrite_prompt = f"""Given this conversation history:

{history_text}

Rewrite this follow-up question as a complete standalone 
question that can be understood without the conversation history.
If the question is already standalone, return it as-is.

Follow-up question: {question}

Rewritten question (return ONLY the question, nothing else):"""

    response  = llm.invoke(rewrite_prompt)
    rewritten = response.content.strip()

    print(f"Original  query : {question}")
    print(f"Rewritten query : {rewritten}")

    return rewritten


# ═══════════════════════════════════════════════
# SECTION 3: INTENT DETECTION
# ═══════════════════════════════════════════════

VAGUE_TRIGGERS = [
    "summary", "summarise", "summarize",
    "conclusion", "conclusions", "overview",
    "main points", "key points", "key conclusions",
    "what is this", "what does this document",
    "tell me about", "explain this",
    "give me an idea", "broad", "general",
    "overall", "highlight", "brief"
]

def detect_intent(question):
    """
    Classify question as vague or specific.
    Vague    → MultiQuery (broad context)
    Specific → Normal retriever (precise)
    Keyword check — fast, free, no API call.
    """
    q_lower = question.lower()

    for trigger in VAGUE_TRIGGERS:
        if trigger in q_lower:
            print(f"Intent: VAGUE (trigger: '{trigger}')")
            return "vague"

    print("Intent: SPECIFIC")
    return "specific"


# ═══════════════════════════════════════════════
# SECTION 4: RETRIEVERS
# ═══════════════════════════════════════════════

def get_normal_retriever(k=3):
    """
    Standard ChromaDB retriever.
    One question → one vector → top k chunks.
    Fast. Cheap. Precise.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )


def multi_query_retrieve(question, k=4):
    """
    Manual MultiQuery — no LangChain dependency.

    How it works:
    1. Ask Gemini to generate 3 query variations
    2. Search ChromaDB with each variation
    3. Merge all results
    4. Deduplicate by content
    5. Return unique chunks

    Covers multiple angles of the same question.
    Gets broader context for vague questions.
    """
    vectorstore = get_vectorstore()

    # Step 1: Generate query variations
    prompt = f"""Generate 3 different versions of this question
to improve document retrieval.

Question: {question}

Return each variation on a new line. No numbering. No bullets."""

    response = llm.invoke(prompt)

    # Clean up generated queries
    # Remove numbering like "1." "2." "1)" etc
    queries = [
        re.sub(r"^\d+[\.\)]\s*", "", q.strip())
        for q in response.content.split("\n")
        if q.strip() and len(q.strip()) > 10
    ]

    print("\nGenerated queries:")
    for q in queries:
        print(f"  - {q}")

    # Step 2: Search ChromaDB with each query
    all_docs = []
    for q in queries:
        docs = vectorstore.similarity_search(q, k=k)
        all_docs.extend(docs)

    # Step 3: Deduplicate by content
    # Using dict — same content = same key = overwrites
    unique_docs = {}
    for doc in all_docs:
        unique_docs[doc.page_content] = doc

    print(f"MultiQuery retrieved {len(unique_docs)} unique chunks")
    return list(unique_docs.values())


# ═══════════════════════════════════════════════
# SECTION 5: SMART ROUTER
# ═══════════════════════════════════════════════

def route_and_retrieve(question, chat_history=None):
    """
    Core routing function.

    Step 1 → Rewrite question using memory
    Step 2 → Detect intent (vague or specific)
    Step 3 → Route to right retriever
    Step 4 → Return (chunks, retriever_type)

    chat_history is optional — defaults to None
    for terminal use via ask()
    """

    # Step 1: Rewrite question with memory context
    # "What happens after that?" becomes
    # "What happens after the probation period?"
    search_query = build_context_aware_query(
        question, chat_history or []
    )

    # Step 2: Detect intent on REWRITTEN query
    intent = detect_intent(search_query)

    if intent == "vague":
        print("Routing → MultiQuery")
        chunks = multi_query_retrieve(search_query, k=4)
        return chunks, "multiquery"

    else:
        print("Routing → Normal Retriever")
        retriever = get_normal_retriever(k=3)
        chunks    = retriever.invoke(search_query)
        print(f"Normal retriever got {len(chunks)} chunks")
        return chunks, "normal"


# ═══════════════════════════════════════════════
# SECTION 6: PROMPT BUILDER WITH MEMORY
# ═══════════════════════════════════════════════

def build_prompt(question, chunks, retriever_type, chat_history=None):
    """
    Build smart prompt based on retriever type.
    Includes conversation history for memory.

    Two memory points:
      build_context_aware_query → fixes RETRIEVAL
      build_prompt with history → fixes ANSWERING

    Both needed for full memory support.
    """

    # Format chunks with source info
    context = "\n\n---\n\n".join([
        f"Source: {chunk.metadata.get('source_file', 'unknown')} "
        f"| Page: {chunk.metadata.get('page', '?')}\n"
        f"{chunk.page_content}"
        for chunk in chunks
    ])

    # Format recent chat history
    # Only last 3 turns — keep prompt short
    history_text   = ""
    history_section = ""
    if chat_history:
        recent = chat_history[-6:]
        for msg in recent:
            role    = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:300]
            history_text += f"{role}: {content}\n"

        history_section = f"""CONVERSATION HISTORY (for context):
{history_text}

"""

    # Switch instruction based on retriever type
    if retriever_type == "multiquery":
        instruction = """You are a helpful document assistant with memory.
Use the conversation history to understand context.
Provide a clear structured summary using the document 
context below. Use headings and bullet points."""

    else:
        instruction = """You are a helpful document assistant with memory.
Use the conversation history to understand context.
Answer strictly based on the document context below.
If not found say:
"I could not find this in the provided documents."
Do NOT use outside knowledge."""

    return f"""{instruction}

{history_section}DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{question}

ANSWER:"""


# ═══════════════════════════════════════════════
# SECTION 7: MAIN ASK FUNCTION
# ═══════════════════════════════════════════════

def ask(question, chat_history=None):
    """
    Full RAG chain with memory.
    Used by main.py for terminal testing.
    app.py calls route_and_retrieve() directly
    to get sources for UI citations.
    """
    print(f"\nQuestion: {question}")
    print("-" * 40)

    chunks, retriever_type = route_and_retrieve(
        question, chat_history
    )

    print("\nChunks retrieved:")
    for i, chunk in enumerate(chunks):
        src  = chunk.metadata.get("source_file", "?")
        page = chunk.metadata.get("page", "?")
        print(f"  {i+1}. [{src}] page {page}")

    prompt   = build_prompt(
        question, chunks, retriever_type, chat_history
    )
    response = llm.invoke(prompt)
    answer   = response.content

    print(f"\nAnswer:\n{answer}")
    return answer