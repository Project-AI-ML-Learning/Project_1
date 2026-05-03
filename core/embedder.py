# embedder.py
# TWO jobs in one file:
# Translator : converts text chunks into numbers (vectors)
# Filing cab : stores those numbers in ChromaDB

import hashlib
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR      = "chroma_db"
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

def get_vectorstore():
    # Open the filing cabinet (existing ChromaDB)
    # Never recreate — always open what exists
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )


def make_chunk_ids(chunks):
    # Give every chunk a unique label
    # Label = filename + position + content fingerprint
    return [
        f"{c.metadata['source_file']}__{i}__{hashlib.md5(c.page_content.encode()).hexdigest()[:8]}"
        for i, c in enumerate(chunks)
    ]


def embed_and_store(chunks):
    # Translate chunks → vectors
    # Store vectors in filing cabinet
    vectorstore = get_vectorstore()
    ids         = make_chunk_ids(chunks)
    vectorstore.add_documents(documents=chunks, ids=ids)
    print(f"Stored {len(chunks)} chunks in ChromaDB.")


def delete_file_vectors(filename):
    # Pull out ALL filing cards belonging to this file
    vectorstore = get_vectorstore()
    existing    = vectorstore.get()
    to_delete   = [
        cid for cid in existing["ids"]
        if cid.startswith(filename)
    ]
    if to_delete:
        vectorstore.delete(ids=to_delete)
        print(f"Deleted {len(to_delete)} vectors for '{filename}'")

"""Translator converts English words into a secret number code that computers understand. Filing cabinet stores those codes with labels. When you search — it finds the closest matching codes and gives you the original words back."""