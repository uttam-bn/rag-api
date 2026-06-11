"""Unit tests for the RAG engine. Run: pytest"""
from app.rag import RagStore


def test_index_counts_chunks():
    store = RagStore(chunk_size=200, chunk_overlap=20)
    n = store.index(["FastAPI is a modern Python web framework for building APIs."])
    assert n >= 1
    assert store.chunks


def test_search_returns_relevant_chunk():
    store = RagStore()
    store.index([
        "FastAPI is a modern Python web framework for building APIs.",
        "FAISS is a library for efficient similarity search over vectors.",
        "Retrieval-augmented generation grounds an LLM in retrieved documents.",
    ])
    hits = store.search("What is FAISS used for?", k=1)
    assert hits and "FAISS" in hits[0]["text"]
