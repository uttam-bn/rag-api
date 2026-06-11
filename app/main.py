"""RAG API — a FastAPI service for retrieval-augmented question answering.

Endpoints:
  GET  /health   -> service + index status
  POST /ingest   -> index your own documents
  POST /query    -> ask a question; returns a grounded answer + sources
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .llm import generate_answer
from .rag import RagStore
from .schemas import (IngestRequest, IngestResponse, QueryRequest,
                      QueryResponse, Source)

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="A small, production-style Retrieval-Augmented Generation API built with LangChain.",
)
store = RagStore()


@app.on_event("startup")
def _load_sample() -> None:
    """Index a sample document so /query works immediately after startup."""
    sample = Path(__file__).resolve().parent.parent / "data" / "sample.md"
    if sample.exists():
        store.index([sample.read_text(encoding="utf-8")])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "indexed_chunks": len(store.chunks),
        "backend": store.backend_name,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        n = store.index(req.documents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IngestResponse(chunks_indexed=n, backend=store.backend_name)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not store.chunks:
        raise HTTPException(status_code=400, detail="No documents indexed. POST /ingest first.")
    hits = store.search(req.question, k=req.k)
    answer = generate_answer(req.question, [h["text"] for h in hits])
    return QueryResponse(answer=answer, sources=[Source(**h) for h in hits])
