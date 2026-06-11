"""Request/response models for the RAG API."""
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    documents: list[str] = Field(..., description="Raw text documents to index.")


class IngestResponse(BaseModel):
    chunks_indexed: int
    backend: str


class QueryRequest(BaseModel):
    question: str
    k: int = Field(3, ge=1, le=10, description="How many chunks to retrieve.")


class Source(BaseModel):
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
