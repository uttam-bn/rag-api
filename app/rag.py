"""Core RAG engine, built with LangChain.

Uses LangChain's text splitter and retriever interface so the same pipeline
scales from a zero-setup TF-IDF retriever (no downloads) up to neural
embeddings with a FAISS vector store — just by changing RAG_BACKEND.
"""
from __future__ import annotations

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RagStore:
    """Indexes documents and retrieves the most relevant chunks for a query."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.backend_name = os.getenv("RAG_BACKEND", "tfidf")
        self.chunks: list[str] = []
        self._retriever = None

    def index(self, documents: list[str]) -> int:
        """Split documents into chunks and build a retriever. Returns chunk count."""
        docs = [Document(page_content=d) for d in documents if d and d.strip()]
        if not docs:
            raise ValueError("No text to index.")
        splits = self.splitter.split_documents(docs)
        self.chunks = [s.page_content for s in splits]
        self._retriever = self._build_retriever(splits)
        return len(self.chunks)

    def _build_retriever(self, splits: list[Document]):
        if self.backend_name == "sentence-transformers":
            # Neural embeddings + FAISS vector store (optional extras).
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            return FAISS.from_documents(splits, embeddings).as_retriever()
        # Default: TF-IDF retriever — fast, no model download.
        from langchain_community.retrievers import TFIDFRetriever
        return TFIDFRetriever.from_documents(splits)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k most relevant chunks for the query."""
        if self._retriever is None:
            raise RuntimeError("Index is empty. Call index() first.")
        try:
            self._retriever.k = k
        except Exception:
            pass
        docs = self._retriever.invoke(query)[:k]
        return [{"text": d.page_content} for d in docs]
