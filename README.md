# RAG API

A small, production-style **Retrieval-Augmented Generation** service built with
**LangChain** and **FastAPI**. Ingest your own documents, then ask questions and
get answers grounded in those documents — with the source chunks returned
alongside.

It runs **out of the box with no API keys and no model downloads** (LangChain
TF-IDF retriever + extractive answers), and upgrades cleanly to neural
embeddings and an LLM by changing one environment variable.

## Features
- **LangChain** throughout — `RecursiveCharacterTextSplitter`, retriever
  interface, and an **LCEL chain** (`prompt | model | parser`) for generation
- **FastAPI** service with typed Pydantic request/response models
- **Pluggable retrieval** — LangChain `TFIDFRetriever` by default, swap to a
  **FAISS** vector store with `sentence-transformers` via one env var
- **Pluggable LLM** — grounded answer via LangChain `ChatOpenAI` (OpenAI or
  Sarvam's OpenAI-compatible endpoint), or extractive fallback with no key
- **Sources returned** for every answer
- **Tests** included (`pytest`)

## Architecture
```
        ┌─────────┐   chunk + embed   ┌──────────────┐
docs ──▶│ /ingest │ ────────────────▶ │  vector index │
        └─────────┘                   └──────┬───────┘
                                             │ top-k similar
        ┌─────────┐   embed query            ▼
query ─▶│ /query  │ ──────────────▶  retrieve ──▶ LLM (optional) ──▶ answer + sources
        └─────────┘
```

## Quickstart
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
```
A sample knowledge base is indexed on startup, so you can query immediately.

## Example
```bash
# Ask a question
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does chunking affect in RAG?", "k": 2}'

# Index your own documents
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": ["your first document text", "your second document text"]}'
```

## Configuration
Copy `.env.example` to `.env`:
| Variable | Default | Purpose |
|---|---|---|
| `RAG_BACKEND` | `tfidf` | `tfidf` (no install) or `sentence-transformers` |
| `OPENAI_API_KEY` | – | generate answers with an OpenAI model |
| `SARVAM_API_KEY` | – | generate answers with a Sarvam model |

## Tests
```bash
pytest
```

## Tech
Python · LangChain (LCEL) · FastAPI · scikit-learn · NumPy · (optional) sentence-transformers, FAISS, OpenAI/Sarvam

## License
MIT
