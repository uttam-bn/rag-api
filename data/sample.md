# Retrieval-Augmented Generation (sample knowledge base)

Retrieval-Augmented Generation (RAG) is a technique that grounds a large
language model in external documents. Instead of relying only on what the model
memorised during training, the system first retrieves relevant passages from a
knowledge base and passes them to the model as context. This reduces
hallucination and lets the model answer using private or up-to-date data.

A typical RAG pipeline has four stages. First, documents are split into smaller
chunks so retrieval is precise. Second, each chunk is converted into a vector
embedding. Third, the embeddings are stored in a vector index for fast
similarity search. Fourth, at query time the question is embedded, the most
similar chunks are retrieved, and the language model generates an answer from
them.

Chunking strategy matters. Chunks that are too large add noise and dilute
relevance, while chunks that are too small lose context. Overlapping chunks help
preserve meaning across boundaries.

Vector stores differ by scale. FAISS is a fast local library with no server.
Chroma is convenient for local development. Pinecone is a managed, scalable
service for production workloads.

FastAPI is a modern Python web framework for building APIs. It uses type hints
and Pydantic models to validate requests and responses automatically, and it
supports asynchronous request handling for I/O-bound workloads such as calling a
database or a language model.
