"""Loads the persisted Chroma vector store (built by notebooks/rag_pipeline.ipynb)
and exposes a retrieve() function. Loaded once at app startup, not per-request.
"""
import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.vector_store_path)
        self.collection = self.client.get_collection(name=settings.collection_name)
        self.embedder = SentenceTransformer(settings.embedding_model)

    @property
    def num_chunks(self) -> int:
        return self.collection.count()

    def retrieve(self, question: str, top_k: int | None = None):
        k = top_k or settings.top_k
        q_emb = self.embedder.encode([question]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=k)
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"text": doc, "source": meta["source"], "score": 1 - dist})
        return hits


# Singleton, created once and reused across requests (see app/main.py lifespan)
retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global retriever
    if retriever is None:
        retriever = Retriever()
    return retriever
