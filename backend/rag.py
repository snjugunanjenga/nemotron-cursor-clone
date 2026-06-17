"""Minimal RAG helper: ingest files and query.

This module prefers chromadb when available; otherwise falls back to a simple in-memory store
suitable for smoke tests and local development. Replace or extend with production-grade
vector DB and embedding pipeline when available.
"""
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    CHROMADB_AVAILABLE = False

class RAG:
    def __init__(self, persist_directory: str | None = None):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        # Simple in-memory fallback store
        self._documents: List[Dict[str, Any]] = []
        if CHROMADB_AVAILABLE:
            self.client = chromadb.Client(Settings())

    def create_collection(self, name: str = 'default'):
        if CHROMADB_AVAILABLE and not self.collection:
            self.collection = self.client.create_collection(name=name)
            return self.collection
        # in-memory: reset list
        self._documents = []
        return self._documents

    def ingest_texts(self, texts: List[str], metadatas: List[dict] | None = None):
        """Ingest a list of texts with optional metadata.
        Returns list of ids (strings).
        """
        metadatas = metadatas or [{}] * len(texts)
        if CHROMADB_AVAILABLE:
            if not self.collection:
                self.create_collection()
            ids = [str(i) for i in range(len(texts))]
            self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
            return ids

        # in-memory fallback: append documents
        start_idx = len(self._documents)
        ids = []
        for i, (t, m) in enumerate(zip(texts, metadatas)):
            doc_id = str(start_idx + i)
            self._documents.append({'id': doc_id, 'text': t, 'metadata': m})
            ids.append(doc_id)
        return ids

    def query(self, query_text: str, n_results: int = 4):
        """Return results. If chromadb available, proxy to it; otherwise do simple substring match."""
        if CHROMADB_AVAILABLE and self.collection:
            return self.collection.query(query_texts=[query_text], n_results=n_results)

        # naive in-memory search: score by substring containment and length
        scored = []
        for d in self._documents:
            score = 0
            txt = d.get('text', '')
            if query_text.lower() in txt.lower():
                score += 10
            # shorter docs that contain query get slight preference
            score += max(0, 1.0 - (len(txt) / 10000.0))
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:n_results]]
        # Return in a shape similar to chromadb for compatibility
        return {
            'ids': [r['id'] for r in results],
            'documents': [r['text'] for r in results],
            'metadatas': [r.get('metadata', {}) for r in results]
        }


# Example usage:
# rag = RAG()
# rag.create_collection('repo')
# rag.ingest_texts(['hello world'], [{'path':'example.txt'}])
# results = rag.query('hello')
