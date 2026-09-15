import os
import shutil
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, DATABASE_DIR
from services.document_loader import Document
from services.ollama_embedder import OllamaEmbeddingFunction

COLLECTION_NAME = "private_rag_collection"


class VectorStoreService:
    def __init__(self, persist_dir: str = str(CHROMA_PERSIST_DIR)):
        self.persist_dir = persist_dir
        self.embedder = OllamaEmbeddingFunction()
        self._collection = None
        self._init_client()

    def _init_client(self):
        """Initialize persistent ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def collection(self):
        """Safely return collection, recovering automatically if collection ID was recreated on disk."""
        try:
            if self._collection is not None:
                self._collection.count()
                return self._collection
        except Exception:
            pass
        self._init_client()
        return self._collection

    def add_documents(self, documents: List[Document], batch_size: int = 20) -> Dict[str, Any]:
        """
        Embed and persist document chunks into the private vector database.
        """
        if not documents:
            return {"added_chunks": 0, "status": "no_documents"}

        total_chunks = len(documents)
        ids = []
        texts = []
        metadatas = []

        # Generate unique ids based on source & chunk index
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            doc_id = f"{abs(hash(source))}_{chunk_idx}_{total_chunks}"
            ids.append(doc_id)
            texts.append(doc.content)
            # Ensure metadata values are str/int/float/bool
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)

        # Batch insert to avoid overloading embedding endpoint
        for i in range(0, total_chunks, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas
            )

        distinct_sources = list({doc.metadata.get("source") for doc in documents})
        return {
            "added_chunks": total_chunks,
            "distinct_sources": distinct_sources,
            "total_documents_in_db": self.collection.count(),
            "status": "success"
        }

    def query(self, query_text: str, top_k: int = 4, min_score: float = 0.30) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most semantically relevant document chunks for a query text.
        Filters out low-confidence chunks where similarity score < min_score (default: 0.30 / 30%).
        If the query score from VectorStoreService is below 30%, the chunk text is not added to the list.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        matched_items: List[Dict[str, Any]] = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(docs)):
                distance = distances[i] if i < len(distances) else 0.0
                # Cosine similarity is 1 - cosine distance
                similarity_score = max(0.0, 1.0 - distance)
                if similarity_score >= min_score:
                    matched_items.append({
                        "content": docs[i],
                        "metadata": metas[i] if i < len(metas) else {},
                        "distance": round(distance, 4),
                        "score": round(similarity_score, 4)
                    })

        return matched_items

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics on the current vector database."""
        total_chunks = self.collection.count()
        sources = set()

        if total_chunks > 0:
            # Sample metadata to find sources
            sample = self.collection.get(limit=min(total_chunks, 500), include=["metadatas"])
            for meta in sample.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])

        # Calculate disk usage of database/ folder
        db_size_bytes = 0
        for p in DATABASE_DIR.rglob("*"):
            if p.is_file():
                db_size_bytes += p.stat().st_size

        return {
            "total_chunks": total_chunks,
            "unique_sources": len(sources),
            "sources_list": sorted(list(sources)),
            "distinct_sources": sorted(list(sources)),
            "database_dir": str(DATABASE_DIR),
            "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
            "collection_name": COLLECTION_NAME,
            "embedder_model": self.embedder.model
        }

    def reset_database(self) -> bool:
        """Clear all records from the vector collection and recreate."""
        try:
            try:
                self.client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass
            self._init_client()
            return True
        except Exception as e:
            print(f"[VectorStore] Reset error: {e}")
            return False

    def switch_embedder(self, new_model: str) -> bool:
        """
        Switch to a new embedding model:
        1. Purge/reset existing vector collection to avoid dimension mismatch.
        2. Update persisted configuration in services/embedder_config.json.
        3. Re-initialize embedder and ChromaDB collection.
        """
        from services.embedder_manager import save_embedder_model
        try:
            # Delete old collection
            try:
                self.client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                pass

            # Update embedder model
            self.embedder = OllamaEmbeddingFunction(model=new_model)

            # Persist to services/embedder_config.json
            save_embedder_model(new_model)

            # Re-initialize collection with new embedding function
            self._init_client()
            return True
        except Exception as e:
            print(f"[VectorStore] Error switching embedder to {new_model}: {e}")
            return False


# Global singleton instance
vector_store = VectorStoreService()
