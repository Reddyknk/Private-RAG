import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, DATABASE_DIR
from services.document_loader import Document, extract_skill_metadata
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
        When adding SKILL.md:
        - Only creates the vectors using the name and description in the file.
        - The chunk is the whole file.
        - During retrieval, provides all the text in the file.
        """
        if not documents:
            return {"added_chunks": 0, "status": "no_documents"}

        total_chunks = len(documents)
        ids = []
        texts = []
        texts_to_embed = []
        metadatas = []

        # Generate unique ids based on source & chunk index
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            doc_id = f"{abs(hash(source))}_{chunk_idx}_{total_chunks}"
            ids.append(doc_id)
            texts.append(doc.content)

            is_skill_md = (
                doc.metadata.get("is_skill_md")
                or doc.metadata.get("filename", "").lower() == "skill.md"
                or Path(str(source)).name.lower() == "skill.md"
            )

            if is_skill_md:
                name = doc.metadata.get("skill_name")
                desc = doc.metadata.get("skill_description")
                if not name or not desc:
                    extracted_name, extracted_desc = extract_skill_metadata(doc.content, default_name=Path(str(source)).parent.name)
                    name = name or extracted_name
                    desc = desc or extracted_desc

                # Only create the vectors using the name and description in the file
                embed_text = f"name: {name}\ndescription: {desc}".strip() if (name or desc) else doc.content
                texts_to_embed.append(embed_text)
            else:
                texts_to_embed.append(doc.content)

            # Ensure metadata values are str/int/float/bool
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            if is_skill_md:
                clean_meta["is_skill_md"] = True
                if name:
                    clean_meta["skill_name"] = name
                if desc:
                    clean_meta["skill_description"] = desc
            metadatas.append(clean_meta)

        # Batch insert with custom embeddings
        for i in range(0, total_chunks, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_texts_to_embed = texts_to_embed[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            # Generate vector embeddings using Ollama: for SKILL.md this uses only name and description
            batch_embeddings = self.embedder.embed_documents(batch_texts_to_embed)

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

        distinct_sources = list({doc.metadata.get("source") for doc in documents})
        return {
            "added_chunks": total_chunks,
            "distinct_sources": distinct_sources,
            "total_documents_in_db": self.collection.count(),
            "status": "success"
        }

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        min_score: float = 0.30,
        min_skill_score: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most semantically relevant document chunks for a query text.
        Filters out low-confidence document chunks where similarity score < min_score (default: 0.30 / 30%).
        For skill vector query, only use skill chunks with score higher than 50% (similarity score > min_skill_score, default: 0.50).
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
                meta = metas[i] if i < len(metas) else {}

                is_skill_chunk = (
                    meta.get("is_skill_md") is True
                    or meta.get("source_type") == "skill"
                    or Path(str(meta.get("source", ""))).name.lower() == "skill.md"
                    or meta.get("filename", "").lower() == "skill.md"
                )

                # For skill vector query only use skill chunks with score higher than 50% (> 0.50)
                if is_skill_chunk:
                    if similarity_score > min_skill_score:
                        matched_items.append({
                            "content": docs[i],
                            "metadata": meta,
                            "distance": round(distance, 4),
                            "score": round(similarity_score, 4),
                            "is_skill": True
                        })
                else:
                    if similarity_score >= min_score:
                        matched_items.append({
                            "content": docs[i],
                            "metadata": meta,
                            "distance": round(distance, 4),
                            "score": round(similarity_score, 4),
                            "is_skill": False
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
