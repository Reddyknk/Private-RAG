import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_DOCS_DIR, CHROMA_SKILLS_DIR, CHROMA_PERSIST_DIR, DATABASE_DIR
from services.document_loader import Document, extract_skill_metadata
from services.ollama_embedder import OllamaEmbeddingFunction

DOCS_COLLECTION_NAME = "documents_collection"
SKILLS_COLLECTION_NAME = "skills_collection"


class DocumentVectorStore:
    """
    Vector store dedicated to user and corporate documents.
    Stores chunked documents with 20% overlap in database/chroma_docs.
    Includes built-in duplicate chunk prevention via content hashing.
    """
    def __init__(self, persist_dir: str = str(CHROMA_DOCS_DIR)):
        self.persist_dir = str(persist_dir)
        self.embedder = OllamaEmbeddingFunction()
        self._collection = None
        self._init_client()

    def _init_client(self):
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self.client.get_or_create_collection(
            name=DOCS_COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"}
        )
        try:
            self.remove_duplicates()
        except Exception:
            pass

    @property
    def collection(self):
        try:
            if self._collection is not None:
                self._collection.count()
                return self._collection
        except Exception:
            pass
        self._init_client()
        return self._collection

    def get_existing_content_hashes(self) -> set:
        """Retrieve set of SHA-256 hashes of all chunks currently in the database."""
        existing_hashes = set()
        count = self.collection.count()
        if count == 0:
            return existing_hashes
        try:
            records = self.collection.get(include=["metadatas", "documents"])
            for meta, text in zip(records.get("metadatas", []) or [], records.get("documents", []) or []):
                if meta and "content_hash" in meta and meta["content_hash"]:
                    existing_hashes.add(meta["content_hash"])
                elif text:
                    existing_hashes.add(hashlib.sha256(text.strip().encode("utf-8")).hexdigest())
        except Exception as e:
            print(f"[DocumentVectorStore] Warning reading existing hashes: {e}")
        return existing_hashes

    def remove_duplicates(self) -> int:
        """
        Scans existing database and removes any duplicate chunks that share identical content hashes.
        Returns the number of pruned duplicate chunks.
        """
        if self._collection is None:
            return 0
        count = self._collection.count()
        if count == 0:
            return 0
        try:
            records = self._collection.get(include=["documents", "metadatas"])
            ids = records.get("ids", [])
            docs = records.get("documents", [])
            seen_hashes = {}
            duplicate_ids = []
            for doc_id, doc in zip(ids, docs):
                h = hashlib.sha256(doc.strip().encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    duplicate_ids.append(doc_id)
                else:
                    seen_hashes[h] = doc_id
            if duplicate_ids:
                for i in range(0, len(duplicate_ids), 100):
                    batch = duplicate_ids[i:i + 100]
                    self._collection.delete(ids=batch)
                print(f"[DocumentVectorStore] Pruned {len(duplicate_ids)} duplicate chunks from database.")
                return len(duplicate_ids)
        except Exception as e:
            print(f"[DocumentVectorStore] Warning pruning duplicates: {e}")
        return 0

    def add_documents(self, documents: List[Document], batch_size: int = 20) -> Dict[str, Any]:
        """
        Embed and persist document chunks with ~20% overlap into the document database.
        Checks for and skips any duplicate chunks that are already in the vector database
        or duplicated within the input batch.
        """
        if not documents:
            return {
                "added_chunks": 0,
                "skipped_duplicates": 0,
                "total_chunks_processed": 0,
                "total_documents_in_db": self.collection.count(),
                "status": "success"
            }

        existing_hashes = self.get_existing_content_hashes()

        unique_docs = []  # List of tuples: (doc, doc_id, content_hash)
        seen_in_batch = set()
        skipped_duplicates = 0

        for doc in documents:
            clean_text = doc.content.strip()
            if not clean_text:
                continue

            content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

            # Skip chunk if its exact content is already stored in ChromaDB or seen in this batch
            if content_hash in existing_hashes or content_hash in seen_in_batch:
                skipped_duplicates += 1
                continue

            seen_in_batch.add(content_hash)
            doc_id = f"chunk_{content_hash[:24]}"
            unique_docs.append((doc, doc_id, content_hash))

        distinct_sources = list({doc.metadata.get("source") for doc in documents if doc.metadata.get("source")})

        if not unique_docs:
            return {
                "added_chunks": 0,
                "skipped_duplicates": skipped_duplicates,
                "total_chunks_processed": len(documents),
                "distinct_sources": distinct_sources,
                "total_documents_in_db": self.collection.count(),
                "status": "success"
            }

        ids = [item[1] for item in unique_docs]
        texts = [item[0].content for item in unique_docs]
        metadatas = []

        for doc, _, content_hash in unique_docs:
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_meta["is_document"] = True
            clean_meta["content_hash"] = content_hash
            metadatas.append(clean_meta)

        for i in range(0, len(unique_docs), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            batch_embeddings = self.embedder.embed_documents(batch_texts)

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

        return {
            "added_chunks": len(unique_docs),
            "skipped_duplicates": skipped_duplicates,
            "total_chunks_processed": len(documents),
            "distinct_sources": distinct_sources,
            "total_documents_in_db": self.collection.count(),
            "status": "success"
        }

    def query(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 4,
        min_score: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Query the document database.
        Filters out low-confidence chunks where similarity score < min_score (default: 0.30 / 30%).
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        try:
            if query_embedding is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=actual_k,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                emb = self.embedder.embed_query(query_text)
                results = self.collection.query(
                    query_embeddings=[emb],
                    n_results=actual_k,
                    include=["documents", "metadatas", "distances"]
                )
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"[DocumentVectorStore] Warning: Embedding dimension mismatch between active model and collection ({e}). Returning empty results.")
                return []
            raise

        matched_items: List[Dict[str, Any]] = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(docs)):
                distance = distances[i] if i < len(distances) else 0.0
                similarity_score = max(0.0, 1.0 - distance)
                meta = metas[i] if i < len(metas) else {}

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
        total_chunks = self.collection.count()
        source_counts: Dict[str, int] = {}
        source_details: List[Dict[str, Any]] = []
        if total_chunks > 0:
            try:
                records = self.collection.get(include=["metadatas"])
                for meta in records.get("metadatas", []):
                    if meta and "source" in meta:
                        src = str(meta["source"])
                        source_counts[src] = source_counts.get(src, 0) + 1
            except Exception as e:
                print(f"[DocumentVectorStore] Warning reading metadata: {e}")

        sources_list = sorted(list(source_counts.keys()))
        for src in sources_list:
            is_url = src.startswith("http://") or src.startswith("https://")
            name = src.split("/")[-1] if "/" in src else src
            source_details.append({
                "source": src,
                "name": name if name else src,
                "is_url": is_url,
                "chunk_count": source_counts[src]
            })

        # Calculate database storage size on disk
        db_size_mb = 0.0
        try:
            p = Path(self.persist_dir)
            if p.exists():
                total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                db_size_mb = round(total_bytes / (1024 * 1024), 2)
        except Exception:
            pass

        return {
            "type": "document_database",
            "database_name": "Document Database (chroma_docs)",
            "total_chunks": total_chunks,
            "unique_sources": len(sources_list),
            "distinct_sources_count": len(sources_list),
            "sources_list": sources_list,
            "sources": sources_list,
            "source_details": source_details,
            "db_size_mb": db_size_mb
        }

    def clear(self):
        try:
            self.client.delete_collection(DOCS_COLLECTION_NAME)
        except Exception:
            pass
        self._init_client()

    def reset_database(self) -> bool:
        """Clear all documents in the document database."""
        try:
            self.clear()
            return True
        except Exception as e:
            print(f"[DocumentVectorStore] Reset error: {e}")
            return False

    def _apply_embedder(self, new_model: str) -> bool:
        """Update internal embedding function to new model and purge document collection."""
        self.embedder = OllamaEmbeddingFunction(model=new_model)
        self.clear()
        return True

    def switch_embedder(self, new_model: str) -> bool:
        """Switch embedder system-wide."""
        return switch_system_embedder(new_model)


class SkillVectorStore:
    """
    Vector store dedicated to Agent skills.
    Stores SKILL.md files in database/chroma_skills.
    Only the name and description are sent to the embedder.
    The vectors and the complete text of SKILL.md are stored in the database.
    """
    def __init__(self, persist_dir: str = str(CHROMA_SKILLS_DIR)):
        self.persist_dir = str(persist_dir)
        self.embedder = OllamaEmbeddingFunction()
        self._collection = None
        self._init_client()

    def _init_client(self):
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self.client.get_or_create_collection(
            name=SKILLS_COLLECTION_NAME,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"}
        )
        # Ensure skills collection dimension matches the active embedder model
        try:
            existing = self._collection.get(include=["embeddings"], limit=1)
            if existing and existing.get("embeddings") and len(existing["embeddings"]) > 0:
                col_dim = len(existing["embeddings"][0])
                test_emb = self.embedder.embed_query("test")
                if test_emb and col_dim != len(test_emb):
                    print(f"[SkillVectorStore] Dimension mismatch detected (collection: {col_dim}, active embedder: {len(test_emb)}). Rebuilding skills collection...")
                    self.client.delete_collection(SKILLS_COLLECTION_NAME)
                    self._collection = self.client.create_collection(
                        name=SKILLS_COLLECTION_NAME,
                        embedding_function=self.embedder,
                        metadata={"hnsw:space": "cosine"}
                    )
        except Exception:
            pass

    @property
    def collection(self):
        try:
            if self._collection is not None:
                self._collection.count()
                return self._collection
        except Exception:
            pass
        self._init_client()
        return self._collection

    def add_skill(
        self,
        skill_id: str,
        name: str,
        description: str,
        full_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Embed only the name and description. Store the vectors and complete text of SKILL.md.
        """
        embed_text = f"name: {name}\ndescription: {description}".strip()
        vector = self.embedder.embed_query(embed_text)

        clean_meta = {
            "skill_id": skill_id,
            "name": name,
            "description": description,
            "is_skill_md": True,
            "chunk_index": 0,
            "total_chunks": 1
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)

        doc_id = f"skill_{skill_id}"
        self.collection.upsert(
            ids=[doc_id],
            documents=[full_content],  # Complete text of SKILL.md
            embeddings=[vector],       # Vector from name and description only
            metadatas=[clean_meta]
        )
        return {
            "skill_id": skill_id,
            "status": "success",
            "total_skills": self.collection.count()
        }

    def query(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        min_score: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Query the skill database.
        Returns candidates with similarity scores strictly higher than 50% (> min_score, default: 0.50).
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        try:
            if query_embedding is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=actual_k,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                emb = self.embedder.embed_query(query_text)
                results = self.collection.query(
                    query_embeddings=[emb],
                    n_results=actual_k,
                    include=["documents", "metadatas", "distances"]
                )
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"[SkillVectorStore] Warning: Embedding dimension mismatch ({e}). Returning empty results.")
                return []
            raise

        matched_skills: List[Dict[str, Any]] = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(docs)):
                distance = distances[i] if i < len(distances) else 0.0
                similarity_score = max(0.0, 1.0 - distance)
                meta = metas[i] if i < len(metas) else {}

                # Candidate with score higher than 50% (> 0.50)
                if similarity_score > min_score:
                    matched_skills.append({
                        "skill_id": meta.get("skill_id") or meta.get("name", "unknown"),
                        "name": meta.get("name", ""),
                        "description": meta.get("description", ""),
                        "content": docs[i],  # Complete text of SKILL.md
                        "metadata": meta,
                        "distance": round(distance, 4),
                        "score": round(similarity_score, 4),
                        "is_skill": True
                    })

        return matched_skills

    def get_stats(self) -> Dict[str, Any]:
        return {
            "type": "skill_database",
            "total_skills": self.collection.count()
        }

    def clear(self):
        try:
            self.client.delete_collection(SKILLS_COLLECTION_NAME)
        except Exception:
            pass
        self._init_client()

    def _apply_embedder(self, new_model: str) -> bool:
        """Update internal embedding function to new model and purge skills collection."""
        self.embedder = OllamaEmbeddingFunction(model=new_model)
        self.clear()
        return True

    def switch_embedder(self, new_model: str) -> bool:
        """Switch embedder system-wide."""
        return switch_system_embedder(new_model)


def switch_system_embedder(new_model: str) -> bool:
    from services.embedder_manager import set_active_embedder_model
    set_active_embedder_model(new_model)
    doc_vector_store._apply_embedder(new_model)
    skill_vector_store._apply_embedder(new_model)
    try:
        from services.skill_runner import auto_index_skills_into_db
        auto_index_skills_into_db()
    except Exception as e:
        print(f"[VectorStore] Warning re-indexing skills after embedder switch: {e}")
    return True


# Create singleton instances for both vector store databases
doc_vector_store = DocumentVectorStore()
skill_vector_store = SkillVectorStore()

# Backwards compatibility alias for components referencing vector_store
vector_store = doc_vector_store
