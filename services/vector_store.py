import os
import shutil
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

    def add_documents(self, documents: List[Document], batch_size: int = 20) -> Dict[str, Any]:
        """
        Embed and persist document chunks with ~20% overlap into the document database.
        """
        if not documents:
            return {"added_chunks": 0, "status": "no_documents"}

        total_chunks = len(documents)
        ids = []
        texts = []
        texts_to_embed = []
        metadatas = []

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            doc_id = f"doc_{abs(hash(source))}_{chunk_idx}_{total_chunks}"
            ids.append(doc_id)
            texts.append(doc.content)
            texts_to_embed.append(doc.content)

            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_meta["is_document"] = True
            metadatas.append(clean_meta)

        for i in range(0, total_chunks, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_texts_to_embed = texts_to_embed[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

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

    def switch_embedder(self, new_model: str) -> bool:
        self.embedder = OllamaEmbeddingFunction(model=new_model)
        self.clear()
        return True


def switch_system_embedder(new_model: str) -> bool:
    from services.embedder_manager import set_active_embedder_model
    set_active_embedder_model(new_model)
    doc_vector_store.switch_embedder(new_model)
    skill_vector_store.switch_embedder(new_model)
    try:
        from services.skill_runner import auto_index_skills_into_db
        auto_index_skills_into_db()
    except Exception as e:
        print(f"[VectorStore] Warning re-indexing skills after embedder switch: {e}")
    return True


# Create singleton instances for both vector store databases
doc_vector_store = DocumentVectorStore()
skill_vector_store = SkillVectorStore()

# Bind switch_embedder to doc_vector_store
doc_vector_store.switch_embedder = switch_system_embedder

# Backwards compatibility alias for components referencing vector_store
vector_store = doc_vector_store
