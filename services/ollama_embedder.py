import time
import requests
from typing import List, Dict, Any, Optional
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL
from services.logger_service import log_call


class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB compatible embedding function that calls local Ollama
    and logs every call to database/logs.json.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of documents/texts using local Ollama."""
        if isinstance(input, str):
            input = [input]
        return self.embed_documents(input)

    def embed_query(self, input: Any) -> Any:
        """Embed query input for ChromaDB or direct usage."""
        if isinstance(input, str):
            res = self.embed_documents([input])
            return res[0] if res else []
        elif isinstance(input, (list, tuple)):
            return self.embed_documents(list(input))
        return self.embed_documents([str(input)])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text strings."""
        embeddings: List[List[float]] = []

        # Attempt batch embedding via /api/embed if available
        start_time = time.time()
        endpoint = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": texts
        }

        try:
            resp = requests.post(endpoint, json=payload, timeout=60)
            duration_ms = (time.time() - start_time) * 1000

            if resp.status_code == 200:
                data = resp.json()
                result_embeddings = data.get("embeddings", [])
                if result_embeddings and len(result_embeddings) == len(texts):
                    log_call(
                        call_type="local_ollama_embed_batch",
                        arguments={
                            "endpoint": endpoint,
                            "model": self.model,
                            "texts_count": len(texts),
                            "preview": [t[:80] + "..." if len(t) > 80 else t for t in texts[:3]]
                        },
                        response={
                            "status_code": resp.status_code,
                            "embeddings_count": len(result_embeddings),
                            "dimensions": len(result_embeddings[0]) if result_embeddings else 0
                        },
                        duration_ms=duration_ms,
                        status="success"
                    )
                    return result_embeddings
        except Exception as e:
            # Fall back to single embedding endpoint
            pass

        # Fallback to /api/embeddings per item
        for text in texts:
            emb = self._embed_single_text(text)
            embeddings.append(emb)

        return embeddings

    def _embed_single_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single query text."""
        start_time = time.time()
        endpoint = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }

        try:
            resp = requests.post(endpoint, json=payload, timeout=30)
            duration_ms = (time.time() - start_time) * 1000

            if resp.status_code == 200:
                data = resp.json()
                embedding = data.get("embedding", [])
                log_call(
                    call_type="local_ollama_embedding",
                    arguments={
                        "endpoint": endpoint,
                        "model": self.model,
                        "text_length": len(text),
                        "text_sample": text[:100] + "..." if len(text) > 100 else text
                    },
                    response={
                        "status_code": resp.status_code,
                        "dimensions": len(embedding)
                    },
                    duration_ms=duration_ms,
                    status="success"
                )
                return embedding
            else:
                error_resp = resp.text
                log_call(
                    call_type="local_ollama_embedding",
                    arguments={"endpoint": endpoint, "model": self.model, "text_sample": text[:100]},
                    response={"status_code": resp.status_code, "error": error_resp},
                    duration_ms=duration_ms,
                    status="error"
                )
                raise RuntimeError(f"Ollama embedding failed ({resp.status_code}): {error_resp}")

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_call(
                call_type="local_ollama_embedding",
                arguments={"endpoint": endpoint, "model": self.model, "text_sample": text[:100]},
                response={"error": str(e)},
                duration_ms=duration_ms,
                status="error"
            )
            raise


def check_ollama_health() -> Dict[str, Any]:
    """Check if local Ollama daemon is running and check model availability."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {
                "available": True,
                "models": models,
                "default_embed_model": OLLAMA_EMBED_MODEL,
                "embed_model_loaded": any(OLLAMA_EMBED_MODEL in m for m in models)
            }
    except Exception as e:
        return {"available": False, "error": str(e)}
    return {"available": False, "error": "Unexpected response"}
