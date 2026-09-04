import os
import json
import unittest
from pathlib import Path

from config import DATABASE_DIR, LOGS_FILE, CHROMA_PERSIST_DIR
from services.logger_service import log_call, get_logs, clear_logs
from services.ollama_embedder import OllamaEmbeddingFunction, check_ollama_health
from services.document_loader import load_from_directory, load_from_url
from services.vector_store import vector_store
from services.gemma_service import gemma_service
from app import app


class TestPrivateRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_ollama_health_and_embedding(self):
        """Test local Ollama connectivity and embedding generation."""
        health = check_ollama_health()
        self.assertTrue(health.get("available"), f"Ollama not reachable: {health.get('error')}")

        embedder = OllamaEmbeddingFunction()
        emb = embedder.embed_query("Testing Ollama embedding for Private RAG")
        self.assertIsInstance(emb, list)
        self.assertGreater(len(emb), 0)

        # Verify call was logged to database/logs.json
        self.assertTrue(LOGS_FILE.exists())
        logs = get_logs(limit=10)
        self.assertGreater(len(logs), 0)
        latest = logs[0]
        self.assertIn("time", latest)
        self.assertIn("type", latest)
        self.assertIn("arguments", latest)
        self.assertIn("response", latest)

    def test_02_directory_ingestion_and_vector_persistence(self):
        """Test scanning local directory and persisting vectors to database/."""
        sample_dir = str(Path(__file__).resolve().parent.parent / "sample_docs")
        docs = load_from_directory(sample_dir, chunk_size=500, chunk_overlap=80)
        self.assertGreater(len(docs), 0, "Failed to load sample documents")

        result = vector_store.add_documents(docs)
        self.assertEqual(result.get("status"), "success")
        self.assertGreater(result.get("total_documents_in_db", 0), 0)

        # Check ChromaDB folder was populated in database/
        self.assertTrue(CHROMA_PERSIST_DIR.exists())

    def test_03_vector_similarity_query(self):
        """Test semantic query retrieval against vector store."""
        matches = vector_store.query("How does RAG combine generative models with vector storage?", top_k=2)
        self.assertGreater(len(matches), 0)
        self.assertIn("content", matches[0])
        self.assertIn("score", matches[0])

    def test_04_gemma_qa_generation(self):
        """Test Google AI Studio Gemma model answer generation and call logging."""
        matches = vector_store.query("What are the key steps in RAG?", top_k=2)
        ans = gemma_service.answer_question("What are the key steps in RAG?", matches)
        self.assertIn("answer", ans)
        self.assertGreater(len(ans["answer"]), 10)

        # Verify Google AI Studio call was logged in database/logs.json
        logs = get_logs(limit=10, call_type="google_ai_studio_gemma")
        self.assertGreater(len(logs), 0)
        gemma_log = logs[0]
        self.assertEqual(gemma_log["type"], "google_ai_studio_gemma")
        self.assertIn("model", gemma_log["arguments"])
        self.assertIn("answer_preview", gemma_log["response"])

    def test_05_flask_api_endpoints(self):
        """Test web app Flask routes."""
        # Test index page
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Private", res.data)
        self.assertIn(b"Vector DB Ingestion", res.data)
        self.assertIn(b"Gemma RAG Query", res.data)
        self.assertIn(b"Audit Logs", res.data)

        # Test health endpoint
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")

        # Test query endpoint
        res = self.client.post("/api/query", json={"question": "What is an autonomous AI agent?", "top_k": 2})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("answer", data)

        # Test logs endpoint
        res = self.client.get("/api/logs")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["logs"]), 0)

        # Test stats endpoint
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("vector_store", data)
        self.assertIn("logs_summary", data)


if __name__ == "__main__":
    unittest.main()
