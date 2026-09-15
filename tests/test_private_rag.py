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
        app.config["TESTING"] = True
        app.testing = True
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
        self.assertIn(b"Chat", res.data)
        self.assertIn(b"Audit Logs", res.data)

        # Test health endpoint
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")

        # Test skills endpoint
        res = self.client.get("/api/skills")
        self.assertEqual(res.status_code, 200)
        skills_data = res.get_json()
        self.assertEqual(skills_data["status"], "success")
        self.assertGreaterEqual(skills_data["count"], 2)

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

        # Verify two tables in HTML for Audit Logs page
        index_res = self.client.get("/")
        self.assertIn(b"conversationsTable", index_res.data)
        self.assertIn(b"eventsTable", index_res.data)

        # Test conversations API
        conv_res = self.client.get("/api/conversations")
        self.assertEqual(conv_res.status_code, 200)
        conv_data = conv_res.get_json()
        self.assertEqual(conv_data["status"], "success")
        self.assertGreater(conv_data["count"], 0)
        first_conv = conv_data["conversations"][0]
        self.assertIn("conversation_id", first_conv)
        self.assertIn("user_query", first_conv)
        self.assertIn("agent_response", first_conv)

        # Test conversation events API
        conv_id = first_conv["conversation_id"]
        ev_res = self.client.get(f"/api/conversations/{conv_id}/events")
        self.assertEqual(ev_res.status_code, 200)
        ev_data = ev_res.get_json()
        self.assertEqual(ev_data["status"], "success")
        self.assertGreater(ev_data["count"], 0)

    def test_06_redaction_and_async_logging(self):
        """Test API key redaction and async logging with size constraints."""
        from services.logger_service import log_event, flush_logs, _trim_to_fit

        test_conv_id = "conv-redaction-test"
        sample_secret = "AIzaSySecretTestKey12345678901234567890"
        entry = log_event(
            event_type="LLM Invocation",
            invoker="Agent",
            target="Google AI Studio",
            short_description=f"Testing key {sample_secret} redaction",
            payload={
                "prompt": f"Please answer with key {sample_secret}",
                "api_key": sample_secret,
                "headers": {"x-goog-api-key": sample_secret, "Authorization": f"Bearer {sample_secret}"}
            },
            conversation_id=test_conv_id
        )

        flush_logs()

        # Check in-memory payload
        self.assertNotIn(sample_secret, json.dumps(entry["payload"]))
        self.assertEqual(entry["payload"]["api_key"], "[REDACTED_API_KEY]")
        self.assertEqual(entry["payload"]["headers"]["x-goog-api-key"], "[REDACTED_API_KEY]")

        # Test 30MB trimming logic with dummy entries
        records = [{"id": i, "content": "x" * 1000} for i in range(50)]
        trimmed = _trim_to_fit(records, max_bytes=20000, target_bytes=10000)
        self.assertLess(len(trimmed), len(records))

    def test_07_all_components_logged_and_displayed(self):
        """Verify Agent, LLM, Embedder, Tools, and External API are logged and returned in message."""
        from services.logger_service import flush_logs

        res = self.client.post("/api/query", json={
            "question": "What is the time and weather in Tokyo?",
            "top_k": 2
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("components", data)
        self.assertIn("answer", data)

        components = data["components"]
        self.assertIsInstance(components, list)
        self.assertGreaterEqual(len(components), 4)

        comp_names = [c["name"] for c in components]
        self.assertIn("Agent", comp_names)
        self.assertIn("Embedder", comp_names)
        self.assertIn("Vector Store", comp_names)
        self.assertIn("Tool", comp_names)
        self.assertIn("External API", comp_names)
        self.assertIn("LLM", comp_names)

        for c in components:
            self.assertIn("request", c)
            self.assertIn("response", c)
            self.assertIn("status", c)
            self.assertIn("duration_ms", c)

        conv_id = data["conversation_id"]
        flush_logs()

        ev_res = self.client.get(f"/api/conversations/{conv_id}/events")
        self.assertEqual(ev_res.status_code, 200)
        events = ev_res.get_json()["events"]
        event_types = [e.get("event_type") or e.get("type") for e in events]

        self.assertTrue(any("Agent" in t for t in event_types))
        self.assertTrue(any("Embedder" in t for t in event_types))
        self.assertTrue(any("Vector" in t for t in event_types))
        self.assertTrue(any("Tool" in t for t in event_types))
        self.assertTrue(any("External API" in t for t in event_types))
        self.assertTrue(any("LLM" in t or "gemma" in t for t in event_types))

    def test_08_text_only_models_and_custom_api_endpoint(self):
        """Test that /api/models returns only text models plus Custom Model API, and test custom endpoint querying."""
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        models = data.get("models", [])
        self.assertGreater(len(models), 0)

        # Verify no non-text models are returned
        non_text_keywords = ["image", "imagen", "tts", "native-audio", "transcribe", "lyria", "music", "banana", "robotics"]
        for m in models:
            mid = m["id"].lower()
            mname = m["name"].lower()
            for kw in non_text_keywords:
                self.assertNotIn(kw, mid, f"Found non-text model {m['id']}")
                self.assertNotIn(kw, mname, f"Found non-text model name {m['name']}")

        # Verify Custom Model API option is present
        custom_entry = next((m for m in models if m["id"] == "custom_model_api"), None)
        self.assertIsNotNone(custom_entry, "Custom Model API option should be present in models list")
        self.assertEqual(custom_entry["name"], "Custom Model API")

        # Test querying with custom_model_api and custom endpoint
        import requests
        from unittest.mock import patch, MagicMock
        real_post = requests.post
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "reply": "This is a response generated by the custom private LLM model server."
        }
        mock_resp.text = "This is a response generated by the custom private LLM model server."

        custom_calls = []

        def selective_post(url, *args, **kwargs):
            if "8000" in url or "api/chat" in url:
                custom_calls.append((url, args, kwargs))
                return mock_resp
            return real_post(url, *args, **kwargs)

        # Ensure sample docs are present in vector store
        if vector_store.collection.count() == 0 or len(vector_store.query("What is RAG architecture?", top_k=1, min_score=0.25)) == 0:
            sample_dir = str(Path(__file__).resolve().parent.parent / "sample_docs")
            docs = load_from_directory(sample_dir, chunk_size=500, chunk_overlap=80)
            vector_store.add_documents(docs)

        with patch("requests.post", side_effect=selective_post):
            query_res = self.client.post("/api/query", json={
                "question": "What is RAG architecture?",
                "top_k": 2,
                "model": "custom_model_api",
                "custom_endpoint": "http://127.0.0.1:8000/api/chat"
            })
            self.assertEqual(query_res.status_code, 200)
            q_data = query_res.get_json()
            self.assertIn("answer", q_data)
            self.assertIn("custom private llm", q_data["answer"].lower())

            # Verify requests.post was called with custom endpoint
            self.assertGreaterEqual(len(custom_calls), 1)
            call_url = custom_calls[0][0]
            self.assertEqual(call_url, "http://127.0.0.1:8000/api/chat")
            call_json = custom_calls[0][2].get("json", {})
            self.assertIn("prompt", call_json)

            # Verify LLM component was returned
            components = q_data.get("components", [])
            llm_comp = next((c for c in components if c["name"] == "LLM"), None)
            self.assertIsNotNone(llm_comp)
            self.assertEqual(llm_comp["status"], "success")
            self.assertIn("endpoint", llm_comp["request"])
            self.assertEqual(llm_comp["request"]["endpoint"], "http://127.0.0.1:8000/api/chat")

    def test_09_conversational_greeting_and_relevance_filtering(self):
        """Test that conversational greeting ('Hello') does not pass irrelevant documents to LLM and greets user warmly."""
        # 1. Test vector store query directly filters matches below 25%
        raw_matches = vector_store.query("Hello", top_k=4, min_score=0.25)
        self.assertEqual(len(raw_matches), 0, "Vector store query for 'Hello' should filter out all low-similarity matches")

        # 2. Test /api/query endpoint with 'Hello'
        res = self.client.post("/api/query", json={"question": "Hello", "top_k": 4})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data.get("sources", [])), 0, "Sources should be empty for a greeting query")
        answer_lower = data.get("answer", "").lower()
        self.assertTrue(
            any(g in answer_lower for g in ["hello", "hi", "welcome", "pleasure", "assist", "help"]),
            f"Expected a warm greeting, got: {data.get('answer')}"
        )
        self.assertNotIn("hnsw", answer_lower, "Should not recite HNSW architecture context for Hello greeting")
        self.assertNotIn("only provides information about the architecture", answer_lower)

        # 3. Check Vector Store component in response indicates threshold filtering
        components = data.get("components", [])
        vs_comp = next((c for c in components if c["name"] == "Vector Store"), None)
        self.assertIsNotNone(vs_comp)
        self.assertEqual(vs_comp["response"]["retrieved_count"], 0)

    def test_10_shutdown_and_branding_elements(self):
        """Verify Agent with RAG branding, icons, shutdown modal, and /api/shutdown endpoint."""
        # Check UI HTML contains new branding and elements
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn("Agent with RAG", html)
        self.assertIn("AI_icon_s.png", html)
        self.assertIn("AI_icon.png", html)
        self.assertIn("btnShutdownApp", html)
        self.assertIn("shutdownModal", html)
        self.assertIn("disable the application for all users", html)
        self.assertIn("conversations-scroll-wrapper", html)

        # Test /api/shutdown endpoint with os.kill mocked to prevent killing test process
        from unittest.mock import patch
        with patch("os.kill") as mock_kill, patch("app.shutdown_ollama_if_started_by_app") as mock_shut_ollama:
            shut_res = self.client.post("/api/shutdown", json={"action": "shutdown"})
            self.assertEqual(shut_res.status_code, 200)
            shut_data = shut_res.get_json()
            self.assertEqual(shut_data["status"], "success")
            self.assertIn("shutting down", shut_data["message"].lower())

    def test_11_low_similarity_sends_prompt_to_model_without_vector_context(self):
        """Verify that when vector store similarity is below 0.25, the prompt is sent to the model without vector store info."""
        res = self.client.post("/api/query", json={"question": "What is the capital of France?", "top_k": 4})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data.get("sources", [])), 0, "No sources should be attached when score is below 0.25")

        answer_lower = data.get("answer", "").lower()
        self.assertIn("paris", answer_lower, f"Expected model to answer 'Paris', got: {data.get('answer')}")

        # Check components: Vector Store had 0 chunks, LLM was called with 0 context chunks
        components = data.get("components", [])
        vs_comp = next((c for c in components if c["name"] == "Vector Store"), None)
        llm_comp = next((c for c in components if c["name"] == "LLM"), None)

        self.assertIsNotNone(vs_comp)
        self.assertEqual(vs_comp["response"]["retrieved_count"], 0)
        self.assertIsNotNone(llm_comp)
        self.assertEqual(llm_comp["status"], "success")
        self.assertEqual(llm_comp["request"]["context_chunks_count"], 0)

    def test_12_cutoff_and_conversations_prompt_entry(self):
        """Verify 30% score cutoff in vector store and immediate entry creation in conversations.json upon prompt receipt."""
        from services.logger_service import flush_logs, CONVERSATIONS_FILE

        # 1. Test 30% vector store cutoff
        low_matches = vector_store.query("hello good morning how are you", top_k=4)
        for m in low_matches:
            self.assertGreaterEqual(m["score"], 0.30, f"Found match below 30% threshold: {m['score']}")

        # 2. Test prompt received conversation logging
        test_conv_id = f"conv-test12-{int(os.getpid())}"
        test_query = "What is the capital of Japan?"
        res = self.client.post("/api/query", json={
            "question": test_query,
            "conversation_id": test_conv_id,
            "top_k": 2
        })
        self.assertEqual(res.status_code, 200)

        flush_logs()
        with open(CONVERSATIONS_FILE, "r") as f:
            conversations = json.load(f)

        record = next((c for c in conversations if c["conversation_id"] == test_conv_id), None)
        self.assertIsNotNone(record, f"Conversation {test_conv_id} was not recorded in conversations.json")
        self.assertEqual(record["user_query"], test_query)
        self.assertNotEqual(record["agent_response"], "[Processing...]")


if __name__ == "__main__":
    unittest.main()


