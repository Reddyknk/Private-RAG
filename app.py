import os
import re
import shutil
import subprocess
import time
import uuid
import threading
import requests
from flask import Flask, render_template, request, jsonify

import config
from services.logger_service import (
    get_logs,
    clear_logs,
    log_call,
    log_event,
    log_conversation,
    get_conversations,
    get_conversation_events
)
from services.ollama_embedder import check_ollama_health, OllamaEmbeddingFunction
from services.document_loader import load_from_url, load_from_directory
from services.vector_store import doc_vector_store, skill_vector_store, vector_store
from services.gemma_service import gemma_service
from services.embedder_manager import (
    get_embedder_catalog,
    get_current_embedder_model,
    init_embedder_config
)
from services.skill_runner import (
    auto_index_skills_into_db,
    get_available_skills,
    execute_skill_tools_if_relevant,
    run_agent_skill_pipeline
)

# Ensure embedder configuration exists in services/
init_embedder_config()

# Minimum relevance similarity score threshold for vector database chunks (0.30 = 30%)
# If the query score from VectorStoreService is below 30%, the text chunk is not added to the list.
MIN_RELEVANCE_SCORE = 0.30
# For skill vector queries, only use skill chunks with score higher than 50% (0.50)
MIN_SKILL_RELEVANCE_SCORE = 0.50

CONVERSATIONAL_GREETINGS = {
    "hello", "hi", "hey", "howdy", "greetings", "hola", "bonjour",
    "good morning", "good afternoon", "good evening", "good day",
    "what can you do", "who are you", "what are you", "help",
    "how are you", "hows it going", "whats up", "sup", "hi there", "hello there"
}


def is_conversational_greeting(text: str) -> bool:
    """Detect if the user query is a greeting, polite pleasantry, or introductory question."""
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    words = cleaned.split()
    if not words:
        return False
    if cleaned in CONVERSATIONAL_GREETINGS:
        return True
    if len(words) <= 2 and words[0] in {"hello", "hi", "hey", "howdy", "greetings"}:
        return True
    return False

import atexit
import signal
import sys

_ollama_process = None
_ollama_started_by_app = False


def shutdown_ollama_if_started_by_app():
    """Shutdown Ollama service only if it was started by this application instance."""
    global _ollama_started_by_app, _ollama_process
    if not _ollama_started_by_app:
        return
    print("\n[Ollama] Application is terminating. Shutting down Ollama daemon started by app...")
    try:
        if _ollama_process and _ollama_process.poll() is None:
            _ollama_process.terminate()
            try:
                _ollama_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _ollama_process.kill()
            print("[Ollama] Ollama daemon terminated cleanly.")
    except Exception as e:
        print(f"[Ollama] Error during daemon termination: {e}")
    _ollama_started_by_app = False


atexit.register(shutdown_ollama_if_started_by_app)


def _handle_exit_signal(sig, frame):
    shutdown_ollama_if_started_by_app()
    sys.exit(0)


try:
    signal.signal(signal.SIGINT, _handle_exit_signal)
    signal.signal(signal.SIGTERM, _handle_exit_signal)
except Exception:
    pass


def ensure_ollama_running(base_url: str = config.OLLAMA_BASE_URL, timeout: int = 8) -> bool:
    """Check if Ollama is running. If not, launch 'ollama serve' as a background daemon."""
    global _ollama_started_by_app, _ollama_process
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            print(f"[Ollama] Service is already active and responsive at {base_url}.")
            _ollama_started_by_app = False
            return True
    except Exception:
        pass

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        print("[Ollama] Warning: 'ollama' executable not found in system PATH.")
        return False

    print(f"[Ollama] Daemon not detected at {base_url}. Auto-starting '{ollama_bin} serve'...")
    try:
        env = os.environ.copy()
        if "127.0.0.1" in base_url or "localhost" in base_url:
            env["OLLAMA_HOST"] = "127.0.0.1:11434"
        _ollama_process = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True
        )
        _ollama_started_by_app = True
    except Exception as e:
        print(f"[Ollama] Failed to launch daemon: {e}")
        return False

    # Poll until responsive
    start_time = time.time()
    while time.time() - start_time < timeout:
        time.sleep(0.5)
        try:
            resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
            if resp.status_code == 200:
                print(f"[Ollama] Daemon started successfully and is ready at {base_url}!")
                return True
        except Exception:
            pass

    print(f"[Ollama] Warning: Daemon launched, but was not responsive within {timeout}s.")
    return False


# Auto-check and start Ollama on application boot
ensure_ollama_running()

# Auto-discover and embed skills into vector DB per SKILL_SPEC.md
try:
    auto_index_skills_into_db()
except Exception as e:
    print(f"[Skills] Notice: Skill discovery deferral ({e})")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)


@app.route("/")
def index():
    """Render the main 3-page / 3-tab user interface."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Check status of backend services, Ollama, and Gemma API."""
    ollama_info = check_ollama_health()
    gemma_ready = bool(config.GEMINI_API_KEY)
    
    return jsonify({
        "status": "online",
        "ollama": ollama_info,
        "gemma": {
            "api_key_configured": gemma_ready,
            "primary_model": config.GEMMA_PRIMARY_MODEL
        },
        "database_dir": str(config.DATABASE_DIR)
    })


@app.route("/api/models", methods=["GET"])
def get_chat_models():
    """Return list of text output models available in Google AI Studio."""
    models = gemma_service.get_available_text_models()
    default_model = next((m["id"] for m in models if m.get("is_default")), config.GEMMA_PRIMARY_MODEL)
    return jsonify({
        "status": "success",
        "default_model": default_model,
        "models": models
    })


@app.route("/api/ingest", methods=["POST"])
def ingest_documents():
    """
    Page 1 backend: Ingest documents from URL or local directory,
    create vector database via local Ollama embeddings, and persist into database/.
    """
    data = request.get_json(force=True, silent=True) or {}
    source_type = data.get("source_type", "").lower().strip()
    path = data.get("path", "").strip()
    chunk_size = int(data.get("chunk_size", 800))
    chunk_overlap = int(data.get("chunk_overlap", 120))

    if not source_type or not path:
        return jsonify({"error": "Both 'source_type' (url or directory) and 'path' are required."}), 400

    try:
        if source_type == "url":
            if not path.startswith("http://") and not path.startswith("https://"):
                path = "https://" + path
            documents = load_from_url(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif source_type == "directory":
            documents = load_from_directory(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            return jsonify({"error": f"Unsupported source_type '{source_type}'. Use 'url' or 'directory'."}), 400

        if not documents:
            return jsonify({
                "status": "warning",
                "message": f"No valid text documents found at {path}.",
                "added_chunks": 0
            }), 200

        # Add to document vector store in database/chroma_docs
        result = doc_vector_store.add_documents(documents)
        return jsonify({
            "status": "success",
            "message": f"Successfully ingested {len(documents)} chunks from {path} into private vector database.",
            "source_type": source_type,
            "path": path,
            "chunks_count": len(documents),
            "total_documents_in_db": result.get("total_documents_in_db", 0),
            "distinct_sources": result.get("distinct_sources", [])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query_rag():
    """
    Page 1 backend: Takes user question, searches private vector database,
    invokes skills/tools if applicable, generates grounded answer via Google AI Studio,
    and logs all Agent, LLM, and Tool invocations asynchronously into database/.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    top_k = int(data.get("top_k", 4))
    selected_model = data.get("model", "").strip() or None
    custom_endpoint = (data.get("custom_endpoint", "") or data.get("custom_model_api", "")).strip() or None
    conversation_id = data.get("conversation_id", "").strip() or f"conv-{uuid.uuid4().hex[:12]}"

    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    query_start_time = time.time()

    try:
        # 1. Agent Component: Log Agent Query (User -> Agent)
        agent_req_payload = {
            "question": question,
            "top_k": top_k,
            "model": selected_model,
            "custom_endpoint": custom_endpoint,
            "conversation_id": conversation_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        log_event(
            event_type="Agent Query",
            invoker="User",
            target="Agent",
            short_description=f"User query: \"{question[:70]}{'...' if len(question) > 70 else ''}\"",
            payload={
                "request": agent_req_payload,
                "response": {"status": "processing"}
            },
            conversation_id=conversation_id,
            status="success"
        )

        # Create an entry in database/conversations.json immediately when prompt is received
        log_conversation(
            conversation_id=conversation_id,
            user_query=question,
            agent_response="[Processing...]",
            timestamp=agent_req_payload["timestamp"],
            duration_ms=0.0
        )

        # Run the full Agent Plan -> Execute -> Synthesize pipeline
        pipeline_res = run_agent_skill_pipeline(
            question=question,
            model=selected_model,
            custom_endpoint=custom_endpoint,
            conversation_id=conversation_id,
            top_k=top_k
        )

        total_duration_ms = (time.time() - query_start_time) * 1000
        answer_text = pipeline_res.get("answer", "")
        sources = pipeline_res.get("sources", [])
        components = pipeline_res.get("components", [])

        # Log Agent Response
        agent_resp_payload = {
            "answer": answer_text,
            "model": pipeline_res.get("model"),
            "sources_count": len(sources),
            "total_duration_ms": round(total_duration_ms, 2)
        }
        log_event(
            event_type="Agent Response",
            invoker="Agent",
            target="User",
            short_description=f"Delivered response ({round(total_duration_ms)}ms)",
            payload={
                "request": agent_req_payload,
                "response": agent_resp_payload
            },
            conversation_id=conversation_id,
            duration_ms=total_duration_ms,
            status="success"
        )

        # Finalize conversation record in database/conversations.json
        log_conversation(
            conversation_id=conversation_id,
            user_query=question,
            agent_response=answer_text,
            timestamp=agent_req_payload["timestamp"],
            duration_ms=round(total_duration_ms, 2)
        )

        agent_component = {
            "name": "Agent",
            "role": "Orchestrator",
            "icon": "🤖",
            "status": "success",
            "duration_ms": round(total_duration_ms, 2),
            "description": f"Agent dispatched query through skill reasoning, tool execution, and grounded synthesis ({round(total_duration_ms)}ms)",
            "request": agent_req_payload,
            "response": agent_resp_payload
        }
        all_components = [agent_component] + components

        return jsonify({
            "status": "success",
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer_text,
            "model": pipeline_res.get("model"),
            "sources": sources,
            "duration_ms": round(total_duration_ms, 2),
            "components": all_components
        })

    except Exception as e:
        total_duration_ms = (time.time() - query_start_time) * 1000
        log_event(
            event_type="Agent Error",
            invoker="Agent",
            target="User",
            short_description=f"Query failure: {str(e)}",
            payload={"request": {"question": question}, "response": {"error": str(e)}},
            conversation_id=conversation_id,
            duration_ms=total_duration_ms,
            status="error"
        )
        log_conversation(
            conversation_id=conversation_id,
            user_query=question,
            agent_response=f"[Error: {str(e)}]",
            duration_ms=total_duration_ms
        )
        return jsonify({"error": str(e)}), 500


@app.route("/api/skills", methods=["GET"])
def get_skills_list():
    """Return discovered skills and their scripts/metadata."""
    skills = get_available_skills()
    return jsonify({
        "status": "success",
        "count": len(skills),
        "skills": skills
    })


@app.route("/api/skills/sync", methods=["POST"])
def sync_skills():
    """Manually trigger discovery and vector indexing of all skills."""
    result = auto_index_skills_into_db()
    return jsonify(result)


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    """Return all recorded user-agent conversations."""
    limit = int(request.args.get("limit", 100))
    convs = get_conversations(limit=limit)
    return jsonify({
        "status": "success",
        "count": len(convs),
        "conversations": convs
    })


@app.route("/api/conversations/<conv_id>/events", methods=["GET"])
def list_conversation_events(conv_id):
    """Return all invocation events for a specific conversation."""
    events = get_conversation_events(conv_id)
    return jsonify({
        "status": "success",
        "conversation_id": conv_id,
        "count": len(events),
        "events": events
    })


@app.route("/api/logs", methods=["GET"])
def logs():
    """
    Page 3 backend: Fetch logged calls and events from database/logs.json.
    """
    limit = int(request.args.get("limit", 200))
    call_type = request.args.get("type", "").strip() or None
    conv_id = request.args.get("conversation_id", "").strip() or None
    entries = get_logs(limit=limit, call_type=call_type, conversation_id=conv_id)
    return jsonify({
        "status": "success",
        "count": len(entries),
        "logs": entries
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Fetch comprehensive statistics about vector database and logs.
    """
    try:
        db_stats = vector_store.get_stats()
    except Exception as e:
        print(f"[Stats] Warning getting stats: {e}. Re-initializing client...")
        vector_store._init_client()
        db_stats = vector_store.get_stats()
    all_logs = get_logs(limit=1000)
    
    # Calculate log counts by type
    type_counts = {}
    total_calls = len(all_logs)
    success_calls = 0
    total_duration_ms = 0.0

    for l in all_logs:
        t = l.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        if l.get("status") == "success":
            success_calls += 1
        if l.get("duration_ms"):
            total_duration_ms += float(l["duration_ms"])

    avg_latency = round(total_duration_ms / total_calls, 1) if total_calls > 0 else 0

    return jsonify({
        "vector_store": db_stats,
        "logs_summary": {
            "total_logs": total_calls,
            "successful_calls": success_calls,
            "error_calls": total_calls - success_calls,
            "type_breakdown": type_counts,
            "avg_latency_ms": avg_latency
        }
    })


@app.route("/api/database/reset", methods=["POST"])
def reset_db():
    """Reset the private vector database."""
    success = vector_store.reset_database()
    if success:
        return jsonify({"status": "success", "message": "Vector database reset successfully."})
    return jsonify({"error": "Failed to reset vector database."}), 500


@app.route("/api/embedder/models", methods=["GET"])
def get_embedders():
    """Return currently active embedder model and catalog of supported Ollama models."""
    catalog = get_embedder_catalog(config.OLLAMA_BASE_URL)
    return jsonify({
        "status": "success",
        "current_model": catalog["current_model"],
        "models": catalog["models"]
    })


@app.route("/api/embedder/change", methods=["POST"])
def change_embedder():
    """
    Switch embedder model:
    Requires confirmation_phrase == 'Change to <new_model>'.
    Purges all documents in vector store, updates config in services/, and sets new active embedder.
    """
    data = request.get_json(force=True, silent=True) or {}
    new_model = data.get("new_model", "").strip()
    confirmation_phrase = data.get("confirmation_phrase", "").strip()

    if not new_model:
        return jsonify({"error": "Field 'new_model' is required."}), 400

    expected_phrase = f"Change to {new_model}"
    if confirmation_phrase != expected_phrase:
        return jsonify({
            "error": f"Confirmation phrase mismatch. Expected '{expected_phrase}', received '{confirmation_phrase}'."
        }), 400

    # Switch model in vector store (purges Chroma collection & updates services/embedder_config.json)
    success = vector_store.switch_embedder(new_model)
    if not success:
        return jsonify({"error": f"Failed to switch embedder to '{new_model}'."}), 500

    log_call(
        call_type="embedder_model_changed",
        arguments={
            "new_model": new_model,
            "confirmation_phrase": confirmation_phrase
        },
        response={
            "status": "purged_and_switched",
            "active_model": new_model
        },
        status="success"
    )

    return jsonify({
        "status": "success",
        "message": f"Successfully switched embedding model to '{new_model}'. Vector database was cleared.",
        "current_model": new_model
    })


@app.route("/api/logs/clear", methods=["POST"])
def clear_all_logs():
    """Clear database/logs.json."""
    success = clear_logs()
    if success:
        return jsonify({"status": "success", "message": "Audit logs cleared successfully."})
    return jsonify({"error": "Failed to clear audit logs."}), 500


@app.route("/api/shutdown", methods=["POST"])
def shutdown_app():
    """
    Shut down the application server cleanly.
    Terminates background Ollama daemon if started by this application instance,
    logs the event, and terminates the Flask server process.
    """
    log_event(
        event_type="Server Shutdown",
        invoker="User",
        target="System",
        short_description="User requested application shutdown via Web UI",
        payload={"request": {"action": "shutdown"}, "response": {"status": "shutting_down"}},
        status="success"
    )

    def _delayed_exit():
        time.sleep(0.5)
        shutdown_ollama_if_started_by_app()
        if app.testing or app.config.get("TESTING"):
            return
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as kill_err:
            print(f"[Shutdown] Signal error: {kill_err}")
            sys.exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()

    return jsonify({
        "status": "success",
        "message": "Agent with RAG server is shutting down. The application has been disabled."
    })


# Startup skill scanning and indexing into Skill Database
try:
    print("[Startup] Scanning skills/ folder and indexing into Skill Database...")
    skill_idx_res = auto_index_skills_into_db()
    print(f"[Startup] Skill Database initialized: {skill_idx_res}")
except Exception as e:
    print(f"[Startup] Warning during initial skill indexing: {e}")


if __name__ == "__main__":
    print(f"Starting Agent with RAG Server on port {config.PORT}...")
    app.run(host="0.0.0.0", port=config.PORT, debug=config.FLASK_DEBUG)
