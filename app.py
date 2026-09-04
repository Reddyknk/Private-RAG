import os
from flask import Flask, render_template, request, jsonify

import config
from services.logger_service import get_logs, clear_logs
from services.ollama_embedder import check_ollama_health
from services.document_loader import load_from_url, load_from_directory
from services.vector_store import vector_store
from services.gemma_service import gemma_service

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

        # Add to vector store in database/
        result = vector_store.add_documents(documents)
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
    Page 2 backend: Takes user question, searches private vector database,
    and generates grounded answer using Gemma on Google AI Studio.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    top_k = int(data.get("top_k", 4))

    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    try:
        # 1. Semantic search in private vector DB
        retrieved_chunks = vector_store.query(question, top_k=top_k)

        if not retrieved_chunks:
            return jsonify({
                "answer": "No documents have been indexed into the private vector database yet, or no relevant matches were found. Please ingest a URL or local directory first in Tab 1.",
                "sources": [],
                "model": "none",
                "duration_ms": 0
            })

        # 2. Call Google AI Studio Gemma model from backend Python code
        response = gemma_service.answer_question(question, retrieved_chunks)

        return jsonify({
            "status": "success",
            "question": question,
            "answer": response.get("answer"),
            "model": response.get("model"),
            "sources": response.get("sources"),
            "duration_ms": response.get("duration_ms"),
            "log_id": response.get("log_id")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs", methods=["GET"])
def logs():
    """
    Page 3 backend: Fetch logged calls from database/logs.json.
    """
    limit = int(request.args.get("limit", 100))
    call_type = request.args.get("type", "").strip() or None
    entries = get_logs(limit=limit, call_type=call_type)
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


@app.route("/api/logs/clear", methods=["POST"])
def clear_all_logs():
    """Clear database/logs.json."""
    success = clear_logs()
    if success:
        return jsonify({"status": "success", "message": "Audit logs cleared successfully."})
    return jsonify({"error": "Failed to clear audit logs."}), 500


if __name__ == "__main__":
    print(f"Starting Private RAG Server on port {config.PORT}...")
    app.run(host="0.0.0.0", port=config.PORT, debug=config.FLASK_DEBUG)
