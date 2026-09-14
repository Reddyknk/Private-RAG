import os
import re
import json
import time
import queue
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

LOGS_FILE = config.LOGS_FILE
CONVERSATIONS_FILE = config.DATABASE_DIR / "conversations.json"
MAX_LOG_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB ceiling per requirement
SAFE_TRIM_SIZE_BYTES = int(28.5 * 1024 * 1024)  # Trim down to 28.5 MB when ceiling breached

_disk_lock = threading.Lock()
_async_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()


def _ensure_files_exist():
    """Ensure database directory and JSON files exist."""
    config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    if not LOGS_FILE.exists() or LOGS_FILE.stat().st_size == 0:
        with open(LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
    if not CONVERSATIONS_FILE.exists() or CONVERSATIONS_FILE.stat().st_size == 0:
        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


_ensure_files_exist()


# ==========================================
# Redaction & Serialization Engine
# ==========================================
def _redact_string(s: str) -> str:
    """Redact sensitive API keys, auth tokens, and secret parameters from a string."""
    if not s or not isinstance(s, str):
        return s

    # 1. Redact exact configured GEMINI_API_KEY if present
    gemini_key = getattr(config, "GEMINI_API_KEY", "")
    if gemini_key and len(gemini_key) > 5 and gemini_key in s:
        s = s.replace(gemini_key, "[REDACTED_API_KEY]")

    # 2. Redact standard Google AI Studio / Gemini API key regex (AIza + 30-45 chars)
    s = re.sub(r"AIza[0-9A-Za-z-_]{30,45}", "[REDACTED_API_KEY]", s)

    # 3. Redact Bearer tokens
    s = re.sub(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED_TOKEN]", s)

    # 4. Redact URL query parameters matching key=, api_key=, token=, secret=
    s = re.sub(r"([?&](?:api_key|apikey|key|token|auth|secret)=)[^&\s]+", r"\1[REDACTED_API_KEY]", s, flags=re.IGNORECASE)

    return s


def _safe_serialize_and_redact(obj: Any) -> Any:
    """
    Recursively sanitize objects:
    - Exclude dense vector float arrays and raw binary data.
    - Redact API keys and authorization secrets.
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, str):
        return _redact_string(obj)
    if isinstance(obj, bytes):
        return f"<binary data length={len(obj)} bytes excluded>"
    if isinstance(obj, (list, tuple)):
        # Exclude dense numeric vectors (list of floats/ints length > 5)
        if len(obj) > 5 and all(isinstance(x, (int, float)) for x in obj[:6]):
            return f"<dense vector dimensions={len(obj)} excluded>"
        return [_safe_serialize_and_redact(item) for item in obj]
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            k_str = str(k)
            k_lower = k_str.lower()
            # If key explicitly refers to vector embeddings, exclude raw array
            if any(term in k_lower for term in ("embedding", "embeddings", "vector", "vectors")) and isinstance(v, (list, tuple)):
                cleaned[k_str] = f"<vector data excluded (count={len(v)})>"
                continue
            # Redact if key indicates an API key, secret, token, authorization
            if any(term in k_lower for term in ("api_key", "apikey", "secret", "password", "x-goog-api-key")) or k_lower in ("key", "token", "authorization", "auth"):
                cleaned[k_str] = "[REDACTED_API_KEY]"
            else:
                cleaned[k_str] = _safe_serialize_and_redact(v)
        return cleaned
    if hasattr(obj, "to_dict"):
        return _safe_serialize_and_redact(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _safe_serialize_and_redact(obj.__dict__)
    return _redact_string(str(obj))


# ==========================================
# 30MB File Size Enforcement & Trimming
# ==========================================
def _trim_to_fit(records: List[Dict[str, Any]], max_bytes: int = MAX_LOG_SIZE_BYTES, target_bytes: int = SAFE_TRIM_SIZE_BYTES) -> List[Dict[str, Any]]:
    """
    If list JSON exceeds max_bytes (30MB), delete older records from the beginning
    to make room for new entries until within target_bytes.
    """
    if not records:
        return records

    try:
        serialized = json.dumps(records, indent=2, ensure_ascii=False).encode("utf-8")
        if len(serialized) <= max_bytes:
            return records

        print(f"[Logger] Log file size ({len(serialized)} bytes) exceeds 30MB ceiling. Trimming older entries...")
        while len(records) > 1 and len(serialized) > target_bytes:
            drop_count = max(1, len(records) // 10)
            records = records[drop_count:]
            serialized = json.dumps(records, indent=2, ensure_ascii=False).encode("utf-8")

        print(f"[Logger] Pruned logs to {len(records)} entries ({len(serialized)} bytes).")
        return records
    except Exception as e:
        print(f"[Logger] Warning during log trimming: {e}")
        return records


# ==========================================
# Asynchronous Background Worker
# ==========================================
def _worker_loop():
    """Worker thread that asynchronously writes log events and conversations to disk."""
    while True:
        task = _async_queue.get()
        try:
            task_type = task.get("task_type")
            if task_type == "log_event":
                _write_event_to_disk(task["entry"])
            elif task_type == "log_conversation":
                _write_conversation_to_disk(task["entry"])
            elif task_type == "clear_logs":
                _clear_logs_on_disk()
        except Exception as e:
            print(f"[Logger Worker] Error processing log task: {e}")
        finally:
            _async_queue.task_done()


def _write_event_to_disk(entry: Dict[str, Any]):
    """Persist event to database/logs.json under lock with 30MB ceiling check."""
    _ensure_files_exist()
    with _disk_lock:
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

        logs.append(entry)
        logs = _trim_to_fit(logs, max_bytes=MAX_LOG_SIZE_BYTES)

        temp_file = LOGS_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        temp_file.replace(LOGS_FILE)


def _write_conversation_to_disk(entry: Dict[str, Any]):
    """Persist or update conversation record in database/conversations.json."""
    _ensure_files_exist()
    with _disk_lock:
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                conversations = json.load(f)
        except Exception:
            conversations = []

        # Check if conversation already exists to update it, or append
        conv_id = entry.get("conversation_id")
        existing_idx = next((i for i, c in enumerate(conversations) if c.get("conversation_id") == conv_id), None)
        if existing_idx is not None:
            conversations[existing_idx].update(entry)
        else:
            conversations.append(entry)

        conversations = _trim_to_fit(conversations, max_bytes=MAX_LOG_SIZE_BYTES)

        temp_file = CONVERSATIONS_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        temp_file.replace(CONVERSATIONS_FILE)


def _clear_logs_on_disk():
    """Wipe database/logs.json and database/conversations.json."""
    _ensure_files_exist()
    with _disk_lock:
        try:
            with open(LOGS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
            with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
        except Exception as e:
            print(f"[Logger] Error clearing files on disk: {e}")


# Launch daemon thread
_worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="AsyncLoggerThread")
_worker_thread.start()


# ==========================================
# Public Logging API
# ==========================================
def log_event(
    event_type: str,
    invoker: str,
    target: str,
    short_description: str,
    payload: Dict[str, Any],
    conversation_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    status: str = "success",
    legacy_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Log an event asynchronously into database/logs.json.
    Non-blocking: enqueues task and returns immediately.
    Guarantees structured request and response fields.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    clean_payload = _safe_serialize_and_redact(payload or {})

    # Ensure standardized 'request' and 'response' fields exist in payload without circular references
    if not isinstance(clean_payload, dict):
        clean_payload = {"value": clean_payload}

    req_data = clean_payload.get("request")
    if req_data is None:
        if "arguments" in clean_payload and isinstance(clean_payload["arguments"], dict):
            req_data = dict(clean_payload["arguments"])
        else:
            req_data = {k: v for k, v in clean_payload.items() if k not in ("request", "response")}
        clean_payload["request"] = req_data

    if "response" not in clean_payload:
        clean_payload["response"] = {}

    arguments = clean_payload.get("arguments") or clean_payload.get("request", {})
    response = clean_payload.get("response", {})

    entry = {
        "id": f"evt-{uuid.uuid4().hex[:12]}",
        "conversation_id": conversation_id or "system",
        "timestamp": now_iso,
        "time": now_iso,  # Backwards compatibility
        "event_type": event_type,
        "type": legacy_type or event_type,  # Backwards compatibility
        "invoker": invoker,
        "target": target,
        "short_description": _redact_string(short_description),
        "status": status,
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        "payload": clean_payload,
        "arguments": arguments,  # Backwards compatibility
        "response": response     # Backwards compatibility
    }

    # Asynchronous enqueue - returns instantly without blocking agent execution
    _async_queue.put({"task_type": "log_event", "entry": entry})
    return entry


def log_conversation(
    conversation_id: str,
    user_query: str,
    agent_response: str,
    timestamp: Optional[str] = None,
    duration_ms: Optional[float] = None,
    events_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Record or update a user-agent conversation turn in database/conversations.json asynchronously.
    """
    now_iso = timestamp or datetime.now(timezone.utc).isoformat()
    entry = {
        "conversation_id": conversation_id,
        "timestamp": now_iso,
        "user_query": _redact_string(user_query),
        "agent_response": _redact_string(agent_response),
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
        "events_count": events_count if events_count is not None else 0
    }

    _async_queue.put({"task_type": "log_conversation", "entry": entry})
    return entry


def log_call(
    call_type: str,
    arguments: Dict[str, Any],
    response: Any,
    duration_ms: Optional[float] = None,
    status: str = "success",
    conversation_id: Optional[str] = None,
    invoker: Optional[str] = None,
    target: Optional[str] = None,
    short_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper that translates legacy log_call into a rich structured event.
    """
    resolved_invoker = invoker or "Agent"
    resolved_target = target or "External Service"
    resolved_desc = short_description or f"Executed {call_type}"
    resolved_event_type = call_type

    if call_type == "google_ai_studio_gemma":
        resolved_event_type = "LLM"
        model_name = arguments.get("model", "Google Gemma")
        resolved_invoker = invoker or "Agent"
        resolved_target = target or f"Google AI Studio ({model_name})"
        resolved_desc = short_description or f"Grounded response query sent to {model_name}"
    elif call_type in ("local_ollama_embedding", "local_ollama_embed_batch"):
        resolved_event_type = "Embedder"
        resolved_invoker = invoker or "Agent"
        resolved_target = target or "Local Ollama (127.0.0.1:11434)"
        resolved_desc = short_description or f"Computed vector embeddings ({call_type})"
    elif call_type == "external_url_fetch":
        resolved_event_type = "External API"
        url = arguments.get("url", "")
        resolved_invoker = invoker or "Agent"
        resolved_target = target or (f"Web URL ({url[:45]}...)" if len(url) > 45 else f"Web URL ({url})")
        resolved_desc = short_description or f"Fetched external webpage content from {url[:50]}"
    elif call_type == "tool_invocation":
        resolved_event_type = "Tool"
        resolved_invoker = invoker or "Agent"
        resolved_target = target or arguments.get("tool", "Skill Tool")
        resolved_desc = short_description or f"Skill tool execution ({arguments.get('tool', 'tool')})"

    payload = {
        "request": arguments,
        "response": response,
        "arguments": arguments
    }

    return log_event(
        event_type=resolved_event_type,
        invoker=resolved_invoker,
        target=resolved_target,
        short_description=resolved_desc,
        payload=payload,
        conversation_id=conversation_id,
        duration_ms=duration_ms,
        status=status,
        legacy_type=call_type
    )


# ==========================================
# Retrieval & Query Functions
# ==========================================
def flush_logs(timeout: Optional[float] = 5.0):
    """Flush pending asynchronous log operations to disk."""
    _async_queue.join()


def get_logs(limit: int = 200, call_type: Optional[str] = None, conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve logged events in reverse chronological order."""
    flush_logs()
    _ensure_files_exist()
    with _disk_lock:
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                logs: List[Dict[str, Any]] = json.load(f)
        except Exception:
            logs = []

    if conversation_id:
        logs = [entry for entry in logs if entry.get("conversation_id") == conversation_id]

    if call_type:
        logs = [entry for entry in logs if entry.get("type") == call_type or entry.get("event_type") == call_type]

    return list(reversed(logs[-limit:]))


def get_conversations(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve conversations list in reverse chronological order."""
    flush_logs()
    _ensure_files_exist()
    with _disk_lock:
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                conversations: List[Dict[str, Any]] = json.load(f)
        except Exception:
            conversations = []

    return list(reversed(conversations[-limit:]))


def get_conversation_events(conversation_id: str) -> List[Dict[str, Any]]:
    """Retrieve all events that occurred during a specific conversation in chronological order."""
    flush_logs()
    _ensure_files_exist()
    with _disk_lock:
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                logs: List[Dict[str, Any]] = json.load(f)
        except Exception:
            logs = []

    events = [entry for entry in logs if entry.get("conversation_id") == conversation_id]
    return events


def clear_logs() -> bool:
    """Clear all records in database/logs.json and database/conversations.json."""
    flush_logs()
    _async_queue.put({"task_type": "clear_logs"})
    flush_logs()
    return True
