import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from config import LOGS_FILE

_lock = threading.Lock()


def _ensure_logs_file_exists():
    """Ensure database/logs.json exists and is initialized as an empty JSON list if empty."""
    LOGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LOGS_FILE.exists() or LOGS_FILE.stat().st_size == 0:
        with open(LOGS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def _safe_serialize(obj: Any) -> Any:
    """Recursively sanitize objects so they are JSON serializable, excluding vectors and non-text binary data."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, bytes):
        return f"<binary data length={len(obj)} bytes excluded>"
    if isinstance(obj, (list, tuple)):
        # If this is a numeric vector (list of floats/ints with length > 5), exclude the raw vector
        if len(obj) > 5 and all(isinstance(x, (int, float)) for x in obj[:6]):
            return f"<dense vector dimensions={len(obj)} excluded>"
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            key_lower = str(k).lower()
            # If key explicitly refers to raw embeddings/vectors, exclude the vector array
            if any(term in key_lower for term in ("embedding", "embeddings", "vector", "vectors")) and isinstance(v, (list, tuple)):
                cleaned[str(k)] = f"<vector data excluded (count={len(v)})>"
            else:
                cleaned[str(k)] = _safe_serialize(v)
        return cleaned
    if hasattr(obj, "to_dict"):
        return _safe_serialize(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _safe_serialize(obj.__dict__)
    return str(obj)


def log_call(
    call_type: str,
    arguments: Dict[str, Any],
    response: Any,
    duration_ms: Optional[float] = None,
    status: str = "success"
) -> Dict[str, Any]:
    """
    Log an external API or container call into database/logs.json.
    
    Requirements:
    - Time of the call
    - Type of the call
    - Arguments passed to the call
    - Response from the call
    """
    _ensure_logs_file_exists()

    entry = {
        "id": str(uuid.uuid4()),
        "time": datetime.now(timezone.utc).isoformat(),
        "type": call_type,
        "status": status,
        "arguments": _safe_serialize(arguments),
        "response": _safe_serialize(response),
        "duration_ms": round(duration_ms, 2) if duration_ms is not None else None
    }

    with _lock:
        try:
            try:
                with open(LOGS_FILE, "r", encoding="utf-8") as f:
                    logs: List[Dict[str, Any]] = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs = []

            logs.append(entry)

            # Atomic write to temporary file then replace
            temp_file = LOGS_FILE.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            temp_file.replace(LOGS_FILE)
        except Exception as e:
            print(f"[Logger] Error writing to {LOGS_FILE}: {e}")

    return entry


def get_logs(limit: int = 100, call_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve logged calls in reverse chronological order."""
    _ensure_logs_file_exists()
    with _lock:
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                logs: List[Dict[str, Any]] = json.load(f)
        except Exception:
            logs = []

    if call_type:
        logs = [entry for entry in logs if entry.get("type") == call_type]

    # Return most recent first
    return list(reversed(logs[-limit:]))


def clear_logs() -> bool:
    """Clear all records in database/logs.json."""
    with _lock:
        try:
            with open(LOGS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
            return True
        except Exception as e:
            print(f"[Logger] Error clearing logs: {e}")
            return False
