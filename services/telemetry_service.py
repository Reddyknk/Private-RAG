import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import LOGS_FILE


def parse_iso_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse various ISO 8601 timestamp formats into UTC datetime."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    clean = ts_str.strip()
    try:
        # Handle trailing Z
        if clean.endswith("Z"):
            clean = clean[:-1] + "+00:00"
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ):
            try:
                dt = datetime.strptime(clean, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
    return None


def extract_llm_events_from_logs(logs_path: Path = LOGS_FILE) -> List[Dict[str, Any]]:
    """
    Extract and normalize all LLM-related call records from database/logs.json.
    Excludes non-LLM calls (e.g. embedder-only, vector search, health checks).
    """
    if not logs_path.exists():
        return []

    try:
        with open(logs_path, "r", encoding="utf-8") as f:
            raw_logs = json.load(f)
    except Exception as e:
        print(f"[TelemetryService] Error reading {logs_path}: {e}")
        return []

    llm_events: List[Dict[str, Any]] = []

    for entry in raw_logs:
        if not isinstance(entry, dict):
            continue

        evt_type = str(entry.get("event_type") or entry.get("type") or entry.get("call_type") or "")
        invoker = str(entry.get("invoker") or "")
        target = str(entry.get("target") or "")
        args = entry.get("arguments") or entry.get("payload") or {}
        resp = entry.get("response") or {}

        # Identify LLM invocation calls
        is_llm = (
            evt_type in ("LLM", "LLM Invocation", "google_ai_studio_gemma", "custom_model_api")
            or "Google AI Studio" in target
            or "Custom Private LLM" in target
            or (evt_type == "Agent Query" and ("model" in args or "model_used" in resp))
        )
        if not is_llm:
            continue

        # Extract model name
        model = (
            resp.get("model_used")
            or resp.get("model")
            or args.get("model")
            or args.get("selected_model")
            or ""
        )
        if not model:
            if "custom_model_api" in evt_type or "Custom" in target:
                model = "Custom Model API"
            elif "gemma" in evt_type or "Google" in target:
                model = "Google AI Studio"
            else:
                model = "Standard LLM"

        # Determine status / errors
        status = str(entry.get("status") or resp.get("status") or "success").lower()
        is_error = status in ("error", "failed", "failure") or bool(resp.get("error"))

        # Extract tokens
        usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
        input_tokens = (
            usage.get("prompt_token_count")
            or usage.get("prompt_tokens")
            or usage.get("input_tokens")
        )
        if input_tokens is None:
            prompt_chars = args.get("prompt_length_chars") or len(str(args.get("question", "") or args.get("prompt", "")))
            input_tokens = max(1, prompt_chars // 4) if prompt_chars > 0 else 0

        output_tokens = (
            usage.get("candidates_token_count")
            or usage.get("completion_tokens")
            or usage.get("output_tokens")
        )
        if output_tokens is None:
            ans_chars = resp.get("answer_chars") or len(str(resp.get("answer", "") or resp.get("text", "") or resp.get("reply", "")))
            output_tokens = max(1, ans_chars // 4) if ans_chars > 0 and not is_error else 0

        ts_dt = parse_iso_timestamp(entry.get("timestamp") or entry.get("time"))
        if not ts_dt:
            continue

        llm_events.append({
            "id": entry.get("id"),
            "timestamp": ts_dt,
            "model": model,
            "is_error": is_error,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "duration_ms": entry.get("duration_ms")
        })

    # Sort chronologically
    llm_events.sort(key=lambda x: x["timestamp"])
    return llm_events


def get_telemetry_data(
    model: Optional[str] = None,
    interval: str = "15m",
    time_range: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    logs_path: Path = LOGS_FILE
) -> Dict[str, Any]:
    """
    Computes top summary statistics and time series bucketing for dual plots:
    - Left plot: Prompts, Responses, Errors per interval.
    - Right plot: Input Tokens, Output Tokens per interval.
    """
    events = extract_llm_events_from_logs(logs_path)

    # Collect distinct model names for dropdown
    all_models = sorted(list({e["model"] for e in events if e.get("model")}))

    # Model filtering
    active_model = model.strip() if model and isinstance(model, str) else ""
    is_all_models = not active_model or active_model.lower() in ("all", "all models", "all_models")

    if not is_all_models:
        filtered_events = [e for e in events if e["model"] == active_model]
    else:
        filtered_events = events

    # 1. Lifetime / Total Summary Statistics for the selected model
    total_prompts = len(filtered_events)
    total_errors = sum(1 for e in filtered_events if e["is_error"])
    total_responses = total_prompts - total_errors
    total_input_tokens = sum(e["input_tokens"] for e in filtered_events)
    total_output_tokens = sum(e["output_tokens"] for e in filtered_events)

    summary = {
        "total_prompts": total_prompts,
        "total_responses": total_responses,
        "total_errors": total_errors,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens
    }

    # 2. Determine Time Window [window_start, window_end]
    now = datetime.now(timezone.utc)
    tr = time_range.lower().strip()

    if tr in ("last hr", "last_hr", "1h", "last hour", "lasthr"):
        window_start = now - timedelta(hours=1)
        window_end = now
    elif tr in ("week", "7d", "7 days", "1 week", "1w"):
        window_start = now - timedelta(days=7)
        window_end = now
    elif tr in ("month", "30d", "30 days", "1 month", "1m_range"):
        window_start = now - timedelta(days=30)
        window_end = now
    elif tr == "custom" and start_date and end_date:
        try:
            s_parts = [int(x) for x in start_date.strip().split("-")]
            e_parts = [int(x) for x in end_date.strip().split("-")]
            window_start = datetime(s_parts[0], s_parts[1], s_parts[2], 0, 0, 0, tzinfo=timezone.utc)
            window_end = datetime(e_parts[0], e_parts[1], e_parts[2], 23, 59, 59, tzinfo=timezone.utc)
            if window_start > window_end:
                window_start, window_end = window_end, window_start
        except Exception:
            window_start = now - timedelta(days=1)
            window_end = now
    else:
        # Default: 1 day (24 hours)
        window_start = now - timedelta(days=1)
        window_end = now

    # Interval seconds
    inv = interval.lower().strip()
    if inv in ("1 min", "1min", "1m"):
        interval_seconds = 60
    elif inv in ("1 hr", "1hr", "1h", "1 hour"):
        interval_seconds = 3600
    elif inv in ("1 day", "1day", "1d", "day"):
        interval_seconds = 86400
    else:
        # Default: 15 min
        interval_seconds = 900

    # Ensure valid duration
    total_duration = max(60, int((window_end - window_start).total_seconds()))
    num_buckets = math.ceil(total_duration / interval_seconds)

    # Cap buckets at 600 for rendering performance
    if num_buckets > 600:
        interval_seconds = math.ceil(total_duration / 600)
        num_buckets = math.ceil(total_duration / interval_seconds)

    # Align starting bucket timestamp to interval boundary
    start_epoch = int(window_start.timestamp())
    bucket_start_epoch = (start_epoch // interval_seconds) * interval_seconds
    end_epoch = int(window_end.timestamp())

    # Build bucket list
    buckets: List[Dict[str, Any]] = []
    curr_epoch = bucket_start_epoch
    while curr_epoch <= end_epoch:
        buckets.append({
            "start_epoch": curr_epoch,
            "end_epoch": curr_epoch + interval_seconds,
            "prompts": 0,
            "responses": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0
        })
        curr_epoch += interval_seconds

    # Bucket index lookup
    if buckets:
        base_epoch = buckets[0]["start_epoch"]
        for e in filtered_events:
            e_ts = int(e["timestamp"].timestamp())
            if e_ts < base_epoch or e_ts > end_epoch + interval_seconds:
                continue
            b_idx = (e_ts - base_epoch) // interval_seconds
            if 0 <= b_idx < len(buckets):
                b = buckets[b_idx]
                b["prompts"] += 1
                if e["is_error"]:
                    b["errors"] += 1
                else:
                    b["responses"] += 1
                b["input_tokens"] += e["input_tokens"]
                b["output_tokens"] += e["output_tokens"]

    # Format labels
    labels = []
    prompts_data = []
    responses_data = []
    errors_data = []
    input_tokens_data = []
    output_tokens_data = []

    # Choose label format based on span
    span_days = total_duration / 86400.0
    for b in buckets:
        dt = datetime.fromtimestamp(b["start_epoch"], tz=timezone.utc)
        if span_days <= 1.5:
            lbl = dt.strftime("%H:%M")
        elif span_days <= 8.0:
            lbl = dt.strftime("%b %d %H:%M") if interval_seconds < 86400 else dt.strftime("%b %d")
        else:
            lbl = dt.strftime("%b %d")

        labels.append(lbl)
        prompts_data.append(b["prompts"])
        responses_data.append(b["responses"])
        errors_data.append(b["errors"])
        input_tokens_data.append(b["input_tokens"])
        output_tokens_data.append(b["output_tokens"])

    return {
        "status": "success",
        "models": all_models,
        "selected_model": "all" if is_all_models else active_model,
        "filters": {
            "model": active_model or "all",
            "interval": interval,
            "time_range": time_range,
            "start_date": start_date,
            "end_date": end_date,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "interval_seconds": interval_seconds,
            "total_buckets": len(buckets)
        },
        "summary": summary,
        "time_series": {
            "labels": labels,
            "prompts": prompts_data,
            "responses": responses_data,
            "errors": errors_data,
            "input_tokens": input_tokens_data,
            "output_tokens": output_tokens_data
        }
    }
