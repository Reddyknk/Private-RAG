import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import requests

from config import BASE_DIR, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

SERVICES_DIR = BASE_DIR / "services"
CONFIG_FILE = SERVICES_DIR / "embedder_config.json"

DEFAULT_MODEL = OLLAMA_EMBED_MODEL or "all-minilm"

# Curated catalog of embedding models supported by Ollama
SUPPORTED_EMBEDDERS: List[Dict[str, Any]] = [
    {
        "id": "all-minilm",
        "name": "All-MiniLM-L6-v2",
        "dimensions": 384,
        "size": "45 MB",
        "context_window": "512 tokens",
        "description": "Ultra-lightweight and blazingly fast. Ideal for general document retrieval with minimal memory footprint.",
        "recommended_for": "Default / Low-resource environments"
    },
    {
        "id": "nomic-embed-text",
        "name": "Nomic Embed Text v1.5",
        "dimensions": 768,
        "size": "274 MB",
        "context_window": "8,192 tokens",
        "description": "High-performance open embedding model. Large context window makes it exceptional for long documents and complex RAG queries.",
        "recommended_for": "Recommended for RAG / Long Documents"
    },
    {
        "id": "bge-m3",
        "name": "BAAI BGE-M3",
        "dimensions": 1024,
        "size": "1.2 GB",
        "context_window": "8,192 tokens",
        "description": "State-of-the-art multilingual model supporting over 100 languages with dense and multi-aspect retrieval.",
        "recommended_for": "Multilingual & Cross-lingual RAG"
    },
    {
        "id": "bge-large",
        "name": "BAAI BGE-Large (en)",
        "dimensions": 1024,
        "size": "670 MB",
        "context_window": "512 tokens",
        "description": "High accuracy English embedding model with deep semantic representation for specialized corpora.",
        "recommended_for": "High-accuracy English retrieval"
    },
    {
        "id": "mxbai-embed-large",
        "name": "Mixedbread AI Embed Large",
        "dimensions": 1024,
        "size": "670 MB",
        "context_window": "512 tokens",
        "description": "Ranked at the top of the MTEB benchmark, outperforming many proprietary embedding APIs.",
        "recommended_for": "Competitive benchmark accuracy"
    },
    {
        "id": "snowflake-arctic-embed",
        "name": "Snowflake Arctic Embed",
        "dimensions": 1024,
        "size": "670 MB",
        "context_window": "512 tokens",
        "description": "Trained specifically for production enterprise search and high-recall information retrieval.",
        "recommended_for": "Enterprise Search & Retrieval"
    },
    {
        "id": "paraphrase-multilingual",
        "name": "Paraphrase Multilingual MiniLM",
        "dimensions": 384,
        "size": "280 MB",
        "context_window": "512 tokens",
        "description": "Lightweight multilingual model mapped onto 384 dimensions across 50+ languages.",
        "recommended_for": "Lightweight Multilingual"
    }
]


def init_embedder_config() -> Dict[str, Any]:
    """Ensure embedder_config.json exists in services/. If missing, create with default model."""
    SERVICES_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        initial_data = {
            "model": DEFAULT_MODEL,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)
            return initial_data
        except Exception as e:
            print(f"Failed to create {CONFIG_FILE}: {e}")
            return {"model": DEFAULT_MODEL}
    
    # Read existing
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "model" not in data or not data["model"]:
                data["model"] = DEFAULT_MODEL
            return data
    except Exception as e:
        print(f"Error reading {CONFIG_FILE}: {e}")
        return {"model": DEFAULT_MODEL}


def get_current_embedder_model() -> str:
    """Return the active embedder model name."""
    cfg = init_embedder_config()
    return cfg.get("model", DEFAULT_MODEL)


def save_embedder_model(model_name: str) -> bool:
    """Persist the new embedder model to services/embedder_config.json."""
    SERVICES_DIR.mkdir(parents=True, exist_ok=True)
    clean_model = model_name.strip()
    data = {
        "model": clean_model,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving to {CONFIG_FILE}: {e}")
        return False


def get_installed_ollama_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """Fetch currently pulled model tags from local Ollama."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            return [m.get("name", "") for m in models_data if m.get("name")]
    except Exception:
        pass
    return []


def get_embedder_catalog(base_url: str = OLLAMA_BASE_URL) -> Dict[str, Any]:
    """Return the active model, supported models catalog, and local installation status."""
    current_model = get_current_embedder_model()
    installed = get_installed_ollama_models(base_url)
    
    catalog = []
    for item in SUPPORTED_EMBEDDERS:
        model_id = item["id"]
        # Match e.g. "all-minilm" against "all-minilm:latest" or "all-minilm"
        is_installed = any(
            model_id == inst or inst.startswith(f"{model_id}:")
            for inst in installed
        )
        catalog.append({
            **item,
            "installed": is_installed,
            "is_current": (model_id == current_model or current_model.startswith(f"{model_id}:"))
        })
        
    return {
        "current_model": current_model,
        "models": catalog
    }
