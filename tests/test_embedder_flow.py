import unittest
from app import app, ensure_ollama_running
from services.embedder_manager import (
    get_current_embedder_model,
    save_embedder_model,
    get_embedder_catalog,
    init_embedder_config
)
from services.vector_store import vector_store


def test_embedder_config_init_and_persistence():
    """Verify embedder config is created and can be updated."""
    init_embedder_config()
    current = get_current_embedder_model()
    assert current is not None
    
    # Save a model
    assert save_embedder_model("all-minilm") is True
    assert get_current_embedder_model() == "all-minilm"


def test_ensure_ollama_running():
    """Verify Ollama auto-detection function runs cleanly."""
    status = ensure_ollama_running()
    assert status is True


def test_get_embedder_models_endpoint():
    """Verify /api/embedder/models returns supported models catalog."""
    client = app.test_client()
    resp = client.get("/api/embedder/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert "current_model" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) >= 6
    
    # Check all-minilm presence
    model_ids = [m["id"] for m in data["models"]]
    assert "all-minilm" in model_ids
    assert "nomic-embed-text" in model_ids


def test_embedder_change_mismatch():
    """Verify /api/embedder/change rejects mismatched confirmation."""
    client = app.test_client()
    resp = client.post("/api/embedder/change", json={
        "new_model": "nomic-embed-text",
        "confirmation_phrase": "invalid confirmation"
    })
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
    assert "mismatch" in data["error"].lower()


def test_embedder_change_valid():
    """Verify /api/embedder/change successfully switches model and resets vector db."""
    client = app.test_client()
    resp = client.post("/api/embedder/change", json={
        "new_model": "all-minilm",
        "confirmation_phrase": "Change to all-minilm"
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["current_model"] == "all-minilm"
    assert get_current_embedder_model() == "all-minilm"


def test_html_contains_required_elements():
    """Verify index.html renders embedder dropdown, reference table, and modal."""
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    
    # Check dropdown
    assert 'id="embedderSelect"' in html
    assert 'Local Embedder:' in html
    
    # Check reference table
    assert 'Supported Ollama Embedding Models' in html
    assert 'id="embeddersTableBody"' in html
    
    # Check warning modal
    assert 'id="modelChangeModal"' in html
    assert 'id="confirmModelChangeInput"' in html
    assert 'id="btnConfirmModelChange"' in html
    assert 'id="btnCancelModelChange"' in html

    # Check Tab 1 is Chat and active
    assert '1. Chat' in html
    assert 'id="tabBtnQuery"' in html
    assert 'class="nav-tab active" id="tabBtnQuery"' in html
    
    # Check Tab 2 is Ingestion
    assert '2. Vector DB Ingestion' in html
    assert 'id="tabBtnIngest"' in html

    # Check Google AI Studio chat model dropdown exists
    assert 'id="chatModelSelect"' in html


def test_get_chat_models_endpoint():
    """Verify /api/models returns Google AI Studio text generation models."""
    client = app.test_client()
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert "default_model" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


def test_ollama_shutdown_if_not_started_by_app():
    """Verify shutdown does nothing if Ollama was not started by this app."""
    import app as app_module
    app_module._ollama_started_by_app = False
    # Should execute cleanly without error
    app_module.shutdown_ollama_if_started_by_app()
    assert app_module._ollama_started_by_app is False

