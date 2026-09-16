import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
LOGS_FILE = DATABASE_DIR / "logs.json"
CHROMA_PERSIST_DIR = DATABASE_DIR / "chroma_db"
CHROMA_DOCS_DIR = DATABASE_DIR / "chroma_docs"
CHROMA_SKILLS_DIR = DATABASE_DIR / "chroma_skills"

# Ensure active database directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

# Google AI Studio configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Default to latest high-capability Gemma models on Google AI Studio
GEMMA_PRIMARY_MODEL = os.getenv("GEMMA_MODEL", "models/gemma-4-26b-a4b-it")
GEMMA_FALLBACK_MODEL = os.getenv("GEMMA_FALLBACK_MODEL", "models/gemma-4-31b-it")

# Local Ollama / container embedding service configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")

# Web server configuration
PORT = int(os.getenv("PORT", 8005)) # Port 8005 for Private RAG app
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")

# Maximum number of tokens to generate. The local model may be a smaller model or quantized model.
MAX_TOKEN_LOCAL=1024
MAX_TOKEN_EXTERNAL=2048