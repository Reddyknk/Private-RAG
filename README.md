# Private RAG (Retrieval-Augmented Generation) System

A private, locally hosted Retrieval-Augmented Generation (RAG) web application with an air-gapped vector database architecture and Google AI Studio Gemma synthesis.

---

## 1. What This System Does

Private RAG enables users to build and query a private knowledge base with complete control over document storage and vector embeddings:

- **Private Vector Database Creation**: Ingests documents from either **live Web URLs** or **local directory paths** (supporting `.txt`, `.md`, `.pdf`, `.py`, `.json`, `.csv`, etc.), partitions them into semantically cohesive text chunks, generates dense vector embeddings locally using **Ollama** (`all-minilm`), and persists them into a local ChromaDB database stored inside `database/chroma_db/`.
- **Grounded Gemma Synthesis**: Takes user questions, retrieves the top semantically relevant passages from the local vector database, and generates accurate, cited answers via **Gemma** (`models/gemma-4-26b-a4b-it`) hosted on Google AI Studio.
- **Backend-Only External Calls**: All external API calls (to Google AI Studio and web scraping) are initiated strictly from the backend Python code, ensuring no sensitive tokens, raw requests, or direct connections originate from the browser client.
- **Audit Logging**: Logs every container/daemon interaction (Ollama embeddings) and external API call (Google AI Studio Gemma, Web URL fetch) into `database/logs.json` with timestamp, call type, arguments passed, and response received (excluding raw vectors and non-text binary data).
- **Three-Page Switchable Interface**: Features an intuitive web UI with switchable tabs at the top of the window for Vector DB Ingestion, Gemma QA Chat, and Audit Logs Inspection.

---

## 🏗️ System Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │      3-Tab Browser User Interface       │
                                  └────┬──────────────┬──────────────┬──────┘
                                       │              │              │
                       [Tab 1: Ingest] │ [Tab 2: RAG] │ [Tab 3: Logs]│
                                       ▼              ▼              ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        Flask Backend Server (Python)                      │
│                                                                           │
│  • Document Loader (BeautifulSoup / PyPDF / Directory Scanner)            │
│  • Vector Store Controller (ChromaDB in database/chroma_db/)             │
│  • Gemma Client (Google AI Studio SDK)                                    │
│  • Structured Call Logger (database/logs.json)                            │
└───────┬───────────────────────────────┬───────────────────────────┬───────┘
        │                               │                           │
        ▼                               ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────┐
│     Local Ollama      │   │   Google AI Studio    │   │  database/ folder │
│  (all-minilm Embeds)  │   │  (Gemma-4-26b-a4b-it) │   │                   │
│   127.0.0.1:11434     │   │     External API      │   │ • chroma_db/      │
│  [Container / Daemon] │   │    [External API]     │   │ • logs.json       │
└───────────────────────┘   └───────────────────────┘   └───────────────────┘
```

---

## 2. How to Install the Components Needed

### System Prerequisites
- Linux (Ubuntu 22.04+ recommended) or macOS
- Python 3.10 to 3.14
- NVIDIA GPU with CUDA (optional, accelerates local embedding inference) or CPU

### Step 1: Set Up Python Virtual Environment
Navigate to the project root and create a virtual environment:

```bash
cd /path/to/privateRAG

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Python Dependencies
Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key packages installed:
- `flask`: Web server framework and API routing
- `chromadb`: Persistent local vector database stored in `database/`
- `google-genai`: Google AI Studio client for Gemma generation
- `requests` & `beautifulsoup4`: Web page fetching and HTML content extraction
- `pypdf`: Text extraction from local and online PDF files
- `python-dotenv`: Management of environment configuration

### Step 3: Install Ollama (Local Embedding Engine)
Ollama runs locally as a daemon/service to compute vector embeddings without data leakage:

```bash
# Download and install Ollama binary
curl -fsSL https://ollama.com/install.sh | sh
```
Prevent Ollama from starting on boot
```bash
sudo systemctl disable ollama
```

### Step 4: Configure Environment Variables
Create or edit `.env` in the root of the project directory:

```bash
GEMINI_API_KEY=your-google-ai-studio-api-key-here
PORT=5050
FLASK_DEBUG=False
```

> **Note**: An API key can be obtained for free at [Google AI Studio](https://aistudio.google.com/).

---

## 3. How to Start All the Services

To run the full Private RAG application, start the two primary backend services in order:

### Service 1: Start Local Ollama Daemon & Pull Model
In your terminal, start the Ollama server and download the embedding model:

```bash
# 1. Start the Ollama background daemon
OLLAMA_HOST=127.0.0.1:11434 ollama serve > /dev/null 2>&1 &

# 2. Verify Ollama is responding
curl -s http://127.0.0.1:11434/api/version
# Output: {"version":"0.33.x"}

# 3. Pull the lightweight embedding model (all-minilm, ~45MB)
ollama pull all-minilm
```

### Service 2: Start the Flask Web Application
Activate the Python virtual environment and run `app.py`:

```bash
source .venv/bin/activate
python app.py
```

*Output:*
```
Starting Private RAG Server on port 5050...
 * Running on http://127.0.0.1:5050
```

### Access the Web Application
Open your browser and navigate to:
**`http://127.0.0.1:5050`**

---

## 4. How to Shutdown All the Services

When you are finished using the application, shut down the running services using the following steps:

### Stopping the Flask Web Application
- **If running in the foreground (terminal)**:
  Press `Ctrl + C` in the terminal running `app.py`.
- **If running in the background**:
  ```bash
  # Terminate Flask app running on port 5050
  fuser -k 5050/tcp 2>/dev/null || kill $(lsof -t -i:5050) 2>/dev/null || pkill -f "python app.py"
  ```

### Stopping the Local Ollama Daemon
- **If started in background / user shell**:
  ```bash
  pkill -f "ollama serve" || pkill ollama
  ```
- **If installed as a systemd service**:
  ```bash
  sudo systemctl stop ollama
  ```

### One-Line Shutdown Command
To stop all Private RAG services simultaneously:
```bash
pkill -f "python app.py" && pkill -f "ollama serve"
```

---

## 5. User Guide: How to Use the System

The application interface consists of three switchable tabs located at the top of the window:

### Tab 1: Vector DB Ingestion (Build Knowledge Base)
1. **Select Source Type**:
   - **🌐 Web URL**: Ingest an online webpage or documentation link (e.g., `https://en.wikipedia.org/wiki/Artificial_intelligence`). The backend fetches the HTML, strips scripts and headers, and extracts the core text.
   - **📁 Local Directory**: Ingest a local directory (e.g., `./sample_docs` or `/home/user/docs`). The backend recursively finds and extracts text from `.txt`, `.md`, `.pdf`, `.py`, and `.json` files.
2. **Configure Chunking (Optional)**:
   - Expand *Advanced Chunking Parameters* to adjust **Chunk Size** (default: `800` characters) and **Overlap** (default: `120` characters).
3. **Index Documents**:
   - Click **Generate Vector Database**.
   - The status panel displays real-time progress while local Ollama computes vector embeddings.
   - All vectors and metadata are stored persistently in `database/chroma_db/`.

### Tab 2: Gemma RAG Query (Ask Questions)
1. **Enter Question**:
   - In the text area, type a question related to your indexed documents (e.g., *"What are the core capabilities of autonomous AI agents?"*).
2. **Choose Top-K Retrieval**:
   - Select how many relevant document chunks to inject into the Gemma context (`3`, `4`, `6`, or `8`).
3. **Ask Gemma**:
   - Click **Ask Gemma** or press `Enter`.
   - The system retrieves semantically matching chunks from `database/chroma_db/`, constructs a grounded prompt, queries the Gemma model on Google AI Studio, and displays the answer.
4. **Review Evidence**:
   - The right-hand panel shows the exact retrieved context passages, their source filenames/URLs, and cosine similarity percentages.

### Tab 3: Audit Logs & DB Explorer (Inspect Calls)
1. **Live Metrics**:
   - View total call counts, Google AI Studio Gemma calls, local Ollama embedding batches, and average response latency.
2. **Filter & Search**:
   - Filter logs by call type (`Google AI Studio (Gemma)`, `Local Ollama Batch Embed`, `Web URL Fetch`).
   - Search arguments or response text using the search filter input.
3. **Inspect Call Details**:
   - Click **Inspect** on any row to open the JSON payload modal displaying the exact arguments sent and response data received.
   - Click **Copy Args** or **Copy Response** to copy sanitized JSON to the clipboard.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status (Ollama daemon, Gemma model, DB path). |
| `POST` | `/api/ingest` | Ingest from URL or directory. Payload: `{"source_type": "url"\|"directory", "path": "...", "chunk_size": 800, "chunk_overlap": 120}` |
| `POST` | `/api/query` | Semantic search and Gemma response. Payload: `{"question": "...", "top_k": 4}` |
| `GET` | `/api/logs` | Fetch audit logs from `database/logs.json`. Query params: `?limit=100&type=...` |
| `GET` | `/api/stats` | Database chunk count, unique sources, storage size, and call counts. |
| `POST` | `/api/database/reset` | Clear all vector embeddings and recreate the collection. |
| `POST` | `/api/logs/clear` | Clear all entries in `database/logs.json`. |

---

## 🧪 Running Automated Tests

A comprehensive unit test suite is included in `tests/test_private_rag.py`:

```bash
source .venv/bin/activate
python -m unittest tests/test_private_rag.py
```

Verified test coverage:
1. Local Ollama embedding generation and `database/logs.json` verification.
2. Local directory recursive scanning, chunking, and ChromaDB vector persistence in `database/chroma_db/`.
3. Vector similarity query retrieval.
4. Google AI Studio Gemma answer generation and call logging.
5. Flask API endpoints (`/`, `/api/health`, `/api/ingest`, `/api/query`, `/api/logs`, `/api/stats`).
