# Private RAG (Retrieval-Augmented Generation) System

A private, locally hosted Retrieval-Augmented Generation (RAG) web application with an air-gapped vector database architecture and Google AI Studio Gemma synthesis.

---

## 1. What This System Does

Private RAG enables users to build and query a private knowledge base with complete control over document storage and vector embeddings:

- **Private Vector Database Creation**: Ingests documents from either **live Web URLs** or **local directory paths** (supporting `.txt`, `.md`, `.pdf`, `.py`, `.json`, `.csv`, etc.), partitions them into semantically cohesive text chunks, generates dense vector embeddings locally using **Ollama** (`all-minilm` or any selected embedder), and persists them into a local ChromaDB database stored inside `database/chroma_db/`.
- **Grounded AI Synthesis & Model Selection**: Takes user questions, retrieves the top semantically relevant passages from the local vector database, and generates accurate, cited answers via any selected text model on Google AI Studio (e.g. **Gemma 4 26B A4B IT**, **Gemini 2.5 Flash**, **Gemini 2.5 Pro**, etc.).
- **Backend-Only External Calls**: All external API calls (to Google AI Studio and web scraping) are initiated strictly from the backend Python code, ensuring no sensitive tokens, raw requests, or direct connections originate from the browser client.
- **Audit Logging**: Logs every container/daemon interaction (Ollama embeddings) and external API call (Google AI Studio, Web URL fetch) into `database/logs.json` with timestamp, call type, arguments passed, and response received (excluding raw vectors and non-text binary data).
- **Three-Page Switchable Interface**: Features an intuitive web UI with switchable tabs at the top of the window for **1. Chat**, **2. Vector DB Ingestion**, and **3. Audit Logs & DB Explorer**.

---

## 🏗️ System Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │      3-Tab Browser User Interface       │
                                  └────┬──────────────┬──────────────┬──────┘
                                       │              │              │
                         [Tab 1: Chat] │[Tab 2:Ingest]│ [Tab 3: Logs]│
                                       ▼              ▼              ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        Flask Backend Server (Python)                      │
│                                                                           │
│  • Document Loader (BeautifulSoup / PyPDF / Directory Scanner)            │
│  • Vector Store Controller (ChromaDB in database/chroma_db/)              │
│  • Google AI Studio Client (Gemma / Gemini Models)                        │
│  • Structured Call Logger (database/logs.json)                            │
└───────┬───────────────────────────────┬───────────────────────────┬───────┘
        │                               │                           │
        ▼                               ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────┐
│     Local Ollama      │   │   Google AI Studio    │   │  database/ folder │
│  (all-minilm Embeds)  │   │  (Gemma / Gemini)     │   │                   │
│   127.0.0.1:11434     │   │     External API      │   │ • chroma_db/      │
│  [Auto-Start Daemon]  │   │    [External API]     │   │ • logs.json       │
└───────────────────────┘   └───────────────────────┘   └───────────────────┘
```

---

## 2. How to Install the Components Needed (Run only Once)

### System Prerequisites
- **Linux**: Ubuntu 22.04+ (or Debian, Fedora, Arch)
- **macOS**: macOS 12 (Monterey) or later (Apple Silicon M1/M2/M3/M4 or Intel)
- **Windows PC**: Windows 10 or 11 (64-bit) with PowerShell, Command Prompt, or WSL2
- **Python**: Python 3.10 to 3.14 (with `pip` and `venv` installed)
- **Hardware**: Dedicated GPU (NVIDIA CUDA on Linux/Windows, Apple Silicon Metal on macOS) or CPU

---

### Step 1: Install Ollama (Local Embedding Engine)
Ollama runs locally as a daemon/service to compute vector embeddings without sending document text to external clouds:

#### Option A: On Linux
```bash
# 1. Download and install Ollama binary
curl -fsSL https://ollama.com/install.sh | sh

# 2. Prevent Ollama from auto-starting on system boot (app.py will auto-start it when needed)
sudo systemctl disable ollama
```

#### Option B: On macOS
- **Using Homebrew**:
  ```bash
  brew install ollama
  ```
- **Or Direct Download**:
  Download the official macOS application bundle from [ollama.com/download/mac](https://ollama.com/download/mac), unzip, and drag `Ollama.app` to your Applications folder.

#### Option C: On Windows PC
1. Download the Windows installer (`OllamaSetup.exe`) from [ollama.com/download/windows](https://ollama.com/download/windows).
2. Run `OllamaSetup.exe` and complete the installer wizard. This automatically adds `ollama.exe` to your Windows system `PATH`.
3. Open a new PowerShell or Command Prompt terminal and verify installation:
   ```powershell
   ollama --version
   ```

---

### Step 2: Set Up Python Virtual Environment
Navigate to the project root and create a virtual environment:

#### On Linux & macOS:
```bash
cd /path/to/privateRAG

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### On Windows PC (PowerShell):
```powershell
cd C:\path\to\privateRAG

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# (Note: If PowerShell displays a script execution policy error, run:
#  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# Upgrade pip
python -m pip install --upgrade pip
```

#### On Windows PC (Command Prompt - `cmd.exe`):
```cmd
cd C:\path\to\privateRAG

:: Create virtual environment
python -m venv .venv

:: Activate virtual environment
.\.venv\Scripts\activate.bat

:: Upgrade pip
python -m pip install --upgrade pip
```

---

### Step 3: Install Python Dependencies
With your virtual environment activated, install dependencies from `requirements.txt`:

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

---

### Step 4: Configure Environment Variables (Run Only Once)
Create or edit the `.env` file in the root directory of `privateRAG`:

```bash
GEMINI_API_KEY=your-google-ai-studio-api-key-here
PORT=8005
FLASK_DEBUG=False
```

> **Note**: A free Gemini API key for Gemma can be obtained at [Google AI Studio](https://aistudio.google.com/).

---

## 3. How to Start All the Services

### Automatic Ollama Auto-Start & Launching the Web App

The application backend (`app.py`) is designed with **automatic service lifecycle management**. When you start `app.py`, it automatically checks whether the local Ollama daemon is active and responsive at `http://127.0.0.1:11434`. 
- If Ollama is not running, **`app.py` will automatically start `ollama serve` in the background** and wait until it is healthy before serving web traffic.
- If Ollama is already running, `app.py` seamlessly connects to it.

#### Starting on Linux & macOS:
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the application
python app.py
```

#### Starting on Windows PC (PowerShell):
```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run the application
python app.py
```

*Console Output:*
```
[Ollama] Daemon not detected at http://127.0.0.1:11434. Auto-starting 'ollama serve'...
[Ollama] Daemon started successfully and is ready at http://127.0.0.1:11434!
Starting Private RAG Server on port 8005...
 * Running on http://127.0.0.1:8005
```

> [!NOTE]
> **Intelligent Ollama Process Lifecycle:**
> - If `app.py` started the Ollama service on boot, it will **automatically terminate the Ollama background process** when the application exits (e.g., on `Ctrl + C`).
> - If Ollama was already running prior to starting `app.py`, the app will **leave Ollama running untouched** when the application exits.

### Access the Web Application
Open your browser and navigate to:
**`http://127.0.0.1:8005`**

---

## 4. Embedding Model Configuration & Dynamic Switching

### Persistent Embedder Configuration
The active embedding model is tracked and persisted in **`services/embedder_config.json`**.
- When `app.py` boots, it reads `services/embedder_config.json`.
- If the file does not exist, it is automatically initialized with the default model (`all-minilm`).

### Dynamic Model Dropdown & Safety Confirmation Modal
In **Tab 2: Vector DB Ingestion**, the *Document Source Configuration* header provides a dropdown selector with compatible Ollama embedding models:

1. **Model Switch Warning**: Because different models output vectors with different dimensionalities and coordinate spaces (e.g., `all-minilm` is 384-d, `nomic-embed-text` is 768-d), previously ingested documents cannot be searched with a new model.
2. **Safety Confirmation**: When a user selects a new model:
   - A modal dialog appears warning that **all previously ingested documents will not work and will be permanently deleted from the Vector Storage DB**.
   - The user must type the exact phrase: `Change to <the new model>` (e.g., `Change to nomic-embed-text`).
   - The **OK** button remains disabled until the exact phrase is entered.
   - Clicking **Cancel** restores the dropdown to the previous model without altering the database.
   - Clicking **OK** executes the purge of `database/chroma_db/`, updates `services/embedder_config.json`, and activates the new embedder.

### Supported Ollama Embedding Models

The application displays a live reference catalog at the bottom of the Ingestion page:

| Model Identifier | Dimensions | Context Window | Model Size | Best For | Ollama Command |
|---|---|---|---|---|---|
| **`all-minilm`** | 384 | 512 tokens | 45 MB | Default / Low resource CPU usage | `ollama pull all-minilm` |
| **`nomic-embed-text`** | 768 | 8,192 tokens | 274 MB | **Recommended for RAG** & long articles | `ollama pull nomic-embed-text` |
| **`bge-m3`** | 1024 | 8,192 tokens | 1.2 GB | Multilingual (100+ languages) | `ollama pull bge-m3` |
| **`bge-large`** | 1024 | 512 tokens | 670 MB | High accuracy English retrieval | `ollama pull bge-large` |
| **`mxbai-embed-large`** | 1024 | 512 tokens | 670 MB | Leading MTEB benchmark performance | `ollama pull mxbai-embed-large` |
| **`snowflake-arctic-embed`** | 1024 | 512 tokens | 670 MB | Production enterprise search | `ollama pull snowflake-arctic-embed` |
| **`paraphrase-multilingual`** | 384 | 512 tokens | 280 MB | Lightweight multilingual | `ollama pull paraphrase-multilingual` |

---

## 5. How to Shutdown All the Services

When you are finished using the application, shut down the running services using the following steps:

### Stopping the Flask Web Application
- **If running in the foreground (terminal)**:
  Press `Ctrl + C` in the terminal running `app.py`. (If `app.py` started Ollama, it shuts it down automatically).
- **If running in the background**:
  - **Linux / macOS**:
    ```bash
    fuser -k 8005/tcp 2>/dev/null || kill $(lsof -t -i:8005) 2>/dev/null || pkill -f "python app.py"
    ```
  - **Windows PC (PowerShell)**:
    ```powershell
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
    ```

### Stopping the Local Ollama Daemon (If started manually)
- **On Linux / macOS**:
  ```bash
  pkill -f "ollama serve" || pkill ollama
  # Or if managed by systemd:
  sudo systemctl stop ollama
  ```
- **On Windows PC (PowerShell)**:
  ```powershell
  Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
  ```

### One-Line Shutdown Command
- **On Linux / macOS**:
  ```bash
  pkill -f "python app.py" && pkill -f "ollama serve"
  ```
- **On Windows PC (PowerShell)**:
  ```powershell
  Stop-Process -Name python,ollama -Force -ErrorAction SilentlyContinue
  ```

---

## 6. User Guide: How to Use the System

The application interface consists of three switchable tabs located at the top of the window:

### Tab 1: Chat (Ask Questions & Select Google AI Studio Models)
1. **Select Model**:
   - In the Chat header, pick any text generation model available from Google AI Studio (e.g., `Gemma 4 26B A4B IT`, `Gemini 2.5 Flash`, `Gemini 2.5 Pro`, `Gemma 4 31B IT`, `Gemini 2.5 Flash-Lite`).
2. **Enter Question**:
   - Type a question related to your indexed documents in the input box.
3. **Choose Top-K Retrieval**:
   - Select how many relevant document chunks to inject into the model context (`3`, `4`, `6`, or `8`).
4. **Send Question**:
   - Click **Send Question** or press `Enter`.
   - The system retrieves semantically matching chunks from `database/chroma_db/`, constructs a grounded prompt, queries your selected model on Google AI Studio, and displays the answer.
5. **Review Evidence**:
   - The right-hand panel shows the exact retrieved context passages, their source filenames/URLs, and cosine similarity percentages.

### Tab 2: Vector DB Ingestion (Build Knowledge Base)
1. **Select Source Type**:
   - **🌐 Web URL**: Ingest an online webpage or documentation link (e.g., `https://en.wikipedia.org/wiki/Artificial_intelligence`). The backend fetches the HTML, strips scripts and headers, and extracts the core text.
   - **📁 Local Directory**: Ingest a local directory (e.g., `./sample_docs` or `/home/user/docs`). The backend recursively finds and extracts text from `.txt`, `.md`, `.pdf`, `.py`, and `.json` files.
2. **Select Embedder Model**:
   - Pick an Ollama embedding model from the *Document Source Configuration* header dropdown (e.g. `all-minilm`, `nomic-embed-text`, `bge-m3`).
3. **Configure Chunking (Optional)**:
   - Expand *Advanced Chunking Parameters* to adjust **Chunk Size** (default: `800` characters) and **Overlap** (default: `120` characters).
4. **Index Documents**:
   - Click **Generate Vector Database**.
   - The status panel displays real-time progress while local Ollama computes vector embeddings.
   - All vectors and metadata are stored persistently in `database/chroma_db/`.

### Tab 3: Audit Logs & DB Explorer (Inspect Conversations & Invocations)
1. **Live Metrics**:
   - View total conversation count, logged events, LLM / Tool calls, and average call latency.
2. **Table 1: User Conversations**:
   - Lists all conversations between user and agent with columns: **Conversation ID**, **Timestamp**, **User Query**, and **Agent Response**.
   - **Interactive Selection**: Click any conversation row to view all chronological events that occurred during that conversation in Table 2 below.
3. **Table 2: Conversation Events Timeline**:
   - Displays all invocation events with columns: **Timestamp**, **Event Type**, **Invoker**, **Target**, and **Short Description**.
   - Filters events by type (`Agent Query`, `LLM Invocation`, `Tool Invocation`, `Vector Search`, `Agent Response`, `Ollama Embedding`, `External API`) or search keywords.
   - Click **Show All Events** to reset filter back to the full event stream across all conversations.
4. **Detailed JSON Payload Modal**:
   - Click any event row in Table 2 to open the detailed JSON log viewer.
   - Shows human-readable, formatted JSON with all arguments passed and responses received.
   - **Security**: Sensitive tokens and Google AI Studio API keys are automatically redacted to `[REDACTED_API_KEY]`.
   - **Asynchronous Execution & 30MB Ceiling**: Writes asynchronously in a background worker thread without blocking agent execution; automatically trims older entries if the log file approaches 30MB.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status (Ollama daemon, Gemma model, DB path) |
| `GET` | `/api/models` | List all text output generation models currently available in Google AI Studio |
| `GET` | `/api/embedder/models` | List supported Ollama embedding models with local installation status |
| `POST` | `/api/embedder/change` | Switch active embedding model and purge vector storage database |
| `POST` | `/api/ingest` | Ingest from URL or directory. Payload: `{"source_type": "url"|"directory", "path": "...", "chunk_size": 800, "chunk_overlap": 120}` |
| `POST` | `/api/query` | Semantic search and grounded answer. Payload: `{"question": "...", "top_k": 4, "model": "..."}` |
| `GET` | `/api/conversations` | List all recorded user-agent conversations (Table 1) |
| `GET` | `/api/conversations/<id>/events` | List all invocation events for a specific conversation (Table 2) |
| `GET` | `/api/logs` | Fetch audit logs from `database/logs.json`. Query params: `?limit=200&type=...&conversation_id=...` |
| `GET` | `/api/stats` | Database chunk count, unique sources, storage size, and call counts |
| `GET` | `/api/skills` | List discovered agent skills and associated scripts |
| `POST` | `/api/skills/sync` | Discover and embed newly added skills into vector database |
| `POST` | `/api/database/reset` | Clear all vector embeddings and recreate the collection |
| `POST` | `/api/logs/clear` | Clear all entries in `database/logs.json` and `database/conversations.json` |

---

## 7. Agent Skills Architecture (`SKILL_SPEC.md`)

The agent supports modular domain skills following the specification in `SKILL_SPEC.md`. When the agent starts up, it auto-discovers all skills in `skills/`, uses local Ollama to embed each `SKILL.md` into ChromaDB, and retrieves skill procedures to answer domain-specific questions in the Chat interface.

```
skills/
├── time-weather-skill/
│   ├── SKILL.md            # Metadata & SOP for city time & weather
│   └── scripts/
│       └── env_tools.py    # Public Open-Meteo & Time logic (Zero API Key Req.)
└── stock-market-skill/
    ├── SKILL.md            # Metadata & SOP for stock screener
    ├── data/
    │   └── registry.csv    # Flat-file database for record resolution
    └── scripts/
        └── stock_tools.py  # Public market screener & gainers/losers calculator
```

### Skill 1: Time and Weather (`skills/time-weather-skill`)
- **Capability**: Queries local time, timezone, temperature, weather conditions, relative humidity, and wind speed for any city worldwide using the public Open-Meteo API.
- **No API Key Required**: Fully open and rate-limit friendly.
- **CLI Runner**:
  ```bash
  python skills/time-weather-skill/scripts/env_tools.py "Tokyo"
  python skills/time-weather-skill/scripts/env_tools.py "New York" --unit fahrenheit
  ```

### Skill 2: Stock Market Screener (`skills/stock-market-skill`)
- **Capability**: Identifies equities with the **highest percentage increase** (top gainers) or the **lowest percentage decrease** (top losers / biggest drops).
- **Public Feeds & Local Fallback**: Uses public market screeners with automatic failover to `skills/stock-market-skill/data/registry.csv`.
- **CLI Runner**:
  ```bash
  python skills/stock-market-skill/scripts/stock_tools.py --gainers --limit 5
  python skills/stock-market-skill/scripts/stock_tools.py --losers --limit 5
  ```

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
5. Flask API endpoints (`/`, `/api/health`, `/api/ingest`, `/api/query`, `/api/logs`, `/api/stats`, `/api/skills`).

