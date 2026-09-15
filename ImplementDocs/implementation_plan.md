# Implementation Plan - Private RAG Web Application

Build a full-featured, private local RAG (Retrieval-Augmented Generation) system with a 3-tab Flask web application according to `SPECIFICATIONS.md`. The system indexes documents from local directories or web URLs into a private vector database stored in `database/`, performs semantic retrieval, generates answers using Gemma models hosted on Google AI Studio, and logs all container and external resource calls into `database/logs.json`.

## User Review Required

> [!IMPORTANT]
> - **Embedding Engine**: Local Ollama (`all-minilm`) is installed and configured in user space running on GPU/CPU for zero-leakage local vector embeddings. AnythingLLM or local container endpoints can also be configured.
> - **Google AI Studio Gemma Model**: Uses `models/gemma-4-26b-a4b-it` (or `models/gemma-4-31b-it`) via the official `google-genai` SDK with fallback resilience.
> - **Three Pages / Tabs**:
>   1. **Knowledge Ingestion (Vector DB Builder)**: Ingest documents from web URLs or local directory paths, chunk and embed into `database/chroma_db/`.
>   2. **Private RAG QA (Gemma Answers)**: User question input, semantic similarity search against local vector database, augmented prompt generation, and Gemma synthesis.
>   3. **Audit Logs & Vector Store Explorer**: Real-time view of `database/logs.json` (timestamps, call types, arguments, responses) and indexed collection metrics.

## Proposed Architecture

```mermaid
flowchart TD
    subgraph UI [Browser Client - 3-Tab Interface]
        Tab1[Tab 1: Vector DB Ingestion]
        Tab2[Tab 2: Private RAG Query]
        Tab3[Tab 3: System Logs & DB Inspector]
    end

    subgraph FlaskApp [Flask Backend Server (Python)]
        IngestAPI[/api/ingest/]
        QueryAPI[/api/query/]
        LogsAPI[/api/logs/]
        StatsAPI[/api/stats/]
        
        DocLoader[Document Parser: Web URL / Local Dir / PDF / TXT / MD]
        Chunker[Text Splitter & Normalizer]
        Embedder[Local Ollama Embedder (all-minilm)]
        VectorDB[(Persistent Chroma Vector Store in database/)]
        GemmaService[Google AI Studio Gemma Service]
        Logger[Structured Call Logger]
    end

    subgraph StorageAndLogs [Persistent Data in database/]
        DBDir[(database/chroma_db/)]
        LogsFile[(database/logs.json)]
    end

    Tab1 --> IngestAPI
    Tab2 --> QueryAPI
    Tab3 --> LogsAPI
    Tab3 --> StatsAPI

    IngestAPI --> DocLoader --> Chunker --> Embedder
    Embedder -- "Embed Chunks" --> VectorDB
    VectorDB --> DBDir

    QueryAPI --> Embedder -- "Embed Query" --> VectorDB
    VectorDB -- "Top K Relevant Chunks" --> GemmaService
    GemmaService -- "Google AI Studio API" --> ExternalGemma[Google AI Studio: Gemma]
    ExternalGemma --> GemmaService
    GemmaService --> QueryAPI

    Embedder -- "Log Call (Local Container/Daemon)" --> Logger
    DocLoader -- "Log URL Fetch (External)" --> Logger
    GemmaService -- "Log Model Call (External API)" --> Logger
    Logger --> LogsFile
```

---

## Proposed Changes

### 1. Configuration & Dependencies

#### [MODIFY] [requirements.txt](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/requirements.txt)
- Include all necessary Python packages: `flask`, `chromadb`, `google-genai`, `requests`, `beautifulsoup4`, `pypdf`, `python-dotenv`.

#### [NEW] [config.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/config.py)
- Configuration management:
  - Database directory: `database/`
  - Logs path: `database/logs.json`
  - Vector store path: `database/chroma_db`
  - Ollama host: `http://127.0.0.1:11434`
  - Default embedding model: `all-minilm`
  - Gemma model: `models/gemma-4-26b-a4b-it` (fallback: `models/gemma-4-31b-it`)
  - Server port: 5000

---

### 2. Services & Core Logic

#### [NEW] [services/logger_service.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/services/logger_service.py)
- Thread-safe structured logging to `database/logs.json` satisfying Requirements 6 & 7:
  - `time`: ISO 8601 UTC timestamp
  - `type`: call type (e.g. `ollama_embedding`, `google_ai_studio_gemma`, `external_url_fetch`, `vllm_container`)
  - `arguments`: full parameters passed (URL, prompt, parameters, model name, text preview, etc.)
  - `response`: parsed response summary, status, tokens/output or error
  - `duration_ms`: execution time in milliseconds.

#### [NEW] [services/ollama_embedder.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/services/ollama_embedder.py)
- Handles vector embeddings via local Ollama (or local container):
  - Sends text batches to `http://127.0.0.1:11434/api/embeddings`
  - Automatically logs each embedding call to `database/logs.json`
  - Handles batching, timeouts, and health checks.

#### [NEW] [services/document_loader.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/services/document_loader.py)
- Handles ingestion from:
  - Web URLs: HTTP GET with polite User-Agent, parses HTML using BeautifulSoup, extracts main article content, logs external fetch to `database/logs.json`.
  - Local Directories: Recursively scans directory for `.txt`, `.md`, `.pdf`, `.json`, `.csv`, `.py`, extracts clean text and metadata.
  - Chunking: Recursive chunking with configurable overlap (default 600 chars with 100 overlap).

#### [NEW] [services/vector_store.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/services/vector_store.py)
- Manages persistent ChromaDB vector collection stored in `database/chroma_db/`:
  - Inserts document chunks with embeddings and metadata (source, chunk index, doc title).
  - Performs similarity query using query vector and returns top-k documents with distance scores.
  - Provides collection statistics (total documents, sources, chunk counts).

#### [NEW] [services/gemma_service.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/services/gemma_service.py)
- Manages Gemma generation on Google AI Studio:
  - Builds grounded prompt with retrieved vector database chunks and user question.
  - Calls `google.genai` client using `GEMINI_API_KEY`.
  - Logs the call details (prompt tokens, parameters, response text, finish reason) to `database/logs.json`.

---

### 3. Web Application & UI

#### [NEW] [app.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/app.py)
- Flask web application routes:
  - `GET /`: Main multi-tab UI.
  - `POST /api/ingest`: Accepts `{"source_type": "url" | "directory", "path": "..."}`, ingests documents, stores embeddings in `database/`, returns result stats.
  - `POST /api/query`: Accepts `{"question": "...", "top_k": 4}`, retrieves chunks, queries Gemma on Google AI Studio, returns answer and sources.
  - `GET /api/logs`: Returns entries from `database/logs.json` with filtering and pagination.
  - `GET /api/stats`: Returns vector DB statistics and logging metrics.
  - `POST /api/logs/clear`: Utility to clear logs if requested.
  - `POST /api/database/reset`: Utility to reset vector DB if requested.

#### [NEW] [templates/index.html](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/templates/index.html)
- Clean, semantic HTML structure with tab switcher at top of window:
  - **Tab 1: Vector DB Builder**: URL or directory input with auto-validation, chunk settings, live ingestion progress bar, document preview table.
  - **Tab 2: Private RAG Query**: Interactive question box, sample queries, response markdown viewer, cited context accordion with similarity scores.
  - **Tab 3: System Logs & Audit**: Statistics cards, real-time log table with filtering by call type, inspect payload modal, copy-to-clipboard.

#### [NEW] [static/css/style.css](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/static/css/style.css)
- Premium modern design system:
  - Curated dark theme (`#0d1117`, `#161b22`, `#21262d`, accent cyan `#38bdf8` and purple `#818cf8`).
  - Google Fonts (Inter + JetBrains Mono for code/logs).
  - Smooth tab switching animations and micro-interactions.
  - Responsive tables, badge tags, and styled modal overlays.

#### [NEW] [static/js/app.js](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/static/js/app.js)
- Asynchronous UI controller:
  - Tab state management.
  - Ingestion submission and live status feedback.
  - Question query handler with typing feedback and rendered sources.
  - Logs viewer with auto-refresh and modal payload viewer.

---

## Verification Plan

### Automated Tests
1. **Embedding & Logging Verification**:
   - Run python test script verifying Ollama embedding generation and checking that `database/logs.json` records the call with timestamp, type, arguments, and response.
2. **Document Ingestion Test**:
   - Ingest a sample local directory and a sample URL, verifying vector persistence in `database/chroma_db/`.
3. **Gemma Query Test**:
   - Query a question related to the ingested documents, verifying Google AI Studio Gemma response generation and corresponding log entry in `database/logs.json`.
4. **Flask Endpoint Tests**:
   - Test `GET /`, `POST /api/ingest`, `POST /api/query`, `GET /api/logs`, `GET /api/stats` with `curl` / `pytest`.

### Manual Browser Verification
- Open the web application in browser:
  1. Test tab switching between Tab 1, Tab 2, and Tab 3.
  2. Ingest a local directory (e.g. `./sample_docs/`) and a URL.
  3. Query a question on Tab 2 and verify answer + citations.
  4. Switch to Tab 3 and inspect logged calls in `database/logs.json` with payload modal.
