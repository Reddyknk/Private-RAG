# Dual Vector Store & Agent Skill Execution Architecture

Refactor the system from a single vector store to two distinct vector store databases: a **Skill Database** for procedural knowledge (`SKILL.md`) and a **Document Database** for user/corporate documents, governed by a two-step agent execution loop (Plan $\to$ Execute $\to$ Synthesize).

---

## User Review Required

> [!IMPORTANT]
> **Dual Database Separation**:
> The persistent storage will be split into two separate ChromaDB stores:
> 1. `database/chroma_skills` (Skill Database)
> 2. `database/chroma_docs` (Document Database)
> Existing document data and skill data will be migrated and cleanly separated.

> [!IMPORTANT]
> **Agent Execution Flow**:
> 1. When a user prompt is received, the prompt is embedded using local Ollama and queried against the **Skill Database**.
> 2. **No Skill Match ($\le 50\%$)**: The prompt is routed directly to the LLM model with the exact required system prompt:
>    `"You are a helpful, respectful, and honest assistant. Always answer as truthfully as possible, use clear markdown formatting, and admit when you do not know an answer rather than guessing."`
> 3. **Skill Match ($> 50\%$)**:
>    - **Step 1 (Plan/Instruction)**: The Agent presents the user prompt and matched `SKILL.md` full text to the LLM. The LLM returns a plan or execution instruction (e.g. tool execution command with parameters).
>    - **Step 2 (Execution)**: The Agent performs the instruction (executes `scripts/doc_tools.py`, `scripts/stock_tools.py`, or `scripts/env_tools.py`).
>    - **Step 3 (Synthesis)**: The Agent sends the execution result, the original prompt, and the `SKILL.md` to the LLM for final grounded synthesis.

---

## Proposed Changes

### Configuration & Storage Layer

#### [MODIFY] [config.py](file:///home/pi-net/Documents/agent_eng_labs/Agent-with-RAG/config.py)
- Define `CHROMA_DOCS_DIR = DATABASE_DIR / "chroma_docs"`
- Define `CHROMA_SKILLS_DIR = DATABASE_DIR / "chroma_skills"`
- Set standard chunk size to `500` and overlap to `100` (~20% overlap) as default for document ingestion.

---

### Vector Store Layer

#### [MODIFY] [services/vector_store.py](file:///home/pi-net/Documents/agent_eng_labs/Agent-with-RAG/services/vector_store.py)
- Separate collection and persistence logic into:
  - `SkillVectorStore`: Points to `CHROMA_SKILLS_DIR`. Stores `SKILL.md` entries where only `name` and `description` are embedded, and the complete text of `SKILL.md` is stored as a single chunk. Exposes `query(query_text, min_score=0.50)`.
  - `DocumentVectorStore`: Points to `CHROMA_DOCS_DIR`. Stores chunked user documents (~20% overlap). Exposes `query(query_text, top_k=4, min_score=0.30)`.
- Export singleton instances `skill_vector_store` and `doc_vector_store` for clean dependency injection.

---

### Document Ingestion Layer

#### [MODIFY] [services/document_loader.py](file:///home/pi-net/Documents/agent_eng_labs/Agent-with-RAG/services/document_loader.py)
- Enforce 20% overlap in text splitting (e.g. `chunk_size=500`, `chunk_overlap=100` = 20%).
- Ensure metadata preserves source path, title, and chunk indices for document chunks.

---

### New Skill: `get-private-doc`

#### [NEW] [skills/get-private-doc/SKILL.md](file:///home/pi-net/Documents/agent_eng_labs/Agent-with-RAG/skills/get-private-doc/SKILL.md)
- Follows `SKILL_SPEC.md` format with frontmatter:
  ```yaml
  ---
  name: get-private-doc
  description: From the RAG document database, retrieve sections of documents that are: private, corporate, internal, or secret. Only sections that are related to the prompt will be included.
  ---
  ```
- Trigger Queries & Capabilities:
  - "What is the summary of internal document"
  - "What is the result of the previous quarter?"
  - "What is the most recent marketing plan for our company?"
- Action: Query the document vector database using the user prompt.

#### [NEW] [skills/get-private-doc/scripts/doc_tools.py](file:///home/pi-net/Documents/agent_eng_labs/Agent-with-RAG/skills/get-private-doc/scripts/doc_tools.py)
- CLI and module tool that queries `DocumentVectorStore` using the provided `--query` string.
- Formats retrieved chunks with document title, path, relevance score, and content snippet. Supports `--json` output.

---

### Agent Execution & Orchestration Layer

#### [MODIFY] [services/skill_runner.py](file:///home/pi-net/Documents/agent_eng_labs/Agent-with-RAG/services/skill_runner.py)
- Update `auto_index_skills_into_db()` to scan `skills/` upon startup and index into `SkillVectorStore`.
- Implement `run_agent_pipeline(question, model, custom_endpoint, conversation_id)`:
  1. Embed prompt and query `SkillVectorStore` with `min_score=0.50`.
  2. If no candidate skill $> 50\%$:
     - Route prompt to LLM with the required general system prompt:
       `"You are a helpful, respectful, and honest assistant. Always answer as truthfully as possible, use clear markdown formatting, and admit when you do not know an answer rather than guessing."`
  3. If candidate skill(s) $> 50\%$:
     - **Phase 1**: Send prompt + `SKILL.md` to LLM for **Plan / Execution Instruction**.
     - **Phase 2**: Agent parses and executes the tool script / action (e.g. `doc_tools.py`, `stock_tools.py`, or `env_tools.py`).
     - **Phase 3**: Agent sends tool output + prompt + `SKILL.md` to LLM for final grounded user answer.

#### [MODIFY] [app.py](file:///home/pi-net/Documents/agent_eng_labs/Agent-with-RAG/app.py)
- On startup, invoke `auto_index_skills_into_db()` targeting `SkillVectorStore`.
- Update `/api/ingest` to route ingested files to `DocumentVectorStore` with 20% overlap.
- Update `/api/query` to execute the new Agent pipeline while maintaining complete audit logging in `database/logs.json` and immediate record creation in `database/conversations.json`.

---

## Verification Plan

### Automated Tests
- Run `.venv/bin/python -m unittest tests/test_private_rag.py` testing:
  1. Separation and persistence of `chroma_skills` and `chroma_docs`.
  2. Startup scanning and indexing of all skills (`get-private-doc`, `stock-market-skill`, `time-weather-skill`).
  3. Document chunking with 20% overlap.
  4. Querying skill database with strict 50% threshold.
  5. Fallback to general system prompt when no skill matches $> 50\%$.
  6. Execution of `get-private-doc` skill when asking internal/corporate document questions.
  7. End-to-end plan $\to$ execute $\to$ synthesize loop.

### Manual Verification
- Ingest `sample_docs/` via the web UI and verify chunks persist in `database/chroma_docs`.
- Ask corporate query: `"What is the summary of internal document"` $\to$ verify `get-private-doc` skill triggers and retrieves internal document context.
- Ask general query: `"What is photosynthesis?"` $\to$ verify no skill matches $> 50\%$ and general system prompt is used.
- Ask stock query: `"Which stocks have the highest percentage increase?"` $\to$ verify `stock-market-skill` triggers.
