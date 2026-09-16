# Walkthrough: Dual Vector Store Refactoring & Agent Pipeline

We have completed the refactoring of **Agent with RAG** into a two-database architecture with dedicated skill indexing, 20% document chunk overlap, 50% skill candidate gating, and the new `get-private-doc` skill.

---

## 1. Architectural Architecture & Key Changes

### A. Dual Vector Databases ([services/vector_store.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/services/vector_store.py))
The application now separates skill metadata from ingested knowledge documents:

1. **Skill Database (`database/chroma_skills`)**:
   - Collection: `skills_collection` (Cosine similarity space).
   - On startup, `auto_index_skills_into_db()` scans the `skills/` folder.
   - **Only the `name` and `description`** of each `SKILL.md` are passed to the local Ollama embedder.
   - When the embedding is returned, the vector and the **complete text of the `SKILL.md`** are stored in the database as a single chunk.
2. **Document Database (`database/chroma_docs`)**:
   - Collection: `documents_collection` (Cosine similarity space).
   - Ingests URLs and local directories via the Web UI.
   - Chunked with **~20% overlap** (`chunk_size=500`, `chunk_overlap=100`).

```
                              User Prompt
                                   │
                                   ▼
                         [Local Ollama Embedder]
                                   │
                                   ▼
                         [Skill Vector Store]
                         (database/chroma_skills)
                                   │
                       Similarity Score > 50%?
                      ┌────────────┴────────────┐
                     YES                        NO
                      │                         │
                      ▼                         ▼
            [Plan -> Execute -> Synthesize]   [Direct LLM Call]
            - LLM receives prompt + SKILL.md  - Exact required system prompt:
            - Agent executes tool script        "You are a helpful, respectful,
            - Tool queries Document DB          and honest assistant. Always answer
              (database/chroma_docs)            as truthfully as possible..."
            - LLM synthesizes final answer
```

---

### B. New Skill: `get-private-doc` ([skills/get-private-doc/](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/skills/get-private-doc/))
Following the specification in `SKILL_SPEC.md`:
- **`SKILL.md`**:
  - `name`: `get-private-doc`
  - `description`: `From the RAG document database, retrieve sections of documents that are: private, corporate, internal, or secret. Explains what is the result of the previous quarter?, what is the summary of internal document, and what is the most recent marketing plan for our company. Only sections that are related to the prompt will be included.`
  - Trigger Queries:
    - `"What is the summary of internal document"`
    - `"What is the result of the previous quarter?"`
    - `"What is the most recent marketing plan for our company?"`
- **`scripts/doc_tools.py`**:
  - Executable CLI and module supporting `--query`, `--top-k`, `--min-score`, and `--json`.
  - Queries `doc_vector_store` (`database/chroma_docs`) and outputs matching private document sections.

---

### C. Agent Execution & Fallback Reasoning Loop ([services/skill_runner.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/services/skill_runner.py))
1. When a user prompt arrives:
   - Ollama generates an embedding for the prompt.
   - The query searches `skill_vector_store` with threshold `min_score = 0.50` (50%).
2. **If no skill matches $> 50\%$**:
   - The prompt is forwarded to the LLM model with the exact required system prompt:
     > *"You are a helpful, respectful, and honest assistant. Always answer as truthfully as possible, use clear markdown formatting, and admit when you do not know an answer rather than guessing."*
   - No vector context chunks are passed to the model.
3. **If skills match $> 50\%$**:
   - **Step 1 (Plan)**: Prompt + matched `SKILL.md` texts sent to LLM to determine the plan and execution instruction.
   - **Step 2 (Execute)**: Agent executes the tool script (`doc_tools.py`, `stock_tools.py`, or `env_tools.py`).
   - **Step 3 (Synthesize)**: Tool execution results + original question + `SKILL.md` are sent to the LLM to generate the final synthesized response with citations.

---

### D. Branding, UI & Telemetry
- Browser tab icon updated to [static/images/AI_icon_s.png](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/static/images/AI_icon_s.png).
- Header brand icon updated to [static/images/AI_icon.png](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/static/images/AI_icon.png).
- Fixed shutdown confirmation modal warning phrase.
- Fixed context duplication by avoiding redundant `prompt` strings when sending chat `messages` over the wire in [services/gemma_service.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/services/gemma_service.py).
- Added component logging for `Embedder`, `Skill Store`, `Vector Store`, `Tool`, `External API`, and `LLM`.

---

## 2. Verification & Test Results

### A. Full Automated Test Suite
Ran all 16 unit and integration tests in [tests/test_private_rag.py](file:///home/pi-net/Documents/agent_eng_labs/privateRAG/tests/test_private_rag.py):

```bash
.venv/bin/python -m unittest tests/test_private_rag.py
```
**Result**:
```
----------------------------------------------------------------------
Ran 16 tests in 78.666s

OK
```

All 16 tests passed:
1. `test_01_health_check` — Server status, Ollama, and Gemma readiness.
2. `test_02_url_and_directory_ingestion` — Ingestion into `database/chroma_docs`.
3. `test_03_query_and_grounded_generation` — Grounded answers with citations.
4. `test_04_audit_logs_structure_and_types` — Logging schema in `database/logs.json`.
5. `test_05_skills_discovery_and_metadata` — Auto-discovery across `skills/`.
6. `test_06_model_switching_and_log_pruning` — Model changes and 30MB log ceiling.
7. `test_07_all_components_logged_and_displayed` — Agent, Embedder, Skill/Vector Store, Tool, External API, LLM telemetry.
8. `test_08_text_only_models_and_custom_api_endpoint` — Dropdown filtering and vLLM custom API endpoint.
9. `test_09_conversational_greeting_and_relevance_filtering` — "Hello" query filtering and warm greeting response.
10. `test_10_shutdown_and_branding_elements` — Shutdown modal and branding icons.
11. `test_11_low_similarity_sends_prompt_to_model_without_vector_context` — Sub-50% fallback with exact required system prompt.
12. `test_12_cutoff_and_conversations_prompt_entry` — 30% document cutoff and immediate `conversations.json` logging.
13. `test_13_dual_vector_stores_separation` — Document DB and Skill DB directory and collection isolation.
14. `test_14_skill_startup_indexing_name_desc_embedding_whole_chunk` — Only name and description embedded, full `SKILL.md` content stored.
15. `test_15_twenty_percent_chunk_overlap` — Chunking with 20% overlap.
16. `test_16_get_private_doc_skill_and_threshold_retrieval` — Trigger retrieval and `doc_tools.py` execution.

---

### B. Live Server Verification
Verified on running server at `http://127.0.0.1:8005`:
1. **Health Check (`GET /api/health`)**: Returns `online` with Ollama embeddings and Gemma LLM ready.
2. **Skill Database Ingestion**: Successfully indexed 3 skills (`get-private-doc`, `stock-market-skill`, `time-weather-skill`).
3. **Live Query (`POST /api/query`)**:
   - Query: `"What is the summary of internal document"`
   - Result: Matched `get-private-doc` skill (relevance > 50%), executed `doc_tools.py` against `database/chroma_docs`, and synthesized grounded answer citing retrieved document evidence.
