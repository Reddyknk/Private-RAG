---
name: get-private-doc
description: From the RAG document database, retrieve sections of documents that are: private, corporate, internal, or secret. Explains what is the result of the previous quarter?, what is the summary of internal document, and what is the most recent marketing plan for our company. Only sections that are related to the prompt will be included.
---

# Get Private Document Skill

This skill enables the autonomous agent to query the private RAG document vector database and retrieve confidential, proprietary, corporate, or internal document sections semantically relevant to the user's prompt.

## 🎯 Trigger Queries & Capabilities
Activate this skill when the user asks:
- "What is the summary of internal document"
- "What is the result of the previous quarter?"
- "What is the most recent marketing plan for our company?"
- Questions asking for corporate secrets, internal memos, financial reports, or private database documents.

---

## 📂 Architecture & Data Assets

Following the `SKILL_SPEC.md` directory layout:
```
skills/get-private-doc/
├── SKILL.md                 # Metadata & Standard Operating Procedure (SOP)
└── scripts/
    └── doc_tools.py         # Document vector database retriever tool
```

### Document Vector Database Tool (`scripts/doc_tools.py`)
Queries the private ChromaDB document database (`database/chroma_docs`) using local embeddings to retrieve the top-matching document sections with high semantic relevance.

---

## 📋 Standard Operating Procedure (SOP)

When answering questions about internal, corporate, or private documents:
1. **Identify the Core Query**:
   Extract the user question or intent to be searched against the document database.
2. **Execute the Document Tool**:
   Run the tool from the repository root:
   ```bash
   python skills/get-private-doc/scripts/doc_tools.py --query "<user prompt>" [--top-k 4] [--json]
   ```
3. **Synthesize the Response**:
   - Provide a clear, factual answer synthesized strictly from the retrieved document sections.
   - Cite source documents and paths provided in the retrieved evidence.
   - If no relevant document sections are found, clearly state that no matching internal documents exist in the database.

---

## 💻 CLI Commands & Examples

```bash
# Query internal documents
python skills/get-private-doc/scripts/doc_tools.py --query "What is the summary of internal document"

# Query with custom top-k limit
python skills/get-private-doc/scripts/doc_tools.py --query "What is the result of the previous quarter?" --top-k 3

# Machine-readable JSON output
python skills/get-private-doc/scripts/doc_tools.py --query "What is the most recent marketing plan for our company?" --json
```
