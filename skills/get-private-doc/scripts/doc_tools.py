#!/usr/bin/env python3
"""
doc_tools.py - Retrieval tool for the get-private-doc skill.
Queries the private document vector store and returns matching document sections.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from services.vector_store import doc_vector_store
except ImportError:
    from services.vector_store import DocumentVectorStore
    doc_vector_store = DocumentVectorStore()


def retrieve_private_documents(query: str, top_k: int = 4, min_score: float = 0.30) -> List[Dict[str, Any]]:
    """Retrieve semantically relevant sections from the private document database."""
    if not query or not query.strip():
        return []
    try:
        return doc_vector_store.query(query_text=query.strip(), top_k=top_k, min_score=min_score)
    except Exception as e:
        print(f"[doc_tools] Warning: Document retrieval failed: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Query private corporate document vector database.")
    parser.add_argument("--query", "-q", type=str, required=True, help="User query or keywords to search for.")
    parser.add_argument("--top-k", "-k", type=int, default=4, help="Maximum number of document sections to retrieve.")
    parser.add_argument("--min-score", "-s", type=float, default=0.30, help="Minimum similarity threshold (default: 0.30).")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")

    args = parser.parse_args()
    results = retrieve_private_documents(args.query, top_k=args.top_k, min_score=args.min_score)

    if args.json:
        output = {
            "query": args.query,
            "count": len(results),
            "documents": [
                {
                    "source": r.get("metadata", {}).get("source", "Unknown"),
                    "title": r.get("metadata", {}).get("title", ""),
                    "score": r.get("score", 0.0),
                    "chunk_index": r.get("metadata", {}).get("chunk_index", 0),
                    "content": r.get("content", "")
                }
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        if not results:
            print(f"No relevant document sections found in the private database for: '{args.query}' (threshold >= {args.min_score})")
            return

        print(f"=== RETRIEVED PRIVATE DOCUMENT SECTIONS ({len(results)} matches) ===")
        for idx, doc in enumerate(results, 1):
            source = doc.get("metadata", {}).get("source", "Unknown")
            title = doc.get("metadata", {}).get("title", source)
            score = doc.get("score", 0.0)
            content = doc.get("content", "").strip()
            print(f"\n--- [Document {idx}] Source: {title} (Relevance: {score}) ---")
            print(f"Path: {source}")
            print(f"Content:\n{content}\n")
        print("=== END PRIVATE DOCUMENT RETRIEVAL ===")


if __name__ == "__main__":
    main()
