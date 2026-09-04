import os
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMMA_PRIMARY_MODEL, GEMMA_FALLBACK_MODEL
from services.logger_service import log_call


class GemmaService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self._client = None
        self._discovered_gemma_model = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def get_best_gemma_model(self) -> str:
        """Discover available Gemma models on Google AI Studio or fall back to primary config."""
        if self._discovered_gemma_model:
            return self._discovered_gemma_model

        try:
            available_models = [
                m.name for m in self.client.models.list()
                if "gemma" in m.name.lower()
            ]
            if available_models:
                # Prefer GEMMA_PRIMARY_MODEL if in list
                for m in available_models:
                    if GEMMA_PRIMARY_MODEL in m or m in GEMMA_PRIMARY_MODEL:
                        self._discovered_gemma_model = m
                        return m
                self._discovered_gemma_model = available_models[0]
                return self._discovered_gemma_model
        except Exception as e:
            print(f"[GemmaService] Could not list models from Google AI Studio: {e}")

        return GEMMA_PRIMARY_MODEL

    def answer_question(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer using Google AI Studio Gemma model, grounded strictly
        on retrieved document chunks from the private vector database.
        """
        start_time = time.time()
        model_name = self.get_best_gemma_model()

        # Build context from chunks
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown Source")
            title = chunk.get("metadata", {}).get("title", source)
            score = chunk.get("score", 0.0)
            text = chunk.get("content", "")
            context_blocks.append(
                f"--- [Document {i}] Source: {title} (Path: {source}, Relevance: {score}) ---\n{text}"
            )

        context_str = "\n\n".join(context_blocks)

        default_system_prompt = (
            "You are a helpful, accurate, and privacy-preserving AI assistant. "
            "Your task is to answer the user's question using ONLY the provided context retrieved from their private vector database. "
            "If the answer cannot be determined from the context, state that clearly without guessing. "
            "Cite your sources using the document numbers or document names provided in the context."
        )

        user_content = (
            f"Context from private vector database:\n\n{context_str}\n\n"
            f"User Question: {question}\n\n"
            f"Please provide a thorough, well-structured answer based strictly on the context above."
        )

        call_args = {
            "model": model_name,
            "system_instruction": system_instruction or default_system_prompt,
            "question": question,
            "retrieved_chunks_count": len(retrieved_chunks),
            "sources": [c.get("metadata", {}).get("source") for c in retrieved_chunks],
            "prompt_length_chars": len(user_content)
        }

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction or default_system_prompt,
                temperature=0.2,
                max_output_tokens=2048,
            )

            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=user_content,
                    config=config
                )
            except Exception as primary_err:
                # If primary model fails, attempt fallback model
                if GEMMA_FALLBACK_MODEL and GEMMA_FALLBACK_MODEL != model_name:
                    print(f"[GemmaService] Model {model_name} failed ({primary_err}), trying fallback {GEMMA_FALLBACK_MODEL}")
                    model_name = GEMMA_FALLBACK_MODEL
                    call_args["model"] = model_name
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=user_content,
                        config=config
                    )
                else:
                    raise primary_err

            duration_ms = (time.time() - start_time) * 1000
            answer_text = response.text or ""

            # Extract usage metadata if present
            usage_metadata = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_metadata = {
                    "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", None),
                    "candidates_token_count": getattr(response.usage_metadata, "candidates_token_count", None),
                    "total_token_count": getattr(response.usage_metadata, "total_token_count", None),
                }

            log_entry = log_call(
                call_type="google_ai_studio_gemma",
                arguments=call_args,
                response={
                    "status": "success",
                    "model_used": model_name,
                    "answer_chars": len(answer_text),
                    "answer_preview": answer_text[:200] + "..." if len(answer_text) > 200 else answer_text,
                    "usage": usage_metadata
                },
                duration_ms=duration_ms,
                status="success"
            )

            return {
                "answer": answer_text,
                "model": model_name,
                "sources": retrieved_chunks,
                "log_id": log_entry.get("id"),
                "duration_ms": round(duration_ms, 2)
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_call(
                call_type="google_ai_studio_gemma",
                arguments=call_args,
                response={"error": str(e)},
                duration_ms=duration_ms,
                status="error"
            )
            raise


# Global singleton instance
gemma_service = GemmaService()
