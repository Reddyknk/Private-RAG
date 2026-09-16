import os
import time
from typing import List, Dict, Any, Optional
import requests
from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMMA_PRIMARY_MODEL, GEMMA_FALLBACK_MODEL, MAX_TOKEN_LOCAL, MAX_TOKEN_EXTERNAL
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

    def get_available_text_models(self) -> List[Dict[str, Any]]:
        """
        Query Google AI Studio for all text generation models.
        Strictly filters out non-text models (image, speech/TTS, audio, transcription, music, robotics, etc.)
        and appends the 'Custom Model API' option for private LLM endpoints.
        """
        preferred_default = self.get_best_gemma_model()

        custom_option = {
            "id": "custom_model_api",
            "name": "Custom Model API",
            "description": "User-defined private LLM API endpoint (e.g., local model server, vLLM, Ollama, OpenAI-compatible)",
            "is_custom": True,
            "is_default": False
        }

        fallback_models = [
            {"id": "models/gemini-3.6-flash", "name": "Gemini 3.6 Flash", "description": "Fast, high-performance model with advanced reasoning", "is_default": True},
            {"id": "models/gemini-flash-latest", "name": "Gemini Flash Latest", "description": "Latest stable Gemini Flash text model", "is_default": False},
            {"id": "models/gemma-4-26b-a4b-it", "name": "Gemma 4 26B A4B IT", "description": "High-capability open Gemma model from Google", "is_default": False},
            {"id": "models/gemini-3.1-flash-lite-preview", "name": "Gemini 3.1 Flash Lite", "description": "Ultra-lightweight, rapid response Gemini", "is_default": False},
            custom_option
        ]

        non_text_keywords = [
            "image", "imagen", "tts", "audio", "native-audio", "transcribe",
            "lyria", "music", "banana", "robotics", "computer-use", "deep-research",
            "antigravity", "embed"
        ]

        try:
            models_list = list(self.client.models.list())
            text_models = []
            for m in models_list:
                actions = getattr(m, 'supported_actions', []) or getattr(m, 'supported_generation_methods', []) or []
                name = m.name
                if 'generateContent' in actions:
                    name_lower = name.lower()
                    display_name = getattr(m, 'display_name', '') or name.replace('models/', '')
                    desc = getattr(m, 'description', '') or ''
                    combined_check = f"{name_lower} {display_name.lower()} {desc.lower()}"

                    if any(kw in combined_check for kw in non_text_keywords):
                        continue

                    text_models.append({
                        "id": name,
                        "name": display_name,
                        "description": desc,
                        "is_default": (name == preferred_default or preferred_default in name)
                    })

            if text_models:
                def sort_key(item):
                    mid = item["id"].lower()
                    if item.get("is_default"):
                        return (0, mid)
                    if "gemma-4-26b" in mid:
                        return (1, mid)
                    if "gemini-2.5-flash" in mid and "preview" not in mid:
                        return (2, mid)
                    if "gemini-2.5-pro" in mid and "preview" not in mid:
                        return (3, mid)
                    if "gemma" in mid:
                        return (4, mid)
                    if "gemini" in mid:
                        return (5, mid)
                    return (6, mid)

                text_models.sort(key=sort_key)
                has_default = any(m["is_default"] for m in text_models)
                if not has_default and text_models:
                    text_models[0]["is_default"] = True

                text_models.append(custom_option)
                return text_models

        except Exception as e:
            print(f"[GemmaService] Could not list models from Google AI Studio: {e}")

        return fallback_models

    def query_custom_llm(
        self,
        endpoint: str,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query a private or custom LLM via HTTP POST endpoint (e.g., local model_serv, vLLM, Ollama, OpenAI API).
        """
        start_time = time.time()
        endpoint = endpoint.strip() if endpoint else "http://127.0.0.1:8000/v1/chat/completions"
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = "http://" + endpoint

        # Smart endpoint normalization for local vLLM server
        ep_lower = endpoint.lower().rstrip("/")
        if ep_lower.endswith(":8000"):
            endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        elif ep_lower.endswith(":8000/v1"):
            endpoint = endpoint.rstrip("/") + "/chat/completions"

        # Build context from chunks
        if retrieved_chunks:
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
                "If the user greets you or includes conversational pleasantries, greet them warmly. "
                "Answer the user's question thoroughly using the provided context retrieved from their private vector database. "
                "If the question cannot be answered from the provided context, state that clearly without guessing. "
                "Cite your sources using the document numbers or document names provided in the context."
            )
            user_content = (
                f"Context from private vector database:\n\n{context_str}\n\n"
                f"User Question: {question}\n\n"
                f"Please provide a thorough, well-structured answer based strictly on the context above."
            )
        else:
            default_system_prompt = (
                "You are a helpful, knowledgeable, and accurate AI assistant for Agent with RAG. "
                "Answer the user's question clearly, thoroughly, and accurately to the best of your knowledge."
            )
            user_content = question

        sys_prompt = system_instruction or default_system_prompt
        full_prompt = f"{sys_prompt}\n\n{user_content}"

        # If endpoint is a chat endpoint, send only messages. For raw completion endpoints, send prompt.
        # This prevents duplicating the large vector database context twice in the payload.
        if "chat" in endpoint.lower() or "v1/chat" in endpoint.lower():
            request_payload = {
                "model": "Llama-3.2-3B-Instruct",
                "system": sys_prompt,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                "target": "model",
                "port": 8000,
                "temperature": 0.2,
                "max_tokens": MAX_TOKEN_LOCAL
            }
        else:
            request_payload = {
                "model": "Llama-3.2-3B-Instruct",
                "prompt": full_prompt,
                "system": sys_prompt,
                "target": "model",
                "port": 8000,
                "temperature": 0.2,
                "max_tokens": MAX_TOKEN_LOCAL
            }

        call_args = {
            "endpoint": endpoint,
            "system_instruction": sys_prompt,
            "question": question,
            "retrieved_chunks_count": len(retrieved_chunks),
            "sources": [c.get("metadata", {}).get("source") for c in retrieved_chunks],
            "prompt_length_chars": len(full_prompt)
        }

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer your-internal-secure-gateway-token-xyz"
            }
            resp = requests.post(
                endpoint,
                json=request_payload,
                headers=headers,
                timeout=120
            )
            duration_ms = (time.time() - start_time) * 1000

            if resp.status_code == 200:
                try:
                    resp_json = resp.json()
                except Exception:
                    resp_json = {}

                answer_text = ""
                if isinstance(resp_json, dict):
                    answer_text = (
                        resp_json.get("reply") or
                        resp_json.get("response") or
                        resp_json.get("answer") or
                        resp_json.get("text") or
                        ""
                    )
                    if not answer_text and "choices" in resp_json and len(resp_json["choices"]) > 0:
                        choice = resp_json["choices"][0]
                        if isinstance(choice, dict):
                            msg = choice.get("message", {})
                            if isinstance(msg, dict):
                                answer_text = msg.get("content", "")
                            elif isinstance(choice.get("text"), str):
                                answer_text = choice.get("text", "")

                if not answer_text:
                    answer_text = resp.text

                log_entry = log_call(
                    call_type="custom_model_api",
                    arguments=call_args,
                    response={
                        "status": "success",
                        "endpoint": endpoint,
                        "status_code": resp.status_code,
                        "answer_chars": len(answer_text),
                        "answer_preview": answer_text[:200] + "..." if len(answer_text) > 200 else answer_text
                    },
                    duration_ms=duration_ms,
                    status="success",
                    conversation_id=conversation_id,
                    invoker="Agent",
                    target=f"Custom Private LLM ({endpoint})",
                    short_description=f"Prompted custom LLM at {endpoint}"
                )

                llm_component = {
                    "name": "LLM",
                    "role": "Private Model Server",
                    "icon": "⚡",
                    "status": "success",
                    "duration_ms": round(duration_ms, 2),
                    "description": f"Generated grounded response using custom private LLM at {endpoint}",
                    "request": {
                        "endpoint": endpoint,
                        "method": "POST",
                        "system_instruction": sys_prompt,
                        "question": question,
                        "prompt_length_chars": len(full_prompt),
                        "context_chunks_count": len(retrieved_chunks)
                    },
                    "response": {
                        "endpoint": endpoint,
                        "status_code": resp.status_code,
                        "answer": answer_text,
                        "duration_ms": round(duration_ms, 2)
                    }
                }

                return {
                    "answer": answer_text,
                    "model": f"Custom API ({endpoint})",
                    "sources": retrieved_chunks,
                    "log_id": log_entry.get("id"),
                    "duration_ms": round(duration_ms, 2),
                    "component": llm_component
                }
            else:
                err_msg = f"Custom model API returned HTTP {resp.status_code}: {resp.text}"
                log_call(
                    call_type="custom_model_api",
                    arguments=call_args,
                    response={"error": err_msg, "status_code": resp.status_code},
                    duration_ms=duration_ms,
                    status="error",
                    conversation_id=conversation_id,
                    invoker="Agent",
                    target=f"Custom Private LLM ({endpoint})",
                    short_description=f"Error from custom LLM {endpoint}: HTTP {resp.status_code}"
                )
                raise RuntimeError(err_msg)

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_call(
                call_type="custom_model_api",
                arguments=call_args,
                response={"error": str(e)},
                duration_ms=duration_ms,
                status="error",
                conversation_id=conversation_id,
                invoker="Agent",
                target=f"Custom Private LLM ({endpoint})",
                short_description=f"Failed to query custom LLM at {endpoint}: {str(e)}"
            )
            raise

    def answer_question(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        conversation_id: Optional[str] = None,
        custom_endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer using Google AI Studio Gemma/Gemini model, or custom model endpoint,
        grounded strictly on retrieved document chunks from the private vector database.
        """
        if model == "custom_model_api" or (model and "custom" in model.lower()) or custom_endpoint:
            endpoint = custom_endpoint or "http://127.0.0.1:8000/api/chat"
            return self.query_custom_llm(
                endpoint=endpoint,
                question=question,
                retrieved_chunks=retrieved_chunks,
                system_instruction=system_instruction,
                conversation_id=conversation_id
            )

        start_time = time.time()
        model_name = model or self.get_best_gemma_model()

        # Build context from chunks
        if retrieved_chunks:
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
                "If the user greets you or includes conversational pleasantries, greet them warmly. "
                "Answer the user's question thoroughly using the provided context retrieved from their private vector database. "
                "If the question cannot be answered from the provided context, state that clearly without guessing. "
                "Cite your sources using the document numbers or document names provided in the context."
            )
            user_content = (
                f"Context from private vector database:\n\n{context_str}\n\n"
                f"User Question: {question}\n\n"
                f"Please provide a thorough, well-structured answer based strictly on the context above."
            )
        else:
            default_system_prompt = (
                "You are a helpful, knowledgeable, and accurate AI assistant for Agent with RAG. "
                "Answer the user's question clearly, thoroughly, and accurately to the best of your knowledge."
            )
            user_content = question

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
                max_output_tokens=MAX_TOKEN_EXTERNAL,
            )

            # Build prioritized fallback list to handle 503 UNAVAILABLE / capacity limits
            candidate_models = [model_name]
            standard_fallbacks = [
                "models/gemini-3.6-flash",
                "models/gemini-flash-latest",
                "models/gemma-4-26b-a4b-it",
                "models/gemini-3-flash-preview",
                "models/gemini-3.1-flash-lite-preview",
                GEMMA_PRIMARY_MODEL,
                GEMMA_FALLBACK_MODEL
            ]
            for fb in standard_fallbacks:
                if fb and fb not in candidate_models:
                    candidate_models.append(fb)

            response = None
            last_err = None
            for candidate in candidate_models:
                try:
                    response = self.client.models.generate_content(
                        model=candidate,
                        contents=user_content,
                        config=config
                    )
                    model_name = candidate
                    call_args["model"] = model_name
                    break
                except Exception as candidate_err:
                    last_err = candidate_err
                    err_str = str(candidate_err)
                    # If 503 capacity limit or temporary server overload, seamlessly switch to next available model
                    if any(kw in err_str.lower() for kw in ["503", "unavailable", "capacity", "overloaded", "resource_exhausted"]):
                        print(f"[GemmaService] Model {candidate} unavailable ({err_str[:120]}), falling back to next model...")
                        time.sleep(0.5)
                        continue
                    else:
                        # For other non-capacity errors, also try next candidate if available
                        print(f"[GemmaService] Model {candidate} error ({err_str[:120]}), trying next fallback...")
                        continue

            if response is None:
                raise last_err or RuntimeError("All candidate Google AI Studio models are currently at capacity.")

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
                status="success",
                conversation_id=conversation_id,
                invoker="Agent",
                target=f"Google AI Studio ({model_name})",
                short_description=f"Prompted {model_name} for grounded QA synthesis"
            )

            llm_component = {
                "name": "LLM",
                "role": "Grounded Generator",
                "icon": "✨",
                "status": "success",
                "duration_ms": round(duration_ms, 2),
                "description": f"Generated grounded response using {model_name}",
                "request": {
                    "model": model_name,
                    "system_instruction": system_instruction or default_system_prompt,
                    "question": question,
                    "prompt_length_chars": len(user_content),
                    "context_chunks_count": len(retrieved_chunks),
                    "temperature": 0.2,
                    "max_output_tokens": MAX_TOKEN_EXTERNAL
                },
                "response": {
                    "model_used": model_name,
                    "answer": answer_text,
                    "usage": usage_metadata,
                    "duration_ms": round(duration_ms, 2)
                }
            }

            return {
                "answer": answer_text,
                "model": model_name,
                "sources": retrieved_chunks,
                "log_id": log_entry.get("id"),
                "duration_ms": round(duration_ms, 2),
                "component": llm_component
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_call(
                call_type="google_ai_studio_gemma",
                arguments=call_args,
                response={"error": str(e)},
                duration_ms=duration_ms,
                status="error",
                conversation_id=conversation_id,
                invoker="Agent",
                target=f"Google AI Studio ({model_name})",
                short_description=f"Error querying {model_name}"
            )
            raise


# Global singleton instance
gemma_service = GemmaService()
