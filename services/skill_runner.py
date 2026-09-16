import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from services.document_loader import Document, extract_skill_metadata
from services.vector_store import skill_vector_store, doc_vector_store
from services.ollama_embedder import OllamaEmbeddingFunction
from services.logger_service import log_event
from services.gemma_service import gemma_service

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
REPO_ROOT = Path(__file__).resolve().parent.parent

NO_SKILL_SYSTEM_PROMPT = (
    "You are a helpful, respectful, and honest assistant. "
    "Always answer as truthfully as possible, use clear markdown formatting, "
    "and admit when you do not know an answer rather than guessing."
)


def get_available_skills() -> List[Dict[str, Any]]:
    """List all skills discovered in the skills/ directory."""
    if not SKILLS_DIR.exists():
        return []

    skills = []
    for skill_folder in sorted(SKILLS_DIR.iterdir()):
        if not skill_folder.is_dir():
            continue
        skill_md = skill_folder / "SKILL.md"
        if not skill_md.exists():
            continue

        title = skill_folder.name
        desc = ""
        try:
            content = skill_md.read_text(encoding="utf-8")
            title, desc = extract_skill_metadata(content, default_name=skill_folder.name)
        except Exception:
            pass

        scripts = [s.name for s in (skill_folder / "scripts").glob("*.py")] if (skill_folder / "scripts").exists() else []
        data_files = [d.name for d in (skill_folder / "data").glob("*.*")] if (skill_folder / "data").exists() else []

        skills.append({
            "id": skill_folder.name,
            "name": title,
            "description": desc,
            "path": str(skill_folder),
            "scripts": scripts,
            "data_files": data_files
        })

    return skills


def auto_index_skills_into_db() -> Dict[str, Any]:
    """
    Scans the skills/ folder to get all skills.
    Only the name and description of each SKILL.md are sent to the embedder.
    When the embedding is received, the vectors and the complete text of the SKILL.md
    are stored in the skill database (database/chroma_skills).
    """
    if not SKILLS_DIR.exists():
        return {"status": "skipped", "message": "skills/ directory does not exist."}

    try:
        added_count = 0
        for skill_folder in sorted(SKILLS_DIR.iterdir()):
            if not skill_folder.is_dir():
                continue
            skill_md = skill_folder / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                content = skill_md.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = skill_md.read_text(encoding="latin-1", errors="ignore")

            if not content.strip():
                continue

            name, desc = extract_skill_metadata(content, default_name=skill_folder.name)

            # Store in skill database: only name and description are embedded, complete text is saved
            skill_vector_store.add_skill(
                skill_id=skill_folder.name,
                name=name,
                description=desc,
                full_content=content,
                metadata={
                    "source": str(skill_md),
                    "relative_path": str(skill_md.relative_to(SKILLS_DIR)),
                    "skill_name": name,
                    "skill_description": desc
                }
            )
            added_count += 1

        return {
            "status": "indexed",
            "message": f"Successfully indexed {added_count} skills into skill database using name & description vectors.",
            "total_skills": skill_vector_store.collection.count()
        }
    except Exception as e:
        return {"status": "error", "message": f"Skill indexing failed: {e}"}


def execute_instruction(
    instruction: str,
    question: str,
    active_skills: List[Dict[str, Any]],
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Agent executes the plan or instruction received from the LLM model.
    Runs the appropriate tool script (e.g. doc_tools.py, stock_tools.py, env_tools.py)
    and returns tool output chunks, logs, and structured UI components.
    """
    start_time = time.time()
    q_lower = question.lower()
    inst_lower = instruction.lower()
    tool_chunks: List[Dict[str, Any]] = []
    components: List[Dict[str, Any]] = []
    raw_output = ""

    # Check which skill is active
    active_skill_names = [s.get("name", "").lower() for s in active_skills] + [s.get("skill_id", "").lower() for s in active_skills]

    # 1. get-private-doc skill
    if any("get-private-doc" in s or "private-doc" in s for s in active_skill_names) or "doc_tools.py" in inst_lower:
        script_path = SKILLS_DIR / "get-private-doc" / "scripts" / "doc_tools.py"
        search_query = question
        # Extract query argument if formatted in instruction
        q_match = re.search(r'--query\s+["\']([^"\']+)["\']', instruction)
        if q_match:
            search_query = q_match.group(1)

        cmd = [sys.executable, str(script_path), "--query", search_query, "--top-k", "4", "--json"]
        tool_req = {"tool": "doc_tools.py", "query": search_query, "command": cmd}

        try:
            res = subprocess.check_output(cmd, cwd=str(REPO_ROOT), timeout=15).decode("utf-8")
            raw_output = res
            parsed_res = json.loads(res) if res.strip().startswith("{") else {"raw_output": res}
            docs = parsed_res.get("documents", [])

            duration_ms = (time.time() - start_time) * 1000
            log_event(
                event_type="Tool",
                invoker="Agent",
                target="doc_tools.py (Private Document Database)",
                short_description=f"Retrieved {len(docs)} private document section(s)",
                payload={"request": tool_req, "response": parsed_res},
                conversation_id=conversation_id,
                duration_ms=duration_ms,
                status="success"
            )

            components.append({
                "name": "Tool",
                "role": "Skill Script",
                "icon": "📄",
                "status": "success",
                "duration_ms": round(duration_ms, 2),
                "description": f"Executed doc_tools.py: retrieved {len(docs)} document chunk(s)",
                "request": tool_req,
                "response": {"count": len(docs), "query": search_query}
            })

            # Format document sections as tool chunks
            for d in docs:
                tool_chunks.append({
                    "content": d.get("content", ""),
                    "metadata": {
                        "source": d.get("source", "private_database"),
                        "title": d.get("title", "Internal Document"),
                        "chunk_index": d.get("chunk_index", 0),
                        "type": "private_doc"
                    },
                    "score": d.get("score", 0.0)
                })

            if not tool_chunks:
                tool_chunks.append({
                    "content": f"No document sections matched query '{search_query}' in the private document database.",
                    "metadata": {"source": str(script_path), "title": "Private Document Database"},
                    "score": 0.0
                })

        except Exception as e:
            raw_output = f"Error executing doc_tools.py: {e}"
            print(f"[SkillRunner] {raw_output}")

    # 2. stock-market-skill
    elif any("stock-market" in s or "stock" in s for s in active_skill_names) or "stock_tools.py" in inst_lower:
        script_path = SKILLS_DIR / "stock-market-skill" / "scripts" / "stock_tools.py"
        is_losers = any(w in inst_lower or w in q_lower for w in ["loser", "losers", "decrease", "drop", "dropped", "fell", "down", "negative"])
        flag = "--losers" if is_losers else "--gainers"
        cmd = [sys.executable, str(script_path), flag, "--limit", "5", "--json"]
        tool_req = {"tool": "stock_tools.py", "metric": flag, "command": cmd}

        try:
            res = subprocess.check_output(cmd, cwd=str(REPO_ROOT), timeout=15).decode("utf-8")
            raw_output = res
            parsed_res = json.loads(res) if res.strip().startswith("{") else {"raw_output": res}

            duration_ms = (time.time() - start_time) * 1000
            log_event(
                event_type="Tool",
                invoker="Agent",
                target="stock_tools.py (Live Screener / Registry)",
                short_description=f"Executed stock screener for {flag.replace('--', '')}",
                payload={"request": tool_req, "response": parsed_res},
                conversation_id=conversation_id,
                duration_ms=duration_ms,
                status="success"
            )

            components.append({
                "name": "Tool",
                "role": "Skill Script",
                "icon": "📈",
                "status": "success",
                "duration_ms": round(duration_ms, 2),
                "description": f"Executed stock_tools.py for {flag.replace('--', '')}",
                "request": tool_req,
                "response": {"metric": parsed_res.get("metric", flag), "count": parsed_res.get("count", 0)}
            })

            # Check for logged external APIs
            for api_info in parsed_res.get("external_apis", []):
                log_event(
                    event_type="External API",
                    invoker="Tool (stock_tools.py)",
                    target=api_info.get("name", "Market API"),
                    short_description=f"HTTP {api_info.get('method', 'GET')} {api_info.get('url', '')[:50]}...",
                    payload={"request": api_info.get("request", {}), "response": api_info.get("response", {})},
                    conversation_id=conversation_id,
                    duration_ms=round(duration_ms / 2, 2),
                    status="success"
                )

            tool_chunks.append({
                "content": f"=== LIVE TOOL EXECUTION: stock_tools.py ({flag}) ===\n{res}\n=== END LIVE TOOL RESULT ===",
                "metadata": {"source": str(script_path), "title": f"Live Stock Output ({flag})", "type": "live_tool_execution"},
                "score": 1.0
            })
        except Exception as e:
            raw_output = f"Error executing stock_tools.py: {e}"
            print(f"[SkillRunner] {raw_output}")

    # 3. time-weather-skill
    elif any("time-weather" in s or "weather" in s for s in active_skill_names) or "env_tools.py" in inst_lower:
        script_path = SKILLS_DIR / "time-weather-skill" / "scripts" / "env_tools.py"
        # Extract city from instruction or question
        city = "London"
        city_m = re.search(r'env_tools\.py\s+["\']?([^"\'\s\-]+)["\']?', instruction)
        if city_m and city_m.group(1).lower() not in ["python", "python3"]:
            city = city_m.group(1)
        else:
            for c in ["Tokyo", "Paris", "London", "New York", "San Francisco", "Berlin", "Sydney", "Chicago"]:
                if c.lower() in q_lower or c.lower() in inst_lower:
                    city = c
                    break

        unit = "fahrenheit" if ("fahrenheit" in q_lower or "fahrenheit" in inst_lower) else "celsius"
        cmd = [sys.executable, str(script_path), city, "--unit", unit, "--json"]
        tool_req = {"tool": "env_tools.py", "city": city, "unit": unit, "command": cmd}

        try:
            res = subprocess.check_output(cmd, cwd=str(REPO_ROOT), timeout=15).decode("utf-8")
            raw_output = res
            parsed_res = json.loads(res) if res.strip().startswith("{") else {"raw_output": res}

            duration_ms = (time.time() - start_time) * 1000
            log_event(
                event_type="Tool",
                invoker="Agent",
                target="env_tools.py (Open-Meteo API)",
                short_description=f"Executed env_tools.py for '{city}' ({unit})",
                payload={"request": tool_req, "response": parsed_res},
                conversation_id=conversation_id,
                duration_ms=duration_ms,
                status="success"
            )

            components.append({
                "name": "Tool",
                "role": "Skill Script",
                "icon": "🌤️",
                "status": "success",
                "duration_ms": round(duration_ms, 2),
                "description": f"Executed env_tools.py for '{city}' ({unit})",
                "request": tool_req,
                "response": {"city": parsed_res.get("city", city), "weather": parsed_res.get("weather", {})}
            })

            # Check for logged external APIs
            for api_info in parsed_res.get("external_apis", []):
                log_event(
                    event_type="External API",
                    invoker="Tool (env_tools.py)",
                    target=api_info.get("name", "Open-Meteo API"),
                    short_description=f"HTTP {api_info.get('method', 'GET')} {api_info.get('url', '')[:50]}...",
                    payload={"request": api_info.get("request", {}), "response": api_info.get("response", {})},
                    conversation_id=conversation_id,
                    duration_ms=round(duration_ms / 2, 2),
                    status="success"
                )

            tool_chunks.append({
                "content": f"=== LIVE TOOL EXECUTION: env_tools.py ({city}) ===\n{res}\n=== END LIVE TOOL RESULT ===",
                "metadata": {"source": str(script_path), "title": f"Live Weather Output: {city}", "type": "live_tool_execution"},
                "score": 1.0
            })
        except Exception as e:
            raw_output = f"Error executing env_tools.py: {e}"
            print(f"[SkillRunner] {raw_output}")

    return {
        "raw_output": raw_output,
        "chunks": tool_chunks,
        "components": components
    }


def run_agent_skill_pipeline(
    question: str,
    model: Optional[str] = None,
    custom_endpoint: Optional[str] = None,
    conversation_id: Optional[str] = None,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Full Agent Reasoning Loop:
    1. Embed prompt using local Ollama.
    2. Query the Skill Database with similarity threshold > 50% (> 0.50).
    3. If NO skill matches > 50%:
       - Send prompt to LLM model with the exact required system prompt:
         "You are a helpful, respectful, and honest assistant. Always answer as truthfully as possible, use clear markdown formatting, and admit when you do not know an answer rather than guessing."
       - Return direct answer.
    4. If skills MATCH (> 50%):
       - Step 1: Send prompt and list of matched SKILL.md files to the LLM model.
         LLM responds with the plan or execution instruction.
       - Step 2: Agent performs the instruction sent from the LLM model (executes script).
       - Step 3: Send results of instruction execution + prompt + SKILL.md to the LLM model.
         LLM responds with the final synthesized user answer.
    """
    total_start_time = time.time()
    embedder = OllamaEmbeddingFunction()

    # Step 1: Pass prompt to embedder
    emb_start = time.time()
    prompt_embedding = embedder.embed_query(question)
    emb_duration = (time.time() - emb_start) * 1000

    log_event(
        event_type="Embedder",
        invoker="Agent",
        target=f"Local Ollama ({embedder.model})",
        short_description=f"Generated vector embedding ({len(prompt_embedding)} dims)",
        payload={"prompt": question, "dimensions": len(prompt_embedding)},
        conversation_id=conversation_id,
        duration_ms=emb_duration,
        status="success"
    )

    # Step 2: Query skill database
    skill_db_start = time.time()
    matched_skills = skill_vector_store.query(
        query_text=question,
        query_embedding=prompt_embedding,
        top_k=5,
        min_score=0.50
    )
    skill_db_duration = (time.time() - skill_db_start) * 1000

    log_event(
        event_type="Vector Store (Skills)",
        invoker="Agent",
        target="ChromaDB (database/chroma_skills)",
        short_description=f"Queried skill database: {len(matched_skills)} candidate(s) > 50%",
        payload={
            "query": question,
            "min_score": 0.50,
            "matched_skills": [{"name": s.get("name"), "score": s.get("score")} for s in matched_skills]
        },
        conversation_id=conversation_id,
        duration_ms=skill_db_duration,
        status="success"
    )

    components: List[Dict[str, Any]] = [
        {
            "name": "Embedder",
            "role": "Local Embedding",
            "icon": "🧠",
            "status": "success",
            "duration_ms": round(emb_duration, 2),
            "description": f"Embedded prompt via Ollama ({len(prompt_embedding)} dims)",
            "request": {"prompt": question},
            "response": {"dimensions": len(prompt_embedding)}
        },
        {
            "name": "Vector Store",
            "role": "Skill Vector DB",
            "icon": "⚡",
            "status": "success",
            "duration_ms": round(skill_db_duration, 2),
            "description": f"Scanned skill database (found {len(matched_skills)} candidate(s) > 50% score)",
            "request": {"query": question, "threshold": 0.50},
            "response": {
                "matches_count": len(matched_skills),
                "retrieved_count": len(matched_skills),
                "skills": [s.get("name") for s in matched_skills]
            }
        },
        {
            "name": "Skill Store",
            "role": "Skill Vector DB",
            "icon": "⚡",
            "status": "success",
            "duration_ms": round(skill_db_duration, 2),
            "description": f"Scanned skill database (found {len(matched_skills)} candidate(s) > 50% score)",
            "request": {"query": question, "threshold": 0.50},
            "response": {
                "matches_count": len(matched_skills),
                "retrieved_count": len(matched_skills),
                "skills": [s.get("name") for s in matched_skills]
            }
        }
    ]

    # Branch A: No skill with > 50% match
    if not matched_skills:
        llm_resp = gemma_service.answer_question(
            question=question,
            retrieved_chunks=[],
            model=model,
            system_instruction=NO_SKILL_SYSTEM_PROMPT,
            conversation_id=conversation_id,
            custom_endpoint=custom_endpoint
        )
        if llm_resp.get("component"):
            components.append(llm_resp["component"])

        total_duration = (time.time() - total_start_time) * 1000
        return {
            "answer": llm_resp.get("answer", ""),
            "model": llm_resp.get("model", ""),
            "sources": [],
            "components": components,
            "has_skill": False,
            "duration_ms": round(total_duration, 2)
        }

    # Branch B: Skills matched (> 50%)
    # Build skills text representation containing complete SKILL.md content
    skills_context = []
    for s in matched_skills:
        skills_context.append(
            f"=== SKILL: {s.get('name')} (Relevance: {s.get('score')}) ===\n"
            f"{s.get('content')}\n"
            f"=== END SKILL ==="
        )
    skills_text = "\n\n".join(skills_context)

    # 1. Ask LLM for Plan / Execution Instruction
    planner_system = (
        "You are the autonomous planning module of Agent with RAG. "
        "Review the user question and the provided skill Standard Operating Procedures (SOPs). "
        "Formulate a concise execution plan or tool command instruction to fulfill the user request. "
        "Specify the exact script command to execute (e.g., `python skills/.../scripts/...py <arguments>`)."
    )
    planner_prompt = (
        f"Available Active Skills:\n\n{skills_text}\n\n"
        f"User Question: {question}\n\n"
        f"Please provide the plan and execution instruction for the agent."
    )

    planner_llm_resp = gemma_service.answer_question(
        question=planner_prompt,
        retrieved_chunks=[],
        model=model,
        system_instruction=planner_system,
        conversation_id=conversation_id,
        custom_endpoint=custom_endpoint
    )
    instruction = planner_llm_resp.get("answer", "").strip()

    components.append({
        "name": "Planner",
        "role": "LLM Planning Engine",
        "icon": "📋",
        "status": "success",
        "duration_ms": planner_llm_resp.get("duration_ms", 0.0),
        "description": "Generated tool execution plan & command from skill SOPs",
        "request": {"question": question, "matched_skills": [s.get("name") for s in matched_skills]},
        "response": {"instruction": instruction[:200] + "..." if len(instruction) > 200 else instruction}
    })

    # 2. Agent performs the instruction
    exec_result = execute_instruction(
        instruction=instruction,
        question=question,
        active_skills=matched_skills,
        conversation_id=conversation_id
    )
    tool_chunks = exec_result.get("chunks", [])
    components.extend(exec_result.get("components", []))

    # 3. Synthesize final answer: Send execution results, prompt, and SKILL.md to LLM
    synthesis_chunks = []
    # Add matched SKILL.md context as Document 1
    for i, s in enumerate(matched_skills, 1):
        synthesis_chunks.append({
            "content": s.get("content", ""),
            "metadata": {
                "source": s.get("metadata", {}).get("source", f"skills/{s.get('skill_id')}/SKILL.md"),
                "title": f"Skill: {s.get('name')}"
            },
            "score": s.get("score", 1.0)
        })
    # Add live tool execution chunks
    synthesis_chunks.extend(tool_chunks)

    synthesis_system = (
        "You are a helpful, accurate, and privacy-preserving AI assistant for Agent with RAG. "
        "Thoroughly answer the user's question using the provided tool execution results and skill guidelines. "
        "Cite the sources and documents accurately."
    )

    final_llm_resp = gemma_service.answer_question(
        question=question,
        retrieved_chunks=synthesis_chunks,
        model=model,
        system_instruction=synthesis_system,
        conversation_id=conversation_id,
        custom_endpoint=custom_endpoint
    )
    if final_llm_resp.get("component"):
        components.append(final_llm_resp["component"])

    total_duration = (time.time() - total_start_time) * 1000
    return {
        "answer": final_llm_resp.get("answer", ""),
        "model": final_llm_resp.get("model", ""),
        "sources": tool_chunks if tool_chunks else synthesis_chunks,
        "components": components,
        "has_skill": True,
        "instruction": instruction,
        "duration_ms": round(total_duration, 2)
    }


def execute_skill_tools_if_relevant(
    question: str,
    retrieved_chunks: List[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Backwards compatibility function for executing skill tools.
    """
    active_skills = []
    # Check if question activates any skill
    emb = OllamaEmbeddingFunction().embed_query(question)
    skills = skill_vector_store.query(question, query_embedding=emb, min_score=0.50)
    if skills:
        active_skills = skills

    res = execute_instruction(
        instruction=question,
        question=question,
        active_skills=active_skills,
        conversation_id=conversation_id
    )
    return {
        "chunks": res.get("chunks", []),
        "components": res.get("components", [])
    }
