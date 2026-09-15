import os
import re
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from services.document_loader import load_from_directory, Document
from services.vector_store import vector_store
from services.logger_service import log_event


SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def get_available_skills() -> List[Dict[str, Any]]:
    """List all skills discovered in the skills/ directory."""
    if not SKILLS_DIR.exists():
        return []

    skills = []
    for skill_folder in SKILLS_DIR.iterdir():
        if not skill_folder.is_dir():
            continue
        skill_md = skill_folder / "SKILL.md"
        if not skill_md.exists():
            continue

        # Parse frontmatter if present
        title = skill_folder.name
        desc = ""
        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                content = f.read()
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        for line in parts[1].splitlines():
                            if line.startswith("name:"):
                                title = line.replace("name:", "").strip()
                            elif line.startswith("description:"):
                                desc = line.replace("description:", "").strip()
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
    Auto-discovers all skills in skills/ and embeds them into the vector database
    using Ollama embeddings as specified in SKILL_SPEC.md.
    """
    if not SKILLS_DIR.exists():
        return {"status": "skipped", "message": "skills/ directory does not exist."}

    try:
        docs = load_from_directory(str(SKILLS_DIR))
        if not docs:
            return {"status": "skipped", "message": "No skill files found to index."}

        # Check existing sources in DB to avoid unnecessary re-indexing if unchanged
        stats = vector_store.get_stats()
        existing_sources = set(stats.get("sources_list", []) or stats.get("distinct_sources", []))
        new_docs = [d for d in docs if d.metadata.get("source") not in existing_sources]

        if not new_docs:
            return {
                "status": "up_to_date",
                "message": "All skills are already indexed in the vector database.",
                "total_documents": stats.get("total_documents", 0)
            }

        result = vector_store.add_documents(new_docs)
        return {
            "status": "indexed",
            "message": f"Successfully indexed {len(new_docs)} new skill chunks into vector database.",
            "added_chunks": len(new_docs),
            "total_documents": result.get("total_documents_in_db", 0)
        }
    except Exception as e:
        return {"status": "error", "message": f"Skill indexing failed: {e}"}


def execute_skill_tools_if_relevant(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Examines retrieved chunks and user question.
    If a skill tool applies, executes the bundled tool script, logs Tool and External API
    invocations asynchronously into database/logs.json with full request/response payloads,
    and returns both chunks and structured component descriptions for the message.
    """
    sources = [c.get("metadata", {}).get("source", "") for c in retrieved_chunks]
    q_lower = question.lower()
    tool_chunks: List[Dict[str, Any]] = []
    components: List[Dict[str, Any]] = []

    # 1. Check for time & weather skill
    has_weather_skill = any("time-weather-skill" in s for s in sources) or any(
        kw in q_lower for kw in ["weather", "time in", "temperature", "humidity", "weather of", "time of"]
    )
    if has_weather_skill:
        # Extract city from question
        patterns = [
            r'(?:time and weather of|time and weather in|weather of|weather in|time of|time in|temperature in|weather for|time for)\s+(?:the city of\s+)?([A-Za-z\s\.\-]+?)(?:\?|\.|\$|,|$)',
            r'in\s+([A-Za-z\s\.\-]+?)(?:\?|\.|\$|,|$)',
            r'of\s+([A-Za-z\s\.\-]+?)(?:\?|\.|\$|,|$)'
        ]
        city = None
        for p in patterns:
            m = re.search(p, question, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if cand and len(cand) > 1 and cand.lower() not in [
                    "the city", "a city", "this city", "city", "my city", "stock", "stocks"
                ]:
                    city = cand
                    break

        if not city:
            for common_city in ["Tokyo", "Paris", "London", "New York", "San Francisco", "Berlin", "Sydney", "Singapore", "Toronto", "Chicago", "Dubai"]:
                if common_city.lower() in q_lower:
                    city = common_city
                    break

        if city:
            unit = "fahrenheit" if ("fahrenheit" in q_lower or " °f" in q_lower) else "celsius"
            script_path = SKILLS_DIR / "time-weather-skill" / "scripts" / "env_tools.py"
            if script_path.exists():
                start_tool_time = time.time()
                cmd = ["python3", str(script_path), city, "--unit", unit, "--json"]
                tool_req = {
                    "tool": "env_tools.py",
                    "city": city,
                    "unit": unit,
                    "command": cmd
                }
                try:
                    res = subprocess.check_output(cmd, timeout=8).decode("utf-8")
                    duration_ms = (time.time() - start_tool_time) * 1000

                    try:
                        parsed_res = json.loads(res)
                    except Exception:
                        parsed_res = {"raw_output": res}

                    # Log Tool Invocation
                    log_event(
                        event_type="Tool",
                        invoker="Agent",
                        target="env_tools.py (Open-Meteo API)",
                        short_description=f"Executed env_tools.py for '{city}' ({unit})",
                        payload={
                            "request": tool_req,
                            "response": parsed_res
                        },
                        conversation_id=conversation_id,
                        duration_ms=duration_ms,
                        status="success"
                    )

                    components.append({
                        "name": "Tool",
                        "role": "Skill Script",
                        "icon": "🛠️",
                        "status": "success",
                        "duration_ms": round(duration_ms, 2),
                        "description": f"Executed env_tools.py for city '{city}' ({unit})",
                        "request": tool_req,
                        "response": {
                            "status": parsed_res.get("status", "success"),
                            "city": parsed_res.get("city", city),
                            "local_time": parsed_res.get("local_time"),
                            "weather": parsed_res.get("weather", {})
                        }
                    })

                    # Log each External API call performed by the tool
                    external_apis = parsed_res.get("external_apis", [])
                    if external_apis:
                        for api_info in external_apis:
                            api_name = api_info.get("name", "Open-Meteo API")
                            api_url = api_info.get("url", "")
                            api_method = api_info.get("method", "GET")
                            api_status = api_info.get("status_code", 200)

                            api_req = api_info.get("request", {"method": api_method, "url": api_url})
                            api_resp = api_info.get("response", {"status_code": api_status})

                            log_event(
                                event_type="External API",
                                invoker="Tool (env_tools.py)",
                                target=api_name,
                                short_description=f"HTTP {api_method} {api_url[:55]}...",
                                payload={
                                    "request": api_req,
                                    "response": api_resp
                                },
                                conversation_id=conversation_id,
                                duration_ms=round(duration_ms / max(1, len(external_apis)), 2),
                                status="success" if api_status == 200 else "error"
                            )

                            components.append({
                                "name": "External API",
                                "role": "Public Web API",
                                "icon": "🌐",
                                "status": "success" if api_status == 200 else "error",
                                "duration_ms": round(duration_ms / max(1, len(external_apis)), 2),
                                "description": f"Called {api_name} ({api_url[:45]}...)",
                                "request": api_req,
                                "response": api_resp
                            })

                    tool_chunks.append({
                        "content": (
                            f"=== LIVE TOOL EXECUTION: env_tools.py ({city}) ===\n"
                            f"{res}\n"
                            f"=== END LIVE TOOL RESULT ==="
                        ),
                        "metadata": {
                            "source": str(script_path),
                            "title": f"Live Weather & Time Output: {city}",
                            "type": "live_tool_execution"
                        },
                        "score": 1.0
                    })
                except Exception as e:
                    duration_ms = (time.time() - start_tool_time) * 1000
                    err_resp = {"error": str(e)}
                    log_event(
                        event_type="Tool",
                        invoker="Agent",
                        target="env_tools.py (Open-Meteo API)",
                        short_description=f"Error executing weather tool for '{city}'",
                        payload={"request": tool_req, "response": err_resp},
                        conversation_id=conversation_id,
                        duration_ms=duration_ms,
                        status="error"
                    )
                    components.append({
                        "name": "Tool",
                        "role": "Skill Script",
                        "icon": "🛠️",
                        "status": "error",
                        "duration_ms": round(duration_ms, 2),
                        "description": f"Failed executing env_tools.py: {str(e)}",
                        "request": tool_req,
                        "response": err_resp
                    })
                    print(f"[SkillRunner] Error executing env_tools.py: {e}")

    # 2. Check for stock market screener skill
    has_stock_skill = any("stock-market-skill" in s for s in sources) or any(
        kw in q_lower for kw in [
            "stock", "stocks", "gainers", "losers", "percentage increase",
            "percentage decrease", "top gainer", "top loser", "market mover"
        ]
    )
    if has_stock_skill:
        is_losers = any(kw in q_lower for kw in [
            "decrease", "loss", "losers", "drop", "dropped", "fell", "fall", "down", "negative", "declined"
        ])
        flag = "--losers" if is_losers else "--gainers"
        script_path = SKILLS_DIR / "stock-market-skill" / "scripts" / "stock_tools.py"

        if script_path.exists():
            start_tool_time = time.time()
            cmd = ["python3", str(script_path), flag, "--limit", "5", "--json"]
            tool_req = {
                "tool": "stock_tools.py",
                "metric": flag,
                "command": cmd
            }
            try:
                res = subprocess.check_output(cmd, timeout=8).decode("utf-8")
                duration_ms = (time.time() - start_tool_time) * 1000

                try:
                    parsed_res = json.loads(res)
                except Exception:
                    parsed_res = {"raw_output": res}

                # Log Tool Invocation
                log_event(
                    event_type="Tool",
                    invoker="Agent",
                    target="stock_tools.py (Yahoo Finance Screener)",
                    short_description=f"Executed stock screener for {flag.replace('--', '')}",
                    payload={
                        "request": tool_req,
                        "response": parsed_res
                    },
                    conversation_id=conversation_id,
                    duration_ms=duration_ms,
                    status="success"
                )

                components.append({
                    "name": "Tool",
                    "role": "Skill Script",
                    "icon": "🛠️",
                    "status": "success",
                    "duration_ms": round(duration_ms, 2),
                    "description": f"Executed stock_tools.py screener for {flag.replace('--', '')}",
                    "request": tool_req,
                    "response": {
                        "metric": parsed_res.get("metric", flag),
                        "count": parsed_res.get("count", len(parsed_res.get("data", []))),
                        "top_ticker": parsed_res.get("data", [{}])[0].get("symbol") if parsed_res.get("data") else None
                    }
                })

                # Log each External API call performed by the tool
                external_apis = parsed_res.get("external_apis", [])
                if external_apis:
                    for api_info in external_apis:
                        api_name = api_info.get("name", "Yahoo Finance API")
                        api_url = api_info.get("url", "")
                        api_method = api_info.get("method", "GET")
                        api_status = api_info.get("status_code", 200)

                        api_req = api_info.get("request", {"method": api_method, "url": api_url})
                        api_resp = api_info.get("response", {"status_code": api_status})

                        log_event(
                            event_type="External API",
                            invoker="Tool (stock_tools.py)",
                            target=api_name,
                            short_description=f"HTTP {api_method} {api_url[:55]}...",
                            payload={
                                "request": api_req,
                                "response": api_resp
                            },
                            conversation_id=conversation_id,
                            duration_ms=round(duration_ms, 2),
                            status="success" if api_status == 200 else "error"
                        )

                        components.append({
                            "name": "External API",
                            "role": "Public Web API",
                            "icon": "🌐",
                            "status": "success" if api_status == 200 else "error",
                            "duration_ms": round(duration_ms, 2),
                            "description": f"Called {api_name}",
                            "request": api_req,
                            "response": api_resp
                        })

                tool_chunks.append({
                    "content": (
                        f"=== LIVE TOOL EXECUTION: stock_tools.py ({flag}) ===\n"
                        f"{res}\n"
                        f"=== END LIVE TOOL RESULT ==="
                    ),
                    "metadata": {
                        "source": str(script_path),
                        "title": f"Live Stock Screener Output ({flag})",
                        "type": "live_tool_execution"
                    },
                    "score": 1.0
                })
            except Exception as e:
                duration_ms = (time.time() - start_tool_time) * 1000
                err_resp = {"error": str(e)}
                log_event(
                    event_type="Tool",
                    invoker="Agent",
                    target="stock_tools.py (Yahoo Finance Screener)",
                    short_description=f"Error executing stock screener ({flag})",
                    payload={"request": tool_req, "response": err_resp},
                    conversation_id=conversation_id,
                    duration_ms=duration_ms,
                    status="error"
                )
                components.append({
                    "name": "Tool",
                    "role": "Skill Script",
                    "icon": "🛠️",
                    "status": "error",
                    "duration_ms": round(duration_ms, 2),
                    "description": f"Failed executing stock_tools.py: {str(e)}",
                    "request": tool_req,
                    "response": err_resp
                })
                print(f"[SkillRunner] Error executing stock_tools.py: {e}")

    return {
        "chunks": tool_chunks,
        "components": components
    }
