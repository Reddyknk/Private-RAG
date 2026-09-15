import os
import re
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from pypdf import PdfReader

from services.logger_service import log_call


class Document:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"<Document source={self.metadata.get('source')} length={len(self.content)}>"


def extract_skill_metadata(text: str, default_name: str = "skill") -> tuple[str, str]:
    """
    Extracts name and description from a SKILL.md file.
    Supports YAML frontmatter delimited by '---' as well as markdown headers.
    """
    name = ""
    description = ""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                stripped = line.strip()
                if stripped.startswith("name:"):
                    name = stripped.replace("name:", "", 1).strip()
                elif stripped.startswith("description:"):
                    description = stripped.replace("description:", "", 1).strip()

    if not name or not description:
        for line in text.splitlines():
            stripped = line.strip()
            if not name and stripped.startswith("# "):
                name = stripped.replace("# ", "", 1).strip()
            elif not description and stripped.lower().startswith("description:"):
                description = stripped.split(":", 1)[1].strip()

    name = name or default_name
    return name, description


def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """
    Split a body of text into overlapping chunks respecting sentence/paragraph boundaries where possible.
    """
    if not text or not text.strip():
        return []

    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Look for sentence boundary near end
        boundary = max(
            text.rfind(". ", start, end),
            text.rfind("? ", start, end),
            text.rfind("! ", start, end),
            text.rfind("\n", start, end)
        )

        if boundary != -1 and boundary > start + (chunk_size // 2):
            chunks.append(text[start:boundary + 1].strip())
            start = boundary + 1 - chunk_overlap
        else:
            # Fall back to word boundary
            space = text.rfind(" ", start, end)
            if space != -1 and space > start + (chunk_size // 2):
                chunks.append(text[start:space].strip())
                start = space + 1 - chunk_overlap
            else:
                chunks.append(text[start:end].strip())
                start = end - chunk_overlap

        # Ensure forward progress
        if start <= 0 and len(chunks) > 0:
            start = end

    return [c for c in chunks if c]


def load_from_url(url: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    """
    Fetch a web page or online document, extract clean text,
    and log the external HTTP call to database/logs.json.
    """
    start_time = time.time()
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        duration_ms = (time.time() - start_time) * 1000

        content_type = resp.headers.get("Content-Type", "").lower()
        extracted_text = ""
        doc_title = url

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            import io
            reader = PdfReader(io.BytesIO(resp.content))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            extracted_text = "\n\n".join(pages_text)
            doc_title = Path(url).name or "Online PDF Document"
        else:
            soup = BeautifulSoup(resp.content, "html.parser")
            # Remove scripts and styles
            for elem in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                elem.decompose()

            if soup.title and soup.title.string:
                doc_title = soup.title.string.strip()

            # Prefer main article container if present
            main_content = soup.find("article") or soup.find("main") or soup.find("div", {"role": "main"}) or soup.body
            if main_content:
                extracted_text = main_content.get_text(separator="\n", strip=True)
            else:
                extracted_text = soup.get_text(separator="\n", strip=True)

        log_call(
            call_type="external_url_fetch",
            arguments={
                "url": url,
                "headers": headers,
                "timeout": 20
            },
            response={
                "status_code": resp.status_code,
                "content_type": content_type,
                "raw_bytes": len(resp.content),
                "extracted_text_chars": len(extracted_text),
                "title": doc_title
            },
            duration_ms=duration_ms,
            status="success" if resp.status_code == 200 else "error"
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch URL ({resp.status_code}): {resp.reason}")

        chunks = split_text_into_chunks(extracted_text, chunk_size, chunk_overlap)
        documents: List[Document] = []
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                content=chunk,
                metadata={
                    "source": url,
                    "source_type": "url",
                    "title": doc_title,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            ))

        return documents

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_call(
            call_type="external_url_fetch",
            arguments={"url": url, "headers": headers},
            response={"error": str(e)},
            duration_ms=duration_ms,
            status="error"
        )
        raise


def load_from_directory(
    dir_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    allowed_extensions: Optional[List[str]] = None
) -> List[Document]:
    """
    Recursively scan a local directory, read documents, and split them into chunks.
    """
    if allowed_extensions is None:
        allowed_extensions = [".txt", ".md", ".markdown", ".pdf", ".py", ".json", ".csv", ".html", ".rst"]

    resolved_path = Path(dir_path).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path} (resolved as {resolved_path})")

    documents: List[Document] = []

    for file_path in resolved_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in allowed_extensions:
            continue
        # Skip hidden files or pycache
        if any(part.startswith(".") or part == "__pycache__" for part in file_path.parts):
            continue

        file_text = ""
        doc_title = file_path.name

        try:
            if file_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(file_path))
                pages_text = []
                for p in reader.pages:
                    txt = p.extract_text()
                    if txt:
                        pages_text.append(txt)
                file_text = "\n\n".join(pages_text)
            else:
                try:
                    file_text = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    file_text = file_path.read_text(encoding="latin-1", errors="ignore")

            if not file_text.strip():
                continue

            is_skill_md = file_path.name.lower() == "skill.md"
            if is_skill_md:
                # When adding SKILL.md to the vector store: The chunk is the whole file!
                skill_name, skill_desc = extract_skill_metadata(file_text, default_name=file_path.parent.name)
                documents.append(Document(
                    content=file_text,
                    metadata={
                        "source": str(file_path),
                        "relative_path": str(file_path.relative_to(resolved_path)),
                        "source_type": "skill",
                        "title": f"Skill: {skill_name}",
                        "filename": file_path.name,
                        "file_extension": file_path.suffix.lower(),
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "is_skill_md": True,
                        "skill_name": skill_name,
                        "skill_description": skill_desc
                    }
                ))
            else:
                chunks = split_text_into_chunks(file_text, chunk_size, chunk_overlap)
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        content=chunk,
                        metadata={
                            "source": str(file_path),
                            "relative_path": str(file_path.relative_to(resolved_path)),
                            "source_type": "directory",
                            "title": doc_title,
                            "file_extension": file_path.suffix.lower(),
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    ))
        except Exception as e:
            print(f"[DocumentLoader] Skipping {file_path} due to error: {e}")

    return documents
