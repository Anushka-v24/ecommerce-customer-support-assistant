"""Local Markdown/text loader, metadata parser, and deterministic chunker."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path


SUPPORTED_SUFFIXES = {".md", ".txt"}
_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def _metadata_and_body(text: str, source: str, root: Path) -> tuple[dict[str, object], str]:
    """Read the small, dependency-free YAML-like metadata header used by KB files."""
    values: dict[str, str] = {}
    body = text
    if text.startswith("---\n") or text.startswith("---\r\n"):
        lines = text.splitlines()
        try:
            closing = lines.index("---", 1)
        except ValueError as exc:
            raise ValueError(f"Knowledge document {source} has an unterminated metadata header") from exc
        for line in lines[1:closing]:
            if not line.strip():
                continue
            key, separator, value = line.partition(":")
            if not separator or not key.strip() or key.strip() in values:
                raise ValueError(f"Knowledge document {source} has malformed or duplicate metadata")
            values[key.strip()] = value.strip().strip('"').strip("'")
        body = "\n".join(lines[closing + 1:]).strip()

    unknown = set(values) - {"document_id", "category", "language", "last_updated", "status"}
    if unknown:
        raise ValueError(f"Knowledge document {source} has unsupported metadata fields: {', '.join(sorted(unknown))}")
    document_id = values.get("document_id") or "doc-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    if not _DOCUMENT_ID.fullmatch(document_id):
        raise ValueError(f"Knowledge document {source} has invalid document_id")
    last_updated = values.get("last_updated") or "unknown"
    if last_updated != "unknown":
        try:
            date.fromisoformat(last_updated)
        except ValueError as exc:
            raise ValueError(f"Knowledge document {source} last_updated must use YYYY-MM-DD") from exc
    relative_parts = Path(source).parts
    is_demo_path = root.name.lower() == "demo" or any(part.lower() == "demo" for part in relative_parts)
    status = values.get("status", "demo" if is_demo_path else "approved").lower()
    if status not in {"approved", "demo"}:
        raise ValueError(f"Knowledge document {source} status must be approved or demo")
    if is_demo_path and status != "demo":
        raise ValueError(f"Knowledge document {source} is in the demo directory but is not marked demo")
    category = values.get("category") or (Path(source).parent.name if "/" in source else "store_faq")
    language = values.get("language", "und").lower()
    if not category.strip() or not re.fullmatch(r"[a-z0-9_-]+", category.lower()):
        raise ValueError(f"Knowledge document {source} has invalid category")
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})?", language):
        raise ValueError(f"Knowledge document {source} has invalid language")
    return {
        "document_id": document_id,
        "filename": Path(source).name,
        "category": category.lower(),
        "language": language,
        "source": source,
        "last_updated": last_updated,
        "status": status,
        "is_demo": status == "demo",
    }, body


def load_chunks(directory: str | Path, chunk_size: int = 1200, overlap: int = 150) -> list[dict[str, object]]:
    """Load approved Markdown/text documents and return source-tagged chunks."""
    root = Path(directory).resolve()
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError("Knowledge-base path must be a directory")

    chunks: list[dict[str, object]] = []
    document_ids: set[str] = set()
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name.lower() == "readme.md"
            or any(part.startswith(".") for part in path.relative_to(root).parts)
            or path.suffix.lower() not in SUPPORTED_SUFFIXES
        ):
            continue
        raw_text = path.read_text(encoding="utf-8").strip()
        source = path.relative_to(root).as_posix()
        metadata, text = _metadata_and_body(raw_text, source, root)
        if not text:
            continue
        document_id = str(metadata["document_id"])
        if document_id in document_ids:
            raise ValueError(f"Duplicate knowledge document_id: {document_id}")
        document_ids.add(document_id)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        version_material = "\n".join(
            str(metadata[key]) for key in ("document_id", "category", "language", "source", "last_updated", "status")
        ) + "\n" + content_hash
        version_hash = hashlib.sha256(version_material.encode("utf-8")).hexdigest()
        start = 0
        chunk_number = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + (1 if text[boundary] == "\n" else 2)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "id": f"eskb-{document_id}-{version_hash[:16]}-{chunk_number}",
                    "text": chunk,
                    **metadata,
                    "topic": Path(source).parent.name if "/" in source else "general",
                    "chunk_index": chunk_number,
                    "content_hash": content_hash,
                    "version_hash": version_hash,
                })
                chunk_number += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
