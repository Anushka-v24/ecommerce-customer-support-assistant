"""Free, local replacements for the paid RAG providers.

``LocalEmbeddingClient`` embeds text with a multilingual Hugging Face model on this
machine (OpenAI-embeddings-shaped: ``.embeddings.create``). ``LocalVectorIndex``
keeps vectors in a JSON file and searches them with numpy (Pinecone-shaped:
``upsert`` / ``query`` / ``list`` / ``delete``). Both plug into ``RAGService``
exactly where the OpenAI and Pinecone clients go, so the retrieval workflow is
unchanged. Answer generation uses Groq's OpenAI-compatible chat API.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.config import Config

logger = logging.getLogger(__name__)
_MAX_TOKENS = 512


class LocalEmbeddingClient:
    """Dense embeddings from a local model (CLS pooling, L2-normalised, as BGE-M3 expects)."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or Config.LOCAL_EMBEDDING_MODEL
        self._lock = threading.Lock()
        self._loaded: tuple[Any, Any, Any, str] | None = None
        self.embeddings = SimpleNamespace(create=self._create)

    def _load(self) -> tuple[Any, Any, Any, str]:
        with self._lock:
            if self._loaded is None:
                from app.rag.service import RAGConfigurationError

                try:
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                    model = AutoModel.from_pretrained(self.model_name, local_files_only=True).eval()
                except Exception as exc:
                    raise RAGConfigurationError(
                        f"Local embedding model {self.model_name} is not downloaded; "
                        "run: python -c \"from huggingface_hub import snapshot_download; "
                        f"snapshot_download('{self.model_name}')\""
                    ) from exc
                device = "mps" if torch.backends.mps.is_available() else "cpu"
                self._loaded = (torch, tokenizer, model.to(device), device)
            return self._loaded

    def _create(self, *, model: str | None = None, input: list[str]) -> SimpleNamespace:  # noqa: A002 (OpenAI's name)
        torch, tokenizer, model_, device = self._load()
        vectors: list[list[float]] = []
        for start in range(0, len(input), 16):
            encoded = tokenizer(input[start:start + 16], padding=True, truncation=True,
                                max_length=_MAX_TOKENS, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model_(**encoded).last_hidden_state[:, 0]
            vectors += torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().tolist()
        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


class LocalVectorIndex:
    """A small on-disk vector index with Pinecone's calling convention (cosine similarity)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or Config.LOCAL_VECTOR_STORE_PATH)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict[str, dict]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def _write(self, data: dict[str, dict[str, dict]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

    def upsert(self, *, vectors: list[dict], namespace: str, **_: Any) -> None:
        with self._lock:
            data = self._read()
            space = data.setdefault(namespace, {})
            for item in vectors:
                space[item["id"]] = {"values": item["values"], "metadata": item.get("metadata") or {}}
            self._write(data)

    def delete(self, *, ids: list[str], namespace: str, **_: Any) -> None:
        with self._lock:
            data = self._read()
            for identifier in ids:
                data.get(namespace, {}).pop(identifier, None)
            self._write(data)

    def list(self, *, prefix: str = "", namespace: str, **_: Any):
        ids = [i for i in self._read().get(namespace, {}) if i.startswith(prefix)]
        yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in ids])

    def query(self, *, vector: list[float], top_k: int, namespace: str, include_metadata: bool = True, **_: Any) -> dict:
        import numpy as np

        space = self._read().get(namespace, {})
        if not space:
            from app.rag.service import RAGConfigurationError

            raise RAGConfigurationError(
                "The local vector index is empty; run: python -m app.rag.ingest" + (" --demo" if "demo" in namespace else "")
            )
        ids = list(space)
        matrix = np.array([space[i]["values"] for i in ids], dtype="float32")
        query = np.array(vector, dtype="float32")
        # Vectors are stored normalised, but renormalise so any embedding source works.
        scores = (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)) @ (query / np.linalg.norm(query))
        best = np.argsort(-scores)[:top_k]
        return {"matches": [{"id": ids[i], "score": float(scores[i]), "metadata": space[ids[i]]["metadata"]} for i in best]}
