"""Explicit command for indexing approved local Markdown/text knowledge documents."""

from __future__ import annotations

import sys
import argparse

from app.config import Config
from app.rag.documents import load_chunks
from app.rag.service import RAGConfigurationError, RAGProviderError, RAGService, get_rag_service


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize approved or explicitly selected demo RAG documents.")
    parser.add_argument("--demo", action="store_true", help="Index only the separate fictional demo corpus in its own Pinecone namespace (development only).")
    parser.add_argument("--prune-empty", action="store_true", help="Delete this indexer's managed vectors when the selected directory is empty.")
    args = parser.parse_args()
    source_directory = Config.RAG_DEMO_KNOWLEDGE_DIR if args.demo else Config.RAG_KNOWLEDGE_DIR
    try:
        chunks = load_chunks(
            source_directory,
            chunk_size=Config.RAG_CHUNK_SIZE,
            overlap=Config.RAG_CHUNK_OVERLAP,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Could not read knowledge documents: {exc}", file=sys.stderr)
        return 2
    if not chunks and not args.prune_empty:
        print(f"No .md or .txt knowledge documents found in {source_directory}; nothing indexed.")
        return 0
    try:
        service = get_rag_service() if not args.demo else RAGService(include_demo=True)
        result = service.ingest(
            source_directory,
            allow_demo=args.demo,
            prune_empty=args.prune_empty,
        )
    except (RAGConfigurationError, RAGProviderError) as exc:
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 2
    target = Config.PINECONE_INDEX_NAME if Config.RAG_VECTOR_STORE == "pinecone" else Config.LOCAL_VECTOR_STORE_PATH
    print(
        f"Indexed {result['documents']} documents as {result['chunks']} chunks "
        f"({result['upserted']} new/changed vectors, {result['unchanged']} unchanged, "
        f"{result['removed']} stale vectors removed) in the {Config.RAG_VECTOR_STORE} vector store ({target})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
