from __future__ import annotations

import argparse
from typing import Any

from app.config import get_config
from rag.embeddings import create_embedding_model
from rag.loader import load_knowledge_documents
from rag.splitter import split_documents
from rag.vectorstore import MilvusVectorStore


def ingest(rebuild: bool = False) -> dict[str, Any]:
    config = get_config()
    documents = load_knowledge_documents(config)
    chunks = split_documents(documents, config)
    embedding_model = create_embedding_model(config)
    vectorstore = MilvusVectorStore(config=config, embedding_model=embedding_model)
    upserted = vectorstore.upsert_documents(chunks, rebuild=rebuild)

    return {
        "knowledge_file": str(config.knowledge_file),
        "milvus_url": config.milvus_url,
        "db_name": config.db_name,
        "collection_name": config.collection_name,
        "documents": len(documents),
        "chunks": len(chunks),
        "upserted": upserted,
        "rebuild": rebuild,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the knowledge base into Milvus."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection before writing.",
    )
    args = parser.parse_args()

    result = ingest(rebuild=args.rebuild)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
