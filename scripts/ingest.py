from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.config import RagConfig, get_config
from rag.embeddings import create_embedding_model
from rag.loader import load_documents_from_path
from rag.partitions import registry_from_config
from rag.splitter import split_documents
from rag.vectorstore import MilvusVectorStore


def ingest(rebuild: bool = False, source: str | Path | None = None) -> dict[str, Any]:
    config = get_config()
    if registry_from_config(config) is not None:
        raise RuntimeError(
            "Partition mode is enabled (PARTITION_CONFIG is set in .env). "
            "Run `python -m scripts.ingest_partitioned` instead; this flat "
            "ingest script only serves the non-partitioned pipeline."
        )
    knowledge_source = Path(source) if source else _default_ingest_source(config)
    if not knowledge_source.is_absolute():
        knowledge_source = config.project_root / knowledge_source

    documents = load_documents_from_path(knowledge_source)
    chunks = split_documents(documents, config)
    embedding_model = create_embedding_model(config)
    vectorstore = MilvusVectorStore(config=config, embedding_model=embedding_model)
    upserted = vectorstore.upsert_documents(chunks, rebuild=rebuild)

    return {
        "knowledge_file": str(knowledge_source),
        "knowledge_source": str(knowledge_source),
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
        description="Split knowledge documents and write them into Milvus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection before writing.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Knowledge file or directory to ingest. Defaults to the parsed document "
            "directory when omitted."
        ),
    )
    args = parser.parse_args()

    result = ingest(rebuild=args.rebuild, source=args.source)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


def _default_ingest_source(config: RagConfig) -> Path:
    return config.mineru_output_dir


if __name__ == "__main__":
    raise SystemExit(main())
