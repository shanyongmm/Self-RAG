from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TypeVar

from langgraph.checkpoint.postgres import PostgresSaver

from app.config import RagConfig, get_config
from rag.vectorstore import MilvusVectorStore
from scripts.ingest import ingest

T = TypeVar("T")


def setup_checkpoint(config: RagConfig) -> None:
    if not config.postgres_uri:
        raise RuntimeError("Missing required environment variable: POSTGRES_URI")

    with PostgresSaver.from_conn_string(config.postgres_uri) as checkpointer:
        checkpointer.setup()


def ensure_knowledge_base(config: RagConfig) -> None:
    vectorstore = MilvusVectorStore(config=config)
    if vectorstore.has_collection():
        print(
            f"Milvus collection '{config.collection_name}' already exists; "
            "skip initial ingestion."
        )
        return

    print("Milvus collection is missing; ingesting the knowledge base.")
    result = ingest(rebuild=False)
    print(
        f"Ingested {result['chunks']} chunks into "
        f"collection '{result['collection_name']}'."
    )


def retry(operation: Callable[[], T], name: str, attempts: int = 30) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[bootstrap] waiting for {name} ({attempt}/{attempts}): {exc}")
            time.sleep(2)

    raise RuntimeError(f"Unable to initialize {name}") from last_error


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    config = get_config()
    retry(lambda: setup_checkpoint(config), "PostgreSQL checkpoint")

    if _env_bool("AUTO_INGEST", default=True):
        retry(lambda: ensure_knowledge_base(config), "Milvus knowledge base")
    else:
        print("[bootstrap] AUTO_INGEST is disabled; skip knowledge-base ingestion.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
