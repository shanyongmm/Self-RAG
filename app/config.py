from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class RagConfig(BaseModel):
    project_root: Path
    knowledge_file: Path

    milvus_url: str = "http://localhost:19530"
    db_name: str | None = None
    collection_name: str = "docs"

    postgres_uri: str | None = Field(default=None, repr=False)
    default_thread_id: str = "default"

    embed_model_name: str
    embed_api_key: str = Field(repr=False)
    embed_dimension: int = Field(default=1024, ge=1)
    embed_batch_size: int = Field(default=20, ge=1, le=20)

    llm_model: str
    llm_api_key: str = Field(repr=False)
    llm_base_url: str | None = None
    llm_temperature: float = 0

    top_k: int = Field(default=3, ge=1, le=50)
    max_retries: int = Field(default=3, ge=0, le=10)
    relevance_threshold: float = Field(default=0.5, ge=0, le=1)

    chunk_size: int = Field(default=200, ge=50)
    chunk_overlap: int = Field(default=80, ge=0)


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _required_env(*names: str) -> str:
    value = _env(*names)
    if value is None:
        joined = " / ".join(names)
        raise RuntimeError(f"Missing required environment variable: {joined}")
    return value


def _int_env(*names: str, default: int) -> int:
    value = _env(*names)
    return default if value is None else int(value)


def _float_env(*names: str, default: float) -> float:
    value = _env(*names)
    return default if value is None else float(value)


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


@lru_cache(maxsize=1)
def get_config() -> RagConfig:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    knowledge_file = _resolve_path(
        project_root,
        _env(
            "KNOWLEDGE_FILE",
            default="data/raw/customer_service_knowledge_base.txt",
        ),
    )

    return RagConfig(
        project_root=project_root,
        knowledge_file=knowledge_file,
        milvus_url=_env("MILVUS_URL", "MILVUS_URI", default="http://localhost:19530"),
        db_name=_env("DB_NAME", "MILVUS_DB_NAME"),
        collection_name=_env(
            "COL_NAME", "MILVUS_COLLECTION", "COLLECTION_NAME", default="docs"
        ),
        postgres_uri=_env("POSTGRES_URI", "POSTGRES_DSN", "DATABASE_URL"),
        default_thread_id=_env("RAG_DEFAULT_THREAD_ID", default="default") or "default",
        embed_model_name=_required_env("EMBED_MODEL_NAME", "EMBEDDING_MODEL"),
        embed_api_key=_required_env("DASHSCOPE_API_KEY", "EMBED_API_KEY"),
        embed_dimension=_int_env("EMBED_DIMENSION", "EMBED_DIE", default=1024),
        embed_batch_size=_int_env(
            "EMBED_BATCH_SIZE", "DASHSCOPE_EMBED_BATCH_SIZE", default=20
        ),
        llm_model=_required_env("LLM_MODEL"),
        llm_api_key=_required_env("LLM_API_KEY", "OPENAI_API_KEY"),
        llm_base_url=_env("LLM_BASE_URL", "OPENAI_BASE_URL"),
        llm_temperature=_float_env("LLM_TEMPERATURE", default=0),
        top_k=_int_env("RAG_TOP_K", "TOP_K", default=3),
        max_retries=_int_env("RAG_MAX_RETRIES", "RAG_MAX_ITERATIONS", default=3),
        relevance_threshold=_float_env("RAG_RELEVANCE_THRESHOLD", default=0.5),
        chunk_size=_int_env("RAG_CHUNK_SIZE", "CHUNK_SIZE", default=200),
        chunk_overlap=_int_env("RAG_CHUNK_OVERLAP", "CHUNK_OVERLAP", default=80),
    )
