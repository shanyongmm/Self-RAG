from __future__ import annotations

from langchain_community.embeddings import DashScopeEmbeddings

from app.config import RagConfig, get_config


def create_embedding_model(config: RagConfig | None = None) -> DashScopeEmbeddings:
    config = config or get_config()
    return DashScopeEmbeddings(
        model=config.embed_model_name,
        dashscope_api_key=config.embed_api_key,
    )
