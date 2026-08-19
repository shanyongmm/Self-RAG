from __future__ import annotations

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.config import RagConfig, get_config


def load_knowledge_documents(config: RagConfig | None = None) -> list[Document]:
    config = config or get_config()
    if not config.knowledge_file.exists():
        raise FileNotFoundError(f"Knowledge file not found: {config.knowledge_file}")

    loader = TextLoader(file_path=str(config.knowledge_file), encoding="utf-8")
    return loader.load()
