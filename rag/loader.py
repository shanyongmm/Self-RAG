from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.config import RagConfig, get_config

SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".markdown"}


def load_documents_from_path(path: str | Path) -> list[Document]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Knowledge source not found: {source_path}")

    file_paths = list(_iter_knowledge_files(source_path))
    if not file_paths:
        supported = ", ".join(sorted(SUPPORTED_KNOWLEDGE_SUFFIXES))
        raise ValueError(
            f"No supported knowledge documents found in {source_path}. "
            f"Supported suffixes: {supported}."
        )

    documents: list[Document] = []
    for file_path in file_paths:
        loader = TextLoader(file_path=str(file_path), encoding="utf-8")
        loaded = loader.load()
        for document in loaded:
            document.metadata = {
                **document.metadata,
                "source": str(file_path),
            }
        documents.extend(loaded)

    return documents


def load_knowledge_documents(config: RagConfig | None = None) -> list[Document]:
    config = config or get_config()
    return load_documents_from_path(config.knowledge_file)


def _iter_knowledge_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_KNOWLEDGE_SUFFIXES:
            return
        yield path
        return

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_KNOWLEDGE_SUFFIXES:
            yield file_path
