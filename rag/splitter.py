from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import RagConfig, get_config


def split_documents(
    documents: Sequence[Document],
    config: RagConfig | None = None,
) -> list[Document]:
    config = config or get_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = splitter.split_documents(list(documents))

    for index, chunk in enumerate(chunks):
        chunk.metadata = {
            **chunk.metadata,
            "chunk_id": chunk.metadata.get("chunk_id", index),
        }

    return chunks
