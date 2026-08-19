from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.documents import Document
from pymilvus import MilvusClient

from app.config import RagConfig, get_config
from app.schemas import RetrievedChunk
from rag.embeddings import create_embedding_model


class MilvusVectorStore:
    def __init__(
        self,
        config: RagConfig | None = None,
        embedding_model: Any | None = None,
        client: MilvusClient | None = None,
    ) -> None:
        self.config = config or get_config()
        self.embedding_model = embedding_model or create_embedding_model(self.config)
        self.client = client or MilvusClient(self.config.milvus_url)
        self._use_database()

    def _use_database(self) -> None:
        if not self.config.db_name:
            return

        databases = self.client.list_databases()
        if self.config.db_name not in databases:
            self.client.create_database(db_name=self.config.db_name)
        self.client.use_database(db_name=self.config.db_name)

    def has_collection(self) -> bool:
        return self.client.has_collection(collection_name=self.config.collection_name)

    def ensure_collection(self, rebuild: bool = False) -> None:
        exists = self.has_collection()
        if exists and rebuild:
            self.client.drop_collection(collection_name=self.config.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                dimension=self.config.embed_dimension,
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
            )

    def upsert_documents(
        self,
        documents: Sequence[Document],
        rebuild: bool = False,
    ) -> int:
        chunks = [doc for doc in documents if doc.page_content.strip()]
        if not chunks:
            return 0

        self.ensure_collection(rebuild=rebuild)

        texts = [chunk.page_content for chunk in chunks]
        vectors = self._embed_documents(texts)

        rows = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            chunk_id = chunk.metadata.get("chunk_id", index)
            source = chunk.metadata.get("source") or str(self.config.knowledge_file)
            rows.append(
                {
                    "id": index,
                    "vector": vector,
                    "text": chunk.page_content,
                    "chunk_id": str(chunk_id),
                    "source": str(source),
                }
            )

        self.client.upsert(collection_name=self.config.collection_name, data=rows)
        self.client.flush(collection_name=self.config.collection_name)
        return len(rows)

    def _embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(texts, self.config.embed_batch_size):
            vectors.extend(self.embedding_model.embed_documents(batch))
        return vectors

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        limit = top_k or self.config.top_k
        query_vector = self.embedding_model.embed_query(query)
        result_sets = self.client.search(
            collection_name=self.config.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text", "chunk_id", "source"],
        )
        hits = result_sets[0] if result_sets else []
        return [
            RetrievedChunk.from_milvus_hit(hit, rank=rank)
            for rank, hit in enumerate(hits, start=1)
            if (hit.get("entity") or {}).get("text") or hit.get("text")
        ]


def _batched(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])
