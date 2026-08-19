from __future__ import annotations

from typing import Any

from app.config import RagConfig, get_config
from app.schemas import RagGeneration, RetrievedChunk
from rag.llm import create_chat_model
from rag.prompts import build_generate_messages
from rag.vectorstore import MilvusVectorStore


class NaiveRagBaseline:
    """Baseline RAG: retrieve Top-K chunks once, then generate directly."""

    def __init__(
        self,
        config: RagConfig | None = None,
        vectorstore: MilvusVectorStore | None = None,
        llm: Any | None = None,
    ) -> None:
        self.config = config or get_config()
        self.vectorstore = vectorstore or MilvusVectorStore(config=self.config)
        self.llm = llm or create_chat_model(self.config)
        self.generate_llm = self.llm.with_structured_output(RagGeneration)

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        if not self.vectorstore.has_collection():
            raise RuntimeError(
                "Milvus collection does not exist. Run `python ingest.py` first."
            )

        resolved_top_k = top_k or self.config.top_k
        documents = self.vectorstore.search(question, top_k=resolved_top_k)
        source_dicts = [_source_dict(chunk) for chunk in documents]
        answer = self.generate_llm.invoke(
            build_generate_messages(
                question=question,
                chunks=documents,
            )
        )

        return {
            "mode": "naive_rag",
            "question": question,
            "answer": answer.answer,
            "is_answerable": answer.is_answerable,
            "citations": answer.citations,
            "missing_info": answer.missing_info,
            "sources": source_dicts,
            "retrieved_sources": source_dicts,
            "rejected_sources": [],
            "retrieval_query": question,
            "retry_count": 0,
            "is_relevant": None,
            "trace": [
                {
                    "step": "retrieve",
                    "iteration": 0,
                    "query": question,
                    "top_k": resolved_top_k,
                    "retrieved_sources": source_dicts,
                },
                {
                    "step": "generate",
                    "iteration": 0,
                    "is_answerable": answer.is_answerable,
                    "citations": answer.citations,
                    "missing_info": answer.missing_info,
                    "final_sources": source_dicts,
                },
            ],
        }


def _source_dict(chunk: RetrievedChunk | dict[str, Any]) -> dict[str, Any]:
    if isinstance(chunk, RetrievedChunk):
        return chunk.to_source_dict()
    return {
        "rank": chunk.get("rank"),
        "chunk_id": chunk.get("chunk_id"),
        "source": chunk.get("source"),
        "score": chunk.get("score"),
        "text": chunk.get("text", ""),
    }
