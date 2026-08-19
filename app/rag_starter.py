from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from langchain_core.messages import HumanMessage

from app.checkpoint import postgres_checkpointer_from_uri
from app.config import RagConfig, get_config
from app.graph import build_graph
from app.schemas import RagGeneration, RetrievedChunk
from rag.llm import create_chat_model
from rag.vectorstore import MilvusVectorStore


class RagStarter:
    def __init__(self, config: RagConfig | None = None) -> None:
        self.config = config or get_config()
        self._checkpointer_context: AbstractContextManager[Any] | None = None
        self.checkpointer = None
        try:
            self.checkpointer = self._create_checkpointer()
            self.vectorstore = MilvusVectorStore(config=self.config)
            self.llm = create_chat_model(self.config)
            self.graph = build_graph(
                config=self.config,
                vectorstore=self.vectorstore,
                llm=self.llm,
                checkpointer=self.checkpointer,
            )
        except Exception:
            self.close()
            raise

    def ask(self, question: str, thread_id: str | None = None) -> dict[str, Any]:
        if not self.vectorstore.has_collection():
            raise RuntimeError(
                "Milvus collection does not exist. Run `python ingest.py` first."
            )

        resolved_thread_id = _resolve_thread_id(
            thread_id,
            self.config.default_thread_id,
        )
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "question": question,
                "retrieval_query": question,
                "trace": [],
                "retry_count": 0,
                "max_retries": self.config.max_retries,
                "top_k": self.config.top_k,
            },
            config={"configurable": {"thread_id": resolved_thread_id}},
        )

        answer = result.get("answer")
        documents = result.get("document", [])
        retrieved_documents = result.get("retrieved_document", [])
        rejected_documents = result.get("rejected_document", [])
        return {
            "mode": "self_rag",
            "question": question,
            "thread_id": resolved_thread_id,
            "answer": result.get("generation", ""),
            "is_answerable": _answer_attr(answer, "is_answerable"),
            "citations": _answer_attr(answer, "citations", []),
            "missing_info": _answer_attr(answer, "missing_info"),
            "sources": [_source_dict(chunk) for chunk in documents],
            "retrieved_sources": [
                _source_dict(chunk) for chunk in retrieved_documents
            ],
            "rejected_sources": [
                _source_dict(chunk) for chunk in rejected_documents
            ],
            "grade": _model_dump(result.get("grade")),
            "trace": result.get("trace", []),
            "retrieval_query": result.get("retrieval_query", question),
            "retry_count": result.get("retry_count", 0),
            "is_relevant": result.get("is_relevant"),
            "checkpoint_enabled": self.checkpointer is not None,
            "memory_message_count": len(result.get("messages", [])),
        }

    def _create_checkpointer(self) -> Any | None:
        if not self.config.postgres_uri:
            raise RuntimeError(
                "POSTGRES_URI is required because PostgreSQL checkpoint "
                "persistence is enabled for short-term memory."
            )

        self._checkpointer_context = postgres_checkpointer_from_uri(
            self.config.postgres_uri
        )
        checkpointer = self._checkpointer_context.__enter__()
        checkpointer.setup()
        return checkpointer

    def close(self) -> None:
        if self._checkpointer_context is not None:
            self._checkpointer_context.__exit__(None, None, None)
            self._checkpointer_context = None


def _answer_attr(
    answer: RagGeneration | dict[str, Any] | None,
    name: str,
    default: Any = None,
) -> Any:
    if answer is None:
        return default
    if isinstance(answer, dict):
        return answer.get(name, default)
    return getattr(answer, name, default)


def _resolve_thread_id(thread_id: str | None, default_thread_id: str) -> str:
    resolved = (thread_id or default_thread_id).strip()
    return resolved or default_thread_id


def _model_dump(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    if isinstance(model, dict):
        return model
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return None


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
