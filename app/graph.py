from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.constants import START
from langgraph.graph import END, MessagesState, StateGraph

from app.config import RagConfig, get_config
from app.schemas import QueryRewrite, RagGeneration, RelevanceGrade, RetrievedChunk
from rag.llm import create_chat_model
from rag.prompts import (
    build_generate_messages,
    build_grade_messages,
    build_rewrite_messages,
)
from rag.vectorstore import MilvusVectorStore


class OverAllState(MessagesState):
    question: str
    retrieval_query: str
    retrieved_document: list[RetrievedChunk]
    document: list[RetrievedChunk]
    rejected_document: list[RetrievedChunk]
    trace: list[dict[str, Any]]
    next_action: str
    decision_reason: str
    generation: str
    retry_count: int
    max_retries: int
    top_k: int
    is_relevant: bool
    grade: RelevanceGrade
    rewrite: QueryRewrite
    answer: RagGeneration


def build_graph(
    config: RagConfig | None = None,
    vectorstore: MilvusVectorStore | None = None,
    llm: Any | None = None,
    checkpointer: Any | None = None,
):
    config = config or get_config()
    vectorstore = vectorstore or MilvusVectorStore(config)
    llm = llm or create_chat_model(config)

    grade_llm = llm.with_structured_output(RelevanceGrade)
    rewrite_llm = llm.with_structured_output(QueryRewrite)
    generate_llm = llm.with_structured_output(RagGeneration)

    def retrieve_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("question", "").strip()
        retrieval_query = (state.get("retrieval_query") or question).strip()
        top_k = state.get("top_k") or config.top_k

        documents = vectorstore.search(retrieval_query, top_k=top_k)
        return {
            "question": question,
            "retrieval_query": retrieval_query,
            "retrieved_document": documents,
            "document": documents,
            "rejected_document": [],
            "trace": _append_trace(
                state,
                {
                    "step": "retrieve",
                    "iteration": state.get("retry_count", 0),
                    "query": retrieval_query,
                    "top_k": top_k,
                    "retrieved_sources": [
                        _source_dict(chunk) for chunk in documents
                    ],
                },
            ),
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", config.max_retries),
            "top_k": top_k,
        }

    def grade_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("question", "")
        retrieval_query = state.get("retrieval_query") or question
        documents = state.get("retrieved_document") or state.get("document", [])

        grade = grade_llm.invoke(
            build_grade_messages(
                question=question,
                retrieval_query=retrieval_query,
                chunks=documents,
            )
        )
        relevant_documents, rejected_documents = _filter_relevant_documents(
            documents=documents,
            grade=grade,
            threshold=config.relevance_threshold,
        )
        is_relevant = bool(relevant_documents)
        relevant_ids = [str(chunk.chunk_id) for chunk in relevant_documents]
        print(
            "[grade] "
            f"is_relevant={is_relevant} relevant_chunk_ids={relevant_ids} "
            f"reason={grade.reason}"
        )
        return {
            "grade": grade,
            "is_relevant": is_relevant,
            "document": relevant_documents,
            "rejected_document": rejected_documents,
            "trace": _append_trace(
                state,
                {
                    "step": "grade",
                    "iteration": state.get("retry_count", 0),
                    "is_relevant": is_relevant,
                    "confidence": grade.confidence,
                    "reason": grade.reason,
                    "supporting_chunk_ids": grade.supporting_chunk_ids,
                    "chunk_grades": [
                        _model_dump(chunk_grade)
                        for chunk_grade in grade.chunk_grades
                    ],
                    "accepted_sources": [
                        _source_dict(chunk) for chunk in relevant_documents
                    ],
                    "rejected_sources": [
                        _source_dict(chunk) for chunk in rejected_documents
                    ],
                },
            ),
        }

    def decide_node(state: OverAllState) -> dict[str, Any]:
        next_action, reason = _decide_next_step(state, config)
        print(f"[route] {reason}")
        return {
            "next_action": next_action,
            "decision_reason": reason,
            "trace": _append_trace(
                state,
                {
                    "step": "decide",
                    "iteration": state.get("retry_count", 0),
                    "next_action": next_action,
                    "reason": reason,
                },
            ),
        }

    def route_after_decide(
        state: OverAllState,
    ) -> Literal["generate_node", "rewrite_node"]:
        if state.get("next_action") == "rewrite_node":
            return "rewrite_node"
        return "generate_node"

    def rewrite_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("question", "")
        retrieval_query = state.get("retrieval_query") or question
        retry_count = state.get("retry_count", 0) + 1
        documents = state.get("retrieved_document") or state.get("document", [])

        rewrite = rewrite_llm.invoke(
            build_rewrite_messages(
                question=question,
                retrieval_query=retrieval_query,
                chunks=documents,
            )
        )
        rewritten_query = rewrite.rewritten_query.strip() or retrieval_query
        print(f"[rewrite] retry={retry_count} query={rewritten_query}")
        return {
            "retrieval_query": rewritten_query,
            "retry_count": retry_count,
            "rewrite": rewrite,
            "trace": _append_trace(
                state,
                {
                    "step": "rewrite",
                    "iteration": retry_count,
                    "previous_query": retrieval_query,
                    "rewritten_query": rewritten_query,
                    "reason": rewrite.reason,
                    "previous_retrieved_sources": [
                        _source_dict(chunk) for chunk in documents
                    ],
                },
            ),
        }

    def generate_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("question", "")
        documents = state.get("document", [])

        answer = generate_llm.invoke(
            build_generate_messages(
                question=question,
                chunks=documents,
            )
        )
        return {
            "answer": answer,
            "generation": answer.answer,
            "messages": [AIMessage(content=answer.answer)],
            "trace": _append_trace(
                state,
                {
                    "step": "generate",
                    "iteration": state.get("retry_count", 0),
                    "is_answerable": answer.is_answerable,
                    "citations": answer.citations,
                    "missing_info": answer.missing_info,
                    "final_sources": [_source_dict(chunk) for chunk in documents],
                },
            ),
        }

    builder = StateGraph(OverAllState)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("grade_node", grade_node)
    builder.add_node("decide_node", decide_node)
    builder.add_node("rewrite_node", rewrite_node)
    builder.add_node("generate_node", generate_node)

    builder.add_edge(START, "retrieve_node")
    builder.add_edge("retrieve_node", "grade_node")
    builder.add_edge("grade_node", "decide_node")
    builder.add_conditional_edges(
        "decide_node",
        route_after_decide,
        {
            "rewrite_node": "rewrite_node",
            "generate_node": "generate_node",
        },
    )
    builder.add_edge("rewrite_node", "retrieve_node")
    builder.add_edge("generate_node", END)

    return builder.compile(checkpointer=checkpointer)


def _decide_next_step(
    state: OverAllState,
    config: RagConfig,
) -> tuple[Literal["generate_node", "rewrite_node"], str]:
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", config.max_retries)

    if is_relevant:
        return "generate_node", "documents are relevant; generating answer"
    if retry_count >= max_retries:
        return (
            "generate_node",
            "max retries reached; generating with filtered documents",
        )

    return "rewrite_node", "documents are not relevant; rewriting query"


def _append_trace(
    state: OverAllState,
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    return [*state.get("trace", []), event]


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


def _model_dump(model: Any) -> dict[str, Any]:
    if isinstance(model, dict):
        return model
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return {"value": model}


def _filter_relevant_documents(
    documents: list[RetrievedChunk],
    grade: RelevanceGrade,
    threshold: float,
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    relevant_ids = {
        str(item.chunk_id)
        for item in grade.chunk_grades
        if item.is_relevant and item.confidence >= threshold
    }
    if not relevant_ids and grade.is_relevant and grade.confidence >= threshold:
        relevant_ids = {str(chunk_id) for chunk_id in grade.supporting_chunk_ids}

    relevant_documents = [
        chunk for chunk in documents if _chunk_id(chunk) in relevant_ids
    ]
    rejected_documents = [
        chunk for chunk in documents if _chunk_id(chunk) not in relevant_ids
    ]
    return relevant_documents, rejected_documents


def _chunk_id(chunk: RetrievedChunk) -> str:
    if chunk.chunk_id is not None:
        return str(chunk.chunk_id)
    return f"rank:{chunk.rank}"
