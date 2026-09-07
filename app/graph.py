from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.constants import START
from langgraph.graph import END, MessagesState, StateGraph

from app.config import RagConfig, get_config
from app.memory import (
    build_history_window,
    compact_memory_for_trace,
    prepare_memory_context,
    update_memory_context,
)
from app.schemas import QueryRewrite, RagGeneration, RelevanceGrade, RetrievedChunk
from rag.llm import create_chat_model
from rag.prompts import (
    build_generate_messages,
    build_grade_messages,
    build_rewrite_messages,
)
from rag.vectorstore import MilvusVectorStore
from rag.mem0_service import get_mem0_service
from rag.web_search import is_configured as _web_search_is_configured
from rag.web_search import web_search as _web_search


class OverAllState(MessagesState):
    question: str
    retrieval_query: str
    contextual_question: str
    partition_id: str | None
    memory_context: dict[str, Any]
    history_window: str
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
    user_id: str | None
    memory_reference: str
    long_term_hits: list[dict[str,Any]]
    answer_source: str
    evidence_origin: str
    web_documents: list[RetrievedChunk]



def build_graph(
    config: RagConfig | None = None,
    vectorstore: MilvusVectorStore | None = None,
    llm: Any | None = None,
    checkpointer: Any | None = None,
):
    config = config or get_config()
    vectorstore = vectorstore or MilvusVectorStore(config)
    llm = llm or create_chat_model(config)

    grade_llm = llm.with_structured_output(
        RelevanceGrade, method="function_calling"
    )
    rewrite_llm = llm.with_structured_output(QueryRewrite, method="function_calling")
    generate_llm = llm.with_structured_output(
        RagGeneration, method="function_calling"
    )

    def memory_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("question", "").strip()
        history_window = build_history_window(state.get("messages", []))
        prepared = prepare_memory_context(
            question=question,
            memory_context=state.get("memory_context"),
        )
        memory_reference,long_term_hits = _load_memory_reference(question,state)
        history_turns = sum(
            1 for line in history_window.splitlines() if line.startswith("用户：")
        )
        return {
            "contextual_question": prepared["contextual_question"],
            "retrieval_query": prepared["retrieval_query"],
            "partition_id": prepared["partition_id"],
            "memory_context": prepared["memory_context"],
            "history_window": history_window,
            "memory_reference": memory_reference,
            "long_term_hits": long_term_hits,
            "trace": _append_trace(
                state,
                {
                    "step": "memory_prepare",
                    "is_follow_up": prepared["is_follow_up"],
                    "partition_id": prepared["partition_id"],
                    "entities": prepared["entities"],
                    "contextual_question": prepared["contextual_question"],
                    "history_window_chars": len(history_window),
                    "history_window_turns": history_turns,
                    "memory": compact_memory_for_trace(
                        prepared["memory_context"]
                    ),
                    "long_term_hits": [h["memory"] for h in long_term_hits],
                },
            ),
        }

    def retrieve_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("question", "").strip()
        retrieval_query = (state.get("retrieval_query") or question).strip()
        top_k = state.get("top_k") or config.top_k
        retry_count = state.get("retry_count", 0)
        # 首次检索收窄到路由分区；rewrite 重试放开到全分区，
        # 避免跨域问题（如高血压+用药）困死在单分区。
        partition_ids = (
            [state["partition_id"]]
            if (state.get("partition_id") and retry_count == 0)
            else None
        )

        documents = vectorstore.search(
            retrieval_query,
            top_k=top_k,
            partition_ids=partition_ids,
        )
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
                    "iteration": retry_count,
                    "query": retrieval_query,
                    "top_k": top_k,
                    "partition_ids": partition_ids,
                    "retrieved_sources": [
                        _source_dict(chunk) for chunk in documents
                    ],
                },
            ),
            "retry_count": retry_count,
            "max_retries": state.get("max_retries", config.max_retries),
            "top_k": top_k,
        }

    def grade_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("contextual_question") or state.get("question", "")
        retrieval_query = state.get("retrieval_query") or question
        documents = state.get("retrieved_document") or state.get("document", [])

        grade = grade_llm.invoke(
            build_grade_messages(
                question=question,
                retrieval_query=retrieval_query,
                chunks=documents,
                history_window=state.get("history_window", ""),
            )
        )
        relevant_documents, rejected_documents = _filter_relevant_documents(
            documents=documents,
            grade=grade,
            threshold=config.relevance_threshold,
        )
        is_relevant = bool(relevant_documents)
        relevant_ids = [str(chunk.chunk_id) for chunk in relevant_documents]
        rerank_scores = _relevance_scores(grade)
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
                    "reranked_chunk_ids": relevant_ids,
                    "rerank_scores": {
                        _chunk_id(chunk): rerank_scores.get(_chunk_id(chunk), 0.0)
                        for chunk in relevant_documents
                    },
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
    ) -> Literal["generate_node", "rewrite_node", "web_search_node"]:
        next_action = state.get("next_action")
        if next_action == "rewrite_node":
            return "rewrite_node"
        if next_action == "web_node":
            return "web_search_node"
        return "generate_node"

    def route_after_web_search(
        state: OverAllState,
    ) -> Literal["generate_node", "coverage_node"]:
        if state.get("web_documents"):
            return "generate_node"
        return "coverage_node"

    def rewrite_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("contextual_question") or state.get("question", "")
        retrieval_query = state.get("retrieval_query") or question
        retry_count = state.get("retry_count", 0) + 1
        documents = state.get("retrieved_document") or state.get("document", [])

        rewrite = rewrite_llm.invoke(
            build_rewrite_messages(
                question=question,
                retrieval_query=retrieval_query,
                chunks=documents,
                history_window=state.get("history_window", ""),
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

    def web_search_node(state: OverAllState) -> dict[str, Any]:
        question = state.get("contextual_question") or state.get("question", "")
        retrieval_query = (state.get("retrieval_query") or question).strip()
        top_k = state.get("top_k") or config.top_k
        retry_count = state.get("retry_count", 0)
        docs = _web_search(config, retrieval_query, top_k=top_k)
        print(
            f"[web_search] query={retrieval_query} hits={len(docs)} "
            f"urls={[chunk.source for chunk in docs][:3]}"
        )
        return {
            "question": question,
            "retrieval_query": retrieval_query,
            "retry_count": retry_count,
            "top_k": top_k,
            "web_documents": docs,
            "document": docs,
            "retrieved_document": docs,
            "rejected_document": [],
            "evidence_origin": "web" if docs else "local",
            "trace": _append_trace(
                state,
                {
                    "step": "web_search",
                    "iteration": retry_count,
                    "query": retrieval_query,
                    "top_k": top_k,
                    "configured": _web_search_is_configured(config),
                    "retrieved_sources": [_source_dict(chunk) for chunk in docs],
                },
            ),
        }

    def generate_node(state: OverAllState) -> dict[str, Any]:
        original_question = state.get("question", "")
        question = state.get("contextual_question") or original_question
        documents = state.get("document", [])
        evidence_origin = state.get("evidence_origin") or "local"
        answer_source = "web" if evidence_origin == "web" else "local"

        answer = generate_llm.invoke(
            build_generate_messages(
                question=question,
                chunks=documents,
                history_window=state.get("history_window", ""),
                memory_reference=state.get("memory_reference", ""),
                evidence_origin=evidence_origin,
            )
        )
        memory_context = update_memory_context(
            state.get("memory_context"),
            question=original_question,
            contextual_question=question,
            partition_id=state.get("partition_id"),
            documents=documents,
            rejected_documents=state.get("rejected_document", []),
            answer_text=answer.answer,
        )
        return {
            "answer": answer,
            "generation": answer.answer,
            "answer_source": answer_source,
            "evidence_origin": evidence_origin,
            "messages": [AIMessage(content=answer.answer)],
            "memory_context": memory_context,
            "trace": _append_trace(
                state,
                {
                    "step": "generate",
                    "iteration": state.get("retry_count", 0),
                    "evidence_origin": evidence_origin,
                    "is_answerable": answer.is_answerable,
                    "citations": answer.citations,
                    "missing_info": answer.missing_info,
                    "final_sources": [_source_dict(chunk) for chunk in documents],
                    "memory": compact_memory_for_trace(memory_context),
                },
            ),
        }

    def coverage_node(state: OverAllState) -> dict[str, Any]:
        answer = RagGeneration(
            answer=(
                "抱歉，本地知识库与联网检索都未找到能支撑回答该问题的资料，"
                "无法可靠作答。请补充更具体的描述或换一种问法再试；"
                "若是专业医疗问题，请以医生意见或正规药品/器械说明书为准。"
            ),
            is_answerable=False,
            citations=[],
            missing_info="本地知识库与联网公开来源均未检索到可支撑回答该问题的资料。",
        )
        return {
            "answer": answer,
            "generation": answer.answer,
            "answer_source": "uncovered",
            "evidence_origin": state.get("evidence_origin") or "local",
            "document": [],
            "web_documents": state.get("web_documents", []),
            "messages": [AIMessage(content=answer.answer)],
            "trace": _append_trace(
                state,
                {
                    "step": "coverage",
                    "iteration": state.get("retry_count", 0),
                    "is_answerable": False,
                    "missing_info": answer.missing_info,
                    "final_sources": [],
                },
            ),
        }

    builder = StateGraph(OverAllState)
    builder.add_node("memory_node", memory_node)
    builder.add_node("retrieve_node", retrieve_node)
    builder.add_node("grade_node", grade_node)
    builder.add_node("decide_node", decide_node)
    builder.add_node("rewrite_node", rewrite_node)
    builder.add_node("web_search_node", web_search_node)
    builder.add_node("generate_node", generate_node)
    builder.add_node("coverage_node", coverage_node)

    builder.add_edge(START, "memory_node")
    builder.add_edge("memory_node", "retrieve_node")
    builder.add_edge("retrieve_node", "grade_node")
    builder.add_edge("grade_node", "decide_node")
    builder.add_conditional_edges(
        "decide_node",
        route_after_decide,
        {
            "rewrite_node": "rewrite_node",
            "generate_node": "generate_node",
            "web_search_node": "web_search_node",
        },
    )
    builder.add_edge("rewrite_node", "retrieve_node")
    builder.add_conditional_edges(
        "web_search_node",
        route_after_web_search,
        {
            "generate_node": "generate_node",
            "coverage_node": "coverage_node",
        },
    )
    builder.add_edge("generate_node", END)
    builder.add_edge("coverage_node", END)

    return builder.compile(checkpointer=checkpointer)


def _decide_next_step(
    state: OverAllState,
    config: RagConfig,
) -> tuple[Literal["generate_node", "rewrite_node", "web_node"], str]:
    is_relevant = state.get("is_relevant", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", config.max_retries)

    if is_relevant:
        return "generate_node", "documents are relevant; generating answer"
    if retry_count >= max_retries:
        # 本地重写耗尽仍无可支撑片段：降级联网检索一次（不进 rewrite 循环），
        # 联网也拿不到支撑片段则由 coverage_node 明确“知识库未覆盖”收尾。
        return (
            "web_node",
            "max retries reached; local documents irrelevant; degrading to web search once",
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
    relevance_scores = _relevance_scores(grade)
    relevant_ids = {
        chunk_id
        for chunk_id, score in relevance_scores.items()
        if score >= threshold
    }
    if not relevant_ids and grade.is_relevant and grade.confidence >= threshold:
        relevant_ids = {str(chunk_id) for chunk_id in grade.supporting_chunk_ids}

    relevant_documents = [
        chunk for chunk in documents if _chunk_id(chunk) in relevant_ids
    ]
    original_ranks = {
        _chunk_id(chunk): index
        for index, chunk in enumerate(documents)
    }
    relevant_documents.sort(
        key=lambda chunk: (
            -relevance_scores.get(_chunk_id(chunk), grade.confidence),
            original_ranks.get(_chunk_id(chunk), len(documents)),
        )
    )
    rejected_documents = [
        chunk for chunk in documents if _chunk_id(chunk) not in relevant_ids
    ]
    return relevant_documents, rejected_documents


def _chunk_id(chunk: RetrievedChunk) -> str:
    if chunk.chunk_id is not None:
        return str(chunk.chunk_id)
    return f"rank:{chunk.rank}"


def _relevance_scores(grade: RelevanceGrade) -> dict[str, float]:
    return {
        str(item.chunk_id): item.confidence
        for item in grade.chunk_grades
        if item.is_relevant
    }

def _load_memory_reference(question: str, state: OverAllState):
    service = get_mem0_service()
    hits = service.search(state.get("user_id"), question)
    if not hits:
        return "", []
    text = "\n".join(f"- {h['memory']}" for h in hits)
    reference = (
        "与该用户此前在其他会话聊过的相关内容(仅供参考,可能存在错误,"
        "一律以本次提供的知识库片段为准):\n" + text
    )
    return reference, hits

