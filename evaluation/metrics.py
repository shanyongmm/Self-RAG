from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def score_eval_result(
    case: dict[str, Any],
    result: dict[str, Any],
    latency_seconds: float,
) -> dict[str, Any]:
    expected_ids = _normalize_ids(case.get("expected_chunk_ids", []))
    citations = _normalize_id_list(result.get("citations", []))
    final_source_ids = _source_ids(result.get("sources", []))
    raw_source_ids = _source_ids(
        result.get("retrieved_sources", result.get("sources", []))
    )
    answer = str(result.get("answer", ""))
    keywords = [str(item) for item in case.get("answer_keywords", [])]

    answer_keyword_recall = _keyword_recall(answer, keywords)
    final_context_precision = _precision(final_source_ids, expected_ids)
    raw_precision_at_k = _precision(raw_source_ids, expected_ids)

    return {
        "case_id": case.get("id"),
        "mode": result.get("mode", "self_rag"),
        "question": case.get("question"),
        "answer": answer,
        "reference_answer": case.get("reference_answer"),
        "expected_chunk_ids": sorted(expected_ids),
        "retrieved_chunk_ids": raw_source_ids,
        "final_chunk_ids": final_source_ids,
        "citations": citations,
        "trace": result.get("trace", []),
        "grade": result.get("grade"),
        "latency_seconds": latency_seconds,
        "raw_precision_at_k": raw_precision_at_k,
        "final_context_precision": final_context_precision,
        "invalid_retrieval_rate": _invalid_rate(final_source_ids, expected_ids),
        "answer_keyword_recall": answer_keyword_recall,
        "answer_correct": answer_keyword_recall == 1.0,
        "citation_accuracy": _citation_accuracy(citations, expected_ids),
        "citation_recall": _citation_recall(citations, expected_ids),
        "retry_count": result.get("retry_count", 0),
        "estimated_input_tokens": estimate_tokens(
            _input_text(
                question=str(case.get("question", "")),
                sources=result.get("sources", []),
            )
        ),
        "estimated_output_tokens": estimate_tokens(answer),
        "is_answerable": result.get("is_answerable"),
        "error": result.get("error"),
    }


def summarize_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_mode[str(record.get("mode", "unknown"))].append(record)

    summaries = {
        mode: _summarize_mode(mode_records)
        for mode, mode_records in sorted(by_mode.items())
    }

    return {
        "total_records": len(records),
        "modes": summaries,
        "comparison": _compare_modes(
            summaries.get("naive_rag"),
            summaries.get("self_rag"),
        ),
    }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    non_cjk_chars = len(text) - cjk_chars
    return cjk_chars + math.ceil(non_cjk_chars / 4)


def _summarize_mode(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_input_tokens = sum(
        int(record["estimated_input_tokens"]) for record in records
    )
    total_output_tokens = sum(
        int(record["estimated_output_tokens"]) for record in records
    )
    return {
        "case_count": len(records),
        "error_count": sum(1 for record in records if record.get("error")),
        "avg_raw_precision_at_k": _avg(records, "raw_precision_at_k"),
        "avg_final_context_precision": _avg(records, "final_context_precision"),
        "avg_invalid_retrieval_rate": _avg(records, "invalid_retrieval_rate"),
        "answer_accuracy": _avg_bool(records, "answer_correct"),
        "avg_answer_keyword_recall": _avg(records, "answer_keyword_recall"),
        "avg_citation_accuracy": _avg(records, "citation_accuracy"),
        "avg_citation_recall": _avg(records, "citation_recall"),
        "avg_latency_seconds": _avg(records, "latency_seconds"),
        "avg_retry_count": _avg(records, "retry_count"),
        "total_estimated_tokens": total_input_tokens + total_output_tokens,
        "avg_estimated_tokens": _safe_divide(
            total_input_tokens + total_output_tokens,
            len(records),
        ),
    }


def _compare_modes(
    naive: dict[str, Any] | None,
    self_rag: dict[str, Any] | None,
) -> dict[str, Any]:
    if not naive or not self_rag:
        return {}

    naive_invalid = naive.get("avg_invalid_retrieval_rate")
    self_invalid = self_rag.get("avg_invalid_retrieval_rate")
    return {
        "invalid_retrieval_rate_reduction_pct": _relative_reduction(
            naive_invalid,
            self_invalid,
        ),
        "raw_precision_at_k_delta": _delta(
            naive.get("avg_raw_precision_at_k"),
            self_rag.get("avg_raw_precision_at_k"),
        ),
        "final_context_precision_delta": _delta(
            naive.get("avg_final_context_precision"),
            self_rag.get("avg_final_context_precision"),
        ),
        "answer_accuracy_delta": _delta(
            naive.get("answer_accuracy"),
            self_rag.get("answer_accuracy"),
        ),
        "citation_accuracy_delta": _delta(
            naive.get("avg_citation_accuracy"),
            self_rag.get("avg_citation_accuracy"),
        ),
        "latency_delta_seconds": _delta(
            naive.get("avg_latency_seconds"),
            self_rag.get("avg_latency_seconds"),
        ),
        "estimated_token_delta": _delta(
            naive.get("avg_estimated_tokens"),
            self_rag.get("avg_estimated_tokens"),
        ),
    }


def _avg(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(record[key])
        for record in records
        if record.get(key) is not None and not record.get("error")
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _avg_bool(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        bool(record[key])
        for record in records
        if record.get(key) is not None and not record.get("error")
    ]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _relative_reduction(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (before - after) / before * 100


def _precision(retrieved_ids: list[str], expected_ids: set[str]) -> float | None:
    if not retrieved_ids or not expected_ids:
        return None
    hits = sum(1 for chunk_id in retrieved_ids if chunk_id in expected_ids)
    return hits / len(retrieved_ids)


def _invalid_rate(retrieved_ids: list[str], expected_ids: set[str]) -> float | None:
    precision = _precision(retrieved_ids, expected_ids)
    if precision is None:
        return None
    return 1 - precision


def _keyword_recall(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    hits = sum(1 for keyword in keywords if keyword in answer)
    return hits / len(keywords)


def _citation_accuracy(citations: list[str], expected_ids: set[str]) -> float | None:
    if not expected_ids:
        return None
    if not citations:
        return 0.0
    hits = sum(1 for citation in citations if citation in expected_ids)
    return hits / len(citations)


def _citation_recall(citations: list[str], expected_ids: set[str]) -> float | None:
    if not expected_ids:
        return None
    if not citations:
        return 0.0
    hits = sum(1 for citation in set(citations) if citation in expected_ids)
    return hits / len(expected_ids)


def _source_ids(sources: list[dict[str, Any]]) -> list[str]:
    return [
        str(source["chunk_id"])
        for source in sources
        if source.get("chunk_id") is not None
    ]


def _normalize_ids(values: list[Any]) -> set[str]:
    return {str(value) for value in values}


def _normalize_id_list(values: list[Any]) -> list[str]:
    return [str(value) for value in values]


def _input_text(question: str, sources: list[dict[str, Any]]) -> str:
    source_text = "\n".join(str(source.get("text", "")) for source in sources)
    return f"{question}\n{source_text}"


def _safe_divide(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
