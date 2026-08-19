from evaluation.metrics import score_eval_result


def test_score_eval_result_tracks_filtered_context_and_citations() -> None:
    case = {
        "id": "qa_001",
        "question": "What is the rule?",
        "reference_answer": "The rule is A.",
        "expected_chunk_ids": ["chunk-a"],
        "answer_keywords": ["rule", "A"],
    }
    result = {
        "mode": "self_rag",
        "answer": "The rule is A.",
        "retrieved_sources": [
            {"chunk_id": "chunk-a"},
            {"chunk_id": "chunk-b"},
        ],
        "sources": [{"chunk_id": "chunk-a"}],
        "citations": ["chunk-a"],
        "retry_count": 1,
    }

    scored = score_eval_result(case, result, latency_seconds=0.1)

    assert scored["raw_precision_at_k"] == 0.5
    assert scored["final_context_precision"] == 1.0
    assert scored["invalid_retrieval_rate"] == 0.0
    assert scored["answer_correct"] is True
    assert scored["citation_accuracy"] == 1.0
    assert scored["citation_recall"] == 1.0
