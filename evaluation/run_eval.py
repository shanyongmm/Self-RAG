from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.naive_rag import NaiveRagBaseline
from app.rag_starter import RagStarter
from evaluation.metrics import score_eval_result, summarize_scores

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "qa_eval.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Self/Corrective RAG with a Naive RAG baseline."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["both", "naive", "self-rag"],
        default="both",
        help="Which pipeline to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override Top-K for Naive RAG. Self-RAG uses RAG_TOP_K from .env.",
    )
    args = parser.parse_args()

    cases = load_dataset(args.dataset, limit=args.limit)
    if not cases:
        raise RuntimeError(f"No eval cases loaded from {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    detail_path = args.output_dir / f"eval_details_{run_id}.jsonl"
    report_path = args.output_dir / f"eval_report_{run_id}.json"

    records = run_evaluation(
        cases=cases,
        mode=args.mode,
        top_k=args.top_k,
        run_id=run_id,
    )
    report = summarize_scores(records)
    report.update(
        {
            "run_id": run_id,
            "dataset": str(args.dataset),
            "case_count": len(cases),
            "mode": args.mode,
            "detail_path": str(detail_path),
        }
    )

    write_jsonl(detail_path, records)
    write_json(report_path, report)

    print(f"Evaluation details written to: {detail_path}")
    print(f"Evaluation report written to: {report_path}")
    print(json.dumps(report["modes"], ensure_ascii=False, indent=2))
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    return 0


def run_evaluation(
    cases: list[dict[str, Any]],
    mode: str,
    top_k: int | None,
    run_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    naive: NaiveRagBaseline | None = None
    self_rag: RagStarter | None = None

    try:
        if mode in {"both", "naive"}:
            naive = NaiveRagBaseline()
        if mode in {"both", "self-rag"}:
            self_rag = RagStarter()

        for case in cases:
            if naive is not None:
                records.append(
                    evaluate_case(
                        case=case,
                        runner=naive,
                        mode="naive_rag",
                        top_k=top_k,
                    )
                )
            if self_rag is not None:
                records.append(
                    evaluate_case(
                        case=case,
                        runner=self_rag,
                        mode="self_rag",
                        thread_id=f"eval-{run_id}-{case['id']}-{uuid4().hex}",
                    )
                )
    finally:
        if self_rag is not None:
            self_rag.close()

    return records


def evaluate_case(
    case: dict[str, Any],
    runner: NaiveRagBaseline | RagStarter,
    mode: str,
    top_k: int | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        if isinstance(runner, NaiveRagBaseline):
            result = runner.ask(question=case["question"], top_k=top_k)
        else:
            result = runner.ask(question=case["question"], thread_id=thread_id)
        result["mode"] = mode
    except Exception as exc:  # noqa: BLE001
        result = {
            "mode": mode,
            "question": case.get("question"),
            "answer": "",
            "sources": [],
            "retrieved_sources": [],
            "citations": [],
            "retry_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    latency_seconds = time.perf_counter() - started_at
    return score_eval_result(case, result, latency_seconds)


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            case = json.loads(text)
            validate_case(case, line_number)
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
    return cases


def validate_case(case: dict[str, Any], line_number: int) -> None:
    required_fields = {
        "id",
        "question",
        "reference_answer",
        "expected_chunk_ids",
        "answer_keywords",
    }
    missing = sorted(required_fields - set(case))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Dataset line {line_number} missing fields: {joined}")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
