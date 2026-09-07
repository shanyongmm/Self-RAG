from __future__ import annotations

import argparse
from pathlib import Path

from rag.mineru_service import MinerUDocumentParser, collect_parse_sources

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_documents(
    sources: list[str | Path] | None = None,
    output_dir: str | Path | None = None,
    *,
    model: str | None = None,
    ocr: bool | None = None,
    language: str | None = None,
    extra_formats: list[str] | None = None,
    timeout: int = 1800,
    with_images: bool = True,
    retry_attempts: int = 3,
    retry_wait: float = 60.0,
    request_interval: float = 3.0,
) -> list[Path]:
    parser = MinerUDocumentParser()
    source_paths = sources or _default_parse_sources()
    output_root = output_dir or parser.output_dir
    file_paths = collect_parse_sources(source_paths, exclude_dirs=[output_root])
    parsed = parser.parse_files(
        file_paths,
        output_dir=output_dir,
        model=model,
        ocr=ocr,
        language=language,
        extra_formats=extra_formats or [],
        timeout=timeout,
        with_images=with_images,
        retry_attempts=retry_attempts,
        retry_wait=retry_wait,
        request_interval=request_interval,
    )
    return [document.markdown_path for document in parsed]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse PDF, image, Office, or HTML documents with MinerU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help=(
            "Files or directories to parse. Directories are scanned recursively. "
            "Defaults to the project data directory when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for parsed markdown files. Defaults to MINERU_OUTPUT_DIR.",
    )
    parser.add_argument("--model", default=None, help="MinerU model, e.g. vlm/pipeline/html.")
    parser.add_argument("--language", default=None, help="Document language hint for MinerU.")
    parser.add_argument("--ocr", dest="ocr", action="store_true", default=None)
    parser.add_argument("--no-ocr", dest="ocr", action="store_false")
    parser.add_argument(
        "--extra-format",
        action="append",
        choices=["docx", "html", "latex"],
        default=[],
        help="Also save an extra format. Can be used multiple times.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--no-images", dest="with_images", action="store_false")
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Total attempts for one file when MinerU returns HTTP 429.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=60.0,
        help="Base seconds to wait before retrying after HTTP 429.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=3.0,
        help="Seconds to wait between two MinerU parse requests.",
    )
    args = parser.parse_args()

    markdown_paths = parse_documents(
        args.sources or None,
        output_dir=args.output_dir,
        model=args.model,
        ocr=args.ocr,
        language=args.language,
        extra_formats=args.extra_format,
        timeout=args.timeout,
        with_images=args.with_images,
        retry_attempts=args.retry_attempts,
        retry_wait=args.retry_wait,
        request_interval=args.request_interval,
    )

    print(f"parsed_documents: {len(markdown_paths)}")
    for path in markdown_paths:
        print(f"markdown: {path}")
    return 0


def _default_parse_sources() -> list[Path]:
    return [PROJECT_ROOT / "data"]


if __name__ == "__main__":
    raise SystemExit(main())
