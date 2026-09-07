from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from app.config import get_config
from rag.loader import load_documents_from_path
from rag.mineru_service import MinerUDocumentParser, collect_parse_sources
from rag.partitions import registry_from_config
from rag.splitter import split_documents
from rag.vectorstore import MilvusVectorStore

DEFAULT_SOURCE_DIR = "source_documents"
DEFAULT_OUTPUT_DIR = "data/parsed/mineru_partitions"

NOISE_SUFFIXES = {".zip", ".xml", ".html", ".htm"}


def ingest_partitioned(
    *,
    rebuild: bool = False,
    source: str | Path | None = None,
    parse: bool = True,
    output_dir: str | Path | None = None,
    include_html: bool = False,
    force_parse: bool = False,
) -> dict[str, Any]:
    config = get_config()
    registry = registry_from_config(config)
    if registry is None:
        raise RuntimeError(
            "Partition mode is disabled. Set PARTITION_CONFIG=knowledge_partitions.json "
            "in .env to enable it, or keep using `python -m scripts.ingest` for the "
            "flat (non-partitioned) pipeline."
        )

    source_root = _resolve(config.project_root, source or DEFAULT_SOURCE_DIR)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Partition source directory not found: {source_root}")

    parsed_root = _resolve(config.project_root, output_dir or DEFAULT_OUTPUT_DIR)
    parsed_root.mkdir(parents=True, exist_ok=True)

    parser = MinerUDocumentParser(config) if parse else None
    vectorstore = MilvusVectorStore(config)
    # 统一在此建库/建分区/重建，避免每个分区各自 rebuild 误删已写数据。
    vectorstore.ensure_collection(rebuild=rebuild)

    force_reparse = force_parse or rebuild
    partitions_summary: dict[str, dict[str, Any]] = {}
    total_upserted = 0

    for partition_id in registry.partition_ids:
        partition = registry.partition(partition_id)
        if partition is None:
            continue
        partition_dir = source_root / partition_id
        if not partition_dir.is_dir():
            continue

        files = _collect_partition_files(
            partition_dir,
            include_html=include_html,
        )
        partition_md_root = parsed_root / partition_id
        parsed_markdown: list[Path] = _existing_markdown(partition_md_root)

        if force_reparse and files:
            _reset_output_dirs(files, partition_md_root)
            parsed_markdown = _existing_markdown(partition_md_root)

        pending_parse = _missing_parse_files(files, partition_md_root)
        if pending_parse:
            if parser is None:
                raise RuntimeError(
                    f"{len(pending_parse)} document(s) under {partition_id} need "
                    "parsing but --no-parse was set. Drop --no-parse or run once with "
                    "parsing enabled first."
                )
            print(
                f"[{partition_id}] parsing {len(pending_parse)} document(s) "
                f"({len(parsed_markdown)} already parsed)"
            )
            parser.parse_files(
                pending_parse,
                output_dir=partition_md_root,
            )
            parsed_markdown = _existing_markdown(partition_md_root)

        if not parsed_markdown:
            print(f"[{partition_id}] no parsed markdown to ingest; skipped")
            continue

        documents = _attach_partition_metadata(
            load_documents_from_path(partition_md_root),
            partition=partition,
            partition_id=partition_id,
        )
        chunks = split_documents(documents, config)
        upserted = vectorstore.upsert_documents(chunks)
        total_upserted += upserted
        partitions_summary[partition_id] = {
            "source_dir": str(partition_dir),
            "source_files": len(files),
            "markdown_files": len(parsed_markdown),
            "documents": len(documents),
            "chunks": len(chunks),
            "upserted": upserted,
        }
        print(
            f"[{partition_id}] documents={len(documents)} "
            f"chunks={len(chunks)} upserted={upserted}"
        )

    return {
        "source_dir": str(source_root),
        "output_dir": str(parsed_root),
        "milvus_url": config.milvus_url,
        "db_name": config.db_name,
        "collection_name": vectorstore.collection_name,
        "partitions": partitions_summary,
        "total_upserted": total_upserted,
        "rebuild": rebuild,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse source_documents/<partition> documents and write each partition "
            "into its own Milvus partition (partition mode)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE_DIR,
        help="Directory whose <partition_id> subdirectories hold raw documents.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where MinerU-parsed markdown is written, per partition.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the collection (and re-parse documents).",
    )
    parser.add_argument(
        "--no-parse",
        dest="parse",
        action="store_false",
        help="Skip MinerU parsing; ingest already-parsed markdown under --output-dir.",
    )
    parser.add_argument(
        "--include-html",
        action="store_true",
        help="Also parse .html/.htm files (noise like download pages is skipped by default).",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Re-parse documents whose markdown output already exists.",
    )
    args = parser.parse_args()

    result = ingest_partitioned(
        rebuild=args.rebuild,
        source=args.source,
        parse=args.parse,
        output_dir=args.output_dir,
        include_html=args.include_html,
        force_parse=args.force_parse,
    )
    print(f"source_dir: {result['source_dir']}")
    print(f"output_dir: {result['output_dir']}")
    print(f"collection_name: {result['collection_name']}")
    for partition_id, summary in result["partitions"].items():
        print(f"{partition_id}: {summary}")
    print(f"total_upserted: {result['total_upserted']}")
    print(f"rebuild: {result['rebuild']}")
    return 0


def _collect_partition_files(
    partition_dir: Path,
    *,
    include_html: bool,
) -> list[Path]:
    files = collect_parse_sources([partition_dir])
    if include_html:
        return files
    return [file_path for file_path in files if file_path.suffix.lower() not in NOISE_SUFFIXES]


def _missing_parse_files(files: list[Path], partition_md_root: Path) -> list[Path]:
    if not files:
        return []
    if not partition_md_root.is_dir():
        return files
    return [
        file_path
        for file_path in files
        if not _canonical_markdown_path(partition_md_root, file_path).is_file()
    ]


def _canonical_markdown_path(partition_md_root: Path, source: Path) -> Path:
    stem = source.stem
    return partition_md_root / stem / f"{stem}.md"


def _reset_output_dirs(files: list[Path], partition_md_root: Path) -> None:
    """Remove canonical <root>/<stem> output dirs before forced re-parse so MinerU
    writes to the stable location instead of a -N suffixed duplicate."""
    if not partition_md_root.is_dir():
        return
    for file_path in files:
        output_dir = partition_md_root / file_path.stem
        if output_dir.is_dir():
            shutil.rmtree(output_dir)


def _existing_markdown(partition_md_root: Path) -> list[Path]:
    if not partition_md_root.is_dir():
        return []
    return sorted(
        path
        for path in partition_md_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}
    )


def _attach_partition_metadata(
    documents: list[Any],
    *,
    partition: Any,
    partition_id: str,
) -> list[Any]:
    for document in documents:
        source = str(document.metadata.get("source", ""))
        document.metadata = {
            **partition.metadata_template,
            "source": source,
            "partition_id": partition_id,
            "doc_title": Path(source).stem,
        }
    return documents


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
