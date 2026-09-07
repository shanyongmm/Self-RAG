from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mineru import MinerU
from mineru.models import ExtractResult

from app.config import RagConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINERU_BASE_URL = "https://mineru.net/api/v4"

SUPPORTED_MINERU_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".html",
    ".htm",
}


@dataclass(frozen=True)
class ParsedDocument:
    source_path: Path
    markdown_path: Path
    filename: str
    extra_paths: dict[str, Path] = field(default_factory=dict)


class MinerUDocumentParser:
    def __init__(self, config: RagConfig | None = None) -> None:
        if config is None:
            load_dotenv(PROJECT_ROOT / ".env")
            self.project_root = PROJECT_ROOT
            token = os.getenv("MINERU_TOKEN") or os.getenv("MINERU_API_KEY")
            base_url = os.getenv("MINERU_BASE_URL") or DEFAULT_MINERU_BASE_URL
            output_dir = _resolve_path(
                self.project_root,
                os.getenv("MINERU_OUTPUT_DIR") or "data/parsed/mineru",
            )
        else:
            self.project_root = config.project_root
            token = config.mineru_token
            base_url = config.mineru_base_url
            output_dir = config.mineru_output_dir

        if not token:
            raise RuntimeError(
                "Missing required environment variable: MINERU_TOKEN. "
                "Put your MinerU token in .env instead of source code."
            )

        self.output_dir = output_dir
        self.client = MinerU(token=token, base_url=base_url)

    def parse_files(
        self,
        file_paths: Sequence[str | Path],
        output_dir: str | Path | None = None,
        *,
        model: str | None = None,
        ocr: bool | None = None,
        language: str | None = None,
        extra_formats: Sequence[str] = (),
        timeout: int = 1800,
        with_images: bool = True,
        retry_attempts: int = 3,
        retry_wait: float = 60.0,
        request_interval: float = 3.0,
    ) -> list[ParsedDocument]:
        sources = [_resolve_existing_file(path) for path in file_paths]
        if not sources:
            return []

        output_root = Path(output_dir) if output_dir else self.output_dir
        output_root = _resolve_path(self.project_root, output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        requested_formats = [item.lower() for item in extra_formats]
        parse_options: dict[str, object] = {"timeout": timeout}
        if model is not None:
            parse_options["model"] = model
        if ocr is not None:
            parse_options["ocr"] = ocr
        if language is not None:
            parse_options["language"] = language
        if requested_formats:
            parse_options["extra_formats"] = requested_formats

        parsed_documents: list[ParsedDocument] = []
        for index, source_path in enumerate(sources):
            print(f"MinerU parsing ({index + 1}/{len(sources)}): {source_path.name}")
            result = self._extract_with_retry(
                source_path,
                parse_options,
                retry_attempts=retry_attempts,
                retry_wait=retry_wait,
            )
            if result.state != "done":
                message = result.error or result.err_code or result.state
                raise RuntimeError(f"MinerU failed to parse {source_path}: {message}")

            stem = _safe_stem(result.filename or source_path.name)
            document_dir = _unique_output_dir(output_root, stem, index)
            markdown_path = document_dir / f"{stem}.md"
            result.save_markdown(str(markdown_path), with_images=with_images)

            extra_paths: dict[str, Path] = {}
            if "html" in requested_formats and result.html is not None:
                extra_paths["html"] = result.save_html(str(document_dir / f"{stem}.html"))
            if "docx" in requested_formats and result.docx is not None:
                extra_paths["docx"] = result.save_docx(str(document_dir / f"{stem}.docx"))
            if "latex" in requested_formats and result.latex is not None:
                extra_paths["latex"] = result.save_latex(str(document_dir / f"{stem}.tex"))

            parsed_documents.append(
                ParsedDocument(
                    source_path=source_path,
                    markdown_path=markdown_path,
                    filename=result.filename or source_path.name,
                    extra_paths=extra_paths,
                )
            )

            if request_interval > 0 and index < len(sources) - 1:
                time.sleep(request_interval)

        return parsed_documents

    def _extract_with_retry(
        self,
        source_path: Path,
        parse_options: dict[str, object],
        *,
        retry_attempts: int,
        retry_wait: float,
    ) -> ExtractResult:
        attempts = max(1, retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                return self.client.extract(str(source_path), **parse_options)
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code != 429 or attempt >= attempts:
                    raise
                wait_seconds = _retry_after_seconds(exc.response, retry_wait, attempt)
                print(
                    f"MinerU rate limited {source_path.name}; "
                    f"retrying in {wait_seconds:.0f}s ({attempt}/{attempts})"
                )
                time.sleep(wait_seconds)
        raise RuntimeError(f"MinerU failed to parse {source_path}")


def collect_parse_sources(
    paths: Sequence[str | Path],
    *,
    exclude_dirs: Sequence[str | Path] = (),
) -> list[Path]:
    excluded = {_resolve_path(PROJECT_ROOT, path).resolve() for path in exclude_dirs}
    sources: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            sources.extend(
                file_path
                for file_path in sorted(path.rglob("*"))
                if file_path.is_file()
                and not _is_under_excluded_dir(file_path, excluded)
                and file_path.suffix.lower() in SUPPORTED_MINERU_SUFFIXES
            )
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_MINERU_SUFFIXES:
                supported = ", ".join(sorted(SUPPORTED_MINERU_SUFFIXES))
                raise ValueError(f"Unsupported file type: {path}. Supported: {supported}")
            sources.append(path)
        else:
            raise FileNotFoundError(f"Parse source not found: {path}")
    return sources


def _is_under_excluded_dir(path: Path, excluded_dirs: set[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == excluded or resolved.is_relative_to(excluded) for excluded in excluded_dirs)


def _retry_after_seconds(
    response: httpx.Response,
    default_wait: float,
    attempt: int,
) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    return max(default_wait * attempt, 1.0)


def _resolve_existing_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Parse source not found: {resolved}")
    return resolved.resolve()


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    return resolved


def _safe_stem(name: str) -> str:
    stem = Path(name).stem or "document"
    invalid_chars = '<>:"/\\|?*'
    safe = "".join("_" if char in invalid_chars else char for char in stem)
    return safe.strip(" .") or "document"


def _unique_output_dir(output_root: Path, stem: str, index: int) -> Path:
    candidate = output_root / stem
    if not candidate.exists():
        return candidate
    return output_root / f"{stem}-{index + 1}"