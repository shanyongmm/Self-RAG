from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import RagConfig, get_config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
SENTENCE_RE = re.compile(r"[^。！？!?；;.!?\n]+[。！？!?；;.!?]?")
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


@dataclass(frozen=True)
class MarkdownBlock:
    content: str
    block_type: str
    section_path: tuple[str, ...]
    block_index: int
    metadata: dict[str, Any]


class HybridMarkdownSplitter:
    """Structure-aware markdown splitter with metadata context enrichment."""

    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.chunk_size = config.chunk_size
        self.context_chars = max(config.chunk_overlap, 0)
        self.table_chunk_size = max(config.chunk_size * 2, 512)

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document_index, document in enumerate(documents):
            blocks = _parse_markdown_blocks(
                document.page_content,
                metadata=document.metadata,
                document_index=document_index,
            )
            for block in blocks:
                chunks.extend(self._split_block(block))

        self._attach_context_metadata(chunks)
        return chunks

    def _split_block(self, block: MarkdownBlock) -> list[Document]:
        if block.block_type == "text":
            return self._split_text_block(block)
        if block.block_type == "table":
            return self._split_table_block(block)
        return self._documents_from_contents(block, [block.content])

    def _split_text_block(self, block: MarkdownBlock) -> list[Document]:
        text_chunks = _pack_text_units(_split_text_units(block.content), self.chunk_size)
        return self._documents_from_contents(block, text_chunks)

    def _split_table_block(self, block: MarkdownBlock) -> list[Document]:
        table_chunks = _split_table(block.content, self.table_chunk_size)
        return self._documents_from_contents(block, table_chunks)

    def _documents_from_contents(
        self,
        block: MarkdownBlock,
        contents: Sequence[str],
    ) -> list[Document]:
        documents: list[Document] = []
        filtered_contents = [content.strip() for content in contents if content.strip()]
        total_chunks = len(filtered_contents)
        source = str(block.metadata.get("source", ""))
        section_path = list(block.section_path)
        section_path_text = " > ".join(section_path)

        for chunk_index, content in enumerate(filtered_contents):
            metadata = {
                **block.metadata,
                "chunk_id": _build_chunk_id(
                    source=source,
                    block_index=block.block_index,
                    chunk_index=chunk_index,
                    content=content,
                ),
                "block_type": block.block_type,
                "block_index": block.block_index,
                "chunk_index": chunk_index,
                "chunk_count": total_chunks,
                "section_path": section_path,
                "section_path_text": section_path_text,
                "section_title": section_path[-1] if section_path else "",
            }
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def _attach_context_metadata(self, chunks: list[Document]) -> None:
        for index, chunk in enumerate(chunks):
            before = ""
            after = ""
            if self.context_chars > 0:
                previous_chunk = _neighbor_from_same_source(chunks, index, -1)
                next_chunk = _neighbor_from_same_source(chunks, index, 1)
                if previous_chunk is not None:
                    before = _tail_snippet(previous_chunk.page_content, self.context_chars)
                if next_chunk is not None:
                    after = _head_snippet(next_chunk.page_content, self.context_chars)

            chunk.metadata = {
                **chunk.metadata,
                "context_before": before,
                "context_after": after,
            }
            chunk.metadata["embedding_text"] = _build_embedding_text(chunk)


def split_documents(
    documents: Sequence[Document],
    config: RagConfig | None = None,
) -> list[Document]:
    config = config or get_config()
    return HybridMarkdownSplitter(config).split_documents(documents)


def _parse_markdown_blocks(
    markdown_content: str,
    metadata: dict[str, Any],
    document_index: int,
) -> list[MarkdownBlock]:
    lines = markdown_content.splitlines()
    blocks: list[MarkdownBlock] = []
    section_stack: list[str] = []
    pending_text: list[str] = []
    block_index = 0
    index = 0

    def current_section_path() -> tuple[str, ...]:
        return tuple(title for title in section_stack if title)

    def append_block(content: str, block_type: str) -> None:
        nonlocal block_index
        stripped = content.strip()
        if not stripped:
            return
        blocks.append(
            MarkdownBlock(
                content=stripped,
                block_type=block_type,
                section_path=current_section_path(),
                block_index=block_index,
                metadata={**metadata, "document_index": document_index},
            )
        )
        block_index += 1

    def flush_text() -> None:
        if pending_text:
            append_block("\n".join(pending_text), "text")
            pending_text.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_text()
            index += 1
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_text()
            level = len(heading_match.group(1))
            title = _clean_heading(heading_match.group(2))
            section_stack = section_stack[: level - 1]
            section_stack.append(title)
            index += 1
            continue

        fence_match = FENCE_RE.match(line)
        if fence_match:
            flush_text()
            fence = fence_match.group(1)[:3]
            end = index + 1
            while end < len(lines):
                if lines[end].lstrip().startswith(fence):
                    end += 1
                    break
                end += 1
            append_block("\n".join(lines[index:end]), "code")
            index = end
            continue

        if _starts_formula(stripped):
            flush_text()
            end = _find_formula_end(lines, index)
            append_block("\n".join(lines[index:end]), "formula")
            index = end
            continue

        if _is_table_start(lines, index):
            flush_text()
            end = index + 2
            while end < len(lines) and "|" in lines[end] and lines[end].strip():
                end += 1
            append_block("\n".join(lines[index:end]), "table")
            index = end
            continue

        if IMAGE_RE.fullmatch(stripped):
            flush_text()
            append_block(line, "image")
            index += 1
            continue

        pending_text.append(line)
        index += 1

    flush_text()
    return blocks


def _split_text_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in text.splitlines():
        stripped = paragraph.strip()
        if not stripped:
            continue
        if _looks_like_list_item(stripped):
            units.append(stripped)
            continue
        sentence_units = [match.group(0).strip() for match in SENTENCE_RE.finditer(stripped)]
        units.extend(unit for unit in sentence_units if unit)
    return units or [text.strip()]


def _pack_text_units(units: Sequence[str], chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append("\n".join(current).strip())
            current.clear()

    for unit in units:
        if len(unit) > chunk_size:
            flush()
            chunks.extend(_hard_split(unit, chunk_size))
            continue

        candidate = "\n".join([*current, unit]).strip()
        if current and len(candidate) > chunk_size:
            flush()
        current.append(unit)

    flush()
    return chunks


def _split_table(table: str, chunk_size: int) -> list[str]:
    lines = [line for line in table.splitlines() if line.strip()]
    if len(table) <= chunk_size or len(lines) <= 3:
        return [table]

    header = lines[:2]
    body = lines[2:]
    chunks: list[str] = []
    current_rows: list[str] = []

    def flush() -> None:
        if current_rows:
            chunks.append("\n".join([*header, *current_rows]))
            current_rows.clear()

    for row in body:
        candidate = "\n".join([*header, *current_rows, row])
        if current_rows and len(candidate) > chunk_size:
            flush()
        current_rows.append(row)

    flush()
    return chunks or [table]


def _hard_split(text: str, chunk_size: int) -> list[str]:
    return [
        text[start : start + chunk_size].strip()
        for start in range(0, len(text), chunk_size)
        if text[start : start + chunk_size].strip()
    ]


def _build_embedding_text(chunk: Document) -> str:
    metadata = chunk.metadata
    parts: list[str] = []
    section_path = metadata.get("section_path_text")
    block_type = metadata.get("block_type")
    context_before = metadata.get("context_before")
    context_after = metadata.get("context_after")

    if section_path:
        parts.append(f"章节路径：{section_path}")
    if block_type:
        parts.append(f"内容类型：{block_type}")
    if context_before:
        parts.append(f"前文上下文：{context_before}")

    parts.append(f"正文：\n{chunk.page_content}")

    if context_after:
        parts.append(f"后文上下文：{context_after}")

    return "\n\n".join(parts)


def _neighbor_from_same_source(
    chunks: Sequence[Document],
    index: int,
    offset: int,
) -> Document | None:
    neighbor_index = index + offset
    if neighbor_index < 0 or neighbor_index >= len(chunks):
        return None

    source = chunks[index].metadata.get("source")
    neighbor = chunks[neighbor_index]
    if neighbor.metadata.get("source") != source:
        return None
    return neighbor


def _build_chunk_id(
    source: str,
    block_index: int,
    chunk_index: int,
    content: str,
) -> str:
    source_name = _safe_id(Path(source).stem if source else "document")
    digest = hashlib.sha1(
        f"{source}|{block_index}|{chunk_index}|{content}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{source_name}:{block_index}:{chunk_index}:{digest}"


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value).strip("_")
    return safe[:40] or "document"


def _clean_heading(title: str) -> str:
    return title.strip().strip("#").strip()


def _starts_formula(stripped: str) -> bool:
    return stripped.startswith("$$") or stripped.startswith("\\[")


def _find_formula_end(lines: Sequence[str], start: int) -> int:
    first = lines[start].strip()
    if first.startswith("$$"):
        if first.count("$$") >= 2 and len(first) > 2:
            return start + 1
        for index in range(start + 1, len(lines)):
            if lines[index].strip().endswith("$$"):
                return index + 1
    if first.startswith("\\["):
        for index in range(start + 1, len(lines)):
            if lines[index].strip().endswith("\\]"):
                return index + 1
    return start + 1


def _is_table_start(lines: Sequence[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and bool(TABLE_SEPARATOR_RE.match(lines[index + 1]))
    )


def _looks_like_list_item(text: str) -> bool:
    return bool(re.match(r"^([-*+]\s+|\d+[.)]\s+|[一二三四五六七八九十]+[、.]\s*)", text))


def _head_snippet(text: str, limit: int) -> str:
    return _compact_text(text)[:limit]


def _tail_snippet(text: str, limit: int) -> str:
    return _compact_text(text)[-limit:]


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
