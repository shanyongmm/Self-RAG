from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.schemas import RetrievedChunk
from rag.partitions import default_registry

MAX_ENTITIES = 8
MAX_SOURCE_IDS = 8
SUMMARY_LIMIT = 180
HISTORY_MAX_TURNS = 4
HISTORY_MESSAGE_CHAR_CAP = 400
HISTORY_MAX_CHARS = 1600

PARTITION_KEYWORDS = {
    "device": [
        "udi",
        "ct",
        "dr",
        "影像",
        "仪器",
        "分类目录",
        "医疗器械",
        "医疗设备",
        "监护仪",
        "设备",
        "器械",
        "注册",
    ],
    "medication": [
        "不良反应",
        "基本药物",
        "抗菌药",
        "处方",
        "剂型",
        "用药",
        "禁忌",
        "规格",
        "药品",
        "药物",
        "药",
    ],
    "diagnosis": [
        "高血压",
        "糖尿病",
        "并发症",
        "慢病",
        "指南",
        "治疗",
        "症状",
        "筛查",
        "诊断",
        "转诊",
    ],
}

FOLLOW_UP_MARKERS = [
    "上面",
    "上述",
    "上一",
    "前面",
    "刚才",
    "这个",
    "这些",
    "这种",
    "该",
    "它",
    "其",
    "继续",
    "还有",
    "那",
    "呢",
]

ENTITY_HINTS = [
    "高血压",
    "糖尿病",
    "基本药物",
    "抗菌药",
    "医疗器械",
    "医疗设备",
    "udi",
    "ct",
    "dr",
]

ATTRIBUTE_TERMS = [
    "禁忌",
    "不良反应",
    "剂型",
    "规格",
    "诊断",
    "治疗",
    "筛查",
    "转诊",
    "变化",
    "区别",
    "新版",
    "流程",
    "条件",
    "标准",
]


def prepare_memory_context(
    question: str,
    memory_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    previous = _normalize_memory(memory_context)
    current_entities = _extract_entities(question)
    current_partition = route_partition(question)
    is_follow_up = _is_follow_up_question(question, current_entities, previous)

    entities = current_entities
    if is_follow_up:
        entities = _merge_unique(
            previous.get("entities", []),
            current_entities,
            limit=MAX_ENTITIES,
        )

    partition_id = current_partition
    if partition_id is None and is_follow_up:
        partition_id = _string_or_none(previous.get("partition_id"))

    contextual_question = _build_contextual_question(
        question=question,
        memory_context=previous,
        entities=entities,
        partition_id=partition_id,
        is_follow_up=is_follow_up,
    )

    prepared_memory = {
        **previous,
        "active_entities": entities,
        "active_partition_id": partition_id,
        "last_follow_up": is_follow_up,
        "last_contextual_question": contextual_question,
    }

    return {
        "memory_context": prepared_memory,
        "contextual_question": contextual_question,
        "retrieval_query": contextual_question,
        "partition_id": partition_id,
        "is_follow_up": is_follow_up,
        "entities": entities,
    }


def update_memory_context(
    memory_context: Mapping[str, Any] | None,
    *,
    question: str,
    contextual_question: str,
    partition_id: str | None,
    documents: Sequence[RetrievedChunk | dict[str, Any]],
    rejected_documents: Sequence[RetrievedChunk | dict[str, Any]],
    answer_text: str,
) -> dict[str, Any]:
    previous = _normalize_memory(memory_context)
    source_entities = _extract_entities_from_sources(documents)
    question_entities = _extract_entities(question)
    entities = _merge_unique(
        previous.get("active_entities", previous.get("entities", [])),
        question_entities,
        source_entities,
        limit=MAX_ENTITIES,
    )
    effective_source_ids = _chunk_ids(documents)
    rejected_source_ids = _chunk_ids(rejected_documents)
    resolved_partition = partition_id or _string_or_none(
        previous.get("active_partition_id") or previous.get("partition_id")
    )
    current_topic = _build_topic(
        question=question,
        entities=entities,
        partition_id=resolved_partition,
        documents=documents,
    )

    return {
        "turn_count": int(previous.get("turn_count", 0) or 0) + 1,
        "current_topic": current_topic,
        "entities": entities,
        "partition_id": resolved_partition,
        "summary": _build_summary(
            current_topic=current_topic,
            question=question,
            answer_text=answer_text,
        ),
        "last_question": question,
        "last_contextual_question": contextual_question,
        "last_effective_sources": effective_source_ids,
        "last_rejected_sources": rejected_source_ids,
        "last_answer_preview": _compact(answer_text)[:SUMMARY_LIMIT],
    }


def compact_memory_for_trace(
    memory_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    memory = _normalize_memory(memory_context)
    return {
        "turn_count": memory.get("turn_count", 0),
        "current_topic": memory.get("current_topic"),
        "entities": memory.get("entities") or memory.get("active_entities") or [],
        "partition_id": memory.get("partition_id")
        or memory.get("active_partition_id"),
        "summary": memory.get("summary"),
        "last_effective_sources": memory.get("last_effective_sources", []),
    }


def build_history_window(
    messages: Sequence[Any] | None,
    *,
    max_turns: int = HISTORY_MAX_TURNS,
    max_chars: int = HISTORY_MAX_CHARS,
) -> str:
    """Render recent complete turns from persisted ``messages`` as plain text.

    Current-turn human message is dropped; window is pure prior context.
    """
    if not messages:
        return ""
    prior = [m for m in messages if _message_type(m) in ("human", "ai")]
    if not prior or _message_type(prior[-1]) != "ai":
        prior = prior[:-1] if prior else []

    selected: list[list[Any]] = []
    used_chars = 0
    index = len(prior) - 1
    while index >= 1 and len(selected) < max_turns:
        human, ai = prior[index - 1], prior[index]
        if _message_type(human) != "human" or _message_type(ai) != "ai":
            break
        turn_chars = len(_message_text(human)) + len(_message_text(ai))
        if selected and used_chars + turn_chars > max_chars:
            break
        selected.append([human, ai])
        used_chars += turn_chars
        index -= 2
    selected.reverse()

    lines: list[str] = []
    for human, ai in selected:
        human_text = _clip(_message_text(human))
        ai_text = _clip(_message_text(ai))
        if human_text:
            lines.append(f"用户：{human_text}")
        if ai_text:
            lines.append(f"助手：{ai_text}")
    return "\n".join(lines)


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return message.get("type", "")
    return getattr(message, "type", "")


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(
        message, "content", message
    )
    return _compact(_flatten_content(content))


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _clip(text: str) -> str:
    if len(text) <= HISTORY_MESSAGE_CHAR_CAP:
        return text
    return f"{text[:HISTORY_MESSAGE_CHAR_CAP]}…"


def route_partition(question: str) -> str | None:
    registry = default_registry()
    if registry is not None:
        return registry.route(question)
    normalized = question.lower()
    for partition_id, keywords in PARTITION_KEYWORDS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            return partition_id
    return None


def _normalize_memory(memory_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not memory_context:
        return {}
    return dict(memory_context)


def _is_follow_up_question(
    question: str,
    current_entities: Sequence[str],
    memory_context: Mapping[str, Any],
) -> bool:
    if not memory_context.get("summary") and not memory_context.get("entities"):
        return False

    stripped = question.strip()
    if any(marker in stripped for marker in FOLLOW_UP_MARKERS):
        return True

    if len(stripped) <= 24 and not current_entities:
        return True

    return (
        len(stripped) <= 28
        and not current_entities
        and any(term in stripped for term in ATTRIBUTE_TERMS)
        and bool(memory_context.get("entities"))
    )


def _build_contextual_question(
    *,
    question: str,
    memory_context: Mapping[str, Any],
    entities: Sequence[str],
    partition_id: str | None,
    is_follow_up: bool,
) -> str:
    if not is_follow_up:
        return question

    parts = [f"用户追问：{question}"]
    topic = _string_or_none(memory_context.get("current_topic"))
    summary = _string_or_none(memory_context.get("summary"))
    if topic:
        parts.append(f"上一轮主题：{topic}")
    if entities:
        parts.append(f"相关实体：{'、'.join(entities)}")
    if partition_id:
        parts.append(f"知识分区：{partition_id}")
    if summary:
        parts.append(f"上下文摘要：{summary}")
    return "\n".join(parts)


def _build_topic(
    *,
    question: str,
    entities: Sequence[str],
    partition_id: str | None,
    documents: Sequence[RetrievedChunk | dict[str, Any]],
) -> str:
    if entities:
        label = _partition_label(partition_id)
        return f"{entities[0]}{label}"

    source_title = _first_source_title(documents)
    if source_title:
        return source_title

    return _compact(question)[:60]


def _first_source_title(
    documents: Sequence[RetrievedChunk | dict[str, Any]],
) -> str | None:
    for document in documents:
        metadata = _metadata(document)
        for key in ("doc_title", "section_title", "section_path_text"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        source = _source(document)
        if source:
            return Path(source).stem

    return None


def _build_summary(
    *,
    current_topic: str,
    question: str,
    answer_text: str,
) -> str:
    answer_preview = _compact(answer_text)[:80]
    summary = f"当前主题：{current_topic}；上一问题：{_compact(question)[:80]}"
    if answer_preview:
        summary = f"{summary}；上一回答摘要：{answer_preview}"
    return summary[:SUMMARY_LIMIT]


def _extract_entities(question: str) -> list[str]:
    normalized = question.lower()
    entities = [
        hint
        for hint in ENTITY_HINTS
        if hint.lower() in normalized
    ]

    quoted = re.findall(r'[\u300a\u201c"]([^\u300b\u201d"]{2,40})[\u300b\u201d"]', question)
    entities.extend(quoted)
    entities.extend(_extract_named_subjects(question))

    return _merge_unique(entities, limit=MAX_ENTITIES)


def _extract_named_subjects(question: str) -> list[str]:
    subjects: list[str] = []
    patterns = [
        r"([A-Za-z0-9一-鿿]{2,30})(?:的)?(?:禁忌|不良反应|剂型|规格|用药|诊断|治疗|筛查|转诊|标准|指南)",
        r"([A-Za-z0-9一-鿿]{2,30})(?:怎么|如何|有哪些|是什么)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, question):
            subject = _strip_subject_prefix(match)
            if subject:
                subjects.append(subject)
    return subjects


def _strip_subject_prefix(value: str) -> str:
    subject = value.strip(" ，。？！?：:、；;")
    for prefix in ("请问", "咨询", "了解", "关于"):
        subject = subject.removeprefix(prefix)
    if subject in {"这个", "这种", "这些", "那个", "那种", "它", "其"}:
        return ""
    return subject.strip()

def _extract_entities_from_sources(
    documents: Sequence[RetrievedChunk | dict[str, Any]],
) -> list[str]:
    values: list[str] = []
    for document in documents:
        metadata = _metadata(document)
        for key in ("doc_title", "section_title", "section_path_text"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
                break

        source = _source(document)
        if source:
            values.append(Path(source).stem)

    return _merge_unique(values, limit=MAX_ENTITIES)


def _merge_unique(*groups: Sequence[Any], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            value = str(item).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
            if len(merged) >= limit:
                return merged
    return merged


def _chunk_ids(
    documents: Sequence[RetrievedChunk | dict[str, Any]],
) -> list[str]:
    ids: list[str] = []
    for document in documents:
        chunk_id = _chunk_id(document)
        if chunk_id:
            ids.append(chunk_id)
        if len(ids) >= MAX_SOURCE_IDS:
            break
    return ids


def _chunk_id(document: RetrievedChunk | dict[str, Any]) -> str | None:
    if isinstance(document, RetrievedChunk):
        return str(document.chunk_id) if document.chunk_id is not None else None
    value = document.get("chunk_id")
    return str(value) if value is not None else None


def _metadata(document: RetrievedChunk | dict[str, Any]) -> dict[str, Any]:
    if isinstance(document, RetrievedChunk):
        return document.metadata
    value = document.get("metadata")
    return value if isinstance(value, dict) else {}


def _source(document: RetrievedChunk | dict[str, Any]) -> str | None:
    if isinstance(document, RetrievedChunk):
        return document.source
    value = document.get("source")
    return str(value) if value is not None else None


def _partition_label(partition_id: str | None) -> str:
    registry = default_registry()
    if registry is not None:
        return registry.label(partition_id)
    return {
        "diagnosis": "诊疗问题",
        "device": "设备问题",
        "medication": "用药问题",
    }.get(partition_id or "", "相关问题")


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
