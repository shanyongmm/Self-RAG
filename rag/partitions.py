from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import RagConfig, get_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTITION_CONFIG = PROJECT_ROOT / "knowledge_partitions.json"

# 路由优先级固定为 设备 -> 用药 -> 诊断，与 app/memory.py 历史顺序保持一致，
# 避免混合问题（如高血压+用药）在不同关键词集合下产生漂移。
_ROUTE_PRIORITY = ("device", "medication", "diagnosis")

_LABELS = {
    "diagnosis": "诊疗问题",
    "device": "设备问题",
    "medication": "用药问题",
}

_SOURCE_TYPE_FALLBACK = {
    "diagnosis": "clinical_guideline",
    "device": "medical_device",
    "medication": "medication_catalog",
}


@dataclass(frozen=True)
class Partition:
    partition_id: str
    name: str
    scope: str
    route_keywords: tuple[str, ...]
    metadata_template: dict[str, Any]

    @property
    def label(self) -> str:
        return _LABELS.get(self.partition_id, self.name)

    @property
    def source_type(self) -> str:
        value = self.metadata_template.get("source_type")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return _SOURCE_TYPE_FALLBACK.get(self.partition_id, "document")


@dataclass(frozen=True)
class PartitionRegistry:
    collection_name: str
    default_partition: str
    partitions: dict[str, Partition]

    @property
    def partition_ids(self) -> tuple[str, ...]:
        ordered = [partition_id for partition_id in _ROUTE_PRIORITY if partition_id in self.partitions]
        ordered.extend(partition_id for partition_id in self.partitions if partition_id not in ordered)
        return tuple(ordered)

    def partition(self, partition_id: str) -> Partition | None:
        return self.partitions.get(partition_id)

    def label(self, partition_id: str | None) -> str:
        if partition_id is None:
            return "相关问题"
        part = self.partitions.get(partition_id)
        return part.label if part else "相关问题"

    def source_type(self, partition_id: str) -> str:
        part = self.partitions.get(partition_id)
        return part.source_type if part else "document"

    def route(self, question: str) -> str | None:
        normalized = question.lower()
        for partition_id in self.partition_ids:
            part = self.partitions[partition_id]
            if any(keyword.lower() in normalized for keyword in part.route_keywords):
                return partition_id
        return None


def load_partition_registry(path: Path | None) -> PartitionRegistry | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    return _load_partition_registry(str(resolved))


@lru_cache(maxsize=4)
def _load_partition_registry(path: str) -> PartitionRegistry | None:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None

    raw_partitions = data.get("partitions") or []
    partitions: dict[str, Partition] = {}
    for raw in raw_partitions:
        partition_id = str(raw.get("id") or raw.get("partition_name") or "").strip()
        if not partition_id:
            continue
        keywords = tuple(
            str(item).strip()
            for item in raw.get("route_keywords") or []
            if str(item).strip()
        )
        partitions[partition_id] = Partition(
            partition_id=partition_id,
            name=str(raw.get("name") or partition_id),
            scope=str(raw.get("scope") or ""),
            route_keywords=keywords,
            metadata_template=dict(raw.get("metadata_template") or {}),
        )

    if not partitions:
        return None

    return PartitionRegistry(
        collection_name=str(
            data.get("milvus_collection_name")
            or raw_partitions[0].get("collection_name")
            or "medical_knowledge_base"
        ),
        default_partition=str(data.get("default_partition") or "diagnosis"),
        partitions=partitions,
    )


def registry_from_config(config: RagConfig | None = None) -> PartitionRegistry | None:
    """分区模式是否开启取决于 config.partition_config（env PARTITION_CONFIG 是否启用）。"""
    config = config or get_config()
    return load_partition_registry(config.partition_config)


def default_registry() -> PartitionRegistry | None:
    """项目自带 knowledge_partitions.json 时返回注册表，供 memory 路由/标签使用。"""
    return load_partition_registry(DEFAULT_PARTITION_CONFIG)
