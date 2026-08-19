from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    rank: int = Field(ge=1)
    text: str = Field(min_length=1)
    chunk_id: str | int | None = None
    source: str | None = None
    score: float | None = None

    @classmethod
    def from_milvus_hit(cls, hit: dict[str, Any], rank: int) -> RetrievedChunk:
        entity = hit.get("entity") or {}
        text = entity.get("text") or hit.get("text") or ""
        chunk_id = entity.get("chunk_id", hit.get("id"))
        source = entity.get("source") or hit.get("source")
        score = hit.get("distance", hit.get("score"))

        return cls(
            rank=rank,
            text=str(text).strip(),
            chunk_id=chunk_id,
            source=str(source) if source is not None else None,
            score=float(score) if score is not None else None,
        )

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "score": self.score,
            "text": self.text,
        }


class ChunkRelevanceGrade(BaseModel):
    chunk_id: str = Field(description="被评分片段的 chunk_id。")
    is_relevant: bool = Field(description="该片段是否能支撑回答用户原始问题。")
    confidence: float = Field(
        ge=0,
        le=1,
        description="该片段相关性判断置信度。",
    )
    reason: str = Field(description="简短说明该片段是否相关的依据。")


class RelevanceGrade(BaseModel):
    is_relevant: bool = Field(description="检索片段是否足以支撑回答用户原始问题。")
    confidence: float = Field(
        ge=0,
        le=1,
        description="相关性判断置信度，0 表示完全不确定，1 表示非常确定。",
    )
    reason: str = Field(description="简短说明整体判断依据。")
    supporting_chunk_ids: list[str] = Field(
        default_factory=list,
        description="支撑相关性判断的 chunk_id 列表；无相关片段时为空。",
    )
    chunk_grades: list[ChunkRelevanceGrade] = Field(
        default_factory=list,
        description="对每个检索片段的独立相关性判断。",
    )


class QueryRewrite(BaseModel):
    rewritten_query: str = Field(
        min_length=1,
        description="用于下一轮向量检索的改写查询。",
    )
    reason: str = Field(description="简短说明改写方向。")


class RagGeneration(BaseModel):
    answer: str = Field(min_length=1, description="基于检索片段生成的最终回答。")
    is_answerable: bool = Field(description="检索片段是否足以回答问题。")
    citations: list[str] = Field(
        default_factory=list,
        description="答案引用到的 chunk_id 列表。",
    )
    missing_info: str | None = Field(
        default=None,
        description="如果无法回答，说明缺失的信息；可以回答时为空。",
    )
