from __future__ import annotations

import logging
import os
from typing import Any

from app.config import RagConfig
from app.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_MAX_SNIPPET_FALLBACK = 1200


def is_configured(config: RagConfig) -> bool:
    return bool((config.tavily_api_key or "").strip())


def _build_tool(config: RagConfig) -> Any | None:
    if not is_configured(config):
        return None
    os.environ.setdefault("TAVILY_API_KEY", config.tavily_api_key)
    try:
        from langchain_tavily import TavilySearchResults  # 延迟导入
    except ImportError:
        try:
            from langchain_community.tools import (
                TavilySearchResults,  # 老版本回退路径
            )
        except ImportError:
            logger.warning(
                "联网检索未启用：缺少 langchain-tavily。请 `pip install langchain-tavily` "
                "并配置 TAVILY_API_KEY，否则该环节静默跳过。"
            )
            return None

    kwargs: dict[str, Any] = {
        "max_results": config.top_k,
        "search_depth": "advanced",
        "include_raw_content": False,
    }
    if config.tavily_allowed_domains:
        kwargs["include_domains"] = config.tavily_allowed_domains
    try:
        return TavilySearchResults(**kwargs)
    except TypeError:
        # 兼容旧版参数名
        legacy: dict[str, Any] = {"k": config.top_k, "search_depth": "advanced"}
        return TavilySearchResults(**legacy)


def web_search(
    config: RagConfig,
    query: str,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """对 query 做一次联网检索，返回规范化的 RetrievedChunk 列表。

    依赖缺失 / 无 key / 调用异常一律返回空列表，由调用方决定如何收尾，
    不让单个外部请求炸掉整条链路。
    """
    if not query:
        return []
    tool = _build_tool(config)
    if tool is None:
        return []
    try:
        raw_results = tool.invoke({"query": query})
    except Exception:
        logger.warning("tavily web_search failed for query=%s", query, exc_info=True)
        return []

    return _to_chunks(raw_results, top_k or config.top_k)


def _to_chunks(raw_results: Any, max_results: int) -> list[RetrievedChunk]:
    if not isinstance(raw_results, list):
        return []
    chunks: list[RetrievedChunk] = []
    for rank, item in enumerate(raw_results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        text = (item.get("content") or "").strip()
        if not text:
            text = (item.get("raw_content") or "").strip()[:_MAX_SNIPPET_FALLBACK]
        if not text or not url:
            continue
        score = item.get("score")
        chunks.append(
            RetrievedChunk(
                rank=rank,
                text=text,
                chunk_id=url,
                source=url,
                score=float(score) if score is not None else None,
                metadata={
                    "title": item.get("title"),
                    "engine": "tavily",
                },
            )
        )
    return chunks
