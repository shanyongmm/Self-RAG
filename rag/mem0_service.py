from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import RagConfig

logger = logging.getLogger(__name__)


class Mem0Service:
    """Mem0 长期记忆封装。MEM0_ENABLED=false 时所有方法静默降级返回空。"""

    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self._memory: Any = None
        if config.mem0_enabled:
            self._memory = self._build_memory(config)

    def _build_memory(self, config: RagConfig) -> Any:
        from mem0 import Memory  # 延迟导入:没装/没开时不影响主链路 import

        history_path = config.mem0_history_db_path or (
            config.project_root / "data/mem0/history.db"
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)

        return Memory.from_config(self._mem0_config(config, str(history_path)))

    @staticmethod
    def _mem0_config(config: RagConfig, history_db_path: str) -> dict[str, Any]:
        return {
            # mem0 自己的 history 落盘(sqlite);不设会默认写到 ~/.mem0
            "history_db_path": history_db_path,
            # 抽取用 LLM:读对话、提炼事实(复用 DeepSeek)
            "llm": {
                "provider": "openai",
                "config": {
                    "model": config.llm_model,
                    "api_key": config.llm_api_key,
                    "openai_base_url": config.llm_base_url,
                    "temperature": 0.1,
                },
            },
            # 向量化模型:记忆句子 → 向量(接 DashScope OpenAI 兼容端点)
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": config.embed_model_name,
                    "api_key": config.embed_api_key,
                    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    # "embedding_dims": config.embed_dimension,
                },
            },
            # 向量库:记忆向量存哪/去哪搜(复用本地 Milvus 的独立 collection)
            "vector_store": {
                "provider": "milvus",
                "config": {
                    "collection_name": config.mem0_collection_name,
                    "embedding_model_dims": config.embed_dimension,
                    "url": config.milvus_url,
                    # Milvus 无鉴权时也给个默认凭据字符串,字段必填
                    "token": "root:Milvus",
                },
            },
        }

    def search(self, user_id, query, top_k=None):
        if self._memory is None or not user_id:
            return []
        try:
            raw = self._memory.search(
                query,
                filters={"user_id": user_id},
                top_k=top_k or self.config.mem0_top_k,
            )
            # mem0 2.x 返回 {"results": [...]};个别版本直接返回 list,兼容两者
            if isinstance(raw, dict):
                results = raw.get("results") or []
            else:
                results = raw or []
            return [normalize_mem0_hit(h) for h in results]
        except Exception:
            logger.warning("mem0.search failed for user=%s", user_id, exc_info=True)
            return []

    def add(self, user_id, run_id, question, answer, metadata=None):
        if self._memory is None or not user_id:
            return None
        try:
            return self._memory.add(
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": answer}],
                user_id=user_id,
                run_id=run_id,
                metadata=metadata or {},
            )
        except Exception:
            logger.warning("mem0.add failed for user=%s run=%s", user_id, run_id, exc_info=True)
            return None


def normalize_mem0_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """把 mem0 返回结构归一成方便展示的字段;键名因版本而异,尽量宽容取值。"""
    return {
        "id": hit.get("id"),
        "memory": hit.get("memory", hit.get("text", "")),
        "score": hit.get("score"),
        "created_at": hit.get("created_at"),
        "metadata": hit.get("metadata") or {},
    }


@lru_cache(maxsize=1)
def get_mem0_service(config: RagConfig | None = None) -> Mem0Service:
    from app.config import get_config
    return Mem0Service(config or get_config())
