from __future__ import annotations

from langchain.chat_models import init_chat_model

from app.config import RagConfig, get_config


def create_chat_model(config: RagConfig | None = None):
    config = config or get_config()
    kwargs = {
        "model": config.llm_model,
        "api_key": config.llm_api_key,
        "temperature": config.llm_temperature,
    }
    if config.llm_base_url:
        kwargs["base_url"] = config.llm_base_url
    return init_chat_model(**kwargs)
