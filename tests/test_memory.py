from langchain_core.messages import AIMessage, HumanMessage

from app.memory import (
    build_history_window,
    prepare_memory_context,
    update_memory_context,
)
from app.schemas import RetrievedChunk


def _turn(user: str, assistant: str) -> list[HumanMessage | AIMessage]:
    return [HumanMessage(content=user), AIMessage(content=assistant)]


def test_build_history_window_returns_empty_without_prior_turns() -> None:
    assert build_history_window(None) == ""
    assert build_history_window([]) == ""
    assert build_history_window([HumanMessage(content="当前问？")]) == ""


def test_build_history_window_excludes_current_turn() -> None:
    messages = [
        *_turn("阿司匹林能长期吃吗？", "需遵医嘱并监测出血风险。"),
        HumanMessage(content="那禁忌呢？"),
    ]
    window = build_history_window(messages)

    assert "阿司匹林能长期吃吗？" in window
    assert "出血风险" in window
    assert "那禁忌呢？" not in window
    assert window.startswith("用户：")
    assert "\n助手：" in window


def test_build_history_window_keeps_newest_turns_within_budget() -> None:
    messages: list[HumanMessage | AIMessage] = []
    for index in range(6):
        messages.extend(_turn(f"问题{index}", f"答案{index}"))
    messages.append(HumanMessage(content="当前"))

    window = build_history_window(messages, max_turns=2)
    user_lines = [
        line for line in window.splitlines() if line.startswith("用户：")
    ]

    assert len(user_lines) == 2
    assert "问题5" in window and "问题4" in window
    assert "问题0" not in window and "问题1" not in window


def test_prepare_memory_context_keeps_explicit_question_standalone() -> None:
    prepared = prepare_memory_context(
        question="糖尿病怎么诊断？",
        memory_context={
            "summary": "当前主题：高血压诊疗问题",
            "entities": ["高血压"],
            "partition_id": "diagnosis",
        },
    )

    assert prepared["is_follow_up"] is False
    assert prepared["contextual_question"] == "糖尿病怎么诊断？"
    assert prepared["partition_id"] == "diagnosis"


def test_prepare_memory_context_resolves_follow_up_with_previous_entities() -> None:
    previous = update_memory_context(
        {},
        question="阿司匹林有哪些用药注意事项？",
        contextual_question="阿司匹林有哪些用药注意事项？",
        partition_id="medication",
        documents=[
            RetrievedChunk(
                rank=1,
                text="阿司匹林用药注意事项。",
                chunk_id="chunk-a",
                source="data/阿司匹林指南.md",
                metadata={"section_title": "阿司匹林"},
            )
        ],
        rejected_documents=[
            RetrievedChunk(
                rank=2,
                text="无关内容。",
                chunk_id="chunk-b",
                source="data/其他.md",
            )
        ],
        answer_text="阿司匹林需要关注禁忌、不良反应等信息。",
    )

    prepared = prepare_memory_context(
        question="那禁忌呢？",
        memory_context=previous,
    )

    assert prepared["is_follow_up"] is True
    assert prepared["partition_id"] == "medication"
    assert "阿司匹林" in prepared["contextual_question"]
    assert "那禁忌呢？" in prepared["contextual_question"]


def test_prepare_memory_context_treats_named_attribute_question_as_new() -> None:
    prepared = prepare_memory_context(
        question="阿司匹林禁忌是什么？",
        memory_context={
            "summary": "当前主题：高血压诊疗问题",
            "entities": ["高血压"],
            "partition_id": "diagnosis",
        },
    )

    assert prepared["is_follow_up"] is False
    assert prepared["contextual_question"] == "阿司匹林禁忌是什么？"
    assert prepared["partition_id"] == "medication"

