from __future__ import annotations

from app.schemas import RetrievedChunk

GRADE_SYSTEM_PROMPT = """
你是 RAG 检索质量评估器。你的任务是逐个判断检索片段是否能够支撑回答用户的原始问题。

判断标准：
1. 必须对每个 chunk_id 分别给出 chunk_grades 判断。
2. 只有当某个片段包含能直接回答问题的事实、规则、流程、限制或条件时，
   才把该片段判定为相关。
3. 仅关键词相似但无法支撑回答的片段，应判定为不相关。
4. 多个片段合起来能支撑回答时，把这些提供有效信息的片段分别判定为相关。
5. supporting_chunk_ids 只能包含被判定为相关的 chunk_id。
6. 顶层 is_relevant 表示是否至少存在一个可支撑回答的相关片段。
7. 不要补充片段中没有的信息。
8. 输出必须符合绑定的结构化字段。
9. 若提供「本会话之前的对话」，它只用于消解追问中的指代（如“它/上面/
   那种药”），不当作检索片段证据，也不应据此放行无关片段。
""".strip()


REWRITE_SYSTEM_PROMPT = """
你是 RAG 查询改写器。你的任务是在保持用户原始意图不变的前提下，
改写一个更适合向量检索的查询。

改写要求：
1. 保留原问题的业务意图和关键约束。
2. 补充可能出现在知识库中的同义词、业务词、规则词。
3. 避免生成答案，避免引入原问题没有的新需求。
4. 查询应简洁清晰，适合直接用于 embedding 检索。
5. 输出必须符合绑定的结构化字段。
6. 若提供「本会话之前的对话」，先据此消解“它/上面/那种药”等指代，
   再改写出独立、可检索的查询。
""".strip()


GENERATE_SYSTEM_PROMPT = """
你是基于知识库片段回答问题的客服助手。你只能使用提供的检索片段回答用户问题。

回答要求：
1. 优先给出直接答案，再补充必要条件、限制、流程或时效。
2. 如果片段不足以回答，明确说明“根据当前知识库片段无法确定”，并指出缺失信息。
3. 不要编造政策、金额、时间、次数或补偿规则。
4. 引用答案实际使用到的 chunk_id。
5. 输出必须符合绑定的结构化字段。
6. 若提供「本会话之前的对话」，它只用于理解追问中的指代（如“它/上面/
   那种药”），不得作为事实依据；回答事实一律以检索片段为准，冲突时以片段为准。
7.若提供「跨会话记忆」段落,它只用于延续用户偏好/上下文的背景提示,不是事实依据; 所有事实、金额、流程一律以「可用知识库片段」为准,冲突时以知识库为准; 不得把记忆内容作为引用来源(citations 只指向知识库 chunk_id)。
""".strip()


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "未检索到任何片段。"

    formatted: list[str] = []
    for chunk in chunks:
        score = f"{chunk.score:.4f}" if chunk.score is not None else "N/A"
        chunk_id = chunk.chunk_id if chunk.chunk_id is not None else "N/A"
        source = chunk.source or "N/A"
        lines = [
            (
                f"[片段 {chunk.rank}] chunk_id={chunk_id} "
                f"score={score} source={source}"
            )
        ]
        lines.extend(_format_metadata_lines(chunk))
        lines.extend(["正文：", chunk.text])
        formatted.append("\n".join(lines))
    return "\n\n".join(formatted)


def build_grade_messages(
    question: str,
    retrieval_query: str,
    chunks: list[RetrievedChunk],
    *,
    history_window: str = "",
) -> list[tuple[str, str]]:
    body = []
    history_section = _history_section(history_window)
    if history_section:
        body.extend([history_section, ""])
    body.extend(
        [
            f"用户原始问题：{question}",
            f"本轮检索查询：{retrieval_query}",
            "",
            "检索片段：",
            format_retrieved_chunks(chunks),
            "",
            "请逐个判断每个 chunk_id 是否足以支撑回答用户原始问题。",
        ]
    )
    return [
        ("system", GRADE_SYSTEM_PROMPT),
        ("human", "\n".join(body)),
    ]


def build_rewrite_messages(
    question: str,
    retrieval_query: str,
    chunks: list[RetrievedChunk],
    *,
    history_window: str = "",
) -> list[tuple[str, str]]:
    body = []
    history_section = _history_section(history_window)
    if history_section:
        body.extend([history_section, ""])
    body.extend(
        [
            f"用户原始问题：{question}",
            f"上一轮检索查询：{retrieval_query}",
            "",
            "上一轮低相关片段：",
            format_retrieved_chunks(chunks),
            "",
            "请输出下一轮用于检索的改写查询。",
        ]
    )
    return [
        ("system", REWRITE_SYSTEM_PROMPT),
        ("human", "\n".join(body)),
    ]


WEB_EVIDENCE_NOTICE = (
    "注意：以上片段来自联网公开检索，可能并非权威、过期或含噪音，只能作为参考答案依据，"
    "不能当作诊疗或政策性结论；请严格基于片段作答，片段不足时明确说明无法确定。"
)


def build_generate_messages(
    question,
    chunks,
    *,
    history_window="",
    memory_reference="",
    evidence_origin: str = "local",
):
    body = []
    history_section = _history_section(history_window)
    if history_section:
        body.extend([history_section, ""])
    if memory_reference:
        body.extend([memory_reference, ""])      # 放历史之后、证据之前
    body.extend([
        f"用户问题：{question}",
        "",
        "可用知识库片段：",
        format_retrieved_chunks(chunks),
        "",
    ])
    if evidence_origin == "web":
        body.append(WEB_EVIDENCE_NOTICE)
        body.append("")
    body.append("请基于这些片段生成最终回答。")
    return [("system", GENERATE_SYSTEM_PROMPT), ("human", "\n".join(body))]


def _history_section(history_window: str) -> str:
    if not history_window:
        return ""
    return (
        "本会话之前的对话（仅用于理解“它/上面/那种药”等指代，"
        "不是事实依据）：\n"
        f"{history_window}"
    )


def _format_metadata_lines(chunk: RetrievedChunk) -> list[str]:
    metadata = chunk.metadata
    if not metadata:
        return []

    lines: list[str] = []
    section_path = metadata.get("section_path_text") or metadata.get("section_path")
    block_type = metadata.get("block_type")
    context_before = metadata.get("context_before")
    context_after = metadata.get("context_after")

    if section_path:
        lines.append(f"章节路径：{section_path}")
    if block_type:
        lines.append(f"内容类型：{block_type}")
    if context_before:
        lines.append(f"前文上下文：{context_before}")
    if context_after:
        lines.append(f"后文上下文：{context_after}")

    return lines
