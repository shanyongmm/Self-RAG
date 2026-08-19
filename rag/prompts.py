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
""".strip()


GENERATE_SYSTEM_PROMPT = """
你是基于知识库片段回答问题的客服助手。你只能使用提供的检索片段回答用户问题。

回答要求：
1. 优先给出直接答案，再补充必要条件、限制、流程或时效。
2. 如果片段不足以回答，明确说明“根据当前知识库片段无法确定”，并指出缺失信息。
3. 不要编造政策、金额、时间、次数或补偿规则。
4. 引用答案实际使用到的 chunk_id。
5. 输出必须符合绑定的结构化字段。
""".strip()


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "未检索到任何片段。"

    formatted: list[str] = []
    for chunk in chunks:
        score = f"{chunk.score:.4f}" if chunk.score is not None else "N/A"
        chunk_id = chunk.chunk_id if chunk.chunk_id is not None else "N/A"
        source = chunk.source or "N/A"
        formatted.append(
            "\n".join(
                [
                    (
                        f"[片段 {chunk.rank}] chunk_id={chunk_id} "
                        f"score={score} source={source}"
                    ),
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(formatted)


def build_grade_messages(
    question: str,
    retrieval_query: str,
    chunks: list[RetrievedChunk],
) -> list[tuple[str, str]]:
    return [
        ("system", GRADE_SYSTEM_PROMPT),
        (
            "human",
            "\n".join(
                [
                    f"用户原始问题：{question}",
                    f"本轮检索查询：{retrieval_query}",
                    "",
                    "检索片段：",
                    format_retrieved_chunks(chunks),
                    "",
                    "请逐个判断每个 chunk_id 是否足以支撑回答用户原始问题。",
                ]
            ),
        ),
    ]


def build_rewrite_messages(
    question: str,
    retrieval_query: str,
    chunks: list[RetrievedChunk],
) -> list[tuple[str, str]]:
    return [
        ("system", REWRITE_SYSTEM_PROMPT),
        (
            "human",
            "\n".join(
                [
                    f"用户原始问题：{question}",
                    f"上一轮检索查询：{retrieval_query}",
                    "",
                    "上一轮低相关片段：",
                    format_retrieved_chunks(chunks),
                    "",
                    "请输出下一轮用于检索的改写查询。",
                ]
            ),
        ),
    ]


def build_generate_messages(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[tuple[str, str]]:
    return [
        ("system", GENERATE_SYSTEM_PROMPT),
        (
            "human",
            "\n".join(
                [
                    f"用户问题：{question}",
                    "",
                    "可用知识库片段：",
                    format_retrieved_chunks(chunks),
                    "",
                    "请基于这些片段生成最终回答。",
                ]
            ),
        ),
    ]
