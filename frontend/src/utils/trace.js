// 把后端返回的执行轨迹(trace)整理成演示友好、易读的分步摘要。
// 目标是让非技术观看者也能大致看懂系统做了什么,而不是堆原始 JSON。

import { prettySource } from "./format.js";

const STEP_LABELS = {
  memory_prepare: "结合上下文与记忆",
  retrieve: "检索知识库",
  grade: "判断片段相关性",
  decide: "决定下一步",
  rewrite: "改写提问",
  web_search: "联网补充检索",
  generate: "生成回答",
  coverage: "未能回答",
};

const PARTITION_NAMES = {
  diagnosis: "诊断分区",
  device: "器械分区",
  medication: "用药分区",
};

function count(list) {
  if (!Array.isArray(list)) return 0;
  return list.length;
}

function short(text, max = 46) {
  const value = String(text ?? "").trim();
  if (!value) return "";
  return value.length > max ? `${value.slice(0, max)}…` : value;
}

function num(value) {
  if (typeof value !== "number") return value ?? "";
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function reasonText(text) {
  const value = String(text ?? "").trim();
  if (!value) return "";
  // 后端 reason 多为英文调试语,演示界面更希望看到中文结论。
  const map = {
    "documents are relevant; generating answer": "已有可用证据,直接生成回答",
    "documents are not relevant; rewriting query": "证据不足,改写提问后重新检索",
    "max retries reached; local documents irrelevant; degrading to web search once":
      "本地重试仍不足,降级为一次联网检索",
    "local documents irrelevant": "本地片段相关性不足",
    "local documents relevant": "本地片段可支撑回答",
  };
  return map[value] || value;
}

export function summarizeTrace(trace = []) {
  return (Array.isArray(trace) ? trace : []).map((step, index) => {
    const s = step || {};
    const summary = {
      label: STEP_LABELS[s.step] || s.step || "步骤",
      round: s.iteration ?? null,
      lines: [],
      key: `${index}-${s.step}-${s.iteration ?? 0}`,
    };

    switch (s.step) {
      case "memory_prepare":
        summary.lines.push(
          s.partition_id
            ? `知识分区:${PARTITION_NAMES[s.partition_id] || s.partition_id}`
            : "知识分区:自动判断",
        );
        if (s.is_follow_up) summary.lines.push("识别为追问,已带上文一起理解");
        if (s.long_term_hits && s.long_term_hits.length) {
          summary.lines.push(`调用了该用户 ${s.long_term_hits.length} 条历史记忆`);
        }
        break;

      case "retrieve":
        summary.lines.push(
          s.query ? `检索词:${short(s.query)}` : "检索词:与问题一致",
        );
        summary.lines.push(`本次召回 ${count(s.retrieved_sources)} 个片段`);
        break;

      case "grade":
        summary.lines.push(
          s.is_relevant
            ? `判断:存在可支撑片段(置信度 ${num(s.confidence)})`
            : "判断:证据不足",
        );
        summary.lines.push(
          `保留 ${count(s.accepted_sources)} 个 · 过滤 ${count(s.rejected_sources)} 个`,
        );
        break;

      case "decide":
        if (s.next_action) {
          summary.lines.push(
            s.next_action === "rewrite_node"
              ? "→ 重新检索"
              : s.next_action === "web_search_node"
                ? "→ 联网检索"
                : "→ 直接回答",
          );
        }
        if (s.reason) summary.lines.push(reasonText(s.reason));
        break;

      case "rewrite":
        summary.lines.push(
          s.rewritten_query
            ? `将提问改写为:「${short(s.rewritten_query, 56)}」`
            : "已改写提问",
        );
        break;

      case "web_search":
        if (s.configured === false) {
          summary.lines.push("未配置联网检索,已跳过");
        } else {
          summary.lines.push(
            count(s.retrieved_sources) > 0
              ? `联网命中 ${count(s.retrieved_sources)} 条公开信息`
              : "联网未命中可引用信息",
          );
        }
        break;

      case "generate":
        summary.lines.push(
          s.evidence_origin === "web" ? "依据:联网公开信息(仅供参考)" : "依据:本地知识库",
        );
        if (s.citations && s.citations.length) {
          summary.lines.push(`引用片段:${s.citations.join("、")}`);
        }
        break;

      case "coverage":
        summary.lines.push("本地知识库与联网公开信息均未能支撑回答");
        break;

      default:
        // 未知步骤退化为“字段:值”文本
        Object.entries(s).forEach(([key, value]) => {
          if (typeof value === "object") return;
          summary.lines.push(`${key}: ${String(value)}`);
        });
    }

    return summary;
  });
}

export function sourceTitle(source) {
  const parts = [];
  if (source && source.rank !== undefined && source.rank !== null) {
    parts.push(`#${source.rank}`);
  }
  const raw =
    (source && typeof source.source === "string" && source.source) ||
    (source && typeof source.chunk_id === "string" && source.chunk_id) ||
    "";
  if (/^https?:\/\//.test(raw)) {
    parts.push(prettySource(raw));
  } else if (source && source.chunk_id !== undefined && source.chunk_id !== null) {
    parts.push(`片段 ${source.chunk_id}`);
  }
  if (source && source.score !== undefined && source.score !== null) {
    parts.push(`相似度 ${formatScore(source.score)}`);
  }
  return parts.join(" · ") || "来源片段";
}

export function isUrl(value) {
  return typeof value === "string" && /^https?:\/\//.test(value);
}

function formatScore(value) {
  if (typeof value !== "number") return String(value ?? "");
  return value > 0 && value < 1 ? value.toFixed(4) : String(value);
}
