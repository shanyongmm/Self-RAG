// 把 LLM 返回的简单 Markdown 片段转成受控的 HTML。
// 安全策略:先整体 HTML 转义,再只允许「加粗 / 链接 / 标题 / 列表」四种结构,
// 其余一律按普通文本处理。

const ESCAPE_MAP = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ESCAPE_MAP[c]);
}

function linkify(text) {
  return text.replace(
    /(https?:\/\/[^\s<>"']+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>',
  );
}

// 行内处理:先加粗,再识别链接。入参必须是已转义文本。
function inline(text) {
  const bold = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return linkify(bold);
}

export function formatAnswer(rawText) {
  const lines = String(rawText ?? "").split(/\r?\n/);
  let html = "";
  let inList = false;

  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      continue;
    }

    const heading = line.match(/^#{1,4}\s+(.+)$/);
    if (heading) {
      closeList();
      html += `<h3>${inline(escapeHtml(heading[1]))}</h3>`;
      continue;
    }

    const bullet = line.match(/^[-*•]\s+(.+)$/);
    if (bullet) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${inline(escapeHtml(bullet[1]))}</li>`;
      continue;
    }

    closeList();
    html += `<p>${inline(escapeHtml(rawLine))}</p>`;
  }

  closeList();
  return html;
}

export function truncate(text, maxLength = 260) {
  const value = String(text ?? "");
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength)}…`;
}

export function formatNumber(value) {
  if (typeof value !== "number") return value ?? "";
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(4);
}

// 提取 URL 的可读短名(host + 首段路径),用于网络来源标题。
export function prettySource(text) {
  const value = String(text ?? "");
  try {
    const url = new URL(value);
    const host = url.hostname.replace(/^www\./, "");
    const firstPath = url.pathname.split("/").filter(Boolean)[0];
    return firstPath ? `${host}/${firstPath}` : host;
  } catch {
    return value;
  }
}
