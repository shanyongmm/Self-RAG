// 与后端 FastAPI 交互的轻量封装。
// 开发模式下通过 Vite 代理访问后端(见 vite.config.js);
// 生产构建后由 FastAPI 在同一 origin 托管,base 路径保持一致。

const TOKEN_KEY = "self-rag-auth-token";
const USER_KEY = "self-rag-auth-user";

export function getAuth() {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    const username = localStorage.getItem(USER_KEY);
    return token && username ? { token, username } : null;
  } catch {
    return null;
  }
}

export function setAuth(token, username) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, username);
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function newThreadId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `t-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

async function request(path, { token, ...options } = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(path, { ...options, headers });
  let data = null;
  try {
    data = await response.json();
  } catch {
    // 非 JSON 错误体(如网关错误),保持 data = null
  }

  if (!response.ok) {
    const error = new Error((data && data.detail) || `请求失败(${response.status})`);
    error.status = response.status;
    throw error;
  }
  return data;
}

export function register(username, password) {
  return request("/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function login(username, password) {
  return request("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function askSelfRag(question, threadId, auth) {
  return request("/ask", {
    token: auth && auth.token,
    method: "POST",
    body: JSON.stringify({ question, thread_id: threadId }),
  });
}

export function askNaive(question) {
  return request("/ask-naive", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}
