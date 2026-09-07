# Self-RAG 前端(Vue 3 + Vite)

面向「医疗知识库问答」演示的对话界面。默认走 **Self/Corrective RAG** 接口 `/ask`,
采用医疗浅色主题、演示友好版信息密度:答案直接展示,引用来源 / 被过滤片段 /
处理过程折叠在消息卡片里。

## 技术栈

- Vue 3(`<script setup>` 组合式 API)+ Vite 5
- 无 UI 框架、无运行时 CDN 依赖,图标为内置少量内联 SVG
- 构建产物(`dist/`)由 FastAPI 在同源托管

## 环境要求

- Node.js 18+(建议 20 LTS)
- 后端 FastAPI 已启动:`uvicorn main:app --port 8001`(后端目录下)

## 开发模式(推荐)

```bash
cd frontend
npm install
npm run dev
```

打开 <http://localhost:5173>。Vite 已把 `/ask`、`/auth`、`/health` 等路径代理到
`http://127.0.0.1:8001`,因此不需要处理 CORS。

如需把后端指向别处:

```bash
# Windows PowerShell
$env:VITE_BACKEND_TARGET = "http://你的后端地址:8001"
npm run dev
```

## 生产构建

```bash
npm run build      # 产物输出到 frontend/dist
```

后端 `app/api.py` 已做了兼容:启动时若检测到 `frontend/dist/index.html` 存在,`GET /`
将优先返回前端构建产物,并把 `/assets/*` 静态文件挂载到同一 FastAPI 进程。

于是生产部署就是:

```bash
cd frontend
npm install
npm run build
cd ..                      # 回到项目根目录
uvicorn main:app --host 0.0.0.0 --port 8001
```

访问 <http://127.0.0.1:8001/> 即为新界面;未构建前端时自动回退到旧的 `app/static/index.html`。

## 功能说明

- 聊天:Enter 发送 / Shift+Enter 换行;发送期间禁用输入,消息以“正在检索…”占位。
- 会话:默认带一个持久化 `thread_id`,支持多轮追问(后端用 LangGraph checkpoint 记忆上文)。
  “新对话”会生成新的 `thread_id` 并清空界面。
- 登录/注册:调用 `/auth/*`。登录后 `/ask` 带 Bearer token,后端按 `user_id` 隔离会话并沉淀 mem0 长期记忆。
  401 时前端会自动清除登录态并弹出登录框。
- 消息卡片:
  - 联网降级回答 / 未覆盖回答会显示顶部提示条(黄色 / 灰色)。
  - “引用来源 / 被过滤片段”折叠展示片段标题、摘要、相关度与原文链接。
  - “处理过程”以时间线形式把后端 trace 转成易懂的中文步骤(记忆/检索/评分/改写/联网/生成)。

## 目录结构

```text
frontend/
├── package.json
├── vite.config.js          # 代理到后端,构建配置
├── index.html
└── src/
    ├── main.js
    ├── assets/main.css     # 全局设计系统(医疗浅色主题)
    ├── api.js              # 后端请求封装 + 本地登录态/token 管理
    ├── App.vue             # 页面骨架与消息状态管理
    ├── utils/
    │   ├── format.js       # 安全 HTML 渲染/截断/数值/URL 短名
    │   └── trace.js        # trace → 中文分步摘要
    └── components/
        ├── AppIcon.vue     # 内联 SVG 图标
        ├── HeaderBar.vue   # 顶栏:品牌 / 新对话 / 登录态
        ├── AuthModal.vue   # 登录 / 注册弹窗
        ├── EmptyState.vue  # 首屏空状态 + 示例问题
        ├── ComposerBar.vue # 底部输入区
        ├── ChatMessage.vue # 消息气泡(答案/提示/来源/轨迹)
        ├── SourceFold.vue  # 折叠“引用来源 / 被过滤片段”
        └── TraceView.vue   # 折叠“处理过程”时间线
```

## 说明与后续可做

- 后端当前是“一次性返回 JSON”,前端因此没有做流式打字效果;如需逐字输出,
  需后端把 `/ask` 改造成 SSE / Stream 后再对接。
- 演示友好版刻意弱化了 Naive RAG 对比入口。需要时可在输入区上方加一个
  `Self-RAG / Naive` 切换,前端已预留 `api.askNaive()`。
