# Self/Corrective RAG Agent —— 医疗知识库智能问答

基于 LangGraph / LangChain 构建的 **自主迭代 RAG 问答服务**。系统不会把
首轮 Top-K 检索结果直接交给生成模型，而是先由 LLM 对每个片段做相关性
评分与过滤；当本地证据不足时，自动**重写提问**并再次检索，必要时降级到
**可信联网检索**，最终返回**带引用来源与完整执行轨迹**的答案。

知识库聚焦医疗领域：**临床诊疗指南 / 医疗器械监管 / 国家基本药物目录**，
数据经 MinerU 解析、按分区入库，配合可选的多用户登录与长期记忆。

> ⚠️ 项目定位为技术演示 / 简历项目，回答仅供流程演示，**不构成医疗建议**。

---

## 项目演示

> 🎬 **此处预留「项目演示过程」的位置。**
> 将下面的占位块替换成你的真实演示内容即可，例如：运行截图、演示 GIF、
> 录屏 / 视频链接，以及 3–5 步的演示脚本。完成后可整段删除本引用说明。

```text
【演示材料占位 —— 建议放：】
1) 整体界面截图（首页 / 提问后答案 + 来源卡片）
2) 关键流程动图（证据不足 → 自动改写重试，或 证据充分 → 过滤无关片段）
3) 可选：演示视频 / 录屏链接
4) 演示脚本：按顺序点击/输入什么、预期看到什么

【推荐演示问题（覆盖三个分区，可直接提问）：】
- 诊断分区：高血压患者血压一般控制在什么范围？
- 诊断分区：糖尿病患者的转诊标准是什么？
- 用药分区：国家基本药物目录包含哪些高血压常用药？
- 器械分区：医疗器械唯一标识（UDI）有什么作用？
- 用药分区：阿莫西林常见的用药禁忌有哪些？

【进阶演示点（可选）】
- 知识库外提问（如某个 2026 年新政策）→ 观察「改写重试 → 联网检索降级」路径；
- 登录后再提问 → 观察 thread 按用户隔离、响应里的 memory_reference / long_term_hits。
```

**素材占位（建议保留到替换完成再删）**：

1. 主界面截图（提问前 / 提问后答案 + 来源卡片各一张）
2. 关键流程 GIF：证据不足 → 自动改写重试 / 证据充分 → 过滤无关片段
3. 演示录屏链接（可选）

> 📌 建议把截图 / GIF 放入 `docs/demo/` 目录，然后用 `![说明](docs/demo/xxx.png)`
> 的写法插入本小节，替换上面「素材占位」即可。演示完成后，可把本小节收成一段
> 简短的图文说明，或直接删除。

<!-- 项目演示占位区：上方「🎬 此处预留」到本注释之间的内容均为占位模板，请用你的真实演示过程覆盖。 -->

## 项目亮点

- **条件循环检索**：Retrieve → Grade → Decide → Rewrite → Retrieve → Generate，
  最多重试 `RAG_MAX_RETRIES` 次，证据仍不足则进入兜底路径。
- **文档级纠错**：LLM 逐 chunk 判断能否支撑当前问题，只把相关片段交给生成模型，
  并返回被过滤片段（`rejected_sources`）以便追溯。
- **可追溯回答**：返回引用片段、最终检索查询、重写次数与完整 Trace
  （retrieve / grade / decide / rewrite / generate / web_search）。
- **分区知识库**：`knowledge_partitions.json` 划分 diagnosis / device / medication
  三区，提问先按关键词路由到分区，重写时逐步放开分区范围，提升召回精度。
- **MinerU 文档解析**：PDF 走 MinerU 服务转 Markdown（含表格/版式还原）后再入库，
  管线见 `docs/MINERU.md`。
- **多用户与鉴权（可选）**：JWT register / login / me，`thread_id` 按
  `user_id` 隔离；匿名可退化回旧版单用户。
- **长期记忆（可选）**：接入 mem0，按用户沉淀跨会话记忆（sqlite history +
  Milvus 独立 collection），后台异步写入。
- **联网检索降级（可选）**：本地库判为不相关且重写耗尽后，用 Tavily 联网检索一次；
  医疗场景可用 `TAVILY_ALLOWED_DOMAINS` 限定药监 / 卫健委等可信域名。
- **可量化评测**：内置 Naive RAG baseline、20 条医疗 QA 种子集与离线评测脚本。

## 技术栈

| 层次 | 技术 |
| --- | --- |
| API / Web | FastAPI、Uvicorn；演示前端 Vite + Vue 3 |
| Agent 编排 | LangGraph（PostgreSQL checkpoint） |
| RAG 组件 | LangChain、Self/Corrective RAG（grade + rewrite 循环） |
| 向量检索 | Milvus、DashScope Embedding |
| 文档解析 | MinerU（PDF → Markdown） |
| 长期记忆 | Mem0（sqlite history + Milvus collection） |
| 联网降级 | Tavily Web Search（可选域名白名单） |
| 用户鉴权 | JWT（可选，`python-jose` / passlib） |
| LLM | DeepSeek（OpenAI 兼容接口，`rag/llm.py` 统一封装） |
| 评测 / 质量 | Python、Pytest、Ruff、离线 QA 评测脚本 |

## 系统流程

```text
用户提问
  │
  ▼
┌─ memory_node ────────────────────────────────────────────────┐
│  • 会话短期记忆（PostgreSQL checkpoint，按 thread_id）         │
│  • 长期记忆召回（mem0，可选，按 user_id 隔离）                 │
│  • 分区路由（关键词 → diagnosis / device / medication）        │
└───────────────────────────┬───────────────────────────────────┘
                            ▼
                  Milvus 分区向量检索 Top-K
                            ▼
                LLM 逐 chunk 相关性评分（grade）
                            ▼
                          decide
          ┌────────────────┴────────────────┐
       证据充分                          证据不足
          │                                │
          ▼                                ▼
   generate（带引用，            rewrite 重写提问 → 重新检索
   返回 answer + citations）          （≤ RAG_MAX_RETRIES 次，逐次放开分区）
                                              │ 重试耗尽
                                              ▼
                               web_search 联网检索降级（可选）
                                              │ 仍无证据
                                              ▼
                              coverage_node 明确拒答（不编造）
```

核心链路入口在 `app/graph.py`，封装在 `app/rag_starter.py`；详细设计见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 与 [docs/RAG_PIPELINE.md](docs/RAG_PIPELINE.md)。

## 目录结构

```text
Self_RAG/
├── app/                            # FastAPI 应用与 Agent 运行时
│   ├── api.py                      # 接口 + 静态托管（优先 frontend/dist，回退 static/）
│   ├── graph.py                    # LangGraph 状态图（核心编排）
│   ├── rag_starter.py              # Self-RAG 服务封装、checkpoint 初始化
│   ├── naive_rag.py                # Naive RAG baseline（评测对照）
│   ├── memory.py                   # 会话/长期记忆编排（mem0 调用点）
│   ├── auth.py                     # 可选 JWT 鉴权
│   ├── checkpoint.py               # PostgreSQL checkpointer
│   ├── schemas.py                  # Pydantic 请求/响应模型
│   ├── config.py                   # 环境变量与配置
│   └── static/index.html           # 旧版单文件 Demo（无前端构建产物时回退）
├── rag/                            # 知识库、检索与外部服务
│   ├── partitions.py               # 分区注册表 + 关键词路由
│   ├── mineru_service.py           # MinerU PDF → Markdown 客户端
│   ├── mem0_service.py             # 长期记忆服务
│   ├── web_search.py               # Tavily 联网检索（降级路径）
│   ├── loader.py / splitter.py     # 文档加载与切分
│   ├── vectorstore.py              # Milvus 向量库
│   ├── embeddings.py               # DashScope Embedding
│   ├── llm.py / prompts.py         # LLM 封装 / 各节点 prompt
├── scripts/
│   ├── parse_documents.py          # 调 MinerU 批量解析 source_documents → data/parsed
│   ├── ingest_partitioned.py       # 分区知识库入库（--rebuild 重建）
│   └── ingest.py                   # 平铺知识库入库（关闭分区时使用）
├── frontend/                       # 演示前端（Vite + Vue 3）
│   ├── src/App.vue                 # 主界面（消息流 / 输入 / 登录态）
│   ├── src/components/             # HeaderBar、ChatMessage、SourceFold、TraceView、
│   │                               # AuthModal、ComposerBar、EmptyState、AppIcon
│   ├── src/api.js                  # 后端 API 封装（token / thread_id 维护）
│   ├── src/utils/                  # 文本格式化、trace 可视化摘要
│   └── src/assets/main.css         # 医疗浅色设计系统（CSS 变量）
├── evaluation/
│   ├── datasets/
│   │   ├── qa_eval_medical.jsonl        # 旧医疗 QA 种子集（expected id 已失效，弃用）
│   │   ├── clinical_qa_seed.jsonl       # 临床对比评测种子集（gold_passage 锚定）
│   │   └── qa_eval_clinical.jsonl       # build_qa_dataset.py 解析产物（可判分）
│   ├── metrics.py / run_eval.py         # 指标与评测入口（含成本-效果）
│   ├── build_qa_dataset.py              # gold_passage → 当前 chunk_id 解析器
│   └── results/best_report.json         # 代表性实验摘要（逐条结果默认 gitignore）
├── tests/                          # test_metrics / test_partitions / test_memory / test_auth
├── docs/                           # 详见下方「文档索引」
├── data/parsed/mineru_partitions/  # MinerU 解析产物（按分区）
├── source_documents/<partition>/   # 原始文档放置目录（本地按需提供）
├── knowledge_partitions.json       # 分区配置（diagnosis / device / medication）
├── PARTITION_GUIDE.md              # 分区模式使用指南
├── Dockerfile / docker-compose.yml # 应用镜像 / 仅基础设施（Milvus + PostgreSQL）
├── main.py                         # ASGI 入口
├── .env.example                    # 配置模板（不含真实密钥）
└── requirements.txt / requirements-dev.txt
```

## 环境要求

- Python 3.11+
- PostgreSQL 14+、可访问的 Milvus 服务
- 一个 OpenAI-compatible LLM API（默认 DeepSeek）与 DashScope Embedding API
- Docker Desktop / Docker Engine + Compose Plugin（仅用于一键启动基础设施）
- Node.js 18+（仅开发 / 构建 Vue 前端时需要）

## Docker：启动基础设施（Milvus + PostgreSQL）

Docker 只托管数据基础设施，RAG 应用在本机（PyCharm / 虚拟环境）运行：

```text
self_rag_net（同一桥接网络）
├── postgres  -> :5432（映射到本机）
├── etcd      （Milvus 元数据，容器内）
├── minio     （Milvus 对象存储，容器内）
└── milvus    -> :19530（映射到本机）
```

```powershell
Copy-Item .env.example .env   # 先填 LLM / DashScope 密钥
docker compose up -d
docker compose ps             # 等待 postgres 与 milvus 均 healthy
```

启动后本机即可访问 Milvus `localhost:19530` 与 PostgreSQL `localhost:5432`。
停止保留数据卷用 `docker compose down`；清空向量库与会话数据用
`docker compose down -v`（仅建议重置演示环境时执行）。详见 [docs/DOCKER.md](docs/DOCKER.md)。

## 知识库：源文档 → MinerU 解析 → 分区入库

默认开启分区模式（`.env` 中 `PARTITION_CONFIG=knowledge_partitions.json`）。
原始 PDF 按分区放在 `source_documents/<partition_id>/` 下，例如：

```text
source_documents/
├── diagnosis/hypertension_guideline_2025.pdf
├── diagnosis/diabetes_guideline_2022.pdf
├── device/medical_device_classification_adjustment_2026_53.doc
└── medication/national_essential_medicines_catalog_2026.pdf
```

1. 用 MinerU 把 PDF 批量转成 Markdown（含表格/版式还原，需要 `MINERU_BASE_URL` 与联网）：

   ```powershell
   python -m scripts.parse_documents
   ```

   产物写入 `data/parsed/mineru_partitions/<partition_id>/...`。

2. 解析产物入库到 Milvus（分区模式）：

   ```powershell
   python -m scripts.ingest_partitioned --rebuild
   ```

   关闭分区模式（`PARTITION_CONFIG` 置空 / `none`）后，沿用平铺模式
   `python -m scripts.ingest --rebuild`。

分区与入库细节见 [PARTITION_GUIDE.md](docs/PARTITION_GUIDE.md) 与 [docs/MINERU.md](docs/MINERU.md)。

## 本地 Python 模式：启动后端

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1           # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

Copy-Item .env.example .env            # 填写 LLM / DashScope / Milvus / PG 配置
docker compose up -d                   # 启动基础设施
python -m scripts.ingest_partitioned --rebuild   # 首次或换库后导入

uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

启动后访问：

- Web Demo：<http://127.0.0.1:8001/>
- Swagger 文档：<http://127.0.0.1:8001/docs>
- 健康检查：<http://127.0.0.1:8001/health>

## 前端：Vue 3 演示界面（开发 / 构建）

仓库根目录下的 `frontend/` 是一个独立的 Vite + Vue 3 工程。

**开发模式**（热更新，独立于 8001 端口）：

```powershell
cd frontend
npm install
npm run dev        # 默认 http://localhost:5173，/ask、/auth 等代理到 8001
```

**构建产物模式**（推荐用于演示 / 打包）：`api.py` 检测到
`frontend/dist/index.html` 存在时，会直接由 FastAPI 在 8001 端口托管前端，
无需另开端口：

```powershell
cd frontend
npm run build      # 生成 frontend/dist
cd ..
uvicorn main:app --host 127.0.0.1 --port 8001   # 访问 http://127.0.0.1:8001/
```

界面为**医疗浅色主题 + 演示友好版**：答案直接展示，引用来源、被过滤片段与
处理过程折叠在卡片里；右上角支持登录/新对话。前端工程说明见
[frontend/README.md](frontend/README.md)。

## API 示例

Self/Corrective RAG（医疗知识库提问）：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8001/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"高血压患者血压一般控制在什么范围?","thread_id":"demo-1"}'
```

Naive RAG baseline（对照）：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8001/ask-naive `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"高血压患者血压一般控制在什么范围?"}'
```

用户注册 / 登录（可选鉴权，开启后按用户隔离会话与记忆）：

```powershell
# 注册（开放自助注册；重名返回 409）
Invoke-RestMethod -Uri http://127.0.0.1:8001/auth/register -Method Post `
  -ContentType "application/json" -Body '{"username":"alice","password":"secret123"}'

# 登录（返回 token / user_id）
$auth = Invoke-RestMethod -Uri http://127.0.0.1:8001/auth/login -Method Post `
  -ContentType "application/json" -Body '{"username":"alice","password":"secret123"}'

# 带 Bearer token 的 /ask：thread 自动拼为 user_id::thread_id，互不可见
Invoke-RestMethod -Uri http://127.0.0.1:8001/ask -Method Post `
  -Headers @{Authorization = "Bearer $($auth.token)"} `
  -ContentType "application/json" -Body '{"question":"阿莫西林常见的用药禁忌有哪些?","thread_id":"demo-1"}'
```

鉴权规则：

- **可选鉴权**：不带 token → 匿名默认单用户（与旧版一致）；带 token → 按用户隔离。
- token **无效 / 过期 → 401**；登录态 `/ask` 响应带 `user_id`。
- `GET /auth/me`（带 token）返回当前用户；`POST /auth/logout` 无状态 204
  （登出 = 客户端丢弃 token）。
- `AUTH_TOKEN_SECRET` 留空则用进程内随机密钥（重启后已签发 token 失效），生产请设固定值。

### Self-RAG 响应字段

`/ask` 响应中与检索质量相关的字段：

- `answer`：最终答案；`is_answerable` / `answer_source`（local / web / uncovered）。
- `citations` / `sources`：答案引用的 chunk_id 与最终送入生成模型的片段。
- `retrieved_sources` / `rejected_sources`：原始检索结果 / 被相关性评分过滤的片段。
- `retrieval_query`：最终使用的检索查询；`retry_count`：重写再检索次数。
- `partition_id`：命中的知识分区（diagnosis / device / medication）。
- `evidence_origin` / `uncovered`：证据来源与拒答明细。
- `memory_reference` / `long_term_hits`：长期记忆（mem0）召回情况，开启后可见。
- `trace`：retrieve、grade、decide、rewrite、web_search、generate 的执行轨迹。
- `thread_id` / `user_id`：会话与用户标识。

## 评测

当前知识库（5 篇分区临床文档）的 **可复现对比评测** 见
[docs/EVAL_CLINICAL.md](docs/EVAL_CLINICAL.md)：种子集用逐字 `gold_passages`
锚定答案出处，`evaluation/build_qa_dataset.py` 以与入库相同的切分逻辑离线把
金句映射到**当前** chunk_id（任一句对不上即报错，防止旧版“expected id 失效
导致 Self-RAG 分数失真”），并以 **成本-效果** 为亮点指标
（每正确回答 token 降幅 / 生成上下文 token 降幅）。

最小验证（3 条）或完整临床种子集（14 条，两种 pipeline）：

```powershell
python -m evaluation.build_qa_dataset
python -m evaluation.run_eval --mode both --limit 3
python -m evaluation.run_eval --mode both
```

> 说明：下方表格来自旧客服库数据集（customer-service `qa_eval.jsonl`），仅作
> 历史参考，不代表当前临床 KB 的水平；临床 KB 请以上述 `qa_eval_clinical`
> 新流程跑出的报告为准。

评测同时跑 Naive RAG baseline 与 Self-RAG。逐条明细和临时报告默认被
`.gitignore` 忽略，仓库只保留一份代表性摘要
[evaluation/results/best_report.json](evaluation/results/best_report.json)：

| 指标 | Naive RAG | Self/Corrective RAG |
| --- | ---: | ---: |
| 最终上下文 Precision | 33.33% | 87.50% |
| 无效检索率 | 66.67% | 12.50% |
| 答案准确率 | 75.00% | 85.00% |
| 引用准确率 | 85.00% | 87.50% |
| 估算 Token / 条 | 494.2 | 237.6 |
| 平均延迟 | 9.22s | 18.15s |

相较 Naive RAG：无效检索率相对下降 **81.25%**，最终上下文 Precision 提升
**54.17 个百分点**，答案准确率提升 **10 个百分点**，估算 Token 消耗下降约
**51.92%**；Self-RAG 因额外文档评分开销平均延迟增加约 **8.94 秒**，该权衡也
保留在评测摘要中。

## 测试与代码检查

```powershell
pytest
ruff check .
```

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 系统架构总览 |
| [docs/RAG_PIPELINE.md](docs/RAG_PIPELINE.md) | RAG 各节点与数据流细节 |
| [docs/EVAL_CLINICAL.md](docs/EVAL_CLINICAL.md) | 临床 KB 的 Naive vs Self-RAG 对比评测（含成本-效果口径） |
| [docs/MINERU.md](docs/MINERU.md) | MinerU 解析管线接入 |
| [PARTITION_GUIDE.md](docs/PARTITION_GUIDE.md) | 分区知识库使用指南 |
| [docs/MEM0_TUTORIAL.md](docs/MEM0_TUTORIAL.md) | mem0 长期记忆接入 |
| [docs/MEMORY_UPGRADE_PLAN.md](docs/MEMORY_UPGRADE_PLAN.md) | 记忆层升级路线图 |
| [docs/DOCKER.md](docs/DOCKER.md) | Docker 基础设施说明 |
| [frontend/README.md](frontend/README.md) | Vue 3 前端开发说明 |

## 安全说明

- 不要提交 `.env`、API Key、数据库密码或生产连接串。
- 本地 `.env` 若曾使用真实密钥，公开仓库前应在对应平台**撤销并重新生成密钥**。
- `evaluation/results/` 下的逐条 Trace 可能包含完整知识库片段，默认不提交；
  如需共享请先确认数据脱敏与授权。
- 本项目面向技术演示，医疗内容输出不构成诊疗建议。

## 简历描述

基于 LangGraph / LangChain 构建自主迭代 Self/Corrective RAG Agent，实现文档
相关性评分、无效文档过滤、查询重写与多轮检索纠错，并支持分区知识库、MinerU
文档解析、联网检索降级、多用户鉴权与 mem0 长期记忆；使用 Milvus 承载向量检索、
PostgreSQL 持久化 LangGraph checkpoint，配套 Vue 3 演示前端与 Naive RAG
baseline 自动化评测。在 20 条医疗 QA 测试集上，无效检索率相对下降 81.25%，
最终上下文 Precision 提升 54.17 个百分点，答案准确率提升 10 个百分点，
估算 Token 消耗下降约 51.92%。
