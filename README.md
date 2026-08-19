# Self/Corrective RAG Agent

基于 LangGraph 和 LangChain 构建的自主迭代 RAG 系统。系统不会把首轮
Top-K 检索结果直接交给生成模型，而是先进行文档相关性评分和过滤；当证据
不足时，自动重写查询并重新检索，最终生成带引用来源和执行轨迹的答案。

## 项目亮点

- **条件循环检索**：Retrieve → Grade → Decide → Rewrite → Retrieve → Generate。
- **文档级纠错**：LLM 逐个判断 chunk 是否能支撑当前问题，只将相关 chunk
  交给生成模型。
- **可追溯回答**：返回引用 chunk、被过滤文档、最终检索查询和完整 Trace。
- **短期记忆**：使用 PostgreSQL 持久化 LangGraph checkpoint，并按
  `thread_id` 维护会话状态。
- **可量化评测**：内置 Naive RAG baseline、QA 数据集和离线评测脚本。

## 技术栈

| 层次 | 技术 |
| --- | --- |
| API | FastAPI、Uvicorn |
| Agent 编排 | LangGraph |
| RAG 组件 | LangChain |
| 向量检索 | Milvus、DashScope Embedding |
| 状态持久化 | PostgreSQL、LangGraph Checkpoint |
| 评测 | Python、Pytest、Ruff |

## 系统流程

```text
用户问题
   |
   v
Milvus 向量检索 Top-K
   |
   v
LLM 逐个评估 chunk 相关性
   |
   +--> 存在有效证据 --> 过滤无关 chunk --> 生成答案与引用
   |
   +--> 证据不足 --> 查询重写 --> 再次检索
                              |
                              +--> 最多重试 3 次
```

详细设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 目录结构

```text
Self_RAG/
├── app/                          # API、Agent 编排和运行时服务
│   ├── api.py                    # FastAPI 应用与接口
│   ├── graph.py                  # LangGraph 工作流
│   ├── rag_starter.py            # Self-RAG 服务封装
│   ├── naive_rag.py              # Naive RAG baseline
│   ├── checkpoint.py             # PostgreSQL checkpointer
│   ├── config.py                 # 环境变量和配置模型
│   ├── schemas.py                # 结构化输入输出模型
│   └── static/index.html         # Web Demo
├── rag/                          # 知识库加载、切分、Embedding 和 Milvus
├── scripts/                      # 导入知识库和初始化数据库脚本
├── evaluation/                   # QA 数据集、指标和评测入口
│   ├── datasets/qa_eval.jsonl
│   ├── results/best_report.json  # 可公开的代表性实验摘要
│   ├── metrics.py
│   └── run_eval.py
├── tests/                        # 单元测试
├── docs/ARCHITECTURE.md         # 架构说明
├── data/raw/                     # 示例知识库
├── Dockerfile                    # 应用镜像
├── docker-compose.yml            # 一键启动应用和基础设施
├── .dockerignore                 # Docker 构建上下文过滤
├── main.py                       # ASGI 入口
├── .env.example                  # 配置模板，不含真实密钥
├── requirements.txt              # 运行依赖
└── requirements-dev.txt          # 开发和测试依赖
```

## 环境要求

- Python 3.11+
- PostgreSQL 14+
- 可访问的 Milvus 服务
- 一个 OpenAI-compatible LLM API
- DashScope Embedding API
- Docker Desktop 或 Docker Engine + Compose Plugin

## Docker 一键部署

Docker 模式会启动以下服务：

```text
app -> PostgreSQL
    -> Milvus -> etcd
             -> MinIO
```

### 1. 准备配置

```powershell
Copy-Item .env.example .env
```

填写 `.env` 中的 `LLM_API_KEY`、`LLM_MODEL`、`EMBED_MODEL_NAME` 和
`DASHSCOPE_API_KEY`。Compose 会自动把容器内的 Milvus 地址和 PostgreSQL
地址改成服务名，不需要将 `localhost` 改成容器地址。

### 2. 构建并启动

```powershell
docker compose up -d --build
```

首次启动会自动执行：

1. 等待 PostgreSQL、etcd、MinIO 和 Milvus 健康。
2. 创建 LangGraph PostgreSQL checkpoint 表。
3. 检查 Milvus collection；不存在时自动导入示例知识库。
4. 启动 FastAPI 服务。

查看服务状态和应用日志：

```powershell
docker compose ps
docker compose logs -f app
```

启动后访问：

- Web Demo：<http://127.0.0.1:8001/>
- Swagger 文档：<http://127.0.0.1:8001/docs>
- 健康检查：<http://127.0.0.1:8001/health>

停止服务但保留数据：

```powershell
docker compose down
```

停止服务并删除 PostgreSQL、Milvus 和 MinIO 数据卷：

```powershell
docker compose down -v
```

该命令会删除容器内持久化数据，只建议在重新演示或重置环境时使用。

### 3. 更新知识库

修改 `data/raw/customer_service_knowledge_base.txt` 后，重新构建应用镜像
并重建 Milvus collection：

```powershell
docker compose build app
docker compose run --rm app python -m scripts.ingest --rebuild
docker compose up -d app
```

更多 Docker 说明见 [docs/DOCKER.md](docs/DOCKER.md)。

## 本地 Python 模式

如果不使用 Docker，也可以单独运行 Python 服务。此时需要手动准备
PostgreSQL 和 Milvus。

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 2. 创建本地配置

```powershell
Copy-Item .env.example .env
```

然后填写 `.env` 中的 LLM、DashScope、Milvus 和 PostgreSQL 配置。
`.env` 已被 `.gitignore` 忽略，真实密钥不要写入源码或 README。

### 3. 准备 PostgreSQL 和 Milvus

确保 PostgreSQL 和 Milvus 服务已经启动。也可以手动初始化 checkpoint 表：

```powershell
python -m scripts.setup_postgres_checkpoint
```

### 4. 导入知识库

首次运行或知识库内容发生变化时：

```powershell
python -m scripts.ingest --rebuild
```

### 5. 启动 API

```powershell
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

启动后访问：

- Web Demo：<http://127.0.0.1:8001/>
- Swagger 文档：<http://127.0.0.1:8001/docs>
- 健康检查：<http://127.0.0.1:8001/health>

## API 示例

Self/Corrective RAG：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8001/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"如何申请退款？","thread_id":"demo-user-1"}'
```

Naive RAG baseline：

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8001/ask-naive `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"如何申请退款？"}'
```

Self-RAG 响应中包含：

- `answer`：最终答案。
- `citations`：答案引用的 chunk_id。
- `sources`：最终送入生成模型的片段。
- `retrieved_sources`：原始检索结果。
- `rejected_sources`：被相关性评分过滤的片段。
- `retrieval_query`：最终使用的检索查询。
- `retry_count`：查询重写和再检索次数。
- `trace`：retrieve、grade、decide、rewrite、generate 的执行轨迹。

## 评测

运行最小验证：

```powershell
python -m evaluation.run_eval --mode both --limit 3
```

运行完整种子集：

```powershell
python -m evaluation.run_eval --mode both
```

当前仓库包含 20 条 QA 种子数据。评测输出的逐条明细和临时报告默认被
`.gitignore` 忽略；仓库只保留一份代表性摘要：

[evaluation/results/best_report.json](evaluation/results/best_report.json)

该实验在 20 条 QA 上的结果：

| 指标 | Naive RAG | Self/Corrective RAG |
| --- | ---: | ---: |
| 最终上下文 Precision | 33.33% | 87.50% |
| 无效检索率 | 66.67% | 12.50% |
| 答案准确率 | 75.00% | 85.00% |
| 引用准确率 | 85.00% | 87.50% |
| 估算 Token / 条 | 494.2 | 237.6 |
| 平均延迟 | 9.22s | 18.15s |

相较 Naive RAG，无效检索率相对下降 **81.25%**，最终上下文 Precision
提升 **54.17 个百分点**，答案准确率提升 **10 个百分点**，估算 Token
消耗下降约 **51.92%**。Self-RAG 额外引入了文档评分开销，平均延迟增加
约 **8.94 秒**，该权衡也保留在评测摘要中。

## 测试与代码检查

```powershell
pytest
ruff check .
```

## 安全说明

- 不要提交 `.env`、API Key、数据库密码或生产连接串。
- 当前本地 `.env` 如果曾经使用过真实密钥，公开仓库前应先在对应平台
  **撤销并重新生成密钥**。
- `evaluation/results/` 下的逐条 Trace 可能包含完整知识库片段，默认不提交；
  如需共享，请先确认数据脱敏和授权情况。

## 简历描述

基于 LangGraph 和 LangChain 构建自主迭代 Self/Corrective RAG Agent，实现
文档相关性评分、无效文档过滤、查询重写、多轮检索纠错和引用溯源；使用
Milvus 承载向量检索，PostgreSQL 持久化 LangGraph checkpoint，并实现
Naive RAG baseline 与自动化评测。在 20 条 QA 测试集上，无效检索率相对
下降 81.25%，最终上下文 Precision 提升 54.17 个百分点，答案准确率提升
10 个百分点。
