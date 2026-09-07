# Docker：Milvus + PostgreSQL 基础设施

`docker-compose.yml` 只编排数据基础设施，共四个容器，全部位于同一个
桥接网络 `self_rag_net` 上，单机多容器由 docker compose 统一管理：

| 服务 | 职责 | 对外端口 |
| --- | --- | --- |
| `postgres` | LangGraph checkpoint / 短期记忆 | `5432` |
| `milvus` | Milvus standalone，向量检索 | `19530` |
| `etcd` | Milvus 元数据存储 | 容器内 |
| `minio` | Milvus 对象存储 | 容器内 |

容器之间通过服务名（`etcd:2379`、`minio:9000`）互相访问；只有
PostgreSQL 和 Milvus 暴露到本机，供在 PyCharm / 虚拟环境里本地运行的
RAG 应用连接。

RAG 应用**不在容器中运行**，因此：

```text
MILVUS_URL=http://localhost:19530
POSTGRES_URI=postgresql://self_rag:self_rag@localhost:5432/self_rag?sslmode=disable
```

保持 `.env` 里的 localhost 连接串即可，无需改成容器地址。

## 启动

```powershell
Copy-Item .env.example .env   # 首次，并填好 LLM / Embedding 密钥
docker compose up -d
docker compose ps             # 等待 postgres、etcd、minio、milvus 均 healthy
```

Milvus 依赖 etcd 和 MinIO，首次启动比 PostgreSQL 慢，属正常现象：

```powershell
docker compose logs -f milvus
```

## 本地运行 RAG 应用

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# 首次或知识库变更后，重建 collection：
python -m scripts.ingest --rebuild

uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

应用启动时会自动连接 PostgreSQL 并执行 `checkpointer.setup()`
（`app/rag_starter.py`），自动创建 LangGraph checkpoint 表。

## 常用命令

```powershell
# 查看各容器健康状态
docker compose ps

# 查看全部日志
docker compose logs -f

# 单独重启某个服务
docker compose restart milvus

# 停止容器，保留数据卷
docker compose down

# 停止并删除所有数据卷（清空向量库与会话，仅用于重置环境）
docker compose down -v
```

## 配置

默认凭据来自 `.env`，未设置时回退到 `.env.example` 里的开发默认值：

| 变量 | 默认 |
| --- | --- |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `self_rag` |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` |

这些默认值只适合本地开发，部署到非本地环境前必须替换。修改凭据后记得
同步更新 `.env` 中的 `POSTGRES_URI`。

## 故障排查

### PostgreSQL 连接被拒

确认 postgres 容器健康，且本机 5432 没有被其它进程占用：

```powershell
docker compose ps
netstat -ano | findstr :5432
```

### Milvus 长时间未 healthy

Milvus 要等 etcd 和 MinIO 先就绪：

```powershell
docker compose ps
docker compose logs --tail=200 milvus
```

### 重置环境

```powershell
docker compose down -v
docker compose up -d
python -m scripts.ingest --rebuild
```
