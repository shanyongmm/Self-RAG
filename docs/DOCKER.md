# Docker Deployment

## Services

`docker-compose.yml` defines five services:

| Service | Responsibility | Exposed port |
| --- | --- | --- |
| `app` | FastAPI and Self/Corrective RAG Agent | `8001` |
| `postgres` | LangGraph checkpoint and short-term memory | internal |
| `milvus` | Vector search | `19530`, `9091` |
| `etcd` | Milvus metadata storage | internal |
| `minio` | Milvus object storage | internal |

The application connects to service names inside the Compose network:

```text
MILVUS_URL=http://milvus:19530
POSTGRES_URI=postgresql://...@postgres:5432/self_rag
```

These values are injected by Compose and override local `localhost` values from
`.env`.

## First Run

```powershell
Copy-Item .env.example .env
# Fill in the LLM and embedding credentials in .env.
docker compose up -d --build
docker compose ps
docker compose logs -f app
```

The application image runs `python -m scripts.bootstrap` before Uvicorn starts.
The bootstrap process:

1. Retries PostgreSQL checkpoint setup until the database is ready.
2. Checks whether the configured Milvus collection exists.
3. Ingests the knowledge base only when the collection is missing.
4. Starts the API process.

This makes container restarts idempotent and avoids rebuilding the collection
on every restart.

## Useful Commands

```powershell
# Rebuild only the application image
docker compose build app

# Follow all service logs
docker compose logs -f

# Run a shell command inside the application image
docker compose run --rm app python -m evaluation.run_eval --help

# Rebuild the vector collection after changing the knowledge base
docker compose run --rm app python -m scripts.ingest --rebuild

# Stop containers and keep named volumes
docker compose down

# Stop containers and remove all named volumes
docker compose down -v
```

## Configuration

`AUTO_INGEST=true` is the default. It performs the first knowledge-base import
when the Milvus collection does not exist. Set it to `false` when you want to
start the API without importing data:

```env
AUTO_INGEST=false
```

The default PostgreSQL and MinIO credentials in `.env.example` are intended only
for local demonstrations. Replace them before deploying outside a local
interview/demo environment.

## Troubleshooting

### The app keeps restarting

Check the application log:

```powershell
docker compose logs --tail=200 app
```

Typical causes are missing LLM/embedding keys or an unavailable external model
endpoint.

### Milvus is still starting

Milvus depends on etcd and MinIO and can take longer than PostgreSQL on the
first run. Check all health states:

```powershell
docker compose ps
docker compose logs --tail=200 milvus
```

### Reset everything

Use the following only when you want to discard all local vector and checkpoint
data:

```powershell
docker compose down -v
docker compose up -d --build
```
