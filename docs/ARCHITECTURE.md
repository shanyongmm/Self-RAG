# Architecture

## Overview

The project implements a corrective RAG workflow instead of sending the first
retrieval result directly to the generation model.

```text
User question
      |
      v
Retrieve from Milvus
      |
      v
Grade each chunk with an LLM
      |
      +--> Relevant chunks --> Generate answer with citations
      |
      +--> Low relevance --> Rewrite query --> Retrieve again
                               |
                               +--> Maximum retry limit: 3
```

## Runtime Components

### API layer

`app/api.py` exposes:

- `GET /`: lightweight web demo.
- `GET /health`: health check.
- `POST /ask`: Self/Corrective RAG.
- `POST /ask-naive`: one-shot Naive RAG baseline.

### Orchestration layer

`app/graph.py` defines the LangGraph state machine:

1. `retrieve_node` searches the Milvus collection.
2. `grade_node` evaluates every retrieved chunk.
3. `decide_node` selects generation or query rewriting.
4. `rewrite_node` creates a retrieval-oriented query.
5. `generate_node` produces a structured answer and citations.

### Persistence layer

PostgreSQL stores LangGraph checkpoints keyed by `thread_id`. This allows the
agent to keep short-term conversation state across requests.

### Evaluation layer

`evaluation/run_eval.py` runs both pipelines against the same JSONL dataset.
The metrics distinguish the initial retrieval quality from the final filtered
context quality, which makes the effect of document grading observable.

## Data Flow

1. `scripts/ingest.py` loads the text knowledge base.
2. LangChain splits it into overlapping chunks.
3. DashScope embeddings are written to Milvus.
4. The API receives a question.
5. LangGraph retrieves and grades candidate chunks.
6. Only accepted chunks are passed to the generation model.
7. The response includes answer, citations, sources, rejected sources, and trace.

## Design Trade-offs

The corrective workflow improves context quality but adds LLM grading overhead.
The checked-in evaluation summary records both quality metrics and estimated
token/latency changes so the trade-off is explicit.
