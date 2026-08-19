from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.naive_rag import NaiveRagBaseline
from app.rag_starter import RagStarter

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    starter = RagStarter()
    app.state.rag_starter = starter
    app.state.naive_baseline = NaiveRagBaseline(
        config=starter.config,
        vectorstore=starter.vectorstore,
        llm=starter.llm,
    )
    try:
        yield
    finally:
        starter.close()


app = FastAPI(
    title="Self/Corrective RAG Agent",
    description=(
        "An iterative RAG service with document grading, "
        "query rewriting, and tracing."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    thread_id: str | None = Field(
        default=None,
        min_length=1,
        description="Conversation identifier used by PostgreSQL checkpointing",
    )


def get_starter() -> RagStarter:
    starter = getattr(app.state, "rag_starter", None)
    if starter is None:
        starter = RagStarter()
        app.state.rag_starter = starter
    return starter


def get_naive_baseline() -> NaiveRagBaseline:
    baseline = getattr(app.state, "naive_baseline", None)
    if baseline is None:
        starter = get_starter()
        baseline = NaiveRagBaseline(
            config=starter.config,
            vectorstore=starter.vectorstore,
            llm=starter.llm,
        )
        app.state.naive_baseline = baseline
    return baseline


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    try:
        return get_starter().ask(
            question=request.question,
            thread_id=request.thread_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask-naive")
async def ask_naive(request: AskRequest) -> dict[str, Any]:
    try:
        return get_naive_baseline().ask(question=request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
