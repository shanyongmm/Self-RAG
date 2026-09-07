from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from psycopg import OperationalError
from pydantic import BaseModel, Field

from app.auth import (
    ensure_auth_schema,
    is_valid_username,
    register_user,
    sign_token,
    verify_login,
    verify_token,
)
from app.config import get_config
from app.naive_rag import NaiveRagBaseline
from app.rag_starter import RagStarter
from fastapi import BackgroundTasks


INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"

# 若存在 Vite 构建产物(frontend/dist),则优先托管 Vue3 前端,否则回退到旧单页 Demo。
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
FRONTEND_DIST_INDEX = FRONTEND_DIST / "index.html"


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
        ensure_auth_schema(
            starter.config.postgres_uri,
            timeout=starter.config.postgres_connect_timeout,
        )
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


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=32, description="登录用户名")
    password: str = Field(..., min_length=6, max_length=128, description="登录密码")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=32, description="登录用户名")
    password: str = Field(..., min_length=1, max_length=128, description="登录密码")


class TokenResponse(BaseModel):
    token: str
    user_id: str
    username: str
    expires_at: str


class CurrentUser(BaseModel):
    user_id: str
    username: str


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


def get_optional_user(
    authorization: str | None = Header(default=None),
) -> CurrentUser | None:
    """Return the authenticated user, or None when no token is present.

    A present-but-invalid/expired token is rejected with 401 rather than
    silently falling back to the anonymous user.
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="登录凭证格式错误")
    claims = verify_token(parts[1].strip(), get_config().auth_token_secret)
    if claims is None:
        raise HTTPException(status_code=401, detail="登录凭证无效或已过期")
    return CurrentUser(user_id=claims["user_id"], username=claims["username"])


def _issue_token(user_id: str, username: str) -> TokenResponse:
    cfg = get_config()
    token = sign_token(
        user_id,
        username,
        cfg.auth_token_secret,
        cfg.auth_token_ttl_hours,
    )
    expires_at = datetime.now(timezone.utc) + timedelta(
        hours=cfg.auth_token_ttl_hours
    )
    return TokenResponse(
        token=token,
        user_id=user_id,
        username=username,
        expires_at=expires_at.isoformat(),
    )


@app.get("/")
async def index() -> FileResponse:
    # 有 Vue 构建产物时优先返回前端,否则回退到 app/static 下的旧单页 Demo。
    if FRONTEND_DIST_INDEX.exists():
        return FileResponse(FRONTEND_DIST_INDEX)
    return FileResponse(INDEX_HTML)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}



@app.post("/ask")
async def ask(
    request: AskRequest,
    background_tasks: BackgroundTasks,
    user: CurrentUser | None = Depends(get_optional_user),
) -> dict[str, Any]:
    try:
        result = get_starter().ask(
            question=request.question,
            thread_id=request.thread_id,
            user_id=user.user_id if user else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if _should_write_memory(result, user):
        run_id = request.thread_id or get_config().default_thread_id
        background_tasks.add_task(
            get_starter().enqueue_memory_write,
            user_id=user.user_id,
            run_id=run_id,
            question=request.question,
            answer=result.get("answer", ""),
        )
    return result


@app.post("/ask-naive")
async def ask_naive(request: AskRequest) -> dict[str, Any]:
    try:
        return get_naive_baseline().ask(question=request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/auth/register", status_code=201)
def register(payload: RegisterRequest) -> TokenResponse:
    username = payload.username.strip()
    if not is_valid_username(username):
        raise HTTPException(
            status_code=400,
            detail="用户名仅支持 1-32 位中英文、数字、下划线、短横线与点",
        )
    cfg = get_config()
    try:
        user_id = register_user(
            cfg.postgres_uri,
            username,
            payload.password,
            timeout=cfg.postgres_connect_timeout,
        )
    except OperationalError as exc:
        raise HTTPException(status_code=503, detail="身份服务暂不可用") from exc
    if user_id is None:
        raise HTTPException(status_code=409, detail="用户名已被占用")
    return _issue_token(user_id, username)


@app.post("/auth/login")
def login(payload: LoginRequest) -> TokenResponse:
    cfg = get_config()
    try:
        user_id = verify_login(
            cfg.postgres_uri,
            payload.username,
            payload.password,
            timeout=cfg.postgres_connect_timeout,
        )
    except OperationalError as exc:
        raise HTTPException(status_code=503, detail="身份服务暂不可用") from exc
    if user_id is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return _issue_token(user_id, payload.username.strip())


@app.get("/auth/me")
def me(user: CurrentUser | None = Depends(get_optional_user)) -> CurrentUser:
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    return user


@app.post("/auth/logout", status_code=204)
def logout() -> None:
    """Stateless logout: the client discards its token; nothing to revoke."""


def _should_write_memory(result: dict, user: CurrentUser | None) -> bool:
    if user is None:
        return False                       # 匿名不沉淀
    if not result.get("is_answerable"):
        return False                       # 没答上来的轮次不沉淀
    # 命中了知识库证据才值得沉淀;没有 KB 支撑的闲聊不写
    if not result.get("sources"):
        return False
    question = (result.get("question") or "").strip()
    if len(question) < 4:
        return False                       # “你好”“谢谢”这类过滤
    return True


# Vite 构建产物托管:api.py 里的业务路由先于此处注册,因此 /ask、/auth 等不受影响;
# 仅当 frontend/dist 已构建时才挂载 /assets 静态目录。
if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )
