from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
import time
import uuid
from typing import Any

from psycopg import Connection, OperationalError
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row

PBKDF2_ITERATIONS = 210_000
USERNAME_PATTERN = re.compile(r"^[\w一-鿿.-]{1,32}$")

_EPHEMERAL_SECRET: str | None = None


def scoped_thread_id(user_id: str | None, thread_id: str) -> str:
    """Namespace a LangGraph thread under a user to isolate conversations."""
    if not user_id:
        return thread_id
    return f"{user_id}::{thread_id}"


def _effective_secret(secret: str) -> str:
    if secret:
        return secret
    global _EPHEMERAL_SECRET
    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_urlsafe(32)
        print(
            "[auth] AUTH_TOKEN_SECRET is not set; using a per-process random "
            "secret. Tokens will become invalid when the server restarts."
        )
    return _EPHEMERAL_SECRET


def _b64e(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> str:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")


def sign_token(
    user_id: str,
    username: str,
    secret: str,
    ttl_hours: int,
) -> str:
    claims = {
        "user_id": user_id,
        "username": username,
        "exp": int(time.time()) + ttl_hours * 3600,
    }
    body = _b64e(json.dumps(claims, separators=(",", ":")))
    signature = _b64e_bytes(_hmac_bytes(_effective_secret(secret), body))
    return f"{body}.{signature}"


def verify_token(token: str, secret: str) -> dict[str, Any] | None:
    """Return claims or None if malformed/tampered/expired."""
    try:
        body, signature = token.split(".", 1)
    except ValueError:
        return None
    try:
        signature_bytes = _b64d_bytes(signature)
    except (ValueError, binascii.Error):
        return None
    if not hmac.compare_digest(_hmac_bytes(_effective_secret(secret), body), signature_bytes):
        return None
    try:
        claims = json.loads(_b64d(body))
    except (ValueError, binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if int(claims.get("exp", 0)) <= int(time.time()):
        return None
    user_id = claims.get("user_id")
    username = claims.get("username")
    if not isinstance(user_id, str) or not isinstance(username, str):
        return None
    return {"user_id": user_id, "username": username, "exp": claims["exp"]}


def _hmac_bytes(secret: str, body: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()


def _b64d_bytes(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _b64e_bytes(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return (
        f"pbkdf2_sha256${PBKDF2_ITERATIONS}"
        f"${_b64e_bytes(salt)}${_b64e_bytes(digest)}"
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        salt = _b64d_bytes(salt_b64)
        expected = _b64d_bytes(digest_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def is_valid_username(username: str) -> bool:
    return bool(USERNAME_PATTERN.fullmatch(username or ""))


def _normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def _connect(postgres_uri: str, *, timeout: int) -> Connection:
    return Connection.connect(
        postgres_uri,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
        connect_timeout=timeout,
    )


def ensure_auth_schema(postgres_uri: str, *, timeout: int = 5) -> None:
    try:
        conn = _connect(postgres_uri, timeout=timeout)
    except OperationalError as exc:
        raise RuntimeError(
            "Auth schema requires the same PostgreSQL database used for "
            "checkpointing, but it is unreachable."
        ) from exc
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id       text PRIMARY KEY,
                    username      text NOT NULL UNIQUE,
                    password_hash text NOT NULL,
                    created_at    timestamptz NOT NULL DEFAULT now()
                )
                """
            )
    finally:
        conn.close()


def register_user(
    postgres_uri: str,
    username: str,
    password: str,
    *,
    timeout: int = 5,
) -> str | None:
    """Create a user; return user_id, or None when the username is taken."""
    normalized = _normalize_username(username)
    user_id = uuid.uuid4().hex
    conn = _connect(postgres_uri, timeout=timeout)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO auth_users (user_id, username, password_hash) "
                "VALUES (%s, %s, %s)",
                (user_id, normalized, hash_password(password)),
            )
    except UniqueViolation:
        return None
    finally:
        conn.close()
    return user_id


def verify_login(
    postgres_uri: str,
    username: str,
    password: str,
    *,
    timeout: int = 5,
) -> str | None:
    normalized = _normalize_username(username)
    conn = _connect(postgres_uri, timeout=timeout)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT user_id, password_hash FROM auth_users "
                "WHERE username = %s",
                (normalized,),
            )
            row = cursor.fetchone()
    finally:
        conn.close()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return row["user_id"]
