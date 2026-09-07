from __future__ import annotations

import time

from app.auth import (
    hash_password,
    is_valid_username,
    register_user,
    scoped_thread_id,
    sign_token,
    verify_login,
    verify_password,
    verify_token,
)

SECRET = "unit-test-secret"


def test_token_roundtrip() -> None:
    token = sign_token("user-abc", "alice", SECRET, ttl_hours=1)
    claims = verify_token(token, SECRET)
    assert claims is not None
    assert claims["user_id"] == "user-abc"
    assert claims["username"] == "alice"


def test_token_rejects_tampered() -> None:
    token = sign_token("user-abc", "alice", SECRET, ttl_hours=1)
    assert verify_token(token + "x", SECRET) is None
    assert verify_token(f"junk.{token}", SECRET) is None
    assert verify_token("no-separator", SECRET) is None
    assert verify_token("", SECRET) is None


def test_token_rejects_wrong_secret() -> None:
    token = sign_token("user-abc", "alice", SECRET, ttl_hours=1)
    assert verify_token(token, "other-secret") is None


def test_token_rejects_expired() -> None:
    token = sign_token("user-abc", "alice", SECRET, ttl_hours=-1)
    assert verify_token(token, SECRET) is None


def test_password_hash_roundtrip() -> None:
    stored = hash_password("s3cret-pw")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("s3cret-pw", stored) is True
    assert verify_password("wrong-pw", stored) is False


def test_username_validation() -> None:
    assert is_valid_username("alice")
    assert is_valid_username("alice_01")
    assert is_valid_username("中文名")
    assert is_valid_username("a.b-c")
    assert not is_valid_username("")
    assert not is_valid_username("has space")
    assert not is_valid_username("x" * 33)


def test_scoped_thread_id() -> None:
    assert scoped_thread_id(None, "t") == "t"
    assert scoped_thread_id("", "t") == "t"
    assert scoped_thread_id("u1", "t") == "u1::t"
    assert scoped_thread_id("u1", "t") != scoped_thread_id("u2", "t")


def test_register_login_with_postgres() -> None:
    from app.auth import ensure_auth_schema
    from app.config import get_config

    cfg = get_config()
    if not cfg.postgres_uri:
        return

    ensure_auth_schema(cfg.postgres_uri, timeout=cfg.postgres_connect_timeout)
    username = f"it_{int(time.time())}"
    user_id = register_user(cfg.postgres_uri, username, "pw-123456", timeout=cfg.postgres_connect_timeout)
    assert user_id is not None

    assert register_user(cfg.postgres_uri, username, "pw-123456", timeout=cfg.postgres_connect_timeout) is None
    assert verify_login(cfg.postgres_uri, username, "pw-123456", timeout=cfg.postgres_connect_timeout) == user_id
    assert verify_login(cfg.postgres_uri, username, "wrong-pw", timeout=cfg.postgres_connect_timeout) is None
    assert verify_login(cfg.postgres_uri, "no-such-user-zz", "pw-123456", timeout=cfg.postgres_connect_timeout) is None
