from __future__ import annotations

import argparse

from langgraph.checkpoint.postgres import PostgresSaver

from app.config import get_config


def setup_checkpoint(postgres_uri: str) -> None:
    with PostgresSaver.from_conn_string(postgres_uri) as checkpointer:
        checkpointer.setup()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create LangGraph checkpoint tables in PostgreSQL."
    )
    parser.parse_args()

    config = get_config()
    if not config.postgres_uri:
        raise RuntimeError("Missing required environment variable: POSTGRES_URI")

    setup_checkpoint(config.postgres_uri)

    print("PostgreSQL checkpoint tables are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
