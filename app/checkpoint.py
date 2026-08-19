from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import Connection
from psycopg.rows import dict_row

from app.schemas import (
    ChunkRelevanceGrade,
    QueryRewrite,
    RagGeneration,
    RelevanceGrade,
    RetrievedChunk,
)

ALLOWED_CHECKPOINT_TYPES = (
    ChunkRelevanceGrade,
    QueryRewrite,
    RagGeneration,
    RelevanceGrade,
    RetrievedChunk,
)


@contextmanager
def postgres_checkpointer_from_uri(postgres_uri: str) -> Iterator[PostgresSaver]:
    conn = Connection.connect(
        postgres_uri,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    try:
        yield PostgresSaver(
            conn,
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=ALLOWED_CHECKPOINT_TYPES,
            ),
        )
    finally:
        conn.close()
