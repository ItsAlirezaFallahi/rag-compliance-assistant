"""Thin database layer: psycopg3 + pgvector, no ORM.

Deliberate choice: for an interview project, raw SQL keeps the retrieval
mechanics visible. You can literally point at the `<=>` cosine-distance
operator in your own code. (You already know SQLAlchemy from Nudgemate;
this project is about RAG, not ORMs.)
"""

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


@contextmanager
def get_conn():
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)  # teaches psycopg to adapt vector <-> numpy
        yield conn


def init_schema(schema_path: str = "db/schema.sql") -> None:
    with open(schema_path) as f:
        sql = f.read()
    with get_conn() as conn:
        conn.execute(sql)
        conn.commit()


def upsert_document(slug: str, title: str, source_url: str | None) -> int:
    """Insert the document row (or reset it if re-ingesting). Returns id."""
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (slug, title, source_url)
            VALUES (%s, %s, %s)
            ON CONFLICT (slug) DO UPDATE
                SET title = EXCLUDED.title,
                    source_url = EXCLUDED.source_url,
                    ingested_at = now()
            RETURNING id
            """,
            (slug, title, source_url),
        ).fetchone()
        doc_id = row[0]
        # Idempotent re-ingest: wipe old chunks for this document.
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
        conn.commit()
    return doc_id


def insert_chunks(doc_id: int, records: list[dict]) -> None:
    """records: dicts with section_path, section_title, chunk_index,
    content, token_count, embedding (list[float])."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (document_id, section_path, section_title, chunk_index,
                     content, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        doc_id,
                        r["section_path"],
                        r["section_title"],
                        r["chunk_index"],
                        r["content"],
                        r["token_count"],
                        r["embedding"],
                    )
                    for r in records
                ],
            )
        conn.commit()
