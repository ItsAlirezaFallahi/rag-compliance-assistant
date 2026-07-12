"""Retrieval: embed the query, cosine-search pgvector.

Key mechanics to be able to explain:

  * `embedding <=> %s` is pgvector's cosine *distance* operator
    (0 = identical, 2 = opposite). similarity = 1 - distance.
  * We embed the query with the SAME model as the chunks -- vectors from
    different models live in different spaces and are not comparable.
  * top-k with an optional similarity floor. The floor matters for
    Phase 2/3: when the best match is weak (say < 0.30), that's the
    signal to refuse rather than hallucinate. Retrieval quality drives
    refusal behavior -- that's the thread connecting Phase 1 to your
    eval harness.
  * `min_similarity` is intentionally None by default: you'll calibrate
    it empirically in Phase 3 against your unanswerable questions,
    rather than guessing now.
"""

from dataclasses import dataclass

from app.db import get_conn
from app.ingest.embed import embed_query


@dataclass
class RetrievedChunk:
    doc_slug: str
    doc_title: str
    section_path: str
    chunk_index: int
    content: str
    similarity: float


def search(
    query: str,
    k: int = 5,
    doc_slug: str | None = None,
    min_similarity: float | None = None,
) -> list[RetrievedChunk]:
    qvec = embed_query(query)

    sql = """
        SELECT d.slug,
               d.title,
               c.section_path,
               c.chunk_index,
               c.content,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    where = "WHERE d.slug = %s" if doc_slug else ""
    params: list = [qvec]
    if doc_slug:
        params.append(doc_slug)
    params.extend([qvec, k])

    with get_conn() as conn:
        rows = conn.execute(sql.format(where=where), params).fetchall()

    results = [
        RetrievedChunk(
            doc_slug=r[0],
            doc_title=r[1],
            section_path=r[2],
            chunk_index=r[3],
            content=r[4],
            similarity=float(r[5]),
        )
        for r in rows
    ]
    if min_similarity is not None:
        results = [r for r in results if r.similarity >= min_similarity]
    return results
