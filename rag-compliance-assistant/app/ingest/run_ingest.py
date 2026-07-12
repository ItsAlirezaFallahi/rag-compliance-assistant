"""End-to-end ingestion: parse -> chunk -> embed -> store."""

from app.db import insert_chunks, upsert_document
from app.ingest.chunk import chunk_document
from app.ingest.embed import embed_texts
from app.ingest.parse import PROFILES, parse_pdf


def ingest(slug: str, pdf_path: str) -> None:
    profile = PROFILES[slug]
    print(f"Parsing {pdf_path} as '{profile.title}'...")

    sections = parse_pdf(pdf_path, slug)
    print(f"  {len(sections)} sections detected")

    chunks = chunk_document(sections, profile.title)
    print(f"  {len(chunks)} chunks produced")

    print("Embedding...")
    vectors = embed_texts([c.embed_text for c in chunks])

    doc_id = upsert_document(slug, profile.title, profile.source_url)
    insert_chunks(
        doc_id,
        [
            {
                "section_path": c.section_path,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "token_count": c.token_count,
                "embedding": v,
            }
            for c, v in zip(chunks, vectors)
        ],
    )
    print(f"Done. Stored {len(chunks)} chunks for '{slug}' (doc id {doc_id}).")
