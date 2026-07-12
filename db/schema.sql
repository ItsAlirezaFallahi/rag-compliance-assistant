-- Phase 1 schema: documents + chunks with pgvector embeddings.
--
-- Interview talking points baked into this schema:
--   * embedding vector(1536)  -> matches text-embedding-3-small output dim
--   * HNSW index with cosine  -> approximate nearest neighbor; O(log n)-ish
--     queries vs. exact scan. IVFFlat is the alternative (faster build,
--     needs training step, worse recall at small scale). At our scale
--     (~1-2k chunks) even a sequential scan is fine -- the index is here
--     because it's what you'd do in production and you should be able to
--     explain the tradeoff.
--   * section_path metadata    -> enables grounded citations in Phase 2
--     ("SR 11-7 > V. Model Validation") and doc-level filtering.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    slug        TEXT UNIQUE NOT NULL,      -- 'sr-11-7', 'nist-ai-rmf'
    title       TEXT NOT NULL,
    source_url  TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    document_id   INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_path  TEXT NOT NULL,           -- breadcrumb, e.g. 'V. Model Validation'
    section_title TEXT,
    chunk_index   INT NOT NULL,            -- position within the section
    content       TEXT NOT NULL,
    token_count   INT NOT NULL,
    embedding     vector(1536) NOT NULL,
    UNIQUE (document_id, section_path, chunk_index)
);

-- Cosine-distance HNSW index. Query operator: <=> (cosine distance).
-- similarity = 1 - distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Helpful for doc-filtered retrieval.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
