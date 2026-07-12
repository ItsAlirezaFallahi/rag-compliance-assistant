# RAG Compliance Assistant — Phase 1: Ingestion & Retrieval

A retrieval-augmented assistant over financial regulatory documents (SR 11-7 and NIST AI RMF 1.0). Phase 1 builds the retrieval foundation: structure-aware chunking, pgvector storage, and cosine-similarity search — all hand-written so every decision is explainable.

## Architecture

```
PDF ──> parse.py ──> chunk.py ──> embed.py ──> Postgres (pgvector)
         sections     chunks       vectors          │
                                                    ▼
                             query ──> embed ──> cosine search ──> top-k chunks
```

Five moving parts, one file each:

1. **`app/ingest/parse.py`** — PDF → sections using each document's own heading structure (Roman numerals for SR 11-7, decimal numbering for NIST AI RMF). This is what "structure-aware" means: the regulator already segmented the document semantically; we reuse that instead of slicing blindly.
2. **`app/ingest/chunk.py`** — sections → chunks. One chunk per section when it fits (≤600 tokens); sentence-boundary splits with 80-token overlap when it doesn't; runt merging; contextual breadcrumb prefix on the embedded text.
3. **`app/ingest/embed.py`** — batched calls to `text-embedding-3-small` (1536-dim).
4. **`app/db.py` + `db/schema.sql`** — raw SQL over psycopg3, HNSW cosine index.
5. **`app/retrieve.py`** — query embedding + `<=>` cosine search, doc filtering, optional similarity floor.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI key

# Postgres + pgvector (Ubuntu):
sudo apt install postgresql postgresql-16-pgvector   # match your PG version
createdb rag_compliance
python -m app.cli init-db
```

Download the source documents into `data/`:
- SR 11-7 attachment: https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf
- NIST AI RMF 1.0: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

## Workflow

```bash
# 1. ALWAYS inspect first — free, no API calls. Verify sections look right.
python -m app.cli inspect sr-11-7 data/sr1107a1.pdf

# 2. Tune heading/noise regexes in parse.py if sections are wrong, re-inspect.

# 3. Ingest (parses, chunks, embeds, stores).
python -m app.cli ingest sr-11-7 data/sr1107a1.pdf
python -m app.cli ingest nist-ai-rmf data/NIST.AI.100-1.pdf

# 4. Query.
python -m app.cli query "What are the core elements of an effective model validation framework?"
python -m app.cli query "How does NIST define AI risk?" --doc nist-ai-rmf -k 3
```

Ingestion is idempotent — re-ingesting a slug deletes its old chunks first.

## Smoke tests (queries with known answers)

| Query | Should retrieve |
|---|---|
| "core elements of model validation" | SR 11-7 § Model Validation |
| "conceptual soundness and developmental evidence" | SR 11-7 validation section |
| "effective challenge" | SR 11-7 governance/validation |
| "GOVERN function categories" | NIST AI RMF §5.1 |
| "characteristics of trustworthy AI" | NIST AI RMF Part 1/§3 |
| "vendor model risk" | SR 11-7 third-party section |

If these come back with the right sections at similarity ≳ 0.45, retrieval is working.

## Interview talking points (the "why" behind each decision)

**Why structure-aware chunking instead of fixed-size?** Fixed windows cut requirements mid-sentence and average multiple topics into one embedding, hurting both retrieval precision and citation quality. Regulatory docs have strong internal structure — using it is free signal. Tradeoff: it's per-document work (heading regexes), which is why naive RAG demos skip it.

**Why 600-token chunks?** Dense regulatory ideas often span several paragraphs — too small and you retrieve fragments that can't support an answer; too large and the embedding gets diluted and generation context gets wasted. 400–800 is the standard band; the honest answer is "I'd tune it against my Phase 3 eval set," which is exactly what the eval harness is for.

**Why prefix the breadcrumb before embedding?** Chunks lose context when isolated ("the bank should ensure..." — which doc? which section?). Prepending `SR 11-7 > V. Model Validation` puts that context into the vector. Cheap version of contextual retrieval. We embed the prefixed text but store/generate from the clean text.

**Why cosine similarity?** OpenAI embeddings are normalized to unit length, so cosine and dot product rank identically; cosine is the conventional choice and pgvector's `vector_cosine_ops` supports it natively. `<=>` returns distance; similarity = 1 − distance.

**Why HNSW over IVFFlat?** HNSW: graph-based ANN, better recall, no training step, slower builds and more memory. IVFFlat: clusters vectors, needs a training pass, recall depends on `lists`/`probes`. At ~1–2k chunks either (or exact scan) is fine — the index exists to demonstrate the production pattern.

**Why pgvector over Pinecone/a vector DB?** Banks already run Postgres; one fewer system, transactional consistency between metadata and vectors, and no data leaving the perimeter. For millions of vectors with high QPS you'd revisit — know the boundary.

**Why does retrieval quality matter for refusal (Phase 2/3)?** A low top similarity (calibrate empirically; often ~0.3 with -3-small) signals the corpus doesn't cover the question — the trigger for "I can't answer from these documents" instead of hallucinating. In a compliance setting, refusal on out-of-scope questions is a feature, and your eval set's unanswerable questions will test it.

**Known limitations (own them proactively):** heading detection is heuristic per-document; no hybrid (BM25 + vector) search yet — exact term matches like "SR 11-7" itself are where pure vector search underperforms; no reranker; tables in NIST AI RMF flatten poorly in text extraction.

## Phase roadmap

- [x] **Phase 1** — ingestion + retrieval (this)
- [ ] **Phase 2** — grounded generation with citations + refusal
- [ ] **Phase 3** — eval harness: 20–30 Q&A incl. unanswerable, LLM-as-judge faithfulness
- [ ] **Phase 4** — input guardrails (PII redaction, prompt-injection checks)
