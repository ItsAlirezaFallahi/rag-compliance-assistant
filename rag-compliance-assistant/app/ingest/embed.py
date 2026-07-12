"""OpenAI embedding calls, batched.

Why batch: one API round trip per chunk is slow and rate-limit-prone;
the embeddings endpoint accepts up to 2048 inputs per request. We use a
conservative batch of 64.

Why text-embedding-3-small: 1536 dims, cheap (~$0.02 / 1M tokens), and
for a two-document corpus the retrieval-quality gap vs. -3-large is
negligible. Right-sizing the model is itself a defensible decision.
"""

import time

from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

BATCH_SIZE = 64
MAX_RETRIES = 3


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                resp = _client.embeddings.create(
                    model=settings.embedding_model, input=batch
                )
                # API returns items with an .index field; sort defensively.
                ordered = sorted(resp.data, key=lambda d: d.index)
                vectors.extend(d.embedding for d in ordered)
                break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)  # 1s, 2s backoff
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return vectors


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
