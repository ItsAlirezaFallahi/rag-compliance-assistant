"""Central configuration.

Chunking parameters are deliberate choices you should be able to defend:

  MAX_CHUNK_TOKENS = 600
      Big enough that a chunk carries a complete regulatory idea (these
      docs are dense; a single validation principle often runs 2-4
      paragraphs). Small enough that retrieval stays precise -- oversized
      chunks dilute the embedding (one vector averaging several topics)
      and waste generation context in Phase 2.

  OVERLAP_TOKENS = 80
      Only applies when we're forced to split *within* a section. Overlap
      prevents an idea that straddles a split boundary from being
      unretrievable from either side. We do NOT overlap across section
      boundaries -- sections are semantically distinct by construction,
      which is the whole point of structure-aware chunking.

  MIN_CHUNK_TOKENS = 120
      Tiny trailing fragments get merged back into the previous chunk.
      A 30-token chunk is mostly noise with an embedding that matches
      everything weakly.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://localhost:5432/rag_compliance"
        )
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    max_chunk_tokens: int = 600
    overlap_tokens: int = 80
    min_chunk_tokens: int = 120

    # tiktoken encoding used for counting (cl100k_base is what the
    # embedding models use).
    encoding_name: str = "cl100k_base"


settings = Settings()
