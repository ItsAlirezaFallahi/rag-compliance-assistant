"""Section -> Chunks.

Algorithm (all interview-explainable):

  1. Respect section boundaries. A section that fits in MAX_CHUNK_TOKENS
     becomes exactly one chunk. Sections are the semantic units the
     regulator itself chose -- we never merge across them.

  2. Oversized sections are split greedily on sentence boundaries into
     ~MAX_CHUNK_TOKENS windows, with OVERLAP_TOKENS of trailing context
     carried into the next chunk so no idea is orphaned at a boundary.

  3. Runt chunks (< MIN_CHUNK_TOKENS) are merged into their predecessor.

  4. Contextual prefix: the text we *embed* is
         "{doc title} > {section path}\n\n{chunk text}"
     so the vector encodes where the chunk lives, not just what it says.
     A chunk that reads "Banks should ensure ongoing monitoring..." is
     ambiguous on its own; prefixed with "SR 11-7 > VI. ..." it embeds
     much closer to model-risk queries. (This is the cheap version of
     Anthropic's "contextual retrieval" idea.) We store the clean text
     and the prefixed text separately -- generation in Phase 2 uses the
     clean text.
"""

import re
from dataclasses import dataclass

import tiktoken

from app.config import settings
from app.ingest.parse import Section

_enc = tiktoken.get_encoding(settings.encoding_name)


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class Chunk:
    section_path: str
    section_title: str
    chunk_index: int
    content: str          # clean text (stored, shown to the LLM in Phase 2)
    embed_text: str       # contextual-prefixed text (what we embed)
    token_count: int


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _pack_sentences(sentences: list[str]) -> list[str]:
    """Greedy pack sentences into windows <= max tokens, with overlap."""
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        stoks = count_tokens(sent)
        if current and current_tokens + stoks > settings.max_chunk_tokens:
            windows.append(" ".join(current))
            # Carry trailing sentences as overlap into the next window.
            overlap: list[str] = []
            otoks = 0
            for prev in reversed(current):
                ptoks = count_tokens(prev)
                if otoks + ptoks > settings.overlap_tokens:
                    break
                overlap.insert(0, prev)
                otoks += ptoks
            current = overlap[:]
            current_tokens = otoks
        current.append(sent)
        current_tokens += stoks

    if current:
        windows.append(" ".join(current))

    # Merge a runt final window into its predecessor.
    if (
        len(windows) >= 2
        and count_tokens(windows[-1]) < settings.min_chunk_tokens
    ):
        windows[-2] = windows[-2] + " " + windows[-1]
        windows.pop()

    return windows


def chunk_section(section: Section, doc_title: str) -> list[Chunk]:
    if count_tokens(section.text) <= settings.max_chunk_tokens:
        windows = [section.text]
    else:
        windows = _pack_sentences(_split_sentences(section.text))

    chunks = []
    for i, window in enumerate(windows):
        prefix = f"{doc_title} > {section.section_path}"
        chunks.append(
            Chunk(
                section_path=section.section_path,
                section_title=section.title,
                chunk_index=i,
                content=window,
                embed_text=f"{prefix}\n\n{window}",
                token_count=count_tokens(window),
            )
        )
    return chunks


def chunk_document(sections: list[Section], doc_title: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(chunk_section(section, doc_title))
    return chunks
