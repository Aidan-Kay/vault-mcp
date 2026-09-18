"""A deterministic stand-in for Ollama, so the eval runs where Ollama does not.

CI has no embedding endpoint, and pinning one would make the retrieval suite a
network test. This produces vectors from the text alone: same input, same vector,
on every machine and every run, with no model and no service.

**What it does and does not prove.** This is the hashing trick, not a language
model. A token is hashed into two of `EMBED_DIMS` buckets and weighted by
sublinear term frequency, so the "dense" arm it feeds scores on shared vocabulary
and nothing else. It has no synonyms, no paraphrase and no notion that a tariff
and a unit rate are related. So a green run against this embedder measures the
lexical arm, the tokeniser, reciprocal rank fusion, the lookup override and the
per-note cap - all of which are ours - and measures the dense arm only in the
weak sense that it puts *something* differently-shaped on the other side of the
fusion. Queries whose answer needs real semantics are tagged `"arm": "dense"` in
queries.json, and eval.py reports them without letting them pass or fail a run.

The nomic prefixes are stripped rather than hashed. `search_document: ` on every
chunk and `search_query: ` on every query would be shared vocabulary across the
whole corpus, which is real signal to nomic and pure dilution here.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from src.config import settings
from src.embedder import l2_normalise

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PREFIX_RE = re.compile(r"^search_(document|query):\s*", re.IGNORECASE)

# Two buckets per token, with the second signed negative, is the standard
# hashing-trick mitigation for collisions: a collision that lands on both
# buckets of another token cancels rather than accumulates.
_BUCKETS = 2


def _digest(token: str) -> bytes:
    # blake2b rather than hash(): PYTHONHASHSEED randomises str hashing per
    # process, which would make this reproducible only within one run.
    return hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()


def vectorise(text: str, dims: int) -> np.ndarray:
    counts: dict[str, int] = {}
    for token in _TOKEN_RE.findall(_PREFIX_RE.sub("", text).lower()):
        counts[token] = counts.get(token, 0) + 1

    vector = np.zeros(dims, dtype=np.float32)
    for token, count in counts.items():
        weight = 1.0 + math.log(count)
        digest = _digest(token)
        for bucket in range(_BUCKETS):
            offset = bucket * 4
            position = int.from_bytes(digest[offset : offset + 4], "big") % dims
            vector[position] += weight if bucket == 0 else -weight
    return vector


class HashingEmbedder:
    """Duck-types src.embedder.Embedder for everything the index and search use."""

    def __init__(self, dims: int | None = None) -> None:
        self.dims = dims or settings.embed_dims
        self.calls = 0

    async def aclose(self) -> None:
        return None

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        self.calls += 1
        matrix = np.vstack([vectorise(text, self.dims) for text in texts])
        return l2_normalise(matrix)

    async def embed_query(self, query: str) -> np.ndarray:
        matrix = await self.embed([query])
        return matrix[0]
