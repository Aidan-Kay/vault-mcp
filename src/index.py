"""The resident index: vector matrix, chunk metadata, and a BM25 model.

No database. At this corpus size a rebuild costs seconds, and a DB would add
migrations, staleness and a failure mode for no benefit.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from . import chunker, vault
from .config import settings
from .embedder import Embedder

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Snowball's English stop list, minus the contraction forms _TOKEN_RE can never
# produce. These are the words whose presence in a query says nothing about
# which note answers it, and whose presence in the corpus is near-universal.
STOP_WORDS = frozenset(
    """
    i me my myself we us our ours ourselves you your yours yourself yourselves
    he him his himself she her hers herself it its itself they them their theirs
    themselves what which who whom this that these those am is are was were be
    been being have has had having do does did doing would should could ought
    a an the and but if or because as until while of at by for with about
    against between into through during before after above below to from up
    down in out on off over under again further then once here there when where
    why how all any both each few more most other some such no nor not only own
    same so than too very will
    """.split()
)

# snowballstemmer's stemmers carry a cursor across a call, so one shared object
# is not safe to hand to two threads. One per thread costs an attribute lookup
# and removes the question.
_LOCAL = threading.local()


def _stemmer() -> snowballstemmer.stemmer:
    stemmer = getattr(_LOCAL, "stemmer", None)
    if stemmer is None:
        stemmer = _LOCAL.stemmer = snowballstemmer.stemmer("english")
    return stemmer


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that is not [a-z0-9], drop stop words, stem.

    Splitting stays deliberately naive: 'nomic-embed-text' becomes three tokens
    rather than one, because a query for any part should still match. The
    stemmer leaves 'nomic' alone and treats 'embed' and 'text' the same way on
    both sides of the index, so it does not disturb that decision.

    What it does fix is the arm this corpus leans on hardest. Without a stemmer
    'readings' does not find a note that only ever writes 'reading', and
    'renewing' does not find one that writes 'renewal' - two misses the
    relevance fixture carried on purpose until this landed.

    Index and query go through this one function, which is the only arrangement
    that cannot drift: a word stemmed on one side and not the other is a term
    that matches nothing.
    """
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOP_WORDS]
    # Filtered again after stemming, because some stop words only become one
    # there: 'having' -> 'have', 'doing' -> 'do'.
    return [s for s in _stemmer().stemWords(words) if s not in STOP_WORDS]


def _bm25_document(chunk: dict) -> list[str]:
    # Title and breadcrumb are indexed alongside the body so a query naming the
    # note matches even when the body never repeats the name.
    return tokenize(f"{chunk['title']} {chunk['breadcrumb']} {chunk['text']}")


def chunk_terms(chunk: dict) -> frozenset[str]:
    """The distinct terms one chunk contributes to the lexical index.

    Exposed for the lookup override, which has to ask whether one chunk holds
    every term of a query rather than merely scoring well on some of them.
    """
    return frozenset(_bm25_document(chunk))


@dataclass(slots=True)
class VaultIndex:
    matrix: np.ndarray  # (N, 768) float32, C-contiguous, L2-normalised
    chunks: list[dict]  # parallel to matrix rows
    bm25: BM25Okapi | None = None
    build_seconds: float = 0.0
    note_count: int = 0
    built_at: float = field(default_factory=time.time)
    # term -> how many chunks hold it. Built here rather than read back out of
    # BM25Okapi, whose per-term counts are an implementation detail, and cached
    # rather than derived per query: asking rank_bm25 for one term's frequency
    # costs a pass over the whole corpus, and the lookup override asks once per
    # term of every query.
    doc_freqs: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bm25 is None and self.chunks:
            documents = [_bm25_document(c) for c in self.chunks]
            self.bm25 = BM25Okapi(documents)
            self.doc_freqs = Counter(term for document in documents for term in set(document))

    @property
    def size(self) -> int:
        return len(self.chunks)

    def summary(self) -> str:
        megabytes = self.matrix.nbytes / 1_048_576
        return (
            f"{self.size} chunks from {self.note_count} notes, "
            f"matrix {self.matrix.shape} ({megabytes:.1f} MB), "
            f"built in {self.build_seconds:.1f}s"
        )

    @classmethod
    def empty(cls) -> VaultIndex:
        return cls(matrix=np.zeros((0, settings.embed_dims), dtype=np.float32), chunks=[])

    @classmethod
    async def build(cls, embedder: Embedder) -> VaultIndex:
        started = time.perf_counter()
        notes = vault.walk_notes()

        chunks: list[dict] = []
        for path in notes:
            try:
                chunks.extend(chunker.chunk_note(path))
            except Exception:
                # One malformed note must not take down the whole index.
                log.exception("skipping unchunkable note %s", path)

        matrix = await embedder.embed([c["embed_text"] for c in chunks])
        index = cls(
            matrix=np.ascontiguousarray(matrix),
            chunks=chunks,
            build_seconds=time.perf_counter() - started,
            note_count=len(notes),
        )
        log.info("index built: %s", index.summary())
        return index

    async def replace_note(self, embedder: Embedder, path: Path) -> VaultIndex:
        """Return a new index with one note's rows swapped out.

        Rebuilding the array beats in-place slot management: np.delete plus
        np.vstack over a few megabytes is sub-millisecond, and it keeps chunks
        trivially parallel to matrix with no tombstones or index drift. BM25 has
        to be rebuilt wholesale regardless - rank_bm25 has no incremental update.
        """
        # resolve() is non-strict, so this still yields the right key for a
        # file that has already been deleted.
        rel = vault.relpath(path)

        new_chunks: list[dict] = []
        if path.exists():
            try:
                new_chunks = chunker.chunk_note(path)
            except FileNotFoundError:
                new_chunks = []  # deleted between the event firing and the read
            except Exception:
                log.exception("re-chunk failed for %s; dropping its rows", rel)
                new_chunks = []

        keep = [i for i, chunk in enumerate(self.chunks) if chunk["path"] != rel]
        kept_matrix = self.matrix[keep] if keep else np.zeros(
            (0, settings.embed_dims), dtype=np.float32
        )
        kept_chunks = [self.chunks[i] for i in keep]

        if new_chunks:
            new_matrix = await embedder.embed([c["embed_text"] for c in new_chunks])
            matrix = np.vstack([kept_matrix, new_matrix])
        else:
            matrix = kept_matrix

        note_paths = {c["path"] for c in kept_chunks} | ({rel} if new_chunks else set())
        return VaultIndex(
            matrix=np.ascontiguousarray(matrix),
            chunks=kept_chunks + new_chunks,
            note_count=len(note_paths),
            build_seconds=self.build_seconds,
        )
