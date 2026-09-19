"""The resident index: vector matrix, chunk metadata, and a BM25 model.

No database. At this corpus size a rebuild costs seconds, and a DB would add
migrations, staleness and a failure mode for no benefit.

What a rebuild does not have to do is repeat work whose inputs have not moved, so
`build()` takes an optional cache of chunks and vectors keyed by content -
see indexcache.py. It is an optimisation and nothing else: a warm build and a
cold build produce the same chunks in the same order and the same matrix row for
row, which is what `tests/cache.py` asserts.
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

from . import chunker, indexcache, vault
from .config import settings
from .embedder import Embedder
from .indexcache import IndexCache

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


def _file_chunks(path: Path, cache: IndexCache | None) -> tuple[list[dict], str]:
    """One file's chunks, and the cache key they are held under.

    The key is returned whether it was a hit or a miss, because the caller needs
    the whole live set to prune against and a miss is just as live as a hit.
    """
    if cache is None:
        return chunker.chunk_note(path), ""

    rel = vault.relpath(path)
    digest = indexcache.file_digest(path)
    key = indexcache.chunk_key(rel, digest)

    cached = cache.chunks_for(rel, digest)
    if cached is not None:
        return cached, key

    produced = chunker.chunk_note(path)
    cache.store_chunks(rel, digest, produced)
    return produced, key


async def _embed_chunks(
    embedder: Embedder, chunks: list[dict], cache: IndexCache | None
) -> np.ndarray:
    """The matrix for these chunks, embedding only the texts the cache lacks.

    Row order follows `chunks` exactly, hit or miss, because everything
    downstream indexes one by the other. The misses go in one batched call rather
    than one per file, so a mostly-warm start makes a single request instead of
    two hundred.

    Identical embed_texts are collapsed into one request. That is not a
    hypothetical saving on a vault of templated notes, and it costs three lines.
    """
    if cache is None:
        return await embedder.embed([c["embed_text"] for c in chunks])

    rows: list[np.ndarray | None] = []
    wanted: dict[str, list[int]] = {}
    for position, chunk in enumerate(chunks):
        row = cache.vector_for(chunk["embed_text"])
        rows.append(row)
        if row is None:
            wanted.setdefault(chunk["embed_text"], []).append(position)

    if wanted:
        texts = list(wanted)
        fresh = await embedder.embed(texts)
        for text, row in zip(texts, fresh):
            cache.store_vector(text, row)
            for position in wanted[text]:
                rows[position] = row

    if not rows:
        return np.zeros((0, settings.embed_dims), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def _bm25_document(chunk: dict) -> list[str]:
    # Title and breadcrumb are indexed alongside the body so a query naming the
    # note matches even when the body never repeats the name.
    return tokenize(f"{chunk['title']} {chunk['breadcrumb']} {chunk['text']}")


def chunk_terms(chunk: dict) -> frozenset[str]:
    """The distinct terms one chunk contributes to the lexical index.

    The live search path reads VaultIndex.term_sets instead, which holds exactly
    this for every chunk and is built once. This stays as the definition of what
    that field contains, and for callers holding a chunk rather than an index.
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
    # chunk index -> the distinct terms it holds, parallel to `chunks`. Built
    # here because __post_init__ already tokenises every chunk to feed BM25, so
    # this costs one frozenset per chunk and no extra tokenising at all. The
    # lookup override asks "does this chunk hold every query term" of up to
    # CANDIDATES chunks per query; doing that by re-tokenising cost 107 ms per
    # query on a 2100-chunk vault, against 0.1 ms for the set comparisons.
    term_sets: list[frozenset[str]] = field(default_factory=list)
    # The tokenised form of every chunk, parallel to `chunks`, kept rather than
    # dropped at the end of __post_init__.
    #
    # Measured on the real vault: tokenising 2223 chunks costs 5349 ms, and
    # BM25Okapi's own init over the result costs 51 ms. replace_note rebuilds the
    # index for one edited note, so before this every note saved in Obsidian
    # re-tokenised the entire corpus - 5.4 seconds of the event loop, on the
    # thread that serves every search, for a change to one file. Carrying the
    # token lists forward turns that into tokenising the edited note alone.
    #
    # It costs 6.4 MB on that vault, against 6.5 MB for the matrix itself.
    # 2.2 MB of that is the lists and 4.2 MB is strings `term_sets` does not
    # already hold - a frozenset keeps one object per distinct term, and a list
    # keeps every occurrence. Worth it at forty times: the alternative is five
    # seconds of a blocked event loop every time a note is saved.
    documents: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chunks:
            return
        if not self.documents:
            self.documents = [_bm25_document(chunk) for chunk in self.chunks]
        if self.bm25 is None:
            self.bm25 = BM25Okapi(self.documents)
            self.doc_freqs = Counter(
                term for document in self.documents for term in set(document)
            )
            self.term_sets = [frozenset(document) for document in self.documents]

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
    async def build(cls, embedder: Embedder, cache: IndexCache | None = None) -> VaultIndex:
        started = time.perf_counter()
        notes = vault.walk_notes()
        if cache is not None:
            cache.reset_stats()

        chunks: list[dict] = []
        live_files: set[str] = set()
        for path in notes:
            try:
                produced, key = _file_chunks(path, cache)
            except Exception:
                # One malformed note must not take down the whole index.
                log.exception("skipping unchunkable note %s", path)
                continue
            chunks.extend(produced)
            live_files.add(key)

        matrix = await _embed_chunks(embedder, chunks, cache)

        if cache is not None:
            # Pruned here, and only here: a full build is the one moment the live
            # set is known exactly, and it happens at every start.
            cache.retain(
                live_files,
                {indexcache.text_digest(chunk["embed_text"]) for chunk in chunks},
            )

        index = cls(
            matrix=np.ascontiguousarray(matrix),
            chunks=chunks,
            note_count=len(notes),
        )
        # Assigned after construction rather than passed in, because an argument
        # is evaluated before __post_init__ - and __post_init__ is where the BM25
        # model and the token lists are built. Passing it in was a 12% error on a
        # 40s cold build, where embedding dominated, and became a 27x one the
        # moment the cache removed the embedding: the server reported 0.12s for a
        # warm start that took 3.28s by its own log timestamps. /readyz publishes
        # this number, so it has to mean the whole build.
        index.build_seconds = time.perf_counter() - started
        if cache is None:
            log.info("index built: %s", index.summary())
        else:
            counts = cache.stats
            log.info(
                "index built: %s (reused %d of %d file(s) and %d of %d vector(s))",
                index.summary(),
                counts.files_reused,
                counts.files_reused + counts.files_chunked,
                counts.vectors_reused,
                counts.vectors_reused + counts.vectors_embedded,
            )
        return index

    async def replace_note(
        self, embedder: Embedder, path: Path, cache: IndexCache | None = None
    ) -> VaultIndex:
        """Return a new index with one note's rows swapped out.

        Rebuilding the array beats in-place slot management: np.delete plus
        np.vstack over a few megabytes is sub-millisecond, and it keeps chunks
        trivially parallel to matrix with no tombstones or index drift. BM25 has
        to be rebuilt wholesale regardless - rank_bm25 has no incremental update.

        Rebuilt from the *tokens*, though, not from the text. Those two were the
        same thing until the token lists were kept, and the difference is 5.4
        seconds against 100 ms on this vault: constructing BM25Okapi over 2223
        pre-tokenised chunks is 51 ms, while deriving those tokens again - which
        is what every note saved in Obsidian used to do - is the rest of it.
        """
        # resolve() is non-strict, so this still yields the right key for a
        # file that has already been deleted.
        rel = vault.relpath(path)

        new_chunks: list[dict] = []
        if path.exists():
            try:
                new_chunks, _ = _file_chunks(path, cache)
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
        # The whole point of carrying `documents`: only the edited note is
        # tokenised, and the other two hundred come across as they are.
        kept_documents = [self.documents[i] for i in keep]

        if new_chunks:
            new_matrix = await _embed_chunks(embedder, new_chunks, cache)
            matrix = np.vstack([kept_matrix, new_matrix])
        else:
            matrix = kept_matrix

        note_paths = {c["path"] for c in kept_chunks} | ({rel} if new_chunks else set())
        return VaultIndex(
            matrix=np.ascontiguousarray(matrix),
            chunks=kept_chunks + new_chunks,
            documents=kept_documents + [_bm25_document(chunk) for chunk in new_chunks],
            note_count=len(note_paths),
            build_seconds=self.build_seconds,
        )
