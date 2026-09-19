"""The index cache: that it is only ever an optimisation, and invalidates right.

A cache over retrieval is a liability unless one property holds outright - a warm
build and a cold build produce the same chunks in the same order and the same
matrix row for row - because everything downstream of a wrong row is a search
result nobody can explain. So the first group here is that equality, asserted on
the bytes of the matrix rather than on a similarity, and every other group exists
to find a way to break it.

The invalidation checks are the ones that earn their keep. Each names a thing
that changes what would be built today, and asserts which of the two stores it
has to take with it:

    a note edited          that file's chunks, and only the vectors that moved
    a note renamed         its chunks; its vectors survive, being text-keyed
    the chunker changed    every chunk; every vector survives
    the model changed      every vector; every chunk survives
    a different vault      both, and it is not asked twice

The pair in the middle is the whole reason there are two stores, and the reason
the test asserts on *both* halves of each: a fingerprint change that also threw
the vectors away would pass a test that only looked at the chunks, and would cost
a full re-embed every time the chunker is touched. Phase 1 touched it twice.

The embedder here counts what it was asked for. That is the measurement the whole
feature is about, and an assertion on a call count is the only way to state "this
did not go to Ollama" - a matrix that came back correct proves nothing about
where it came from.

The last group is not about the file at all. With chunking and embedding both
free on a warm start, what was left of an *incremental* reindex turned out to be
re-tokenising the whole corpus for a one-note edit, so the index now carries its
token lists forward too. It is the same question - what does a rebuild not have
to redo - and it was found by measuring this one.

Needs a writable vault, since half of these checks edit one, so it copies the
fixture to a temp tree and points VAULT_PATH at it before importing src.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "vault"

_ROOT = Path(tempfile.mkdtemp(prefix="vault-cache-"))
_VAULT = _ROOT / "vault"
shutil.copytree(FIXTURES, _VAULT)
os.environ["VAULT_PATH"] = str(_VAULT)
os.environ["INDEX_CACHE_PATH"] = str(_ROOT / "cache" / "index.npz")

from src import chunker, documents, indexcache, vault  # noqa: E402
from src.config import settings  # noqa: E402
from src.index import VaultIndex  # noqa: E402
from src.indexcache import IndexCache  # noqa: E402

from .relevance.hashing_embedder import HashingEmbedder  # noqa: E402

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual == expected:
        return
    FAILURES.append(f"{name}\n    expected: {expected!r}\n    actual:   {actual!r}")


def report() -> int:
    if not FAILURES:
        return 0
    print(f"{len(FAILURES)} failure(s):\n")
    for failure in FAILURES:
        print(failure + "\n")
    return 1


class CountingEmbedder(HashingEmbedder):
    """A deterministic embedder that remembers how much it was asked to do.

    Deterministic matters as much as counting: a cache is only provably
    transparent if the uncached answer is reproducible, so the same text has to
    give the same vector on both sides of the comparison.

    `calls` is the parent's own counter, left alone deliberately - it skips an
    empty list, which is the difference between "asked for nothing" and "not
    asked", and that distinction is what half of these checks turn on.
    """

    def __init__(self) -> None:
        super().__init__()
        self.texts = 0

    async def embed(self, texts: list[str]) -> np.ndarray:
        self.texts += len(texts)
        return await super().embed(texts)


def fresh_cache() -> IndexCache:
    """A cache object over the configured path, as the server would open one."""
    cache = IndexCache.open()
    assert cache is not None, "INDEX_CACHE_PATH was set, so this cannot be None"
    return cache


def build(cache: IndexCache | None) -> tuple[VaultIndex, CountingEmbedder]:
    embedder = CountingEmbedder()
    index = asyncio.run(VaultIndex.build(embedder, cache))
    return index, embedder


def note(relative: str) -> Path:
    return _VAULT / relative


# --------------------------------------------------------------------------
# 1  A warm build is the cold build
# --------------------------------------------------------------------------


def transparency() -> None:
    cold, cold_embedder = build(None)

    # Populate, save, and reopen from the file - so this is the on-disk round
    # trip rather than a dictionary that happened to still be in memory.
    populating = fresh_cache()
    build(populating)
    check("the cache is dirty after a cold build", populating.dirty, True)
    check("and writes", populating.save(), True)
    check("and is then clean", populating.dirty, False)
    check("and writing again does nothing", populating.save(), False)

    reopened = fresh_cache()
    check("a written cache is read back", reopened.loaded, True)
    warm, warm_embedder = build(reopened)

    check("the same number of chunks", warm.size, cold.size)
    check("the same chunks, in the same order", warm.chunks, cold.chunks)
    check("the same note count", warm.note_count, cold.note_count)
    check(
        "the same matrix, row for row",
        np.array_equal(warm.matrix, cold.matrix),
        True,
    )
    check("and it is still float32", warm.matrix.dtype, np.dtype("float32"))
    check("and still C-contiguous", warm.matrix.flags["C_CONTIGUOUS"], True)

    # The point of all of the above.
    check("the cold build embedded every chunk", cold_embedder.texts, cold.size)
    check("the warm build embedded nothing", warm_embedder.texts, 0)
    check("and made no request at all", warm_embedder.calls, 0)
    check(
        "every file was reused",
        (reopened.stats.files_reused, reopened.stats.files_chunked),
        (len(vault.walk_notes()), 0),
    )


# --------------------------------------------------------------------------
# 2  One note edited
# --------------------------------------------------------------------------


def one_edit() -> None:
    cache = fresh_cache()
    build(cache)
    cache.save()

    target = note("Home/Utilities/Electricity.md")
    text = target.read_text(encoding="utf-8")
    target.write_text(text + "\n\nThe meter cupboard key lives in the hall drawer.\n",
                      encoding="utf-8")

    cache = fresh_cache()
    index, embedder = build(cache)
    files = len(vault.walk_notes())

    check("one file was re-chunked", cache.stats.files_chunked, 1)
    check("and the rest were reused", cache.stats.files_reused, files - 1)
    check("one request was made", embedder.calls, 1)
    # The edit falls in one chunk of a multi-chunk note, so the count is what
    # says the other chunks of the same note were not re-embedded along with it.
    check(
        "for fewer texts than the note has chunks in the index",
        embedder.texts <= len([c for c in index.chunks if c["path"] == "Home/Utilities/Electricity.md"]),
        True,
    )
    check("and for at least the changed one", embedder.texts >= 1, True)

    target.write_text(text, encoding="utf-8")  # put it back for the next group


# --------------------------------------------------------------------------
# 3  One note renamed
# --------------------------------------------------------------------------


def one_rename() -> None:
    """A rename invalidates the chunks and keeps the vectors.

    The path is in the cache key and in every chunk, so the chunks cannot be
    reused. The embed_text is built from the note's *frontmatter* title, its
    description and its body - none of which a rename touches - so every vector
    still applies, and this is the check that says so.
    """
    cache = fresh_cache()
    build(cache)
    cache.save()

    source = note("Home/Utilities/Electricity.md")
    renamed = note("Home/Utilities/Electricity Supply.md")
    source.rename(renamed)

    cache = fresh_cache()
    index, embedder = build(cache)

    check("the renamed file is re-chunked", cache.stats.files_chunked, 1)
    check("but nothing is re-embedded", embedder.texts, 0)
    check(
        "and the chunks carry the new path",
        any(c["path"] == "Home/Utilities/Electricity Supply.md" for c in index.chunks),
        True,
    )

    renamed.rename(source)


# --------------------------------------------------------------------------
# 4  The header decides which store survives
#
# Driven through _read rather than through settings, because settings resolve at
# import and these are three different processes' worth of configuration. What is
# being tested is the decision, and the decision lives in _read.
# --------------------------------------------------------------------------


def header_invalidation() -> None:
    cache = fresh_cache()
    build(cache)
    cache.save()
    files, vectors = len(cache.chunks_by_file), len(cache.vectors)
    check("the saved cache holds files", files > 0, True)
    check("and vectors", vectors > 0, True)

    def reopen(**changes) -> IndexCache:
        fields = {
            "fingerprint": cache.fingerprint,
            "model": cache.model,
            "vault": cache.vault,
        }
        fields.update(changes)
        other = IndexCache(path=cache.path, **fields)
        other._read()
        return other

    same = reopen()
    check("an unchanged header keeps both stores",
          (len(same.chunks_by_file), len(same.vectors)), (files, vectors))

    chunker_changed = reopen(fingerprint="something else")
    check("a changed chunk pipeline drops the chunks",
          len(chunker_changed.chunks_by_file), 0)
    check("and keeps every vector", len(chunker_changed.vectors), vectors)

    model_changed = reopen(model="some-other-model@768")
    check("a changed model drops the vectors", len(model_changed.vectors), 0)
    check("and keeps every chunk", len(model_changed.chunks_by_file), files)

    other_vault = reopen(vault="/somewhere/else")
    check("another vault's cache is not used at all",
          (len(other_vault.chunks_by_file), len(other_vault.vectors)), (0, 0))


def fingerprint_moves_with_the_code() -> None:
    """The fingerprint has to notice a chunker change without being told.

    A hand-bumped version constant is a gate that fires when somebody remembers,
    and the failure it misses is silent: stale chunks, served indefinitely. So
    this asserts the derivation - that the digest follows the *source* of the
    modules that produce chunks, and the settings that change their output.

    The source it edits is a stand-in in a temp directory, not the real chunker.
    A test that rewrites the repository it is testing leaves the repository
    rewritten if it is interrupted, and this one is asserting a mechanism rather
    than a file.
    """
    baseline = indexcache.pipeline_fingerprint()
    check("it is stable when nothing changes", indexcache.pipeline_fingerprint(), baseline)

    # Both are named because being in this tuple is the whole claim: documents.py
    # is the easy one to leave out, and a filed PDF's chunks *are* its extraction.
    check("the chunker is part of it", chunker in indexcache._PIPELINE_MODULES, True)
    check("and so is the extractor", documents in indexcache._PIPELINE_MODULES, True)

    stand_in = _ROOT / "stand_in.py"
    stand_in.write_text("# version one\n", encoding="utf-8")

    class Module:
        __name__ = "stand_in"
        __file__ = str(stand_in)

    real = indexcache._PIPELINE_MODULES
    indexcache._PIPELINE_MODULES = (Module(),)
    try:
        first = indexcache.pipeline_fingerprint()
        stand_in.write_text("# version one, with a comment added\n", encoding="utf-8")
        check("editing a pipeline module's source moves it",
              indexcache.pipeline_fingerprint() != first, True)
        stand_in.write_text("# version one\n", encoding="utf-8")
        check("and putting it back brings it back", indexcache.pipeline_fingerprint(), first)

        stand_in.unlink()
        check("an unreadable source is never trusted as unchanged",
              indexcache.pipeline_fingerprint() != indexcache.pipeline_fingerprint(), True)
    finally:
        indexcache._PIPELINE_MODULES = real

    check("and the real fingerprint is unaffected", indexcache.pipeline_fingerprint(), baseline)

    # A chunk setting is as invalidating as a line of the chunker, and easier to
    # forget: the same code over the same note produces different chunks.
    original = settings.chunk_target_tokens
    object.__setattr__(settings, "chunk_target_tokens", original + 1)
    try:
        check("a chunk setting moves it too",
              indexcache.pipeline_fingerprint() != baseline, True)
    finally:
        object.__setattr__(settings, "chunk_target_tokens", original)


# --------------------------------------------------------------------------
# 5  Pruning
# --------------------------------------------------------------------------


def pruning() -> None:
    """A full build drops everything it did not use.

    Without this the file grows with the vault's history rather than its size:
    every edit leaves the old chunks and the old vectors behind, and nothing ever
    asks after them again.
    """
    cache = fresh_cache()
    build(cache)
    before = (len(cache.chunks_by_file), len(cache.vectors))

    stray = note("Home/Utilities/Scratch.md")
    stray.write_text(
        "---\ntitle: Scratch\ndescription: Deleted in a moment.\n---\n\n"
        "# Scratch\n\nA line about the immersion heater timer switch.\n",
        encoding="utf-8",
    )
    build(cache)
    check(
        "a new note adds an entry",
        len(cache.chunks_by_file) == before[0] + 1,
        True,
    )
    grown = len(cache.vectors)
    check("and at least one vector", grown > before[1], True)

    stray.unlink()
    build(cache)
    check("deleting it takes its entry away", len(cache.chunks_by_file), before[0])
    check("and its vectors", len(cache.vectors), before[1])


# --------------------------------------------------------------------------
# 6  Every failure is the cache's own
# --------------------------------------------------------------------------


def failure_is_contained() -> None:
    """A cache that cannot be read or written must cost a slower start only.

    Logging is turned down across this group. Every failure below is one this
    test caused on purpose, and each one traces itself at ERROR - which is
    correct in production and, here, four stack traces burying the one line that
    says whether the checks passed.
    """
    import logging

    logging.disable(logging.ERROR)
    try:
        _failure_is_contained()
    finally:
        logging.disable(logging.NOTSET)


def _failure_is_contained() -> None:
    cache = fresh_cache()
    build(cache)
    cache.save()

    # Truncated: the archive's own directory is at the end of a zip, so cutting
    # the tail is exactly what a build killed mid-write would have left behind -
    # which is the thing the atomic replace exists to make impossible, and the
    # thing this asserts is survivable if it ever happens anyway.
    raw = cache.path.read_bytes()
    cache.path.write_bytes(raw[: len(raw) // 2])
    salvaged = fresh_cache()
    check("a truncated cache reads as empty", len(salvaged.chunks_by_file), 0)
    check("and empty of vectors too", len(salvaged.vectors), 0)

    index, embedder = build(salvaged)
    check("and the build still happens", index.size > 0, True)
    check(
        "paying the full embedding for it",
        embedder.texts,
        len({chunk["embed_text"] for chunk in index.chunks}),
    )

    cache.path.write_bytes(b"not an archive at all")
    check("nor does a file that is not an archive raise", len(fresh_cache().vectors), 0)

    # Unwritable: the directory is replaced by a file, so mkdir and the temp
    # write both fail. Windows and POSIX disagree about chmod, and they do not
    # disagree about this.
    blocked = _ROOT / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    stuck = IndexCache(
        path=blocked / "index.npz",
        fingerprint="f",
        model="m",
        vault=str(_VAULT),
    )
    stuck.store_vector("some text", np.zeros(settings.embed_dims, dtype=np.float32))
    check("a save that cannot happen reports false", stuck.save(), False)
    check("and counts itself", stuck.write_failures, 1)
    check("and stays dirty, so a later flush can try again", stuck.dirty, True)
    check("and says so in its status", stuck.as_json()["write_failures"], 1)

    # The garbage above is left where the next group would read it, and a group
    # that has to survive the previous one's rubble is testing two things at once.
    cache.path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# 7  Switched off
# --------------------------------------------------------------------------


def disabled() -> None:
    """INDEX_CACHE_PATH= is a supported configuration, not a degraded one."""
    check("a configured path opens a cache", IndexCache.open() is not None, True)

    saved = settings.index_cache_path
    object.__setattr__(settings, "index_cache_path", None)
    try:
        check("an empty one opens nothing", IndexCache.open(), None)
        index, embedder = build(None)
        check("and the build is the plain one", index.size > 0, True)
        check("embedding everything", embedder.texts, index.size)
    finally:
        object.__setattr__(settings, "index_cache_path", saved)


# --------------------------------------------------------------------------
# 8  What it writes
# --------------------------------------------------------------------------


def on_disk() -> None:
    cache = fresh_cache()
    build(cache)
    cache.save()

    with np.load(cache.path, allow_pickle=False) as archive:
        header = json.loads(bytes(archive["manifest"]).decode("utf-8"))
        matrix = archive["vectors"]

    check("the format is named", header["format"], indexcache.FORMAT)
    check("the vault is named", header["vault"], str(settings.vault_path))
    check("the model is named", header["model"], indexcache.model_id())
    check("one row per key", len(matrix), len(header["vector_keys"]))
    check("stored as float32", matrix.dtype, np.dtype("float32"))
    check("no half-written file is left behind",
          list(cache.path.parent.glob("*.writing")), [])

    # A vector is a lookup on the text, so the text has to be enough to find it.
    any_chunk = next(iter(cache.chunks_by_file.values()))[0]
    row = cache.vector_for(any_chunk["embed_text"])
    check("a chunk's text finds its vector", row is not None, True)
    check(
        "and it is the row that was written",
        np.array_equal(row, matrix[header["vector_keys"].index(
            indexcache.text_digest(any_chunk["embed_text"]))]),
        True,
    )


# --------------------------------------------------------------------------
# 9  The other thing a rebuild does not have to redo
#
# Not the file cache, and here because it was found by measuring it. With
# chunking and embedding both free on a warm start, what was left of an
# incremental reindex turned out to be tokenising: replace_note rebuilds the
# BM25 model, and rebuilding it re-derived the tokens of every chunk in the
# vault - 5.4 seconds of the event loop for a change to one note, on the thread
# that serves every search. BM25Okapi's own init over those tokens is 51 ms.
#
# Asserted by counting rather than by timing. A stopwatch in a test is a flake on
# somebody else's machine, and the claim is not "it is fast" but "it does not do
# that work at all".
# --------------------------------------------------------------------------


def incremental_tokenising() -> None:
    from src import index as index_module

    cache = fresh_cache()
    index, _ = build(cache)

    target = note("Home/Utilities/Electricity.md")
    rel = "Home/Utilities/Electricity.md"
    before = len([c for c in index.chunks if c["path"] == rel])
    check("the note under test has chunks", before > 0, True)
    check("tokens are kept, one list per chunk", len(index.documents), len(index.chunks))
    check(
        "and each is that chunk's",
        index.documents[0],
        index_module._bm25_document(index.chunks[0]),
    )

    original = target.read_text(encoding="utf-8")
    target.write_text(
        original + "\n\nThe immersion heater runs on an off-peak circuit.\n",
        encoding="utf-8",
    )

    counted = {"n": 0}
    real = index_module._bm25_document

    def counting(chunk: dict) -> list[str]:
        counted["n"] += 1
        return real(chunk)

    index_module._bm25_document = counting
    try:
        updated = asyncio.run(index.replace_note(CountingEmbedder(), target, cache))
    finally:
        index_module._bm25_document = real

    after = len([c for c in updated.chunks if c["path"] == rel])
    check("only the edited note is tokenised", counted["n"], after)
    check("which is far fewer than the corpus", counted["n"] < len(updated.chunks), True)

    # Carrying tokens forward is only allowed if the result is the same model.
    rebuilt, _ = build(fresh_cache())
    check("the same chunks either way", sorted(c["text"] for c in updated.chunks),
          sorted(c["text"] for c in rebuilt.chunks))
    order = {c["text"]: i for i, c in enumerate(rebuilt.chunks)}
    check(
        "and the same term set for each of them",
        [updated.term_sets[i] for i, c in enumerate(updated.chunks)],
        [rebuilt.term_sets[order[c["text"]]] for c in updated.chunks],
    )
    check("and the same corpus-wide term counts", updated.doc_freqs, rebuilt.doc_freqs)

    target.write_text(original, encoding="utf-8")


# --------------------------------------------------------------------------
# 10  The number /readyz publishes
# --------------------------------------------------------------------------


def build_seconds_covers_the_build() -> None:
    """build_seconds has to mean the whole build, including the BM25 model.

    It did not. It was passed as an argument to the constructor, so it was
    evaluated before __post_init__ - and __post_init__ is where the tokenising
    and the BM25 model happen. On a cold build, where embedding dominated, that
    was a 12% under-report nobody would notice. With the cache it is most of the
    work: the server logged a warm start taking 3.28s and reported 0.12s on the
    readiness endpoint.

    Asserted by making __post_init__ take a known minimum rather than by timing
    the real thing. A sleep is a floor, so the check cannot flake on a slow
    machine - it can only fail if the span genuinely excludes the delay.
    """
    import time as timing

    from src import index as index_module

    cache = fresh_cache()
    build(cache)
    cache.save()

    injected = 0.004
    real = index_module._bm25_document

    def slow(chunk: dict) -> list[str]:
        timing.sleep(injected)
        return real(chunk)

    index_module._bm25_document = slow
    try:
        # A warm build, so tokenising is nearly all of the work that is left and
        # the floor below is not competing with chunking or embedding.
        warm, embedder = build(fresh_cache())
    finally:
        index_module._bm25_document = real

    floor = injected * len(warm.chunks)
    check("nothing was embedded, so the delay is the build", embedder.texts, 0)
    check(
        f"build_seconds ({warm.build_seconds:.3f}s) covers __post_init__ (>= {floor:.3f}s)",
        warm.build_seconds >= floor,
        True,
    )


def main() -> int:
    transparency()
    one_edit()
    one_rename()
    header_invalidation()
    fingerprint_moves_with_the_code()
    pruning()
    failure_is_contained()
    disabled()
    on_disk()
    incremental_tokenising()
    build_seconds_covers_the_build()

    if report():
        return 1
    print("cache: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)
