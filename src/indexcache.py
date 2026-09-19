"""A chunk-and-vector cache, so a restart is not a re-read and a re-embed.

The README declined a *database* for the index, and that still stands: there is
no schema here, no migration, no query language and nothing to keep in step with
the code. There is one file, and it is discarded wholesale the moment its header
stops describing what would be built today.

**Why it is worth having.** Measured on the real vault - 201 notes, 7 filed
documents, 2200 chunks - a cold build spends 23.6 s embedding, 8.0 s extracting
PDFs, 1.7 s chunking and 0.7 s walking the tree. Hashing every file in the vault
costs 0.15 s. The key is cheaper than a rounding error on the thing it avoids,
which is what makes content addressing the right shape here rather than mtimes.

**Two stores, and the reason they are two.**

    chunks    (vault path, sha256 of the file's bytes)  ->  the chunks it made
    vectors   sha256 of a chunk's embed_text            ->  its embedding

Keeping them apart is what makes each invalidation proportionate to its cause:

- A **chunker or extractor change** invalidates the chunk store and leaves the
  vector store alone. Re-chunking the vault then costs the 10 s of chunking and
  extraction rather than the 34 s of a cold build, because an embedding is a
  function of the embed_text alone and almost every chunk comes back with the
  text it had. Phase 1 changed the chunker twice; this is the difference between
  that being free and that being a full re-embed each time.
- A **model change** invalidates the vector store and leaves the chunks alone.
- A **note edit** invalidates that one file's chunks, and only the vectors of
  the chunks whose text actually moved.
- A **rename** invalidates the chunks, because the path is in the key and the
  path is in the chunk, and keeps the vectors, because the path is not in the
  embed_text unless the title moved with it.

**Nothing here may break a build.** Every entry point swallows its own failures:
a corrupt file, an unreadable directory or a full disk costs the cache and not
the index. That is not defensive habit but the whole contract - a warm build and
a cold build have to produce the same chunks and the same matrix, so the cache is
only ever allowed to be an optimisation.

**The fingerprint is derived, not declared.** A hand-maintained version constant
is a gate that fires only when somebody remembers, and the cost of getting it
wrong is asymmetric: an unnecessary invalidation costs ten seconds once, while a
missed one serves stale chunks until somebody notices search has gone strange. So
the chunk store is keyed by a digest of the source of the modules that produce
chunks, plus the settings that change their output. Editing a comment in
chunker.py does invalidate it, and that is the cheap direction to be wrong in.

**Where it lives, and what is in it.** Not in the vault: writing there would trip
the watcher and leave a cache inside somebody's Obsidian. It holds the vault's
text - which is to say finances, insurance and addresses - outside the vault, so
the directory is created 0o700 and the file 0o600. `INDEX_CACHE_PATH=` empty
turns the whole thing off.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import chunker, documents
from .config import settings

log = logging.getLogger(__name__)

# Bumped only when the *file layout* changes. Everything else that invalidates is
# carried in the header as a value, so it can invalidate one store and leave the
# other; a format change can never be that selective.
FORMAT = "vault-mcp/index-cache/1"

# The modules whose output the chunk store holds. Their source is digested into
# the fingerprint, so a change to either re-chunks rather than being trusted to
# have changed nothing. documents.py is here because a filed PDF's chunks *are*
# its extraction, and an extractor change is exactly as invalidating as a chunker
# change while being much easier to forget.
_PIPELINE_MODULES = (chunker, documents)


def _pipeline_settings() -> list[str]:
    """The settings that change what a chunk is.

    Not the exclusions: those decide which files are walked, which the walk
    re-decides on every build, so a folder that comes back into scope should find
    its chunks waiting rather than pay for them a second time.
    """
    return [
        f"chunk_target_tokens={settings.chunk_target_tokens}",
        f"chunk_overlap_tokens={settings.chunk_overlap_tokens}",
        f"chunk_min_tokens={settings.chunk_min_tokens}",
        f"doc_suffixes={','.join(sorted(settings.doc_suffixes))}",
        f"doc_files_dir={settings.doc_files_dir}",
        f"doc_ocr={settings.doc_ocr}",
    ]


def pipeline_fingerprint() -> str:
    """One hex digest standing for "what chunking means right now"."""
    parts = _pipeline_settings()
    for module in _PIPELINE_MODULES:
        source = getattr(module, "__file__", None)
        try:
            digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        except (OSError, TypeError):
            # No source to read - a frozen build, or a module loaded from an
            # archive. Fall back to something that differs on every start, so the
            # chunk store is bypassed rather than trusted blind.
            log.warning("cannot read the source of %s; chunk reuse disabled", module.__name__)
            digest = f"unreadable-{time.time_ns()}"
        parts.append(f"{module.__name__}={digest}")
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def model_id() -> str:
    """What the vector store's contents are embeddings *of*."""
    return f"{settings.embed_model}@{settings.embed_dims}"


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """The key half that says "these bytes". The upload path's hash, reused."""
    return documents.sha256_file(path)


def chunk_key(rel: str, digest: str) -> str:
    # NUL cannot appear in either half, so the join needs no escaping to stay
    # unambiguous.
    return f"{rel}\x00{digest}"


@dataclass(slots=True)
class CacheStats:
    """What the last build got from the cache, for /readyz to report.

    Counted in files and chunks rather than in seconds: the seconds are what the
    build log already says, and these are what explain them.
    """

    files_reused: int = 0
    files_chunked: int = 0
    vectors_reused: int = 0
    vectors_embedded: int = 0

    def as_json(self) -> dict:
        return {
            "files_reused": self.files_reused,
            "files_chunked": self.files_chunked,
            "vectors_reused": self.vectors_reused,
            "vectors_embedded": self.vectors_embedded,
        }


@dataclass(slots=True)
class IndexCache:
    path: Path
    fingerprint: str
    model: str
    vault: str
    chunks_by_file: dict[str, list[dict]] = field(default_factory=dict)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)
    dirty: bool = False
    loaded: bool = False
    write_failures: int = 0

    # ------------------------------------------------------------------ open

    @classmethod
    def open(cls) -> IndexCache | None:
        """The configured cache, or None when it is switched off.

        Never raises. A cache that cannot be opened is a cache that is not used,
        which costs a slower start and nothing else.
        """
        if settings.index_cache_path is None:
            return None
        cache = cls(
            path=settings.index_cache_path,
            fingerprint=pipeline_fingerprint(),
            model=model_id(),
            vault=str(settings.vault_path),
        )
        try:
            cache._read()
        except Exception:
            log.exception("cannot read the index cache at %s; starting cold", cache.path)
            cache.chunks_by_file.clear()
            cache.vectors.clear()
        return cache

    def _read(self) -> None:
        if not self.path.is_file():
            log.info("no index cache at %s yet", self.path)
            return

        # allow_pickle stays off. This file is read at startup with the vault's
        # own privileges, and a cache is not a place to accept arbitrary objects
        # from - not even ones we wrote.
        with np.load(self.path, allow_pickle=False) as archive:
            header = json.loads(bytes(archive["manifest"]).decode("utf-8"))
            if header.get("format") != FORMAT:
                log.info(
                    "index cache is format %r, not %r; discarded",
                    header.get("format"), FORMAT,
                )
                return
            if header.get("vault") != self.vault:
                log.info(
                    "index cache belongs to vault %r, not %r; discarded",
                    header.get("vault"), self.vault,
                )
                return

            # The two stores are checked separately, and that is the point of
            # having two: here is where a chunker change keeps its embeddings and
            # a model change keeps its chunks.
            cached_chunks = header.get("chunks") or {}
            if header.get("fingerprint") == self.fingerprint:
                self.chunks_by_file = cached_chunks
            elif cached_chunks:
                log.info(
                    "the chunk pipeline has changed; %d cached file(s) will be re-chunked",
                    len(cached_chunks),
                )

            keys = header.get("vector_keys") or []
            if header.get("model") == self.model:
                matrix = archive["vectors"]
                if len(keys) != len(matrix):
                    raise ValueError(
                        f"cache holds {len(keys)} vector keys for {len(matrix)} rows"
                    )
                # matrix[i] is a view, so this costs the read and no copy.
                self.vectors = {key: matrix[i] for i, key in enumerate(keys)}
            elif keys:
                log.info(
                    "the embedding model is %r, not the cached %r; %d vector(s) dropped",
                    self.model, header.get("model"), len(keys),
                )

        self.loaded = True
        log.info(
            "index cache: %d file(s) and %d vector(s) from %s",
            len(self.chunks_by_file), len(self.vectors), self.path,
        )

    # ----------------------------------------------------------------- reads

    def chunks_for(self, rel: str, digest: str) -> list[dict] | None:
        found = self.chunks_by_file.get(chunk_key(rel, digest))
        if found is None:
            self.stats.files_chunked += 1
            return None
        self.stats.files_reused += 1
        return found

    def vector_for(self, embed_text: str) -> np.ndarray | None:
        found = self.vectors.get(text_digest(embed_text))
        if found is None:
            self.stats.vectors_embedded += 1
            return None
        self.stats.vectors_reused += 1
        return found

    # ---------------------------------------------------------------- writes

    def store_chunks(self, rel: str, digest: str, chunks: list[dict]) -> None:
        self.chunks_by_file[chunk_key(rel, digest)] = chunks
        self.dirty = True

    def store_vector(self, embed_text: str, row: np.ndarray) -> None:
        self.vectors[text_digest(embed_text)] = np.asarray(row, dtype=np.float32)
        self.dirty = True

    def retain(self, file_keys: set[str], vector_keys: set[str]) -> None:
        """Drop everything the build that just finished did not use.

        Called only after a full build, which is the one moment the live set is
        known exactly. That makes the cache self-pruning: a deleted note, an
        edited paragraph's superseded chunks and a renamed file's old key all
        leave on the next start, so the file is bounded by the size of the vault
        rather than by the length of its history.
        """
        before = (len(self.chunks_by_file), len(self.vectors))
        self.chunks_by_file = {
            key: value for key, value in self.chunks_by_file.items() if key in file_keys
        }
        self.vectors = {key: value for key, value in self.vectors.items() if key in vector_keys}
        after = (len(self.chunks_by_file), len(self.vectors))
        if after != before:
            self.dirty = True
            log.info(
                "index cache pruned: %d -> %d file(s), %d -> %d vector(s)",
                before[0], after[0], before[1], after[1],
            )

    def reset_stats(self) -> None:
        self.stats = CacheStats()

    # ------------------------------------------------------------------ save

    def save(self) -> bool:
        """Write the cache atomically. Returns whether anything was written.

        Call this from a worker thread, and not while a build is mutating the
        stores - server.py holds the reindex lock across it for that reason. It
        serialises inside rather than taking a snapshot first because the
        serialising is the expensive half, and a snapshot would double the peak
        memory of the one thing in this process measured in megabytes.
        """
        if not self.dirty:
            return False
        try:
            self._write()
        except Exception as exc:
            # A cache that cannot be written costs a slower next start. Nothing
            # in here is the index, so nothing in here is worth failing for.
            #
            # Traced once and then counted. Every reason this fails is sticky - a
            # permission, a path, a full disk - so the flush timer would otherwise
            # write a stack trace a minute for as long as the container runs, and
            # bury whatever the operator actually went to the log for.
            self.write_failures += 1
            if self.write_failures == 1:
                log.exception("cannot write the index cache to %s", self.path)
            else:
                log.warning(
                    "index cache still not writable (%d attempts): %s",
                    self.write_failures, exc,
                )
            return False
        self.write_failures = 0
        self.dirty = False
        return True

    def _write(self) -> None:
        keys = list(self.vectors)
        if keys:
            matrix = np.stack([self.vectors[key] for key in keys]).astype(np.float32, copy=False)
        else:
            matrix = np.zeros((0, settings.embed_dims), dtype=np.float32)

        manifest = json.dumps(
            {
                "format": FORMAT,
                "fingerprint": self.fingerprint,
                "model": self.model,
                "vault": self.vault,
                "written_at": time.time(),
                "chunks": self.chunks_by_file,
                "vector_keys": keys,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 0o700 and 0o600. The vault's text is in here, and it is now outside the
        # vault, where nobody has yet thought about who can read it. Suppressed
        # rather than checked because a filesystem without modes is not a reason
        # to refuse to cache.
        with contextlib.suppress(OSError):
            self.path.parent.chmod(0o700)

        writing = self.path.with_name(self.path.name + ".writing")
        with open(writing, "wb") as handle:
            # One archive rather than a manifest beside a matrix: two files
            # replaced in sequence can be interrupted between them, and a header
            # describing rows that are not there is worse than no cache at all.
            np.savez(
                handle,
                vectors=matrix,
                manifest=np.frombuffer(manifest, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(OSError):
            writing.chmod(0o600)
        os.replace(writing, self.path)

        log.info(
            "index cache written: %d file(s), %d vector(s), %.1f MB",
            len(self.chunks_by_file), len(keys), self.path.stat().st_size / 1_048_576,
        )

    # ---------------------------------------------------------------- status

    def as_json(self) -> dict:
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {
            "enabled": True,
            "path": str(self.path),
            "loaded": self.loaded,
            "files": len(self.chunks_by_file),
            "vectors": len(self.vectors),
            "bytes": size,
            "pending_write": self.dirty,
            "write_failures": self.write_failures,
            "last_build": self.stats.as_json(),
        }
