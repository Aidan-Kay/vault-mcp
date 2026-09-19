"""Tokenising and result selection, at the unit level.

The relevance suite scores whole queries against a corpus and is the right place
to ask whether retrieval got better. It is the wrong place to ask why: an
aggregate of 29 queries moves for reasons that are hard to attribute, and two
changes that cancel out look like no change at all. These are the pieces, each
asserted on its own.

The cases that matter most here are the ones the fixture cannot reach. A corpus
of 27 notes never exhausts its candidates, so the cap's backfill branch never
runs in the relevance suite - and a branch nobody has seen run is not known to
work.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_VAULT = Path(tempfile.mkdtemp(prefix="vault-retrieval-"))
os.environ["VAULT_PATH"] = str(_VAULT)

from src import search  # noqa: E402
from src.index import STOP_WORDS, tokenize  # noqa: E402

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


def tokeniser() -> None:
    # The deliberate part, unchanged by the stemmer: three tokens, not one, so a
    # query for any part of a hyphenated name still matches.
    check("nomic-embed-text splits", len(tokenize("nomic-embed-text")), 3)
    check("nomic survives the stemmer", tokenize("nomic")[0], "nomic")

    # The two misses the relevance fixture carried until stemming landed.
    check("readings meets reading", tokenize("readings"), tokenize("reading"))
    check("renewing meets renewal", tokenize("renewing"), tokenize("renewals"))
    check("policies meets policy", tokenize("policies"), tokenize("policy"))

    check("stop words go", tokenize("the escape of water claim"), tokenize("escape water claim"))
    check("a query of only stop words is empty", tokenize("what is the"), [])
    # Stemmed into one: dropped on the second pass, not the first.
    check("having is dropped once stemmed", tokenize("having"), [])

    # Identifiers are the arm this corpus leans on, and the stemmer must not
    # touch them. Case and punctuation are all it is allowed to change.
    for identifier in ("KE-8842071", "T4204X14-889201", "10.0.4.12", "E12N4471088", "235/45 R18"):
        check(f"{identifier} is unchanged by the stemmer", tokenize(identifier), tokenize(identifier.lower()))
    check("an identifier keeps its pieces", tokenize("KE-8842071"), ["ke", "8842071"])
    check("no identifier piece is a stop word", {"ke", "8842071", "nx8010"} & STOP_WORDS, set())


def normalisation() -> None:
    both = search.fuse([7], [7])
    check("rank one in both lists is 1.0", round(both[0][1], 10), 1.0)

    dense_only = search.fuse([7], [])
    check("dense alone at rank one", round(dense_only[0][1], 4), round(1 / (1 + search.SPARSE_WEIGHT), 4))

    sparse_only = search.fuse([], [7])
    expected = search.SPARSE_WEIGHT / (1 + search.SPARSE_WEIGHT)
    check("BM25 alone at rank one", round(sparse_only[0][1], 4), round(expected, 4))

    ranked = search.fuse(list(range(20)), list(range(19, -1, -1)))
    check("every score lands in (0, 1]", all(0 < s <= 1 for _, s in ranked), True)
    check("the order still descends", [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True), True)


def caps() -> None:
    check("k=5 caps a note at two slots", search.per_path_cap(5), 2)
    check("k=1 still allows one", search.per_path_cap(1), 1)
    check("k=3 caps at one", search.per_path_cap(3), 1)
    check("k=20 caps at seven", search.per_path_cap(20), 7)


def _chunks(paths: list[str]) -> list[dict]:
    return [{"path": path} for path in paths]


CHUNKS = _chunks(["long.md"] * 6 + ["a.md", "b.md", "c.md", "d.md"])

# Six chunks of one long note ahead of every other note - the single-source
# dominance case. Taking the top five off this list unselected is five chunks of
# long.md and four notes hidden, which is the defect the cap exists for.
DOMINANT = [(i, 1.0 - i * 0.01) for i in range(6)] + [(i, 0.2 - i * 0.01) for i in range(6, 10)]

# The same six chunks, but the other notes are close enough behind to compete.
CLOSE = [(i, 1.0 - i * 0.02) for i in range(10)]


def selection() -> None:
    chosen = search.diversify(DOMINANT, CHUNKS, 5)
    paths = [CHUNKS[doc]["path"] for doc, _ in chosen]
    check("five results come back", len(chosen), 5)
    check("the cap stops the fifth slot going to one note", paths.count("long.md"), 2)
    check("four other notes get in", len(set(paths)), 4)
    check("the top hit is still the top hit", chosen[0][0], 0)
    check("scores are the fused ones, not the MMR ones", chosen[0][1], 1.0)
    check("a clear second chunk is not refused", [doc for doc, _ in chosen][:2], [0, 1])

    # The cap is a ceiling, not a quota. When other notes are competitive the
    # penalty spends the slots on them and the cap never binds.
    paths = [CHUNKS[doc]["path"] for doc, _ in search.diversify(CLOSE, CHUNKS, 5)]
    check("a competitive field pushes concentration below the cap", paths.count("long.md"), 1)
    check("and fills the rest with distinct notes", len(set(paths)), 5)


def pinning() -> None:
    chosen = search.diversify(CLOSE, CHUNKS, 5, pinned=7)
    check("the pinned hit leads", chosen[0][0], 7)
    check("it carries its own fused score", chosen[0][1], dict(CLOSE)[7])
    check("nothing is returned twice", len({doc for doc, _ in chosen}), 5)
    check(
        "a pin outranks a chunk that beat it on fusion",
        chosen[0][1] < max(score for _, score in chosen),
        True,
    )

    # Pinning a chunk of the dominant note must not buy that note a third slot.
    chosen = search.diversify(DOMINANT, CHUNKS, 5, pinned=3)
    paths = [CHUNKS[doc]["path"] for doc, _ in chosen]
    check("a pin counts against its own cap", paths.count("long.md"), 2)
    check("the pin is still first", chosen[0][0], 3)


def backfill() -> None:
    """The branch the fixture corpus is too large to reach.

    When every candidate left belongs to a note already at its cap there is no
    other note being hidden, which is the only thing the cap protects. The
    caller asked for k, so the cap reorders rather than truncates.
    """
    chunks = _chunks(["only.md"] * 6)
    ranked = [(i, 1.0 - i * 0.05) for i in range(6)]

    chosen = search.diversify(ranked, chunks, 5)
    check("k results come back from one note", len(chosen), 5)
    check("they are the best five, in order", [doc for doc, _ in chosen], [0, 1, 2, 3, 4])

    # Nothing to backfill from: fewer than k is the honest answer.
    chosen = search.diversify(ranked[:2], _chunks(["only.md"] * 2), 5)
    check("fewer candidates than k returns fewer", len(chosen), 2)
    check("an empty candidate list returns nothing", search.diversify([], [], 5), [])


def main() -> int:
    tokeniser()
    normalisation()
    caps()
    selection()
    pinning()
    backfill()

    if report():
        return 1
    print("retrieval: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_VAULT, ignore_errors=True)
