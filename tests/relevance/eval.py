"""Retrieval relevance: recall@k, MRR and per-note concentration, against a baseline.

    python -m tests.relevance.eval
    python -m tests.relevance.eval --vault /media/Share/Vault --queries private.json
    python -m tests.relevance.eval --embedder ollama
    python -m tests.relevance.eval --update-baseline

The README makes measured claims about this retrieval stack - that dense scores
0/40 on exact identifiers where BM25 scores 40/40, that the lookup override earns
its place - and until now nothing in the repo could re-run them. This is the
harness those claims belong in, and the gate every later change to SPARSE_WEIGHT,
LOOKUP_MAX_MATCHES, the tokeniser or the chunker has to pass.

Two corpora, deliberately:

- The committed fixture under tests/fixtures/vault, which runs anywhere, in CI,
  with the hashing embedder. It reproduces the real vault's *shapes* - identifier
  density, one flat-list note, one long note that chunks six ways, three notes
  sharing a heading name - not its contents.
- The real vault, with a queries file that stays out of git because the queries
  name real accounts. `--vault` and `--queries` are what make that possible, and
  the reason neither is hardcoded.

What a run reports is not only whether a query hit. `max_per_path` is the number
of the k returned chunks that came from one note, and it is the measurement item 1
exists to move: a query answered by five chunks of one file has hidden four other
files. Watch it fall.

Baseline semantics: a query that hit at rank r must not miss, and must not return
a worse rank. An improvement is reported and does not fail. That way the stemming
queries can sit in the file today, failing honestly, and their fix is visible as a
baseline update rather than as a test somebody had to remember to write.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_VAULT = REPO / "tests" / "fixtures" / "vault"
DEFAULT_QUERIES = Path(__file__).parent / "queries.json"
DEFAULT_BASELINE = Path(__file__).parent / "baseline.json"

# Arms whose result decides the exit code. The others are printed and ignored:
# `dense` because the hashing embedder has no semantics to test, `threshold`
# because there is no score floor yet for it to pass.
SCORED_ARMS = frozenset({"lexical", "either"})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tests.relevance.eval", description=__doc__)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--embedder",
        choices=("hashing", "ollama"),
        default="hashing",
        help="hashing is deterministic and offline; ollama is the real one and needs OLLAMA_URL",
    )
    parser.add_argument("--k", type=int, default=None, help="defaults to SEARCH_DEFAULT_K")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="overwrite the baseline with this run instead of comparing against it",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="report only; exit 0 unless a `never` path appeared",
    )
    return parser.parse_args(argv)


# src.config resolves settings at import, so the vault has to be in the
# environment before anything under src is touched. Same reason tests/indexdoc.py
# sets it at module top: there is no second chance after the import.
ARGS = parse_args(sys.argv[1:])
if not ARGS.vault.is_dir():
    sys.exit(f"--vault {ARGS.vault} is not a directory")
os.environ["VAULT_PATH"] = str(ARGS.vault.resolve())
os.environ.setdefault("VAULT_MCP_API_KEY", "eval")

from src import search as search_module  # noqa: E402
from src.config import settings  # noqa: E402
from src.index import VaultIndex  # noqa: E402

from .hashing_embedder import HashingEmbedder  # noqa: E402


def load_queries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload["queries"] if isinstance(payload, dict) else payload
    for entry in queries:
        entry.setdefault("arm", "either")
        entry.setdefault("expect", [])
        entry.setdefault("never", [])
    return queries


def _best_rank(results: list[dict], expected: list[str]) -> int | None:
    wanted = set(expected)
    for position, result in enumerate(results, start=1):
        if result["path"] in wanted:
            return position
    return None


def _max_per_path(results: list[dict]) -> int:
    counts: dict[str, int] = {}
    for result in results:
        counts[result["path"]] = counts.get(result["path"], 0) + 1
    return max(counts.values(), default=0)


async def run(queries: list[dict], k: int, embedder_name: str) -> list[dict]:
    if embedder_name == "ollama":
        from src.embedder import Embedder

        embedder = Embedder()
    else:
        embedder = HashingEmbedder()

    try:
        index = await VaultIndex.build(embedder)
        print(f"index: {index.summary()}\n")
        if index.size == 0:
            sys.exit("the index is empty - is --vault pointing at a vault with notes in it?")

        rows: list[dict] = []
        for entry in queries:
            results = await search_module.search(index, embedder, entry["q"], k)
            paths = [r["path"] for r in results]
            rows.append(
                {
                    "q": entry["q"],
                    "arm": entry["arm"],
                    "rank": _best_rank(results, entry["expect"]),
                    "returned": len(results),
                    "max_per_path": _max_per_path(results),
                    "forbidden": sorted(set(paths) & set(entry["never"])),
                    "top": paths[0] if paths else None,
                }
            )
        return rows
    finally:
        await embedder.aclose()


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["arm"] in SCORED_ARMS]
    hits = [r for r in scored if r["rank"] is not None]
    reciprocal = sum(1.0 / r["rank"] for r in hits)
    concentration = [r["max_per_path"] for r in scored if r["returned"]]
    return {
        "scored_queries": len(scored),
        "hits": len(hits),
        "recall": round(len(hits) / len(scored), 4) if scored else 0.0,
        "mrr": round(reciprocal / len(scored), 4) if scored else 0.0,
        "mean_max_per_path": (
            round(sum(concentration) / len(concentration), 3) if concentration else 0.0
        ),
        "worst_max_per_path": max(concentration, default=0),
    }


def report(rows: list[dict], aggregate: dict, k: int) -> None:
    width = max(len(r["q"]) for r in rows)
    print(f"{'query'.ljust(width)}  arm        rank  1/path  result")
    print("-" * (width + 34))
    for row in rows:
        rank = "-" if row["rank"] is None else str(row["rank"])
        if row["forbidden"]:
            verdict = "EXCLUDED PATH RETURNED: " + ", ".join(row["forbidden"])
        elif row["arm"] not in SCORED_ARMS:
            # The top path matters most on the informational misses: a threshold
            # query returning five chunks is only legible once you see what they
            # were, and "the least-bad chunk in the vault" is the finding.
            verdict = (
                "hit (informational)"
                if row["rank"]
                else f"miss (informational, top was {row['top']})"
            )
        elif row["rank"] == 1:
            verdict = "hit"
        elif row["rank"] is not None:
            verdict = f"hit at {row['rank']}"
        else:
            verdict = f"MISS (top was {row['top']})"
        print(
            f"{row['q'].ljust(width)}  {row['arm']:<9}  {rank:>4}  "
            f"{row['max_per_path']:>6}  {verdict}"
        )

    print(
        f"\nk={k}  scored={aggregate['scored_queries']}  "
        f"recall@{k}={aggregate['recall']}  MRR={aggregate['mrr']}  "
        f"mean chunks from one note={aggregate['mean_max_per_path']}  "
        f"worst={aggregate['worst_max_per_path']}"
    )


def compare(rows: list[dict], baseline: dict) -> tuple[list[str], list[str]]:
    """Regressions and improvements, per query, against a previous run."""
    was = baseline.get("queries", {})
    regressions: list[str] = []
    improvements: list[str] = []

    for row in rows:
        if row["arm"] not in SCORED_ARMS:
            continue
        previous = was.get(row["q"])
        if previous is None:
            improvements.append(f"{row['q']!r} is new to the baseline")
            continue
        before, after = previous.get("rank"), row["rank"]
        if before is not None and after is None:
            regressions.append(f"{row['q']!r} hit at {before} and now misses")
        elif before is not None and after > before:
            regressions.append(f"{row['q']!r} fell from rank {before} to {after}")
        elif before is None and after is not None:
            improvements.append(f"{row['q']!r} now hits at rank {after} (was a miss)")
        elif before is not None and after < before:
            improvements.append(f"{row['q']!r} rose from rank {before} to {after}")

    for query in set(was) - {r["q"] for r in rows}:
        improvements.append(f"{query!r} is in the baseline but no longer in the queries file")

    return regressions, improvements


def main() -> int:
    k = ARGS.k or settings.search_default_k
    queries = load_queries(ARGS.queries)
    print(
        f"vault={settings.vault_path}  embedder={ARGS.embedder}  "
        f"queries={ARGS.queries.name} ({len(queries)})"
    )
    if ARGS.embedder == "hashing":
        print(
            "note: the hashing embedder has no semantics, so `dense` queries are "
            "reported but not scored"
        )
    print()

    rows = asyncio.run(run(queries, k, ARGS.embedder))
    aggregate = summarise(rows)
    report(rows, aggregate, k)

    leaked = [r for r in rows if r["forbidden"]]
    if leaked:
        print(
            f"\n{len(leaked)} query/queries returned a path listed under `never`. "
            "That is a correctness failure regardless of the baseline: an excluded "
            "folder reached the index."
        )

    snapshot = {
        "k": k,
        "vault": str(settings.vault_path.name),
        "embedder": ARGS.embedder,
        "queries_file": ARGS.queries.name,
        "aggregate": aggregate,
        "queries": {
            r["q"]: {"rank": r["rank"], "max_per_path": r["max_per_path"]}
            for r in rows
            if r["arm"] in SCORED_ARMS
        },
    }

    if ARGS.update_baseline:
        ARGS.baseline.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nbaseline written to {ARGS.baseline}")
        return 1 if leaked else 0

    if ARGS.no_baseline:
        return 1 if leaked else 0

    if not ARGS.baseline.exists():
        print(
            f"\nno baseline at {ARGS.baseline} - run with --update-baseline once the "
            "numbers above look right"
        )
        return 1 if leaked else 0

    baseline = json.loads(ARGS.baseline.read_text(encoding="utf-8"))
    # A rank is only comparable against a rank measured the same way. k changes
    # what "rank 5" even means, and the two embedders disagree about everything.
    mismatched = {
        key: (baseline.get(key), mine)
        for key, mine in (
            ("embedder", ARGS.embedder),
            ("queries_file", ARGS.queries.name),
            ("k", k),
        )
        if baseline.get(key) != mine
    }
    if mismatched:
        detail = ", ".join(f"{key} {was!r} != {now!r}" for key, (was, now) in mismatched.items())
        print(f"\nnot comparing - {detail}. Keep one baseline per corpus and k.")
        return 1 if leaked else 0

    regressions, improvements = compare(rows, baseline)
    for line in improvements:
        print(f"  improved: {line}")
    for line in regressions:
        print(f"  REGRESSION: {line}")

    if regressions:
        print(
            f"\n{len(regressions)} regression(s) against {ARGS.baseline.name}. If the "
            "change was intended, re-run with --update-baseline and commit the diff."
        )
        return 1
    if improvements:
        print("\nno regressions, and some queries improved - --update-baseline to record it")
    else:
        print("\nno change against the baseline")
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
