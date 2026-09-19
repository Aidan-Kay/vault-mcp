"""Run the whole suite.

    python -m tests.run                 # everything that runs offline
    python -m tests.run --real-vault    # point the vault readers at VAULT_PATH
    python -m tests.run resolve_all rest

Until now the suite was a set of scripts and a README listing them, which means it
was several things to remember rather than one thing to run, and nothing in the
repo could fail as a unit. This is that one thing.

One subprocess per script, not one process, and that is not incidental:
`src.config` resolves settings at import and `tests/indexdoc.py` points VAULT_PATH
at a temp tree before importing src. Two scripts wanting two different vaults
cannot coexist in one interpreter, so process isolation is the only arrangement
that works - and it also means one script segfaulting or calling sys.exit cannot
take the run with it.

The vault readers default to the committed fixture rather than the real vault, so
a clean checkout passes with nothing mounted. `--real-vault` is how you get the
old behaviour, and it is worth running before a release: the fixture reproduces
the real vault's shapes, but only the real vault has the notes somebody actually
wrote.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "vault"

# Scripts that read a vault and assert against whatever is in it. They run
# against the fixture by default and the real vault under --real-vault.
VAULT_READERS = ("resolve_all", "resolve_leaves", "primitives")

# Scripts that build their own temp vault at import. VAULT_PATH is irrelevant to
# them, and passing one would be misleading rather than harmful - chunker and
# retrieval go further and never touch the tree at all, because they test
# functions that take text rather than paths. They point VAULT_PATH at an empty
# one only because src.config refuses to resolve without it.
SELF_CONTAINED = (
    "write_scope", "documents", "chunker", "retrieval", "indexdoc", "rest",
    "cache", "embedder",
)

RELEVANCE = ("relevance.eval",)

ALL = VAULT_READERS + SELF_CONTAINED + RELEVANCE


def run_one(module: str, env: dict[str, str]) -> tuple[bool, float, str]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", f"tests.{module}"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, elapsed, output


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tests.run", description=__doc__)
    parser.add_argument(
        "only",
        nargs="*",
        help=f"scripts to run; defaults to all of {', '.join(ALL)}",
    )
    parser.add_argument(
        "--real-vault",
        action="store_true",
        help="run the vault readers against VAULT_PATH instead of the fixture",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print output only for scripts that fail",
    )
    args = parser.parse_args(argv)

    selected = args.only or list(ALL)
    unknown = [name for name in selected if name not in ALL]
    if unknown:
        return parser.error(f"unknown script(s): {', '.join(unknown)}") or 2

    vault = os.environ.get("VAULT_PATH") if args.real_vault else str(FIXTURE)
    if args.real_vault and not vault:
        return parser.error("--real-vault needs VAULT_PATH set") or 2
    if not args.real_vault and not FIXTURE.is_dir():
        return parser.error(f"fixture vault missing at {FIXTURE}") or 2

    print(f"vault readers against: {vault}\n")

    results: list[tuple[str, bool, float]] = []
    for module in selected:
        env = dict(os.environ)
        env["VAULT_MCP_API_KEY"] = env.get("VAULT_MCP_API_KEY", "test")
        env["PYTHONIOENCODING"] = "utf-8"
        # No script may write an index cache into whoever ran this suite's home
        # directory. tests/cache.py points it at its own temp tree before it
        # imports src, so this is the default rather than a restriction.
        env["INDEX_CACHE_PATH"] = ""
        if module in VAULT_READERS or module in RELEVANCE:
            env["VAULT_PATH"] = vault
        else:
            # Let the script choose its own temp vault, as it does at import.
            env.pop("VAULT_PATH", None)

        passed, elapsed, output = run_one(module, env)
        results.append((module, passed, elapsed))
        status = "pass" if passed else "FAIL"
        print(f"{status}  tests.{module}  ({elapsed:.1f}s)")
        if output.strip() and (not passed or not args.quiet):
            print("\n".join("    " + line for line in output.strip().splitlines()))
            print()

    failed = [name for name, passed, _ in results if not passed]
    total = sum(elapsed for _, _, elapsed in results)
    print(f"\n{len(results) - len(failed)}/{len(results)} passed in {total:.1f}s")
    if failed:
        print("failed: " + ", ".join(f"tests.{name}" for name in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
