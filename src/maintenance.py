"""Run the vault's own maintenance checkers and return what they printed.

The eight scripts in `.scripts/` are the vault's integrity tests - links,
frontmatter, index coverage, encoding hygiene, generated-series gaps, expiry
dates, unaddressable headings and callout conformance. They are stdlib-only Python 3 and already sit
inside the mount at `/vault/.scripts/`, so this container is the one place that
can run them unchanged, on a schedule, with no second copy to keep in step.

This exists so n8n can run them. n8n's container has neither Python nor the
vault mounted, and the alternative - an SSH node with a key and a shell on the
host - buys a far larger blast radius than one read-only route.

Three things are deliberate:

- **The script list is fixed, not a glob.** `.scripts/` lives in a read-write
  mount, and a directory listing turned into a command line is a directory
  listing turned into code execution. Writes through this service are confined
  to `.md` files, which already forbids planting a `.py` here, but a route that
  runs whatever it finds would be one relaxed guard away from arbitrary
  execution. Adding a checker means adding a line below.
- **No caller input reaches the command line.** There are no query parameters
  that become arguments. Every argv is built from the constants here.
- **Nothing here writes.** `check_vault_hygiene.py --fix` repairs line endings
  and BOMs; it is never passed. A report that says what is wrong is safe to run
  unattended, a repair that runs at 05:00 on a Sunday is not.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

# Per-script wall clock. The whole suite walks ~390 notes in a few seconds; a
# script still running after two minutes is wedged, not slow, and the report is
# more useful with one timeout recorded than with no report at all.
TIMEOUT_SECONDS = 120

SCRIPTS_DIR = ".scripts"


@dataclass(frozen=True, slots=True)
class Check:
    """One checker, and what a reader needs to know to weigh its output."""

    script: str
    title: str
    # What a non-zero exit means for this script specifically. The scripts do
    # not agree on it - the hygiene check exits 0 while printing 27 warnings,
    # the expiry check exits 1 only on a lapsed date - so the exit code alone
    # is not a finding count and a report that treats it as one will be wrong.
    nonzero_means: str
    args: tuple[str, ...] = ()


CHECKS: tuple[Check, ...] = (
    Check(
        "check_vault_links.py",
        "Broken internal links",
        "at least one link resolves to nothing",
    ),
    Check(
        "compare_index_descriptions.py",
        "index.md against frontmatter",
        "index.md has drifted from the notes - the generator is not running",
    ),
    Check(
        "check_frontmatter.py",
        "OKF frontmatter conformance",
        "a note is missing a required field or has a malformed one",
    ),
    Check(
        "check_vault_hygiene.py",
        "Encoding and filesystem hygiene",
        "mixed line endings or a case collision; BOMs and missing final "
        "newlines are warnings and still exit 0",
    ),
    Check(
        "check_generated_output.py",
        "Generated series output",
        "a workflow wrote an empty note, or a dated series has a gap",
    ),
    Check(
        "check_expiries.py",
        "Expiry dates",
        "a date has lapsed or an expires block is malformed; dates due inside "
        "the warning window are warnings and still exit 0",
    ),
    Check(
        "check_duplicate_headings.py",
        "Unaddressable duplicate headings",
        "a note repeats a full heading path, so that section cannot be patched",
    ),
    Check(
        "check_documents.py",
        "Filed document conformance",
        "a document in a Files/ folder has no note linking to it, or a PDF's "
        "text cannot be extracted so its contents reach no index - either one "
        "makes the document unreachable; a document outside a Files/ folder is "
        "a warning and still exits 0",
    ),
    Check(
        "check_callouts.py",
        "Callout conformance",
        "a blockquote has no type, carries a type outside the allowed five, or "
        "a callouts opt-out cannot be honoured - each one drops its contents "
        "out of the callout sweep; wrong-case types and missing titles are "
        "warnings and still exit 0",
    ),
)


# Handed to every checker, and nothing else is. Built rather than inherited
# because this process holds VAULT_MCP_API_KEY, and a stdlib script that walks
# markdown has no use for it - a child process is one `os.environ` dump away
# from putting it in stdout, and stdout here goes into a note.
#
# PYTHONDONTWRITEBYTECODE keeps `__pycache__` out of `.scripts/`. This route is
# meant to leave the vault exactly as it found it, and a directory the host and
# Obsidian both see is not the place to shed build artefacts.
#
# PYTHONIOENCODING pins stdout to UTF-8, so a note title with an em dash
# survives into the report whatever the container locale is.
SUBPROCESS_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    # PATH is not needed - the interpreter is invoked by absolute path - but
    # an empty environment is unusual enough to be worth not being the first
    # thing to blame when a script misbehaves.
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
}


def _run(check: Check, scripts_dir: Path, vault_path: Path) -> dict:
    script_path = scripts_dir / check.script
    result: dict = {
        "script": check.script,
        "title": check.title,
        "nonzero_means": check.nonzero_means,
    }

    if not script_path.is_file():
        # Reported rather than raised: one missing script should not cost the
        # others their run, and "it is not there" is itself a finding.
        result |= {"exit_code": None, "stdout": "", "stderr": "",
                   "duration_ms": 0, "error": f"not found at {script_path}"}
        return result

    argv = [sys.executable, str(script_path), "--vault-path", str(vault_path), *check.args]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            cwd=str(vault_path),
            env=SUBPROCESS_ENV,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return result | {
            "exit_code": None,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": "",
            "duration_ms": round((time.monotonic() - started) * 1000),
            "error": f"timed out after {TIMEOUT_SECONDS}s",
        }
    except OSError as exc:
        return result | {
            "exit_code": None, "stdout": "", "stderr": "",
            "duration_ms": round((time.monotonic() - started) * 1000),
            "error": f"could not be launched: {exc}",
        }

    return result | {
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "duration_ms": round((time.monotonic() - started) * 1000),
        "error": None,
    }


def _render(checks: list[dict], meta: dict) -> str:
    """The same results as one markdown block, ready to paste under a prompt.

    Rendered here rather than in n8n because everything it needs is here: the
    order, the titles, and `nonzero_means`. A Code node would have to be handed
    all three anyway, and putting it there would mean the workflow's copy and
    this list drifting the first time a checker is added.

    What deliberately stays in the workflow is the *instruction* - what to
    prioritise, what to write, what never to touch. That is editorial policy
    about one report, it changes far more often than the check list, and it
    belongs where it can be read and edited without a container rebuild.

    `nonzero_means` is repeated per block for the same reason it exists: the
    scripts do not agree on what a non-zero exit is for. `check_vault_hygiene`
    exits 0 with 27 warnings printed; a reader given one blanket rule would
    file that as a clean check.
    """
    lines = [
        f"Ran at {meta['generated_at']} against `{meta['vault']}` in "
        f"{meta['duration_ms'] / 1000:.1f}s. "
        f"{meta['summary']['exit_zero']} of {meta['summary']['total']} exited zero"
        + (f", {meta['summary']['errored']} failed to run"
           if meta["summary"]["errored"] else "")
        + ".",
    ]

    for position, check in enumerate(checks, start=1):
        lines.append("")
        lines.append(f"### {position}. {check['title']} - `{check['script']}`")
        lines.append("")
        if check["error"]:
            lines.append(f"**This check did not run: {check['error']}**")
            continue
        lines.append(
            f"Exit code `{check['exit_code']}`. A non-zero exit here means "
            f"{check['nonzero_means']}."
        )
        lines.append("")
        output = "\n\n".join(part for part in (check["stdout"], check["stderr"]) if part)
        lines.append("```text")
        lines.append(output or "(no output)")
        lines.append("```")

    return "\n".join(lines)


def run_all() -> dict:
    """Run every checker in order and return one JSON-serialisable result.

    Sequential on purpose. They walk the same tree over the same mount, so
    running them at once trades disk contention for a saving measured against a
    total of a few seconds, and it would scramble the order the report reads in.

    Blocking - `subprocess.run` is. The caller hands this to a thread.
    """
    vault_path = settings.vault_path
    scripts_dir = vault_path / SCRIPTS_DIR
    started = time.monotonic()

    checks = [_run(check, scripts_dir, vault_path) for check in CHECKS]

    result = {
        "vault": str(vault_path),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_ms": round((time.monotonic() - started) * 1000),
        "summary": {
            "total": len(checks),
            "exit_zero": sum(1 for c in checks if c["exit_code"] == 0),
            "exit_nonzero": sum(1 for c in checks if (c["exit_code"] or 0) > 0),
            "errored": sum(1 for c in checks if c["error"]),
        },
        "checks": checks,
    }
    # Both shapes, because they answer different questions. `checks` is what a
    # caller branches on; `markdown` is what goes under a prompt. Deriving the
    # second from the first in every consumer is the duplication this avoids.
    result["markdown"] = _render(checks, result)
    return result
