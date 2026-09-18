"""Run the vault's callout extractor and hand back what it found, as JSON.

`.scripts/extract_callouts.py` gathers every open callout in the vault - the
warnings, the tokens with a renewal date, the claims a fact-check could not
settle - into one list. Scattered through the tree they are invisible; in a
list they can be raised. The script deliberately does not judge which ones
still matter, because only reading the text can, so the caller is an n8n
workflow that hands the list to Lyra.

This route exists for the same reason `/maintenance` does: n8n's container has
neither Python nor the vault mounted, and this one has both. The same three
rules hold, for the same reasons -

- **One named script, not a glob.** `.scripts/` is a read-write mount; a
  directory listing turned into a command line is arbitrary execution one
  relaxed guard away.
- **No caller input reaches the command line.** The script takes filters
  (`--type`, `--path`, `--all-types`); none of them are exposed here. Every
  argv below is a constant. The answer is a structured list, so a workflow
  that wants one folder or one type filters the JSON it gets back, which costs
  a workflow expression and no attack surface.
- **Nothing here writes.** Extraction reads notes and prints; that is all.

Unlike `/maintenance`, there is no markdown rendering. The checkers print prose
that had to be wrapped to be usable under a prompt; this script already emits
the shape a consumer wants - one object per callout, sorted by severity - and a
second rendering of it would be a copy to keep in step for no reader.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .config import settings
from .maintenance import SCRIPTS_DIR, SUBPROCESS_ENV, TIMEOUT_SECONDS

SCRIPT = "extract_callouts.py"


class ExtractionError(RuntimeError):
    """The extractor did not produce a result. Distinct from it finding nothing.

    An empty vault is a successful extraction and a 200; only this is a 500.
    That split is what lets the workflow's error path fire on a broken run
    rather than on a quiet week.
    """


def run() -> dict:
    """Extract every open callout and return the script's own JSON, plus timing.

    Blocking - `subprocess.run` is. The caller hands this to a thread.
    """
    vault_path: Path = settings.vault_path
    script_path = vault_path / SCRIPTS_DIR / SCRIPT

    if not script_path.is_file():
        raise ExtractionError(f"{SCRIPT} is not there: looked at {script_path}")

    argv = [sys.executable, str(script_path), "--vault-path", str(vault_path), "--json"]
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
        raise ExtractionError(f"{SCRIPT} timed out after {TIMEOUT_SECONDS}s") from exc
    except OSError as exc:
        raise ExtractionError(f"{SCRIPT} could not be launched: {exc}") from exc

    duration_ms = round((time.monotonic() - started) * 1000)

    # The script exits 0 whether or not it found anything - finding callouts is
    # a normal state of the vault, not a fault - so a non-zero exit here is the
    # script itself having failed, and stderr is the only thing that will say
    # why. Truncated because a traceback belongs in the log, not in a note.
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ExtractionError(
            f"{SCRIPT} exited {completed.returncode}"
            + (f": {detail[-500:]}" if detail else " and said nothing")
        )

    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        raise ExtractionError(f"{SCRIPT} did not emit JSON: {exc}") from exc

    if not isinstance(payload, dict) or "callouts" not in payload:
        raise ExtractionError(f"{SCRIPT} emitted JSON without a callouts list")

    # Returned as the script wrote it - `generated`, `vault`, `counts`,
    # `callouts` - so the workflow reads the same field names whether it runs
    # the script here or on the host. Only the timing is added.
    return payload | {"duration_ms": duration_ms}
