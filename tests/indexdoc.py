"""index.md generation: coverage, ordering, incremental updates, and churn.

The document this builds is the vault's navigation surface, and it is now
derived rather than written, so the things worth asserting are the ones a human
editor used to guarantee by hand: every note appears exactly once, the headings
follow the folders, a description change reaches the index, and a note that
vanishes takes its entry with it.

The last property is the one that is easy to lose and expensive to have lost: an
edit that changes nothing an entry displays must not rewrite index.md at all,
because this vault is in git and a document that churns on every write buries
its own history.

Needs a writable vault, so VAULT_PATH is pointed at a temp tree before src is
imported - src.config resolves settings once, at import.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_VAULT = Path(tempfile.mkdtemp(prefix="vault-indexdoc-"))
os.environ["VAULT_PATH"] = str(_VAULT)

from src import indexdoc, operations, vault  # noqa: E402

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual == expected:
        return
    FAILURES.append(f"{name}\n    expected: {expected!r}\n    actual:   {actual!r}")


def contains(name: str, haystack: str, needle: str) -> None:
    if needle not in haystack:
        FAILURES.append(f"{name}\n    missing line: {needle!r}")


def missing(name: str, haystack: str, needle: str) -> None:
    if needle in haystack:
        FAILURES.append(f"{name}\n    unexpected line: {needle!r}")


def refused(name: str, fn) -> None:
    try:
        fn()
    except vault.VaultError:
        return
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{name}\n    expected VaultError, got {type(exc).__name__}: {exc}")
        return
    FAILURES.append(f"{name}\n    expected VaultError, but the call succeeded")


def report() -> int:
    if not FAILURES:
        return 0
    print(f"{len(FAILURES)} failure(s):\n")
    for failure in FAILURES:
        print(failure + "\n")
    return 1


def note(title: str, description: str) -> str:
    return f"---\ntype: note\ntitle: {title}\ndescription: {description}\n---\n\nbody\n"


def write(rel: str, text: str) -> None:
    path = _VAULT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def seed() -> None:
    write("AGENTS.md", note("AGENTS", "Instructions at the vault root."))
    write("Home/Home.md", note("Home", "The home landing note."))
    write("Home/Boiler.md", note("Boiler", "A combi boiler."))
    write("Home/Council Tax.md", note("Council Tax", "Band and billing."))
    write("Home/Utilities/Gas.md", note("Gas", "The gas account."))
    write("Pets/Levi.md", note("Levi", "A dog."))
    # A generated series: excluded by config, so it must never gain an entry.
    write("Workflows/Approvals/2026-01-01 Thing.md", "proposal body\n")
    # No frontmatter at all, like the prompt-injected files in the real vault.
    write("Meta/Raw.md", "# Raw Heading\n\nFirst sentence here. Second one after it.\n")
    # Both of YAML's ways of wrapping a description across lines. The index
    # entry is one line, so each has to arrive as one line - and the block
    # scalar's `>-` header is syntax, not part of the value.
    write(
        "Meta/Folded.md",
        "---\ntitle: Folded\ndescription: >-\n  A description the author wrapped\n"
        "  across two lines.\n---\n\nbody\n",
    )
    write(
        "Meta/Plain.md",
        "---\ntitle: Plain\ndescription: A description the author wrapped\n"
        "  across two lines.\n---\n\nbody\n",
    )


def rebuild() -> tuple[indexdoc.IndexDoc, str]:
    doc = indexdoc.IndexDoc.build()
    doc.write_if_changed()
    return doc, (_VAULT / "index.md").read_text(encoding="utf-8")


def main() -> int:
    seed()
    doc, text = rebuild()

    # --- coverage -----------------------------------------------------------
    check("every note but the generated one has an entry", len(doc.entries), 9)
    missing("a generated series is never indexed", text, "2026-01-01 Thing")
    missing("index.md does not index itself", text, "](index.md)")

    # --- headings mirror the folder tree ------------------------------------
    contains("top-level folder is an H2", text, "\n## Home\n")
    contains("nested folder is an H3", text, "\n### Utilities\n")
    contains("a folder with no notes of its own still gets a heading", text, "\n## Workflows\n")
    contains("the excluded folder is explained, not silently absent", text,
             "> [!NOTE] `Workflows/Approvals/` is not indexed note by note")

    # --- entries ------------------------------------------------------------
    contains("entry carries the note's own description", text, "- [Levi](Pets/Levi.md) - A dog.")
    contains("spaces in a target are percent-encoded", text, "](Home/Council%20Tax.md)")
    missing("no unencoded space survives", text, "](Home/Council Tax.md)")

    # A root note is listed before the first heading, where it is actually findable.
    check(
        "root notes precede the first folder heading",
        text.index("](AGENTS.md)") < text.index("\n## Home\n"),
        True,
    )

    # The landing note leads its section regardless of the alphabet: B sorts
    # before H, so ordering by title alone would bury Home.md under Boiler.
    home = text[text.index("\n## Home\n") :]
    check(
        "the folder's landing note comes first",
        home.index("](Home/Home.md)") < home.index("](Home/Boiler.md)"),
        True,
    )

    # --- fallbacks for a note with no frontmatter ---------------------------
    contains("title falls back to the note's H1", text, "- [Raw Heading](Meta/Raw.md)")
    contains("description falls back to the first sentence", text, "Raw.md) - First sentence here.")

    # --- a wrapped description arrives as one line, either way it is written --
    wrapped = "A description the author wrapped across two lines."
    contains("a plain wrapped description is joined", text, f"](Meta/Plain.md) - {wrapped}")
    contains("a block scalar is joined the same way", text, f"](Meta/Folded.md) - {wrapped}")
    # The header is YAML syntax. Left in, it renders into the navigation
    # document as literal punctuation in front of every folded description.
    missing("and its '>-' header never reaches the page", text, ">- A description")

    # --- no churn -----------------------------------------------------------
    # The rendered timestamp changes on every render, so comparing whole files
    # would report a change every time. Only the body counts.
    check("a second write is a no-op", indexdoc.IndexDoc.build().write_if_changed(), None)

    before = (_VAULT / "index.md").read_bytes()
    write("Home/Boiler.md", note("Boiler", "A combi boiler.") + "\nan extra paragraph\n")
    doc = doc.replace_note(_VAULT / "Home" / "Boiler.md")
    doc.write_if_changed()
    check(
        "editing a body the index does not show leaves index.md alone",
        (_VAULT / "index.md").read_bytes(),
        before,
    )

    # --- incremental updates track the vault --------------------------------
    operations.set_frontmatter("Pets/Levi.md", "description", "A labrador.")
    doc = doc.replace_note(_VAULT / "Pets" / "Levi.md")
    doc.write_if_changed()
    text = (_VAULT / "index.md").read_text(encoding="utf-8")
    contains("a description change reaches the index", text, "- [Levi](Pets/Levi.md) - A labrador.")

    operations.write("Pets/Cat.md", note("Cat", "A cat."))
    doc = doc.replace_note(_VAULT / "Pets" / "Cat.md")
    doc.write_if_changed()
    text = (_VAULT / "index.md").read_text(encoding="utf-8")
    contains("a new note gains an entry", text, "- [Cat](Pets/Cat.md) - A cat.")

    operations.move("Pets/Cat.md", "Home/Cat.md")
    doc = doc.replace_note(_VAULT / "Pets" / "Cat.md")
    doc = doc.replace_note(_VAULT / "Home" / "Cat.md")
    doc.write_if_changed()
    text = (_VAULT / "index.md").read_text(encoding="utf-8")
    contains("a moved note lands under its new folder", text, "- [Cat](Home/Cat.md) - A cat.")
    missing("and leaves nothing behind at the old path", text, "](Pets/Cat.md)")

    operations.delete("Home/Cat.md")
    doc = doc.replace_note(_VAULT / "Home" / "Cat.md")
    doc.write_if_changed()
    text = (_VAULT / "index.md").read_text(encoding="utf-8")
    missing("a deleted note loses its entry", text, "](Home/Cat.md)")

    # A note the index does not carry must not even cost a render.
    approval = _VAULT / "Workflows" / "Approvals" / "2026-01-01 Thing.md"
    check("a generated-series change is a no-op", doc.replace_note(approval) is doc, True)

    # --- index.md is generated, so writing it by hand is refused ------------
    before = (_VAULT / "index.md").read_bytes()
    refused("write to index.md", lambda: operations.write("index.md", "x", overwrite=True))
    refused("append to index.md", lambda: operations.append("index.md", "x"))
    refused("patch index.md", lambda: operations.patch("index.md", "Home", "replace", "x"))
    refused("delete index.md", lambda: operations.delete("index.md"))
    refused(
        "set frontmatter on index.md",
        lambda: operations.set_frontmatter("index.md", "title", "hijacked"),
    )
    check("index.md untouched by any of them", (_VAULT / "index.md").read_bytes(), before)

    # Reads are not restricted - a model still has to be able to navigate.
    try:
        vault.read_note("index.md")
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"index.md must stay readable\n    got {type(exc).__name__}: {exc}")

    if report():
        return 1
    print("indexdoc: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_VAULT, ignore_errors=True)
