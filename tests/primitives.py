"""Write primitives: the traps that are silent corruption rather than errors.

No pytest on this host (the slim image has no pip), so this is a plain runner.
Filesystem writes go to a temp directory, never the vault; the vault is touched
read-only, to assert that path safety rejects what it should.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

from src import edit, target, vault

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual == expected:
        return
    FAILURES.append(f"{name}\n    expected: {expected!r}\n    actual:   {actual!r}")


def raises(name: str, fn, fragment: str) -> None:
    try:
        fn()
    except vault.VaultError as exc:
        if fragment not in str(exc):
            FAILURES.append(f"{name}\n    expected error containing {fragment!r}\n    got: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - a non-VaultError is itself the failure
        FAILURES.append(f"{name}\n    raised {type(exc).__name__}: {exc}")
        return
    FAILURES.append(f"{name}\n    expected VaultError containing {fragment!r}, nothing raised")


def allows(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{name}\n    unexpectedly raised {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------

FM = """---
type: reference
title: Example
description: A note.
tags: [one, two]
timestamp: 2026-01-01T00:00:00Z
---

# Example

Body text.

## Section

Section text.
"""


def test_line_endings() -> None:
    check("CRLF folded to LF", vault.normalise_body("a\r\nb\r\n"), "a\nb\n")
    check("bare CR folded to LF", vault.normalise_body("a\rb"), "a\nb\n")
    check("final newline added", vault.normalise_body("a"), "a\n")
    check("trailing newlines collapsed", vault.normalise_body("a\n\n\n"), "a\n")
    check("empty stays empty", vault.normalise_body(""), "")


def test_frontmatter_order_preserved() -> None:
    out = vault.set_frontmatter(FM, "timestamp", "2026-09-02T12:00:00Z")
    keys = [line.split(":")[0] for line in out.splitlines()[1:6]]
    check(
        "field order survives a timestamp bump",
        keys,
        ["type", "title", "description", "tags", "timestamp"],
    )
    check("tags stay inline", "tags: [one, two]" in out, True)
    check("body untouched", out.endswith("## Section\n\nSection text.\n"), True)


def test_frontmatter_insert_position() -> None:
    stripped = FM.replace("description: A note.\n", "")
    out = vault.set_frontmatter(stripped, "description", "Restored.")
    keys = [line.split(":")[0] for line in out.splitlines()[1:6]]
    check(
        "a missing key lands in OKF order, not appended",
        keys,
        ["type", "title", "description", "tags", "timestamp"],
    )


def test_expires_block() -> None:
    out = vault.set_frontmatter(
        FM, "expires", [{"date": "2027-03-14", "what": "Nissan Qashqai MOT due"}]
    )
    check(
        "expires renders as a block sequence after timestamp",
        "timestamp: 2026-01-01T00:00:00Z\nexpires:\n  - date: 2027-03-14\n"
        "    what: Nissan Qashqai MOT due\n" in out,
        True,
    )

    two = vault.set_frontmatter(
        out,
        "expires",
        [
            {"date": "2027-10-31", "what": "ScottishPower tariff ends"},
            {"date": "2027-02-09", "what": "akay.io renews"},
        ],
    )
    check("every entry kept, none collapsed", two.count("  - date:"), 2)

    none = vault.set_frontmatter(two, "expires", "none")
    check("expires: none replaces the whole block", "  - date:" in none, False)
    check("expires: none is a scalar", "expires: none" in none, True)

    reasoned = vault.set_frontmatter(none, "expires_reason", "historical record")
    check(
        "expires_reason follows expires",
        "expires: none\nexpires_reason: historical record" in reasoned,
        True,
    )

    removed = vault.set_frontmatter(reasoned, "expires", delete=True)
    check("delete removes the key", "expires:" in removed, False)
    check("delete leaves siblings alone", "expires_reason:" in removed, True)


def test_no_quotes_added() -> None:
    out = vault.set_frontmatter(FM, "title", "Example")
    check("scalar written unquoted", "title: Example" in out, True)



def test_body_is_content_without_frontmatter() -> None:
    """`body` is textual, so it removes a block whether or not it parses.

    Nothing is lost by that, because `content` is untouched - which is the
    trade the structured read is built on.
    """
    note = "---\ntitle: Alpha\n---\n\n# Alpha\n\nProse.\n"
    check("frontmatter removed", vault.without_frontmatter(note), "# Alpha\n\nProse.\n")

    # No block at all: the note is already its own body.
    check("no frontmatter is a no-op", vault.without_frontmatter("# Alpha\n"), "# Alpha\n")

    # A leading `---` is a fence everywhere in this server - frontmatter_span,
    # iter_headings and metadata all read it that way - so body agrees with them
    # rather than inventing a third rule for what opens a note.
    check(
        "a leading fence is frontmatter here, whatever it was meant to be",
        vault.without_frontmatter("---\n\nintro\n\n---\n\n# Alpha\n"),
        "# Alpha\n",
    )

    # An unparseable block is still a block. metadata() gives up on this one.
    broken = "---\ntitle: Alpha: beta\n---\n\n# Alpha\n"
    check("malformed block still removed", vault.without_frontmatter(broken), "# Alpha\n")
    check("and metadata gives up on it", vault.metadata(broken), {})


def test_a_string_stays_a_string() -> None:
    """A value must read back as the value it was set to.

    The case that forced this: a Discord thread id written bare is an integer,
    and one larger than JavaScript's MAX_SAFE_INTEGER, so n8n's JSON.parse
    rounds it and the reply goes to a thread that does not exist. The plugin
    quoted it; losing that on the way to Vault MCP would corrupt it silently.
    """
    import yaml

    def roundtrip(key, value):
        out = vault.set_frontmatter(FM, key, value)
        line = next(ln for ln in out.splitlines() if ln.startswith(f"{key}:"))
        return line, vault.metadata(out).get(key)

    line, back = roundtrip("thread_id", "1548070648281038848")
    check("a numeric string is quoted", line, 'thread_id: "1548070648281038848"')
    check("and reads back as that exact string", back, "1548070648281038848")

    # The whole point of quoting it: bare, YAML makes it an int.
    check("bare, it would not have been", yaml.safe_load("1548070648281038848"), 1548070648281038848)

    for value in ("true", "false", "null", "no", "0123", "1.0", "2026-09-12"):
        line, back = roundtrip("thread_id", value)
        check(f"{value!r} survives the round trip", back, value)

    # A colon would otherwise make the rest of the line a mapping, and
    # metadata() swallows a malformed block as {} - so the note would silently
    # lose every field rather than one.
    line, back = roundtrip("description", "Outlook: dispatch an order")
    check("a colon is quoted rather than left to parse as a mapping", back, "Outlook: dispatch an order")


def test_null_is_refused_rather_than_written() -> None:
    """`None` has no YAML spelling, so there is nothing honest to write.

    Two callers send it without meaning to: vault_set_frontmatter with `value`
    omitted - its default is None and `delete` defaults to False - and a REST
    frontmatter PATCH whose body is `null`. Written out, `key: None` reads back
    as the *string* "None", which is a field that compares equal to nothing and
    looks deliberate in the note.
    """
    raises(
        "a bare None is refused, not written",
        lambda: vault.set_frontmatter(FM, "expires"),
        "delete=true",
    )
    raises("and names the key it refused", lambda: vault.set_frontmatter(FM, "expires"), "'expires'")

    # A None *inside* a list takes the same route, one level down.
    # Not `tags`: that branch renders bare words without going near _scalar.
    raises(
        "a None inside a list is refused too",
        lambda: vault.set_frontmatter(FM, "aliases", ["a", None]),
        "cannot be null",
    )

    # delete=True is the call that legitimately passes no value.
    out = vault.set_frontmatter(FM, "tags", delete=True)
    check("delete still needs no value", "tags:" in out, False)

    # The string "none" is a real value this vault writes, and must be unharmed.
    check("the string 'none' is untouched", "expires: none" in vault.set_frontmatter(FM, "expires", "none"), True)


def test_bom_does_not_hide_the_frontmatter() -> None:
    """A BOM'd note must not read as a note with no frontmatter block.

    A leading U+FEFF makes the first line the mark plus '---' rather than '---',
    and str.strip() does not remove it - so every line scanner misses the block
    and the note silently loses its metadata instead of failing.
    """
    bommed = "\ufeff" + FM
    check("metadata still parses", vault.metadata(bommed)["title"], "Example")
    check("the prose is still found", vault.without_frontmatter(bommed).startswith("# Example"), True)
    check("a key is still queryable", vault.frontmatter_values(bommed, "tags"), ["one", "two"])

    out = vault.set_frontmatter(bommed, "title", "Renamed")
    check("the block is found and edited", "title: Renamed" in out, True)
    check("and the mark is normalised away rather than kept", out.startswith("---"), True)


def test_ordinary_scalars_stay_bare() -> None:
    """Quoting must be the exception, or every note churns on its next write."""
    for key, value in (
        ("status", "approved"),
        ("title", "Example"),
        ("risk", "medium"),
        ("description", "A note about things."),
    ):
        out = vault.set_frontmatter(FM, key, value)
        check(f"{key} written bare", f"{key}: {value}" in out, True)

    # timestamp is the one that would have been caught by a cruder rule: YAML
    # resolves it to a datetime, but it renders back to the identical text, so
    # there is nothing to protect and quoting it would fight the convention.
    out = vault.set_frontmatter(FM, "timestamp", "2026-09-12T13:57:17Z")
    check("timestamp stays unquoted", "timestamp: 2026-09-12T13:57:17Z" in out, True)
    check("and still reads back as written", vault.metadata(out)["timestamp"], "2026-09-12T13:57:17Z")


def test_numbers_are_still_numbers() -> None:
    """rev is a number and must stay one; quoting everything would break it."""
    out = vault.set_frontmatter(FM, "rev", 2)
    check("an int is written bare", "rev: 2" in out, True)
    check("and reads back as an int", vault.metadata(out)["rev"], 2)


def test_patch_operations() -> None:
    replaced, path = edit.patch_section(FM, "Section", "replace", "New text.")
    check("resolved path reported back", path, "Example::Section")
    check("content replaced", replaced.endswith("## Section\n\nNew text.\n"), True)

    appended, _ = edit.patch_section(FM, "Section", "append", "Added.")
    check("append keeps existing text", appended.endswith("Section text.\n\nAdded.\n"), True)

    prepended, _ = edit.patch_section(FM, "Section", "prepend", "First.")
    check("prepend goes above", prepended.endswith("## Section\n\nFirst.\n\nSection text.\n"), True)


def test_append_does_not_widen_gaps() -> None:
    """The plugin widens the gap by a line on every append; this must not."""
    text = FM
    for i in range(4):
        text, _ = edit.patch_section(text, "Example", "append", f"Line {i}.")
    check("no blank-line drift before the next heading", "\n\n\n" in text, False)
    check("all four appends landed", text.count("Line "), 4)
    check("following heading intact", "## Section" in text, True)


def test_marker_scope() -> None:
    out, _ = edit.patch_section(FM, "Section", "replace", "## Renamed", target_scope="marker")
    check("marker scope rewrites the heading line", "## Renamed" in out, True)
    check("marker scope leaves content", "Section text." in out, True)

    both, _ = edit.patch_section(
        FM, "Section", "replace", "## Gone", target_scope="markerAndContent"
    )
    check("markerAndContent replaces both", "Section text." in both, False)


def test_ambiguity_is_actionable() -> None:
    doc = "# Recipe\n\n## Mince\n\n### Method\n\nA.\n\n## Mash\n\n### Method\n\nB.\n"
    try:
        target.resolve(doc, "Method", note="Recipe.md")
    except vault.VaultError as exc:
        message = str(exc)
        check("error names the collision count", "2 headings match" in message, True)
        check("error offers a usable path", "Recipe::Mince::Method" in message, True)
        check("error offers the other path", "Recipe::Mash::Method" in message, True)
    else:
        FAILURES.append("test_ambiguity_is_actionable\n    expected an ambiguity error")

    found = target.resolve(doc, "Mince::Method", note="Recipe.md")
    check("prepending an ancestor narrows it", found.display, "Recipe::Mince::Method")


def test_normalisation() -> None:
    check("em dash folds to hyphen", target.normalise("A — B"), "a - b")
    check("backticks stripped", target.normalise("`code` heading"), "code heading")
    check("emphasis stripped", target.normalise("**Bold** _it_"), "bold it")
    check("link text kept", target.normalise("[Text](/a/b.md)"), "text")
    check("case folded", target.normalise("MiXeD"), "mixed")
    check("whitespace collapsed", target.normalise("a    b"), "a b")
    check("hash markers stripped", target.normalise("## Heading ##"), "heading")
    check("trailing hash kept when not a closing sequence", target.normalise("C#"), "c#")


def test_atomic_write() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        note = root / "note.md"

        vault.atomic_write(note, "created")
        check("new file gets NEW_FILE_MODE", stat.S_IMODE(note.stat().st_mode), vault.NEW_FILE_MODE)
        check("content written with final newline", note.read_text(), "created\n")

        os.chmod(note, 0o777)
        vault.atomic_write(note, "overwritten\r\n")
        check("existing mode preserved", stat.S_IMODE(note.stat().st_mode), 0o777)
        check("CRLF normalised on write", note.read_bytes(), b"overwritten\n")

        nested = root / "a" / "b" / "deep.md"
        vault.atomic_write(nested, "deep")
        check("parent directories created", nested.exists(), True)

        leftovers = [p.name for p in root.iterdir() if p.name.startswith(".vault-mcp-")]
        check("no temp files left behind", leftovers, [])


def test_path_safety() -> None:
    raises("traversal rejected", lambda: vault.safe_resolve("../../etc/passwd"), "escapes the vault")
    raises(
        "traversal rejected when creating",
        lambda: vault.safe_resolve("../escape.md", must_exist=False, writing=True),
        "escapes the vault",
    )
    raises(
        ".git is not writable",
        lambda: vault.safe_resolve(".git/config", must_exist=False, writing=True),
        "protected",
    )
    raises(
        "non-.md is not writable",
        lambda: vault.safe_resolve("notes.txt", must_exist=False, writing=True),
        "only .md files",
    )
    raises("null byte rejected", lambda: vault.safe_resolve("a\x00b.md"), "null byte")

    # Reads are unrestricted by decision: the n8n cutover needs Workflows/.
    allows(
        "generated series are readable",
        lambda: vault.safe_resolve("Workflows/Daily Tasks/Task List.md"),
    )
    allows(
        "generated series are writable",
        lambda: vault.safe_resolve("Reports/scratch.md", must_exist=False, writing=True),
    )
    check(
        "generated series stay out of the index",
        vault.is_search_excluded(Path("Workflows/Daily Tasks/Task List.md")),
        True,
    )
    check("but are not protected", vault.is_protected(Path("Workflows/x.md")), False)
    check(".git is protected", vault.is_protected(Path(".git/config")), True)


def test_walk_scopes_differ() -> None:
    """The two walks answer different questions and must not be confused.

    walk_notes() is "what belongs in the index"; walk_all_notes() is "what could
    contain a link". Using the indexing walk for a link rewrite silently skipped
    every note in Workflows/ and Reports/ - found in integration, not here, which
    is why it is pinned now.

    Neither walk contains the other, and that only became true when documents
    were filed. Each holds something the other must not: the index walk holds
    PDFs, which have no links to rewrite and would be decoded as text if the
    rewriter saw them; the link walk holds Workflows/ and Reports/, which are
    linked constantly and searched never. A subset assertion held here until
    Phase 2 and would now be the wrong shape to restore.
    """
    indexed = {p.relative_to(vault.ROOT).as_posix() for p in vault.walk_notes()}
    everything = {p.relative_to(vault.ROOT).as_posix() for p in vault.walk_all_notes()}

    check(
        "every note in the index walk is in the link walk",
        {p for p in indexed if p.endswith(".md")} <= everything,
        True,
    )
    check(
        "documents are indexed",
        [p for p in indexed if p.endswith(".pdf")] != [],
        True,
    )
    check(
        "and never offered to the link rewriter, which would decode them as text",
        [p for p in everything if not p.endswith(".md")],
        [],
    )
    generated = {p for p in everything if p.startswith(("Workflows/", "Reports/"))}
    check("generated series exist to be linked", len(generated) > 0, True)
    check("but are absent from the index walk", generated & indexed, set())
    check(
        "and present in the link-rewrite walk",
        generated <= everything,
        True,
    )
    check(
        "neither walk sees hidden directories",
        [p for p in everything if p.startswith(".")],
        [],
    )


def test_symlink_rejected() -> None:
    """Two cases, rejected by two different rules.

    A link pointing out of the vault fails containment - the more fundamental
    violation, and the one resolve() was always going to catch. A link pointing
    *inside* the vault passes containment and is caught only by the symlink rule,
    which is the case that rule exists for.
    """
    outward = vault.ROOT / "Reports" / "symtest-out.md"
    inward = vault.ROOT / "Reports" / "symtest-in.md"
    for link in (outward, inward):
        if link.exists() or link.is_symlink():
            FAILURES.append(f"test_symlink_rejected\n    {link.name} already exists; skipped")
            return
    try:
        outward.symlink_to("/etc/passwd")
        inward.symlink_to(vault.ROOT / "index.md")
    except OSError as exc:
        FAILURES.append(f"test_symlink_rejected\n    could not create the symlink: {exc}")
        return
    try:
        raises(
            "symlink out of the vault fails containment",
            lambda: vault.safe_resolve("Reports/symtest-out.md"),
            "escapes the vault",
        )
        raises(
            "symlink inside the vault fails the symlink rule",
            lambda: vault.safe_resolve("Reports/symtest-in.md"),
            "symlink",
        )
        raises(
            "and on the write path too",
            lambda: vault.safe_resolve("Reports/symtest-in.md", writing=True),
            "symlink",
        )
        check(
            "the symlink target is provably unmodified",
            Path("/etc/passwd").read_text().count("root") > 0,
            True,
        )
    finally:
        for link in (outward, inward):
            if link.is_symlink():
                link.unlink()


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()

    if FAILURES:
        print(f"primitives: {len(FAILURES)} failure(s)\n")
        for failure in FAILURES:
            print(f"  {failure}\n")
        return 1
    print(f"primitives: {len(tests)} test groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
