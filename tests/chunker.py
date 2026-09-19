"""Chunking: the flat-list rule, and what disqualifies a list from it.

The relevance suite measures whether the rule *helps*; this one measures whether
it fires where it is meant to. The two are not the same question, and the gap
between them is where a rule that is right on average hides a shape it gets
wrong - a '## Related notes' turned into four chunks holding one link each, a
nested tree read as twelve top-level items, a shell block read as a list.

Every case here is a shape this vault holds. None of them needs a vault on disk,
because _chunk_spans takes text; VAULT_PATH points at an empty temp tree only
because src.config resolves settings at import and refuses an absent one.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_VAULT = Path(tempfile.mkdtemp(prefix="vault-chunker-"))
os.environ["VAULT_PATH"] = str(_VAULT)

from src import chunker  # noqa: E402

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


def spans(body: str) -> list[tuple[str, int, str]]:
    """Chunk a note body, with a frontmatter block on it as a real note has."""
    return chunker._chunk_spans(f"---\ntitle: T\n---\n\n{body}")


def texts(body: str) -> list[str]:
    return [text for _, _, text in spans(body)]


LOG = "\n".join(f"- Item {n} about something unrelated to all the others" for n in range(1, 13))


def flat_list_fires() -> None:
    out = spans(f"# Note\n\nIntro line.\n\n## Log\n\n{LOG}\n")
    items = [text for _, _, text in out if text.startswith("- Item")]
    check("every item is its own chunk", len(items), 12)
    check("the first item is alone in its chunk", items[0], "- Item 1 about something unrelated to all the others")
    check("items keep the section's breadcrumb", {b for b, _, t in out if t.startswith("- Item")}, {"Note > Log"})

    # Line numbers are what a search result cites, and per-item chunking makes
    # them per-item. Frontmatter is 3 lines, blank 4, '# Note' 5, blank 6,
    # 'Intro line.' 7, blank 8, '## Log' 9, blank 10, first item 11.
    lines = [line for _, line, text in out if text.startswith("- Item")]
    check("line numbers count from the first item", lines[:3], [11, 12, 13])
    check("line numbers are consecutive", lines == list(range(11, 23)), True)


def too_few_items() -> None:
    short = "\n".join(f"- Field {n}: value" for n in range(1, 5))
    out = texts(f"# Note\n\nIntro that is long enough on its own. {'Padding. ' * 60}\n\n## Account\n\n{short}\n")
    check("a four-line fact block is not split", [t for t in out if t.startswith("- Field")], [])
    check("it stays one chunk with its heading", sum("- Field 1: value" in t for t in out), 1)


def bare_links_disqualify() -> None:
    links = "\n".join(f"- [Note {n}](/Some/Note{n}.md)" for n in range(1, 9))
    out = texts(f"# Note\n\n{'Body text. ' * 80}\n\n## Related notes\n\n{links}\n")
    check("a list of bare links is not split", [t for t in out if t.startswith("- [Note")], [])
    # And it is never a chunk of its own: on its own it would win a lexical
    # query on length normalisation while carrying no answer.
    check("the links ride along with a neighbour", any("[Note 1]" in t and "Body text." in t for t in out), True)

    described = "\n".join(f"- [Note {n}](/Some/Note{n}.md) - what this one covers and why" for n in range(1, 9))
    out = texts(f"# Note\n\nIntro.\n\n## Reading\n\n{described}\n")
    check("links with prose on them are a real list", sum(t.startswith("- [Note") for t in out), 8)


def two_lists_are_not_one() -> None:
    first = "\n".join(f"- Alpha {n} with enough words to look like a real entry" for n in range(1, 6))
    second = "\n".join(f"- Beta {n} with enough words to look like a real entry" for n in range(1, 6))
    out = texts(f"# Note\n\nIntro.\n\n## Both\n\n{first}\n\nA paragraph between them.\n\n{second}\n")
    check("items either side of a paragraph are two lists", [t for t in out if t.startswith("- Alpha")], [])

    mixed = first + "\n" + "\n".join(f"{n}. Gamma {n} with enough words to look real" for n in range(1, 6))
    out = texts(f"# Note\n\nIntro.\n\n## Mixed\n\n{mixed}\n")
    check("a bullet list and an ordered one are two lists", [t for t in out if t.startswith("- Alpha")], [])


def sequences_and_registers() -> None:
    """The two guards the real vault added, neither of which the fixture had.

    A numbered list is a sequence - a recipe method, a release procedure - and
    step four answers nothing without the three before it. A list whose items
    are two words each is a register of names, not of subjects: splitting one
    note's film list produced 45 chunks holding one title apiece.
    """
    ordered = "\n".join(f"{n}. Step {n} of a procedure worth recording, at length" for n in range(1, 9))
    out = texts(f"# Note\n\nIntro.\n\n## Method\n\n{ordered}\n")
    check("a numbered list is a sequence, not a flat list", [t for t in out if t.startswith("1.")], [])

    films = "\n".join(f"- Film Title {n}" for n in range(1, 20))
    out = texts(f"# Note\n\nIntro.\n\n## Favourites\n\n{films}\n")
    check("a register of short names is not split", [t for t in out if t.startswith("- Film Title")], [])

    fields = "\n".join(f"- **Field {n}:** value" for n in range(1, 12))
    out = texts(f"# Note\n\nIntro.\n\n## Profile\n\n{fields}\n")
    check("a key-value profile is not split", [t for t in out if t.startswith("- **Field")], [])

    # The floor is on the median, so a handful of terse entries among real ones
    # does not disqualify the list.
    mixed = ["- Short one"] * 4 + [f"- A log entry from {d} March with enough detail to stand alone" for d in range(1, 9)]
    out = texts("# Note\n\nIntro.\n\n## Log\n\n" + "\n".join(mixed) + "\n")
    check("a few terse entries do not sink a real log", sum(t.startswith("- A log entry") for t in out), 8)


def nesting_and_fences() -> None:
    tree = "\n".join(f"- Parent {n}\n  - child a\n  - child b" for n in range(1, 9))
    out = texts(f"# Note\n\nIntro.\n\n## Tree\n\n{tree}\n")
    parents = [t for t in out if t.startswith("- Parent")]
    check("a two-space nested tree is eight items, not twenty-four", len(parents), 8)
    check("children travel with their parent", parents[0], "- Parent 1\n  - child a\n  - child b")

    fenced = f"# Note\n\nIntro.\n\n## Usage\n\n```bash\n{chr(10).join(f'- not an item {n}' for n in range(1, 13))}\n```\n"
    out = texts(fenced)
    check("a fenced block is not a list", [t for t in out if t.startswith("- not an item")], [])


def prose_around_the_list() -> None:
    out = texts(f"# Note\n\nIntro.\n\n## Log\n\nWhat this log is for.\n\n{LOG}\n\nAnd a closing note.\n")
    check("the lead prose is its own chunk", "What this log is for." in out, True)
    check("the trailing prose is its own chunk", "And a closing note." in out, True)
    check("neither is folded into an item", any(t.startswith("- Item 1") and len(t.splitlines()) > 1 for t in out), False)


def ordinary_notes_are_unchanged() -> None:
    prose = "# Note\n\n" + "A sentence that says something. " * 40 + "\n\n## Detail\n\n" + "More prose. " * 40
    out = texts(prose)
    check("prose still packs rather than splitting", all(not t.startswith("- ") for t in out), True)
    check("prose still produces chunks", len(out) >= 1, True)


def main() -> int:
    flat_list_fires()
    too_few_items()
    bare_links_disqualify()
    two_lists_are_not_one()
    sequences_and_registers()
    nesting_and_fences()
    prose_around_the_list()
    ordinary_notes_are_unchanged()

    if report():
        return 1
    print("chunker: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_VAULT, ignore_errors=True)
