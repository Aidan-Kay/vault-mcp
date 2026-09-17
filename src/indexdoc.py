"""Generate the vault's root index.md from the notes on disk.

index.md is the vault's navigation document: one line per note carrying that
note's own `description`, grouped under headings that mirror the folder tree.

It used to be maintained by hand, which made it the one convention every write
depended on the model remembering - and the one it forgot. Nothing here is a
judgement call: the heading is the folder, the title and description are the
note's own frontmatter, and the order is fixed. So it is derived, not authored,
and deriving it on every change is cheaper than checking it after the fact.

The rendered document is compared against what is already on disk and only
written when it differs, so a note edit that does not touch a title or a
description leaves index.md - and the vault's git history - untouched.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from . import vault
from .config import settings

log = logging.getLogger(__name__)

INDEX_NAME = "index.md"

# The document's fixed head. Everything below it is derived.
PREAMBLE = """# Vault Index

Master navigation document for this vault. Read this first to understand the
structure and locate content.

Generated from the folder tree and each note's `description` frontmatter. Do not
edit by hand - every entry is rewritten whenever the vault changes. To change a
line here, change the note's `title` or `description`; to change a heading, move
the note."""

_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

# A YAML block scalar header: '|' or '>', then an indentation digit and a
# chomping '+'/'-' in either order. The value is on the lines below it, so the
# header itself must never reach the rendered entry.
_BLOCK_SCALAR = re.compile(r"^[|>](?:[+-]?\d|\d?[+-]?)$")

_FALLBACK_DESCRIPTION_CHARS = 240


@dataclass(frozen=True, slots=True)
class Entry:
    rel: str  # vault-relative POSIX path
    title: str
    description: str


# --------------------------------------------------------------------------
# Exclusions
#
# INDEX_EXCLUDE_DIRS, read here; SEARCH_EXCLUDE_DIRS, read by
# vault.is_search_excluded(). The names are parallel but the lists are not
# interchangeable, in two ways. That one drops Workflows/ and Reports/
# wholesale; this one must not, because the curated notes inside them -
# Workflows/Email Triage/Rules.md, Reports/PC/ - belong in the navigation
# document, so only the generated series come out. And that one matches bare
# folder names against any part of a path, where these are root-relative
# prefixes: "Approvals" alone would match nothing here.
# --------------------------------------------------------------------------


def _excluded_prefixes() -> tuple[str, ...]:
    return tuple(prefix.lower().rstrip("/") + "/" for prefix in settings.index_exclude_dirs)


def is_generated_series(rel: str) -> bool:
    """True if this path sits in a workflow-generated series."""
    return rel.lower().startswith(_excluded_prefixes())


def _indexable(rel: str) -> bool:
    return rel != INDEX_NAME and not is_generated_series(rel)


# --------------------------------------------------------------------------
# Reading a note's frontmatter
#
# Line-scanned rather than parsed. The write path never round-trips a note
# through a YAML library (see vault.py), and this runs over every note in the
# vault on every change, so it stays as cheap and as tolerant as that: a note
# with malformed YAML still gets an entry rather than breaking the whole index.
# --------------------------------------------------------------------------


def _folded(lines: list[str], end: int, key: str) -> str | None:
    """One frontmatter value, with YAML folded continuation lines joined.

    A description wrapped across lines has to compare - and render - as the
    single line the index entry is. Both of YAML's ways of wrapping one are
    handled: a plain scalar continued by indentation, and a block scalar, whose
    `>-` or `|` header announces the wrap and is not part of the value.
    """
    parts: list[str] = []
    capturing = False
    for i in range(1, end):
        line = lines[i]
        if capturing:
            if line[:1] in (" ", "\t") and line.strip():
                parts.append(line.strip())
                continue
            capturing = False
        match = vault.FM_KEY.match(line)
        if match and match.group(1) == key:
            inline = match.group(2).strip()
            parts.append("" if _BLOCK_SCALAR.match(inline) else inline)
            capturing = True
    if not parts:
        return None
    value = " ".join(parts).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value or None


def _first_sentence(lines: list[str], start: int) -> str | None:
    """The note's opening sentence, for a note with no `description`."""
    for line in lines[start:]:
        text = line.strip()
        if not text or text.startswith(("#", ">", "-", "*", "|", "```", "---")):
            continue
        sentence = _SENTENCE_END.split(text, 1)[0].strip()
        if len(sentence) > _FALLBACK_DESCRIPTION_CHARS:
            sentence = sentence[:_FALLBACK_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "..."
        return sentence
    return None


def read_entry(path: Path, rel: str) -> Entry:
    """Build one index entry from a note.

    Every fallback here exists because a real note in this vault needs it: the
    prompt-injected files carry no frontmatter at all, and a note can be created
    with a body before anyone gives it a title.
    """
    lines = vault.strip_bom(vault.read_text(path)).split("\n")

    # 0 for a note with no block at all, which _folded and _first_sentence both
    # read as "there is no frontmatter to look in".
    bounds = vault.frontmatter_bounds(lines)
    end = bounds[1] if bounds else 0

    title = _folded(lines, end, "title") if end else None
    if not title:
        title = next(
            (m.group(1).strip() for m in map(_H1.match, lines[end:]) if m),
            Path(rel).stem,
        )

    description = _folded(lines, end, "description") if end else None
    if description is None:
        description = _first_sentence(lines, end + 1 if end else 0) or ""

    return Entry(rel=rel, title=title, description=description)


def _group(entries: dict[str, Entry]) -> dict[str, list[Entry]]:
    """Entries bucketed by the folder they sit in; the vault root is ""."""
    grouped: dict[str, list[Entry]] = {}
    for rel, entry in entries.items():
        folder = Path(rel).parent
        key = "" if folder == Path(".") else folder.as_posix()
        grouped.setdefault(key, []).append(entry)
    return grouped


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _encode(rel: str) -> str:
    """A link target this vault's conventions accept.

    Spaces percent-encoded because Obsidian will not resolve them otherwise, and
    check_vault_links.py reports an unencoded one as broken even when the file is
    there. Apostrophes are left alone, matching how the vault already writes them.
    """
    return urllib.parse.quote(rel, safe="/'")


def _sort_key(entry: Entry, folder: str) -> tuple[int, str]:
    """The folder's landing note first, then the rest by title.

    A folder's landing note is named after the folder (Home/Home.md), per
    Meta/Conventions.md. It is the one that describes everything under it, so it
    leads the section rather than falling wherever the alphabet puts it.
    """
    landing = Path(folder).name if folder else ""
    is_landing = bool(landing) and Path(entry.rel).stem == landing
    return (0 if is_landing else 1, entry.title.casefold())


def _live_exclusions() -> list[str]:
    """Configured generated-series folders that actually exist in this vault.

    The list is configuration, not observation, so it can name a folder that was
    never created - a workflow configured but not yet run, or a vault this
    config is merely pointed at. Announcing "this folder is not indexed" for a
    folder nobody has, under a heading invented to hold the announcement, would
    describe the config rather than the vault.
    """
    live = []
    for prefix in settings.index_exclude_dirs:
        clean = prefix.strip("/")
        if clean and (vault.ROOT / clean).is_dir():
            live.append(clean)
    return sorted(live, key=str.casefold)


def _folders(grouped: dict[str, list[Entry]], exclusions: list[str]) -> list[str]:
    """Every folder needing a heading, in document order.

    Ancestors of a folder holding notes are included even when they hold none
    themselves - Tech/ has no notes of its own, but dropping its heading would
    leave Linux Server hanging off nothing.
    """
    needed: set[str] = set()
    sources = [f for f in grouped if f] + [str(Path(e).parent) for e in exclusions]
    for folder in sources:
        if folder == ".":
            continue
        parts = folder.replace("\\", "/").split("/")
        for depth in range(1, len(parts) + 1):
            needed.add("/".join(parts[:depth]))
    return sorted(needed, key=lambda f: tuple(part.casefold() for part in f.split("/")))


def _exclusion_notes(folder: str, exclusions: list[str]) -> list[str]:
    """The 'this folder is not indexed' lines for one section.

    A reader who finds no entry for Workflows/Approvals/ should learn that it is
    deliberate rather than assume the index is stale.
    """
    lines: list[str] = []
    for clean in exclusions:
        parent = Path(clean).parent
        if (parent.as_posix() if parent != Path(".") else "") != folder:
            continue
        # A typed callout, per the Callouts section of Meta/Conventions.md: every blockquote in
        # the vault carries one of the five allowed types, and .scripts/check_callouts.py
        # enforces it. index.md is exempt from that check because it is generated, but emitting
        # the bare form here would be the one place the vault contradicts its own convention.
        lines.append(f"> [!NOTE] `{clean}/` is not indexed note by note")
        lines.append(
            "> It is a generated series, written by a workflow rather than by hand."
        )
        lines.append("")
    return lines


def render(entries: dict[str, Entry]) -> str:
    """The complete index.md, ready to write."""
    grouped = _group(entries)
    exclusions = _live_exclusions()
    out = [
        "---",
        'okf_version: "0.1"',
        f"timestamp: {vault.utc_now()}",
        "---",
        "",
        PREAMBLE,
        "",
    ]

    def emit(folder: str) -> None:
        for entry in sorted(grouped.get(folder, []), key=lambda e: _sort_key(e, folder)):
            line = f"- [{entry.title}]({_encode(entry.rel)})"
            out.append(f"{line} - {entry.description}" if entry.description else line)
        if grouped.get(folder):
            out.append("")
        out.extend(_exclusion_notes(folder, exclusions))

    emit("")  # notes at the vault root, before the first heading
    for folder in _folders(grouped, exclusions):
        # Six is markdown's floor; a seventh level of nesting would stop being a
        # heading at all, so it flattens into its parent's depth instead.
        depth = min(folder.count("/") + 2, 6)
        out.append(f"{'#' * depth} {Path(folder).name}")
        out.append("")
        emit(folder)

    return vault.normalise_body("\n".join(out))


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _body(text: str) -> str:
    """Everything after the frontmatter, which is what 'changed' means here.

    The timestamp is regenerated on every render, so comparing whole documents
    would report a change every time and rewrite index.md on every note edit.
    """
    return vault.without_frontmatter(text)


@dataclass(slots=True)
class IndexDoc:
    """The index's entries, held in memory so a change re-reads only one note.

    A full scan walks every note in the vault, which is a few seconds across the
    Samba mount. Doing that on every write would put those seconds between a
    model's call and its answer, so the scan happens once at startup and each
    change swaps a single entry, exactly as VaultIndex does for search.
    """

    entries: dict[str, Entry]

    @classmethod
    def build(cls) -> IndexDoc:
        entries: dict[str, Entry] = {}
        for path in vault.walk_all_notes():
            rel = vault.relpath(path)
            if _indexable(rel):
                try:
                    entries[rel] = read_entry(path, rel)
                except Exception:
                    # One unreadable note must not cost the whole index.
                    log.exception("skipping unreadable note %s", rel)
        return cls(entries=entries)

    def replace_note(self, path: Path) -> IndexDoc:
        """A new document with one note's entry re-read, added, or dropped.

        Returns self when the path has no bearing on the index, so the caller
        can compare identity and skip the render entirely.
        """
        rel = vault.relpath(path)
        if not _indexable(rel):
            return self

        entries = dict(self.entries)
        if path.exists():
            try:
                entries[rel] = read_entry(path, rel)
            except FileNotFoundError:
                entries.pop(rel, None)  # deleted between the event and the read
            except Exception:
                log.exception("cannot read %s; leaving its entry as it was", rel)
                return self
        else:
            entries.pop(rel, None)

        return IndexDoc(entries=entries) if entries != self.entries else self

    def render(self) -> str:
        return render(self.entries)

    def write_if_changed(self) -> str | None:
        """Write index.md when the rendered body differs. Returns what it did.

        Writing unconditionally would touch index.md on every note edit and fill
        the vault's git history with commits whose only change is a timestamp.
        """
        rendered = self.render()
        path = vault.ROOT / INDEX_NAME

        if path.exists() and _body(vault.read_text(path)) == _body(rendered):
            return None

        vault.atomic_write(path, rendered)
        return f"wrote index.md ({len(self.entries)} entries)"
