"""Filesystem access to the vault, reads and writes.

Security-critical: every path argument reaching this module originated in an
LLM tool call and is untrusted. All access funnels through safe_resolve(), which
is now the only barrier between that string and destructive writes to the share -
the :ro mount that used to back it up is gone.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from . import documents
from .config import settings


class VaultError(ValueError):
    """Raised for a rejected path or an unreadable note."""


class NotFound(VaultError):
    """Raised when a path that had to exist does not.

    A subclass rather than a message the caller matches on, because the REST
    surface answers 404 for this and 400 for every other VaultError, and a
    status code derived from a string is a status code that breaks the next time
    the string is reworded. Every MCP caller still sees a plain VaultError.
    """


# Resolved once. If VAULT_PATH is itself a symlink, this is the real target,
# which is what every containment check below compares against.
ROOT = settings.vault_path.resolve()

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")

# One frontmatter key and everything after its colon. Public because
# src.indexdoc scans the same blocks and a second copy of this pattern is a
# second place for the vault's key syntax to drift.
FM_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):(.*)$")

BOM = "\ufeff"


def strip_bom(text: str) -> str:
    """Drop a leading byte-order mark.

    Written out rather than spelled with the character itself, which is
    invisible in an editor: a literal BOM inside a string literal is deletable
    by accident and leaves no visible diff when it goes.

    Every line scanner over a frontmatter block needs this. A BOM'd note's first
    line is the mark followed by '---' rather than '---' alone, and str.strip()
    does not remove it - so the block is simply not found and the note silently
    loses its metadata.
    """
    return text.lstrip(BOM)


@dataclass(frozen=True, slots=True)
class Heading:
    depth: int
    text: str
    line: int  # 1-based, relative to the whole file


def _is_hidden(rel: Path) -> bool:
    """True if any component is a dotted directory - .git, .obsidian, .trash."""
    return any(part.startswith(".") for part in rel.parts)


INDEX_DOC = "index.md"


def is_protected(rel: Path) -> bool:
    """True if this path must never be written.

    This is the *only* thing standing between Lyra and `rm -rf .git`, now
    that SEARCH_EXCLUDE_DIRS has been narrowed to indexing. Reads are
    unrestricted; writes go through here.

    index.md joins the hidden directories because it is now generated from the
    notes themselves. A write to it is not dangerous, it is futile - the next
    change to any note overwrites it - and a tool that accepts a write it is
    about to discard teaches the caller the edit worked. src.indexdoc writes it
    through atomic_write directly, which is the same door _rewrite_links uses.
    """
    return _is_hidden(rel) or rel.as_posix() == INDEX_DOC


# The write scope for the request being served, or None for "anywhere".
#
# A ContextVar rather than a module global because the server is ASGI and
# concurrent: two requests can be in flight, and a global would leak one
# caller's scope into another's writes. Set per request and never inherited -
# an unscoped request sees None however many scoped ones came before it.
_WRITE_SCOPE: ContextVar[str | None] = ContextVar("vault_write_scope", default=None)


@contextlib.contextmanager
def write_scope(scope: str | None):
    """Confine writes to one note, or to one directory, for this request.

    `scope` is a vault-relative path. A `.md` path allows exactly that note; any
    other path is treated as a directory prefix. None restores the unscoped
    default, which is what every caller that does not ask for this gets.
    """
    token = _WRITE_SCOPE.set(scope.strip().lstrip("/") if scope else None)
    try:
        yield
    finally:
        _WRITE_SCOPE.reset(token)


def current_write_scope() -> str | None:
    """The scope in force, for callers that must refuse work rather than narrow it."""
    return _WRITE_SCOPE.get()


def _out_of_scope(rel: Path) -> str | None:
    """The scope this path violates, or None if it is allowed.

    Reads are never scoped. The agent revising a proposal still has to read the
    conventions, the note it is revising, and whatever that note refers to; it
    is writing outside its remit that has to be impossible, not knowing things.
    """
    scope = _WRITE_SCOPE.get()
    if scope is None:
        return None

    target = rel.as_posix()
    if scope.lower().endswith(".md"):
        return None if target == scope else scope
    prefix = scope.rstrip("/") + "/"
    return None if target.startswith(prefix) else scope


def is_search_excluded(rel: Path) -> bool:
    """True if this path is kept out of the vector index and BM25.

    Workflows/ and Reports/ are machine-generated series - noise in search, but
    ordinary notes to read and write. That distinction is the whole reason this
    is separate from is_protected().

    index.md is excluded too, and for a sharper reason: every line in it is a
    copy of a description that already sits in the note it points at, so
    indexing it puts each of those sentences in the corpus twice and lets one
    note win two slots in the same result set. It is also rewritten whenever the
    vault changes, which would mean re-embedding the whole document each time.
    """
    return (
        _is_hidden(rel)
        or rel.as_posix() == INDEX_DOC
        or any(part in settings.search_exclude_dirs for part in rel.parts)
    )


def _reject_symlinks(raw: Path) -> None:
    """Refuse any path with a symlink component.

    Must be given the *unresolved* path: resolve() follows symlinks by
    definition, so walking its output would never find one. `..` is removed
    lexically first, with no filesystem access, so this cannot itself traverse.

    The vault contains no symlinks and is not going to. Rejecting them outright
    is cheaper than reasoning about the window between resolve() and os.replace,
    and it fails loudly if one ever appears. They *are* creatable on this
    fuseblk mount - verified - so this is not a theoretical rule.
    """
    lexical = Path(os.path.normpath(raw))
    try:
        parts = lexical.relative_to(ROOT).parts
    except ValueError:
        return  # outside the vault; containment has already rejected it

    current = ROOT
    for part in parts:
        current = current / part
        if current.is_symlink():  # lstat, does not follow
            rel = current.relative_to(ROOT).as_posix()
            raise VaultError(f"path component is a symlink, which is not allowed: {rel!r}")
        if not current.exists():
            return  # nothing beyond this can exist either


def safe_resolve(
    rel_path: str,
    *,
    must_exist: bool = True,
    writing: bool = False,
    allow_documents: bool = False,
) -> Path:
    """Resolve a vault-relative path, or raise.

    Rejects traversal, absolute escapes and symlinks. When writing=True it also
    rejects protected paths and anything that is not a .md file. Never falls
    back to a default.

    `allow_documents` widens that last rule to the DOC_SUFFIXES allowlist, and
    only the three verbs that treat a file as opaque bytes pass it: upload,
    move and delete. Every content verb - patch, append, write, set_body,
    set_frontmatter - leaves it False and so stays markdown-only, because there
    is no structure inside a PDF for any of them to address. The default is the
    strict one so that a new caller has to say it means this.
    """
    if "\x00" in rel_path:
        raise VaultError("path contains a null byte")

    # A leading slash is treated as vault-root-relative rather than rejected -
    # models write "/Pets/Levi.md" often enough that failing it is pure
    # friction. Containment is still enforced below, so this is not a shortcut.
    cleaned = rel_path.strip().lstrip("/")

    raw = ROOT / cleaned
    candidate = raw.resolve()

    if candidate != ROOT and not candidate.is_relative_to(ROOT):
        raise VaultError(f"path escapes the vault: {rel_path!r}")

    rel = candidate.relative_to(ROOT)
    _reject_symlinks(raw)

    if writing:
        if is_protected(rel):
            raise VaultError(f"path is protected and cannot be written: {rel.as_posix()!r}")
        suffix = candidate.suffix.lower()
        if suffix != ".md" and not (allow_documents and suffix in settings.doc_suffixes):
            allowed = ".md"
            if allow_documents:
                allowed = ", ".join([".md", *sorted(settings.doc_suffixes)])
            raise VaultError(
                f"only {allowed} files may be written, got: {rel.as_posix()!r}"
            )
        scope = _out_of_scope(rel)
        if scope is not None:
            # Deliberately not phrased as something to retry. A caller that
            # reaches this has been handed a narrower remit than it thinks it
            # has, and the useful thing is for it to say so rather than to go
            # looking for a path that gets through.
            raise VaultError(
                f"this request may only write to {scope!r}, so "
                f"{rel.as_posix()!r} was refused. Nothing was changed. Report this "
                "rather than trying another path."
            )

    if must_exist and not candidate.exists():
        raise NotFound(f"no such path in the vault: {rel.as_posix()!r}")

    return candidate


def relpath(path: Path) -> str:
    """Vault-relative POSIX path, for display and chunk metadata."""
    return path.resolve().relative_to(ROOT).as_posix()


def frontmatter_span(text: str) -> int:
    """Number of leading lines occupied by the YAML frontmatter block."""
    match = _FRONTMATTER_RE.match(text)
    return match.group(0).count("\n") if match else 0



def without_frontmatter(text: str) -> str:
    """The prose: everything after the YAML block, if the note opens with one.

    Textual, not semantic - a block that does not parse is still a block and is
    still removed. That is deliberate: metadata() answers {} for a malformed
    block, so a caller reading only the parsed fields would otherwise find the
    raw YAML pasted on the front of the note it asked for.
    """
    text = strip_bom(text)
    match = _FRONTMATTER_RE.match(text)
    return text[match.end() :].lstrip("\r\n") if match else text


def iter_headings(text: str) -> list[Heading]:
    """ATX headings, skipping frontmatter and fenced code blocks.

    Fence tracking matters: the vault is full of bash blocks whose comments
    start with '#', and every one of them would otherwise parse as a heading.
    """
    lines = text.splitlines()
    start = frontmatter_span(text)
    headings: list[Heading] = []
    fence: str | None = None

    for offset, line in enumerate(lines[start:], start=start):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            headings.append(
                Heading(
                    depth=len(heading_match.group(1)),
                    text=heading_match.group(2).strip(),
                    line=offset + 1,
                )
            )
    return headings


def read_text(path: Path) -> str:
    """The UTF-8 text of a note, and only ever of a note.

    The suffix check is the fix for a latent defect rather than tidiness.
    `errors="replace"` decodes anything at all, so before documents existed
    read_text() on a PDF returned a page of replacement characters and called it
    a note - and every caller downstream of it, from the chunker to the link
    rewriter, believed it. Refusing here means a path that is not markdown can
    only be reached through the door that knows what to do with it.
    """
    suffix = path.suffix.lower()
    if suffix != ".md":
        if suffix in settings.doc_suffixes:
            raise VaultError(
                f"{relpath(path)!r} is a document, not a note. Read it with "
                "vault_read, which extracts its text; it has no markdown source "
                "to edit."
            )
        raise VaultError(
            f"{relpath(path)!r} is not a note. Only .md files, and documents "
            f"with an allowlisted suffix, can be read from this vault."
        )
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except IsADirectoryError as exc:
        raise VaultError(f"{relpath(path)!r} is a directory, not a note") from exc
    except OSError as exc:
        raise VaultError(f"cannot read {relpath(path)!r}: {exc}") from exc


def read_any(path: Path) -> str:
    """Markdown for a note, extracted markdown for a document, or raise.

    The one dispatch point. A caller that wants "the text of this path" asks
    here and does not care which of the two it got, which is what lets search,
    the chunker and the read surface treat a filed bill as a note that happens
    to have arrived as bytes.
    """
    if documents.is_document(path):
        extraction = documents.extract(path)
        if extraction.searchable:
            return extraction.markdown
        raise VaultError(
            f"no text could be extracted from {relpath(path)!r}: "
            f"{extraction.detail or extraction.status}"
        )
    return read_text(path)


def extract_section(text: str, section: str) -> str:
    """Return one heading's content: the heading line through to the next
    heading of equal or shallower depth."""
    headings = iter_headings(text)
    wanted = section.strip().lstrip("#").strip().casefold()

    start_heading = next((h for h in headings if h.text.casefold() == wanted), None)
    if start_heading is None:
        available = ", ".join(h.text for h in headings) or "(none)"
        raise VaultError(f"no heading {section!r} in this note. Headings: {available}")

    end_line = None
    for heading in headings:
        if heading.line > start_heading.line and heading.depth <= start_heading.depth:
            end_line = heading.line
            break

    lines = text.splitlines()
    body = lines[start_heading.line - 1 : (end_line - 1) if end_line else None]
    return "\n".join(body).rstrip() + "\n"


def read_note(rel_path: str, section: str | None = None) -> str:
    path = safe_resolve(rel_path)
    if path.is_dir():
        raise VaultError(
            f"{relpath(path)!r} is a directory, not a note - list it rather "
            "than reading it"
        )
    text = read_any(path)
    if section and documents.is_document(path):
        # A document's headings are inferred by the extractor rather than
        # written by anyone, so naming one is guesswork the caller cannot
        # verify. Reading the whole thing is the honest offer.
        raise VaultError(
            f"{relpath(path)!r} is a document; its headings are inferred during "
            "extraction and cannot be addressed by name. Read the whole document."
        )
    return extract_section(text, section) if section else text


def list_dir(rel_path: str = "") -> list[dict]:
    path = safe_resolve(rel_path or ".")
    if not path.is_dir():
        raise VaultError(f"{relpath(path)!r} is not a directory")

    entries: list[dict] = []
    for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        try:
            rel = child.resolve().relative_to(ROOT)
        except ValueError:
            continue  # symlink out of the vault
        if _is_hidden(rel):
            continue  # tidiness, not safety - reads are unrestricted
        try:
            stat = child.stat()
        except OSError:
            continue
        entries.append(
            {
                "name": child.name,
                "path": rel.as_posix(),
                "type": "dir" if child.is_dir() else "file",
                "size": None if child.is_dir() else stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            }
        )
    return entries


def _as_written(value):
    """YAML's typed scalars, back to the text the note actually carries.

    PyYAML resolves `timestamp: 2026-09-12T09:00:00Z` to a datetime, and every
    note in this vault carries a timestamp. Passed on that way it is not JSON
    at all, and anything that does serialise it renders a *different string*
    from the one in the file - so a consumer comparing timestamps would be
    comparing against a shape the vault has never written. Rendered back in the
    vault's own format, `frontmatter.timestamp` is the text on disk, which is
    the only thing a caller can sensibly have meant.

    Recursive, because `expires` is a list of mappings with a date in each.
    """
    if isinstance(value, datetime):  # before date - datetime is a subclass
        utc = value.astimezone(timezone.utc) if value.tzinfo else value
        return utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _as_written(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_written(item) for item in value]
    return value


def metadata(text: str) -> dict:
    """A note's frontmatter as a dict, or {} if it has none or it is malformed.

    Malformed YAML must not make a note unreadable: a caller that asked for the
    content gets the content, and an empty block, rather than an error about a
    field it never mentioned.
    """
    # Imported here, not at module scope: reading YAML metadata is the only
    # thing in this module that needs it. The write path is deliberately
    # surgical and never round-trips a note through a parser, so the resolver
    # and its tests must not drag the dependency in.
    import frontmatter

    try:
        parsed = frontmatter.loads(strip_bom(text)).metadata
        return {key: _as_written(value) for key, value in parsed.items()}
    except Exception:
        return {}


def parse_note(rel_path: str) -> dict:
    path = safe_resolve(rel_path)
    if documents.is_document(path):
        # A document has no frontmatter and no headings anyone wrote, so it has
        # no patch targets - which is the question this answers. Saying that
        # outright, with the extraction verdict beside it, is more use than
        # either an error or an empty outline: it tells a caller both why it
        # cannot edit this file and whether search can see inside it.
        extraction = documents.extract(path)
        return {
            "path": relpath(path),
            "document": True,
            "frontmatter": {},
            "headings": [],
            "extraction": extraction.as_json(),
        }
    text = read_text(path)
    return {
        "path": relpath(path),
        "document": False,
        "frontmatter": metadata(text),
        "headings": [
            {"depth": h.depth, "text": h.text, "line": h.line} for h in iter_headings(text)
        ],
    }


def note_json(rel_path: str, section: str | None = None) -> dict:
    """A note as {path, content, body, frontmatter} - the structured read.

    The shape obsidian-local-rest-api returned for
    `Accept: application/vnd.olrapi.note+json`, minus `tags` and `stat`, which
    no caller reads. It exists so a consumer that wants one frontmatter field
    does not have to parse YAML out of a markdown string itself.

    `content` is the file, byte for byte, and is the only field that is. The
    parsed `frontmatter` cannot reconstruct the block it came from, and answers
    {} outright when the YAML does not parse - four notes in this vault do not.
    Stripping the block from `content` would hand those back with no trace of
    their frontmatter in either field, so `body` carries the prose and
    `content` keeps everything.

    `section` narrows `content` exactly as read_note does; a section has no
    frontmatter of its own, so `body` is then the same text. `frontmatter` is
    always the whole note's.
    """
    path = safe_resolve(rel_path)
    if path.is_dir():
        raise VaultError(
            f"{relpath(path)!r} is a directory, not a note - list it rather "
            "than reading it"
        )
    if documents.is_document(path):
        # No frontmatter and no section: a document carries neither, and
        # answering {} for one while answering a parsed block for the other
        # would be the same shape describing two different things.
        content = read_note(rel_path, section)
        return {
            "path": relpath(path),
            "content": content,
            "body": content,
            "frontmatter": {},
        }
    text = read_text(path)
    content = extract_section(text, section) if section else text
    return {
        "path": relpath(path),
        "content": content,
        "body": content if section else without_frontmatter(content),
        "frontmatter": metadata(text),
    }


def walk_all_notes() -> list[Path]:
    """Every readable markdown file, hidden directories aside.

    Distinct from walk_notes(): that one answers "what belongs in the index",
    this one answers "what could contain a link". Since the exclusion split,
    Workflows/ and Reports/ are ordinary notes for every purpose except search,
    so a link rewrite that used the indexing walk would silently skip 388 notes.

    Markdown only, and it stays that way now documents exist. The plan had both
    walks gaining the document suffixes; checking the three callers says
    otherwise, and each one breaks differently. _rewrite_links would decode a
    PDF and write it back as UTF-8 - the exact corruption the byte-preserving
    move() branch exists to prevent, arriving by another door. indexdoc.build
    would put a Files/ folder into the generated index.md, contradicting the
    one thing the vault proposal wanted confirmed. find_by_frontmatter would
    scan PDF bytes for YAML. All three want notes, because a document is a link
    *target* and never a link source.
    """
    notes: list[Path] = []
    for path in sorted(ROOT.rglob("*.md")):
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            continue
        if _is_hidden(rel) or not path.is_file():
            continue
        notes.append(path)
    return notes


def walk_notes() -> list[Path]:
    """Every indexable file, exclusions applied: notes and documents alike.

    Documents are deliberately not excluded. Indexing them is the entire point
    of filing them here, and a chunk from one is attributed to the document's
    own path, so a hit reads as the PDF it came from rather than as the note
    beside it.
    """
    notes: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            continue
        if path.suffix.lower() != ".md" and not documents.is_document(path):
            continue
        if is_search_excluded(rel) or not path.is_file():
            continue
        notes.append(path)
    return notes


# --------------------------------------------------------------------------
# Frontmatter query
#
# Line-scanned rather than YAML-parsed, for the reason indexdoc.py is: this
# reads every note in the vault on every call, and a note with malformed YAML
# should drop out of one query rather than break it.
# --------------------------------------------------------------------------

_FM_LIST_ITEM = re.compile(r"^\s+-\s+(.*)$")


def _unquoted(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def frontmatter_values(text: str, key: str) -> list[str] | None:
    """Every scalar one frontmatter key carries, as written.

    None when the note has no frontmatter block or does not carry the key at
    all - which is not the same as carrying it empty, and the caller can tell.

    Three shapes are recognised, which is every shape this vault writes:

        key: value     ->  ["value"]
        key: [a, b]    ->  ["a", "b"]
        key:               ["a", "b"]
          - a
          - b

    An inline list is split on commas, so a value containing one is not
    addressable. Nothing in this vault writes one, and the alternative is a YAML
    parser on every note of every query.
    """
    lines = text.lstrip("\ufeff").split("\n")
    bounds = frontmatter_bounds(lines)
    if bounds is None:
        return None
    start, end = bounds

    for i in range(start, end):
        match = FM_KEY.match(lines[i])
        if not match or match.group(1) != key:
            continue

        inline = match.group(2).strip()
        if inline.startswith("[") and inline.endswith("]"):
            return [_unquoted(part) for part in inline[1:-1].split(",") if part.strip()]
        if inline:
            return [_unquoted(inline)]

        # Empty after the colon: a block sequence, or a key with no value.
        items: list[str] = []
        for j in range(i + 1, end):
            item = _FM_LIST_ITEM.match(lines[j])
            if not item:
                break
            items.append(_unquoted(item.group(1)))
        return items

    return None


def find_by_frontmatter(key: str, value: str, prefix: str | None = None) -> list[str]:
    """Vault-relative paths of every note whose `key` carries `value`.

    String equality against the value as written, and membership when the key
    holds a list - so tags=lyra matches `tags: [lyra, ops]`. Frontmatter is
    text here, so 2 and "2" are the same query; nothing that asks this asks it
    of a number.

    Deliberately a filesystem walk and never the semantic index. Workflows/ is
    in SEARCH_EXCLUDE_DIRS and therefore absent from search entirely, and every note
    this exists to find lives there. `prefix` narrows it to one folder and is
    worth passing whenever the caller knows it - it is what turns reading the
    whole vault into reading one directory.
    """
    scope = None
    if prefix:
        scope = safe_resolve(prefix)
        if not scope.is_dir():
            raise VaultError(f"{relpath(scope)!r} is not a directory")

    matches: list[str] = []
    for path in walk_all_notes():
        if scope is not None and not path.is_relative_to(scope):
            continue
        try:
            values = frontmatter_values(read_text(path), key)
        except VaultError:
            continue  # unreadable note drops out of the query, not the query out
        if values and value in values:
            matches.append(relpath(path))
    return matches


# --------------------------------------------------------------------------
# Write primitives
#
# Nothing below round-trips a note through a parser. Every operation is a
# surgical edit on the line list, because the vault's own checkers
# (.scripts/check_frontmatter.py, check_vault_hygiene.py) reject exactly the
# formatting a naive yaml.dump or frontmatter.dumps would produce.
# --------------------------------------------------------------------------

NEW_FILE_MODE = 0o644

# OKF v0.1 spec order, per Meta/Conventions.md. A key that does not exist yet is
# inserted at its ordained position, not appended - appending 'description' to
# the end of the block is a convention violation the checker will not catch.
FIELD_ORDER = (
    "type",
    "title",
    "description",
    "tags",
    "timestamp",
    "expires",
    "expires_reason",
)


def utc_now() -> str:
    """The vault's timestamp format: YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_body(text: str) -> str:
    """LF line endings and exactly one trailing newline.

    check_vault_hygiene.py treats mixed endings as an error and a missing final
    newline as a warning, so this is not cosmetic. A leading BOM goes the same
    way and for the same reason: it is not content, and left in place it hides
    the frontmatter block from every line scanner that follows.
    """
    if not text:
        return ""
    text = strip_bom(text).replace("\r\n", "\n").replace("\r", "\n")
    return text.rstrip("\n") + "\n"


def atomic_write(path: Path, text: str) -> None:
    """Replace a note's contents in one step, preserving its mode.

    The temporary file is created in the same directory because os.replace is
    only atomic within a filesystem. A partial write on the fuseblk mount would
    leave a corrupt note that the watcher indexes immediately.

    mkstemp creates 0600 and os.replace keeps the *new* inode's mode, so without
    the chmod every note Lyra touches would silently change mode. Nothing breaks
    if it does - Samba forces uid 1000 for every accessor - but the vault has a
    settled mix of 644 and 777 and there is no reason to churn it.
    """
    _atomic(path, normalise_body(text).encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace a file's contents with exact bytes, preserving its mode.

    The same guarantee as atomic_write and deliberately not the same function.
    Every transformation that one applies - newline normalisation, UTF-8
    encoding - is correct for a note and destroys a PDF, so the shared part is
    the temp-file-and-replace dance and nothing above it.
    """
    _atomic(path, data)


def _atomic(path: Path, payload: bytes) -> None:
    mode = (path.stat().st_mode & 0o777) if path.exists() else NEW_FILE_MODE
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".vault-mcp-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


# --------------------------------------------------------------------------
# Frontmatter: surgical, never round-tripped
# --------------------------------------------------------------------------


def frontmatter_bounds(lines: list[str]) -> tuple[int, int] | None:
    """Half-open [start, end) line range of the frontmatter *content*.

    Excludes both '---' fences. None if the note has no frontmatter block.
    """
    if not lines or lines[0].strip() != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return 1, i
    return None


def _key_span(lines: list[str], start: int, end: int, key: str) -> tuple[int, int] | None:
    """Half-open line range occupied by one frontmatter key, value included.

    A key's value runs until the next top-level key or the closing fence, which
    is what makes this work for a block value like the `expires` sequence rather
    than only for `key: value` lines.
    """
    for i in range(start, end):
        match = FM_KEY.match(lines[i])
        if match and match.group(1) == key:
            j = i + 1
            while j < end and not FM_KEY.match(lines[j]):
                j += 1
            return i, j
    return None


def _scalar(value) -> str:
    """One frontmatter value as YAML that reads back as the value it was set to.

    Only strings can need anything doing to them, and most need nothing: written
    bare, `approved` reads back as "approved". The exception is a string whose
    bare form YAML resolves to some *other* type, and the case that forced this
    is a Discord thread id:

        thread_id: 1548070648281038848

    written bare, that is an integer. It is also larger than JavaScript's
    Number.MAX_SAFE_INTEGER, so n8n's JSON.parse rounds it to ...038800 and the
    reply goes to a thread that does not exist. obsidian-local-rest-api quoted
    it and this has to as well, or moving n8n across silently corrupts it.

    The test is a round trip rather than a list of dangerous-looking shapes: a
    value is quoted exactly when reading it back would not return the same text.
    That keeps `timestamp: 2026-09-12T13:57:17Z` bare - YAML resolves it to a
    datetime, but `_as_written` renders that to the identical string, so nothing
    is lost and the vault's convention is undisturbed - while quoting `"true"`,
    `"null"`, `"0123"` and anything carrying a `: ` that would otherwise parse
    as a mapping.

    The round trip is the rule for every type, not only for strings. int, float
    and bool all render to text YAML reads back as the same value, so str() is
    the whole of their handling. None does not: `None` is not YAML's null, it is
    the string "None", so there is no rendering of it that survives - which is
    why it is refused rather than written.
    """
    if value is None:
        # set_frontmatter refuses a bare None before it reaches here, with a
        # message naming both spellings of delete. This catches a None *inside*
        # a list, where `- None` would come back as the string "None" with
        # nothing to see.
        raise VaultError(
            "a frontmatter value cannot be null. Pass the value to set, or "
            "delete the field."
        )
    if not isinstance(value, str):
        return str(value)

    import yaml

    try:
        resolved = _as_written(yaml.safe_load(value))
        if isinstance(resolved, str) and resolved == value:
            return value
    except Exception:  # noqa: BLE001 - unparseable bare means it must be quoted
        pass

    # json.dumps emits a double-quoted scalar with JSON's escapes, which YAML's
    # double-quoted style accepts unchanged - and is the form the plugin wrote.
    return json.dumps(value, ensure_ascii=False)


def _render_value(key: str, value) -> list[str]:
    """Render one frontmatter key, in the shape Conventions mandates for it.

    Three shapes, chosen explicitly rather than by a general YAML dumper:
    inline flow for `tags`, a block sequence for `expires`, scalar for the rest.
    yaml.dump would sort the keys and render tags as a block list, both of which
    check_frontmatter.py reports as violations.
    """
    if isinstance(value, (list, tuple)):
        if key == "tags":
            # Left bare deliberately. Conventions constrains tags to plain
            # lowercase words, and quoting them would be the violation rather
            # than the fix.
            joined = ", ".join(str(item) for item in value)
            return [f"{key}: [{joined}]"]
        out = [f"{key}:"]
        for entry in value:
            if isinstance(entry, dict):
                # 'date' first, per the expires schema; everything else follows.
                keys = [k for k in ("date", "what") if k in entry]
                keys += [k for k in entry if k not in keys]
                first, *rest = keys
                out.append(f"  - {first}: {_scalar(entry[first])}")
                out += [f"    {k}: {_scalar(entry[k])}" for k in rest]
            else:
                out.append(f"  - {_scalar(entry)}")
        return out
    return [f"{key}: {_scalar(value)}"]


def _insert_at(lines: list[str], start: int, end: int, key: str) -> int:
    """Line index at which a new key belongs, honouring FIELD_ORDER."""
    if key not in FIELD_ORDER:
        return end
    rank = FIELD_ORDER.index(key)
    for i in range(start, end):
        match = FM_KEY.match(lines[i])
        if not match:
            continue
        existing = match.group(1)
        if existing in FIELD_ORDER and FIELD_ORDER.index(existing) > rank:
            return i
    return end


def set_frontmatter(text: str, key: str, value=None, *, delete: bool = False) -> str:
    """Set or remove one frontmatter key, leaving every other byte untouched.

    `value=None` is the signature's placeholder for "deleting, so no value" and
    is refused on its own. It is what a caller sends by omitting the argument
    and by PATCHing a JSON `null`, and neither means "write the text None" -
    which is all that could be written, YAML having no way back from it.
    """
    if not delete and value is None:
        raise VaultError(
            f"no value given for {key!r}. Pass the value to set, or delete the "
            "field instead - delete=true as a tool argument, Operation: delete "
            "as a header - a null is not a value this vault's frontmatter "
            "carries."
        )

    lines = normalise_body(text).split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # split() leaves a trailing empty from the final newline

    bounds = frontmatter_bounds(lines)
    if bounds is None:
        raise VaultError("note has no frontmatter block to edit")
    start, end = bounds

    span = _key_span(lines, start, end, key)

    if delete:
        if span is None:
            return normalise_body("\n".join(lines))
        lines[span[0] : span[1]] = []
        return normalise_body("\n".join(lines))

    rendered = _render_value(key, value)
    if span is None:
        at = _insert_at(lines, start, end, key)
        lines[at:at] = rendered
    else:
        lines[span[0] : span[1]] = rendered
    return normalise_body("\n".join(lines))


def bump_timestamp(text: str) -> str:
    """Set `timestamp` to now. Applied to every write, so a convention this
    mechanical never depends on the model remembering it."""
    return set_frontmatter(text, "timestamp", utc_now())
