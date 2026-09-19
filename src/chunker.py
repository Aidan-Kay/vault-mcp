"""Markdown -> chunk records.

A chunk is a heading-scoped span of one note, plus enough scaffolding that its
embedding still identifies what the fragment is about.

Two shapes get different treatment. Prose packs to a token target. A section
whose root-level content is mostly one list of top-level items is a *flat list*
- a log, an inbox, a register of tasks written as bullets - and packing it
averages a dozen unrelated subjects into one embedding, so each item becomes its
own chunk instead. See _flat_list_pieces for what disqualifies a list.
"""

from __future__ import annotations

import logging
import re
import statistics
from pathlib import Path
from typing import NamedTuple

import frontmatter

from . import documents, vault
from .config import settings

log = logging.getLogger(__name__)

# Characters per token. A tokeniser dependency is not warranted: the target is
# soft, and nomic truncates at 8192 regardless.
CHARS_PER_TOKEN = 3.6

_PARAGRAPH_RE = re.compile(r"\n{2,}")
_SECTION_HEADING_MAX_DEPTH = 3  # '#' to '###'; deeper headings stay inline

# A top-level list item, bullet or ordered. Anchored hard at column zero rather
# than allowing CommonMark its three spaces of slack, because this vault indents
# nested items by two and reading those as top-level would flatten every tree
# into a flat list.
_LIST_ITEM_RE = re.compile(r"^([-*+]|\d+[.)])\s+(?=\S)")
_LIST_MARKER_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+", re.M)
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
# Wikilinks and inline markdown links, image embeds included.
_LINK_RE = re.compile(r"\[\[[^\]]*\]\]|!?\[[^\]]*\]\([^)]*\)")

# A section is a flat list when at least this share of its root-level blocks are
# items of a single list. The ratio is upstream's; the floor on item count is
# ours, and it is what stops a three-line list becoming three chunks that each
# say less than the section they came from.
FLAT_LIST_RATIO = 0.7
FLAT_LIST_MIN_ITEMS = 7

# And the median item has to say something. Measured over this vault: a log
# entry or a task runs 10 to 30 tokens, while an ingredient ("750 g beef
# mince"), a film title and a '**Owner:** Aidan' profile line run 3 to 7. The
# short ones are a register of names, not a list of subjects, and a chunk per
# name is 160 embeddings of two words each. Anywhere between 8 and 10 picks out
# exactly the same sections here, so the cut sits in a gap rather than on an
# edge that wants defending.
FLAT_LIST_MIN_ITEM_TOKENS = 8


class Section(NamedTuple):
    """A heading-scoped span, before it is cut into chunks."""

    breadcrumb: str
    line: int
    body: str
    heading: bool  # body's first line is the heading that named it


class Block(NamedTuple):
    """A root-level block of a section body: one list item, or one paragraph."""

    offset: int  # line index within the section body
    lines: list[str]
    item: bool
    marker: str  # 'bullet' or 'ordered' for an item, '' otherwise


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """(char offset, paragraph) pairs. Offsets let sub-chunks keep line numbers."""
    parts: list[tuple[int, str]] = []
    pos = 0
    for match in _PARAGRAPH_RE.finditer(text):
        parts.append((pos, text[pos : match.start()]))
        pos = match.end()
    parts.append((pos, text[pos:]))
    return [(offset, body) for offset, body in parts if body.strip()]


def _split_oversized(text: str, start_line: int) -> list[tuple[int, str]]:
    """Split at paragraph boundaries, carrying overlap into each next chunk."""
    target_chars = int(settings.chunk_target_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(settings.chunk_overlap_tokens * CHARS_PER_TOKEN)
    min_chars = int(settings.chunk_min_tokens * CHARS_PER_TOKEN)

    out: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_offset: int | None = None
    carried = ""

    def flush() -> None:
        nonlocal buffer, buffer_offset, carried
        if not buffer:
            return
        body = "\n\n".join(buffer)
        line = start_line + text[: buffer_offset or 0].count("\n")
        out.append((line, (carried + body) if carried else body))
        # Snap the overlap to a whitespace boundary so a chunk never opens
        # mid-word.
        tail = body[-overlap_chars:] if overlap_chars else ""
        if tail and len(tail) < len(body):
            space = tail.find(" ")
            tail = tail[space + 1 :] if space != -1 else tail
        carried = (tail + "\n\n") if tail.strip() else ""
        buffer, buffer_offset = [], None

    for offset, paragraph in _paragraphs(text):
        if buffer_offset is None:
            buffer_offset = offset
        held = sum(len(p) + 2 for p in buffer)
        # Only break if what is already held stands on its own. Without this, a
        # heading line followed by an unsplittable table (no blank lines, so one
        # paragraph) is emitted as a chunk containing nothing but the heading.
        if buffer and held + len(paragraph) > target_chars and held >= min_chars:
            flush()
            buffer_offset = offset
        buffer.append(paragraph)
    flush()

    # A trailing remainder below the threshold belongs to the piece before it,
    # not in the index on its own.
    if len(out) > 1 and len(out[-1][1]) < min_chars:
        line, tail = out.pop()
        head_line, head_text = out[-1]
        out[-1] = (head_line, f"{head_text}\n\n{tail}")
    return out


def _sections(text: str) -> list[Section]:
    """One span per heading-scoped region, in document order."""
    lines = text.splitlines()
    headings = [h for h in vault.iter_headings(text) if h.depth <= _SECTION_HEADING_MAX_DEPTH]
    body_start = vault.frontmatter_span(text)

    spans: list[Section] = []

    first_heading_line = headings[0].line if headings else len(lines) + 1
    raw_preamble = "\n".join(lines[body_start : first_heading_line - 1])
    preamble = raw_preamble.strip()
    if preamble:
        # Count what the strip removed rather than assuming it removed nothing:
        # a note with a blank line under its frontmatter would otherwise report
        # its preamble one line early, and the flat-list rule makes those line
        # numbers per-item rather than per-section.
        blanks = raw_preamble[: len(raw_preamble) - len(raw_preamble.lstrip())].count("\n")
        spans.append(Section("", body_start + 1 + blanks, preamble, False))

    stack: list[tuple[int, str]] = []
    for position, heading in enumerate(headings):
        while stack and stack[-1][0] >= heading.depth:
            stack.pop()
        breadcrumb = " > ".join([*(t for _, t in stack), heading.text])
        stack.append((heading.depth, heading.text))

        end = headings[position + 1].line - 1 if position + 1 < len(headings) else len(lines)
        body = "\n".join(lines[heading.line - 1 : end]).strip()
        if body:
            spans.append(Section(breadcrumb, heading.line, body, True))
    return spans


def _root_blocks(lines: list[str]) -> list[Block]:
    """Split a section body into its root-level blocks.

    Not a Markdown parser, and does not need to be - the only question asked of
    the result is how many root-level blocks are items of one list. The rules: a
    zero-indent list marker opens an item, an indented line continues whatever
    is open, a zero-indent line after a blank one opens a paragraph, and a
    zero-indent line straight after an item is that item's lazy continuation.
    Fences swallow everything until they close, or a shell block whose options
    start with '-' would read as a list.
    """
    blocks: list[Block] = []
    fence = ""
    previous_blank = True

    for offset, line in enumerate(lines):
        stripped = line.strip()

        if fence:
            blocks[-1].lines.append(line)
            if stripped.startswith(fence):
                fence = ""
            continue

        if not stripped:
            if blocks:
                blocks[-1].lines.append(line)
            previous_blank = True
            continue

        indented = line[:1] in (" ", "\t")
        match = None if indented else _LIST_ITEM_RE.match(line)

        if match:
            marker = "ordered" if match.group(1)[0].isdigit() else "bullet"
            blocks.append(Block(offset, [line], True, marker))
        elif blocks and (indented or not previous_blank):
            blocks[-1].lines.append(line)
        else:
            blocks.append(Block(offset, [line], False, ""))

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            fence = fence_match.group(1)
        previous_blank = False

    return blocks


def _block_text(blocks: list[Block]) -> str:
    return "\n".join("\n".join(block.lines) for block in blocks).strip()


def _is_bare_link(text: str) -> bool:
    """True when an item is a pointer and nothing else.

    '## Related notes' is a flat list by every structural measure and must not
    be chunked as one: splitting it produces four chunks that each hold one
    link, competing for a top-k slot while carrying no answer. What separates it
    from a real list is that nothing survives removing the links.
    """
    residue = _LINK_RE.sub("", _LIST_MARKER_RE.sub("", text))
    return not residue.strip(" \t\r\n-–—:;,.|")


def _flat_list_pieces(section: Section) -> list[tuple[int, str]] | None:
    """(line, text) per item when the section is a flat list, else None.

    Each disqualifier below is a shape this vault actually holds, and every one
    of them was added because the vault held it: too few items to be worth
    splitting, a section that is mostly prose with a list in it, two lists
    rather than one, a numbered list, items too short to answer anything, and a
    list of bare links.
    """
    lines = section.body.splitlines()
    skip = 1 if section.heading else 0
    blocks = _root_blocks(lines[skip:])
    items = [block for block in blocks if block.item]

    if len(items) < FLAT_LIST_MIN_ITEMS or len(items) < FLAT_LIST_RATIO * len(blocks):
        return None

    positions = [i for i, block in enumerate(blocks) if block.item]
    if positions[-1] - positions[0] != len(positions) - 1:
        return None  # items either side of a paragraph: two lists, not one
    if len({block.marker for block in items}) != 1:
        return None  # a bullet list and an ordered one: ditto
    if items[0].marker == "ordered":
        # A numbered list is a sequence and a sequence is one unit. Step four of
        # a recipe, or of a release procedure, answers nothing on its own and
        # loses the order that made it worth numbering.
        return None

    sizes = [estimate_tokens(_block_text([block])) for block in items]
    if statistics.median(sizes) < FLAT_LIST_MIN_ITEM_TOKENS:
        return None
    if sum(_is_bare_link(_block_text([b])) for b in items) >= FLAT_LIST_RATIO * len(items):
        return None

    base = section.line + skip
    pieces: list[tuple[int, str]] = []

    # Prose either side of the list stays whole, and is not folded into the
    # first or last item: the point of the rule is that an item carries its own
    # signal and nothing else's.
    lead = blocks[: positions[0]]
    if _block_text(lead):
        pieces.append((base + lead[0].offset, _block_text(lead)))
    for block in items:
        pieces.append((base + block.offset, _block_text([block])))
    trail = blocks[positions[-1] + 1 :]
    if _block_text(trail):
        pieces.append((base + trail[0].offset, _block_text(trail)))
    return pieces


def _merge_small(spans: list[Section]) -> list[Section]:
    """Fold sub-threshold sections into the following sibling.

    A bare '## Related' with four links is not a retrievable unit; on its own it
    competes with real content for a top-k slot while carrying no answer.
    """
    merged: list[Section] = []
    pending: Section | None = None

    for span in spans:
        if pending is not None:
            span = Section(
                pending.breadcrumb,
                pending.line,
                f"{pending.body}\n\n{span.body}",
                pending.heading,
            )
            pending = None
        if estimate_tokens(span.body) < settings.chunk_min_tokens:
            pending = span
            continue
        merged.append(span)

    if pending is not None:
        if merged:  # trailing runt: attach to the previous chunk instead
            last = merged[-1]
            merged[-1] = Section(
                last.breadcrumb, last.line, f"{last.body}\n\n{pending.body}", last.heading
            )
        else:
            merged.append(pending)
    return merged


def _chunk_spans(text: str) -> list[tuple[str, int, str]]:
    """(breadcrumb, line, chunk text) for one note, in document order.

    Flat-list sections are barriers: never merged into a neighbour and never
    absorbing one, because a merge would bury the list inside a larger body and
    the rule would silently stop applying to it.

    A barrier strands whatever runt follows it - a '## Related notes' whose only
    possible host was the section just split. Stranding it is not an option: a
    chunk holding four links and nothing else is short enough to win a lexical
    query outright on length normalisation alone, and it carries no answer when
    it does. It attaches to the chunk before it instead, which is the answer
    _merge_small already gives a runt at the end of a note.
    """
    out: list[tuple[str, int, str]] = []
    ordinary: list[Section] = []

    def emit(breadcrumb: str, line: int, body: str) -> None:
        if estimate_tokens(body) > settings.chunk_target_tokens:
            out.extend((breadcrumb, at, piece) for at, piece in _split_oversized(body, line))
        else:
            out.append((breadcrumb, line, body))

    def flush() -> None:
        merged = _merge_small(ordinary)
        ordinary.clear()
        for position, section in enumerate(merged):
            # _merge_small returns a sub-threshold section in one case only:
            # the whole run was a single runt with nothing to merge into. That
            # is the stranded one, and it can only ever be first.
            stranded = position == 0 and estimate_tokens(section.body) < settings.chunk_min_tokens
            if stranded and out:
                breadcrumb, line, body = out[-1]
                out[-1] = (breadcrumb, line, f"{body}\n\n{section.body}")
                continue
            emit(section.breadcrumb, section.line, section.body)

    for section in _sections(text):
        pieces = _flat_list_pieces(section)
        if pieces is None:
            ordinary.append(section)
            continue
        flush()
        for line, piece in pieces:
            emit(section.breadcrumb, line, piece)
    flush()
    return out


def build_embed_text(title: str, description: str, breadcrumb: str, text: str) -> str:
    """Scaffold a chunk for embedding. Never returned to the model.

    The 'search_document: ' prefix is mandatory - nomic-embed-text is
    asymmetric and silently loses recall without it.

    The scaffold carries more weight since the flat-list rule landed: a one-line
    bullet is only retrievable at all because the title, the description and the
    breadcrumb are embedded alongside it.
    """
    header = title
    if description:
        header = f"{title} - {description}" if title else description
    lines = [f"search_document: {header}".rstrip()]
    if breadcrumb:
        lines.append(breadcrumb)
    return "\n".join(lines) + "\n\n" + text


def chunk_note(path: Path) -> list[dict]:
    if documents.is_document(path):
        return chunk_document(path)

    text = vault.read_text(path)
    try:
        meta = dict(frontmatter.loads(text).metadata)
    except Exception:
        meta = {}

    rel = vault.relpath(path)
    title = str(meta.get("title") or Path(rel).stem)
    description = str(meta.get("description") or "")

    return [
        {
            "path": rel,
            "title": title,
            "description": description,
            "breadcrumb": breadcrumb,
            "text": body,
            "embed_text": build_embed_text(title, description, breadcrumb, body),
            "line": line,
        }
        for breadcrumb, line, body in _chunk_spans(text)
    ]


def chunk_document(path: Path) -> list[dict]:
    """A filed document as chunks, attributed to the document's own path.

    Extraction happens at the edge and everything after it is the note path
    unchanged - the same sectioning, the same token target, the same flat-list
    rule. A bill's summary table is a section like any other once it is
    markdown.

    Two things differ, and both come from a document having no frontmatter.
    The title is the filename, which in this vault is not a fallback but the
    convention: `YYYY-MM-DD Issuer - Document Type.pdf` names the date, the
    issuer and the kind, and those are exactly the terms somebody searches for.
    The description is the folder the document sits beside, because `Files` on
    its own says nothing and the owning note's name says what the document is
    about. A document that yielded no text produces no chunks rather than an
    empty one: an unsearchable file should be absent from search, not present
    and silent, and the vault's checker is what surfaces it.
    """
    extraction = documents.extract(path)
    rel = vault.relpath(path)
    if not extraction.searchable:
        log.info(
            "no chunks from %s (%s: %s)", rel, extraction.status, extraction.detail
        )
        return []

    title = path.stem
    parent = path.parent
    if parent.name == settings.doc_files_dir and parent.parent != vault.ROOT:
        description = parent.parent.name
    else:
        description = parent.name

    return [
        {
            "path": rel,
            "title": title,
            "description": description,
            "breadcrumb": breadcrumb,
            "text": body,
            "embed_text": build_embed_text(title, description, breadcrumb, body),
            "line": line,
        }
        for breadcrumb, line, body in _chunk_spans(extraction.markdown)
    ]
