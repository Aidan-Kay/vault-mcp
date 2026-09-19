"""File-level vault operations, shared by the MCP tools and the REST routes.

Everything here resolves a path, reads, edits in memory, and replaces the file
atomically. Both surfaces call these, so the resolver, the conventions and every
trap are handled exactly once regardless of which one the caller used.

Each function returns the confirmation string the caller reports back, naming
the resolved heading path where there was one. A successful call is meant to be
self-evidencing: the model can see it hit the section it meant.
"""

from __future__ import annotations

import os
import re
import shutil
import urllib.parse

from . import documents
from . import edit
from . import vault
from .config import settings
from .vault import VaultError

_LINK_TARGET = re.compile(r"\]\(([^)]+)\)")

# "appendd" is not a word. Reported back to the model verbatim, so it matters.
_PAST_TENSE = {"replace": "replaced", "append": "appended", "prepend": "prepended"}


def _timestamped(text: str) -> tuple[str, str]:
    """Bump `timestamp`, or say why it was not bumped.

    A note written without frontmatter has nothing to bump. That is reported
    rather than silently skipped - a missing timestamp is a convention breach
    the model should see immediately, not discover in a later checker run.
    """
    lines = vault.normalise_body(text).split("\n")
    if vault.frontmatter_bounds(lines) is None:
        return text, " (no frontmatter, so timestamp not bumped)"
    return vault.bump_timestamp(text), ""


def patch(
    path: str,
    target: str,
    operation: str,
    content: str,
    target_scope: str = "content",
) -> str:
    """Replace, prepend to, or append to one heading's section."""
    resolved = vault.safe_resolve(path, writing=True)
    rel = vault.relpath(resolved)
    text = vault.read_text(resolved)
    updated, heading = edit.patch_section(
        text, target, operation, content, target_scope=target_scope, note=rel
    )
    updated, note = _timestamped(updated)
    vault.atomic_write(resolved, updated)
    return f"{_PAST_TENSE[operation]} {heading!r} in {rel}{note}"


def append(path: str, content: str, create_if_missing: bool = False) -> str:
    """Append a block to the end of a note."""
    resolved = vault.safe_resolve(path, must_exist=not create_if_missing, writing=True)
    rel = vault.relpath(resolved)

    if not resolved.exists():
        # The same tail as write(). Nothing is invented here either - frontmatter
        # still has to arrive in `content` - but a note created down this branch
        # now reaches disk under the rules every other write obeys. It used to be
        # the one write path that skipped _timestamped, which meant the branch a
        # model is told to prefer was also the only one that quietly broke the
        # convention, and said "no frontmatter added" even when it was given some.
        updated, note = _timestamped(content)
        vault.atomic_write(resolved, updated)
        return f"created {rel} with the supplied content{note}"

    updated = edit.append_to_note(vault.read_text(resolved), content)
    updated, note = _timestamped(updated)
    vault.atomic_write(resolved, updated)
    return f"appended to {rel}{note}"


def write(path: str, content: str, overwrite: bool = False) -> str:
    """Create a note, or replace one wholesale.

    The only guard in a system with no safety net: a create that silently
    clobbers is indistinguishable from a create that worked.
    """
    resolved = vault.safe_resolve(path, must_exist=False, writing=True)
    rel = vault.relpath(resolved)
    existed = resolved.exists()

    if existed and not overwrite:
        raise VaultError(
            f"{rel} already exists. Pass overwrite=true to replace it, or use "
            "vault_patch to edit one section."
        )

    updated, note = _timestamped(content)
    vault.atomic_write(resolved, updated)
    return f"{'overwrote' if existed else 'created'} {rel}{note}"


def set_frontmatter(path: str, key: str, value=None, delete: bool = False) -> str:
    """Set or remove one frontmatter key, leaving every other byte untouched."""
    resolved = vault.safe_resolve(path, writing=True)
    rel = vault.relpath(resolved)
    text = vault.read_text(resolved)

    updated = vault.set_frontmatter(text, key, value, delete=delete)
    if key != "timestamp":
        updated, _ = _timestamped(updated)
    vault.atomic_write(resolved, updated)
    return f"{'removed' if delete else 'set'} {key!r} in {rel}"


def set_body(path: str, content: str) -> str:
    """Replace everything after the frontmatter, leaving the block untouched.

    For a note whose prose is regenerated wholesale but whose frontmatter is
    written once and kept - the weekly summaries. Their only option before was
    PUT, which takes the frontmatter with it, so those notes ended up with none
    at all and nothing in the vault could describe them.

    A note with no frontmatter has no prefix to keep, so this is the same as a
    PUT for it. That is the honest answer rather than an error: the caller asked
    for the body to be the content, and afterwards it is.
    """
    resolved = vault.safe_resolve(path, writing=True)
    rel = vault.relpath(resolved)
    text = vault.read_text(resolved)

    # Sliced rather than re-rendered. The frontmatter reaches disk as the exact
    # bytes it already had, including any this server could not parse - four
    # notes in the vault cannot be round-tripped through a YAML dump without
    # losing everything in the block.
    body = vault.without_frontmatter(text)
    prefix = text[: len(text) - len(body)]

    updated, note = _timestamped(prefix + vault.normalise_body(content))
    vault.atomic_write(resolved, updated)
    return f"replaced the body of {rel}{note}"


def delete(path: str) -> str:
    """Delete a note or a filed document. There is no trash - git is the undo."""
    resolved = vault.safe_resolve(path, writing=True, allow_documents=True)
    rel = vault.relpath(resolved)
    if resolved.is_dir():
        raise VaultError(f"{rel} is a directory; only files can be deleted")
    resolved.unlink()
    return f"deleted {rel}"


def _link_forms(rel: str) -> set[str]:
    """Every way this vault writes a link to one note.

    Both encoded and unencoded, both vault-root-absolute and bare. The unencoded
    forms are broken links by convention, but they exist and a move must not
    leave them pointing at nothing.
    """
    encoded = urllib.parse.quote(rel)
    return {rel, encoded, f"/{rel}", f"/{encoded}"}


def _rewrite_links(source_rel: str, dest_rel: str) -> int:
    """Repoint every internal link from source to dest. Returns the note count.

    Obsidian's fileManager.renameFile did this for free; this is the one place
    the migration genuinely loses something, so it is deliberately conservative:
    only link targets inside `](...)` are touched, never prose, and the
    replacement is always written in the encoded root-absolute form the
    conventions require.
    """
    stale = _link_forms(source_rel)
    replacement = "/" + urllib.parse.quote(dest_rel)
    touched = 0

    for note in vault.walk_all_notes():
        text = vault.read_text(note)

        def swap(match: re.Match) -> str:
            target = match.group(1)
            anchor = ""
            if "#" in target:
                target, _, fragment = target.partition("#")
                anchor = "#" + fragment
            return f"]({replacement}{anchor})" if target in stale else match.group(0)

        updated = _LINK_TARGET.sub(swap, text)
        if updated != text:
            # No timestamp bump: repointing a link is a mechanical consequence of
            # someone else's move, not an edit to this note's content.
            vault.atomic_write(note, updated)
            touched += 1

    return touched


def move(source: str, destination: str, update_links: bool = True) -> str:
    """Move or rename a note, optionally repointing every link to it."""
    # Refused outright under a write scope rather than narrowed to fit one.
    # _rewrite_links() writes through vault.atomic_write directly, over every
    # note in the vault, without going near safe_resolve - so the one guard that
    # would contain this does not see it. Allowing a "scoped" move would mean a
    # confined caller could still rewrite the whole vault, which is worse than
    # not offering the tool.
    scope = vault.current_write_scope()
    if scope is not None:
        raise VaultError(
            f"this request may only write to {scope!r}, and moving a note touches "
            "every note that links to it, so it cannot be scoped. Nothing was changed."
        )

    src = vault.safe_resolve(source, writing=True, allow_documents=True)
    dest = vault.safe_resolve(destination, must_exist=False, writing=True, allow_documents=True)

    if dest.exists():
        raise VaultError(f"{vault.relpath(dest)} already exists")
    if src.is_dir():
        raise VaultError(f"{vault.relpath(src)} is a directory; only files can be moved")

    source_rel = vault.relpath(src)
    dest_rel = dest.resolve().relative_to(vault.ROOT).as_posix()

    if documents.is_document(src) != documents.is_document(dest):
        # A move is a move, not a conversion. Renaming a PDF to .md would leave
        # a file the read path decodes as text and the chunker parses for
        # frontmatter, which is the corruption below by a slower route.
        raise VaultError(
            f"cannot move {source_rel} to {dest_rel}: a move may not change a "
            "file between a note and a document. Rename it within its own kind."
        )

    if documents.is_document(src):
        # The Files/ parent rule is deliberately not enforced here, where upload
        # enforces it. That rule contains the credential that carries *bytes*,
        # and a move carries none - it moves bytes already in the vault. The
        # pipeline token that can upload is scoped, and a scoped caller cannot
        # move at all (see above), so the only caller who could move a document
        # out of a Files/ folder is one that could already write any note
        # anywhere. check_documents.py reports a misfiled document as a warning,
        # which is the right weight for a convention breach that costs nothing
        # but discoverability.
        #
        # Byte-preserving, and every step of the note path is wrong for it.
        # read_text would decode the PDF with errors="replace", _timestamped
        # would give it a YAML header, and atomic_write would re-encode the
        # result as UTF-8 - which is not a move but a shredder. There is no
        # timestamp to bump either: a document has no frontmatter, and the
        # bytes are the same bytes wherever they sit.
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(src, dest)
        except OSError:
            # Different filesystems under one vault root - a bind mount, a
            # mounted share. copy2 keeps the mtime, which is the closest thing
            # a document has to a timestamp.
            shutil.move(str(src), str(dest))
        note = ""
    else:
        text = vault.read_text(src)
        updated, note = _timestamped(text)

        dest.parent.mkdir(parents=True, exist_ok=True)
        vault.atomic_write(dest, updated)
        src.unlink()

    message = f"moved {source_rel} to {dest_rel}{note}"
    if update_links:
        touched = _rewrite_links(source_rel, dest_rel)
        message += f"; repointed links in {touched} note(s)"
    else:
        message += "; links NOT updated"
    return message


def upload(path: str, data: bytes, overwrite: bool = False) -> dict:
    """File raw bytes as a document, extract its text, and report what landed.

    The containment control is the parent folder, not the credential. A
    document may only be written directly inside a folder named `Files`, which
    is where this vault's convention already puts them, so a pipeline token that
    can carry bytes cannot drop a PDF anywhere a note lives. That is stricter
    than a suffix allowlist alone and it replaced an earlier design that scoped
    the pipeline's credential to a staging folder - the convention turned out to
    be the better enforcement point, because it is the thing a human reading the
    vault can also check.

    The `Files` folder is created on demand. Lyra picks the destination inside
    the n8n workflow and the first document filed beside a note would otherwise
    need a separate folder-creation capability to land at all.

    Re-uploading identical bytes is a satisfied no-op rather than a conflict.
    The same attachment *will* arrive twice - a thread reprocessed, a statement
    forwarded on, a run retried - and a pipeline should not have to interpret an
    error to find out that what it wanted is already true. Different bytes at
    the same path are a real collision and are refused unless overwrite is set.

    Extraction runs here, on the server's own read of the file, rather than
    taking whatever the uploading workflow says the document contains. The two
    reads serve different questions and only one of them decides what search can
    find.
    """
    if not data:
        raise VaultError("refusing to file an empty document")

    resolved = vault.safe_resolve(path, must_exist=False, writing=True, allow_documents=True)
    rel = vault.relpath(resolved)

    if not documents.is_document(resolved):
        raise VaultError(
            f"{rel} is not an allowlisted document. Uploadable suffixes are: "
            f"{documents.suffix_list()}."
        )

    if resolved.parent.name != settings.doc_files_dir:
        raise VaultError(
            f"a document must be filed directly inside a folder named "
            f"{settings.doc_files_dir!r}, and {rel} is not. File it beside the "
            f"note that owns it, as "
            f"<folder>/{settings.doc_files_dir}/<YYYY-MM-DD Issuer - Type>"
            f"{resolved.suffix}."
        )

    digest = documents.sha256_bytes(data)
    existed = resolved.exists()
    status = "filed"

    if existed:
        if documents.sha256_file(resolved) == digest:
            status = "unchanged"
        elif overwrite:
            status = "replaced"
        else:
            raise VaultError(
                f"{rel} already holds a different document. Pass overwrite=true "
                "to replace it, or file this one under a name of its own."
            )

    if status != "unchanged":
        resolved.parent.mkdir(parents=True, exist_ok=True)
        vault.atomic_write_bytes(resolved, data)

    # `status` says what happened to the file and `extraction` what happened to
    # its text, and they are separate because the interesting case is the one
    # where they disagree: a scan files perfectly and extracts to nothing, and a
    # caller that had only one field would have to guess which it was being
    # told about.
    extraction = documents.extract(resolved)
    return {
        "path": rel,
        "sha256": digest,
        "size": len(data),
        "status": status,
        "extraction": extraction.status,
        "pages": extraction.pages,
        "has_text_layer": extraction.has_text_layer,
        "extracted_chars": extraction.extracted_chars,
        "detail": extraction.detail,
    }
