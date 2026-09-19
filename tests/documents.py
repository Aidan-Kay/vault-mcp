"""Filed documents: the containment rule, the byte path, and extraction.

Three things are being asserted, and they fail in three different ways.

The containment rule is a security control. A pipeline credential that can
carry raw bytes into the vault is the widest capability this server offers, and
the only thing narrowing it is that a document must land directly inside a
folder named `Files`. Every way of not doing that has a check here.

The byte path is a correctness control, and the failure it prevents is silent.
Before this, `move()` read through `read_text`, stamped a YAML header on the
result and wrote it back as UTF-8 - which for a PDF is not a move but a
shredder, and nothing would have said so until somebody opened the file months
later. So the move checks compare hashes rather than existence.

Extraction is measured rather than asserted wherever it can be. The one thing
worth asserting outright is the negative: a document with no text layer must be
*visibly* unsearchable, because the file is already in the vault by the time
anyone finds out and "refuse it" is no longer on offer.

Like write_scope, this needs a writable vault and so points VAULT_PATH at a
temp tree before importing src.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

_VAULT = Path(tempfile.mkdtemp(prefix="vault-docs-"))
os.environ["VAULT_PATH"] = str(_VAULT)

from src import chunker, documents, operations, vault  # noqa: E402
from src.config import settings  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "vault"

# The fixture PDFs, reused rather than regenerated: one with a text layer and
# one deliberately without, which is the pair every check below needs.
WITH_TEXT = FIXTURES / "Home/Utilities/Files/2026-09-17 Kestrel Energy - Contract Confirmation.pdf"
NO_TEXT = FIXTURES / "Home/Utilities/Files/2026-08-06 Brightvale Power - Tariff Change Notice.pdf"

FAILURES: list[str] = []


def check(name: str, actual, expected) -> None:
    if actual == expected:
        return
    FAILURES.append(f"{name}\n    expected: {expected!r}\n    actual:   {actual!r}")


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


def allowlist() -> None:
    check("a pdf is a document", documents.is_document(Path("a/Files/x.pdf")), True)
    check("case does not matter", documents.is_document(Path("a/Files/X.PDF")), True)
    check("a note is not", documents.is_document(Path("a/x.md")), False)
    check("nor is anything else", documents.is_document(Path("a/Files/x.exe")), False)
    check("nor a suffixless file", documents.is_document(Path("a/Files/x")), False)


def containment() -> None:
    """A document may only land directly inside a folder named `Files`."""
    data = WITH_TEXT.read_bytes()

    refused("the vault root is not Files/", lambda: operations.upload("x.pdf", data))
    refused("a note's folder is not Files/", lambda: operations.upload("Home/x.pdf", data))
    refused(
        "a folder merely named like it does not count",
        lambda: operations.upload("Home/Filesx/x.pdf", data),
    )
    refused(
        "nor does one nested below it",
        lambda: operations.upload("Home/Files/Sub/x.pdf", data),
    )
    refused(
        "case matters, because the convention is a literal folder name",
        lambda: operations.upload("Home/files/x.pdf", data),
    )
    refused("traversal is still traversal", lambda: operations.upload("../Files/x.pdf", data))
    refused(
        "a suffix off the allowlist is refused before the folder is considered",
        lambda: operations.upload("Home/Files/x.exe", data),
    )
    refused("an empty body is not a document", lambda: operations.upload("Home/Files/x.pdf", b""))

    # And the rule the containment exists to permit.
    result = operations.upload("Home/Utilities/Files/2026-09-17 Kestrel Energy - Contract.pdf", data)
    check("a correctly filed document lands", result["status"], "filed")
    check(
        "its Files/ folder was created on demand",
        (_VAULT / "Home/Utilities/Files").is_dir(),
        True,
    )
    check("the bytes on disk are the bytes sent",
          (_VAULT / "Home/Utilities/Files/2026-09-17 Kestrel Energy - Contract.pdf").read_bytes(),
          data)


def duplicates() -> None:
    """The same attachment arriving twice is a satisfied no-op, not a conflict."""
    data = WITH_TEXT.read_bytes()
    path = "Home/Utilities/Files/dup.pdf"

    first = operations.upload(path, data)
    check("first upload files it", first["status"], "filed")

    again = operations.upload(path, data)
    check("identical bytes are a no-op", again["status"], "unchanged")
    check("and report the same hash", again["sha256"], first["sha256"])
    check("the hash is of the bytes sent", again["sha256"], hashlib.sha256(data).hexdigest())

    other = NO_TEXT.read_bytes()
    refused("different bytes at the same path are a conflict", lambda: operations.upload(path, other))
    check(
        "and the refusal changed nothing",
        (_VAULT / path).read_bytes(),
        data,
    )

    replaced = operations.upload(path, other, overwrite=True)
    check("overwrite is how a caller means it", replaced["status"], "replaced")
    check("and the bytes really changed", (_VAULT / path).read_bytes(), other)


def opacity() -> None:
    """A document is read, moved and deleted. It is never edited."""
    data = WITH_TEXT.read_bytes()
    path = "Home/Utilities/Files/opaque.pdf"
    operations.upload(path, data)

    refused("patch", lambda: operations.patch(path, "Page 1", "replace", "x"))
    refused("append", lambda: operations.append(path, "x"))
    refused("set_body", lambda: operations.set_body(path, "x"))
    refused("set_frontmatter", lambda: operations.set_frontmatter(path, "title", "x"))
    refused("write, which would replace it with markdown", lambda: operations.write(path, "x", overwrite=True))
    check("and none of them touched it", (_VAULT / path).read_bytes(), data)

    # Reading works, and reads the extracted text rather than the bytes.
    text = vault.read_note(path)
    check("the read is markdown, not bytes", text.startswith("## Page 1"), True)
    check("and it holds what the document says", "KE-8842071" in text, True)
    refused("a section of a document cannot be addressed", lambda: vault.read_note(path, "Page 1"))

    # read_text is the note door, and a document must not fit through it. This
    # is the latent defect: errors="replace" decodes anything at all, so before
    # documents existed this returned a page of replacement characters and every
    # caller downstream believed it was a note.
    refused("read_text refuses a document outright", lambda: vault.read_text(_VAULT / path))
    refused(
        "and refuses anything else that is not a note",
        lambda: vault.read_text(_VAULT / "Home/Utilities/Files/x.png"),
    )


def moving() -> None:
    """A move preserves every byte, and may not change a file's kind."""
    data = WITH_TEXT.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    source = "Home/Utilities/Files/2026-09-17 Kestrel - Confirmation.pdf"
    operations.upload(source, data)

    # A note links to it, so the rewrite has something to repoint. The link is
    # written in the encoded root-absolute form the conventions require.
    note = _VAULT / "Home/Utilities/Electricity.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "---\ntitle: Electricity\n---\n\n## Documents\n\n"
        "| Document | Contents |\n| --- | --- |\n"
        "| [Confirmation](/Home/Utilities/Files/2026-09-17%20Kestrel%20-%20Confirmation.pdf) | The contract |\n",
        encoding="utf-8",
    )

    dest = "Home/Utilities/Files/2026-09-17 Kestrel Energy - Contract Confirmation.pdf"
    operations.move(source, dest)

    moved = _VAULT / dest
    check("the move is byte-preserving", hashlib.sha256(moved.read_bytes()).hexdigest(), digest)
    check("the source is gone", (_VAULT / source).exists(), False)
    check("no YAML header was stapled on", moved.read_bytes()[:5], b"%PDF-")
    check(
        "and the link was repointed",
        "2026-09-17%20Kestrel%20Energy%20-%20Contract%20Confirmation.pdf"
        in note.read_text(encoding="utf-8"),
        True,
    )

    refused(
        "a move may not turn a document into a note",
        lambda: operations.move(dest, "Home/Utilities/Files/x.md"),
    )
    refused(
        "nor a note into a document",
        lambda: operations.move("Home/Utilities/Electricity.md", "Home/Utilities/Files/x.pdf"),
    )
    check("and the note is untouched", note.read_bytes()[:3], b"---")

    check("a document can be deleted", operations.delete(dest), f"deleted {dest}")


def extraction() -> None:
    """What comes out of a PDF, and what is said when nothing does."""
    good = documents.extract(WITH_TEXT)
    check("a text layer is found", good.has_text_layer, True)
    check("the status says so", good.status, "extracted")
    check("it is searchable", good.searchable, True)
    check("the page count is real", good.pages, 1)
    check("the identifier survives", "KE-8842071" in good.markdown, True)
    check(
        "a document with no headings of its own is structured by page",
        good.markdown.startswith("## Page 1"),
        True,
    )

    # The scanned-bill case. It must be visibly unsearchable, not silently so:
    # the file is in the vault either way, and only one of those can be acted on.
    blank = documents.extract(NO_TEXT)
    check("no text layer is detected", blank.has_text_layer, False)
    check("it is not searchable", blank.searchable, False)
    check("nothing is invented", blank.markdown, "")
    check("the page is counted even so", blank.pages, 1)
    check(
        "and the status names the reason rather than reporting success",
        blank.status in {"needs_ocr", "ocr", "no_text"},
        True,
    )
    check("which is never a bare empty extraction", blank.status == "extracted", False)

    # The threshold between the two, which is what separates them. Measured on
    # real documents: 637 to 4296 characters per page with a text layer, and
    # exactly 0 for the same documents rasterised.
    check("the threshold sits between them", 0 < documents.MIN_CHARS_PER_PAGE < 637, True)


def chunking() -> None:
    """A document chunks like a note, and is attributed to its own path."""
    data = WITH_TEXT.read_bytes()
    path = "Home/Utilities/Files/2026-09-17 Kestrel Energy - Contract Confirmation.pdf"
    operations.upload(path, data)

    chunks = chunker.chunk_note(_VAULT / path)
    check("it produced chunks", len(chunks) > 0, True)
    check("attributed to the document, not the note beside it", chunks[0]["path"], path)
    check(
        "titled by the filename, which in this vault names date, issuer and kind",
        chunks[0]["title"],
        "2026-09-17 Kestrel Energy - Contract Confirmation",
    )
    check(
        "described by the folder that owns it, since `Files` says nothing",
        chunks[0]["description"],
        "Utilities",
    )
    check(
        "and the embedding scaffold carries both",
        "Kestrel" in chunks[0]["embed_text"] and "Utilities" in chunks[0]["embed_text"],
        True,
    )

    # A document that yielded no text contributes nothing to the index. This is
    # a behavioural check, not a check of the early return in chunk_document:
    # `searchable` means "the markdown is not empty", so the section pass
    # produces nothing either way and deleting that branch changes no output.
    # The behaviour is still worth asserting - an empty chunk would be a row in
    # the index that can never answer anything - but the guarantee comes from
    # the definition rather than from the branch, and a reader of this file
    # should not be left thinking it is the branch under test.
    blank_path = "Home/Utilities/Files/2026-08-06 Brightvale Power - Tariff Change Notice.pdf"
    operations.upload(blank_path, NO_TEXT.read_bytes())
    check("an unsearchable document is absent from the index", chunker.chunk_note(_VAULT / blank_path), [])


def walking() -> None:
    """The index walk sees documents. The link-rewriting walk must not.

    The plan had both walks gaining the document suffixes. Each of the three
    callers of walk_all_notes breaks differently on a PDF, so only the index
    walk got them, and this is the check that keeps it that way.
    """
    indexed = {vault.relpath(p) for p in vault.walk_notes()}
    linkable = {vault.relpath(p) for p in vault.walk_all_notes()}

    check(
        "documents are indexed",
        any(p.endswith(".pdf") for p in indexed),
        True,
    )
    check(
        "and are never offered as things that could contain a link",
        [p for p in linkable if p.endswith(".pdf")],
        [],
    )
    check("notes are still in both", "Home/Utilities/Electricity.md" in indexed, True)


def main() -> int:
    allowlist()
    containment()
    duplicates()
    opacity()
    moving()
    extraction()
    chunking()
    walking()

    if report():
        return 1
    print("documents: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(_VAULT, ignore_errors=True)
