"""The document edge: what counts as one, and how its text is got out.

Everything format-specific lives here and nowhere else. The chunker, the index
and the search path all take markdown, and they take it from a document by the
same route they take it from a note - which is what makes adding `.docx` later
the work of writing an extractor rather than of teaching the chunker what a
document is.

Extraction is deliberately independent of whatever read the file upstream. The
n8n triage workflow reads a PDF to decide where to file it; this reads it again
to decide what the vault can find. If those were one read, what search contains
would depend on which workflow happened to upload the file.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)

# Below this many characters per page, the document is treated as having no text
# layer and is a candidate for OCR. Measured rather than chosen: the seven real
# documents this was built against extract 637 to 4296 characters per page, and
# the same documents rasterised - which is what a scanner produces and what a
# utility company emails - extract exactly 0. Anything in that gap separates
# them, so the threshold sits well clear of both edges rather than on one.
MIN_CHARS_PER_PAGE = 50

_HEADING_RE = re.compile(r"^#{1,6} ", re.M)


def is_document(path: Path | str) -> bool:
    """True when this path names an allowlisted binary document."""
    suffix = Path(path).suffix.lower()
    return suffix in settings.doc_suffixes


def suffix_list() -> str:
    """The allowlist as prose, for the error messages that have to name it."""
    return ", ".join(sorted(settings.doc_suffixes))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Extraction:
    """What one document yielded, and whether that counts as searchable.

    `status` is the field a caller branches on, and it is separate from
    `has_text_layer` on purpose: a scan that OCR read successfully has no text
    layer and is perfectly searchable, while an encrypted file has neither.
    """

    markdown: str
    pages: int
    has_text_layer: bool
    status: str  # extracted | ocr | needs_ocr | no_text | failed
    detail: str = ""

    @property
    def extracted_chars(self) -> int:
        return len(self.markdown.strip())

    @property
    def searchable(self) -> bool:
        return self.extracted_chars > 0

    def as_json(self) -> dict:
        return {
            "pages": self.pages,
            "has_text_layer": self.has_text_layer,
            "extracted_chars": self.extracted_chars,
            "status": self.status,
            "detail": self.detail,
        }


def _failed(detail: str, pages: int = 0) -> Extraction:
    return Extraction(
        markdown="", pages=pages, has_text_layer=False, status="failed", detail=detail
    )


def ocr_available() -> tuple[bool, str]:
    """Whether OCR can run here, and why not when it cannot.

    PyMuPDF shells out to a Tesseract binary and needs its tessdata; neither is
    a Python dependency, so neither arrives with the wheel. Asking once per
    document is cheap and the answer can change under a running server - a
    volume mounted, an image rebuilt - so this is not cached.
    """
    if not settings.doc_ocr:
        return False, "OCR is disabled by DOC_OCR"
    try:
        import pymupdf

        tessdata = pymupdf.get_tessdata()
    except Exception as exc:  # pragma: no cover - depends on the host
        return False, f"Tesseract is not available: {exc}"

    # get_tessdata() hands back TESSDATA_PREFIX verbatim when it is set, without
    # checking anything is there. An environment variable pointing at a folder
    # that does not exist would otherwise read as "OCR is available" and fail
    # only once a scan arrived, which is the worst moment to find out.
    if not tessdata or not Path(tessdata).is_dir():
        return False, f"the Tesseract language folder {tessdata!r} does not exist"
    return True, ""


def _assemble(pages: list[str]) -> str:
    """Per-page markdown into one document, with a structure the chunker can use.

    When layout recovery found headings they are the structure, and the pages
    are joined plainly - a heading that opens page three describes a section
    that began on page two, and inserting a page break between them would cut
    it in half.

    When it found none - a one-page bill is mostly a table and carries no
    heading at all - the page becomes the structural unit, because the chunker
    works in heading-scoped sections and a document with no headings is one
    undivided section however long it is.
    """
    joined = "\n\n".join(page.strip() for page in pages if page.strip())
    if not joined:
        return ""
    if _HEADING_RE.search(joined):
        return joined
    numbered = [
        f"## Page {number}\n\n{page.strip()}"
        for number, page in enumerate(pages, start=1)
        if page.strip()
    ]
    return "\n\n".join(numbered)


def _pdf_pages(path: Path, *, force_ocr: bool) -> list[str]:
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(
        str(path), page_chunks=True, show_progress=False, force_ocr=force_ocr
    )
    return [chunk.get("text", "") for chunk in chunks]


def extract_pdf(path: Path) -> Extraction:
    """A PDF to markdown, OCR'd if it has no text layer and OCR is available."""
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        return _failed(f"cannot open the PDF: {exc}")

    try:
        if doc.needs_pass:
            # Encrypted. Nothing to extract and nothing OCR can do, since the
            # page content cannot be rendered either.
            return _failed("the PDF is password-protected", pages=doc.page_count)
        pages = doc.page_count
        raw = "".join(page.get_text() for page in doc).strip()
    except Exception as exc:
        return _failed(f"cannot read the PDF: {exc}")
    finally:
        doc.close()

    if pages == 0:
        return _failed("the PDF has no pages")

    has_text_layer = len(raw) / pages >= MIN_CHARS_PER_PAGE

    if has_text_layer:
        try:
            markdown = _assemble(_pdf_pages(path, force_ocr=False))
        except Exception as exc:
            log.exception("extraction failed for %s", path)
            return _failed(f"extraction failed: {exc}", pages=pages)
        if not markdown.strip():
            # A text layer that yielded nothing through layout recovery. Rare,
            # and reported rather than smoothed over: the document is in the
            # vault and unsearchable, which is a finding.
            return Extraction("", pages, True, "no_text", "the text layer yielded no markdown")
        return Extraction(markdown, pages, True, "extracted")

    available, why = ocr_available()
    if not available:
        # The file is already in the vault - it was written before this ran -
        # so refusing it is not on offer. Saying plainly that it is unsearchable
        # is, and it is what the vault's unsearchable-document checker reads.
        return Extraction("", pages, False, "needs_ocr", why)

    try:
        markdown = _assemble(_pdf_pages(path, force_ocr=True))
    except Exception as exc:
        log.exception("OCR failed for %s", path)
        return _failed(f"OCR failed: {exc}", pages=pages)

    if not markdown.strip():
        return Extraction("", pages, False, "no_text", "OCR found no text on any page")
    return Extraction(markdown, pages, False, "ocr")


_EXTRACTORS = {".pdf": extract_pdf}


def extract(path: Path) -> Extraction:
    """One allowlisted document to markdown.

    Never raises for a document it was asked to read: an extraction failure
    leaves a file that is already in the vault, so the only useful answer is one
    that says so and can be acted on. A suffix with no extractor is a
    configuration error rather than a document problem, and does raise.
    """
    suffix = Path(path).suffix.lower()
    extractor = _EXTRACTORS.get(suffix)
    if extractor is None:
        raise KeyError(
            f"{suffix!r} is in DOC_SUFFIXES but has no extractor in documents.py"
        )
    return extractor(Path(path))
