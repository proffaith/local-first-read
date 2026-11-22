"""Helpers for the student writing-feedback feature.

Text extraction runs in the web request (in memory only — no S3 persistence)
and the resulting plain text is handed to the Celery task as a task argument.
"""
from __future__ import annotations

from io import BytesIO
from typing import Optional


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB guardrail


class DraftExtractionError(ValueError):
    """Raised when an uploaded draft cannot be parsed to text."""


def extract_draft_text(file_storage, filename: Optional[str] = None) -> str:
    """Extract plain text from an uploaded draft file (in memory).

    Accepts a Werkzeug ``FileStorage`` (or any object with ``.read()``) and
    returns the concatenated plain text. The original binary is held only as
    a ``BytesIO`` buffer for the duration of this call and is never written
    to disk or to S3.
    """
    name = (filename or getattr(file_storage, "filename", "") or "").lower()
    raw = file_storage.read()
    if not raw:
        raise DraftExtractionError("The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DraftExtractionError(
            f"The uploaded file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    if name.endswith(".pdf"):
        return _extract_pdf(raw)
    if name.endswith(".docx"):
        return _extract_docx(raw)
    if name.endswith(".txt") or name.endswith(".md"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".doc"):
        raise DraftExtractionError(
            "Legacy .doc format is not supported. Please save as .docx and re-upload."
        )
    raise DraftExtractionError(
        "Unsupported file format. Please upload PDF, DOCX, or plain text."
    )


def _extract_pdf(raw: bytes) -> str:
    text = ""
    try:
        import fitz  # PyMuPDF — best text ordering when it is installed
        with fitz.open(stream=raw, filetype="pdf") as doc:
            text = "\n\n".join((page.get_text() or "") for page in doc).strip()
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 — fall back to PyPDF2 below
        text = ""
    if not text:
        from PyPDF2 import PdfReader

        reader = PdfReader(BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
    text = _rejoin_word_per_line(text)
    if not text:
        raise DraftExtractionError(
            "No readable text was found in the PDF (it may be a scanned image)."
        )
    return text


def _rejoin_word_per_line(text: str, threshold: float = 0.8) -> str:
    """Repair PDFs whose text extracts as one word per line.

    Word/Google-Docs "Save as PDF" output often reaches PyPDF2 as a separate
    text run per word, which extract_text() renders as ``word\\n \\nword``.
    Citations then never occur on one line ("503 Pa. 178" becomes three
    lines) and eyecite finds nothing. When most non-blank lines are single
    tokens, rejoin tokens with spaces and keep only the longer blank runs
    (which is how paragraph breaks survive in such files) as paragraphs.
    """
    if not text:
        return text
    lines = text.split("\n")
    nonblank = [ln for ln in lines if ln.strip()]
    if len(nonblank) < 40:
        return text
    single = sum(1 for ln in nonblank if len(ln.split()) == 1)
    if single / len(nonblank) < threshold:
        return text
    paragraphs, current, blank_run = [], [], 0
    for ln in lines:
        if ln.strip():
            if blank_run >= 3 and current:
                paragraphs.append(" ".join(current))
                current = []
            current.append(ln.strip())
            blank_run = 0
        else:
            blank_run += 1
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _extract_docx(raw: bytes) -> str:
    """Accepted-changes view of the document, tables included.

    ``paragraph.text`` in python-docx only sees runs that are direct children
    of the paragraph, so text a student INSERTED with Track Changes on
    (``w:ins``) silently vanished and deleted text (``w:delText``) is not
    part of ``w:t`` anyway. A redlined submission therefore read as the
    untouched original. This walks the XML instead and returns what the
    document says with all tracked changes accepted.
    """
    paragraphs, _ = _docx_paragraphs(raw, mode="final")
    text = "\n\n".join(paragraphs).strip()
    if not text:
        raise DraftExtractionError("The DOCX appears to contain no readable text.")
    return text


def extract_docx_redline(raw: bytes, include_notes: bool = False) -> dict:
    """Tracked-changes views of a DOCX for reviewing a redlined submission.

    Returns ``{"final", "markup", "original", "insertions", "deletions"}``:
    ``final`` = all changes accepted (what ``extract_draft_text`` returns),
    ``markup`` = the same text with ``[INSERTED: …]`` / ``[DELETED: …]``
    markers so a reviewer (or model) can see exactly what the author changed,
    ``original`` = all changes rejected. The counts are tracked-change runs.
    ``include_notes`` appends footnotes and endnotes (see ``_docx_paragraphs``).
    """
    final, stats = _docx_paragraphs(raw, mode="final", include_notes=include_notes)
    markup, _ = _docx_paragraphs(raw, mode="markup", include_notes=include_notes)
    original, _ = _docx_paragraphs(raw, mode="original", include_notes=include_notes)
    return {
        "final": "\n\n".join(final).strip(),
        "markup": "\n\n".join(markup).strip(),
        "original": "\n\n".join(original).strip(),
        "insertions": stats["insertions"],
        "deletions": stats["deletions"],
    }


# Footnote / endnote parts, by relationship type suffix → marker prefix.
_NOTE_PARTS = (("/footnotes", "footnote", ""), ("/endnotes", "endnote", "e"))
_NOTE_SKIP_TYPES = {"separator", "continuationSeparator", "continuationNotice"}


def _docx_paragraphs(raw: bytes, mode: str = "final", include_notes: bool = False):
    """Paragraph texts for every ``w:p`` in the body (tables included) under
    one of three tracked-change views: ``final`` (accept all), ``original``
    (reject all) or ``markup`` (both, with markers).

    The body is all python-docx walks, so footnotes and endnotes (where
    Bluebook and Chicago-notes citations live) are not part of it. With
    ``include_notes`` a reference in the body becomes ``[^n]`` (endnotes
    ``[^en]``) and, after the body, a ``Notes`` line is followed by one
    ``[^n] <note text>`` paragraph per note, under the same view. Off by
    default: production extraction is unchanged."""
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(BytesIO(raw))
    W_P, W_T, W_DELTEXT, W_INS = qn("w:p"), qn("w:t"), qn("w:delText"), qn("w:ins")
    W_TAB, W_BR = qn("w:tab"), qn("w:br")
    W_FNREF, W_ENREF, W_ID = qn("w:footnoteReference"), qn("w:endnoteReference"), qn("w:id")
    stats = {"insertions": 0, "deletions": 0}

    def para_text(p) -> str:
        pieces: list[tuple[str, str]] = []   # (kind, text)
        for el in p.iter():
            if include_notes and el.tag in (W_FNREF, W_ENREF):
                prefix = "e" if el.tag == W_ENREF else ""
                pieces.append(("text", f"[^{prefix}{el.get(W_ID)}]"))
            elif el.tag == W_T:
                inserted = any(a.tag == W_INS for a in el.iterancestors())
                if inserted:
                    stats["insertions"] += 1
                    if mode == "original":
                        continue
                pieces.append(("ins" if inserted and mode == "markup" else "text",
                               el.text or ""))
            elif el.tag == W_DELTEXT:
                stats["deletions"] += 1
                if mode == "final":
                    continue
                pieces.append(("del" if mode == "markup" else "text",
                               el.text or ""))
            elif el.tag == W_TAB:
                pieces.append(("text", "\t"))
            elif el.tag == W_BR:
                pieces.append(("text", "\n"))
        # merge adjacent runs of the same kind so markers wrap whole phrases
        merged: list[list[str]] = []
        for kind, txt in pieces:
            if merged and merged[-1][0] == kind:
                merged[-1][1] += txt
            else:
                merged.append([kind, txt])
        out = []
        for kind, txt in merged:
            if kind == "ins":
                out.append(f"[INSERTED: {txt}]")
            elif kind == "del":
                out.append(f"[DELETED: {txt}]")
            else:
                out.append(txt)
        return "".join(out)

    paragraphs = []
    for p in doc.element.body.iter(W_P):
        text = para_text(p)
        if text.strip():
            paragraphs.append(text)
    if include_notes:
        notes = []
        for suffix, tag, prefix in _NOTE_PARTS:
            for note in _docx_note_elements(doc, suffix, qn(f"w:{tag}")):
                if note.get(qn("w:type")) in _NOTE_SKIP_TYPES:
                    continue
                body = " ".join(t.strip() for t in map(para_text, note.iter(W_P))
                                if t.strip())
                if body:
                    notes.append(f"[^{prefix}{note.get(W_ID)}] {body}")
        if notes:
            paragraphs += ["Notes"] + notes
    return paragraphs, stats


def _docx_note_elements(doc, reltype_suffix: str, note_tag: str) -> list:
    """The ``w:footnote`` / ``w:endnote`` elements of a document's notes part
    (empty when the document has none)."""
    from lxml import etree
    for rel in doc.part.rels.values():
        if rel.is_external or not rel.reltype.endswith(reltype_suffix):
            continue
        part = rel.target_part
        root = getattr(part, "element", None)
        if root is None:
            root = etree.fromstring(part.blob)
        return list(root.iter(note_tag))
    return []
