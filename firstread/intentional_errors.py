"""Helpers for the `intentional_errors` (find-the-flaw) activity type.

An Intentional Errors assignment hands the student a deliberately DEFECTIVE
document — a legal memo, a court filing/pleading, or an accessibility-defective
document — and asks them to (1) find the planted errors, (2) explain what is
wrong and how they would fix each, and (3) submit a REDLINED version of the
document with their proposed changes.

Three artifacts come out of generation (built in
celery_tasks/instructor_assignment_tasks.py):

  1. A student instructions handout (the normal TILT DOCX) — find / diagnose /
     fix / redline. Never contains the answer key.
  2. The defective document itself, rendered by `render_defective_document_docx`
     and attached as an InstructorAssignmentFile (so it exports to the LMS as a
     downloadable handout). For the accessibility document type the renderer
     injects REAL, Word-Accessibility-Checker-detectable defects (untagged
     headings, alt-less images, header-less tables, low-contrast text, vague
     link text) rather than merely describing them.
  3. An instructor-only answer key, rendered by `render_answer_key_docx`, stored
     at an instructor-only S3 key (NOT an InstructorAssignmentFile, so it never
     flows into an LMS export).

The structured `intentional_errors` block the LLM produces (and that we persist
in InstructorAssignmentIntentionalErrors.errors_json) looks like::

    {
      "document_type": "legal_memo" | "pleading" | "accessibility",
      "document_title": "Memorandum re: ...",
      "document_blocks": [
        {
          "block_type": "title|heading|subheading|paragraph|caption|signature|list_item|table|image|link",
          "text": "rendered text (with the planted error present, if substantive)",
          "error_id": "E1" | null,            # set when this block carries a planted error
          # accessibility document_type only:
          "accessibility_defect": "untagged_heading|missing_alt_text|no_table_header|low_contrast|vague_link_text" | null,
          "table_rows": [["Col A","Col B"], ["1","2"]],   # block_type == "table"
          "image_caption": "Figure 1: ...",               # block_type == "image"
          "href": "https://..."                            # block_type == "link"
        }
      ],
      "planted_errors": [                       # the ANSWER KEY — instructor only
        {"id":"E1","category":"...","severity":"high|medium|low",
         "locator":"...","whats_wrong":"...","why_it_matters":"...","correct_fix":"..."}
      ],
      "answer_key_summary": "one or two sentences orienting the grader"
    }
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.opc.constants import RELATIONSHIP_TYPE as RT

log = logging.getLogger(__name__)


# ── Accessibility defect catalogue ─────────────────────────────────────────
# Keys are the enum values the LLM may put in block.accessibility_defect and
# that render_defective_document_docx() dispatches on to inject a REAL defect.
# Values describe the WCAG issue (used in prompt guidance + as a grader hint).
ACCESSIBILITY_DEFECTS: Dict[str, Dict[str, str]] = {
    "untagged_heading": {
        "wcag": "1.3.1 Info and Relationships",
        "desc": "A line that looks like a heading (bold/large) but is NOT styled "
                "with a real Heading style, so assistive tech can't navigate by heading.",
        "fix": "Apply a true Heading 1/2/3 style instead of manual bold + large font.",
    },
    "missing_alt_text": {
        "wcag": "1.1.1 Non-text Content",
        "desc": "An image with no alternative (alt) text, invisible to screen-reader users.",
        "fix": "Add concise, meaningful alt text describing the image's purpose.",
    },
    "no_table_header": {
        "wcag": "1.3.1 Info and Relationships",
        "desc": "A data table with no designated header row, so screen readers "
                "can't associate data cells with their column headers.",
        "fix": "Mark the first row as a header row (Table Tools → 'Repeat Header Row').",
    },
    "low_contrast": {
        "wcag": "1.4.3 Contrast (Minimum)",
        "desc": "Body text whose contrast against the background is below the "
                "WCAG minimum of 4.5:1 (here rendered in light gray).",
        "fix": "Use a dark text color (e.g., near-black) that meets 4.5:1 contrast.",
    },
    "vague_link_text": {
        "wcag": "2.4.4 Link Purpose (In Context)",
        "desc": "A hyperlink labeled 'click here' / 'read more' that conveys no "
                "purpose when read out of context.",
        "fix": "Rewrite the link text to describe its destination (e.g., 'Maryland "
               "Rules of Professional Conduct').",
    },
}


# ── Document-type registry ──────────────────────────────────────────────────
# Each entry drives (a) the form selector + hint, (b) the prompt's error
# taxonomy, and (c) how the student task is framed. The accessibility entry
# additionally lists which structural defects the renderer can inject.
DOCUMENT_TYPES: Dict[str, Dict[str, Any]] = {
    "legal_memo": {
        "label": "Legal memo (IRAC/CREAC)",
        "blurb": "An office legal memorandum with planted analytical and citation defects.",
        "structure": (
            "Build a standard office legal memorandum with these blocks, in order: "
            "a heading block (TO / FROM / DATE / RE), QUESTION PRESENTED, BRIEF ANSWER, "
            "STATEMENT OF FACTS, DISCUSSION (organized as IRAC/CREAC), and CONCLUSION. "
            "Use 'heading' blocks for the section names and 'paragraph' blocks for prose. "
            "Write a realistic, full-length memo — roughly 600-900 words across the sections "
            "— so there is real substance to review and enough surface for the planted errors. "
            "Ground the facts and legal issue in the supplied sources / learning objective."
        ),
        "error_categories": [
            "Misstated or overbroad statement of the governing rule",
            "Conclusory analysis that never applies the rule to the facts",
            "Missing counter-analysis / failure to address the other side",
            "Citation/authority errors that are DETECTABLY wrong on inspection — a malformed "
            "citation (wrong reporter abbreviation, missing pincite, improper short form, "
            "wrong signal) OR a source cited for a proposition it does not support. Keep cited "
            "cases hypothetical (do not impersonate real cases), but make each defect "
            "unambiguous to a student who knows the rules — NOT plausibly correct",
            "Omitted element of a multi-part legal test",
            "Wrong or unstated standard of review / burden of proof",
            "Unsupported factual assertion (a fact asserted in the Discussion that never appears "
            "in the Statement of Facts)",
            "Internal inconsistency (Brief Answer or Conclusion contradicts the analysis)",
            "Brief Answer that does not actually answer the Question Presented",
        ],
        "min_errors": 7,
        "is_accessibility": False,
    },
    "pleading": {
        "label": "Court filing / pleading",
        "blurb": "A pleading (e.g., a complaint) with planted procedural and formal defects.",
        "structure": (
            "Build a court filing (default: a civil COMPLAINT) with these blocks, in order: "
            "a 'caption' block (court name, parties, case number), a 'title' block naming the "
            "pleading, then 'heading'/'paragraph' blocks for JURISDICTION AND VENUE, PARTIES, "
            "FACTUAL ALLEGATIONS (numbered), one or more COUNTS / CAUSES OF ACTION (each "
            "stating its elements), a PRAYER FOR RELIEF, a 'signature' block (attorney name, "
            "bar number, address), and a CERTIFICATE OF SERVICE. Write a realistic, full-length "
            "filing with several numbered factual allegations and at least one full count, so "
            "there is real substance to review. Ground the dispute in the supplied sources / "
            "learning objective."
        ),
        "error_categories": [
            "Defective or incomplete caption (missing court, party, or case number)",
            "Missing or inadequate basis for subject-matter jurisdiction",
            "Improper or unstated venue",
            "A count that omits a required element of the cause of action",
            "Prayer for relief missing, or relief that does not match the counts pleaded",
            "Missing certificate of service",
            "Missing/defective signature block (no bar number, no address, unsigned)",
            "Numbered-paragraph defects (skips, duplicates, allegations not numbered)",
            "Missing verification where the pleading requires one",
        ],
        "min_errors": 7,
        "is_accessibility": False,
    },
    "accessibility": {
        "label": "Accessibility (ADA/WCAG) document",
        "blurb": "A normal-looking document carrying REAL, checker-detectable accessibility defects.",
        "structure": (
            "Build an ordinary informational document on the learning-objective topic "
            "(e.g., a client-facing handout, an office policy memo, or a procedures guide). "
            "It should read as a real document, but you MUST plant accessibility defects using "
            "the 'accessibility_defect' field on the affected blocks. Include AT LEAST ONE block "
            "of EACH of these defect types: untagged_heading, missing_alt_text, no_table_header, "
            "low_contrast, vague_link_text. For an 'image' block set image_caption; for a 'table' "
            "block set table_rows (first row = intended column headers); for a 'link' block set "
            "href. Also include some CORRECT content (properly styled headings, normal paragraphs) "
            "so the document is realistic and the defects must actually be found."
        ),
        "error_categories": [
            f"{k} — {v['desc']} (WCAG {v['wcag']})" for k, v in ACCESSIBILITY_DEFECTS.items()
        ],
        "min_errors": 5,
        "is_accessibility": True,
    },
}

DOCUMENT_TYPE_LABELS = {k: v["label"] for k, v in DOCUMENT_TYPES.items()}
DEFAULT_DOCUMENT_TYPE = "legal_memo"


def normalize_document_type(value: Optional[str]) -> str:
    """Coerce a form/payload value to a known document type (default legal_memo)."""
    v = (value or "").strip().lower()
    return v if v in DOCUMENT_TYPES else DEFAULT_DOCUMENT_TYPE


# ── Prompt guidance ─────────────────────────────────────────────────────────
def build_document_type_guidance(document_type: str) -> str:
    """Return the doc-type-specific guidance block injected into the system prompt."""
    dt = DOCUMENT_TYPES.get(normalize_document_type(document_type))
    lines = [
        f"DOCUMENT TYPE: {dt['label']}",
        dt["structure"],
        "",
        "PLANT ERRORS FROM THESE CATEGORIES (mix several; vary severity):",
    ]
    lines += [f"  - {c}" for c in dt["error_categories"]]
    lines += [
        "",
        f"Plant AT LEAST {dt['min_errors']} distinct, pedagogically meaningful errors. Each "
        "planted error MUST appear in the actual document_blocks (the student has to be able "
        "to find it by reading the document) AND be listed in planted_errors with a matching "
        "error id. Do NOT label or hint at the errors inside the document itself — the document "
        "must look like a real, finished work product.",
        "",
        "Spread the difficulty: include some errors a careful reader catches on first read and "
        "some that require domain knowledge. Every planted error must be a genuine, defensible "
        "defect a reviewer could mark up — never something a competent professional would judge "
        "correct. Avoid 'errors' that are merely stylistic preferences.",
    ]
    if dt["is_accessibility"]:
        lines += [
            "",
            "ACCESSIBILITY DEFECTS — set the block's accessibility_defect to one of: "
            + ", ".join(ACCESSIBILITY_DEFECTS.keys()) + ".",
            "Each planted_errors entry's category should name the WCAG criterion (e.g., "
            "'WCAG 1.1.1 Non-text Content').",
        ]
    return "\n".join(lines)


def build_intentional_errors_schema_block() -> str:
    """Return the `intentional_errors` JSON sub-schema appended to the output schema."""
    return """
  "intentional_errors": {
    "document_type": "legal_memo | pleading | accessibility (echo the requested type)",
    "document_title": "string (the document's own title/RE line)",
    "document_blocks": [
      {
        "block_type": "title|heading|subheading|paragraph|caption|signature|list_item|table|image|link",
        "text": "string (the rendered content WITH any planted error present)",
        "error_id": "string id (e.g. 'E1') if this block carries a planted error, else null",
        "accessibility_defect": "one of untagged_heading|missing_alt_text|no_table_header|low_contrast|vague_link_text (accessibility document_type only), else null",
        "table_rows": [["header1","header2"],["r1c1","r1c2"]],
        "image_caption": "string (block_type 'image' only)",
        "href": "string URL (block_type 'link' only)"
      }
    ],
    "planted_errors": [
      {
        "id": "string (matches an error_id used in document_blocks)",
        "category": "string (the error type / WCAG criterion)",
        "severity": "high|medium|low",
        "locator": "string (quote the offending text or describe where to find it)",
        "whats_wrong": "string (what is defective)",
        "why_it_matters": "string (the professional / legal / accessibility consequence)",
        "correct_fix": "string (the specific correction the student should make)"
      }
    ],
    "answer_key_summary": "string (1-2 sentences orienting the grader)"
  },"""


# ── Student-facing task text ────────────────────────────────────────────────
def render_student_task_text(errors_block: Dict[str, Any]) -> str:
    """Build the canonical student instructions (find / diagnose / fix / redline).

    Deliberately does NOT reveal the planted errors or their count — the student
    must find them. The defective document is delivered as an attached file.
    """
    document_type = normalize_document_type(errors_block.get("document_type"))
    dt = DOCUMENT_TYPES[document_type]
    title = (errors_block.get("document_title") or dt["label"]).strip()

    lines = [
        f"You have been given a defective {dt['label']}: “{title}”. It is attached to "
        "this assignment as a downloadable file, and it contains a number of deliberately "
        "planted errors. Your job is to act as the reviewing professional: find the errors, "
        "explain them, and fix them.",
        "",
        "First: Read the attached document closely and identify every error you can find. "
        "Examine both the substance (is it correct, complete, and well-reasoned?) and the form "
        "(structure, formatting, required elements).",
        "Next: For EACH error, (a) quote it or pinpoint where it appears, (b) explain what is "
        "wrong and why it matters, and (c) state the specific correction you would make.",
        "Then: Produce a REDLINED version of the document. Open the attached file in your word "
        "processor, turn on Track Changes (Word) or Suggesting mode (Google Docs), and insert "
        "your corrections directly into the document so your edits are visible as markup.",
        "Finally: Submit (1) your redlined document and (2) a short memo listing each error "
        "you found, your diagnosis, and your proposed fix.",
    ]
    if dt["is_accessibility"]:
        lines += [
            "",
            "Tip: this document has accessibility defects. Run your word processor's "
            "Accessibility Checker (in Word: Review → Check Accessibility) to help locate "
            "issues, then remediate each one and explain the barrier it created for users of "
            "assistive technology.",
        ]
    return "\n".join(lines)


# ── DOCX low-level helpers ──────────────────────────────────────────────────
def _add_hyperlink(paragraph, url: str, text: str, *, color: str = "0563C1") -> None:
    """Append a real (clickable) external hyperlink run to a paragraph.

    Using a true w:hyperlink (not just blue text) is what makes Word's
    Accessibility Checker treat vague link text as a finding.
    """
    part = paragraph.part
    r_id = part.relate_to(url or "https://example.com", RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), color)
    rPr.append(c)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)
    new_run.append(rPr)

    t = OxmlElement("w:t")
    t.text = text or url or "link"
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def _set_table_header_row(row) -> None:
    """Designate a table row as a (repeating) header row — the OOXML signal the
    Accessibility Checker reads as 'this table has headers'."""
    tr = row._tr
    trPr = tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)


def _set_picture_alt_text(picture, alt: str) -> None:
    """Set alt text (descr) on an inline picture's docPr element."""
    try:
        docPr = picture._inline.docPr
        docPr.set("descr", alt or "")
        docPr.set("title", alt or "")
    except Exception:
        log.debug("could not set picture alt text", exc_info=True)


def _make_placeholder_png(caption: str, w: int = 520, h: int = 240) -> Optional[bytes]:
    """Generate a simple placeholder image so the document has a real inline
    picture (the missing-alt-text defect needs an actual image). Returns None if
    Pillow is unavailable, in which case the caller renders a text fallback."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    img = Image.new("RGB", (w, h), (216, 226, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, w - 4, h - 4], outline=(78, 110, 165), width=3)
    label = (caption or "Figure").strip()
    if len(label) > 60:
        label = label[:57] + "..."
    d.text((24, h // 2 - 8), label, fill=(40, 60, 100))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Defective-document renderer ─────────────────────────────────────────────
def render_defective_document_docx(errors_block: Dict[str, Any]) -> bytes:
    """Render the student-facing DEFECTIVE document from document_blocks.

    Substantive errors (wrong rule, bad citation, missing element) live in the
    block text and render as ordinary styled content. For the accessibility
    document type, blocks flagged with accessibility_defect are rendered so the
    defect is genuinely present in the file (Word's Accessibility Checker will
    flag it).
    """
    document_type = normalize_document_type(errors_block.get("document_type"))
    blocks: List[Dict[str, Any]] = errors_block.get("document_blocks") or []

    doc = Document()
    title = (errors_block.get("document_title") or "").strip()
    if title:
        doc.add_heading(title, level=0)

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = (block.get("block_type") or "paragraph").strip().lower()
        text = (block.get("text") or "").strip()
        defect = (block.get("accessibility_defect") or "").strip().lower() or None

        # ── Accessibility defects: inject the real structural problem ──
        if defect == "untagged_heading":
            # Looks like a heading; is a plain bold/large paragraph (no Heading style).
            p = doc.add_paragraph()
            run = p.add_run(text or "Section")
            run.bold = True
            run.font.size = Pt(16)
            continue
        if defect == "low_contrast":
            p = doc.add_paragraph()
            run = p.add_run(text)
            run.font.color.rgb = RGBColor(0xBC, 0xBC, 0xBC)  # light gray on white
            continue
        if defect == "vague_link_text":
            p = doc.add_paragraph()
            if text:
                p.add_run(text + " ")
            _add_hyperlink(p, block.get("href") or "https://example.com",
                           "click here")
            continue
        if defect == "missing_alt_text" or btype == "image":
            caption = block.get("image_caption") or text or "Figure"
            png = _make_placeholder_png(caption)
            if png:
                pic = doc.add_picture(BytesIO(png), width=Inches(4.5))
                # Defect: leave alt text unset. Otherwise give it meaningful alt.
                if defect != "missing_alt_text":
                    _set_picture_alt_text(pic, caption)
            else:
                # No Pillow: fall back to a captioned line (still a11y-imperfect).
                doc.add_paragraph(f"[Image: {caption}]")
            continue
        if defect == "no_table_header" or btype == "table":
            rows = block.get("table_rows") or []
            rows = [r for r in rows if isinstance(r, list) and r]
            if rows:
                ncols = max(len(r) for r in rows)
                t = doc.add_table(rows=len(rows), cols=ncols)
                try:
                    t.style = "Table Grid"
                except Exception:
                    pass
                for ri, row in enumerate(rows):
                    for ci in range(ncols):
                        t.cell(ri, ci).text = str(row[ci]) if ci < len(row) else ""
                # Defect: do NOT designate a header row. Otherwise tag row 0.
                if defect != "no_table_header":
                    _set_table_header_row(t.rows[0])
                    for cell in t.rows[0].cells:
                        for par in cell.paragraphs:
                            for r in par.runs:
                                r.bold = True
            elif text:
                doc.add_paragraph(text)
            continue

        # ── Ordinary (correctly-structured) blocks ──
        if not text and btype not in ("title", "heading", "subheading"):
            continue
        if btype == "title":
            doc.add_heading(text, level=0)
        elif btype == "heading":
            doc.add_heading(text, level=1)
        elif btype == "subheading":
            doc.add_heading(text, level=2)
        elif btype == "caption":
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    p = doc.add_paragraph(ln)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif btype == "signature":
            for ln in text.splitlines():
                if ln.strip():
                    doc.add_paragraph(ln)
        elif btype == "list_item":
            doc.add_paragraph(text, style="List Bullet")
        elif btype == "link":
            p = doc.add_paragraph()
            _add_hyperlink(p, block.get("href") or "https://example.com", text or "link")
        else:  # paragraph + any unknown type
            doc.add_paragraph(text)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── Answer-key renderer (instructor only) ───────────────────────────────────
_SEVERITY_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}


def render_answer_key_docx(errors_block: Dict[str, Any], assignment_title: str = "") -> bytes:
    """Render the INSTRUCTOR-ONLY answer key for grading. Never attach this to
    the assignment as a student file and never export it to an LMS."""
    document_type = normalize_document_type(errors_block.get("document_type"))
    dt = DOCUMENT_TYPES[document_type]
    planted: List[Dict[str, Any]] = errors_block.get("planted_errors") or []

    doc = Document()
    doc.add_heading("Answer Key — Instructor Only", level=0)

    banner = doc.add_paragraph()
    run = banner.add_run("DO NOT DISTRIBUTE TO STUDENTS")
    run.bold = True
    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    subtitle = assignment_title or errors_block.get("document_title") or dt["label"]
    doc.add_paragraph(f"Assignment: {subtitle}")
    doc.add_paragraph(f"Document type: {dt['label']}")

    summary = (errors_block.get("answer_key_summary") or "").strip()
    if summary:
        doc.add_paragraph(summary)

    doc.add_paragraph(f"Planted errors: {len(planted)}")

    for i, e in enumerate(planted, 1):
        if not isinstance(e, dict):
            continue
        eid = (e.get("id") or f"E{i}").strip()
        category = (e.get("category") or "").strip()
        severity = _SEVERITY_LABEL.get((e.get("severity") or "").strip().lower(), "")
        head = f"{eid} — {category}" if category else eid
        if severity:
            head += f"  [{severity}]"
        doc.add_heading(head, level=2)

        def _kv(label: str, value: str):
            value = (value or "").strip()
            if not value:
                return
            p = doc.add_paragraph()
            r = p.add_run(f"{label}: ")
            r.bold = True
            p.add_run(value)

        _kv("Where", e.get("locator"))
        _kv("What's wrong", e.get("whats_wrong"))
        _kv("Why it matters", e.get("why_it_matters"))
        _kv("Correct fix", e.get("correct_fix"))

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── Validation ──────────────────────────────────────────────────────────────
def validate_intentional_errors_block(errors_block: Any) -> List[str]:
    """Return a list of human-readable problems with an intentional_errors block.
    Empty list == valid. Used by validate_assignments and by normalization."""
    problems: List[str] = []
    if not isinstance(errors_block, dict):
        return ["intentional_errors must be an object"]

    if not str(errors_block.get("document_title") or "").strip():
        problems.append("intentional_errors.document_title is required")

    blocks = errors_block.get("document_blocks")
    if not isinstance(blocks, list) or not blocks:
        problems.append("intentional_errors.document_blocks must be a non-empty list")

    planted = errors_block.get("planted_errors")
    if not isinstance(planted, list) or len(planted) < 3:
        problems.append("intentional_errors.planted_errors must list at least 3 errors")
    else:
        for i, e in enumerate(planted):
            if not isinstance(e, dict):
                problems.append(f"planted_errors[{i}] must be an object")
                continue
            for k in ("id", "whats_wrong", "correct_fix"):
                if not str(e.get(k) or "").strip():
                    problems.append(f"planted_errors[{i}].{k} is required")

    document_type = normalize_document_type(errors_block.get("document_type"))
    if DOCUMENT_TYPES[document_type]["is_accessibility"] and isinstance(blocks, list):
        present = {(_b.get("accessibility_defect") or "").strip().lower()
                   for _b in blocks if isinstance(_b, dict)}
        if not (present & set(ACCESSIBILITY_DEFECTS.keys())):
            problems.append(
                "accessibility document must flag at least one block with an "
                "accessibility_defect"
            )
    return problems


# ── Answer-key detection check (grading side) ───────────────────────────────
#
# The generation side above produces the defective document and its answer
# key. These helpers are the grading side: given a student's submission (the
# redlined document and/or the cover memo) and the planted_errors answer
# key, ask a model which planted errors the student found and fixed. Pure
# functions — no model client here — so the same prompt can run against
# Anthropic in the LMS path or a localhost model in utils/local_eval.py.

DETECTION_STATUSES = [
    "fixed_and_explained",    # corrected in the document AND diagnosed
    "fixed_silently",         # corrected, but never explained/flagged
    "identified_not_fixed",   # flagged/explained, but the defect remains
    "missed",                 # neither flagged nor corrected
    "unclear",                # cannot tell from this submission
]
DETECTED_STATUSES = {"fixed_and_explained", "fixed_silently",
                     "identified_not_fixed"}

DETECTION_STATUS_LABELS = {
    "fixed_and_explained": "found, explained and fixed",
    "fixed_silently": "fixed in the document, not explained",
    "identified_not_fixed": "flagged/explained, but not actually fixed",
    "missed": "MISSED",
    "unclear": "unclear from this file",
}


def render_document_text(errors_block: Dict[str, Any]) -> str:
    """Plain text of the defective document from ``document_blocks`` (the
    original the student was asked to redline)."""
    out = []
    for b in (errors_block or {}).get("document_blocks") or []:
        t = (b or {}).get("text") or ""
        if not t.strip():
            continue
        if (b.get("block_type") or "") == "heading":
            out.append(t.strip().upper())
        else:
            out.append(t.strip())
    return "\n\n".join(out)


def answer_key_text(errors_block: Dict[str, Any]) -> str:
    lines = []
    for e in (errors_block or {}).get("planted_errors") or []:
        lines.append(f"[{e.get('id')}] severity={e.get('severity', '?')} — "
                     f"{e.get('category', '')}")
        if e.get("locator"):
            lines.append(f"    where: {e['locator']}")
        if e.get("whats_wrong"):
            lines.append(f"    what's wrong: {e['whats_wrong']}")
        if e.get("correct_fix"):
            lines.append(f"    correct fix: {e['correct_fix']}")
    return "\n".join(lines)


def build_detection_prompt(errors_block: Dict[str, Any], submission_text: str,
                           *, assignment_ctx: str = "",
                           original_text: str = "",
                           has_tracked_changes: bool = False,
                           max_submission_chars: int = 40000) -> str:
    """Prompt: which planted errors did this submission find / fix?

    ``submission_text`` should be the tracked-changes MARKUP view when the
    student redlined the document (``[INSERTED: …]`` / ``[DELETED: …]``), so
    the model can distinguish the student's edits from the defective
    original. A cover memo that only lists errors is also a valid submission
    — the statuses allow for "identified but not fixed".
    """
    title = (errors_block or {}).get("document_title") or "the document"
    n = len((errors_block or {}).get("planted_errors") or [])
    sub = (submission_text or "")[:max_submission_chars]
    parts = [
        "You are assisting an instructor grading a find-the-flaws exercise. "
        f"Students received a deliberately defective document (“{title}”) "
        f"containing {n} planted errors, and were asked to (1) identify each "
        "error, (2) explain what is wrong and why it matters, and (3) fix it, "
        "submitting a redlined document and a short cover memo.",
        "",
        "Below is the INSTRUCTOR'S ANSWER KEY (never shown to students), then "
        "ONE of the student's submitted files. That file may be the redlined "
        "document, the cover memo, or a combined document.",
    ]
    if assignment_ctx:
        parts += ["", "ASSIGNMENT CONTEXT:", assignment_ctx]
    parts += ["", "ANSWER KEY — PLANTED ERRORS:", answer_key_text(errors_block)]
    if original_text:
        parts += ["", "ORIGINAL DEFECTIVE DOCUMENT (as issued to the student):",
                  original_text[:20000]]
    if has_tracked_changes:
        parts += ["", "The student's file carried Track Changes. In the text "
                  "below, [INSERTED: …] marks text the student added and "
                  "[DELETED: …] marks text the student struck. Text outside "
                  "those markers is the original document, unchanged."]
    parts += ["", "STUDENT'S SUBMITTED FILE:", sub, ""]
    parts += [
        "For EACH planted error in the answer key decide, from THIS file only:",
        "- fixed_and_explained: the defect is corrected in the document AND the "
        "student explains what was wrong (in a comment, cover memo, or note).",
        "- fixed_silently: the defect is corrected but never explained.",
        "- identified_not_fixed: the student flags or explains the defect but "
        "the defective text still stands (or the fix does not cure it).",
        "- missed: the student neither flagged nor corrected it.",
        "- unclear: this file cannot show it (e.g. a cover memo cannot show "
        "whether the document was actually edited — if the memo explains the "
        "error and states the fix, use identified_not_fixed; if the redline "
        "shows a fix with no explanation anywhere, use fixed_silently).",
        "Quote the student's own words or edit as evidence (short). Be strict: "
        "restating the defect is not a fix; a fix that introduces a different "
        "error is not a cure — say so in the note.",
        "",
        "Also list ADDITIONAL issues the student raised that are NOT in the "
        "answer key, and judge each: genuine_issue (a real defect the key did "
        "not plant), not_an_error (the student was wrong), or unclear.",
        "",
        "Finish with a two-sentence summary for the instructor: how thorough "
        "the detection was, and whether the corrections are legally sound.",
    ]
    return "\n".join(parts)


def detection_schema(error_ids) -> Dict[str, Any]:
    """JSON schema for the detection result; error ids are enum-locked so a
    small model cannot invent or garble them."""
    ids = list(error_ids) or []
    id_schema = {"type": "string", "enum": ids} if ids else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "errors": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": id_schema,
                    "status": {"type": "string", "enum": DETECTION_STATUSES},
                    "evidence": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["id", "status", "evidence", "note"],
            }},
            "additional_issues": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "assessment": {"type": "string",
                                   "enum": ["genuine_issue", "not_an_error",
                                            "unclear"]},
                },
                "required": ["description", "assessment"],
            }},
            "summary": {"type": "string"},
        },
        "required": ["errors", "additional_issues", "summary"],
    }


def tidy_detection(result: Dict[str, Any],
                   errors_block: Dict[str, Any]) -> Dict[str, Any]:
    """Dedupe rows (first mention wins), restore answer-key order, attach
    severity/category from the key, and add an 'unclear' placeholder for any
    planted error the model skipped. Adds ``total`` / ``detected`` counts."""
    key = {e.get("id"): e for e in (errors_block or {}).get("planted_errors") or []}
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in (result or {}).get("errors") or []:
        rid = r.get("id")
        if rid in key and rid not in by_id:
            by_id[rid] = r
    rows = []
    for rid, e in key.items():
        r = by_id.get(rid) or {"id": rid, "status": "unclear",
                               "evidence": "",
                               "note": "model returned no row for this "
                                       "error — check manually"}
        if r.get("status") not in DETECTION_STATUSES:
            r["status"] = "unclear"
        rows.append({**r, "severity": e.get("severity", ""),
                     "category": e.get("category", ""),
                     "locator": e.get("locator", "")})
    detected = sum(1 for r in rows if r["status"] in DETECTED_STATUSES)
    high_total = sum(1 for r in rows if r["severity"] == "high")
    high_detected = sum(1 for r in rows if r["severity"] == "high"
                        and r["status"] in DETECTED_STATUSES)
    return {
        "errors": rows,
        "additional_issues": (result or {}).get("additional_issues") or [],
        "summary": (result or {}).get("summary") or "",
        "total": len(rows), "detected": detected,
        "high_total": high_total, "high_detected": high_detected,
    }


def detection_summary_text(det: Dict[str, Any]) -> str:
    """Compact block for a downstream grading prompt."""
    if not det:
        return ""
    lines = [f"ANSWER KEY CHECK: {det.get('detected', 0)} of "
             f"{det.get('total', 0)} planted errors detected "
             f"({det.get('high_detected', 0)}/{det.get('high_total', 0)} "
             "high-severity)."]
    for r in det.get("errors") or []:
        lines.append(f"- {r.get('id')} ({r.get('severity')}): "
                     f"{DETECTION_STATUS_LABELS.get(r.get('status'), r.get('status'))}"
                     + (f" — {r['note']}" if r.get("note") else ""))
    extra = [a for a in det.get("additional_issues") or []
             if a.get("assessment") == "genuine_issue"]
    if extra:
        lines.append(f"Student also raised {len(extra)} genuine issue(s) "
                     "beyond the key: " + "; ".join(
                         a.get("description", "")[:120] for a in extra))
    wrong = [a for a in det.get("additional_issues") or []
             if a.get("assessment") == "not_an_error"]
    if wrong:
        lines.append(f"Student flagged {len(wrong)} non-error(s): " + "; ".join(
            a.get("description", "")[:120] for a in wrong))
    if det.get("summary"):
        lines.append(det["summary"])
    return "\n".join(lines)
