"""
Readings context for the first read: what the student could have read.

In ProfFaith this block is gathered from the course database (the module's
textbook sections, finished readings and lecture outlines). In this snapshot
the instructor lists the readings in the assignment pack instead::

    "readings": [
      {"title": "Week 2 notes", "file": "readings/week2.md"},
      {"title": "Rule statement handout", "text": "..."}
    ]

``readings_to_module_context`` turns that list into the same
``pack["module_context"]`` shape the prompts read, trimming each reading to
a budget (head and tail kept, never head-only truncation). Give the model
student-facing material only, never an answer key.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_BUDGET = 32000          # chars across all readings (~8k tokens)
MIN_ITEM = 3000                 # never shrink one reading below this


def _trim(text: str, share: int) -> str:
    """Keep the head and tail of an over-budget reading (the parts that
    state and sum up the rule), marking the gap."""
    if len(text) <= share:
        return text
    half = max(share // 2 - 40, 200)
    return text[:half].rstrip() + "\n\n[... middle of reading omitted ...]\n\n" + text[-half:].lstrip()


def readings_to_module_context(pack: dict, pack_dir=None,
                               budget_chars: int = DEFAULT_BUDGET) -> Optional[dict]:
    """``pack["readings"]`` → the ``module_context`` dict, or None when the
    pack lists no readings. ``file`` paths are relative to the pack's folder."""
    readings = pack.get("readings") or []
    items = []
    for n, r in enumerate(readings, 1):
        if isinstance(r, str):
            r = {"title": f"Reading {n}", "text": r}
        text = r.get("text") or ""
        if not text and r.get("file"):
            path = Path(r["file"])
            if not path.is_absolute() and pack_dir is not None:
                path = Path(pack_dir) / path
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                log.warning("reading not found: %s", path)
                continue
        if text.strip():
            items.append({"ref": f"r{n}", "kind": r.get("kind", "finished_material"),
                          "title": r.get("title") or f"Reading {n}", "text": text.strip(),
                          **({"digest": r["digest"]} if r.get("digest") else {})})
    if not items:
        return None
    total = sum(len(i["text"]) for i in items)
    out = []
    for i in items:
        share = (len(i["text"]) if total <= budget_chars
                 else max(MIN_ITEM, int(budget_chars * len(i["text"]) / total)))
        text = _trim(i["text"], share)
        out.append({**i, "text": text, "chars": len(text),
                    "full_chars": len(i["text"]), "excerpted": len(text) < len(i["text"])})
    return {"module": {"module_title": pack.get("module_title", "")} if pack.get("module_title") else {},
            "items": out, "budget_chars": budget_chars,
            "total_chars": sum(i["chars"] for i in out),
            "full_chars": sum(i["full_chars"] for i in out)}


KIND_LABELS = {"textbook_section": "textbook section", "finished_material": "reading",
               "lecture": "lecture outline"}


def module_context_text(pack: dict, max_chars: Optional[int] = None) -> str:
    """Prompt block for grading: what the student could have read. Uses an
    item's digest when the instructor made one, else its (excerpted) text."""
    mc = (pack or {}).get("module_context") or {}
    items = mc.get("items") or []
    if not items:
        return ""
    mod = mc.get("module") or {}
    head = "WHAT THE STUDENT COULD HAVE READ"
    if mod.get("module_title"):
        num = f"Module {mod['module_number']}: " if mod.get("module_number") else ""
        head += (f" — {num}{mod['module_title']}"
                 + (f" ({mod['weeks']})" if mod.get("weeks") else ""))
    lines = [head + " — the assigned, student-facing materials for this "
             "assignment. Judge the submission against these: reward correct use "
             "of the frameworks, rules and cases presented here; note when the "
             "student contradicts or ignores them; do not penalize what these "
             "materials did not cover."]
    for i in items:
        body = i.get("digest") or i.get("text") or ""
        tag = "digest" if i.get("digest") else ("excerpts" if i.get("excerpted") else "full text")
        lines.append(f"\n### {KIND_LABELS.get(i.get('kind'), i.get('kind'))}: {i.get('title')} ({tag})\n{body}")
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n[... module context truncated ...]"
    return text


DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "heading": {"type": "string"},
                "bullets": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["heading", "bullets"],
        }},
    },
    "required": ["sections"],
}

# A digest shorter than this share of its source is a failed reduction
# (observed: gemma returning the assignment title as the whole digest).
MIN_DIGEST_RATIO = 0.08
MIN_DIGEST_CHARS = 600


def digest_prompt(item: dict, assignment_title: str = "") -> str:
    return "\n".join([
        "Reduce this course reading to the points a student would need in order "
        f"to complete the assignment \u201c{assignment_title or 'this assignment'}\u201d. "
        "Return sections, each with a short heading and 3-8 bullets: the rules "
        "and tests (with their elements), key definitions, the cases or "
        "authorities named and what each stands for, worked examples' takeaways, "
        "and the reading's warnings or pitfalls. Keep the reading's own "
        "terminology and be specific — a bullet must carry the actual rule or "
        "fact, not a topic label. Aim for roughly a quarter of the original "
        "length overall.",
        "", f"READING: {item.get('title', '')}", "", item.get("text", "")])


def digest_to_markdown(data: dict) -> str:
    out = []
    for sec in (data or {}).get("sections") or []:
        h = (sec.get("heading") or "").strip()
        bullets = [b.strip() for b in (sec.get("bullets") or []) if b and b.strip()]
        if not bullets:
            continue
        if h:
            out.append(f"## {h}")
        out += [f"- {b}" for b in bullets]
        out.append("")
    return "\n".join(out).strip()


def digest_acceptable(digest: str, source: str) -> bool:
    d, s = len(digest or ""), len(source or "")
    return d >= min(MIN_DIGEST_CHARS, s // 2) and d >= s * MIN_DIGEST_RATIO
