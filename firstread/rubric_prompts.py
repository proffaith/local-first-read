"""
The rubric-read prompts.

Snapshot of the prompt builders ProfFaith's hosted writing evaluation uses
(``utils/moodle_writing_eval.py`` there). The local tool runs these same
prompts on a localhost model, so a local first read and the hosted one judge
against the same instructions. The Anthropic plumbing, the database lookups
and the HTML renderers that sit next to them in ProfFaith are left out.
"""
from __future__ import annotations

import re


def _truncate(text, max_chars=40000):
    if not text or len(text) <= max_chars:
        return text or ""
    half = max_chars // 2
    return text[:half] + "\n\n[... truncated for analysis ...]\n\n" + text[-half:]


def _plain(s):
    return re.sub(r"<[^>]+>", " ", s or "").strip()


def assignment_context(pack: dict) -> str:
    """The assignment's design context as a prompt block, built from a pack's
    plain fields (the same layout ProfFaith builds from its database row):

    ``title``, ``activity_type``, ``bloom_level``, ``points_possible``,
    ``purpose`` / ``task`` / ``criteria`` (TILT instructions), ``ai_policy``,
    ``objectives`` (list of strings) and ``sources`` (list of strings).
    """
    parts = [f"ASSIGNMENT: {pack.get('title') or 'Untitled assignment'}"]
    if pack.get("activity_type"):
        parts.append(f"Type: {pack['activity_type']}")
    if pack.get("bloom_level"):
        parts.append(f"Target cognitive level (Bloom): {pack['bloom_level']}")
    if pack.get("points_possible"):
        parts.append(f"Points possible: {pack['points_possible']}")

    tilt = []
    if pack.get("purpose"):
        tilt.append(f"Purpose: {_plain(pack['purpose'])}")
    if pack.get("task"):
        tilt.append(f"Task: {_plain(pack['task'])}")
    if pack.get("criteria"):
        tilt.append(f"Criteria for success: {_plain(pack['criteria'])}")
    if tilt:
        parts.append("INSTRUCTIONS (TILT):\n" + "\n".join(tilt))
    if pack.get("ai_policy"):
        parts.append(f"AI policy for this assignment: {_plain(pack['ai_policy'])}")

    aligns = [_plain(a) for a in (pack.get("objectives") or []) if a]
    if aligns:
        parts.append("ALIGNED OBJECTIVES:\n" + "\n".join(f"- {a}" for a in aligns))
    srcs = [_plain(s) for s in (pack.get("sources") or []) if s]
    if srcs:
        parts.append("ASSIGNED/RELATED SOURCES:\n" + "\n".join(f"- {x}" for x in srcs))
    return "\n\n".join(parts)


def _rubric_text(rubric_v2):
    lines = []
    for g in (rubric_v2 or {}).get("groups", []) or []:
        if g.get("name"):
            lines.append(f"[{g['name']}]")
        level_names = [l.get("name") for l in (g.get("levels") or []) if l.get("name")]
        for c in (g.get("criteria") or []):
            lines.append(f"- Criterion: {c.get('name')}")
            for ln in level_names:
                desc = (c.get("cells") or {}).get(ln, "")
                if desc:
                    lines.append(f"    {ln}: {desc}")
    return "\n".join(lines)


def _rubric_level_names(rubric_v2):
    names = []
    for g in (rubric_v2 or {}).get("groups", []) or []:
        for l in (g.get("levels") or []):
            if l.get("name") and l["name"] not in names:
                names.append(l["name"])
    return names


def writing_review_prompt(paper_text, assignment_ctx, source_summary,
                          rubric_text=""):
    """Stage-1 formative-review prompt. Shared with scripts/local_grade.py so
    local preliminary grading judges against the same prompt as production."""
    rubric_block = ""
    if rubric_text:
        rubric_block = (
            "\nGRADING RUBRIC — this is exactly what the instructor will assess. "
            "Anchor your formative feedback to these criteria so the student "
            "knows where they stand against the real grading standard. Do NOT "
            "assign scores or rubric levels (that is the instructor's job); only "
            "point out where the draft is on-track or needs work relative to each "
            "criterion:\n" + rubric_text + "\n")
    prompt = f"""You are an experienced legal-writing teaching assistant giving a student FORMATIVE feedback on a DRAFT, before their instructor grades it. Catch the fixable things — responsiveness to the prompt, organization, support for claims, and clarity/citation mechanics — so the student can revise. Do NOT grade the substance of the law; that is the instructor's job. Be specific and constructive: point to concrete passages and give actionable fixes.

{assignment_ctx}
{rubric_block}
SOURCE CHECK (already run on the cited authorities — treat as ground truth):
{source_summary}

STUDENT DRAFT:
{_truncate(paper_text)}

Return a JSON object:
{{
  "summary": "2-3 sentence formative overview",
  "dimensions": [
    {{"name": "Responsiveness to the prompt", "status": "strong|adequate|needs_work", "comment": "...", "suggestions": ["...", "..."]}},
    {{"name": "Organization & structure", "status": "strong|adequate|needs_work", "comment": "...", "suggestions": ["..."]}},
    {{"name": "Support for claims & use of authority", "status": "strong|adequate|needs_work", "comment": "...", "suggestions": ["..."]}},
    {{"name": "Clarity, grammar & citation form", "status": "strong|adequate|needs_work", "comment": "...", "suggestions": ["..."]}}
  ],
  "top_priorities": ["the 2-3 most important things to fix before submitting"]
}}
Respond ONLY with the JSON object."""
    return prompt


def content_and_rubric_prompt(paper_text, assignment_ctx, rubric_v2, source_summary):
    """Stage-2 substance/rubric prompt. Shared with scripts/local_grade.py."""
    rubric_text = _rubric_text(rubric_v2)
    level_names = _rubric_level_names(rubric_v2)
    levels_hint = (" Use ONLY these level names: " + ", ".join(level_names) + ".") if level_names else ""
    prompt = f"""You are assisting a law instructor about to grade a SUBMITTED paper. The mechanical/structural review and a source check are already done. Your job is the SUBSTANCE: assess the legal analysis and content, and map the paper to the instructor's rubric so they can focus on adjudicating the law. Be candid; flag where the instructor's expert judgment is most needed. Do not invent rubric criteria.

{assignment_ctx}

SOURCE CHECK (cited authorities):
{source_summary}

RUBRIC:
{rubric_text or "(no structured rubric provided)"}

SUBMITTED PAPER:
{_truncate(paper_text)}

Return a JSON object:
{{
  "content_assessment": "3-5 sentences on the strength of the legal analysis/argument and use of authority — what is strong, what is weak, where the instructor should look closely",
  "rubric": [
    {{"criterion": "<exact criterion name from the rubric>", "suggested_level": "<one of the rubric's level names>", "rationale": "1-2 sentences tied to the paper", "instructor_check": "what to verify to confirm or adjust"}}
  ],
  "focus_for_instructor": ["2-4 substance/law questions most worth the instructor's expert attention"],
  "integrity_note": "anything about sourcing/authority that bears on grading (e.g. a key cite that does not support its proposition), else empty string"
}}
{levels_hint} Respond ONLY with the JSON object."""
    return prompt
