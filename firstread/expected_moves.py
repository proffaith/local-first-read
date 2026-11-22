"""
"Expected moves" keys for open writing assignments (story 276).

Planted-error keys (utils/intentional_errors.py) fit closed exercises. For
open writing — a memo, a brief, a contract clause — the honest key is a
checklist of moves a competent submission must make: state the governing
rule, apply it to these facts, engage the controlling authority, address
the strongest counterargument, reach a conclusion that follows. Moves are
scored for PRESENCE with evidence, never for matching a model answer.

The checklist lives beside the assignment pack as ``ia<id>.moves.json``:

    {"moves": [{"id": "M1", "move": "States the governing rule for ...",
                "severity": "high", "looks_like": "..."}, ...],
     "drafted_by": "gemma4:26b" | "instructor", "updated_at": "..."}

Pure functions here (prompt builders, schema, tidy-up, summary); model
calls live in utils/local_eval.py so the same prompts can run on Anthropic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MOVE_STATUSES = ["present_strong", "present_weak", "missing", "not_applicable"]
PRESENT_STATUSES = {"present_strong", "present_weak"}
MOVE_STATUS_LABELS = {
    "present_strong": "present and done well",
    "present_weak": "present but weak",
    "missing": "MISSING",
    "not_applicable": "not applicable to this submission",
}
SEVERITIES = ["high", "medium", "low"]


# ── sidecar file ─────────────────────────────────────────────────────────────

def moves_path(pack_path) -> Path:
    p = Path(pack_path)
    return p.with_name(p.stem + ".moves.json")


def load_moves(pack_path) -> Optional[dict]:
    p = moves_path(pack_path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    moves = [m for m in (d.get("moves") or []) if m.get("id") and m.get("move")]
    if not moves:
        return None
    d["moves"] = moves
    return d


def save_moves(pack_path, moves: List[dict], drafted_by: str = "instructor") -> dict:
    doc = {"moves": normalize_moves(moves), "drafted_by": drafted_by,
           "updated_at": datetime.now(timezone.utc).isoformat()}
    moves_path(pack_path).write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc


def normalize_moves(moves: List[dict]) -> List[dict]:
    out = []
    for m in moves:
        text = (m.get("move") or "").strip()
        if not text:
            continue
        sev = (m.get("severity") or "medium").strip().lower()
        out.append({"id": (m.get("id") or f"M{len(out) + 1}").strip()[:8],
                    "move": text[:300], "severity": sev if sev in SEVERITIES else "medium",
                    "looks_like": (m.get("looks_like") or "").strip()[:400]})
    return out


def parse_moves_text(text: str) -> List[dict]:
    """Instructor-friendly editor format, one move per line:
    ``M1 | high | Applies the rule to these facts | looks like: ...``.
    The id and the trailing 'looks like' are optional; severity defaults
    to medium."""
    moves = []
    for i, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        mid, sev, move, looks = "", "medium", "", ""
        if len(parts) >= 3 and parts[1].lower() in SEVERITIES:
            mid, sev, move = parts[0], parts[1].lower(), parts[2]
            looks = " | ".join(parts[3:])
        elif len(parts) >= 2 and parts[0].lower() in SEVERITIES:
            sev, move = parts[0].lower(), parts[1]
            looks = " | ".join(parts[2:])
        else:
            move = parts[0]
            looks = " | ".join(parts[1:])
        if looks.lower().startswith("looks like:"):
            looks = looks[len("looks like:"):].strip()
        moves.append({"id": mid or f"M{len(moves) + 1}", "severity": sev,
                      "move": move, "looks_like": looks})
    return normalize_moves(moves)


def moves_to_text(moves: List[dict]) -> str:
    return "\n".join(
        f"{m['id']} | {m['severity']} | {m['move']}"
        + (f" | looks like: {m['looks_like']}" if m.get("looks_like") else "")
        for m in moves)


# ── drafting a checklist from the assignment design ─────────────────────────

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "moves": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "move": {"type": "string"},
                "severity": {"type": "string", "enum": SEVERITIES},
                "looks_like": {"type": "string"},
            },
            "required": ["move", "severity", "looks_like"],
        }},
    },
    "required": ["moves"],
}


def draft_moves_prompt(pack: dict, max_moves: int = 8) -> str:
    from firstread.rubric_prompts import _rubric_text
    rubric = _rubric_text(pack.get("rubric_v2"))
    return "\n".join([
        "You are helping an instructor write an 'expected moves' checklist for an "
        "open writing assignment: the analytical moves a competent submission must "
        "make, scored later for presence (not for matching a model answer).",
        "",
        "ASSIGNMENT DESIGN:", pack.get("assignment_ctx", ""),
        "", "RUBRIC:", rubric or "(none)",
        "",
        f"Propose {min(max_moves, 8)} or fewer moves, most consequential first. Each "
        "move is one concrete, observable thing in the text (e.g. 'States the "
        "elements of negligence before applying them', 'Cites and applies the "
        "controlling Pennsylvania authority on invitee duty', 'Addresses the "
        "strongest counterargument'). Not style, not formatting, not vague "
        "virtues. severity: high for moves without which the analysis fails, "
        "medium for moves that materially strengthen it, low for polish. "
        "looks_like: one sentence describing what a strong version reads like.",
    ])


# ── scoring a submission against the checklist ──────────────────────────────

def build_moves_prompt(moves_doc: dict, submission_text: str, *,
                       assignment_ctx: str = "",
                       max_submission_chars: int = 40000) -> str:
    moves = moves_doc.get("moves") or []
    lines = [
        "You are assisting an instructor grading an open writing assignment. The "
        "instructor's checklist below lists the analytical MOVES a competent "
        "submission must make. Judge each move for PRESENCE in this submission — "
        "not whether it matches any particular answer.",
    ]
    if assignment_ctx:
        lines += ["", "ASSIGNMENT CONTEXT:", assignment_ctx]
    lines += ["", "EXPECTED MOVES:"]
    for m in moves:
        lines.append(f"[{m['id']}] severity={m['severity']} — {m['move']}")
        if m.get("looks_like"):
            lines.append(f"    a strong version looks like: {m['looks_like']}")
    lines += ["", "SUBMISSION:", (submission_text or "")[:max_submission_chars], "",
              "For EACH move decide:",
              "- present_strong: the move is there and done competently.",
              "- present_weak: the move is attempted but thin, partly wrong, or "
              "unsupported (say what is missing in the note).",
              "- missing: the submission does not make this move.",
              "- not_applicable: this submission's task or facts make the move moot "
              "(rare; explain).",
              "Quote the student's own words as evidence (short). Be strict: naming a "
              "rule is not applying it; mentioning a case is not engaging it.",
              "",
              "Finish with a two-sentence summary of the analytical strengths and the "
              "most important gap."]
    return "\n".join(lines)


def moves_schema(move_ids) -> Dict[str, Any]:
    ids = list(move_ids) or []
    id_schema = {"type": "string", "enum": ids} if ids else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "moves": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": id_schema,
                    "status": {"type": "string", "enum": MOVE_STATUSES},
                    "evidence": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["id", "status", "evidence", "note"],
            }},
            "summary": {"type": "string"},
        },
        "required": ["moves", "summary"],
    }


def tidy_moves(result: Dict[str, Any], moves_doc: dict) -> Dict[str, Any]:
    key = {m["id"]: m for m in (moves_doc.get("moves") or [])}
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in (result or {}).get("moves") or []:
        rid = r.get("id")
        if rid in key and rid not in by_id:
            by_id[rid] = r
    rows = []
    for mid, m in key.items():
        r = by_id.get(mid) or {"id": mid, "status": "missing", "evidence": "",
                               "note": "model returned no row for this move — "
                                       "check manually"}
        if r.get("status") not in MOVE_STATUSES:
            r["status"] = "missing"
        rows.append({**r, "move": m["move"], "severity": m.get("severity", "medium")})
    applicable = [r for r in rows if r["status"] != "not_applicable"]
    present = sum(1 for r in applicable if r["status"] in PRESENT_STATUSES)
    strong = sum(1 for r in applicable if r["status"] == "present_strong")
    high = [r for r in applicable if r["severity"] == "high"]
    return {
        "moves": rows, "summary": (result or {}).get("summary") or "",
        "total": len(applicable), "present": present, "strong": strong,
        "high_total": len(high),
        "high_present": sum(1 for r in high if r["status"] in PRESENT_STATUSES),
    }


def moves_summary_text(det: Dict[str, Any]) -> str:
    if not det:
        return ""
    lines = [f"EXPECTED MOVES CHECK: {det.get('present', 0)} of {det.get('total', 0)} "
             f"moves present ({det.get('strong', 0)} strong; "
             f"{det.get('high_present', 0)}/{det.get('high_total', 0)} high-severity)."]
    for r in det.get("moves") or []:
        lines.append(f"- {r.get('id')} ({r.get('severity')}) {r.get('move')}: "
                     f"{MOVE_STATUS_LABELS.get(r.get('status'), r.get('status'))}"
                     + (f" — {r['note']}" if r.get("note") else ""))
    if det.get("summary"):
        lines.append(det["summary"])
    return "\n".join(lines)
