"""
Instructor override log + calibration exemplars for the local evaluation
package (story 277).

The instructor confirms or overrides each suggested rubric level (and each
planted-error status) on a file's report page. Every decision is recorded
twice: ``<stem>.overrides.json`` in the run folder (that file's decisions)
and one JSONL line per decision in ``<results root>/_calibration/
course<id>.jsonl`` (the instructor's running record for that course). No
student text is stored — only criterion, model level, instructor level, and
the instructor's own reason.

At grading time ``calibration_block()`` pulls the most relevant past
decisions for the assignment's criteria (same assignment first, then the
course; disagreements with reasons first) and hands them to the rubric
prompt as calibration exemplars. Retrieval, not weights: auditable, per
instructor and per course, switchable, and it produces the agreement
statistics (``agreement_stats``) that decide whether training is ever worth
it. Stores are per course on purpose — pooling faculty averages out the
signal.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CALIBRATION_DIR = "_calibration"
OVERRIDES_SUFFIX = ".overrides.json"
MAX_REASON = 400


def _course_file(root, course_id) -> Path:
    return Path(root) / CALIBRATION_DIR / f"course{course_id or 'none'}.jsonl"


def _stem_token(run_id: str, stem: str) -> str:
    return hashlib.sha256(f"{run_id}/{stem}".encode("utf-8")).hexdigest()[:12]


# ── recording ────────────────────────────────────────────────────────────────

def record_overrides(root, pack: dict, run_id: str, outdir, stem: str,
                     result_record: dict, form: dict) -> dict:
    """Persist the instructor's decisions for one file.

    ``form`` = ``{"rubric": {criterion: {"level": str, "reason": str}},
    "planted": {error_id: {"status": str, "reason": str}}}``; entries with an
    empty level/status are dropped (no decision). Model values are taken
    from ``result_record`` so each entry carries the pair. Returns the
    overrides document written to the run folder."""
    ts = datetime.now(timezone.utc).isoformat()
    model_levels = {r.get("criterion"): r.get("suggested_level")
                    for r in ((result_record.get("result") or {}).get("rubric") or [])}
    model_status = {r.get("id"): r.get("status")
                    for r in ((result_record.get("planted") or {}).get("errors") or [])}
    doc = {"run_id": run_id, "stem": stem, "saved_at": ts,
           "assignment_id": pack.get("assignment_id"),
           "course_id": pack.get("course_id"), "rubric": {}, "planted": {}}
    lines = []
    token = _stem_token(run_id, stem)
    for crit, d in (form.get("rubric") or {}).items():
        level = (d.get("level") or "").strip()
        if not level or crit not in model_levels:
            continue
        reason = (d.get("reason") or "").strip()[:MAX_REASON]
        doc["rubric"][crit] = {"model": model_levels[crit], "instructor": level,
                               "reason": reason}
        lines.append({"ts": ts, "kind": "rubric", "run_id": run_id,
                      "item": token, "assignment_id": pack.get("assignment_id"),
                      "assignment_title": pack.get("title", ""), "key": crit,
                      "model": model_levels[crit], "instructor": level,
                      "reason": reason})
    for eid, d in (form.get("planted") or {}).items():
        status = (d.get("status") or "").strip()
        if not status or eid not in model_status:
            continue
        reason = (d.get("reason") or "").strip()[:MAX_REASON]
        doc["planted"][eid] = {"model": model_status[eid], "instructor": status,
                               "reason": reason}
        lines.append({"ts": ts, "kind": "planted", "run_id": run_id,
                      "item": token, "assignment_id": pack.get("assignment_id"),
                      "assignment_title": pack.get("title", ""), "key": eid,
                      "model": model_status[eid], "instructor": status,
                      "reason": reason})
    (Path(outdir) / f"{stem}{OVERRIDES_SUFFIX}").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False))
    cf = _course_file(root, pack.get("course_id"))
    cf.parent.mkdir(parents=True, exist_ok=True)
    with open(cf, "a", encoding="utf-8") as fh:
        for ln in lines:
            fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
    return doc


def load_overrides(outdir, stem: str) -> Optional[dict]:
    p = Path(outdir) / f"{stem}{OVERRIDES_SUFFIX}"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


# ── loading ──────────────────────────────────────────────────────────────────

def load_calibration(root, course_id) -> list[dict]:
    """All decisions for a course, latest decision per (item, kind, key)."""
    p = _course_file(root, course_id)
    if not p.is_file():
        return []
    latest: dict[tuple, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        latest[(e.get("item"), e.get("kind"), e.get("key"))] = e   # file order = time
    entries = list(latest.values())
    entries.sort(key=lambda e: e.get("ts") or "", reverse=True)
    return entries


def list_courses(root) -> list[dict]:
    d = Path(root) / CALIBRATION_DIR
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("course*.jsonl")):
        cid = p.stem[len("course"):]
        entries = load_calibration(root, cid)
        titles = sorted({e.get("assignment_title", "") for e in entries if e.get("assignment_title")})
        out.append({"course_id": cid, "decisions": len(entries),
                    "assignments": titles})
    return out


# ── exemplars → prompt block ─────────────────────────────────────────────────

def exemplars_for(entries: list[dict], assignment_id, key: str,
                  kind: str = "rubric", n: int = 3) -> list[dict]:
    """Most useful past decisions for one criterion: same assignment before
    the rest of the course; disagreements with a reason first; newest
    first within each tier."""
    rows = [e for e in entries if e.get("kind") == kind and e.get("key") == key]

    def rank(e):
        same = 0 if str(e.get("assignment_id")) == str(assignment_id) else 1
        disagree = 0 if e.get("model") != e.get("instructor") else 1
        reasoned = 0 if e.get("reason") else 1
        return (same, disagree, reasoned, e.get("ts") or "")

    rows.sort(key=lambda e: (rank(e)[:3], ))
    # newest first inside the tiers
    rows.sort(key=lambda e: (rank(e)[:3], "".join(reversed(e.get("ts") or ""))))
    return rows[:n]


def calibration_block(root, pack: dict, criteria: list[str],
                      n_per_criterion: int = 3) -> str:
    """Text for the rubric prompt: this instructor's past calls on these
    criteria. Empty string when there is no history."""
    entries = load_calibration(root, pack.get("course_id"))
    if not entries:
        return ""
    lines = []
    for crit in criteria:
        ex = exemplars_for(entries, pack.get("assignment_id"), crit,
                           n=n_per_criterion)
        if not ex:
            continue
        lines.append(f"- {crit}:")
        for e in ex:
            if e.get("model") == e.get("instructor"):
                lines.append(f"    · confirmed '{e['instructor']}'"
                             + (f" — {e['reason']}" if e.get("reason") else ""))
            else:
                lines.append(f"    · model suggested '{e.get('model')}', instructor "
                             f"set '{e.get('instructor')}'"
                             + (f" — {e['reason']}" if e.get("reason") else ""))
    if not lines:
        return ""
    return ("INSTRUCTOR CALIBRATION — this instructor's past decisions on this "
            "rubric (apply the same standards; the instructor's level is the "
            "reference):\n" + "\n".join(lines))


# ── agreement statistics ────────────────────────────────────────────────────

def agreement_stats(entries: list[dict], level_names: Optional[list] = None) -> dict:
    """Per criterion (rubric) and per planted-error status: n, exact
    agreement, adjacent agreement (one level apart, when level order is
    known), and the model's tendency (higher / lower than the instructor)."""
    order = {name: i for i, name in enumerate(level_names or [])}
    out: dict[str, dict] = {}
    by_key: dict[tuple, list] = defaultdict(list)
    for e in entries:
        by_key[(e.get("kind"), e.get("key"))].append(e)
    for (kind, key), rows in by_key.items():
        n = len(rows)
        exact = sum(1 for e in rows if e.get("model") == e.get("instructor"))
        adjacent = higher = lower = 0
        for e in rows:
            m, i = order.get(e.get("model")), order.get(e.get("instructor"))
            if m is None or i is None:
                continue
            if abs(m - i) == 1:
                adjacent += 1
            if m > i:
                higher += 1
            elif m < i:
                lower += 1
        tendency = ("model reads higher" if higher > lower else
                    "model reads lower" if lower > higher else "no lean")
        out[f"{kind}:{key}"] = {
            "kind": kind, "key": key, "n": n, "exact": exact,
            "exact_rate": round(exact / n, 2) if n else 0.0,
            "adjacent": adjacent,
            "within_one_rate": round((exact + adjacent) / n, 2) if n else 0.0,
            "model_higher": higher, "model_lower": lower, "tendency": tendency,
            "recent_disagreements": [
                {"ts": e.get("ts"), "model": e.get("model"),
                 "instructor": e.get("instructor"), "reason": e.get("reason", ""),
                 "assignment_title": e.get("assignment_title", "")}
                for e in rows if e.get("model") != e.get("instructor")][:5],
        }
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["kind"], kv[1]["key"])))
