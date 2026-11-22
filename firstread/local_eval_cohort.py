"""
Cohort views over one local-eval run (utils/local_eval.py):

  * per-student pairing — a student usually submits several files (a redlined
    document AND a cover memo); group them and score the student as the
    UNION of what the files show;
  * class heat map — planted errors × students, citation problems, rubric
    level distributions; and
  * a "next class" brief — themes to reteach plus a pseudonymous announcement
    draft, synthesized by the local model from the aggregates only.

Everything here reads the run folder the pipeline already wrote and adds
``_students.json``, ``_cohort.json`` and ``_next_class.md`` beside it.
Nothing leaves the machine.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from firstread.local_eval import (EvalOptions, HISTORY_ALERTS, PROBLEM_STATUSES,
                              SOURCE_LABELS, ACADEMIC_LABELS, SOURCE_KIND_LABELS,
                              D2L_PREFIX_RE, chat_json,
                              finding_status, is_academic,
                              d2l_prefix,
                              LocalLLMError)

log = logging.getLogger(__name__)

_STATUS_PHRASE = {"not_found": "not found in Semantic Scholar or OpenAlex",
                  "found_conflated": "cited with the wrong authors or DOI",
                  "ambiguous": "too incomplete to identify"}


def _problem_key(f: dict) -> str:
    """De-duplication key for a citation problem: the citation string for a
    case, the source key (DOI, or title + year) for an academic source."""
    if is_academic(f):
        return f.get("key") or f.get("citation") or ""
    return f.get("citation") or ""


def _problem_display(f: dict) -> str:
    if is_academic(f):
        return (f.get("cited") or {}).get("title") or f.get("citation") or ""
    return f.get("citation") or ""


def _student_problem(key: str, f: dict) -> dict:
    row = {"citation": _problem_display(f), "status": f.get("status"),
           "case_name": f.get("case_name", ""), "note": (f.get("note") or "")[:200]}
    if is_academic(f):
        row.update(key=key, type="academic", source_kind=f.get("source_kind", ""))
    return row


def _problem_label(cp: dict) -> str:
    if cp.get("type") == "academic":
        return ACADEMIC_LABELS.get(cp.get("status"), "")
    return SOURCE_LABELS.get(cp.get("status"), "")

# Brightspace "Download all submissions" file-name prefix (shared with the
# pipeline, which derives scrub identities from it).
_D2L_PREFIX = D2L_PREFIX_RE

# Union of detection statuses across a student's files.
_EXPLAINED = {"fixed_and_explained", "identified_not_fixed"}
_FIXED = {"fixed_and_explained", "fixed_silently"}


# ── Brightspace download index ───────────────────────────────────────────────

_DATE_RE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? "
                      r"\d{1,2},? \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}:\d{2}", re.I)


def _looks_like_name(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 80 or _DATE_RE.search(t) or t.replace(" ", "").isdigit():
        return False
    if re.search(r"\.(docx?|pdf|txt|md|html?|zip)$", t, re.I):
        return False
    return ("," in t and len(t.split(",")) == 2) or len(t.split()) >= 2


def parse_submission_index(folder) -> dict:
    """Read Brightspace's ``index.html`` (the roster page that ships at the
    root of a bulk submissions download) and map each linked file, by
    folder-relative path and by basename, to the student named on its row.

    The page is a table; a row carries a name cell and one or more file
    links, or a name row is followed by file rows. Any parser is a guess
    against an undocumented export, so the result is tolerant: rows with no
    name-like cell fall back to the most recent name; files not linked from
    the index are left to the other pairing methods.

    Returns ``{"files": {key: label}, "names": [labels...], "path": str}``
    or an empty dict when there is no index."""
    from html.parser import HTMLParser
    from urllib.parse import unquote

    folder = Path(folder)
    index = next((p for p in folder.iterdir()
                  if p.is_file() and p.name.lower() == "index.html"), None) \
        if folder.is_dir() else None
    if not index:
        return {}

    class _Rows(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self._row, self._cell, self._href = [], None, None, None
            self._cell_has_link = False
            self.title, self.heading, self._text_tag = "", "", None

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag in ("title", "h1", "h2") and not (tag != "title" and self.heading):
                self._text_tag = tag
            if tag == "tr":
                self._row = {"cells": [], "links": []}
            elif tag in ("td", "th") and self._row is not None:
                self._cell, self._cell_has_link = [], False
            elif tag == "a" and self._row is not None and a.get("href"):
                self._href = a["href"]
                self._cell_has_link = True
                self._row["links"].append({"href": a["href"], "text": ""})

        def handle_endtag(self, tag):
            if tag == self._text_tag:
                self._text_tag = None
            if tag in ("td", "th") and self._row is not None and self._cell is not None:
                # a cell full of file links is never the name cell
                if not self._cell_has_link:
                    self._row["cells"].append(" ".join("".join(self._cell).split()))
                self._cell = None
            elif tag == "a":
                self._href = None
            elif tag == "tr" and self._row is not None:
                self.rows.append(self._row)
                self._row = None

        def handle_data(self, data):
            if self._text_tag == "title":
                self.title += data
            elif self._text_tag in ("h1", "h2"):
                self.heading += data
            if self._cell is not None:
                self._cell.append(data)
            if self._href is not None and self._row is not None and self._row["links"]:
                self._row["links"][-1]["text"] += data

    parser = _Rows()
    try:
        parser.feed(index.read_text(errors="ignore"))
    except Exception:  # noqa: BLE001
        return {}

    heading = " ".join(parser.heading.split())
    title = " ".join(parser.title.split())
    files, names, current = {}, [], None
    for row in parser.rows:
        name = next((c for c in row["cells"] if _looks_like_name(c)), None)
        if name:
            current = name
            if name not in names:
                names.append(name)
        if not current:
            continue
        for link in row["links"]:
            href = unquote(link["href"].split("#")[0].split("?")[0])
            if not href or href.lower().endswith((".html", ".htm")) or "://" in href:
                continue
            rel = href.lstrip("./")
            files[rel] = current
            files[Path(rel).name] = current
    if not files and not (heading or title):
        return {}
    return {"files": files, "names": names, "path": str(index),
            "heading": heading, "title": title}


# ── pairing ──────────────────────────────────────────────────────────────────

def student_key(filename: str, identities: Optional[list] = None,
                index: Optional[dict] = None) -> dict:
    """``{"label", "method"}`` for a submission file name.

    Order: the Brightspace index.html (``index``), then the bulk-download
    file-name prefix (deterministic), then a roster name found in the file
    name (``Last, First`` / ``First Last`` / ``FLast``), else the file's
    own stem (one student per file)."""
    stem = Path(filename).stem
    files = (index or {}).get("files") or {}
    if files.get(filename):                    # exact folder-relative path
        return {"label": files[filename], "method": "index"}
    # the Brightspace prefix (file name or per-submission folder) beats a
    # bare-name index match: in the folder layout every student may have
    # a "memo.docx", so the bare name is ambiguous
    m = d2l_prefix(filename)
    if m:
        return {"label": m["name"], "method": "d2l_prefix", "user_id": m["uid"]}
    if files.get(Path(filename).name):
        return {"label": files[Path(filename).name], "method": "index"}
    hay = re.sub(r"[^a-z0-9]+", " ", stem.lower())
    for ident in identities or []:
        name = (ident.get("name") or "").strip()
        if not name:
            continue
        parts = [p for p in re.split(r"[\s,]+", name) if p]
        if len(parts) < 2:
            continue
        if "," in name:                       # "Last, First"
            last, first = parts[0], parts[1]
        else:                                 # "First Last"
            first, last = parts[0], parts[-1]
        cands = {f"{first} {last}", f"{last} {first}", f"{first[0]}{last}",
                 f"{first} {last[0]}"}
        for c in cands:
            c = re.sub(r"[^a-z0-9]+", " ", c.lower()).strip()
            if c and re.search(rf"(?<![a-z0-9]){re.escape(c)}(?![a-z0-9])", hay):
                return {"label": name, "method": "roster"}
    return {"label": stem, "method": "file"}


def _union_status(statuses: list[str]) -> str:
    s = set(statuses)
    explained = bool(s & _EXPLAINED)
    fixed = bool(s & _FIXED)
    if explained and fixed:
        return "fixed_and_explained"
    if fixed:
        return "fixed_silently"
    if explained:
        return "identified_not_fixed"
    if "unclear" in s and "missed" not in s:
        return "unclear"
    if "missed" in s:
        return "missed"
    return "unclear"


PAIRING_FILE = "_pairing.json"


def load_pairing(outdir) -> dict:
    """Instructor's manual grouping: ``{file name: student label}``. Written
    by the UI; empty when absent."""
    p = Path(outdir) / PAIRING_FILE
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text())
        return {k: str(v).strip() for k, v in d.items() if str(v).strip()}
    except Exception:  # noqa: BLE001
        return {}


def group_students(manifest: dict, results: dict, identities=None,
                   overrides: Optional[dict] = None,
                   index: Optional[dict] = None) -> list[dict]:
    """Pair files into students and union their findings.

    ``results`` maps stem → the per-file JSON record. ``overrides`` (file
    name → label) is the instructor's manual grouping and wins over the
    automatic key; ``index`` is the parsed Brightspace index.html. Returns
    one entry per student: label, method, files, union flags, union
    planted-error rows, citation problems across files, and the per-file
    rubric levels."""
    groups: dict[str, dict] = {}
    order: list[str] = []
    overrides = overrides or {}
    for f in manifest.get("files", []):
        if f["name"] in overrides:
            key = {"label": overrides[f["name"]], "method": "manual"}
        else:
            key = student_key(f["name"], identities, index)
        label = key["label"]
        if label not in groups:
            groups[label] = {"label": label, "method": key["method"],
                             "user_id": key.get("user_id"), "files": []}
            order.append(label)
        groups[label]["files"].append(f)

    students = []
    for label in order:
        g = groups[label]
        recs = [results.get(f["stem"]) for f in g["files"]]
        recs = [r for r in recs if r]
        ok_files = [f for f in g["files"] if f.get("status") == "ok"]
        # planted errors: union
        by_id: dict[str, list] = defaultdict(list)
        meta: dict[str, dict] = {}
        for r in recs:
            for row in ((r.get("planted") or {}).get("errors") or []):
                by_id[row["id"]].append(row)
                meta.setdefault(row["id"], row)
        planted = []
        for eid, rows in by_id.items():
            st = _union_status([x.get("status") for x in rows])
            ev = next((x.get("evidence") for x in rows
                       if x.get("status") == st and x.get("evidence")), "") \
                or next((x.get("evidence") for x in rows if x.get("evidence")), "")
            planted.append({"id": eid, "severity": meta[eid].get("severity"),
                            "category": meta[eid].get("category"),
                            "status": st, "evidence": ev,
                            "per_file": {x.get("_stem", ""): x.get("status")
                                         for x in rows}})
        # keep the key order (E1, E2, ...) as the first record listed it
        first = next((r for r in recs if r.get("planted")), None)
        if first:
            idx = {row["id"]: i for i, row in enumerate(first["planted"]["errors"])}
            planted.sort(key=lambda x: idx.get(x["id"], 999))
        detected = sum(1 for p in planted if p["status"] in
                       {"fixed_and_explained", "fixed_silently",
                        "identified_not_fixed"})
        # citations: problems across files, de-duplicated by citation string
        problems: dict[str, dict] = {}
        alerts: dict[str, dict] = {}
        n_cites = 0
        for r in recs:
            for fnd in ((r.get("source") or {}).get("findings") or []):
                n_cites += 1
                if finding_status(fnd) in PROBLEM_STATUSES:
                    # one academic source cited two ways is still one source
                    problems.setdefault(_problem_key(fnd), fnd)
                if (fnd.get("history") or {}).get("status") in HISTORY_ALERTS:
                    alerts.setdefault(fnd["citation"], fnd)
        levels = {}
        for f in ok_files:
            if f.get("levels"):
                levels[f["stem"]] = f["levels"]
        students.append({
            "label": label, "method": g["method"], "user_id": g.get("user_id"),
            "files": [{"name": f["name"], "stem": f["stem"],
                       "status": f.get("status"), "flags": f.get("flags")}
                      for f in g["files"]],
            "flags": {"citations": n_cites, "problems": len(problems),
                      "history_alerts": len(alerts),
                      "key_total": len(planted) if planted else None,
                      "key_detected": detected if planted else None},
            "planted": planted,
            "citation_problems": [_student_problem(k, v) for k, v in problems.items()],
            "levels": levels,
        })
    return students


# ── heat map / aggregates ───────────────────────────────────────────────────

def cohort_aggregate(students: list[dict], results: dict,
                     pack: Optional[dict]) -> dict:
    key = ((pack or {}).get("intentional_errors") or {}).get("planted_errors") or []
    if not key:
        # No pack at hand (rebuilt later, pack moved): the per-file rows carry
        # id / severity / category, which is enough for the grid.
        first = next((r for r in results.values() if (r.get("planted") or {}).get("errors")), None)
        if first:
            key = [{"id": e["id"], "severity": e.get("severity", ""),
                    "category": e.get("category", ""), "whats_wrong": "",
                    "why_it_matters": ""} for e in first["planted"]["errors"]]
    graded = [s for s in students if any(f.get("status") == "ok" for f in s["files"])]

    # planted errors × students
    heat = []
    if key:
        for e in key:
            eid = e.get("id")
            cells = {}
            for s in graded:
                row = next((p for p in s["planted"] if p["id"] == eid), None)
                cells[s["label"]] = row["status"] if row else "unclear"
            n = len(graded)
            missed = sum(1 for v in cells.values() if v == "missed")
            unclear = sum(1 for v in cells.values() if v == "unclear")
            not_fixed = sum(1 for v in cells.values() if v == "identified_not_fixed")
            heat.append({"id": eid, "severity": e.get("severity"),
                         "category": e.get("category"),
                         "whats_wrong": e.get("whats_wrong", ""),
                         "why_it_matters": e.get("why_it_matters", ""),
                         "cells": cells, "students": n, "missed": missed,
                         "identified_not_fixed": not_fixed, "unclear": unclear,
                         "miss_rate": round(missed / n, 2) if n else 0.0})
        heat.sort(key=lambda h: (-h["miss_rate"], -h["identified_not_fixed"],
                                 h["severity"] != "high"))

    # citation problems across the cohort
    cite_counter: Counter = Counter()
    cite_meta: dict[str, dict] = {}
    status_counter: Counter = Counter()
    for s in graded:
        for cp in s["citation_problems"]:
            k = cp.get("key") or cp["citation"]
            cite_counter[k] += 1
            cite_meta.setdefault(k, cp)
            status_counter[cp["status"]] += 1
    citations = []
    for c, n in cite_counter.most_common(10):
        cp = cite_meta[c]
        row = {"citation": cp["citation"], "students": n, "status": cp["status"],
               "label": _problem_label(cp), "case_name": cp.get("case_name", ""),
               "note": cp.get("note", "")}
        if cp.get("type") == "academic":
            row.update(type="academic",
                       kind=SOURCE_KIND_LABELS.get(cp.get("source_kind"), ""))
        citations.append(row)
    # Academic sources (Story 1028): every verdict across the class, and the
    # same unfindable reference in two or more papers (spec §2.4.2).
    academic_counter: Counter = Counter()
    for rec in results.values():
        for fnd in ((rec.get("source") or {}).get("findings") or []):
            if is_academic(fnd):
                academic_counter[fnd.get("status") or "unknown"] += 1
    shared_unfindable = [{"citation": cite_meta[c]["citation"], "students": n,
                          "kind": SOURCE_KIND_LABELS.get(cite_meta[c].get("source_kind"), "")}
                         for c, n in cite_counter.most_common()
                         if n >= 2 and cite_meta[c].get("type") == "academic"
                         and cite_meta[c]["status"] == "not_found"]

    # rubric level distribution + focus lines
    level_dist: dict[str, Counter] = defaultdict(Counter)
    focus: list[str] = []
    integrity: list[str] = []
    for rec in results.values():
        res = rec.get("result") or {}
        for row in res.get("rubric") or []:
            level_dist[row.get("criterion", "?")][row.get("suggested_level", "?")] += 1
        focus += [x for x in (res.get("focus_for_instructor") or []) if x]
        if res.get("integrity_note"):
            integrity.append(res["integrity_note"])
    extra_issues: Counter = Counter()
    for rec in results.values():
        for a in ((rec.get("planted") or {}).get("additional_issues") or []):
            if a.get("assessment") == "genuine_issue":
                extra_issues[(a.get("description") or "")[:120]] += 1

    return {
        "students": len(graded),
        "files": sum(len(s["files"]) for s in students),
        "heat": heat,
        "citation_problems": citations,
        "citation_status_counts": dict(status_counter),
        "academic_status_counts": dict(academic_counter),
        "shared_unfindable": shared_unfindable,
        "rubric_levels": {c: dict(v) for c, v in level_dist.items()},
        "focus_lines": focus[:40],
        "integrity_notes": integrity[:20],
        "extra_issues": [{"issue": k, "students": v}
                         for k, v in extra_issues.most_common(8)],
    }


# ── next-class brief (local model) ──────────────────────────────────────────

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "prevalence": {"type": "string"},
                "what_to_teach": {"type": "string"},
                "in_class_move": {"type": "string"},
            },
            "required": ["title", "prevalence", "what_to_teach", "in_class_move"],
        }},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "announcement": {"type": "string"},
        "instructor_notes": {"type": "string"},
    },
    "required": ["themes", "strengths", "announcement", "instructor_notes"],
}


def brief_prompt(agg: dict, pack: Optional[dict]) -> str:
    title = (pack or {}).get("title") or "this assignment"
    lines = [
        "You are helping an instructor plan the follow-up to a graded assignment. "
        f"Assignment: “{title}”. {agg['students']} student(s), {agg['files']} file(s). "
        "Below are ONLY class-level aggregates from a preliminary local review — "
        "no student names. Do not invent names or details.",
        "",
    ]
    if agg.get("heat"):
        lines.append("PLANTED-ERROR HEAT MAP (sorted by miss rate):")
        for h in agg["heat"]:
            lines.append(f"- {h['id']} [{h['severity']}] {h['category']}: "
                         f"missed by {h['missed']}/{h['students']}, "
                         f"flagged-but-not-fixed {h['identified_not_fixed']}, "
                         f"unclear {h['unclear']}. What's wrong: {h['whats_wrong'][:300]}")
        lines.append("")
    if agg.get("citation_problems"):
        lines.append("CITATION PROBLEMS ACROSS THE CLASS:")
        for c in agg["citation_problems"]:
            if c.get("type") == "academic":
                lines.append(f"- {c.get('kind') or 'source'} {_STATUS_PHRASE.get(c['status'], c['status'])} "
                             f"({c['students']} student(s)): “{c['citation']}”")
                continue
            lines.append(f"- {c['citation']} ({c['students']} student(s)): {c['label']}"
                         + (f" — {c['case_name']}" if c.get("case_name") else ""))
        for u in agg.get("shared_unfindable") or []:
            lines.append(f"- the same unfindable {u.get('kind') or 'source'} appears in "
                         f"{u['students']} students' papers: “{u['citation']}”")
        lines.append("")
    if agg.get("rubric_levels"):
        lines.append("RUBRIC LEVEL DISTRIBUTION (per file):")
        for crit, dist in agg["rubric_levels"].items():
            lines.append(f"- {crit}: " + ", ".join(f"{k} {v}" for k, v in dist.items()))
        lines.append("")
    if agg.get("extra_issues"):
        lines.append("GENUINE ISSUES STUDENTS RAISED BEYOND THE KEY:")
        lines += [f"- {x['issue']} ({x['students']})" for x in agg["extra_issues"]]
        lines.append("")
    if agg.get("focus_lines"):
        lines.append("RECURRING 'FOCUS FOR INSTRUCTOR' LINES FROM THE PER-PAPER READS:")
        lines += [f"- {x[:220]}" for x in agg["focus_lines"][:25]]
        lines.append("")
    lines += [
        "Produce, as JSON:",
        "- themes: 2–4 things worth reteaching next class, most consequential first. "
        "For each: title, prevalence (e.g. 'missed by 6 of 9'), what_to_teach (the "
        "rule or skill, stated correctly, 2–3 sentences), in_class_move (a 5–10 minute "
        "activity: a quick poll question, a pair exercise, or a worked example).",
        "- strengths: 1–3 things the class did well (encourage before correcting).",
        "- announcement: a short LMS announcement to the whole class (120–200 words), "
        "warm and specific, no names, no grades, no percentages that identify anyone; "
        "state what most people handled well, the one or two things to revisit, and "
        "what will happen next class.",
        "- instructor_notes: 2–3 sentences for the instructor only — where the local "
        "review is least reliable and what to double-check by hand.",
    ]
    return "\n".join(lines)


def next_class_brief(agg: dict, pack: Optional[dict], opts: EvalOptions) -> dict:
    """One local-model call over the aggregates. Raises LocalLLMError."""
    return chat_json(brief_prompt(agg, pack), model=opts.model, host=opts.host,
                     num_ctx=opts.num_ctx, temperature=opts.temperature,
                     think=opts.think, schema=BRIEF_SCHEMA)


# ── rendering ────────────────────────────────────────────────────────────────

_ST = {"fixed_and_explained": "✓✓", "fixed_silently": "✓", "identified_not_fixed": "!",
       "missed": "✗", "unclear": "?"}


def render_next_class_md(agg: dict, brief: Optional[dict], students: list[dict],
                         pack: Optional[dict], model: str) -> str:
    title = (pack or {}).get("title") or "Assignment"
    out = [f"# {title} — next class brief",
           f"*Class-level view of a preliminary local review ({model}, localhost). "
           f"{agg['students']} student(s), {agg['files']} file(s). Not grades.*", ""]
    if agg.get("heat"):
        labels = [s["label"] for s in students
                  if any(f.get("status") == "ok" for f in s["files"])]
        out += ["## Planted errors × students", "",
                "✓✓ found, explained and fixed · ✓ fixed silently · ! flagged but not "
                "fixed · ✗ missed · ? unclear", "",
                "| # | Sev | Planted error | Missed | " + " | ".join(labels) + " |",
                "|---|---|---|---|" + "---|" * len(labels)]
        for h in agg["heat"]:
            out.append(f"| {h['id']} | {h['severity']} | {h['category']} | "
                       f"{h['missed']}/{h['students']} | "
                       + " | ".join(_ST.get(h["cells"].get(l, "unclear"), "?")
                                    for l in labels) + " |")
        out.append("")
    if agg.get("citation_problems"):
        out += ["## Citation problems across the class", ""]
        out += [f"- **{c['citation']}** ({c['students']}): {c['label']}"
                + (f" — {c['case_name']}" if c.get("case_name") else "")
                for c in agg["citation_problems"]] + [""]
    if agg.get("shared_unfindable"):
        out += ["## The same unfindable source in several papers", ""]
        out += [f"- **{u['citation']}** ({u['students']} students)"
                for u in agg["shared_unfindable"]] + [""]
    if agg.get("rubric_levels"):
        out += ["## Rubric levels (per file)", ""]
        for crit, dist in agg["rubric_levels"].items():
            out.append(f"- **{crit}**: " + ", ".join(f"{k} {v}" for k, v in dist.items()))
        out.append("")
    if brief:
        out += ["## What to reteach", ""]
        for t in brief.get("themes") or []:
            out += [f"### {t.get('title')}  ·  {t.get('prevalence')}", "",
                    t.get("what_to_teach", ""), "",
                    f"*In class:* {t.get('in_class_move', '')}", ""]
        if brief.get("strengths"):
            out += ["## What went well", ""] + [f"- {s}" for s in brief["strengths"]] + [""]
        out += ["## Announcement draft (edit before posting)", "", "> " +
                (brief.get("announcement") or "").replace("\n", "\n> "), ""]
        if brief.get("instructor_notes"):
            out += ["## Instructor notes", "", brief["instructor_notes"], ""]
    out += ["## Students", ""]
    for s in students:
        fl = s["flags"]
        key = (f"key {fl['key_detected']}/{fl['key_total']}"
               if fl.get("key_total") else "")
        out.append(f"- **{s['label']}** ({len(s['files'])} file(s), {s['method']}): "
                   f"{fl['citations']} cite(s), {fl['problems']} problem(s), "
                   f"{fl['history_alerts']} history alert(s) {key}")
    return "\n".join(out) + "\n"


# ── entry point ──────────────────────────────────────────────────────────────

def build_cohort(outdir, manifest: dict, pack: Optional[dict],
                 opts: EvalOptions, identities=None,
                 with_brief: bool = True) -> dict:
    """Read the run folder, write ``_students.json``, ``_cohort.json`` and
    ``_next_class.md``. Never raises; a failed brief is recorded in the
    cohort record's ``errors``."""
    outdir = Path(outdir)
    results = {}
    for f in manifest.get("files", []):
        p = outdir / f"{f['stem']}.json"
        if p.is_file():
            try:
                rec = json.loads(p.read_text())
                for row in ((rec.get("planted") or {}).get("errors") or []):
                    row["_stem"] = f["stem"]
                results[f["stem"]] = rec
            except Exception:  # noqa: BLE001
                log.warning("unreadable result %s", p, exc_info=True)
    index = {}
    try:
        index = parse_submission_index(manifest.get("folder") or "")
    except Exception:  # noqa: BLE001
        log.warning("index.html parse failed", exc_info=True)
    students = group_students(manifest, results, identities,
                              overrides=load_pairing(outdir), index=index)
    agg = cohort_aggregate(students, results, pack)
    agg["index"] = {"found": bool(index), "names": len(index.get("names", [])),
                    "matched_files": sum(1 for s in students for f in s["files"]
                                         if s["method"] == "index")} if index else None
    errors = []
    brief = None
    if with_brief and results:
        try:
            brief = next_class_brief(agg, pack, opts)
        except LocalLLMError as e:
            errors.append(f"brief failed: {e}")
    cohort = {"aggregate": agg, "brief": brief, "errors": errors,
              "model": opts.model}
    (outdir / "_students.json").write_text(json.dumps(students, indent=2,
                                                      ensure_ascii=False))
    (outdir / "_cohort.json").write_text(json.dumps(cohort, indent=2,
                                                    ensure_ascii=False))
    (outdir / "_next_class.md").write_text(
        render_next_class_md(agg, brief, students, pack, opts.model))
    return {"students": students, **cohort}
