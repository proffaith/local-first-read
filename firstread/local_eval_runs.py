"""Organizing a results folder's runs for the Past runs page: which
semester, class and section each run belongs to, archiving, and deleting
duplicate or failed runs.

Each run's filing lives in its own folder as ``_filing.json``: only the
fields the instructor set (``semester``, ``course``, ``section``) plus
``archived`` / ``archived_at``. A field the instructor never set is
detected from what the run already holds, first hit wins:

1. the semester & section it was sent to course design under
   (``_debrief_sent.json``: "Fall 2026 · W03");
2. the class: the pack's course title, shortened to its course code
   ("BUSL 101 Business Law I rev. 2026" -> "BUSL 101"), so two versions of
   one course design file together;
3. a discussion print view's course name ("BUSL101 W01 F26");
4. the folders the submissions sit in ("eval/W03 Fall 2026/<download>").
   A Brightspace download folder's own name is the assignment title
   ("... Download Oct 22, 2026 644 AM"), so it is never read;
5. semester only: the run's start date (Jan-May Spring, Jun-Jul Summer,
   Aug-Dec Fall), shown as a guess.

Archiving only hides a run from the Past runs list; portfolios, calibration
and the run's own page are unchanged. Deleting removes the run folder.

No DB, no model: everything comes from the run folder and the pack file
its manifest names.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

FILING_FILE = "_filing.json"
FIELDS = ("semester", "course", "section")
SENT_FILE = "_debrief_sent.json"          # firstread.local_eval_send.SENT_FILE

# A run whose manifest still says running but has not been written for this
# long was interrupted (server restart, crash, closed terminal).
STALE_AFTER_S = 15 * 60

SOURCE_LABELS = {
    "you": "set by you",
    "send": "from the semester & section it was sent to course design under",
    "pack": "from the pack's course",
    "discussion": "from the discussion print view",
    "folder": "from the submissions folder's name",
    "date": "guessed from the run's date",
}

NO_SEMESTER = "No semester"
NO_COURSE = "No class"


# ── reading semesters, sections and course codes out of text ────────────────

_SEASONS = {"fall": "Fall", "autumn": "Fall", "spring": "Spring",
            "summer": "Summer", "winter": "Winter"}
_SEASON_YEAR = re.compile(r"\b(fall|autumn|spring|summer|winter)\s*(?:term\s*)?['’]?(\d{4}|\d{2})\b",
                          re.I)
_YEAR_SEASON = re.compile(r"\b(\d{4})\s*(fall|autumn|spring|summer|winter)\b", re.I)
_TERM_ABBR = {"F": "Fall", "FA": "Fall", "S": "Spring", "SP": "Spring",
              "SU": "Summer", "SM": "Summer", "WI": "Winter"}
_TERM_ABBR_RE = re.compile(r"^(FA|SP|SU|SM|WI|F|S)(\d{2})$")
_SECTION_WORD = re.compile(r"\bsect(?:ion|\.)?\s*#?\s*([A-Za-z]{0,2}\d{1,3}[A-Za-z]?)\b", re.I)
_SECTION_TOKEN = re.compile(r"^[A-Z]{1,2}\d{2,3}$")
_COURSE_CODE = re.compile(r"\b([A-Z]{2,5})\s*-?\s*(\d{3}[A-Z]?)\b")
_D2L_DOWNLOAD = re.compile(r"\bDownload\s+[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}")


def _year(y: str) -> str:
    return y if len(y) == 4 else f"20{y}"


def parse_semester(text) -> Optional[str]:
    """"W03 Fall 2026" / "2026 Fall" / "BUSL101 W01 F26" -> "Fall 2026"."""
    text = str(text or "")
    m = _SEASON_YEAR.search(text)
    if m:
        return f"{_SEASONS[m.group(1).lower()]} {_year(m.group(2))}"
    m = _YEAR_SEASON.search(text)
    if m:
        return f"{_SEASONS[m.group(2).lower()]} {m.group(1)}"
    for tok in re.findall(r"[A-Za-z0-9]+", text):
        m = _TERM_ABBR_RE.match(tok)
        if m:
            return f"{_TERM_ABBR[m.group(1)]} {_year(m.group(2))}"
    return None


def parse_section(text) -> Optional[str]:
    """The section code a course or folder name carries: "Section 003",
    or a whole token of one or two capitals and two or three digits
    ("W03" in "LGST-203-W03 Fall 2026"). Term abbreviations ("F26") and
    course codes ("BUSL101") are not sections."""
    text = str(text or "")
    m = _SECTION_WORD.search(text)
    if m:
        return m.group(1).upper()
    for tok in re.findall(r"[A-Za-z0-9]+", text):
        if _SECTION_TOKEN.match(tok) and not _TERM_ABBR_RE.match(tok):
            return tok
    return None


def course_code(title) -> str:
    """"BUSL 101 Business Law I rev. 2026" -> "BUSL 101"; "BUSL101 W01 F26"
    -> "BUSL 101"; a title with no course code comes back whole."""
    title = " ".join(str(title or "").split())
    m = _COURSE_CODE.search(title)
    return f"{m.group(1)} {m.group(2)}" if m else title


def _find_course_code(text) -> Optional[str]:
    m = _COURSE_CODE.search(str(text or ""))
    return f"{m.group(1)} {m.group(2)}" if m else None


def semester_from_date(iso) -> Optional[str]:
    try:
        d = datetime.fromisoformat(str(iso or ""))
    except ValueError:
        return None
    season = "Spring" if d.month <= 5 else "Summer" if d.month <= 7 else "Fall"
    return f"{season} {d.year}"


_SEASON_ORDER = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}


def semester_sort_key(label: str) -> tuple:
    """Newest first when sorted in reverse; anything unparsed sorts last."""
    m = re.match(r"^(Winter|Spring|Summer|Fall)\s+(\d{4})$", label or "")
    if not m:
        return (0, 0, label or "")
    return (int(m.group(2)), _SEASON_ORDER[m.group(1)] + 1, "")


# ── the run's filing ─────────────────────────────────────────────────────────

def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def load_filing(outdir) -> dict:
    data = _load_json(Path(outdir) / FILING_FILE)
    return data if isinstance(data, dict) else {}


def normalize_field(field: str, value) -> str:
    """Typed values file with detected ones: "fall 2026" -> "Fall 2026",
    "w03" -> "W03", "busl101" -> "BUSL 101"."""
    v = " ".join(str(value or "").split())[:80]
    if not v:
        return ""
    if field == "semester":
        return parse_semester(v) or v
    if field == "section" and re.fullmatch(r"[A-Za-z]{0,3}\d{1,3}[A-Za-z]?", v):
        return v.upper()
    if field == "course":
        m = re.fullmatch(r"([A-Za-z]{2,5})\s*-?\s*(\d{3}[A-Za-z]?)", v)
        if m:
            return f"{m.group(1).upper()} {m.group(2).upper()}"
    return v


def save_filing(outdir, *, archived: Optional[bool] = None, reset: bool = False,
                **fields) -> dict:
    """Update the run's ``_filing.json``. ``fields`` are semester/course/
    section: a non-blank value sets it, a blank or missing one leaves it as
    it was. ``reset`` drops every field the instructor set, so detection
    takes over again. ``archived`` True/False archives/restores."""
    outdir = Path(outdir)
    filing = load_filing(outdir)
    if reset:
        for f in FIELDS:
            filing.pop(f, None)
    for f in FIELDS:
        v = normalize_field(f, fields.get(f))
        if v:
            filing[f] = v
    if archived is True and not filing.get("archived"):
        filing.update(archived=True,
                      archived_at=datetime.now().isoformat(timespec="minutes"))
    elif archived is False:
        filing.pop("archived", None)
        filing.pop("archived_at", None)
    (outdir / FILING_FILE).write_text(json.dumps(filing, indent=2, ensure_ascii=False))
    return filing


_PACK_CACHE: dict = {}


def _pack_course(pack_path) -> Optional[str]:
    """The pack's course title (assignment or activity pack), cached by
    path and modification time — packs carry whole readings."""
    if not pack_path:
        return None
    p = Path(pack_path)
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return None
    if key not in _PACK_CACHE:
        pack = _load_json(p) or {}
        _PACK_CACHE[key] = (pack.get("course_title") or pack.get("course") or "").strip() or None
    return _PACK_CACHE[key]


def _sent_offering(outdir: Path) -> tuple[Optional[str], Optional[str]]:
    """(semester, section) of the latest course-design send, if any."""
    data = _load_json(outdir / SENT_FILE)
    if not isinstance(data, dict):
        return None, None
    label = (data.get("offering_label") or "").strip()
    if not label:
        return None, None
    sem, _, sec = (part.strip() for part in label.partition("·"))
    return (sem or None), (sec or None)


def _discussion_courses(outdir: Path, manifest: dict) -> list[str]:
    if manifest.get("mode") != "discussion":
        return []
    out = []
    for f in manifest.get("files") or []:
        if not f.get("stem"):
            continue
        disc = (_load_json(outdir / f"{f['stem']}.discussion.json") or {}).get("discussion") or {}
        course = " ".join(str(disc.get("course") or "").split())
        if course and course not in out:
            out.append(course)
    return out


def _folder_names(folder) -> list[str]:
    """The folders the submissions sit in, innermost first — at most three,
    skipping a Brightspace download folder (its name is the assignment)."""
    if not folder:
        return []
    parts = list(Path(str(folder)).parts)
    if parts and _D2L_DOWNLOAD.search(parts[-1]):
        parts = parts[:-1]
    return list(reversed(parts[-3:])) if parts else []


def detect_filing(outdir, manifest: dict) -> dict:
    """``{field: (value, source)}`` for each field detection can fill."""
    outdir = Path(outdir)
    out: dict = {}

    def put(field, value, source):
        if value and field not in out:
            out[field] = (value, source)

    sem, sec = _sent_offering(outdir)
    put("semester", sem, "send")
    put("section", sec, "send")

    course = _pack_course(manifest.get("pack"))
    put("course", course_code(course) if course else None, "pack")

    for text in _discussion_courses(outdir, manifest):
        put("course", _find_course_code(text) or course_code(text), "discussion")
        put("semester", parse_semester(text), "discussion")
        put("section", parse_section(text), "discussion")

    for name in _folder_names(manifest.get("folder")):
        put("semester", parse_semester(name), "folder")
        put("section", parse_section(name), "folder")
        put("course", _find_course_code(name), "folder")

    put("semester", semester_from_date(manifest.get("started_at")), "date")
    return out


def run_filing(outdir, manifest: dict) -> dict:
    """The run's semester/course/section (instructor's choice, else
    detected), where each came from, and whether it is archived."""
    filing = load_filing(outdir)
    detected = detect_filing(outdir, manifest)
    out = {"archived": bool(filing.get("archived")),
           "archived_at": filing.get("archived_at"), "sources": {}}
    for f in FIELDS:
        if filing.get(f):
            out[f], out["sources"][f] = filing[f], "you"
        elif f in detected:
            out[f], out["sources"][f] = detected[f]
        else:
            out[f] = ""
    return out


# ── the Past runs page ───────────────────────────────────────────────────────

def _is_stale(outdir: Path, manifest: dict, live_ids) -> bool:
    if manifest.get("status") not in ("running", "queued"):
        return False
    if manifest.get("run_id") in live_ids:
        return False
    try:
        return time.time() - (outdir / "_run.json").stat().st_mtime > STALE_AFTER_S
    except OSError:
        return True


def is_live(outdir, manifest: dict, live_ids=()) -> bool:
    """Still being written: started from this UI and not finished, or a
    command-line run whose manifest was written in the last few minutes."""
    if manifest.get("run_id") in live_ids:
        return True
    return (manifest.get("status") in ("running", "queued")
            and not _is_stale(Path(outdir), manifest, ()))


def annotate_runs(runs: list[dict], live_ids=()) -> list[dict]:
    """Add what the Past runs page shows about each run (``list_runs``
    order, newest first): ``filing``, ``ok_files``, ``interrupted``,
    ``failed``, ``superseded_by`` (a newer run of the same folder and pack
    that evaluated at least as many files), ``sent`` (a debrief went to
    course design), ``decisions`` (the instructor recorded rubric
    decisions), and ``cleanup`` (failed, interrupted or superseded, and
    neither sent nor decided on — what "Select failed & older duplicates"
    ticks)."""
    newest: dict = {}
    for r in runs:
        outdir = Path(r.get("outdir") or "")
        files = r.get("files") or []
        r["ok_files"] = sum(1 for f in files if f.get("status") == "ok")
        r["live"] = is_live(outdir, r, live_ids)
        r["interrupted"] = (not r["live"]) and r.get("status") in ("running", "queued")
        r["failed"] = (r.get("status") == "failed"
                       or (not r["live"] and bool(files) and r["ok_files"] == 0))
        r["filing"] = run_filing(outdir, r) if outdir.is_dir() else {
            "archived": False, "sources": {}, **{f: "" for f in FIELDS}}
        r["sent"] = (outdir / SENT_FILE).is_file()
        try:
            r["decisions"] = any(outdir.glob("*.overrides.json"))
        except OSError:
            r["decisions"] = False
        key = (r.get("mode") or "submissions", r.get("folder") or "", r.get("pack") or "")
        r["superseded_by"] = None
        if r["live"] or not key[1]:
            continue
        if key in newest and newest[key]["ok_files"] >= r["ok_files"]:
            r["superseded_by"] = newest[key]["run_id"]
        elif r["ok_files"] or key not in newest:
            newest[key] = r
    for r in runs:
        r["cleanup"] = (not r["live"] and not r["sent"] and not r["decisions"]
                        and bool(r["failed"] or r["interrupted"] or r["superseded_by"]))
    return runs


def group_runs(runs: list[dict]) -> list[dict]:
    """``[{"semester", "count", "classes": [{"course", "section", "runs"}]}]``
    — semesters newest first (unfiled last), classes by course then
    section, runs newest first."""
    by_sem: dict = {}
    for r in runs:
        f = r["filing"]
        sem = by_sem.setdefault(f.get("semester") or NO_SEMESTER, {})
        sem.setdefault((f.get("course") or NO_COURSE, f.get("section") or ""), []).append(r)
    out = []
    for sem in sorted(by_sem, key=lambda s: (s != NO_SEMESTER, semester_sort_key(s)),
                      reverse=True):
        classes = [{"course": c, "section": s, "runs": rs}
                   for (c, s), rs in sorted(by_sem[sem].items(),
                                            key=lambda kv: (kv[0][0] == NO_COURSE,
                                                            kv[0][0].lower(),
                                                            kv[0][1] == "", kv[0][1]))]
        out.append({"semester": sem, "classes": classes,
                    "count": sum(len(c["runs"]) for c in classes)})
    return out


def delete_run(results_root, run_id: str) -> None:
    """Remove one run folder. Refuses anything that is not a run folder
    directly inside the results root (ValueError)."""
    root = Path(results_root).resolve()
    if (not run_id or run_id in (".", "..") or "/" in run_id or "\\" in run_id
            or "\x00" in run_id or run_id.startswith("_")):
        raise ValueError(f"not a run: {run_id!r}")
    d = (root / run_id).resolve()
    if d.parent != root or not (d / "_run.json").is_file():
        raise ValueError(f"not a run: {run_id!r}")
    shutil.rmtree(d)
