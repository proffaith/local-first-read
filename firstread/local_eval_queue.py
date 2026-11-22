"""One grading queue across every run in a results folder, and the record
of what the instructor has already graded.

A run folder holds one set of work (one assignment, one section). Facing a
list of sets is the hard part of grading, so the queue flattens them: every
student still waiting, oldest set first, handed over a few at a time.

What is graded lives in each run folder as ``_graded.json``::

    {"marks": {"<file stem>": {"graded": true, "at": "2026-11-10T09:14"},
               "<file stem>": {"graded": true, "at": "...", "bulk": true},
               "<file stem>": {"graded": false, "later": "2026-11-10T09:20"}}}

Marks are keyed by file stem because stems are stable; the student a file
belongs to can change when the instructor regroups the class view. A
student (the unit the queue hands over) is graded when every one of their
files is. ``later`` sets a student aside to the end of the queue.

Running a folder again makes a new run. Its files inherit the marks of the
older runs of the same folder and pack, so a second download of a dropbox
shows only the work that came in since. A mark saved on the newer run wins.

Left out of the queue: archived runs (archiving a set is how the instructor
says "not grading this one by hand"), runs still being written, failed runs
and older duplicates.

A **kept line** is a sentence from a student's paper the instructor wants
to remember (the best line of the set). Kept lines live beside the marks as
``_kept_lines.json`` and never leave the results folder: they are student
text, so they are not part of anything sent to course design.

No DB, no model: run folders only.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

GRADED_FILE = "_graded.json"
KEPT_FILE = "_kept_lines.json"
MAX_KEPT_CHARS = 600
DEFAULT_BATCH = 5
MAX_BATCH = 25


# ── marks on disk ────────────────────────────────────────────────────────────

def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def load_marks(outdir) -> dict:
    """``{stem: mark}`` saved in this run folder (no inheritance)."""
    data = _read_json(Path(outdir) / GRADED_FILE)
    marks = data.get("marks") if isinstance(data, dict) else None
    return {str(k): v for k, v in marks.items() if isinstance(v, dict)} \
        if isinstance(marks, dict) else {}


def _write_marks(outdir, marks: dict) -> None:
    (Path(outdir) / GRADED_FILE).write_text(
        json.dumps({"marks": marks}, indent=2, ensure_ascii=False))


def _now(now=None) -> str:
    return (now or datetime.now()).isoformat(timespec="minutes")


def set_marks(outdir, stems: Iterable[str], state: str, now=None,
              bulk: bool = False) -> dict:
    """Record ``state`` for each stem: ``graded``, ``ungraded`` (an explicit
    "not graded", which also overrides a mark inherited from an older run)
    or ``later``. ``bulk`` marks a whole set at once (work graded before the
    queue existed), which the day's count leaves out. Returns the marks
    these stems had before, for undo (``None`` = no mark of its own)."""
    if state not in ("graded", "ungraded", "later"):
        raise ValueError(f"unknown state: {state!r}")
    marks = load_marks(outdir)
    ts = _now(now)
    before = {}
    for stem in stems:
        before[stem] = marks.get(stem)
        if state == "graded":
            marks[stem] = {"graded": True, "at": ts, **({"bulk": True} if bulk else {})}
        elif state == "later":
            marks[stem] = {"graded": False, "later": ts}
        else:
            marks[stem] = {"graded": False}
    _write_marks(outdir, marks)
    return before


def restore_marks(outdir, before: dict) -> None:
    """Undo a ``set_marks``: put back what each stem had."""
    marks = load_marks(outdir)
    for stem, mark in before.items():
        if mark is None:
            marks.pop(stem, None)
        else:
            marks[stem] = mark
    _write_marks(outdir, marks)


# ── marks across runs of the same folder ─────────────────────────────────────

def run_key(run: dict) -> Optional[tuple]:
    """Runs of the same submissions folder against the same pack are one set
    of work (``local_eval_runs.annotate_runs`` uses the same key). A run
    with no folder stands alone."""
    if not run.get("folder"):
        return None
    return (run.get("mode") or "submissions", run["folder"], run.get("pack") or "")


def effective_marks(runs: list[dict]) -> dict:
    """``{run_id: {stem: mark}}`` — each run's own marks over the marks its
    files inherit from older runs of the same folder and pack. An inherited
    mark carries ``from`` (the run it was saved on)."""
    handed: dict = {}
    out: dict = {}
    for r in sorted(runs, key=lambda r: r.get("started_at") or ""):
        own = load_marks(r.get("outdir") or "")
        key = run_key(r)
        inherited = handed.get(key, {}) if key else {}
        stems = {f.get("stem") for f in r.get("files") or []}
        eff = {s: m for s, m in inherited.items() if s in stems}
        eff.update(own)
        out[r.get("run_id")] = eff
        if key:
            carried = dict(inherited)
            carried.update({s: {**m, "from": m.get("from") or r.get("run_id")}
                            for s, m in own.items()})
            handed[key] = carried
    return out


def hand_down_marks(runs: list[dict], run_id: str) -> Optional[str]:
    """Before a run is deleted: copy the marks it holds (its own and the
    ones it inherited) onto the next newer run of the same folder and pack,
    so deleting an older duplicate never un-grades anything. Returns the
    run that received them."""
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    key = run_key(run) if run else None
    if not key:
        return None
    marks = effective_marks(runs).get(run_id) or {}
    if not marks:
        return None
    newer = sorted((r for r in runs if run_key(r) == key and r.get("run_id") != run_id
                    and (r.get("started_at") or "") >= (run.get("started_at") or "")),
                   key=lambda r: r.get("started_at") or "")
    if not newer:
        return None
    heir = newer[0]
    theirs = load_marks(heir["outdir"])
    merged = {s: {k: v for k, v in m.items() if k != "from"} for s, m in marks.items()}
    merged.update(theirs)
    _write_marks(heir["outdir"], merged)
    return heir.get("run_id")


# ── the units of grading in one run ──────────────────────────────────────────

def _is_graded(mark) -> bool:
    return bool(mark and mark.get("graded"))


def _name_in(filename) -> Optional[str]:
    """The student a Brightspace download names in the file name, for a run
    with no class view yet ("1234-5678 - Rivera, Jordan - Oct 20 - Memo")."""
    from firstread.local_eval_cohort import student_key
    try:
        key = student_key(filename or "", None, None)
    except Exception:  # noqa: BLE001
        return None
    return key["label"] if key.get("method") == "d2l_prefix" else None


def run_units(run: dict, marks: dict) -> list[dict]:
    """What there is to grade in one run, in the run's own order.

    Papers: one unit per student when the class view has paired the files
    (``_students.json``), else one per file. Interview notes: one per
    sheet. Discussions: one per topic. A file the pipeline could not read
    is still a submission, so it stays in the list marked not evaluated.
    An exit ticket (Story 648) has nothing to grade: it is an auto-graded
    extra-credit quiz, read here only for its class-level debrief."""
    mode = run.get("mode") or "submissions"
    if mode == "ticket":
        return []
    files = [f for f in run.get("files") or [] if f.get("stem")]
    label_of: dict = {}
    if mode == "submissions":
        students = _read_json(Path(run.get("outdir") or "") / "_students.json")
        for s in students if isinstance(students, list) else []:
            if not isinstance(s, dict) or not s.get("label"):
                continue
            for f in s.get("files") or []:
                if isinstance(f, dict) and f.get("stem"):
                    label_of[f["stem"]] = s["label"]

    groups: dict = {}
    for f in files:
        paired = label_of.get(f["stem"])
        if mode == "discussion":
            label = f.get("topic") or f.get("name") or f["stem"]
        else:
            label = paired or _name_in(f.get("name")) or f.get("name") or f["stem"]
        g = groups.setdefault(("s", paired) if paired else ("f", f["stem"]),
                              {"label": label, "paired": bool(paired), "files": []})
        g["files"].append(f)

    units = []
    for g in groups.values():
        fs = g["files"]
        ok = [f for f in fs if f.get("status") == "ok"]
        stems = [f["stem"] for f in fs]
        own = [marks.get(s) for s in stems]
        graded = all(_is_graded(m) for m in own)
        ats = [m.get("at") for m in own if m and m.get("at")]
        bulk = any(m.get("bulk") for m in own if m)
        laters = [m.get("later") for m in own if m and m.get("later")]
        if mode == "interview":
            target = ("sheet", fs[0]["stem"])
        elif mode == "discussion":
            target = ("discussion", ok[0]["stem"]) if ok else None
        elif g["paired"] and len(fs) > 1:
            target = ("student", g["label"])
        else:
            target = ("file", ok[0]["stem"]) if ok else None
        flags = [f.get("flags") or {} for f in ok]
        key_total = max((fl.get("key_total") or 0 for fl in flags), default=0)
        levels = next((f.get("levels") for f in ok if f.get("levels")), None) or {}
        units.append({
            "label": g["label"], "stems": stems, "files": fs, "target": target,
            "graded": graded,
            "graded_at": max(ats) if graded and ats else None,
            "bulk": graded and bulk,
            "later": None if graded else (max(laters) if laters else None),
            "evaluated": bool(ok),
            "problems": sum(fl.get("problems") or 0 for fl in flags),
            "history_alerts": sum(fl.get("history_alerts") or 0 for fl in flags),
            "possibly_fabricated": sum(fl.get("possibly_fabricated") or 0 for fl in flags),
            "key_total": key_total or None,
            "key_detected": (max((fl.get("key_detected") or 0 for fl in flags), default=0)
                             if key_total else None),
            "levels": [{"criterion": c, "level": lv} for c, lv in levels.items()],
        })
    return units


def run_title(run: dict) -> str:
    return (run.get("assignment_title") or run.get("pack_title")
            or str(run.get("folder") or run.get("run_id") or "").rstrip("/").rsplit("/", 1)[-1])


# ── the queue ────────────────────────────────────────────────────────────────

def in_queue(run: dict) -> bool:
    """``run`` has been through ``local_eval_runs.annotate_runs``."""
    return not (run.get("live") or run.get("failed") or run.get("superseded_by")
                or (run.get("filing") or {}).get("archived"))


def annotate_graded(runs: list[dict]) -> list[dict]:
    """Add ``units``, ``graded_n`` and ``gradable_n`` to each run (for the
    Past runs page and the queue)."""
    eff = effective_marks(runs)
    for r in runs:
        r["units"] = run_units(r, eff.get(r.get("run_id")) or {})
        r["gradable_n"] = len(r["units"])
        r["graded_n"] = sum(1 for u in r["units"] if u["graded"])
    return runs


def build_queue(runs: list[dict], *, only: Optional[str] = None, today=None) -> dict:
    """The queue over annotated runs (``annotate_runs`` then
    ``annotate_graded``).

    ``sets``: every run in the queue with work left, oldest first.
    ``items``: every ungraded unit — oldest set first, each set in its own
    order, anything set aside for later at the very end. ``only`` narrows
    both to one run. ``done_today`` counts units graded today in any run
    that is not archived."""
    day = (today or datetime.now().date()).isoformat()
    sets, items, later = [], [], []
    done_today = total = 0
    for r in sorted(runs, key=lambda r: r.get("started_at") or ""):
        if not (r.get("filing") or {}).get("archived"):
            done_today += sum(1 for u in r.get("units") or []
                              if u["graded"] and not u["bulk"]
                              and str(u["graded_at"] or "")[:10] == day)
        if not in_queue(r) or (only and r.get("run_id") != only):
            continue
        units = r.get("units") or []
        left = [u for u in units if not u["graded"]]
        total += len(units)
        if not left:
            continue
        filing = r.get("filing") or {}
        info = {"run_id": r.get("run_id"), "title": run_title(r),
                "mode": r.get("mode") or "submissions",
                "course": filing.get("course") or "", "section": filing.get("section") or "",
                "started_at": r.get("started_at"), "left": len(left), "total": len(units)}
        sets.append(info)
        for u in left:
            (later if u["later"] else items).append({**u, "set": info})
    later.sort(key=lambda u: u["later"])
    items += later
    return {"sets": sets, "items": items, "left": len(items), "total": total,
            "done_today": done_today}


def clamp_batch(value, default: int = DEFAULT_BATCH) -> int:
    try:
        return max(1, min(MAX_BATCH, int(value)))
    except (TypeError, ValueError):
        return default


# ── kept lines ───────────────────────────────────────────────────────────────

def load_kept(outdir) -> list[dict]:
    data = _read_json(Path(outdir) / KEPT_FILE)
    lines = data.get("lines") if isinstance(data, dict) else None
    return [x for x in lines if isinstance(x, dict) and x.get("id") and x.get("text")] \
        if isinstance(lines, list) else []


def _write_kept(outdir, lines: list[dict]) -> None:
    (Path(outdir) / KEPT_FILE).write_text(
        json.dumps({"lines": lines}, indent=2, ensure_ascii=False))


def keep_line(outdir, stem: str, text, label: str = "", now=None) -> Optional[dict]:
    """Save one line from a student's paper. Blank text saves nothing."""
    text = " ".join(str(text or "").split())[:MAX_KEPT_CHARS]
    if not text:
        return None
    entry = {"id": uuid.uuid4().hex[:10], "stem": stem, "label": label[:120],
             "text": text, "at": _now(now)}
    _write_kept(outdir, load_kept(outdir) + [entry])
    return entry


def drop_line(outdir, line_id: str) -> bool:
    lines = load_kept(outdir)
    rest = [x for x in lines if x["id"] != line_id]
    if len(rest) == len(lines):
        return False
    _write_kept(outdir, rest)
    return True


def hand_down_kept(runs: list[dict], run_id: str) -> Optional[str]:
    """Before a run is deleted: move its kept lines to the next newer run of
    the same folder and pack, if there is one."""
    run = next((r for r in runs if r.get("run_id") == run_id), None)
    key = run_key(run) if run else None
    mine = load_kept(run["outdir"]) if key else []
    if not mine:
        return None
    newer = sorted((r for r in runs if run_key(r) == key and r.get("run_id") != run_id
                    and (r.get("started_at") or "") >= (run.get("started_at") or "")),
                   key=lambda r: r.get("started_at") or "")
    if not newer:
        return None
    theirs = load_kept(newer[0]["outdir"])
    have = {x["id"] for x in theirs}
    _write_kept(newer[0]["outdir"], theirs + [x for x in mine if x["id"] not in have])
    return newer[0].get("run_id")


def kept_lines(runs: list[dict], today=None) -> list[dict]:
    """Every kept line in the runs that are not archived, newest first, each
    with its run's ``run_id`` and ``title`` and whether it was kept today."""
    day = (today or datetime.now().date()).isoformat()
    out = []
    for r in runs:
        if (r.get("filing") or {}).get("archived"):
            continue
        for x in load_kept(r.get("outdir") or ""):
            out.append({**x, "run_id": r.get("run_id"), "title": run_title(r),
                        "today": str(x.get("at") or "")[:10] == day})
    out.sort(key=lambda x: x.get("at") or "", reverse=True)
    return out
