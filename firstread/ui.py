#!/usr/bin/env python3
"""
Localhost web UI for the local first read (firstread/local_eval.py).

    python -m firstread ui            # http://127.0.0.1:5077

Point it at a folder of submissions. Every docx/pdf/txt/md file is
extracted, PII-scrubbed, citation-checked (does the authority exist, does
it say what the student says it says, is there subsequent appellate
history) and, when an assignment pack is chosen, given a preliminary
rubric read by the local Ollama model.

Binds to 127.0.0.1 only. No cloud model. Runs are one at a time (the local
model is serial anyway) and every run's outputs land in results/<run-id>/
so they survive a restart.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import threading
import webbrowser
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True) or (REPO_ROOT / ".env"))
except ImportError:
    pass

from flask import (Flask, Response, abort, jsonify, redirect,  # noqa: E402
                   render_template, request, url_for)

from firstread.local_eval import (EvalOptions, HISTORY_ALERTS,  # noqa: E402
                              HISTORY_LABELS, PROBLEM_STATUSES,
                              RESULTS_DIR, SOURCE_LABELS, ACADEMIC_LABELS,
                              SOURCE_KIND_LABELS, detect_assignment_title,
                              finding_status, finding_label, is_academic,
                              indexed_line, academic_why, source_flags,
                              discover_packs, list_runs, list_submission_files,
                              load_manifest, load_pack, match_pack, pack_dirs,
                              read_roster, rubric_read, run_folder)
from firstread.local_eval_runs import (FIELDS as FILING_FIELDS,  # noqa: E402
                                   SOURCE_LABELS as FILING_SOURCES,
                                   annotate_runs, delete_run, group_runs,
                                   is_live, normalize_field, run_filing,
                                   save_filing)
from firstread.local_eval_queue import (annotate_graded, build_queue,  # noqa: E402
                                    clamp_batch, effective_marks,
                                    drop_line, hand_down_kept, hand_down_marks,
                                    keep_line, kept_lines, restore_marks, set_marks)
from firstread.local_llm import (DEFAULT_HOST, DEFAULT_MODEL,  # noqa: E402
                             DEFAULT_NUM_CTX, LocalLLMError, list_models)

log = logging.getLogger("local_grade_ui")

RESULTS_ROOT = Path.cwd() / RESULTS_DIR      # default; UI config lives here
_TERMINAL = ("done", "failed", "stopped")


def _results_root() -> Path:
    """Where run folders go. Configurable from the form and remembered, so
    an instructor can point it at a OneDrive/SharePoint folder and keep every
    report (which quotes scrubbed student text) inside the college tenant.
    The UI's own config file stays under the default local root."""
    try:
        custom = (_load_ui_state().get("results_root") or "").strip()
    except Exception:  # noqa: BLE001
        custom = ""
    return Path(custom).expanduser() if custom else RESULTS_ROOT


def _result_path(run_id: str, name: str = "") -> Path:
    """Resolve <results>/<run_id>/<name> or 404. Submission stems carry
    spaces, commas and parentheses ("Rivera, Jordan - Memo 1 (2)"), so the
    guard is path containment, not a character allowlist."""
    for seg in (run_id, name):
        if seg and (seg in (".", "..") or "/" in seg or "\\" in seg
                    or "\x00" in seg):
            abort(404)
    if not run_id:
        abort(404)
    root = _results_root().resolve()
    p = (root / run_id / name).resolve() if name else (root / run_id).resolve()
    if p != root and root not in p.parents:
        abort(404)
    return p

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"),
            static_folder=None)


@app.before_request
def _same_origin_posts():
    """Every POST comes from this UI's own pages. Binding to 127.0.0.1 keeps
    other machines out, not other web pages open in the same browser: a
    page elsewhere could otherwise submit a form here (delete runs). Browsers send Origin on cross-site POSTs; no Origin (the test
    client, curl) is let through."""
    if request.method != "POST":
        return None
    origin = request.headers.get("Origin")
    if origin is None:
        return None
    from urllib.parse import urlsplit
    if origin == "null" or urlsplit(origin).netloc != request.host:
        abort(403)
    return None


# ── run state ────────────────────────────────────────────────────────────────

class RunState:
    def __init__(self, run_id, folder, pack_path, pack, opts, outdir,
                 mode: str = "submissions", runner_kwargs: dict | None = None):
        self.run_id = run_id
        self.mode = mode
        self.runner_kwargs = runner_kwargs or {}
        self.reader = self.runner_kwargs.get("reader") or ""
        self.reader_note = ""
        self.folder = str(folder)
        self.pack_path = pack_path
        self.pack_title = pack.get("title") if pack else None
        self.pack = pack
        self.opts = opts
        self.outdir = Path(outdir)
        self.status = "queued"
        self.error = None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at = None
        self.files = [{"name": p.name, "stem": p.stem, "status": "pending"}
                      for p in list_submission_files(folder)]
        self.current = {}
        self.log = deque(maxlen=300)
        self.stop = False
        self.pack_auto = None
        self.lock = threading.Lock()

    def progress(self, ev):
        with self.lock:
            self.current = ev
            if ev.get("stage") == "reader" and ev.get("message"):
                self.reader_note = ev["message"]
            self.log.append({"ts": datetime.now().strftime("%H:%M:%S"),
                             "file": ev.get("file", ""),
                             "stage": ev.get("stage", ""),
                             "message": ev.get("message", "")})
            if ev.get("stage") == "file_done":
                for f in self.files:
                    if f["name"] == ev.get("file"):
                        f.update(status=ev.get("status"),
                                 flags=ev.get("flags"), levels=ev.get("levels"),
                                 rubric_read=ev.get("rubric_read"),
                                 error=(ev.get("message") if ev.get("status")
                                        not in ("ok", "skipped") else None))
            elif ev.get("file"):
                for f in self.files:
                    if f["name"] == ev.get("file") and f["status"] == "pending":
                        f["status"] = "running"

    def snapshot(self):
        with self.lock:
            done = sum(1 for f in self.files
                       if f["status"] in ("ok", "skipped", "failed", "stopped"))
            return {
                "run_id": self.run_id, "mode": self.mode, "folder": self.folder,
                "pack_title": self.pack_title, "pack_auto": self.pack_auto,
                "model": self.opts.model,
                "reader": self.reader, "reader_note": self.reader_note,
                "options": self.opts.public(), "status": self.status,
                "error": self.error, "started_at": self.started_at,
                "finished_at": self.finished_at,
                "current": dict(self.current), "files": list(self.files),
                "done": done, "total": len(self.files),
                "log": list(self.log)[-40:],
                "outdir": str(self.outdir),
            }


RUNS: dict[str, RunState] = {}
_ACTIVE: dict = {"run": None}
_LOCK = threading.Lock()


def _snapshot_from_manifest(m: dict) -> dict:
    files = m.get("files", [])
    return {
        "run_id": m.get("run_id"), "mode": m.get("mode") or "submissions",
        "folder": m.get("folder"),
        "pack_title": m.get("assignment_title"), "model": m.get("model"),
        "reader": m.get("reader") or "", "reader_note": m.get("reader_note") or "",
        "options": m.get("options", {}), "status": m.get("status", "done"),
        "error": None, "started_at": m.get("started_at"),
        "finished_at": m.get("finished_at"), "current": {},
        "files": files,
        "done": sum(1 for f in files if f.get("status") != "pending"),
        "total": len(files), "log": [], "outdir": m.get("outdir", ""),
        "scholarly": m.get("scholarly") or {},
    }


def _worker(state: RunState):
    state.status = "running"
    runner = run_folder
    try:
        manifest = runner(state.folder, state.pack, state.opts,
                          state.outdir, progress=state.progress,
                          should_stop=lambda: state.stop,
                          run_id=state.run_id, pack_path=state.pack_path,
                          **state.runner_kwargs)
        with state.lock:
            state.files = manifest["files"]
            state.status = manifest["status"]
    except Exception as e:  # noqa: BLE001
        log.exception("run %s failed", state.run_id)
        with state.lock:
            state.status = "failed"
            state.error = str(e)
    finally:
        state.finished_at = datetime.now(timezone.utc).isoformat()
        with _LOCK:
            if _ACTIVE["run"] is state:
                _ACTIVE["run"] = None


# ── ui state (recent folders, last choices) ─────────────────────────────────

def _ui_state_path() -> Path:
    return RESULTS_ROOT / "_ui_state.json"


def _load_ui_state() -> dict:
    try:
        return json.loads(_ui_state_path().read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_ui_state(d: dict):
    try:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        _ui_state_path().write_text(json.dumps(d, indent=2))
    except Exception:  # noqa: BLE001
        log.warning("could not save ui state", exc_info=True)


# ── template helpers ─────────────────────────────────────────────────────────

def badge(status: str) -> str:
    if status in PROBLEM_STATUSES or status in ("reversed", "vacated",
                                                "failed", "error"):
        return "bad"
    if status in ("caution", "unverified", "stopped", "found_mismatch"):
        return "warn"
    if status in ("found_relevant", "good_standing", "none_found", "ok",
                  "done", "skipped", "found"):
        return "ok"
    if status == "running":
        return "run"
    return "muted"


from firstread.intentional_errors import DETECTION_STATUS_LABELS  # noqa: E402
from firstread.expected_moves import MOVE_STATUS_LABELS  # noqa: E402

app.jinja_env.globals.update(
    DETECTION_STATUS_LABELS=DETECTION_STATUS_LABELS,
    MOVE_STATUS_LABELS=MOVE_STATUS_LABELS,
    SOURCE_LABELS=SOURCE_LABELS, HISTORY_LABELS=HISTORY_LABELS,
    PROBLEM_STATUSES=PROBLEM_STATUSES, HISTORY_ALERTS=HISTORY_ALERTS,
    badge=badge, finding_status=finding_status, DEFAULT_MODEL=DEFAULT_MODEL, DEFAULT_NUM_CTX=DEFAULT_NUM_CTX,
    ACADEMIC_LABELS=ACADEMIC_LABELS, SOURCE_KIND_LABELS=SOURCE_KIND_LABELS,
    finding_label=finding_label, is_academic=is_academic,
    indexed_line=indexed_line, academic_why=academic_why,
)


def scholarly_key_status() -> dict:
    """The Checks section's key line (spec §3.1): booleans only, never a key."""
    from firstread import semantic_scholar_http as ssh
    return {"ss_key": bool(ssh.api_key()),
            "openalex_key": bool((os.getenv("OPENALEX_API_KEY") or "").strip())}


app.jinja_env.globals.update(scholarly_key_status=scholarly_key_status)


@app.template_filter("local_when")
def _local_when(iso):
    """A stored UTC timestamp in this machine's time: "Oct 22, 2026 6:44 AM"."""
    try:
        d = datetime.fromisoformat(str(iso or "")).astimezone()
    except ValueError:
        return iso or ""
    return f"{d:%b} {d.day}, {d.year} {d.hour % 12 or 12}:{d:%M} {d:%p}"


def _model_status():
    try:
        # Embedding models can't chat; keep them out of the picker.
        return [m for m in list_models(DEFAULT_HOST)
                if "embed" not in m.lower()], None
    except LocalLLMError as e:
        return [], str(e)


def _index(error=None, status=200, form=None):
    models, ollama_error = _model_status()
    state = _load_ui_state()
    active = _ACTIVE["run"]
    return render_template(
        "local_eval/index.html", packs=discover_packs(), models=models,
        ollama_error=ollama_error, error=error, form=form or state.get("last", {}),
        recent=state.get("recent_folders", []),
        active=active.snapshot() if active else None,
        results_root=str(_results_root()), host=DEFAULT_HOST,
        pack_dirs=[str(d) for d in pack_dirs()],
    ), status


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return _index()


@app.get("/detect")
def detect():
    """Which assignment is this folder? → detected title + best local pack."""
    folder = Path((request.args.get("folder") or "").strip()).expanduser()
    if not str(folder) or not folder.is_dir():
        return jsonify({"error": "not a folder"}), 400
    det = detect_assignment_title(folder)
    m = match_pack(det["title"])
    files = list_submission_files(folder)
    return jsonify({"title": det["title"], "source": det["source"],
                    "files": len(files),
                    "match": ({"path": m["best"]["path"],
                               "assignment_id": m["best"]["assignment_id"],
                               "title": m["best"]["title"],
                               "has_key": m["best"].get("has_key"),
                               "score": m["score"], "confident": m["confident"]}
                              if m["best"] else None),
                    "suggestions": m["suggestions"]})


@app.post("/pick-folder")
def pick_folder():
    """Native folder picker (macOS only — browsers cannot hand a page a
    folder path)."""
    if platform.system() != "Darwin":
        return jsonify({"error": "native picker only on macOS; type the path"}), 400
    try:
        out = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt '
                                '"Folder of submissions to evaluate")'],
            capture_output=True, text=True, timeout=300)
        path = out.stdout.strip().rstrip("/")
        if not path:
            return jsonify({"error": "cancelled"}), 400
        return jsonify({"path": path})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


def _bool(name):
    return request.form.get(name) in ("on", "1", "true", "yes")


@app.post("/start")
def start():
    form = {k: v for k, v in request.form.items()}
    folder = Path((form.get("folder") or "").strip()).expanduser()
    if not str(folder) or not folder.is_dir():
        return _index(f"Not a folder: {folder}", 400, form)
    mode = "submissions"
    runner_kwargs = {}
    files = list_submission_files(folder)
    if not files:
        return _index("No docx/pdf/txt/md files in that folder.", 400, form)

    pack_path = (form.get("pack") or "").strip()
    pack = None
    pack_auto = None
    if pack_path == "none":
        pack_path = ""                       # explicit citations-only
    elif not pack_path:
        # Auto-detect: the folder name / index.html names the assignment.
        det = detect_assignment_title(folder)
        m = match_pack(det["title"])
        if m["confident"] and m["best"]:
            pack_path = m["best"]["path"]
            pack_auto = {"title": det["title"], "source": det["source"],
                         "score": m["score"]}
        else:
            hint = (f" Closest local pack: {m['best']['assignment_id']} "
                    f"“{m['best']['title']}” (score {m['score']}) — pick it "
                    "in the list if that is right." if m["best"] else
                    " No local pack matches — add one to packs/ (see "
                    "packs/example-memo.json), or choose “citations only”.")
            return _index(f"This looks like “{det['title']}” ({det['source']}), "
                          f"but no pack was chosen.{hint}", 400, form)
    if pack_path and pack is None:
        try:
            pack = load_pack(pack_path)
        except Exception as e:  # noqa: BLE001
            return _index(f"Could not read pack {pack_path}: {e}", 400, form)

    roster = (form.get("roster") or "").strip()
    try:
        identities = read_roster(Path(roster).expanduser()) if roster else []
    except Exception as e:  # noqa: BLE001
        return _index(f"Could not read roster {roster}: {e}", 400, form)

    try:
        opts = EvalOptions(
            model=(form.get("model") or DEFAULT_MODEL).strip(),
            num_ctx=int(form.get("num_ctx") or DEFAULT_NUM_CTX),
            with_review=_bool("with_review"),
            think=_bool("think"),
            check_citations=_bool("check_citations"),
            check_history=_bool("check_history"),
            max_history_checks=int(form.get("max_history_checks") or 3),
            check_sources=_bool("check_sources"),
            support_reads=_bool("support_reads"),
            refresh_sources=_bool("refresh_sources"),
            identities=identities,
        )
    except ValueError as e:
        return _index(f"Bad option: {e}", 400, form)
    if mode == "submissions" and not pack and not opts.check_citations:
        return _index("Nothing to do: choose a pack and/or keep the "
                      "citation check on.", 400, form)

    root_field = (form.get("results_root") or "").strip()
    root = Path(root_field).expanduser() if root_field else RESULTS_ROOT
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not os.access(root, os.W_OK):
            raise PermissionError("not writable")
    except Exception as e:  # noqa: BLE001
        return _index(f"Results folder is not usable: {root} ({e})", 400, form)
    if root.resolve() == folder.resolve():
        return _index("Results folder must not be the submissions folder "
                      "itself (use a subfolder).", 400, form)

    with _LOCK:
        if _ACTIVE["run"] is not None:
            return _index("A run is already in progress — wait for it to "
                          "finish or stop it.", 409, form)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = re.sub(r"[^\w.-]+", "_", folder.name)[:40] or "run"
        run_id = f"{stamp}-{label}"
        ui = _load_ui_state()
        ui["results_root"] = str(root) if root_field else ""
        _save_ui_state(ui)          # before RunState so routes resolve here
        state = RunState(run_id, folder, pack_path, pack, opts,
                         root / run_id, mode=mode, runner_kwargs=runner_kwargs)
        state.pack_auto = pack_auto
        RUNS[run_id] = state
        _ACTIVE["run"] = state

    recent = [str(folder)] + [r for r in ui.get("recent_folders", [])
                              if r != str(folder)]
    ui["recent_folders"] = recent[:8]
    form.pop("folder", None)
    ui["last"] = form
    _save_ui_state(ui)

    threading.Thread(target=_worker, args=(state,), daemon=True,
                     name=f"local-eval-{run_id}").start()
    return redirect(url_for("run_page", run_id=run_id))


def _get_snapshot(run_id):
    outdir = _result_path(run_id)
    state = RUNS.get(run_id)
    if state:
        snap = state.snapshot()
        if snap["status"] in _TERMINAL:      # the rate-limit banner lives in the manifest
            snap["scholarly"] = (load_manifest(outdir) or {}).get("scholarly") or {}
        return snap
    m = load_manifest(outdir)
    if not m:
        abort(404)
    m["outdir"] = str(outdir)
    return _snapshot_from_manifest(m)


def _load_json(run_id, name):
    p = _result_path(run_id, name)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


@app.get("/run/<run_id>")
def run_page(run_id):
    snap = _get_snapshot(run_id)
    live = snap["status"] not in _TERMINAL
    students = cohort = None
    if not live:
        students = _load_json(run_id, "_students.json")
        cohort = _load_json(run_id, "_cohort.json")
        _with_rubric_reads(run_id, snap)
        marks = effective_marks(list_runs(_results_root())).get(run_id) or {}
        snap["files"] = [{**f, "graded": bool((marks.get(f.get("stem")) or {}).get("graded"))}
                         for f in snap["files"]]
    paired = bool(students) and (
        any(len(s["files"]) > 1 for s in students)
        or any(s.get("method") != "file" for s in students))
    # file name → current student label, for the regroup form
    labels = {}
    for s in students or []:
        for f in s["files"]:
            labels[f["name"]] = s["label"]
    return render_template("local_eval/run.html", snap=snap, run_id=run_id,
                           live=live, can_stop=run_id in RUNS,
                           students=students, paired=paired, cohort=cohort,
                           labels=labels, filing=_filing_for(run_id))


def _with_rubric_reads(run_id, snap):
    """A run made before the run page showed the model's reasons keeps only
    criterion -> level in its manifest: read each file's report for them
    (once per page view; new runs carry ``rubric_read`` already)."""
    missing = [f for f in snap["files"]
               if f.get("status") == "ok" and f.get("rubric_read") is None]
    if not missing:
        return snap
    names = (_pack_for_run(run_id) or {}).get("level_names") or []
    for f in missing:
        try:
            res = json.loads(_result_path(run_id, f"{f['stem']}.json").read_text())
            rows = (res.get("result") or {}).get("rubric")
        except Exception:  # noqa: BLE001 — report gone: levels only
            rows = None
        if rows is None:
            rows = [{"criterion": c, "suggested_level": lv}
                    for c, lv in (f.get("levels") or {}).items()]
        f["rubric_read"] = rubric_read(rows, names)
    return snap


def _filing_for(run_id):
    m = load_manifest(_result_path(run_id))
    return run_filing(_result_path(run_id), m) if m else None


@app.get("/run/<run_id>/student/<label>")
def student_page(run_id, label):
    students = _load_json(run_id, "_students.json")
    if students is None:
        # a run from before the class view existed: group on the fly
        # a run with no class view yet: build it from the run page first
        return redirect(url_for("run_page", run_id=run_id) + "#class")
    students = students or []
    s = next((x for x in students if x["label"] == label), None)
    if not s:
        abort(404)
    key = None
    m = load_manifest(_result_path(run_id))
    if m and m.get("pack"):
        try:
            key = {e["id"]: e for e in (load_pack(m["pack"])
                                        .get("intentional_errors") or {})
                   .get("planted_errors") or []}
        except Exception:  # noqa: BLE001
            key = None
    return render_template("local_eval/student.html", s=s, run_id=run_id,
                           key=key or {})


@app.post("/run/<run_id>/pairing")
def save_pairing(run_id):
    """Manual grouping: one 'student' label per file (blank = automatic).
    Saves _pairing.json in the run folder and rebuilds the class view."""
    outdir = _result_path(run_id)
    m = load_manifest(outdir)
    if not m or m.get("status") in ("running", "queued"):
        abort(404)
    names = {f["name"] for f in m.get("files", [])}
    pairing = {}
    for k, v in request.form.items():
        if k.startswith("label:"):
            name = k[len("label:"):]
            if name in names and v.strip():
                pairing[name] = v.strip()[:120]
    from firstread.local_eval_cohort import PAIRING_FILE
    (outdir / PAIRING_FILE).write_text(json.dumps(pairing, indent=2,
                                                  ensure_ascii=False))
    return rebuild_class_view(run_id)


@app.post("/run/<run_id>/rebuild-class-view")
def rebuild_class_view(run_id):
    """(Re)build the class view for a finished run from the files on disk —
    for runs made before the class view existed, after editing a pack, or
    after the instructor regroups files by student."""
    outdir = _result_path(run_id)
    m = load_manifest(outdir)
    if not m or m.get("status") in ("running", "queued"):
        abort(404)
    pack = None
    if m.get("pack"):
        try:
            pack = load_pack(m["pack"])
        except Exception:  # noqa: BLE001
            pack = None
    from firstread.local_eval_cohort import build_cohort
    opts = EvalOptions(model=m.get("model") or DEFAULT_MODEL)
    with _LOCK:
        if _ACTIVE["run"] is not None:
            return _index("A run is in progress — build the class view when "
                          "it finishes.", 409)
    cohort = build_cohort(outdir, m, pack, opts)
    m["cohort"] = {"students": len(cohort["students"]),
                   "brief": bool(cohort.get("brief")),
                   "errors": cohort.get("errors") or []}
    (outdir / "_run.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    return redirect(url_for("run_page", run_id=run_id) + "#class")


@app.get("/run/<run_id>/next-class")
def next_class_md(run_id):
    p = _result_path(run_id, "_next_class.md")
    if not p.is_file():
        abort(404)
    return Response(p.read_text(), mimetype="text/plain; charset=utf-8")




@app.get("/run/<run_id>/status")
def run_status(run_id):
    return jsonify(_get_snapshot(run_id))


@app.post("/run/<run_id>/stop")
def run_stop(run_id):
    state = RUNS.get(run_id)
    if state:
        state.stop = True
        state.progress({"stage": "stop", "message":
                        "stop requested — finishing the current file"})
    return redirect(url_for("run_page", run_id=run_id))


def _load_result(run_id, stem):
    p = _result_path(run_id, f"{stem}.json")
    if not p.is_file():
        abort(404)
    return json.loads(p.read_text())


def _pack_for_run(run_id):
    """The assignment pack a run was read against, if it can still be read."""
    m = load_manifest(_result_path(run_id)) or {}
    if m.get("pack"):
        try:
            return load_pack(m["pack"])
        except Exception:  # noqa: BLE001
            return None
    return None


@app.get("/run/<run_id>/file/<stem>")
def file_page(run_id, stem):
    from firstread.local_eval_calibration import (load_overrides, load_calibration,
                                              agreement_stats)
    from firstread.intentional_errors import DETECTION_STATUSES
    res = _load_result(run_id, stem)
    if (res.get("source") or {}).get("checked"):
        # Recount from the findings so runs made before "unverifiable"
        # existed don't still report statutes / WL numbers as problems.
        res["flags"] = {**(res.get("flags") or {}), **source_flags(res["source"])}
    pack = _pack_for_run(run_id)
    level_names = (pack or {}).get("level_names") or sorted(
        {r.get("suggested_level") for r in ((res.get("result") or {}).get("rubric") or [])
         if r.get("suggested_level")})
    overrides = load_overrides(_result_path(run_id), stem)
    stats = {}
    if pack:
        entries = load_calibration(_results_root(), pack.get("course_id"))
        stats = agreement_stats(entries, level_names)
    return render_template("local_eval/file.html", res=res, run_id=run_id,
                           stem=stem, level_names=level_names,
                           overrides=overrides, stats=stats, pack=pack,
                           DETECTION_STATUSES=DETECTION_STATUSES)


@app.post("/run/<run_id>/file/<stem>/overrides")
def save_overrides(run_id, stem):
    """Instructor confirms/overrides suggested levels and planted-error
    statuses; recorded for this file and in the course calibration log."""
    from firstread.local_eval_calibration import record_overrides
    res = _load_result(run_id, stem)
    pack = _pack_for_run(run_id)
    if not pack:
        return _index("This run has no pack, so there is no rubric to "
                      "calibrate against.", 400)
    form = {"rubric": {}, "planted": {}}
    for k, v in request.form.items():
        if k.startswith("level:"):
            crit = k[len("level:"):]
            form["rubric"].setdefault(crit, {})["level"] = v
        elif k.startswith("reason:"):
            crit = k[len("reason:"):]
            form["rubric"].setdefault(crit, {})["reason"] = v
        elif k.startswith("pstatus:"):
            eid = k[len("pstatus:"):]
            form["planted"].setdefault(eid, {})["status"] = v
        elif k.startswith("preason:"):
            eid = k[len("preason:"):]
            form["planted"].setdefault(eid, {})["reason"] = v
    record_overrides(_results_root(), pack, run_id, _result_path(run_id), stem,
                     res, form)
    return redirect(url_for("file_page", run_id=run_id, stem=stem) + "#decisions")


def _pack_by_id(assignment_id):
    for p in discover_packs():
        if str(p.get("assignment_id")) == str(assignment_id):
            return p["path"], load_pack(p["path"])
    abort(404)


@app.get("/packs")
def packs_page():
    return render_template("local_eval/packs.html", packs=discover_packs(),
                           pack_dirs=[str(d) for d in pack_dirs()],
                           error=request.args.get("error"), notice=request.args.get("notice"))


@app.get("/packs/<assignment_id>/moves")
def moves_page(assignment_id):
    from firstread.expected_moves import moves_to_text
    path, pack = _pack_by_id(assignment_id)
    doc = pack.get("expected_moves") or {}
    text = moves_to_text(doc.get("moves") or []) if doc else ""
    draft = request.args.get("draft") == "1"
    return render_template("local_eval/moves.html", pack=pack, pack_path=path,
                           text=text, doc=doc, error=None,
                           models=_model_status()[0], drafted=draft)


@app.post("/packs/<assignment_id>/moves")
def moves_save(assignment_id):
    from firstread.expected_moves import parse_moves_text, save_moves, moves_to_text
    path, pack = _pack_by_id(assignment_id)
    action = request.form.get("action") or "save"
    text = request.form.get("moves_text") or ""
    if action == "draft":
        with _LOCK:
            if _ACTIVE["run"] is not None:
                return render_template("local_eval/moves.html", pack=pack,
                                       pack_path=path, text=text, doc=None,
                                       error="A run is in progress — draft "
                                             "when it finishes.",
                                       models=_model_status()[0], drafted=False), 409
        from firstread.local_eval import draft_expected_moves
        opts = EvalOptions(model=(request.form.get("model") or DEFAULT_MODEL))
        try:
            moves = draft_expected_moves(pack, opts)
        except LocalLLMError as e:
            return render_template("local_eval/moves.html", pack=pack,
                                   pack_path=path, text=text, doc=None,
                                   error=str(e), models=_model_status()[0],
                                   drafted=False), 502
        # Show the draft in the editor; the instructor saves it explicitly.
        return render_template("local_eval/moves.html", pack=pack, pack_path=path,
                               text=moves_to_text(moves), doc=None, error=None,
                               models=_model_status()[0], drafted=True)
    moves = parse_moves_text(text)
    if not moves:
        from firstread.expected_moves import moves_path
        p = moves_path(path)
        if p.exists():
            p.unlink()
        return redirect(url_for("packs_page"))
    save_moves(path, moves, drafted_by="instructor")
    return redirect(url_for("moves_page", assignment_id=assignment_id))


@app.get("/calibration")
def calibration_page():
    from firstread.local_eval_calibration import (list_courses, load_calibration,
                                              agreement_stats)
    root = _results_root()
    courses = list_courses(root)
    cid = request.args.get("course") or (courses[0]["course_id"] if courses else None)
    entries = load_calibration(root, cid) if cid else []
    # level order: from any pack of that course we can find
    level_names = []
    for p in discover_packs():
        if str(p.get("assignment_id")) and cid:
            try:
                pk = load_pack(p["path"])
            except Exception:  # noqa: BLE001
                continue
            if str(pk.get("course_id")) == str(cid) and pk.get("level_names"):
                level_names = pk["level_names"]
                break
    stats = agreement_stats(entries, level_names)
    return render_template("local_eval/calibration.html", courses=courses,
                           course_id=cid, entries=entries[:50], stats=stats,
                           level_names=level_names, results_root=str(root))


@app.get("/run/<run_id>/md/<stem>")
def file_md(run_id, stem):
    p = _result_path(run_id, f"{stem}.md")
    if not p.is_file():
        abort(404)
    return Response(p.read_text(), mimetype="text/plain; charset=utf-8")


@app.get("/run/<run_id>/txt/<kind>/<stem>")
def file_txt(run_id, stem, kind):
    """The scrubbed text the model saw (``text``) or the tracked-changes
    view (``redline``) — so extraction problems are visible, not guessed."""
    suffix = {"text": ".txt", "redline": ".redline.txt"}.get(kind)
    if not suffix:
        abort(404)
    p = _result_path(run_id, f"{stem}{suffix}")
    if not p.is_file():
        abort(404)
    return Response(p.read_text(), mimetype="text/plain; charset=utf-8")


def _live_ids():
    return {rid for rid, st in RUNS.items() if st.status not in _TERMINAL}


@app.get("/runs")
def runs_page():
    root = _results_root()
    runs = list_runs(root)
    # Merge in-memory (live) runs that have not finished writing yet.
    known = {r.get("run_id") for r in runs}
    for rid, st in RUNS.items():
        if rid not in known and Path(st.outdir).parent == root:
            runs.insert(0, st.snapshot())
    annotate_runs(runs, _live_ids())
    annotate_graded(runs)
    archived = request.args.get("tab") == "archived"
    shown = [r for r in runs if r["filing"]["archived"] == archived]
    known_values = {f: sorted({r["filing"][f] for r in runs if r["filing"][f]},
                              key=str.lower) for f in FILING_FIELDS}
    return render_template("local_eval/runs.html", groups=group_runs(shown),
                           archived=archived, results_root=str(root),
                           n_active=sum(1 for r in runs if not r["filing"]["archived"]),
                           n_archived=sum(1 for r in runs if r["filing"]["archived"]),
                           n_cleanup=sum(1 for r in shown if r["cleanup"]),
                           known_values=known_values, sources=FILING_SOURCES,
                           note=request.args.get("note", "")[:300])


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


@app.post("/runs/organize")
def runs_organize():
    """The Past runs action bar: archive, restore, file under a semester /
    class / section, go back to the detected filing, or delete — for every
    ticked run. Runs still being written are never deleted."""
    root = _results_root()
    action = request.form.get("action", "")
    tab = "archived" if request.form.get("tab") == "archived" else None
    picked = list(dict.fromkeys(request.form.getlist("run_id")))
    if action not in ("archive", "unarchive", "file", "reset", "delete") or not picked:
        return redirect(url_for("runs_page", tab=tab, note="Tick at least one run first."))
    live_ids = _live_ids()
    done, unknown, running = 0, 0, 0
    for rid in picked:
        outdir = root / rid
        m = load_manifest(outdir) if "/" not in rid and "\\" not in rid else None
        if not m or outdir.resolve().parent != root.resolve():
            unknown += 1
            continue
        m["run_id"] = rid
        if action == "delete":
            if is_live(outdir, m, live_ids):
                running += 1
                continue
            hand_down_marks(list_runs(root), rid)
            hand_down_kept(list_runs(root), rid)
            delete_run(root, rid)
            RUNS.pop(rid, None)
        elif action == "file":
            save_filing(outdir, **{f: request.form.get(f, "") for f in FILING_FIELDS})
        else:
            save_filing(outdir, archived={"archive": True, "unarchive": False}.get(action),
                        reset=action == "reset")
        done += 1
    runs = _plural(done, "run")
    note = {"archive": f"Archived {runs}.", "unarchive": f"Restored {runs} to the list.",
            "reset": f"{runs} back to the detected semester, class and section.",
            "delete": f"Deleted {runs}.",
            "file": f"Filed {runs} under "
                    + (" · ".join(v for v in (normalize_field(f, request.form.get(f))
                                               for f in FILING_FIELDS) if v) or "nothing new")
                    + "."}[action]
    if running:
        note += f" Kept {_plural(running, 'run')} still being written."
    if unknown:
        note += f" Skipped {_plural(unknown, 'run')} no longer in the results folder."
    return redirect(url_for("runs_page", tab=tab, note=note[:1].upper() + note[1:]))


# ── the grading queue (utils/local_eval_queue.py) ───────────────────────────

_LAST_MARK: dict = {}        # the latest mark, for Undo (one instructor, one machine)


def _queue_runs():
    runs = list_runs(_results_root())
    annotate_runs(runs, _live_ids())
    return annotate_graded(runs)


def _unit_href(run_id, target):
    """Where a queue item opens: the file report or the student's page
    (several files)."""
    if not target:
        return None
    kind, value = target
    if kind == "student":
        return url_for("student_page", run_id=run_id, label=value)
    if kind != "file":
        return None
    return url_for("file_page", run_id=run_id, stem=value)


def graded_state(run_id, stems):
    """For a report page's graded bar: ``{"graded", "at"}`` for these files
    of one run, counting marks inherited from older runs of the folder."""
    marks = effective_marks(list_runs(_results_root())).get(run_id) or {}
    own = [marks.get(s) or {} for s in stems]
    graded = bool(own) and all(m.get("graded") for m in own)
    return {"graded": graded,
            "at": max((m.get("at") or "" for m in own), default="") if graded else ""}


def run_graded_counts(run_id):
    """``(graded, to grade)`` for one run's page."""
    run = next((r for r in _queue_runs() if r.get("run_id") == run_id), None)
    return (run["graded_n"], run["gradable_n"]) if run else (0, 0)


app.jinja_env.globals.update(unit_href=_unit_href, graded_state=graded_state,
                             run_graded_counts=run_graded_counts)


@app.get("/queue")
def queue_page():
    """Every student still to grade, across every active run, a few at a
    time: oldest set first."""
    state = _load_ui_state()
    if request.args.get("n"):
        n = clamp_batch(request.args.get("n"))
        if n != state.get("queue_batch"):
            state["queue_batch"] = n
            _save_ui_state(state)
    else:
        n = clamp_batch(state.get("queue_batch"))
    runs = _queue_runs()
    only = request.args.get("set") or None
    whole = build_queue(runs)
    q = build_queue(runs, only=only) if only else whole
    focus = next((s for s in whole["sets"] if s["run_id"] == only), None) if only else None
    kept = kept_lines(runs)
    goal = _queue_goal(state)
    return render_template("local_eval/queue.html", q=q, whole=whole, batch=q["items"][:n],
                           n=n, only=only, focus=focus, kept=kept,
                           goal=goal, reward=(state.get("queue_reward") or "")[:120],
                           goal_met=bool(goal) and whole["done_today"] >= goal,
                           kept_today=sum(1 for x in kept if x["today"]),
                           all_lines=request.args.get("lines") == "all",
                           undo=_LAST_MARK if _LAST_MARK.get("run_id") else None,
                           note=request.args.get("note", "")[:300],
                           results_root=str(_results_root()))


def _safe_back(path):
    """A page of this UI to go back to, or None."""
    path = (path or "").strip()
    return path if path.startswith("/") and not path.startswith("//") and "\\" not in path else None


@app.post("/queue/mark")
def queue_mark():
    """Record that a student's work (one or more files of one run) is
    graded, not graded after all, or set aside for later — or, for a whole
    run, that everything left in it is graded."""
    run_id = request.form.get("run_id", "")
    outdir = _result_path(run_id)
    m = load_manifest(outdir)
    if not m:
        abort(404)
    known = [f["stem"] for f in m.get("files", []) if f.get("stem")]
    action = request.form.get("action", "")
    state = {"graded": "graded", "rest": "graded", "ungraded": "ungraded",
             "later": "later"}.get(action)
    if not state:
        abort(400)
    if action == "rest":
        marks = effective_marks(list_runs(_results_root())).get(run_id) or {}
        stems = [s for s in known if not (marks.get(s) or {}).get("graded")]
    else:
        stems = [s for s in dict.fromkeys(request.form.getlist("stem")) if s in known]
    label = " ".join((request.form.get("label") or "").split())[:120]
    if stems:
        before = set_marks(outdir, stems, state, bulk=action == "rest")
        _LAST_MARK.clear()
        _LAST_MARK.update(run_id=run_id, before=before, label=label, action=action)
    what = label or _plural(len(stems), "file")
    note = ("Nothing to change." if not stems else
            {"graded": f"Marked {what} graded.",
             "rest": f"Marked the remaining {_plural(len(stems), 'file')} of {what} graded.",
             "ungraded": f"{what} is back in the queue.",
             "later": f"Set {what} aside to the end of the queue."}[action])
    back = None if request.args.get("next") == "queue" else _safe_back(request.form.get("back"))
    if back and not back.startswith(url_for("queue_page")):
        return redirect(back)
    return redirect(url_for("queue_page", set=request.form.get("set") or None, note=note))


def _queue_goal(state) -> int:
    try:
        return max(0, min(200, int(state.get("queue_goal") or 0)))
    except (TypeError, ValueError):
        return 0


@app.post("/queue/goal")
def queue_goal():
    """Today's goal (students to grade in a day; 0 turns it off) and the
    reward the instructor names for reaching it. Both live in the UI's own
    state file on this machine."""
    state = _load_ui_state()
    state["queue_goal"] = _queue_goal({"queue_goal": request.form.get("goal")})
    state["queue_reward"] = " ".join((request.form.get("reward") or "").split())[:120]
    _save_ui_state(state)
    return redirect(url_for("queue_page", set=request.form.get("set") or None))


@app.post("/queue/keep")
def queue_keep():
    """Keep a line from a student's paper (stays in the run folder), or drop
    one kept earlier."""
    run_id = request.form.get("run_id", "")
    outdir = _result_path(run_id)
    m = load_manifest(outdir)
    if not m:
        abort(404)
    back = _safe_back(request.form.get("back"))
    if request.form.get("drop"):
        drop_line(outdir, request.form["drop"])
        return redirect(back or url_for("queue_page", note="Dropped that line."))
    stem = request.form.get("stem", "")
    if stem not in {f.get("stem") for f in m.get("files", [])}:
        abort(400)
    label = " ".join((request.form.get("label") or "").split())
    kept = keep_line(outdir, stem, request.form.get("text"), label)
    if back:
        return redirect(back + ("#kept" if kept else ""))
    return redirect(url_for("queue_page", note="Kept." if kept else "Nothing to keep."))


def kept_for(run_id, stems):
    """The lines kept from these files of one run, for the report page."""
    from firstread.local_eval_queue import load_kept
    return [x for x in load_kept(_result_path(run_id)) if x.get("stem") in stems]


app.jinja_env.globals.update(kept_for=kept_for)


@app.post("/queue/undo")
def queue_undo():
    last = dict(_LAST_MARK)
    _LAST_MARK.clear()
    note = "Nothing to undo."
    if last.get("run_id"):
        outdir = _result_path(last["run_id"])
        if outdir.is_dir():
            restore_marks(outdir, last["before"])
            note = f"Undone{': ' + last['label'] if last.get('label') else ''}."
    return redirect(url_for("queue_page", set=request.form.get("set") or None, note=note))


# ── entry point ──────────────────────────────────────────────────────────────

def serve(port: int = 5077, open_browser: bool = True):
    url = f"http://127.0.0.1:{port}/"
    print(f"Local first read: {url}  (results in {_results_root()}/)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True,
            use_reloader=False)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=5077)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
