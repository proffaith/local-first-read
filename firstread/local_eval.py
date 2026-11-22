"""
Local-only evaluation pipeline, shared by firstread/cli.py (CLI) and
firstread/ui.py (localhost web UI).

Per submission file:

  1. extract text (docx / pdf / txt / md)          local
  2. firstread.pii_gate.scrub                          local
  3. citation check
       extract citations (eyecite)                 local
       existence lookup                            CourtListener / Cornell LII
                                                   (citation identifiers ONLY)
       academic sources (reference list, notes,   local extraction;
       DOIs — utils/academic_citations)           Semantic Scholar / OpenAlex /
                                                   doi.org get a DOI, or the
                                                   title + year, ONLY
       "says what the student says it says"        localhost model, using the
                                                   production relevance prompt
       subsequent appellate history                CourtListener forward cites
                                                   + localhost model disposition
  4. rubric read (only when an assignment pack is given)
                                                   localhost model, production
                                                   grading prompt
  5. <stem>.json + <stem>.md, _summary.md, _run.json, _emissions.jsonl

Student writing never leaves this machine. The only outbound traffic is the
citation identifiers sent to the research services: volume/reporter/page,
party names and year for cases and statutes — the same boundary production
draws in firstread.pii_gate.audit_citation_emissions — and, for academic
sources, a DOI / arXiv / PubMed id or the source's title and year (author
names are compared locally, never sent; see utils/academic_lookup).
firstread.local_llm refuses any non-localhost model host by design.

Everything produced here is labelled PRELIMINARY: a triage draft for the
instructor, not a grade.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from firstread.local_llm import (chat_json, list_models, LocalLLMError,
                             DEFAULT_MODEL, DEFAULT_HOST, DEFAULT_NUM_CTX)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKS_DIR = "packs"
RESULTS_DIR = "results"
SUPPORTED_EXTS = (".docx", ".pdf", ".txt", ".md")
CITATION_CAP = 25
SOURCE_CAP = 40

ProgressFn = Callable[[dict], None]


# ── options ──────────────────────────────────────────────────────────────────

@dataclass
class EvalOptions:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = 1.0          # gemma degenerates below ~0.7 w/ JSON
    think: bool = False               # gemma4 thinking mode: slow, and it
                                      # can exhaust num_predict → empty reply
    with_review: bool = False         # stage-1 formative review (slow)
    with_brief: bool = True           # next-class brief from the class view
    calibration_root: str = ""        # where _calibration/ lives (default:
                                      # the results root); "" = results root
    use_calibration: bool = True      # inject past instructor decisions
    check_citations: bool = True
    check_history: bool = True        # subsequent appellate authority
    max_history_checks: int = 3       # citing opinions judged per citation
    max_citations: int = CITATION_CAP
    cl_rate_limit: float = 1.0        # seconds between CourtListener calls
    # academic sources (Story 1028): Semantic Scholar, OpenAlex, doi.org
    check_sources: bool = True
    max_sources: int = SOURCE_CAP
    support_reads: bool = True        # abstract-based "does it support the point"
    max_support_reads: int = 10       # per paper (~10–20 s each on gemma)
    scholarly_fallback: bool = True   # OpenAlex when Semantic Scholar has nothing
    doi_registry_check: bool = True   # doi.org: is the cited DOI registered?
    refresh_sources: bool = False     # ignore the results root's lookup cache
    identities: list = field(default_factory=list)   # roster for scrubbing

    def public(self) -> dict:
        d = asdict(self)
        d["identities"] = len(self.identities)   # never write names to disk
        return d


def read_roster(path) -> list[dict]:
    """Optional roster file: one student per line, 'Name' or 'Name, email'.
    A Brightspace download's ``index.html`` is accepted too — its student
    names become the roster (an instructor did exactly this, and the HTML
    lines were being scrubbed as if they were nine students)."""
    path = Path(path)
    if path.suffix.lower() in (".html", ".htm"):
        from firstread.local_eval_cohort import parse_submission_index
        idx = parse_submission_index(path.parent) if path.name.lower() == "index.html" else {}
        return [{"sub": n, "name": n, "email": ""} for n in idx.get("names", [])]
    idents = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, email = line.partition(",")
        idents.append({"sub": name.strip(), "name": name.strip(),
                       "email": email.strip()})
    return idents


# Brightspace "Download all submissions" names files
#   "<userId>-<submissionId> - <Last, First> - <Mon d, yyyy hhmm AM> - <file>"
D2L_PREFIX_RE = re.compile(
    r"^(?P<uid>\d+)-(?P<sid>\d+)\s+-\s+(?P<name>.+?)\s+-\s+"
    r"(?P<date>[A-Z][a-z]{2} \d{1,2}, \d{4} \d{3,4} [AP]M)\s+-\s+(?P<file>.+)$")


# ... or, in the layout Brightspace also produces, one FOLDER per submission
#   "<userId>-<submissionId> - <First Last> - <Mon d, yyyy hhmm AM>/<file>"
D2L_FOLDER_RE = re.compile(
    r"^(?P<uid>\d+)-(?P<sid>\d+)\s+-\s+(?P<name>.+?)\s+-\s+"
    r"(?P<date>[A-Z][a-z]{2} \d{1,2}, \d{4} \d{3,4} [AP]M)$")


def d2l_prefix(path) -> Optional[dict]:
    """``{uid, sid, name, date}`` from a Brightspace bulk download, whether
    the prefix is on the file name itself or on the per-submission folder
    it sits in (innermost folder wins). ``path`` may be a bare name, a
    folder-relative path (a manifest's file name) or an absolute path.
    None for anything else."""
    p = Path(str(path))
    m = D2L_PREFIX_RE.match(p.name)
    if not m:
        for part in reversed(p.parts[:-1]):
            m = D2L_FOLDER_RE.match(part)
            if m:
                break
    if not m:
        return None
    return {"uid": m.group("uid"), "sid": m.group("sid"),
            "name": m.group("name").strip(), "date": m.group("date")}


def identities_from_filename(filename) -> list[dict]:
    """The student a Brightspace bulk-download file name (or submission
    folder) says it belongs to, as a scrub identity — the name in the prefix
    is exactly the name the student writes in their header, so it MUST be on
    the roster even when the instructor gave none. Empty for any other path."""
    m = d2l_prefix(filename)
    if not m:
        return []
    return [{"sub": f"d2l:{m['uid']}", "name": m["name"], "email": ""}]


def _identity_key(ident: dict) -> str:
    return " ".join(sorted(re.sub(r"[^a-z0-9]+", " ", (ident.get("name") or "")
                                  .lower()).split()))


def merge_identities(*groups) -> list[dict]:
    """Union of identity lists, first occurrence of a name wins (so an
    explicit roster entry keeps its sub/email over a derived one)."""
    seen, out = set(), []
    for group in groups:
        for ident in group or []:
            key = _identity_key(ident)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(ident)
    return out


def folder_identities(folder, files, extra=None) -> tuple[list[dict], dict]:
    """Everything a submissions folder itself tells us about who wrote
    what: the explicit roster (``extra``), the names in Brightspace's
    ``index.html``, and every file's bulk-download prefix. Returns
    ``(identities, sources)`` where ``sources`` counts each origin — the
    counts (never the names) go into the run manifest."""
    roster = list(extra or [])
    index_names = []
    try:
        from firstread.local_eval_cohort import parse_submission_index
        idx = parse_submission_index(folder)
        index_names = [{"sub": n, "name": n, "email": ""}
                       for n in idx.get("names", [])]
    except Exception:  # noqa: BLE001
        log.warning("could not read index.html roster", exc_info=True)
    from_files = []
    for f in files or []:
        from_files.extend(identities_from_filename(f))
    merged = merge_identities(roster, index_names, from_files)
    sources = {"roster_file": len(roster), "index_html": len(index_names),
               "file_names": len({_identity_key(i) for i in from_files}),
               "total": len(merged)}
    return merged, sources


# ── packs / files ────────────────────────────────────────────────────────────

def normalize_pack(pack: dict, pack_path=None) -> dict:
    """Fill in what a hand-written pack leaves out, so a pack can be just a
    title, the instructions and a rubric:

    * ``assignment_id`` — the file name when absent;
    * ``course_id`` — a slug of ``course_title`` when absent (calibration
      decisions are kept per course);
    * ``rubric`` (v1 or v2 shape) → ``rubric_v2`` + ``level_names``;
    * ``assignment_ctx`` — built from title / purpose / task / criteria /
      ai_policy / objectives / sources (firstread.rubric_prompts);
    * ``readings`` → ``module_context`` (firstread.local_eval_context).

    See packs/example-memo.json and the README for the format."""
    from firstread.rubric_prompts import _rubric_level_names, assignment_context
    from firstread.rubric_schema import normalize_rubric_json
    if pack_path is not None and not pack.get("assignment_id"):
        pack["assignment_id"] = Path(pack_path).stem
    if not pack.get("course_id") and pack.get("course_title"):
        # calibration logs are kept per course (_calibration/course<id>.jsonl)
        pack["course_id"] = re.sub(r"[^a-z0-9]+", "-", pack["course_title"].lower()).strip("-")
    raw = pack.get("rubric_v2") or pack.get("rubric")
    if raw:
        pack["rubric_v2"] = normalize_rubric_json(raw)
    if pack.get("rubric_v2") and not pack.get("level_names"):
        pack["level_names"] = _rubric_level_names(pack["rubric_v2"])
    if not pack.get("assignment_ctx"):
        pack["assignment_ctx"] = assignment_context(pack)
    if pack.get("readings") and not pack.get("module_context"):
        from firstread.local_eval_context import readings_to_module_context
        mc = readings_to_module_context(
            pack, Path(pack_path).parent if pack_path is not None else None)
        if mc:
            pack["module_context"] = mc
    return pack


def load_pack(path) -> dict:
    """The pack (normalized, see ``normalize_pack``) plus, when present, its
    expected-moves sidecar (``<pack>.moves.json``, see firstread.expected_moves)."""
    pack = normalize_pack(json.loads(Path(path).read_text()), path)
    try:
        from firstread.expected_moves import load_moves
        moves = load_moves(path)
        if moves:
            pack["expected_moves"] = moves
    except Exception:  # noqa: BLE001
        log.warning("could not read moves sidecar for %s", path, exc_info=True)
    return pack


def pack_dirs() -> list[Path]:
    """``./packs`` and the repo's own ``packs/``; set FIRSTREAD_PACKS to a
    folder of your own to look there first."""
    import os
    dirs = []
    if os.getenv("FIRSTREAD_PACKS"):
        dirs.append(Path(os.environ["FIRSTREAD_PACKS"]).expanduser())
    dirs += [Path.cwd() / PACKS_DIR, REPO_ROOT / PACKS_DIR]
    seen, out = set(), []
    for d in dirs:
        r = d.resolve()
        if r in seen or not r.is_dir():
            continue
        seen.add(r)
        out.append(r)
    return out


def discover_packs() -> list[dict]:
    """Every assignment pack reachable from here, by course and then by
    assignment title (alphabetical, ignoring case), so a pack can be found
    by name in the picker."""
    packs = []
    for d in pack_dirs():
        for p in sorted(d.glob("*.json")):
            if p.name.endswith(".moves.json"):
                continue
            try:
                pk = load_pack(p)
            except Exception:  # noqa: BLE001
                continue
            packs.append({
                "path": str(p),
                "assignment_id": pk.get("assignment_id"),
                "title": pk.get("title", p.stem),
                "course_title": pk.get("course_title", ""),
                "has_rubric": bool(pk.get("rubric_v2")),
                "has_key": bool((pk.get("intentional_errors") or {})
                                .get("planted_errors")),
                "has_moves": bool((pk.get("expected_moves") or {}).get("moves")),
                "readings": len((pk.get("module_context") or {}).get("items") or []),
                "reading_chars": (pk.get("module_context") or {}).get("total_chars", 0),
                "digested": bool((pk.get("module_context") or {}).get("digest_chars")),
                "exported_at": pk.get("exported_at", ""),
            })
    packs.sort(key=lambda x: ((x["course_title"] or "").casefold(),
                              (x["title"] or "").casefold(), str(x["assignment_id"] or "")))
    return packs


# ── which assignment is this folder? ────────────────────────────────────────

_D2L_FOLDER_SUFFIX = re.compile(
    r"\s+Download\s+[A-Z][a-z]{2} \d{1,2}, \d{4} \d{3,4} [AP]M\s*$")
_GENERIC_TITLES = {"assignment submissions", "submissions", "index", ""}


def detect_assignment_title(folder) -> dict:
    """Best guess at the assignment a submissions folder belongs to:
    the Brightspace index.html heading, else the folder name minus the
    ``Download <date>`` suffix Brightspace appends. ``{"title", "source"}``."""
    folder = Path(folder)
    try:
        from firstread.local_eval_cohort import parse_submission_index
        idx = parse_submission_index(folder)
    except Exception:  # noqa: BLE001
        idx = {}
    for key in ("heading", "title"):
        t = (idx.get(key) or "").strip()
        t = re.sub(r"^(Assignment|Submissions?)\s*[:\-–]\s*", "", t, flags=re.I).strip()
        if t and t.lower() not in _GENERIC_TITLES:
            return {"title": t, "source": f"index.html {key}"}
    name = _D2L_FOLDER_SUFFIX.sub("", folder.name).strip()
    return {"title": name, "source": "folder name"}


def _norm_title(s: str) -> str:
    s = (s or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def match_pack(title: str, packs: Optional[list] = None) -> dict:
    """Rank packs against a detected title. Returns ``{"best", "score",
    "confident", "suggestions"}``; ``best`` is a discover_packs() entry."""
    import difflib
    packs = discover_packs() if packs is None else packs
    q = _norm_title(title)
    scored = []
    for p in packs:
        t = _norm_title(p.get("title", ""))
        if not q or not t:
            continue
        ratio = difflib.SequenceMatcher(None, q, t).ratio()
        if q == t:
            ratio = 1.0
        elif q in t or t in q:
            ratio = max(ratio, 0.9)
        scored.append((ratio, p))
    scored.sort(key=lambda x: -x[0])
    best = scored[0] if scored else None
    return {
        "best": best[1] if best and best[0] >= 0.55 else None,
        "score": round(best[0], 2) if best else 0.0,
        "confident": bool(best and best[0] >= 0.8),
        "suggestions": [{"path": p["path"], "assignment_id": p["assignment_id"],
                         "title": p["title"], "score": round(r, 2),
                         "has_key": p.get("has_key"), "has_rubric": p.get("has_rubric")}
                        for r, p in scored[:3] if r >= 0.4],
    }


def list_submission_files(folder, recursive: bool = True) -> list[Path]:
    """Supported files in ``folder`` and (by default) its subfolders.
    Skips hidden / lock / underscore-prefixed names, and any directory that
    is itself a results folder (has ``_run.json``) so a results root nested
    under the submissions is never re-graded."""
    folder = Path(folder)
    files = []
    if not recursive:
        candidates = sorted(folder.iterdir())
    else:
        candidates = sorted(folder.rglob("*"))
    for p in candidates:
        if not p.is_file():
            continue
        rel = p.relative_to(folder)
        if any(part.startswith((".", "~$", "_")) for part in rel.parts):
            continue
        if any((folder / Path(*rel.parts[:i]) / "_run.json").exists()
               for i in range(1, len(rel.parts))):
            continue
        if p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files


def submission_ids(folder, path) -> tuple[str, str]:
    """``(display name, stem)`` for a file under ``folder``: the display name
    is the folder-relative path; the stem folds subfolders into the file
    name with ``__`` so two students' ``memo.docx`` in different subfolders
    never collide in the flat results folder."""
    rel = Path(path).relative_to(Path(folder))
    name = rel.as_posix()
    stem = "__".join(list(rel.parts[:-1]) + [rel.stem])
    return name, stem


def extract_text(path: Path) -> str:
    from firstread.writing_feedback import extract_draft_text
    with open(path, "rb") as fh:
        return extract_draft_text(fh, filename=path.name)


def extract_submission(path: Path) -> dict:
    """``{"text", "markup", "insertions", "deletions"}`` — for a DOCX with
    Track Changes, ``text`` is the accepted-changes view and ``markup`` the
    redline view; for everything else ``markup`` is empty. A DOCX's
    footnotes and endnotes are included (``[^n]`` markers + a ``Notes``
    section) so footnoted citations reach the citation check."""
    path = Path(path)
    if path.suffix.lower() == ".docx":
        from firstread.writing_feedback import (extract_docx_redline,
                                            DraftExtractionError)
        raw = path.read_bytes()
        if not raw:
            raise DraftExtractionError("The uploaded file is empty.")
        r = extract_docx_redline(raw, include_notes=True)
        if not r["final"].strip():
            raise DraftExtractionError("The DOCX appears to contain no readable text.")
        tracked = bool(r["insertions"] or r["deletions"])
        return {"text": r["final"], "markup": r["markup"] if tracked else "",
                "insertions": r["insertions"], "deletions": r["deletions"]}
    return {"text": extract_text(path), "markup": "", "insertions": 0,
            "deletions": 0}


# ── local LLM judge (relevance + disposition, production prompts) ───────────

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "topically_relevant": {"type": "string", "enum": ["YES", "NO"]},
        "accurately_described": {"type": "string",
                                 "enum": ["YES", "NO", "CANNOT_DETERMINE"]},
        "summary": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["topically_relevant", "accurately_described", "summary",
                 "note"],
}

DISPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "disposition": {"type": "string",
                        "enum": ["AFFIRMED", "REVERSED", "VACATED", "REMANDED",
                                 "DISTINGUISHED", "DISCUSSED", "OTHER"]},
        "note": {"type": "string"},
    },
    "required": ["disposition", "note"],
}

_JSON_TAIL = {
    "relevance": ("\n\nReturn the answer as a JSON object with exactly these "
                  "keys: topically_relevant (YES or NO), accurately_described "
                  "(YES, NO or CANNOT_DETERMINE), summary, note."),
    "disposition": ("\n\nReturn the answer as a JSON object with exactly these "
                    "keys: disposition (one of AFFIRMED, REVERSED, VACATED, "
                    "REMANDED, DISTINGUISHED, DISCUSSED, OTHER) and note."),
}


class LocalJudge:
    """``LLMFn`` for firstread.citation_verifier: runs the production relevance /
    disposition prompts on the localhost model.

    Small models follow the prompts' 'KEY: value' line format unreliably, so
    the answer is decoded under a JSON schema (which gemma honours) and
    re-emitted in the line format the production parsers expect. Counts
    calls and failures so a run can report how much of the judging actually
    happened."""

    def __init__(self, opts: EvalOptions):
        self.opts = opts
        self.calls = 0
        self.failures = 0

    def __call__(self, prompt: str, kind: str) -> str:
        self.calls += 1
        schema = RELEVANCE_SCHEMA if kind == "relevance" else DISPOSITION_SCHEMA
        try:
            data = chat_json(prompt + _JSON_TAIL.get(kind, ""),
                             model=self.opts.model, host=self.opts.host,
                             num_ctx=self.opts.num_ctx,
                             temperature=self.opts.temperature,
                             think=self.opts.think, schema=schema)
        except LocalLLMError:
            self.failures += 1
            raise
        if kind == "relevance":
            return (f"TOPICALLY_RELEVANT: {data.get('topically_relevant', 'YES')}\n"
                    f"ACCURATELY_DESCRIBED: {data.get('accurately_described', 'CANNOT_DETERMINE')}\n"
                    f"SUMMARY: {data.get('summary', '')}\n"
                    f"NOTE: {data.get('note', '') or 'N/A'}")
        return (f"DISPOSITION: {data.get('disposition', 'OTHER')}\n"
                f"NOTE: {data.get('note', '')}")


# ── citation check ───────────────────────────────────────────────────────────

SOURCE_LABELS = {
    "found": "exists (no opinion text available — support not verified)",
    "found_relevant": "exists and supports the stated proposition",
    "found_miscited": "exists, but the paper mischaracterizes what it says",
    "found_unrelated": "exists, but does NOT support the stated proposition",
    "not_found": "could not be found (possibly inaccurate or fabricated)",
    "unverifiable": "could not be verified automatically — a statute / "
                    "database / secondary source the checker doesn't index; "
                    "check by hand",
    "ambiguous": "several possible matches — confirm which case is meant",
    "unknown": "could not be checked",
    # academic sources only (Story 1028)
    "found_mismatch": "exists, but the reference gets details wrong",
    "found_conflated": "a real work cited with the wrong authors or another "
                       "work's DOI — possibly fabricated or conflated; check by hand",
}

# Academic rows (type "academic") read their status through these labels.
from firstread.academic_lookup import VERDICT_LABELS as ACADEMIC_LABELS  # noqa: E402

SOURCE_KIND_LABELS = {
    "journal_article": "journal article", "law_review": "law-review article",
    "conference_paper": "conference paper", "preprint": "preprint", "book": "book",
    "book_chapter": "book chapter", "report": "report", "web": "web page",
    "news": "news article", "personal_communication": "personal communication",
    "course_material": "course material", "legal": "legal authority",
    "other": "source",
}

HISTORY_LABELS = {
    "none_found": "no higher-court treatment found",
    "good_standing": "cited by a higher court without adverse action",
    "caution": "distinguished or remanded by a higher court — check",
    "reversed": "REVERSED by a higher court",
    "vacated": "VACATED by a higher court",
    "error": "history check failed",
    "skipped": "court of last resort — no history check",
    "not_run": "history check not run",
    "unverified": "cited by a higher court, but the local model could not "
                  "classify the treatment — read the citing opinions",
}

PROBLEM_STATUSES = {"not_found", "found_unrelated", "found_miscited", "ambiguous",
                    "found_conflated"}
HISTORY_ALERTS = {"reversed", "vacated", "caution"}
HISTORY_UNVERIFIED = {"unverified", "error"}

# "could not be found" is only a student's error for a case citation in a
# reporter the checker indexes. The lookup also misses statutes it doesn't
# carry (42 Pa. Cons. Stat. § 7102), Westlaw / Lexis / public-domain numbers
# CourtListener never has, and secondary sources — those are "unverifiable".
# eyecite also reads dates in APA references ("10 Mar. 2015") as reporter
# citations; those aren't citations at all. Seen in every real course so far.
_STATUTE_RE = re.compile(r"§|\bU\.?S\.?C\b|\bC\.?F\.?R\b|\bStat\.|\bCons\.|\bCode\b|\bRev\.", re.I)
_DATABASE_RE = re.compile(r"\b(WL|LEXIS|Lexis)\b|-Ohio-\d")
_SECONDARY_RE = re.compile(r"\bA\.\s?L\.\s?R\.|\bL\.\s?Rev\.|\bL\.\s?J\.|\bRestatement\b", re.I)
_DATE_MISPARSE_RE = re.compile(
    r"^\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,? \d{4}$", re.I)

UNVERIFIABLE_NOTES = {
    "statute": "Statute the checker doesn't index (it looks up U.S.C. and "
               "C.F.R. sections at Cornell LII only).",
    "federal_statute": "Cornell LII returned no page for this section — "
                       "check the title and section number by hand.",
    "database": "Westlaw / Lexis / public-domain number — CourtListener "
                "doesn't index these; look it up by name.",
    "secondary": "Secondary source (A.L.R., law review, Restatement) — the "
                 "checker looks up cases and statutes only.",
}


def citation_kind(citation: str) -> str:
    """'case' | 'statute' | 'database' | 'secondary' | 'misparse' by shape."""
    c = (citation or "").strip()
    if _DATE_MISPARSE_RE.match(c):
        return "misparse"
    if _STATUTE_RE.search(c):
        return "statute"
    if _SECONDARY_RE.search(c):
        return "secondary"
    if _DATABASE_RE.search(c):
        return "database"
    return "case"


def finding_status(f: dict) -> str:
    """A finding's status, reading runs made before ``unverifiable`` existed
    the way ``check_citations`` now writes them: a not-found citation that
    isn't a case reporter citation was never evidence of a student error."""
    st = f.get("status") or "unknown"
    if f.get("type") == "academic":
        return st                  # written with today's statuses; never re-read
    if st == "not_found" and citation_kind(f.get("citation")) != "case":
        return "unverifiable"
    return st


def finding_label(f: dict) -> str:
    """The instructor-facing label for a finding's status."""
    st = finding_status(f)
    if f.get("type") == "academic":
        return ACADEMIC_LABELS.get(st, st)
    return SOURCE_LABELS.get(st, st)


def is_academic(f: dict) -> bool:
    return f.get("type") == "academic"


def _emit(progress: Optional[ProgressFn], **event):
    if progress:
        try:
            progress(event)
        except Exception:  # noqa: BLE001 — a UI glitch must not stop a run
            log.warning("progress callback failed", exc_info=True)


def check_citations(clean_text: str, opts: EvalOptions,
                    judge: Optional[LocalJudge] = None,
                    progress: Optional[ProgressFn] = None,
                    checker=None, file_label: str = "") -> dict:
    """Extract → existence → relevance → subsequent history for every
    case and statute in already-scrubbed text, then (``check_sources``) the
    academic sources: reference list, footnotes, bare DOIs → Semantic
    Scholar / OpenAlex / doi.org (utils/academic_lookup). Never raises;
    per-citation failures become ``status: unknown`` rows so nothing
    silently disappears. ``checker`` is the run's
    ``academic_lookup.SourceChecker`` (its cache and rate-limit state);
    one without a persistent cache is made when it is None."""
    judge = judge or LocalJudge(opts)
    cites = _legal_cites(clean_text) if opts.check_sources else None
    academic = _academic_sources(clean_text, opts, cites) if opts.check_sources else None
    # Each authority has one owner. A case or statute listed among the
    # references is the legal check's (claim_legal took it out of
    # ``academic``); an eyecite row inside a source that stayed academic is
    # that source's.
    spans = [sp for s in (academic or {}).get("sources", [])
             if s.get("source_kind") != "legal" for sp in s.get("spans", [])]
    source = _check_legal_citations(clean_text, opts, judge, progress, spans, cites)
    if academic is None:
        return source
    if not source.get("checked") and academic["sources"]:
        source.update(checked=True)       # eyecite failed; the sources still count
    checker = checker or make_source_checker(opts)
    rows, em, meta = checker.check(
        academic["sources"], judge=judge if opts.support_reads else None,
        progress=(lambda ev: _emit(progress, **ev)) if progress else None,
        file_label=file_label)
    source["findings"] = list(source.get("findings") or []) + rows
    source["emissions"] = {**source.get("emissions", {}), **em}
    source.update(meta, citation_form=academic["citation_form"],
                  judge_calls=judge.calls, judge_failures=judge.failures)
    return source


def _legal_cites(clean_text: str) -> Optional[list]:
    """eyecite's case / statute citations, minus dates it misread as
    reporter citations. None when extraction fails (the legal check then
    runs it again and reports the error)."""
    from firstread.citation_verifier import extract_citations
    try:
        cites = extract_citations(clean_text) or []
    except Exception:  # noqa: BLE001
        return None
    return [c for c in cites
            if citation_kind(c.get("citation_text", "")) != "misparse"]


def _academic_sources(clean_text: str, opts: EvalOptions,
                      cites: Optional[list] = None) -> Optional[dict]:
    """firstread.academic_citations.extract_sources, less the cases and statutes
    in the reference list (``claim_legal``: those belong to the legal check,
    whose ``cites`` these are), with the localhost model reading reference
    entries the heuristics can't (spec §1.3). None when extraction itself
    fails (logged; the case check still runs)."""
    from firstread.academic_citations import (extract_sources, fill_unparsed,
                                          claim_legal, parse_prompt, PARSE_SCHEMA)
    try:
        out = extract_sources(clean_text)
        claim_legal(clean_text, out, cites or [])
    except Exception:  # noqa: BLE001
        log.warning("academic source extraction failed", exc_info=True)
        return None

    def parse_fn(raws):
        prompt = parse_prompt(raws)
        res = chat_json(prompt, model=opts.model, host=opts.host,
                        num_ctx=context_window_for(prompt, opts.num_ctx),
                        temperature=opts.temperature, think=False,
                        schema=PARSE_SCHEMA)
        return res.get("entries") or []
    fill_unparsed(out["sources"], parse_fn)
    return out


def make_source_checker(opts: EvalOptions, cache_root=None):
    """The academic-source checker for a run: one Semantic Scholar circuit
    breaker, and the lookup cache in ``<cache_root>/_scholarly_cache.json``
    (the results root) when given."""
    from firstread.academic_lookup import (SourceChecker, SourceOptions,
                                       ScholarCache, CACHE_FILE)
    cache = ScholarCache(Path(cache_root) / CACHE_FILE if cache_root else None,
                         refresh=opts.refresh_sources)
    return SourceChecker(SourceOptions(
        max_sources=opts.max_sources,
        max_support_reads=opts.max_support_reads if opts.support_reads else 0,
        openalex=opts.scholarly_fallback, doi_registry=opts.doi_registry_check),
        cache)


def _inside(cite: dict, spans) -> bool:
    s, e = cite.get("start"), cite.get("end")
    if s is None or e is None:
        return False
    return any(a <= s and e <= b for a, b in spans)


def _check_legal_citations(clean_text, opts, judge, progress, drop_spans=(),
                           cites=None) -> dict:
    """The case / statute check. An eyecite row inside an academic source's
    span (a law-review cite read as a "case") is dropped: the academic row
    covers it. ``cites`` are the already-extracted citations, when the
    caller has them."""
    from firstread.citation_verifier import (
        extract_citations, lookup_citation, lookup_statute,
        check_relevance, check_subsequent_history, statute_range_start,
    )
    emissions = {"courtlistener": [], "cornell_lii": []}
    if cites is None:
        try:
            cites = extract_citations(clean_text) or []
        except Exception as e:  # noqa: BLE001
            log.warning("extract_citations failed", exc_info=True)
            return {"checked": False, "count": 0, "findings": [],
                    "error": f"citation extraction failed: {e}",
                    "emissions": emissions}
        # Dates eyecite misread as reporter citations aren't citations at all.
        cites = [c for c in cites
                 if citation_kind(c.get("citation_text", "")) != "misparse"]
    if drop_spans:
        cites = [c for c in cites if not _inside(c, drop_spans)]
    if not cites:
        return {"checked": True, "count": 0, "findings": [],
                "emissions": emissions}

    findings = []
    total = min(len(cites), opts.max_citations)
    for i, cite in enumerate(cites[:opts.max_citations], 1):
        ctype = cite.get("citation_type", "case")
        ctext = cite.get("citation_text", "")
        proposition = cite.get("proposition", "") or ""
        row = {"citation": ctext, "proposition": proposition[:300],
               "type": ctype, "status": "unknown", "case_name": "",
               "url": "", "note": "", "court": "", "date_filed": "",
               "cluster_id": "", "court_id": "", "history": None}
        _emit(progress, stage="citations", index=i, total=total,
              message=f"{ctext}: looking up")
        try:
            if ctype == "statute":
                emissions["cornell_lii"].append(ctext)
                # "9 U.S.C. §§ 9-10": LII has no page for a range; check
                # that its first section exists, and skip the relevance
                # read — one section's text can't support the whole range.
                first = statute_range_start(cite)
                lk = lookup_statute(dict(cite, section=first) if first else cite,
                                    rate_limit=opts.cl_rate_limit) or {}
                if lk.get("found"):
                    row.update(status="found",
                               case_name=lk.get("statute_name", ""),
                               url=lk.get("source_url", ""))
                    if first:
                        row["note"] = (f"A range of sections: § {first} "
                                       f"exists; the rest weren't checked.")
                    elif lk.get("statute_text") and proposition:
                        _emit(progress, stage="citations", index=i, total=total,
                              message=f"{ctext}: checking what it says")
                        rel = check_relevance(proposition, row["case_name"],
                                              lk.get("statute_text", ""),
                                              citation_type="statute",
                                              llm=judge) or {}
                        row["status"] = rel.get("status", "found")
                        row["note"] = _join_notes(rel)
                else:
                    federal = re.search(r"U\.?S\.?C|C\.?F\.?R",
                                        cite.get("reporter", "") or "")
                    row.update(status="unverifiable", unverifiable_reason="statute",
                               note=UNVERIFIABLE_NOTES["federal_statute" if federal
                                                       else "statute"])
            else:
                emissions["courtlistener"].append(ctext)
                lk = lookup_citation(cite, rate_limit=opts.cl_rate_limit) or {}
                if lk.get("found"):
                    row.update(status="found",
                               case_name=lk.get("case_name", ""),
                               url=lk.get("courtlistener_url", ""),
                               court=lk.get("court", ""),
                               date_filed=lk.get("date_filed", ""),
                               cluster_id=str(lk.get("cluster_id") or ""),
                               court_id=str(lk.get("court_id") or ""))
                    if lk.get("opinion_snippet") and proposition:
                        _emit(progress, stage="citations", index=i, total=total,
                              message=f"{ctext}: checking what it says")
                        rel = check_relevance(proposition, row["case_name"],
                                              lk.get("opinion_snippet", ""),
                                              citation_type="case",
                                              llm=judge) or {}
                        row["status"] = rel.get("status", "found")
                        row["note"] = _join_notes(rel)
                    else:
                        row["note"] = lk.get("confidence_note", "")
                    if opts.check_history and row["cluster_id"]:
                        _emit(progress, stage="citations", index=i, total=total,
                              message=f"{ctext}: subsequent history")
                        row["history"] = _history(check_subsequent_history,
                                                  row, opts, judge)
                    elif opts.check_history:
                        row["history"] = {"status": "not_run",
                                          "summary": "no cluster id from lookup",
                                          "citing_cases": []}
                elif lk.get("match_status") == "ambiguous":
                    cands = [c.get("case_name", "") for c in
                             (lk.get("candidate_matches") or [])[:3]]
                    row.update(status="ambiguous",
                               note=(lk.get("confidence_note", "") +
                                     (" Candidates: " + "; ".join(
                                         c for c in cands if c) if cands else "")).strip())
                elif (kind := citation_kind(ctext)) != "case":
                    row.update(status="unverifiable", unverifiable_reason=kind,
                               note=UNVERIFIABLE_NOTES[kind])
                else:
                    row.update(status="not_found",
                               note=lk.get("confidence_note", ""))
        except Exception as e:  # noqa: BLE001
            log.warning("citation check failed for %s", ctext, exc_info=True)
            row.update(status="unknown", note=f"check failed: {str(e)[:200]}")
        findings.append(row)

    return {"checked": True, "count": len(cites), "findings": findings,
            "capped": len(cites) > opts.max_citations,
            "judge_calls": judge.calls, "judge_failures": judge.failures,
            "emissions": emissions}


def _join_notes(rel: dict) -> str:
    parts = [rel.get("summary", ""), rel.get("confidence_note", "")]
    return " ".join(p for p in parts if p).strip()


def _history(fn, row, opts, judge):
    failures_before = judge.failures
    try:
        h = fn(row["cluster_id"], row["court_id"], row["case_name"],
               date_filed=row.get("date_filed", ""),
               max_llm_checks=opts.max_history_checks,
               rate_limit=opts.cl_rate_limit, llm=judge) or {}
        status = h.get("status", "error")
        summary = h.get("summary", "")
        if status == "good_standing" and judge.failures > failures_before:
            # The verifier maps a failed disposition call to "other" (severity
            # 1) which rolls up to good_standing — production's fail-open.
            # Locally, nothing was actually verified: say so.
            n = judge.failures - failures_before
            status = "unverified"
            summary = (f"{n} citing opinion(s) could not be classified by "
                       f"the local model; {summary}")
        return {"status": status, "summary": summary,
                "citing_cases": h.get("citing_cases", []) or []}
    except Exception as e:  # noqa: BLE001
        log.warning("history check failed for %s", row["citation"],
                    exc_info=True)
        return {"status": "error", "summary": f"{str(e)[:200]}",
                "citing_cases": []}


def source_summary_text(source: dict) -> str:
    """Compact text block for the grading prompt (same shape production's
    moodle_writing_eval._source_summary_text feeds it, plus history)."""
    sources_ran = "sources_count" in source
    if not source.get("checked"):
        n = source.get("count", 0)
        return (f"{n} citation(s) detected (not individually verified)."
                if n else "No legal citations detected.")
    findings = source.get("findings", [])
    if not findings:
        return ("No citations or sources detected." if sources_ran
                else "No verifiable legal citations detected.")
    legal = [f for f in findings if not is_academic(f)]
    academic = [f for f in findings if is_academic(f)]
    blocks = []
    if legal:
        lines = []
        for f in legal:
            label = SOURCE_LABELS.get(f.get("status"), f.get("status") or "unknown")
            nm = f" — {f['case_name']}" if f.get("case_name") else ""
            line = f"- {f.get('citation')}: {label}{nm}"
            h = f.get("history") or {}
            if h.get("status") in HISTORY_ALERTS:
                line += f" [subsequent history: {h.get('summary') or h['status']}]"
            lines.append(line)
        extra = (" (only the first %d shown)" % len(legal)
                 if source.get("capped") else "")
        blocks.append(f"{len(legal)} citation(s) checked{extra}:\n" + "\n".join(lines))
    if academic:
        extra = (" (only the first %d)" % len(academic)
                 if source.get("sources_capped") else "")
        blocks.append(f"Academic sources ({len(academic)} checked{extra}):\n"
                      + "\n".join(academic_summary_line(f) for f in academic))
    return "\n\n".join(blocks)


def academic_summary_line(f: dict) -> str:
    """One SOURCE CHECK line for an academic source (spec §5.4)."""
    c = f.get("cited") or {}
    who = ", ".join(s.title() for s in (c.get("author_surnames") or [])[:2])
    if len(c.get("author_surnames") or []) > 2:
        who += " et al."
    year = c.get("year") or "n.d."
    title = c.get("title") or f.get("citation") or ""
    head = (f"{who} ({year}), “{title}”" if who else f"“{title}” ({year})")
    st = f.get("status")
    if st == "found":
        tail = "verified"
    elif st == "found_mismatch":
        tail = "exists, but " + "; ".join(
            f"cited {d['field']} {d['cited']}, indexed {d['indexed']}"
            for d in f.get("differences") or [])
    elif st in ("found_conflated", "not_found"):
        base = ("a real work cited with the wrong authors or DOI" if st == "found_conflated"
                else "not found in Semantic Scholar or OpenAlex")
        sig = (f.get("note") or "").strip()
        tail = f"{base}. {sig} Possibly fabricated — instructor to verify".replace(". .", ".")
    elif st == "ambiguous":
        tail = "several possible matches — instructor to confirm which work"
    elif st == "unverifiable":
        tail = "not indexed — not checked automatically"
    else:
        tail = "not checked (" + (f.get("note") or "lookup failed").rstrip(".") + "); do not treat as missing"
    sup = f.get("support") or {}
    if sup.get("status") in ("found_unrelated", "found_miscited"):
        tail += "; the abstract doesn't obviously support the stated point"
    return f"- {head}: {tail}"


def source_flags(source: dict) -> dict:
    findings = source.get("findings", []) if source else []
    flags = {
        "citations": source.get("count", 0) if source else 0,
        "problems": sum(1 for f in findings if finding_status(f) in PROBLEM_STATUSES),
        "unverifiable": sum(1 for f in findings
                            if finding_status(f) == "unverifiable"),
        "history_alerts": sum(1 for f in findings
                              if (f.get("history") or {}).get("status") in HISTORY_ALERTS),
        "unchecked": sum(1 for f in findings
                         if f.get("status") == "unknown"
                         or (f.get("history") or {}).get("status")
                         in HISTORY_UNVERIFIED),
    }
    academic = [f for f in findings if is_academic(f)]
    if academic:
        flags.update(
            sources=len(academic),
            source_problems=sum(1 for f in academic if f.get("status") in PROBLEM_STATUSES),
            reference_errors=sum(1 for f in academic if f.get("status") == "found_mismatch"),
            possibly_fabricated=sum(1 for f in academic
                                    if f.get("status") == "found_conflated"
                                    or (f.get("status") == "not_found"
                                        and f.get("fabrication_signals"))))
    return flags


# ── rubric read (production grading prompt, schema-locked) ──────────────────

def criteria_names(rubric_v2):
    names = []
    for g in (rubric_v2 or {}).get("groups", []) or []:
        for c in (g.get("criteria") or []):
            if c.get("name") and c["name"] not in names:
                names.append(c["name"])
    return names


def rubric_result_schema(level_names, criteria):
    """Rubric levels AND criterion names are enum-locked to the pack's actual
    rubric so the local model cannot invent or garble either (observed: gemma
    emitting a criterion literally named ",")."""
    level = {"type": "string"}
    if level_names:
        level = {"type": "string", "enum": list(level_names)}
    criterion = {"type": "string"}
    if criteria:
        criterion = {"type": "string", "enum": list(criteria)}
    return {
        "type": "object",
        "properties": {
            "content_assessment": {"type": "string"},
            "rubric": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "criterion": criterion,
                    "suggested_level": level,
                    "rationale": {"type": "string"},
                    "instructor_check": {"type": "string"},
                },
                "required": ["criterion", "suggested_level", "rationale",
                             "instructor_check"],
            }},
            "focus_for_instructor": {"type": "array",
                                     "items": {"type": "string"}},
            "integrity_note": {"type": "string"},
        },
        "required": ["content_assessment", "rubric", "focus_for_instructor",
                     "integrity_note"],
    }


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "dimensions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "status": {"type": "string",
                           "enum": ["strong", "adequate", "needs_work"]},
                "comment": {"type": "string"},
                "suggestions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "status", "comment", "suggestions"],
        }},
        "top_priorities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "dimensions", "top_priorities"],
}


def tidy_rubric_rows(rows, criteria):
    """Dedupe (keep first mention) and restore the rubric's own criterion
    order; append a placeholder for any criterion the model skipped."""
    by_name = {}
    for r in rows:
        name = r.get("criterion")
        if name and name not in by_name:
            by_name[name] = r
    ordered = [by_name[c] for c in criteria if c in by_name]
    ordered += [r for n, r in by_name.items() if n not in criteria]
    for c in criteria:
        if c not in by_name:
            ordered.append({"criterion": c, "suggested_level": "(not assessed)",
                            "rationale": "model returned no row for this "
                            "criterion — assess manually",
                            "instructor_check": ""})
    return ordered


# Rubric level words by how well they read, for colouring a level when the
# pack's own level order is unavailable. Checked bad -> warn -> ok so
# "Approaching proficient" and "Does not meet" land where they should.
_LEVEL_WORDS = (
    ("bad", ("beginning", "novice", "unsatisfactory", "insufficient", "missing",
             "incomplete", "not met", "does not", "poor", "inadequate", "needs")),
    ("warn", ("developing", "emerging", "approaching", "partial", "basic",
              "adequate", "satisfactory", "fair")),
    ("ok", ("exemplary", "excellent", "advanced", "distinguished", "mastery",
            "exceeds", "proficient", "accomplished", "competent", "meets", "strong")),
)


def _word_tier(level: str) -> Optional[str]:
    low = (level or "").lower()
    for tier, words in _LEVEL_WORDS:
        if any(w in low for w in words):
            return tier
    return None


def level_tier(level: str, level_names=()) -> str:
    """ok / warn / bad / muted for a suggested rubric level, by its place in
    the pack's level names (exported low -> high, rubric_v2 order; flipped
    when they read high -> low), else by the level's own words."""
    names = list(level_names or [])
    if level in names and len(names) > 1:
        i = names.index(level)
        if _word_tier(names[0]) == "ok" and _word_tier(names[-1]) == "bad":
            i = len(names) - 1 - i
        r = i / (len(names) - 1)
        return "ok" if r >= 0.6 else "warn" if r >= 0.3 else "bad"
    return _word_tier(level) or "muted"


# Words a period can follow without ending a sentence: the listed legal and
# business abbreviations (any case), a capitalized word of up to three
# letters ("Md.", "Pa.", "Co."), and dotted initials ("U.S.", "N.Y.").
_ABBREV_END = re.compile(
    r"(?:^|\s|\()(?:(?i:v|vs|no|nos|inc|co|corp|ltd|llc|mr|ms|mrs|dr|st|ct|cir|"
    r"app|supp|stat|rev|ann|art|sec|para|cf|al|id)|[A-Z][a-z]{0,2}|"
    r"(?:[A-Za-z]\.)+[A-Za-z])\.$")


def first_sentence(text: str, limit: int = 220) -> str:
    """The first sentence of ``text``, cut at a word boundary past ``limit``.
    A period after a legal abbreviation ("Smith v. Jones", "1 U.S. 1",
    "Md. Code") does not end the sentence."""
    text = " ".join(str(text or "").split())
    s = text
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"“(])", text):
        if not _ABBREV_END.search(text[max(0, m.start() - 12):m.start()]):
            s = text[:m.start()]
            break
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return s


def rubric_read(rubric_rows, level_names=()) -> list[dict]:
    """The run page's line per criterion under a submission: criterion, the
    model's level, its colour tier and the first sentence of its reason."""
    return [{"criterion": r.get("criterion") or "",
             "level": r.get("suggested_level") or "",
             "tier": level_tier(r.get("suggested_level") or "", level_names),
             "why": first_sentence(r.get("rationale") or "")}
            for r in rubric_rows or []]


def grade(clean_text: str, pack: dict, source_summary: str,
          opts: EvalOptions, progress: Optional[ProgressFn] = None,
          calibration_text: str = ""):
    """Stage-2 rubric read (+ optional stage-1 review). Returns
    ``(result, review)``. Raises LocalLLMError. ``calibration_text`` (past
    instructor decisions, see firstread.local_eval_calibration) rides along in
    the assignment context so the production prompt stays untouched."""
    from firstread.rubric_prompts import (writing_review_prompt,
                                           content_and_rubric_prompt,
                                           _rubric_text)
    criteria = criteria_names(pack.get("rubric_v2"))
    _emit(progress, stage="grading", message="rubric read")
    ctx = pack["assignment_ctx"]
    if calibration_text:
        ctx = ctx + "\n\n" + calibration_text
    prompt = content_and_rubric_prompt(clean_text, ctx,
                                       pack.get("rubric_v2"), source_summary)
    result = chat_json(prompt, model=opts.model, host=opts.host,
                       num_ctx=context_window_for(prompt, opts.num_ctx),
                       temperature=opts.temperature, think=opts.think,
                       schema=rubric_result_schema(pack.get("level_names"),
                                                   criteria))
    result["rubric"] = tidy_rubric_rows(result.get("rubric") or [], criteria)
    review = None
    if opts.with_review:
        _emit(progress, stage="grading", message="formative review")
        rprompt = writing_review_prompt(
            clean_text, pack["assignment_ctx"], source_summary,
            rubric_text=_rubric_text(pack.get("rubric_v2")))
        review = chat_json(rprompt, model=opts.model, host=opts.host,
                           num_ctx=context_window_for(rprompt, opts.num_ctx),
                           temperature=opts.temperature,
                           think=opts.think, schema=REVIEW_SCHEMA)
    return result, review


def context_window_for(prompt: str, floor: int) -> int:
    """Ollama truncates from the FRONT when a prompt exceeds num_ctx, so size
    the window to the prompt: ~1 token per 3.2 chars, plus room for the
    reply, rounded up to 8k, never below the configured floor."""
    need = int(len(prompt) / 3.2) + 4096
    return max(floor, ((need + 8191) // 8192) * 8192)


# ── answer-key check (intentional_errors assignments) ───────────────────────

def check_planted_errors(submission_text: str, pack: dict, opts: EvalOptions,
                         has_tracked_changes: bool = False,
                         progress: Optional[ProgressFn] = None) -> dict:
    """Which of the pack's planted errors did this file find / fix? Uses the
    shared prompt in firstread.intentional_errors so the LMS path can run the
    identical check against Anthropic. Raises LocalLLMError."""
    from firstread.intentional_errors import (build_detection_prompt,
                                          detection_schema, tidy_detection)
    block = pack.get("intentional_errors") or {}
    ids = [e.get("id") for e in block.get("planted_errors") or [] if e.get("id")]
    _emit(progress, stage="answer_key",
          message=f"checking {len(ids)} planted errors")
    prompt = build_detection_prompt(
        block, submission_text, assignment_ctx=pack.get("assignment_ctx", ""),
        original_text=pack.get("defective_document_text", ""),
        has_tracked_changes=has_tracked_changes)
    raw = chat_json(prompt, model=opts.model, host=opts.host,
                    num_ctx=context_window_for(prompt, max(opts.num_ctx, 32768)),
                    temperature=opts.temperature,
                    think=opts.think, schema=detection_schema(ids))
    return tidy_detection(raw, block)


def check_expected_moves(submission_text: str, pack: dict, opts: EvalOptions,
                         progress: Optional[ProgressFn] = None) -> dict:
    """Score the submission against the pack's expected-moves checklist
    (firstread.expected_moves). Raises LocalLLMError."""
    from firstread.expected_moves import build_moves_prompt, moves_schema, tidy_moves
    doc = pack.get("expected_moves") or {}
    ids = [m["id"] for m in doc.get("moves") or []]
    _emit(progress, stage="moves", message=f"checking {len(ids)} expected moves")
    prompt = build_moves_prompt(doc, submission_text,
                                assignment_ctx=pack.get("assignment_ctx", ""))
    raw = chat_json(prompt, model=opts.model, host=opts.host,
                    num_ctx=context_window_for(prompt, max(opts.num_ctx, 32768)),
                    temperature=opts.temperature,
                    think=opts.think, schema=moves_schema(ids))
    return tidy_moves(raw, doc)


def draft_expected_moves(pack: dict, opts: EvalOptions) -> list:
    """Ask the local model for a first-draft checklist from the assignment
    design (and what the student read, when the pack carries it); the
    instructor edits and saves it. Raises LocalLLMError."""
    from firstread.expected_moves import draft_moves_prompt, DRAFT_SCHEMA, normalize_moves
    from firstread.local_eval_context import module_context_text
    reading = module_context_text(pack)
    dpack = pack
    if reading:
        dpack = dict(pack)
        dpack["assignment_ctx"] = pack.get("assignment_ctx", "") + "\n\n" + reading
    prompt = draft_moves_prompt(dpack)
    raw = chat_json(prompt, model=opts.model, host=opts.host,
                    num_ctx=context_window_for(prompt, opts.num_ctx),
                    temperature=opts.temperature,
                    think=opts.think, schema=DRAFT_SCHEMA)
    return normalize_moves(raw.get("moves") or [])


def digest_module_context(pack_path, opts: EvalOptions,
                          progress: Optional[ProgressFn] = None) -> dict:
    """Reduce each reading in the pack's module context to salient points
    with the local model and save the digests into the pack file. Raises
    LocalLLMError."""
    from firstread.local_eval_context import (digest_prompt, DIGEST_SCHEMA,
                                          digest_to_markdown, digest_acceptable)
    path = Path(pack_path)
    pack = json.loads(path.read_text())
    mc = pack.get("module_context") or {}
    items = mc.get("items") or []
    rejected = []
    for i, item in enumerate(items, 1):
        _emit(progress, stage="digest", index=i, total=len(items),
              message=item.get("title", ""))
        prompt = digest_prompt(item, pack.get("title", ""))
        data = chat_json(prompt, model=opts.model, host=opts.host,
                         num_ctx=context_window_for(prompt, opts.num_ctx),
                         temperature=opts.temperature, think=opts.think,
                         schema=DIGEST_SCHEMA)
        md = digest_to_markdown(data)
        if digest_acceptable(md, item.get("text", "")):
            item["digest"] = md
            item["digest_model"] = opts.model
            item.pop("digest_error", None)
        else:
            # keep the full text in play rather than a degenerate summary
            item.pop("digest", None)
            item["digest_error"] = (f"digest too short ({len(md)} chars for "
                                    f"{len(item.get('text', ''))}) — full text kept")
            rejected.append(item.get("title", ""))
    mc["digest_chars"] = sum(len(i.get("digest") or "") for i in items)
    mc["digest_rejected"] = rejected
    path.write_text(json.dumps(pack, indent=2, ensure_ascii=False))
    return mc


# ── per-file pipeline ────────────────────────────────────────────────────────

def evaluate_file(path: Path, pack: Optional[dict], opts: EvalOptions,
                  progress: Optional[ProgressFn] = None,
                  context_id: str = "local-eval",
                  calibration_root=None, name: Optional[str] = None,
                  stem: Optional[str] = None, source_checker=None) -> dict:
    """Run the whole pipeline on one file. Returns the result record (also
    what gets written to <stem>.json). Extraction failure raises
    DraftExtractionError; model failures are recorded in ``errors`` and the
    partial record is still returned.

    The scrubbed text the model saw is returned under ``_text`` (and the
    tracked-changes view under ``_markup``) so the runner can save it next
    to the report; those keys are not part of the JSON record."""
    from firstread.pii_gate import scrub
    from firstread.intentional_errors import detection_summary_text
    from firstread.local_eval_context import module_context_text
    path = Path(path)
    t0 = time.time()
    timings, errors = {}, []

    # What the student could have read rides in the assignment context so
    # every downstream prompt (answer key, moves, rubric) sees it.
    reading_block = module_context_text(pack) if pack else ""
    if reading_block:
        pack = dict(pack)
        pack["assignment_ctx"] = (pack.get("assignment_ctx", "") + "\n\n" + reading_block)

    _emit(progress, stage="extract", message="extracting text")
    sub = extract_submission(path)
    text = sub["text"]
    # The roster plus whoever the file name says this is (Brightspace
    # prefix) — a direct caller without a roster still gets the header name.
    identities = merge_identities(opts.identities,
                                  identities_from_filename(path))
    clean, report = scrub(text, identities=identities, context_id=context_id)
    clean_markup = ""
    if sub["markup"]:
        clean_markup, _ = scrub(sub["markup"], identities=identities,
                                context_id=context_id)
    timings["extract_s"] = round(time.time() - t0, 1)

    source = {"checked": False, "count": 0, "findings": [],
              "emissions": {"courtlistener": [], "cornell_lii": []}}
    if opts.check_citations:
        t1 = time.time()
        source = check_citations(clean, opts, LocalJudge(opts), progress,
                                 checker=source_checker, file_label=name or path.name)
        timings["citations_s"] = round(time.time() - t1, 1)
        if source.get("error"):
            errors.append(source["error"])

    planted = None
    has_key = bool(pack and (pack.get("intentional_errors") or {})
                   .get("planted_errors"))
    if has_key:
        t3 = time.time()
        try:
            planted = check_planted_errors(clean_markup or clean, pack, opts,
                                           has_tracked_changes=bool(clean_markup),
                                           progress=progress)
        except LocalLLMError as e:
            errors.append(f"answer-key check failed: {e}")
        timings["answer_key_s"] = round(time.time() - t3, 1)

    moves = None
    has_moves = bool(pack and (pack.get("expected_moves") or {}).get("moves"))
    if has_moves:
        t4 = time.time()
        try:
            moves = check_expected_moves(clean, pack, opts, progress)
        except LocalLLMError as e:
            errors.append(f"expected-moves check failed: {e}")
        timings["moves_s"] = round(time.time() - t4, 1)

    result, review = None, None
    if pack:
        from firstread.expected_moves import moves_summary_text
        t2 = time.time()
        summary = (source_summary_text(source) if opts.check_citations else
                   "(source check not run — verify cited authorities manually)")
        gpack = pack
        extra_ctx = []
        if planted:
            # The rubric read must weigh detection against the actual key.
            extra_ctx.append(detection_summary_text(planted))
        if moves:
            extra_ctx.append(moves_summary_text(moves))
        if extra_ctx:
            gpack = dict(pack)
            gpack["assignment_ctx"] = "\n\n".join(
                [pack.get("assignment_ctx", "")] + extra_ctx)
        calibration_text = ""
        if opts.use_calibration and calibration_root:
            try:
                from firstread.local_eval_calibration import calibration_block
                calibration_text = calibration_block(
                    calibration_root, pack, criteria_names(pack.get("rubric_v2")))
            except Exception:  # noqa: BLE001 — never block grading on it
                log.warning("calibration block failed", exc_info=True)
        try:
            result, review = grade(clean, gpack, summary, opts, progress,
                                   calibration_text=calibration_text)
        except LocalLLMError as e:
            errors.append(f"rubric read failed: {e}")
        timings["grading_s"] = round(time.time() - t2, 1)
        timings["calibrated"] = bool(calibration_text)
    timings["total_s"] = round(time.time() - t0, 1)
    timings["module_context_chars"] = len(reading_block)

    flags = source_flags(source)
    if has_key:
        flags["key_total"] = planted["total"] if planted else len(
            pack["intentional_errors"]["planted_errors"])
        flags["key_detected"] = planted["detected"] if planted else None
    if has_moves:
        flags["moves_total"] = moves["total"] if moves else len(
            pack["expected_moves"]["moves"])
        flags["moves_present"] = moves["present"] if moves else None
    return {
        "file": name or path.name,
        "stem": stem or path.stem,
        "assignment_id": pack.get("assignment_id") if pack else None,
        "assignment_title": pack.get("title") if pack else None,
        "model": opts.model,
        "preliminary": True,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "chars_in": len(text),
        "tracked_changes": {"insertions": sub["insertions"],
                            "deletions": sub["deletions"]},
        "redactions": report,
        "source": source,
        "planted": planted,
        "moves": moves,
        "flags": flags,
        "result": result,
        "review": review,
        "errors": errors,
        "timings": timings,
        "_text": clean,
        "_markup": clean_markup,
    }


# ── rendering ────────────────────────────────────────────────────────────────

def render_md(res: dict) -> str:
    fname = res["file"]
    lines = [f"# {fname}"]
    if res.get("assignment_title"):
        lines.append(f"*Assignment: {res['assignment_title']}*")
    lines += ["", "> **PRELIMINARY — local model triage draft. Not a grade.**",
              ""]
    fl = res.get("flags") or {}
    src = res.get("source") or {}
    if src.get("checked"):
        lines += [f"**Citations:** {fl.get('citations', 0)} found · "
                  f"{fl.get('problems', 0)} problem(s) · "
                  f"{fl.get('history_alerts', 0)} subsequent-history alert(s) · "
                  f"{fl.get('unchecked', 0)} unchecked"
                  + (f" · {fl['unverifiable']} couldn't verify automatically"
                     if fl.get("unverifiable") else "")
                  + (f" · {fl['sources']} academic source(s), "
                     f"{fl.get('source_problems', 0)} problem(s), "
                     f"{fl.get('reference_errors', 0)} reference error(s)"
                     if fl.get("sources") else ""), ""]
    for e in res.get("errors") or []:
        lines.append(f"> ⚠ {e}")
    if res.get("errors"):
        lines.append("")

    all_findings = src.get("findings") or []
    findings = [f for f in all_findings if not is_academic(f)]
    academic = [f for f in all_findings if is_academic(f)]
    if findings:
        lines += ["## Cases and statutes", "",
                  "| Citation | Status | Authority | Subsequent history |",
                  "|---|---|---|---|"]
        for f in findings:
            label = SOURCE_LABELS.get(finding_status(f), f.get("status"))
            auth = f.get("case_name") or ""
            if f.get("url"):
                auth = f"[{auth or 'view'}]({f['url']})"
            h = f.get("history") or {}
            hist = ""
            if h:
                hist = HISTORY_LABELS.get(h.get("status"), h.get("status"))
                if h.get("summary") and h.get("status") in (
                        HISTORY_ALERTS | HISTORY_UNVERIFIED | {"good_standing"}):
                    hist += f" — {h['summary']}"
            lines.append(f"| {_cell(f.get('citation'))} | {_cell(label)} | "
                         f"{_cell(auth)} | {_cell(hist)} |")
        lines.append("")
        detail = [f for f in findings
                  if f.get("note") or (f.get("history") or {}).get("citing_cases")]
        if detail:
            lines += ["### Notes", ""]
            for f in detail:
                lines.append(f"- **{f.get('citation')}**")
                if f.get("proposition"):
                    lines.append(f"  - Student's proposition: “{f['proposition']}”")
                if f.get("note"):
                    lines.append(f"  - {f['note']}")
                for c in (f.get("history") or {}).get("citing_cases") or []:
                    nm = c.get("citing_case_name", "")
                    if c.get("courtlistener_url"):
                        nm = f"[{nm}]({c['courtlistener_url']})"
                    yr = (c.get("citing_date_filed") or "")[:4]
                    lines.append(f"  - {c.get('disposition', '').title()} in {nm} "
                                 f"({c.get('citing_court', '')} {yr}): "
                                 f"{c.get('disposition_note', '')}")
            lines.append("")
    elif src.get("checked") and not academic:
        lines += ["## Cases and statutes", "", "No legal citations detected.", ""]
    if academic:
        lines += academic_md(academic, src.get("citation_form") or {},
                             src.get("sources_capped"))

    planted = res.get("planted")
    if planted:
        from firstread.intentional_errors import DETECTION_STATUS_LABELS
        tc = res.get("tracked_changes") or {}
        lines += [f"## Answer key: {planted['detected']} of {planted['total']} "
                  f"planted errors detected "
                  f"({planted['high_detected']}/{planted['high_total']} high-severity)",
                  ""]
        if tc.get("insertions") or tc.get("deletions"):
            lines += [f"*Track Changes seen: {tc.get('insertions', 0)} inserted "
                      f"run(s), {tc.get('deletions', 0)} deleted run(s) — the "
                      "redline view was what the model judged.*", ""]
        lines += ["| # | Severity | Planted error | Student | Evidence |",
                  "|---|---|---|---|---|"]
        for r in planted.get("errors") or []:
            lines.append(f"| {r.get('id')} | {r.get('severity')} | "
                         f"{_cell(r.get('category'))} | "
                         f"**{_cell(DETECTION_STATUS_LABELS.get(r.get('status'), r.get('status')))}** | "
                         f"{_cell(r.get('evidence'))}"
                         + (f" — {_cell(r.get('note'))}" if r.get("note") else "")
                         + " |")
        lines.append("")
        extra = planted.get("additional_issues") or []
        if extra:
            lines += ["### Issues the student raised beyond the key", ""]
            lines += [f"- ({a.get('assessment')}) {a.get('description')}"
                      for a in extra] + [""]
        if planted.get("summary"):
            lines += [planted["summary"], ""]

    moves = res.get("moves")
    if moves:
        from firstread.expected_moves import MOVE_STATUS_LABELS
        lines += [f"## Expected moves: {moves['present']} of {moves['total']} present "
                  f"({moves['strong']} strong; {moves['high_present']}/{moves['high_total']} "
                  "high-severity)", "",
                  "| # | Severity | Move | Status | Evidence |", "|---|---|---|---|---|"]
        for r in moves.get("moves") or []:
            lines.append(f"| {r.get('id')} | {r.get('severity')} | {_cell(r.get('move'))} | "
                         f"**{_cell(MOVE_STATUS_LABELS.get(r.get('status'), r.get('status')))}** | "
                         f"{_cell(r.get('evidence'))}"
                         + (f" — {_cell(r.get('note'))}" if r.get("note") else "") + " |")
        lines.append("")
        if moves.get("summary"):
            lines += [moves["summary"], ""]

    result = res.get("result") or {}
    if result.get("content_assessment"):
        lines += ["## Content assessment", result["content_assessment"], ""]
    rub = result.get("rubric") or []
    if rub:
        lines += ["## Rubric suggestions (verify before adopting)", ""]
        for r in rub:
            lines.append(f"- **{r.get('criterion', '?')}** → "
                         f"*{r.get('suggested_level', '?')}*")
            if r.get("rationale"):
                lines.append(f"  - {r['rationale']}")
            if r.get("instructor_check"):
                lines.append(f"  - Check: {r['instructor_check']}")
        lines.append("")
    focus = result.get("focus_for_instructor") or []
    if focus:
        lines += ["## Where your judgment is most needed", ""]
        lines += [f"- {f}" for f in focus] + [""]
    if result.get("integrity_note"):
        lines += ["## Integrity note", result["integrity_note"], ""]
    review = res.get("review")
    if review:
        lines += ["## Formative review (stage 1)", "",
                  review.get("summary", ""), ""]
        for d in review.get("dimensions") or []:
            lines.append(f"- **{d.get('name')}**: {d.get('status')} — "
                         f"{d.get('comment', '')}")
        lines.append("")
    redacted = {k: v for k, v in (res.get("redactions") or {}).items() if v}
    lines += ["---", f"*PII scrubbed before any model call: "
              f"{redacted if redacted else 'nothing found to redact'}. "
              f"Model: {res.get('model')} (localhost). "
              f"Time: {(res.get('timings') or {}).get('total_s', '?')}s.*"]
    return "\n".join(lines) + "\n"


def _cell(s) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ")


def indexed_line(f: dict) -> str:
    """'Title — O. Ray · American Psychologist 2004 (from Semantic Scholar)'."""
    ix = f.get("indexed") or {}
    if not ix:
        return ""
    authors = ", ".join((ix.get("authors") or [])[:3])
    if len(ix.get("authors") or []) > 3:
        authors += " et al."
    where = " ".join(str(x) for x in (ix.get("venue"), ix.get("year")) if x)
    src = {"semantic_scholar": "Semantic Scholar", "openalex": "OpenAlex"}.get(ix.get("source"), "")
    return " · ".join(x for x in (authors, where) if x) + (f" (from {src})" if src else "")


def academic_why(f: dict) -> str:
    """The "What differs / why" cell: differences, signals, the support read."""
    parts = []
    if f.get("note"):
        parts.append(f["note"])
    sup = f.get("support") or {}
    if sup.get("note"):
        parts.append(f"Abstract read: {sup['note']}")
    return " ".join(parts)


def academic_md(rows: list, form: dict, capped: bool = False) -> list:
    """The per-file report's Academic sources table (spec §5.3)."""
    lines = ["## Academic sources", "",
             "| Reference | Verdict | Indexed record | What differs / why |",
             "|---|---|---|---|"]
    for f in rows:
        ref = f.get("citation") or ""
        if f.get("in_text"):
            ref += f" — cited as {', '.join(f['in_text'][:2])}"
        rec = f.get("case_name") or ""
        if f.get("url") and rec:
            rec = f"[{rec}]({f['url']})"
        line = indexed_line(f)
        rec = f"{rec} — {line}" if rec and line else (rec or line)
        lines.append(f"| {_cell(ref)} | **{_cell(finding_status(f))}**: "
                     f"{_cell(finding_label(f))} | {_cell(rec)} | {_cell(academic_why(f))} |")
    if capped:
        lines.append(f"\n*Only the first {len(rows)} sources were checked.*")
    lines.append("")
    props = [f for f in rows if f.get("proposition")]
    if props:
        lines += ["### What each source is cited for", ""]
        lines += [f"- **{f.get('case_name') or (f.get('cited') or {}).get('title') or f.get('citation')}**: "
                  f"“{f['proposition']}”" for f in props] + [""]
    if form.get("in_text_missing") or form.get("uncited_entry"):
        lines += ["### Citation form", ""]
        lines += [f"- Cited in the text but missing from the references: {c}"
                  for c in form.get("in_text_missing") or []]
        lines += [f"- In the references but never cited in the text: {t}"
                  for t in form.get("uncited_entry") or []]
        lines.append("")
    return lines


def write_summary(outdir: Path, manifest: dict):
    files = manifest.get("files", [])
    criteria = []
    for f in files:
        for c in (f.get("levels") or {}):
            if c not in criteria:
                criteria.append(c)
    title = manifest.get("assignment_title") or Path(manifest["folder"]).name
    has_key = any("key_total" in (f.get("flags") or {}) for f in files)
    has_moves = any("moves_total" in (f.get("flags") or {}) for f in files)
    has_sources = any((f.get("flags") or {}).get("sources") for f in files)
    key_col = (" Answer key |" if has_key else "") + (" Moves |" if has_moves else "")
    src_col = " Sources (problems) |" if has_sources else ""
    lines = [f"# {title} — preliminary triage",
             f"*Local model: {manifest.get('model')}. Folder: "
             f"`{manifest.get('folder')}`. Suggested levels are a draft read "
             "— not grades.*", "",
             "| Submission | Cites |" + src_col + " Problems | History alerts | Unchecked |"
             + key_col + (" " + " | ".join(criteria) + " |" if criteria else ""),
             "|---|---|" + ("---|" if has_sources else "") + "---|---|---|"
             + ("---|" if has_key else "")
             + ("---|" if has_moves else "") + "---|" * len(criteria)]
    for f in files:
        fl = f.get("flags") or {}
        if f.get("status") != "ok":
            lines.append(f"| {f['name']} | — |" + (" — |" if has_sources else "")
                         + " — | — | — |"
                         + (" — |" if has_key else "") + (" — |" if has_moves else "")
                         + " — |" * len(criteria)
                         + f" {f.get('error', f.get('status'))}")
            continue
        levels = f.get("levels") or {}
        row = (f"| [{f['name']}]({f['stem']}.md) | {fl.get('citations', 0)} |"
               + (f" {fl.get('sources', 0)} ({fl.get('source_problems', 0)}) |"
                  if has_sources else "")
               + f" {fl.get('problems', 0)} | {fl.get('history_alerts', 0)} | "
               f"{fl.get('unchecked', 0)} |")
        if has_key:
            kd, kt = fl.get("key_detected"), fl.get("key_total")
            row += (f" {kd}/{kt} |" if kd is not None else
                    (f" ?/{kt} |" if kt else " — |"))
        if has_moves:
            mp, mt = fl.get("moves_present"), fl.get("moves_total")
            row += (f" {mp}/{mt} |" if mp is not None else
                    (f" ?/{mt} |" if mt else " — |"))
        if criteria:
            row += " " + " | ".join(levels.get(c, "—") or "—"
                                    for c in criteria) + " |"
        lines.append(row)
    (outdir / "_summary.md").write_text("\n".join(lines) + "\n")


# ── folder runner ────────────────────────────────────────────────────────────

def run_folder(folder, pack: Optional[dict], opts: EvalOptions, outdir,
               progress: Optional[ProgressFn] = None,
               should_stop: Optional[Callable[[], bool]] = None,
               run_id: str = "", pack_path: str = "",
               files: Optional[list] = None) -> dict:
    """Evaluate every supported file in ``folder`` (or the explicit ``files``
    list); write per-file outputs, ``_summary.md`` and ``_run.json`` into
    ``outdir``. Returns the manifest. Preflights the model once (fails fast
    if Ollama is down)."""
    from firstread.writing_feedback import DraftExtractionError
    folder, outdir = Path(folder), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = ([Path(f) for f in files] if files
             else list_submission_files(folder))
    ids = []
    for p in files:
        try:
            ids.append(submission_ids(folder, p))
        except ValueError:                      # explicit file outside folder
            ids.append((p.name, p.stem))

    # Names to scrub: the given roster + what the folder itself reveals
    # (index.html, Brightspace file-name prefixes). Every file is scrubbed
    # against the whole class so cross-references ("as Jordan argued") go
    # too — same as the production NRPS-roster path.
    identities, roster_sources = folder_identities(folder, files,
                                                   opts.identities)
    opts = replace(opts, identities=identities)
    _emit(progress, stage="roster",
          message=(f"scrubbing {roster_sources['total']} name(s): "
                   f"{roster_sources['roster_file']} from roster, "
                   f"{roster_sources['index_html']} from index.html, "
                   f"{roster_sources['file_names']} from file names"))

    manifest = {
        "run_id": run_id or outdir.name,
        "folder": str(folder),
        "pack": pack_path or None,
        "assignment_title": pack.get("title") if pack else None,
        "model": opts.model,
        "options": opts.public(),
        "roster": roster_sources,             # counts only, never names
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "status": "running",
        "files": [{"name": n, "stem": s, "status": "pending"}
                  for n, s in ids],
    }
    _save_manifest(outdir, manifest)

    available = list_models(opts.host)   # raises LocalLLMError when down
    if opts.model not in available:
        raise LocalLLMError(f"model {opts.model!r} not pulled; available: "
                            f"{available}")

    context_id = (f"local-eval-ia{pack['assignment_id']}" if pack
                  else f"local-eval-{folder.name}")
    calibration_root = (Path(opts.calibration_root) if opts.calibration_root
                        else outdir.parent)
    # One academic-source checker per run: the results root's lookup cache
    # and one Semantic Scholar circuit breaker across every file.
    source_checker = (make_source_checker(opts, cache_root=outdir.parent)
                      if opts.check_citations and opts.check_sources else None)
    if source_checker:
        manifest["scholarly"] = source_checker.status()
    audit = open(outdir / "_emissions.jsonl", "a")
    try:
        for i, path in enumerate(files):
            entry = manifest["files"][i]
            fname, fstem = ids[i]
            if should_stop and should_stop():
                entry["status"] = "stopped"
                continue
            _emit(progress, stage="file", index=i + 1, total=len(files),
                  file=fname, message="starting")

            def fprogress(ev, _name=fname, _i=i + 1):
                ev.setdefault("file", _name)
                ev.setdefault("file_index", _i)
                ev.setdefault("file_total", len(files))
                _emit(progress, **ev)

            try:
                res = evaluate_file(path, pack, opts, fprogress, context_id,
                                    calibration_root=calibration_root,
                                    name=fname, stem=fstem,
                                    source_checker=source_checker)
            except DraftExtractionError as e:
                entry.update(status="skipped", error=f"extract failed: {e}")
                _emit(progress, stage="file_done", file=fname,
                      status="skipped", message=str(e))
                _save_manifest(outdir, manifest)
                continue
            except Exception as e:  # noqa: BLE001
                log.exception("evaluate_file failed for %s", fname)
                entry.update(status="failed", error=str(e)[:300])
                _emit(progress, stage="file_done", file=fname,
                      status="failed", message=str(e)[:300])
                _save_manifest(outdir, manifest)
                continue

            # What the model actually saw (scrubbed), for instructor
            # verification of extraction — local results dir only.
            clean_text, clean_markup = res.pop("_text", ""), res.pop("_markup", "")
            (outdir / f"{fstem}.txt").write_text(clean_text)
            if clean_markup:
                (outdir / f"{fstem}.redline.txt").write_text(clean_markup)
            (outdir / f"{fstem}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False))
            (outdir / f"{fstem}.md").write_text(render_md(res))
            em = res["source"].get("emissions") or {}
            audit.write(json.dumps({
                "ts": res["evaluated_at"], "file": fname,
                "model": opts.model, "destination": "localhost",
                "chars_in": res["chars_in"], "redactions": res["redactions"],
                "research_lookups": {k: v for k, v in em.items() if v},
            }) + "\n")
            audit.flush()
            rows = (res.get("result") or {}).get("rubric") or []
            levels = {r.get("criterion"): r.get("suggested_level") for r in rows}
            read = rubric_read(rows, (pack or {}).get("level_names"))
            entry.update(status="ok", flags=res["flags"], levels=levels,
                         rubric_read=read, errors=res.get("errors") or [],
                         seconds=res["timings"].get("total_s"))
            if source_checker:
                manifest["scholarly"] = source_checker.status()
            _emit(progress, stage="file_done", file=fname, status="ok",
                  flags=res["flags"], levels=levels, rubric_read=read,
                  message=f"done in {res['timings'].get('total_s')}s")
            _save_manifest(outdir, manifest)
    finally:
        audit.close()
        stopped = any(f["status"] == "stopped" for f in manifest["files"])
        manifest["status"] = "stopped" if stopped else "done"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_manifest(outdir, manifest)
        write_summary(outdir, manifest)
    if any(f["status"] == "ok" for f in manifest["files"]):
        # Class view: pair files into students, heat map, next-class brief.
        _emit(progress, stage="cohort", message="building the class view")
        try:
            from firstread.local_eval_cohort import build_cohort
            cohort = build_cohort(outdir, manifest, pack, opts,
                                  identities=opts.identities,
                                  with_brief=opts.with_brief)
            manifest["cohort"] = {
                "students": len(cohort["students"]),
                "brief": bool(cohort.get("brief")),
                "errors": cohort.get("errors") or [],
            }
        except Exception as e:  # noqa: BLE001 — the per-file work is done
            log.exception("cohort build failed")
            manifest["cohort"] = {"students": 0, "brief": False,
                                  "errors": [str(e)[:300]]}
        _save_manifest(outdir, manifest)
        _emit(progress, stage="cohort_done",
              message=f"class view ready ({manifest['cohort']['students']} student(s))")
    return manifest


def _save_manifest(outdir: Path, manifest: dict):
    (outdir / "_run.json").write_text(json.dumps(manifest, indent=2,
                                                 ensure_ascii=False))


def load_manifest(outdir) -> Optional[dict]:
    p = Path(outdir) / "_run.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def list_runs(results_root) -> list[dict]:
    root = Path(results_root)
    if not root.is_dir():
        return []
    runs = []
    for d in root.iterdir():
        if d.is_dir():
            m = load_manifest(d)
            if m:
                m["outdir"] = str(d)
                # Route by directory name: a CLI run with -o names its
                # folder freely, and the UI locates results by folder.
                m["run_id"] = d.name
                runs.append(m)
    runs.sort(key=lambda m: m.get("started_at") or "", reverse=True)
    return runs
