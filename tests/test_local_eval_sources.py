"""Academic sources in the local evaluation pipeline (Story 1028): how
check_citations, the flags, the SOURCE CHECK text, the reports, the run
folder, the UI, the class view, the portfolio and the debrief carry them.

Real eyecite and the real extractor run on the paper; Semantic Scholar,
OpenAlex and doi.org are stubbed at utils/semantic_scholar_http and the
SourceChecker's OpenAlex / doi.org methods, and a socket guard fails any
test that tries to reach the network. CourtListener is stubbed too.

    python3 -m pytest tests/test_local_eval_sources.py
"""
import json
import socket
import sys
from pathlib import Path

import pytest

from firstread import academic_lookup as al
from firstread import citation_verifier as cv
from firstread import local_eval as le
from firstread import local_eval_cohort as co
from firstread import semantic_scholar_http as ssh

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    real = socket.socket.connect

    def guarded(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"test tried to reach the network: {addr}")
        return real(self, addr)
    monkeypatch.setattr(socket.socket, "connect", guarded)


PAPER = """Memo on covenants

Courts enforce reasonable covenants (Ray, 2004). Burnout follows unclear notice
(Doe, 2023). Precaution has limits.[^1]

References

Ray, O. (2004). How the mind hurts and heals the body. American Psychologist, 59(1), 29–40. https://doi.org/10.1037/0003-066X.59.1.29

Doe, J. (2023). Quantum jurisprudence and paralegal burnout. Journal of Legal Studies, 12(1), 1-20.

Notes

[^1] Cass R. Sunstein, Beyond the Precautionary Principle, 151 U. Pa. L. Rev. 1003, 1010 (2003).
"""

RAY = {"paperId": "676d40a8", "externalIds": {"DOI": "10.1037/0003-066X.59.1.29"},
       "title": "How the mind hurts and heals the body.", "venue": "American Psychologist",
       "year": 2004, "publicationTypes": ["JournalArticle"],
       "journal": {"name": "The American psychologist", "volume": "59 1", "pages": " 29-40 "},
       "authors": [{"name": "O. Ray"}], "abstract": "Stress and illness."}
SUNSTEIN = {"paperId": "s1", "externalIds": {}, "title": "Beyond the Precautionary Principle",
            "venue": "University of Pennsylvania Law Review", "year": 2003,
            "publicationTypes": ["JournalArticle"], "journal": {"volume": "151", "pages": "1003"},
            "authors": [{"name": "Cass R. Sunstein"}], "abstract": ""}


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400

    def json(self):
        return self._body


@pytest.fixture
def scholar(monkeypatch):
    """Semantic Scholar knows Ray (by DOI and title) and Sunstein (by title);
    OpenAlex finds nothing; doi.org is not asked (no unknown DOI)."""
    calls = {"batch": [], "get": [], "openalex": []}

    def post_batch(ids, fields):
        calls["batch"].append(list(ids))
        return [RAY if i.lower() == "doi:10.1037/0003-066x.59.1.29" else None for i in ids]

    def get(path, params):
        calls["get"].append((path, params["query"]))
        q = params["query"].lower()
        if "mind hurts" in q:
            return _Resp(200, {"data": [RAY]})
        if "precautionary" in q:
            return _Resp(200, {"data": [SUNSTEIN]})
        return _Resp(404, {"error": "Title match not found"})

    def openalex(self, ctx, *, doi="", title="", year=None):
        calls["openalex"].append(doi or title)
        ctx.emissions["openalex"].append(f"title: {title}")
        return []
    monkeypatch.setattr(ssh, "post_batch", post_batch)
    monkeypatch.setattr(ssh, "get", get)
    monkeypatch.setattr(al.SourceChecker, "_openalex", openalex)
    monkeypatch.setattr(al.SourceChecker, "_doi_registered", lambda self, doi, ctx: None)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    return calls


@pytest.fixture
def courtlistener(monkeypatch):
    seen = []

    def lookup(cite, rate_limit=1.0):
        seen.append(cite["citation_text"])
        return {"found": False, "match_status": "not_found", "confidence_note": "no hits"}
    monkeypatch.setattr(cv, "lookup_citation", lookup)
    return seen


def _opts(**kw):
    return le.EvalOptions(check_history=False, support_reads=False, **kw)


def _by_title(src):
    return {(f.get("cited") or {}).get("title"): f for f in src["findings"] if le.is_academic(f)}


# ── check_citations ──────────────────────────────────────────────────────────

def test_academic_rows_beside_cases_and_the_law_review_is_not_a_case(scholar, courtlistener):
    src = le.check_citations(PAPER, _opts())
    by = _by_title(src)
    assert by["How the mind hurts and heals the body"]["status"] == "found"
    assert by["How the mind hurts and heals the body"]["match"]["via"] == "doi"
    doe = by["Quantum jurisprudence and paralegal burnout"]
    assert doe["status"] == "not_found" and "indexable_not_found" in doe["fabrication_signals"]
    assert doe["proposition"] == "Burnout follows unclear notice (Doe, 2023)."   # its own sentence only
    assert by["Beyond the Precautionary Principle"]["proposition"] == "Precaution has limits.[^1]"
    sun = by["Beyond the Precautionary Principle"]
    assert sun["status"] == "found" and sun["origin"] == "note"
    # eyecite's reading of "151 U. Pa. L. Rev. 1003" never went to CourtListener
    assert not any("L. Rev." in c for c in courtlistener)
    assert src["sources_count"] == 3 and src["cache_hits"] == 0
    assert src["emissions"]["semantic_scholar"][0] == "DOI:10.1037/0003-066x.59.1.29"
    assert "title: Quantum jurisprudence and paralegal burnout" in src["emissions"]["semantic_scholar"]
    assert src["scholarly"]["ss_key"] is False


def test_flags_and_problem_statuses(scholar, courtlistener):
    src = le.check_citations(PAPER, _opts())
    fl = le.source_flags(src)
    assert fl["sources"] == 3 and fl["source_problems"] == 1
    assert fl["possibly_fabricated"] == 1 and fl["reference_errors"] == 0
    assert fl["problems"] >= 1
    assert "found_conflated" in le.PROBLEM_STATUSES
    # case-only sources keep the old flag shape
    assert set(le.source_flags({"count": 0, "findings": []})) == {
        "citations", "problems", "unverifiable", "history_alerts", "unchecked"}


def test_source_check_text_for_the_rubric_prompt(scholar, courtlistener):
    txt = le.source_summary_text(le.check_citations(PAPER, _opts()))
    assert "Academic sources (3 checked):" in txt
    assert "- Ray (2004), “How the mind hurts and heals the body”: verified" in txt
    assert "Doe (2023), “Quantum jurisprudence and paralegal burnout”: not found in " \
           "Semantic Scholar or OpenAlex" in txt
    assert "Possibly fabricated — instructor to verify" in txt
    assert "fabricated" not in txt.split("Ray (2004)")[1].split("\n")[0]
    assert le.source_summary_text({"checked": True, "findings": [], "sources_count": 0}) == \
        "No citations or sources detected."


def test_check_sources_off_is_the_old_behaviour(scholar, courtlistener):
    src = le.check_citations(PAPER, _opts(check_sources=False))
    assert not any(le.is_academic(f) for f in src["findings"])
    assert "sources_count" not in src and scholar["batch"] == [] and scholar["get"] == []


def test_mismatch_and_its_label(scholar, courtlistener):
    paper = PAPER.replace("Ray, O. (2004)", "Ray, O. (2007)").replace("(Ray, 2004)", "(Ray, 2007)")
    f = _by_title(le.check_citations(paper, _opts()))["How the mind hurts and heals the body"]
    assert f["status"] == "found_mismatch"
    assert f["differences"][0]["field"] == "year"
    assert le.finding_label(f).startswith("exists, but the reference gets details wrong")
    assert le.finding_status(f) == "found_mismatch"     # never re-read as a case


def test_support_read_rides_the_local_judge(scholar, courtlistener):
    class Judge(le.LocalJudge):
        def __call__(self, prompt, kind):
            self.calls += 1
            return ("TOPICALLY_RELEVANT: NO\nACCURATELY_DESCRIBED: NO\nSUMMARY: s\nNOTE: N/A")
    opts = le.EvalOptions(check_history=False)
    src = le.check_citations(PAPER, opts, judge=Judge(opts))
    ray = _by_title(src)["How the mind hurts and heals the body"]
    assert ray["status"] == "found" and ray["support"]["status"] == "found_unrelated"
    assert src["judge_calls"] >= 1


def test_unparsed_entries_go_to_the_local_model_only(scholar, courtlistener, monkeypatch):
    seen = {}

    def chat(prompt, **kw):
        seen["prompt"] = prompt
        return {"entries": [{"index": 0, "title": "Notice and Assent Reconsidered",
                             "authors": ["Kim"], "year": "2020", "venue": "",
                             "kind": "journal_article"}]}
    monkeypatch.setattr(le, "chat_json", chat)
    paper = ("Body text (Kim, 2020).\n\nReferences\n\n"
             "kim notice and assent reconsidered twenty twenty journal of contracts\n")
    src = le.check_citations(paper.replace("twenty twenty", "Notice and Assent Reconsidered 2020"),
                             _opts())
    assert "Body text" not in seen["prompt"]
    assert any((f.get("cited") or {}).get("title") == "Notice and Assent Reconsidered"
               for f in src["findings"])


# ── reports ──────────────────────────────────────────────────────────────────

def _result(src):
    return {"file": "memo.txt", "stem": "memo", "model": "m", "preliminary": True,
            "flags": le.source_flags(src), "source": src, "result": None, "review": None,
            "errors": [], "redactions": {}, "timings": {}, "evaluated_at": "now"}


def test_render_md_has_the_academic_table(scholar, courtlistener):
    md = le.render_md(_result(le.check_citations(PAPER, _opts())))
    assert "## Academic sources" in md
    assert "| Reference | Verdict | Indexed record | What differs / why |" in md
    assert "**not_found**: not found in Semantic Scholar or OpenAlex" in md
    assert "(from Semantic Scholar)" in md
    assert "3 academic source(s), 1 problem(s)" in md
    assert "### What each source is cited for" in md


def _run(tmp_path, monkeypatch, root=None):
    monkeypatch.setattr(le, "list_models", lambda host: [le.DEFAULT_MODEL])
    monkeypatch.setattr(co, "chat_json", lambda p, **k: {
        "themes": [], "strengths": [], "announcement": "", "instructor_notes": ""})
    sub = tmp_path / "subs"
    sub.mkdir(exist_ok=True)
    (sub / "Rivera, Jordan - memo.txt").write_text(PAPER)
    (sub / "Kim, Ana - memo.txt").write_text(PAPER.replace("Ray, O.", "Ray, Oakley"))
    root = root or tmp_path / "res"
    return le.run_folder(sub, None, _opts(), root / "run1"), root / "run1"


def test_run_folder_writes_sources_everywhere_and_caches_lookups(tmp_path, monkeypatch,
                                                                  scholar, courtlistener):
    m, out = _run(tmp_path, monkeypatch)
    assert m["scholarly"]["ss_skipped"] is False
    summary = (out / "_summary.md").read_text()
    assert "Sources (problems)" in summary and "| 3 (1) |" in summary
    audit = [json.loads(line) for line in (out / "_emissions.jsonl").read_text().splitlines()]
    assert all("semantic_scholar" in a["research_lookups"] for a in audit[:1])
    cache = out.parent / al.CACHE_FILE
    assert cache.is_file()
    text = cache.read_text()
    assert "Rivera" not in text and "memo" not in text and "covenants" not in text
    # the second paper's identical sources came from the run's cache
    second = json.loads((out / "Rivera, Jordan - memo.json").read_text())
    first = json.loads((out / "Kim, Ana - memo.json").read_text())
    assert first["source"]["cache_hits"] + second["source"]["cache_hits"] >= 2
    calls_before = len(scholar["get"]) + len(scholar["batch"])
    le.run_folder(tmp_path / "subs", None, _opts(), out.parent / "run2")
    assert len(scholar["get"]) + len(scholar["batch"]) == calls_before   # all from the cache
    # the class view: the same unfindable source in both papers
    cohort = json.loads((out / "_cohort.json").read_text())
    agg = cohort["aggregate"]
    assert agg["shared_unfindable"][0]["citation"] == "Quantum jurisprudence and paralegal burnout"
    assert agg["shared_unfindable"][0]["students"] == 2
    assert agg["academic_status_counts"]["found"] == 4
    assert agg["academic_status_counts"]["not_found"] == 2
    lines = co.brief_prompt(agg, None)
    assert "journal article not found in Semantic Scholar or OpenAlex (2 student(s))" in lines
    assert "the same unfindable journal article appears in 2 students' papers" in lines


@pytest.fixture
def ui(tmp_path, monkeypatch):
    from firstread import ui as mod
    monkeypatch.setattr(mod, "RESULTS_ROOT", tmp_path / "res")
    monkeypatch.setattr(mod, "list_models", lambda host: ["gemma4:26b"])
    mod.RUNS.clear()
    mod._ACTIVE["run"] = None
    return mod


def test_ui_file_page_shows_academic_sources_and_what_was_sent(ui, tmp_path, monkeypatch,
                                                              scholar, courtlistener):
    _run(tmp_path, monkeypatch, root=ui._results_root())
    c = ui.app.test_client()
    page = c.get("/run/run1/file/Kim, Ana - memo").data.decode()
    assert "<h2>Academic sources</h2>" in page
    assert "possibly fabricated" in page and "indexable not found" in page
    assert "Semantic Scholar: DOI:10.1037/0003-066x.59.1.29" in page
    run_page = c.get("/run/run1").data.decode()
    assert "The same unfindable source in several papers" in run_page


def test_ui_index_offers_the_source_checks_and_says_there_is_no_key(ui, monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    page = ui.app.test_client().get("/").data.decode()
    assert 'name="check_sources"' in page and 'name="support_reads"' in page
    assert "no API key" in page and "SEMANTIC_SCHOLAR_API_KEY" in page
    assert "Author names, the sentence a source is cited for" in page
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "k")
    assert "API key set (1 request/s)" in ui.app.test_client().get("/").data.decode()


def test_ui_banner_when_semantic_scholar_was_skipped(ui):
    run = ui._results_root() / "r"
    run.mkdir(parents=True)
    (run / "_run.json").write_text(json.dumps({
        "run_id": "r", "status": "done", "files": [], "started_at": "t",
        "folder": "/tmp/subs", "scholarly": {"ss_skipped": True, "ss_skipped_after": "b.docx", "ss_skipped_sources": 4}}))
    page = ui.app.test_client().get("/run/r").data.decode()
    assert "Semantic Scholar was rate-limiting" in page and "b.docx" in page


# ── portfolio and debrief ────────────────────────────────────────────────────


# ── one owner per authority: cases in the reference list ─────────────────────

CASES_IN_REFS = """Memo on warnings

Police must warn a suspect first. Miranda v. Arizona, 384 U.S. 436 (1966). Juveniles have
due process rights too (In re Gault, 1967). Section 1983 gives a remedy. Stress matters
(Ray, 2004).

References

Brown v. Board of Education, 347 U.S. 483 (1954).

Civil Rights Act of 1871, 42 U.S.C. § 1983 (2018).

In re Gault, 387 U.S. 1 (1967).

Miranda v. Arizona, 384 U.S. 436 (1966).

Obergefell v. Hodges (2015).

Ray, O. (2004). How the mind hurts and heals the body. American Psychologist, 59(1), 29–40. https://doi.org/10.1037/0003-066X.59.1.29
"""


def test_a_case_in_the_reference_list_is_checked_once_and_not_as_an_academic_source(
        scholar, monkeypatch):
    looked_up, statutes = [], []

    def lookup(cite, rate_limit=1.0):
        looked_up.append(cite["citation_text"])
        return {"found": True, "case_name": "A Case", "courtlistener_url": "https://cl/x"}

    def statute(cite, rate_limit=1.0):
        statutes.append(cite["citation_text"])
        return {"found": True, "statute_name": "42 U.S.C. § 1983", "source_url": "https://lii/x"}
    monkeypatch.setattr(cv, "lookup_citation", lookup)
    monkeypatch.setattr(cv, "lookup_statute", statute)
    # the cases are never handed to the local model to be re-read as publications
    monkeypatch.setattr(le, "chat_json", lambda *a, **k: pytest.fail("model asked"))

    src = le.check_citations(CASES_IN_REFS, _opts())

    # each case went to CourtListener exactly once, including the two that
    # carry a reporter citation only in the reference list
    assert sorted(looked_up) == ["347 U.S. 483", "384 U.S. 436", "387 U.S. 1"]
    assert statutes == ["42 U.S.C. § 1983"]
    legal = {f["citation"]: f for f in src["findings"] if not le.is_academic(f)}
    assert all(f["status"] == "found" for f in legal.values()) and len(legal) == 4

    # nothing about a case or statute reached Semantic Scholar or OpenAlex
    sent = json.dumps([scholar, src["emissions"].get("semantic_scholar"),
                       src["emissions"].get("openalex")])
    for name in ("Miranda", "Gault", "Brown", "Obergefell", "Civil Rights", "U.S."):
        assert name not in sent, name

    academic = [f for f in src["findings"] if le.is_academic(f)]
    assert [f["status"] for f in academic if f["source_kind"] != "legal"] == ["found"]   # Ray
    ober = [f for f in academic if f["source_kind"] == "legal"]
    assert len(ober) == 1 and ober[0]["citation"] == "Obergefell v. Hodges (2015)."
    assert ober[0]["status"] == "unverifiable" and ober[0]["unverifiable_reason"] == "legal"
    assert not any(f["status"] == "not_found" for f in src["findings"])
    fl = le.source_flags(src)
    assert fl["possibly_fabricated"] == 0 and fl["source_problems"] == 0
    assert le.SOURCE_KIND_LABELS["legal"] == "legal authority"
