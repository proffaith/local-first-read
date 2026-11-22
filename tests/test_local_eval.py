"""Local evaluation pipeline (utils/local_eval.py) + the LLM injection seam
in utils/citation_verifier.py + the localhost UI (firstread/ui.py).

Everything network-bound (CourtListener, Ollama) is stubbed; the point is
that the production relevance / disposition prompts and parsers are reused
unchanged, that failures surface as ``unknown`` rather than vanishing, and
that the folder runner writes the files the UI reads back.
"""
import json
import sys
import re
import time
from pathlib import Path

import pytest

from firstread import citation_verifier as cv
from firstread import local_eval as le

REPO = Path(__file__).resolve().parent.parent


# ── injection seam ───────────────────────────────────────────────────────────

def test_check_relevance_injected_llm_miscited():
    seen = {}

    def llm(prompt, kind):
        seen["kind"] = kind
        seen["prompt"] = prompt
        return ("TOPICALLY_RELEVANT: YES\nACCURATELY_DESCRIBED: NO\n"
                "SUMMARY: About restrictive covenants.\n"
                "NOTE: The court ruled for the defendant.")

    r = cv.check_relevance("The court held for the plaintiff.", "Foo v. Bar",
                           "opinion text " * 50, llm=llm)
    assert r["status"] == "found_miscited"
    assert r["confidence_note"] == "The court ruled for the defendant."
    assert seen["kind"] == "relevance"
    assert "PASSAGE FROM BRIEF" in seen["prompt"]   # production prompt reused


def test_check_relevance_injected_llm_failure_is_unknown_not_fail_open():
    def llm(prompt, kind):
        raise RuntimeError("ollama down")

    r = cv.check_relevance("prop", "Foo v. Bar", "text", llm=llm)
    assert r["status"] == "unknown"


def test_check_relevance_needs_a_local_llm():
    """This snapshot has no cloud-model path: without an injected llm the
    judgment calls refuse rather than send text anywhere."""
    with pytest.raises(ValueError):
        cv.check_relevance("prop", "Foo v. Bar", "text")
    with pytest.raises(ValueError):
        cv._analyze_disposition("Low v. Er", "High v. Er", "text")
    # nothing to judge → no model needed
    assert cv.check_relevance("prop", "Foo v. Bar", "")["status"] == "found_relevant"


def test_parse_relevance_accepts_markdown_decoration():
    text = ("**TOPICALLY_RELEVANT:** NO\n- **ACCURATELY_DESCRIBED:** YES\n"
            "SUMMARY: x\nNOTE: N/A")
    assert cv._parse_relevance_response(text)["status"] == "found_unrelated"


def test_analyze_disposition_injected():
    r = cv._analyze_disposition(
        "Low v. Er", "High v. Er", "text",
        llm=lambda p, k: "**DISPOSITION:** Reversed and remanded\nNOTE: gone")
    assert r == {"disposition": "reversed", "note": "gone"}


def test_analyze_disposition_excerpt_reaches_the_citing_passage():
    """A long citing opinion whose only discussion of the cited case sits in
    the middle must still show that passage to the model (no head
    truncation)."""
    filler = "The procedural history of this matter is lengthy. " * 200
    mention = ("We find the reasoning of Leonard v. Pepsico persuasive and "
               "apply it here to the televised statement. ")
    text = filler + mention + filler
    seen = {}

    def llm(prompt, kind):
        seen["prompt"] = prompt
        return "DISPOSITION: DISCUSSED\nNOTE: applied"

    cv._analyze_disposition("Leonard v. Pepsico, Inc.", "Kolodziej v. Mason",
                            text, llm=llm)
    # (the chunker splits at "v." so the two halves arrive as adjacent
    # chunks — what matters is that the passage is present at all)
    assert "reasoning of Leonard" in seen["prompt"]
    assert "Pepsico persuasive and apply it here" in seen["prompt"]
    assert "[...]" in seen["prompt"]
    assert len(seen["prompt"]) < len(text)


# ── LocalJudge ───────────────────────────────────────────────────────────────

def test_local_judge_reemits_line_format(monkeypatch):
    seen = {}

    def fake_chat_json(prompt, **kw):
        seen["schema"] = kw["schema"]
        seen["prompt"] = prompt
        return {"topically_relevant": "YES",
                "accurately_described": "CANNOT_DETERMINE",
                "summary": "S", "note": ""}

    monkeypatch.setattr(le, "chat_json", fake_chat_json)
    j = le.LocalJudge(le.EvalOptions())
    out = j("prompt", "relevance")
    assert seen["schema"] is le.RELEVANCE_SCHEMA
    assert "JSON object" in seen["prompt"]
    assert cv._parse_relevance_response(out)["status"] == "found_relevant"
    assert (j.calls, j.failures) == (1, 0)

    monkeypatch.setattr(le, "chat_json",
                        lambda prompt, **kw: {"disposition": "VACATED",
                                              "note": "n"})
    assert cv._parse_disposition_response(j("p", "disposition")) == {
        "disposition": "vacated", "note": "n"}


def test_local_judge_counts_failures(monkeypatch):
    def boom(*a, **k):
        raise le.LocalLLMError("down")

    monkeypatch.setattr(le, "chat_json", boom)
    j = le.LocalJudge(le.EvalOptions())
    with pytest.raises(le.LocalLLMError):
        j("p", "disposition")
    assert j.failures == 1


# ── citation pipeline ────────────────────────────────────────────────────────

def _stub_verifier(monkeypatch, history_status="reversed"):
    # run_folder ends with the class-view brief (one model call) — never let a
    # test reach a real Ollama.
    monkeypatch.setattr(co, "chat_json", lambda p, **k: {
        "themes": [], "strengths": [], "announcement": "", "instructor_notes": ""})
    monkeypatch.setattr(cv, "extract_citations", lambda text: [
        {"citation_type": "case", "citation_text": "123 F.3d 456",
         "volume": "123", "reporter": "F.3d", "page": "456",
         "proposition": "The court held X."},
        {"citation_type": "case", "citation_text": "999 U.S. 1",
         "proposition": "Y."},
    ])

    def lookup(cite, rate_limit=1.0):
        if cite["citation_text"] == "999 U.S. 1":
            return {"found": False, "match_status": "not_found",
                    "confidence_note": "no hits"}
        return {"found": True, "case_name": "Foo v. Bar",
                "courtlistener_url": "https://cl/x", "court": "D. Md.",
                "court_id": "mdd", "cluster_id": "42",
                "date_filed": "2001-01-01", "opinion_snippet": "opinion text"}

    monkeypatch.setattr(cv, "lookup_citation", lookup)
    monkeypatch.setattr(
        cv, "check_relevance",
        lambda prop, name, snip, citation_type="case", llm=None: {
            "status": "found_miscited", "summary": "Says Z.",
            "confidence_note": "contradicts"})
    monkeypatch.setattr(
        cv, "check_subsequent_history",
        lambda cid, court, name, **kw: {
            "status": history_status,
            "summary": "Reversed by 4th Cir. in Bar v. Foo (2003)",
            "citing_cases": [{"citing_case_name": "Bar v. Foo",
                              "citing_court": "4th Cir.",
                              "citing_date_filed": "2003-05-05",
                              "disposition": "reversed",
                              "disposition_note": "n",
                              "courtlistener_url": "https://cl/y"}]})


def test_check_citations_pipeline(monkeypatch):
    _stub_verifier(monkeypatch)
    events = []
    opts = le.EvalOptions()
    src = le.check_citations("text", opts, judge=le.LocalJudge(opts),
                             progress=events.append)
    assert src["checked"] and src["count"] == 2
    f0, f1 = src["findings"]
    assert f0["status"] == "found_miscited"
    assert f0["note"] == "Says Z. contradicts"
    assert f0["history"]["status"] == "reversed"
    assert f1["status"] == "not_found" and f1["history"] is None
    assert src["emissions"]["courtlistener"] == ["123 F.3d 456", "999 U.S. 1"]
    assert le.source_flags(src) == {"citations": 2, "problems": 2,
                                    "unverifiable": 0,
                                    "history_alerts": 1, "unchecked": 0}
    txt = le.source_summary_text(src)
    assert "mischaracterizes" in txt
    assert "[subsequent history: Reversed by 4th Cir." in txt
    assert "possibly inaccurate or fabricated" in txt
    assert any(e["stage"] == "citations" for e in events)


def _stub_mixed(monkeypatch):
    """course-84 shapes: a real miss, the things the checker can't index,
    and dates eyecite misreads in APA references."""
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(cv, "extract_citations", lambda text: [
        {"citation_type": "case", "citation_text": "999 U.S. 1", "proposition": "Y."},
        {"citation_type": "statute", "citation_text": "42 Pa. Cons. Stat. § 7102",
         "reporter": "Pa. Cons. Stat.", "title": "42", "section": "7102",
         "proposition": "Comparative negligence."},
        {"citation_type": "statute", "citation_text": "42 U.S.C. § 99999",
         "reporter": "U.S.C.", "title": "42", "section": "99999", "proposition": ""},
        {"citation_type": "case", "citation_text": "2017 WL 1100915", "proposition": ""},
        {"citation_type": "case", "citation_text": "2025-Ohio-4521", "proposition": ""},
        {"citation_type": "case", "citation_text": "74 A.L.R. Fed. 2d 341", "proposition": ""},
        {"citation_type": "case", "citation_text": "10 Mar. 2015", "proposition": ""},
        {"citation_type": "case", "citation_text": "19 Mar., 1986", "proposition": ""},
    ])
    monkeypatch.setattr(cv, "lookup_citation", lambda cite, rate_limit=1.0: {
        "found": False, "match_status": "not_found", "confidence_note": "no hits"})
    monkeypatch.setattr(cv, "lookup_statute",
                        lambda cite, rate_limit=1.5: {"found": False})


def test_check_citations_separates_unverifiable_from_not_found(monkeypatch):
    _stub_mixed(monkeypatch)
    src = le.check_citations("t", le.EvalOptions(check_history=False))
    by = {f["citation"]: f for f in src["findings"]}
    # misread dates are not citations: not listed, not counted, not looked up
    assert "10 Mar. 2015" not in by and "19 Mar., 1986" not in by
    assert src["count"] == 6
    assert "10 Mar. 2015" not in src["emissions"]["courtlistener"]
    assert by["999 U.S. 1"]["status"] == "not_found"
    for c, reason in [("42 Pa. Cons. Stat. § 7102", "statute"),
                      ("42 U.S.C. § 99999", "statute"),
                      ("2017 WL 1100915", "database"),
                      ("2025-Ohio-4521", "database"),
                      ("74 A.L.R. Fed. 2d 341", "secondary")]:
        assert by[c]["status"] == "unverifiable", c
        assert by[c]["unverifiable_reason"] == reason, c
        assert by[c]["note"], c
    assert "Cornell LII" in by["42 U.S.C. § 99999"]["note"]
    assert "unverifiable" not in le.PROBLEM_STATUSES
    fl = le.source_flags(src)
    assert fl["problems"] == 1 and fl["unverifiable"] == 5
    assert "check by hand" in le.source_summary_text(src)


def test_statute_range_checks_first_section_without_relevance(monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(cv, "extract_citations", lambda text: [
        {"citation_type": "statute", "citation_text": "9 U.S.C. §§ 9-10",
         "reporter": "U.S.C.", "title": "9", "section": "9-10",
         "proposition": "Confirmation and vacatur of awards."},
        {"citation_type": "statute", "citation_text": "42 U.S.C. § 2000e-2",
         "reporter": "U.S.C.", "title": "42", "section": "2000e-2",
         "proposition": "Title VII."},
    ])
    looked_up = []

    def statute(cite, rate_limit=1.5):
        looked_up.append(cite["section"])
        return {"found": True, "statute_name": "s", "source_url": "u",
                "statute_text": "text"}
    monkeypatch.setattr(cv, "lookup_statute", statute)
    src = le.check_citations("t", le.EvalOptions(check_history=False))
    rng, single = src["findings"]
    assert looked_up == ["9", "2000e-2"]
    # one section's text can't support a range: exists, relevance not read
    assert rng["status"] == "found" and "§ 9 exists" in rng["note"]
    assert single["status"] == "found_miscited"      # stub relevance ran


def test_statute_range_start():
    f = cv.statute_range_start
    assert f({"citation_text": "9 U.S.C. §§ 9-10", "section": "9-10"}) == "9"
    assert f({"citation_text": "9 U.S.C. §§ 1-16", "section": "1-16"}) == "1"
    assert f({"citation_text": "9 U.S.C. §§ 10, 11", "section": "10, 11"}) == "10"
    # hyphens inside real section numbers are not ranges
    assert f({"citation_text": "42 U.S.C. § 2000e-2", "section": "2000e-2"}) == ""
    assert f({"citation_text": "42 U.S.C. §§ 2000e-2", "section": "2000e-2"}) == ""
    assert f({"citation_text": "9 U.S.C. § 9-10", "section": "9-10"}) == ""
    assert f({"citation_text": "9 U.S.C. § 2", "section": "2"}) == ""


def test_finding_status_reads_older_runs():
    fs = le.finding_status
    assert fs({"citation": "42 Pa. Cons. Stat. § 7102", "status": "not_found"}) == "unverifiable"
    assert fs({"citation": "2017 WL 1100915", "status": "not_found"}) == "unverifiable"
    assert fs({"citation": "12 A.3d 99", "status": "not_found"}) == "not_found"
    assert fs({"citation": "12 A.3d 99", "status": "found_unrelated"}) == "found_unrelated"
    assert le.source_flags({"findings": [
        {"citation": "2017 WL 1100915", "status": "not_found"},
        {"citation": "12 A.3d 99", "status": "not_found"}]})["problems"] == 1


def test_check_citations_lookup_exception_becomes_unknown(monkeypatch):
    _stub_verifier(monkeypatch)

    def boom(cite, rate_limit=1.0):
        raise RuntimeError("network")

    monkeypatch.setattr(cv, "lookup_citation", boom)
    src = le.check_citations("t", le.EvalOptions(check_history=False))
    assert [f["status"] for f in src["findings"]] == ["unknown", "unknown"]
    assert "network" in src["findings"][0]["note"]
    assert le.source_flags(src)["unchecked"] == 2


def test_history_good_standing_downgraded_when_judge_failed(monkeypatch):
    """The verifier maps a failed disposition call to 'other' → good_standing
    (production fail-open). Locally that must surface as 'unverified'."""
    _stub_verifier(monkeypatch)
    opts = le.EvalOptions()
    judge = le.LocalJudge(opts)

    def history(cid, court, name, **kw):
        judge.failures += 2      # what _analyze_disposition's except does
        return {"status": "good_standing", "summary": "Other by X in Y",
                "citing_cases": [{"disposition": "other"}] * 2}

    monkeypatch.setattr(cv, "check_subsequent_history", history)
    src = le.check_citations("t", opts, judge=judge)
    h = src["findings"][0]["history"]
    assert h["status"] == "unverified"
    assert h["summary"].startswith("2 citing opinion(s) could not be classified")
    assert le.source_flags(src)["unchecked"] == 1
    assert le.source_flags(src)["history_alerts"] == 0


def test_judge_and_grade_pass_think_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(le, "chat_json", lambda p, **kw: (
        seen.append(kw.get("think")) or {
            "disposition": "AFFIRMED", "note": "", "content_assessment": "",
            "rubric": [], "focus_for_instructor": [], "integrity_note": ""}))
    le.LocalJudge(le.EvalOptions())("p", "disposition")
    le.LocalJudge(le.EvalOptions(think=True))("p", "disposition")
    le.grade("t", PACK, "s", le.EvalOptions(think=True))
    assert seen == [False, True, True]


def test_check_citations_ambiguous(monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(cv, "lookup_citation", lambda cite, rate_limit=1.0: {
        "found": False, "match_status": "ambiguous",
        "confidence_note": "Two near matches.",
        "candidate_matches": [{"case_name": "A v. B"}, {"case_name": "C v. D"}]})
    src = le.check_citations("t", le.EvalOptions(check_history=False))
    f = src["findings"][0]
    assert f["status"] == "ambiguous"
    assert "A v. B; C v. D" in f["note"]
    assert le.source_flags(src)["problems"] == 2


# ── folder runner ────────────────────────────────────────────────────────────

PACK = {
    "assignment_id": 7, "title": "Memo 1", "course_id": 1,
    "assignment_ctx": "ctx", "points_possible": 10,
    "rubric_v2": {"groups": [{
        "name": "g",
        "levels": [{"name": "Good"}, {"name": "Poor"}],
        "criteria": [{"name": "Analysis", "cells": {"Good": "g", "Poor": "p"}},
                     {"name": "Citations", "cells": {}}]}]},
    "level_names": ["Good", "Poor"],
}


def test_run_folder_end_to_end(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    prompts = []

    def fake_chat_json(prompt, **kw):
        prompts.append(prompt)
        return {"content_assessment": "ok",
                "rubric": [{"criterion": "Citations", "suggested_level": "Poor",
                            "rationale": "r", "instructor_check": "c"}],
                "focus_for_instructor": ["f"], "integrity_note": "n"}

    monkeypatch.setattr(le, "chat_json", fake_chat_json)
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "alice.txt").write_text("Alice argues X. 123 F.3d 456. "
                                   "Reach me at alice@example.com")
    (sub / "~$lock.docx").write_bytes(b"x")
    (sub / "empty.txt").write_text("")
    (sub / "notes.xyz").write_text("ignored")
    out = tmp_path / "out"

    events = []
    m = le.run_folder(sub, PACK, le.EvalOptions(), out, progress=events.append,
                      run_id="t1", pack_path="p.json")
    assert m["status"] == "done" and m["run_id"] == "t1"
    by = {f["name"]: f for f in m["files"]}
    assert set(by) == {"alice.txt", "empty.txt"}
    assert by["alice.txt"]["status"] == "ok"
    assert by["alice.txt"]["levels"] == {"Citations": "Poor",
                                         "Analysis": "(not assessed)"}
    assert by["alice.txt"]["flags"]["problems"] == 2
    assert by["empty.txt"]["status"] == "skipped"

    # the citation findings reached the grading prompt
    assert "mischaracterizes" in prompts[0]

    rec = json.loads((out / "alice.json").read_text())
    assert rec["preliminary"] and rec["stem"] == "alice"
    assert rec["redactions"]   # the email was scrubbed and reported
    assert "alice@example.com" not in json.dumps(rec["source"])
    md = (out / "alice.md").read_text()
    assert "PRELIMINARY" in md and "REVERSED" in md and "Bar v. Foo" in md
    summary = (out / "_summary.md").read_text()
    assert "alice.txt" in summary and "Citations" in summary
    manifest = json.loads((out / "_run.json").read_text())
    assert manifest["status"] == "done" and manifest["pack"] == "p.json"
    audit = [json.loads(line) for line in
             (out / "_emissions.jsonl").read_text().splitlines()]
    assert audit[0]["destination"] == "localhost"
    assert audit[0]["research_lookups"]["courtlistener"] == [
        "123 F.3d 456", "999 U.S. 1"]
    assert [e["stage"] for e in events if e["stage"] == "file_done"] == [
        "file_done", "file_done"]


def test_run_folder_citations_only_without_pack(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    monkeypatch.setattr(le, "chat_json", lambda *a, **k: pytest.fail(
        "no grading call expected without a pack"))
    (tmp_path / "a.txt").write_text("123 F.3d 456")
    m = le.run_folder(tmp_path, None, le.EvalOptions(), tmp_path / "out")
    assert m["files"][0]["status"] == "ok" and m["files"][0]["levels"] == {}
    rec = json.loads((tmp_path / "out" / "a.json").read_text())
    assert rec["result"] is None and rec["flags"]["citations"] == 2


def test_list_runs_routes_by_directory_name(tmp_path):
    d = tmp_path / "cli-custom"
    d.mkdir()
    (d / "_run.json").write_text(json.dumps({"run_id": "20260903-1-ia1",
                                             "status": "done", "files": [],
                                             "started_at": "t"}))
    runs = le.list_runs(tmp_path)
    assert runs[0]["run_id"] == "cli-custom"
    assert runs[0]["outdir"] == str(d)


def test_run_folder_model_missing_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(le, "list_models", lambda host: ["other"])
    with pytest.raises(le.LocalLLMError):
        le.run_folder(tmp_path, None, le.EvalOptions(), tmp_path / "o")


def test_run_folder_stop_between_files(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    for n in ("a", "b", "c"):
        (tmp_path / f"{n}.txt").write_text("123 F.3d 456")
    seen = []

    def stop():
        return len(seen) >= 1

    def prog(ev):
        if ev["stage"] == "file_done":
            seen.append(ev["file"])

    m = le.run_folder(tmp_path, None, le.EvalOptions(check_history=False),
                      tmp_path / "out", progress=prog, should_stop=stop)
    assert m["status"] == "stopped"
    assert [f["status"] for f in m["files"]] == ["ok", "stopped", "stopped"]


# ── tracked changes + answer key ─────────────────────────────────────────────

def _redlined_docx(tmp_path) -> Path:
    """A DOCX whose second paragraph carries a tracked deletion and a tracked
    insertion, plus a table cell — the shapes python-docx's paragraph.text
    silently drops."""
    from docx import Document
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    doc = Document()
    doc.add_paragraph("Pennsylvania imposes a duty of reasonable care.")
    p = doc.add_paragraph("See ")
    p._p.append(parse_xml(
        f'<w:del {nsdecls("w")} w:id="1" w:author="s"><w:r><w:delText>'
        f'Carrender v. Fitterer, 469 Pa. 46 (1983)</w:delText></w:r></w:del>'))
    p._p.append(parse_xml(
        f'<w:ins {nsdecls("w")} w:id="2" w:author="s"><w:r><w:t>'
        f'Carrender v. Fitterer, 503 Pa. 178, 469 A.2d 120 (1983)</w:t></w:r></w:ins>'))
    p.add_run(".")
    t = doc.add_table(rows=1, cols=1)
    t.cell(0, 0).text = "Cell text survives"
    out = tmp_path / "Rivera, Jordan - redline (2).docx"
    doc.save(out)
    return out


def test_extract_docx_includes_tracked_insertions_and_tables(tmp_path):
    from firstread.writing_feedback import extract_draft_text, extract_docx_redline
    path = _redlined_docx(tmp_path)
    final = extract_draft_text(open(path, "rb"), filename=path.name)
    assert "503 Pa. 178, 469 A.2d 120" in final          # insertion kept
    assert "469 Pa. 46" not in final                       # deletion dropped
    assert "Cell text survives" in final                   # tables included
    r = extract_docx_redline(path.read_bytes())
    assert r["insertions"] == 1 and r["deletions"] == 1
    assert "[DELETED: Carrender v. Fitterer, 469 Pa. 46 (1983)]" in r["markup"]
    assert "[INSERTED: Carrender v. Fitterer, 503 Pa. 178" in r["markup"]
    assert "469 Pa. 46" in r["original"] and "503 Pa. 178" not in r["original"]
    sub = le.extract_submission(path)
    assert sub["insertions"] == 1 and "[INSERTED:" in sub["markup"]
    # the inserted citation is now visible to eyecite
    assert any(c["citation_text"] == "503 Pa. 178"
               for c in cv.extract_citations(sub["text"]))


def test_pdf_word_per_line_rejoin():
    from firstread.writing_feedback import _rejoin_word_per_line
    words = ("See Carrender v. Fitterer, 503 Pa. 178, 469 A.2d 120 (1983). "
             "The court held that a possessor owes invitees a duty of care.").split()
    # word\n \nword …, with a paragraph break rendered as a longer blank run
    broken = "\n \n".join(words[:13]) + "\n \n \n \n" + "\n \n".join(words[13:])
    broken = (broken + "\n \n") * 5          # enough lines to trip the threshold
    fixed = _rejoin_word_per_line(broken)
    assert "503 Pa. 178, 469 A.2d 120 (1983)." in fixed
    assert "\n\n" in fixed                    # paragraph break kept
    assert any(c["citation_text"] == "503 Pa. 178"
               for c in cv.extract_citations(fixed))
    # ordinary prose is left alone
    prose = "A normal paragraph with several words per line.\n" * 50
    assert _rejoin_word_per_line(prose) == prose


KEY = {
    "document_type": "legal_memo", "document_title": "MEMO — Rivera",
    "planted_errors": [
        {"id": "E1", "severity": "high", "category": "Misstated rule",
         "locator": "'strict liability'", "whats_wrong": "PA is negligence",
         "correct_fix": "duty of reasonable care"},
        {"id": "E2", "severity": "medium", "category": "Citation error",
         "locator": "'469 Pa. 46'", "whats_wrong": "wrong reporter",
         "correct_fix": "503 Pa. 178"},
        {"id": "E3", "severity": "high", "category": "Omitted element",
         "locator": "elements", "whats_wrong": "no causation",
         "correct_fix": "add causation"},
    ],
    "document_blocks": [{"text": "MEMORANDUM", "block_type": "heading"},
                        {"text": "Strict liability applies.", "block_type": "paragraph"}],
}
KEY_PACK = {**PACK, "assignment_id": 391, "title": "Redline the memo",
            "intentional_errors": KEY,
            "defective_document_text": "MEMORANDUM\n\nStrict liability applies."}


def test_detection_helpers():
    from firstread import intentional_errors as ie
    assert ie.render_document_text(KEY) == "MEMORANDUM\n\nStrict liability applies."
    prompt = ie.build_detection_prompt(KEY, "[INSERTED: fixed]", assignment_ctx="ctx",
                                       original_text="orig", has_tracked_changes=True)
    for needle in ("[E1] severity=high", "correct fix: 503 Pa. 178", "ctx",
                   "ORIGINAL DEFECTIVE DOCUMENT", "[INSERTED: fixed]",
                   "Track Changes", "fixed_silently"):
        assert needle in prompt
    schema = ie.detection_schema(["E1", "E2", "E3"])
    assert schema["properties"]["errors"]["items"]["properties"]["id"]["enum"] == ["E1", "E2", "E3"]
    raw = {"errors": [{"id": "E2", "status": "fixed_and_explained", "evidence": "e", "note": ""},
                      {"id": "E2", "status": "missed", "evidence": "", "note": "dup"},
                      {"id": "E1", "status": "bogus", "evidence": "", "note": ""},
                      {"id": "ZZ", "status": "missed", "evidence": "", "note": ""}],
           "additional_issues": [{"description": "typo", "assessment": "genuine_issue"}],
           "summary": "ok"}
    det = ie.tidy_detection(raw, KEY)
    assert [r["id"] for r in det["errors"]] == ["E1", "E2", "E3"]   # key order
    assert det["errors"][0]["status"] == "unclear"                  # bogus → unclear
    assert det["errors"][1]["status"] == "fixed_and_explained"     # first wins
    assert det["errors"][2]["note"].startswith("model returned no row")
    assert det["errors"][0]["severity"] == "high"
    assert (det["total"], det["detected"], det["high_total"], det["high_detected"]) == (3, 1, 2, 0)
    txt = ie.detection_summary_text(det)
    assert txt.startswith("ANSWER KEY CHECK: 1 of 3 planted errors detected (0/2 high-severity)")
    assert "E3 (high): unclear" in txt and "genuine issue" in txt


def test_evaluate_file_with_answer_key(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    prompts = []

    def fake_chat_json(prompt, **kw):
        prompts.append((prompt, kw))
        if "ANSWER KEY — PLANTED ERRORS" in prompt:
            assert "[INSERTED:" in prompt and "Track Changes" in prompt
            return {"errors": [{"id": "E2", "status": "fixed_silently",
                                "evidence": "503 Pa. 178", "note": ""}],
                    "additional_issues": [], "summary": "Only the cite."}
        assert "ANSWER KEY CHECK: 1 of 3" in prompt   # fed to the rubric read
        return {"content_assessment": "ok", "rubric": [],
                "focus_for_instructor": [], "integrity_note": ""}

    monkeypatch.setattr(le, "chat_json", fake_chat_json)
    path = _redlined_docx(tmp_path / "subs") if (tmp_path / "subs").mkdir() is None else None
    out = tmp_path / "out"
    m = le.run_folder(tmp_path / "subs", KEY_PACK, le.EvalOptions(check_history=False), out)
    f = m["files"][0]
    assert f["status"] == "ok"
    assert f["flags"]["key_total"] == 3 and f["flags"]["key_detected"] == 1
    rec = json.loads((out / f"{path.stem}.json").read_text())
    assert "_text" not in rec and "_markup" not in rec
    assert rec["planted"]["errors"][1]["status"] == "fixed_silently"
    assert rec["tracked_changes"] == {"insertions": 1, "deletions": 1}
    assert (out / f"{path.stem}.txt").read_text().startswith("Pennsylvania")
    assert "[DELETED:" in (out / f"{path.stem}.redline.txt").read_text()
    md = (out / f"{path.stem}.md").read_text()
    assert "Answer key: 1 of 3 planted errors detected" in md
    assert "unclear from this file" in md        # E1/E3 had no row → unclear
    assert "fixed in the document, not explained" in md
    assert "Answer key |" in (out / "_summary.md").read_text()
    assert "1/3" in (out / "_summary.md").read_text()
    # detection ran with a widened context window
    assert prompts[0][1]["num_ctx"] >= 32768


def test_evaluate_file_key_check_failure_is_reported(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])

    def fake_chat_json(prompt, **kw):
        if "ANSWER KEY — PLANTED ERRORS" in prompt:
            raise le.LocalLLMError("empty response")
        return {"content_assessment": "ok", "rubric": [],
                "focus_for_instructor": [], "integrity_note": ""}

    monkeypatch.setattr(le, "chat_json", fake_chat_json)
    (tmp_path / "a.txt").write_text("memo text")
    m = le.run_folder(tmp_path, KEY_PACK, le.EvalOptions(check_citations=False),
                      tmp_path / "out")
    f = m["files"][0]
    assert f["status"] == "ok" and f["errors"] == ["answer-key check failed: empty response"]
    assert f["flags"]["key_total"] == 3 and f["flags"]["key_detected"] is None


# ── cohort: pairing, heat map, brief ─────────────────────────────────────────

from firstread import local_eval_cohort as co  # noqa: E402


def test_student_key_parsing():
    d2l = "12345-67890 - Rivera, Jordan - Sep 3, 2026 1128 PM - Redline memo (2).docx"
    k = co.student_key(d2l)
    assert k == {"label": "Rivera, Jordan", "method": "d2l_prefix", "user_id": "12345"}
    roster = [{"sub": "Lopez, Ana", "name": "Lopez, Ana"},
              {"sub": "Kim Park", "name": "Kim Park"}]
    assert co.student_key("ALopez Redline AI Draft Memo.pdf", roster)["label"] == "Lopez, Ana"
    assert co.student_key("Ana Lopez - cover memo.docx", roster)["label"] == "Lopez, Ana"
    assert co.student_key("park kim memo.docx", roster)["label"] == "Kim Park"
    assert co.student_key("Untitled document.docx", roster) == {
        "label": "Untitled document", "method": "file"}


D2L_INDEX = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Assignment Submissions</title></head>
<body><h1>Redline the AI-Drafted Memo</h1>
<table>
<tr><th>Student</th><th>Submitted</th><th>Files</th></tr>
<tr><td>Alvarado, Luz</td><td>Sep 3, 2026 11:28 PM</td>
    <td><a href="LAlvarado%20Redline%20AI%20Draft%20Memo.pdf">LAlvarado Redline AI Draft Memo.pdf</a><br>
        <a href="./Redline the AI-Drafted Memo Find the Flaws (defective) Final.docx">Redline ... Final.docx</a></td></tr>
<tr><td>Park, Kim</td><td>Sep 3, 2026 9:02 PM</td><td><a href="sub/memo.docx">memo.docx</a></td></tr>
<tr><td>Rivera, Jordan</td><td>Sep 2, 2026 4:10 PM</td><td></td></tr>
<tr><td></td><td></td><td><a href="late.pdf">late.pdf</a></td></tr>
<tr><td>Totals</td><td>3</td><td><a href="https://example.edu/x.docx">external</a> <a href="notes.html">notes</a></td></tr>
</table></body></html>"""


def test_parse_submission_index(tmp_path):
    assert co.parse_submission_index(tmp_path) == {}          # no index.html
    (tmp_path / "Index.HTML").write_text(D2L_INDEX)
    idx = co.parse_submission_index(tmp_path)
    assert idx["names"] == ["Alvarado, Luz", "Park, Kim", "Rivera, Jordan"]
    f = idx["files"]
    assert f["LAlvarado Redline AI Draft Memo.pdf"] == "Alvarado, Luz"      # url-decoded
    assert f["Redline the AI-Drafted Memo Find the Flaws (defective) Final.docx"] == "Alvarado, Luz"
    assert f["sub/memo.docx"] == "Park, Kim" and f["memo.docx"] == "Park, Kim"
    assert f["late.pdf"] == "Rivera, Jordan"                  # name row, then file row
    assert "x.docx" not in f and "notes.html" not in f       # external + html ignored
    # index wins over prefix/roster/stem
    k = co.student_key("LAlvarado Redline AI Draft Memo.pdf", [{"name": "Lopez, Ana"}], idx)
    assert k == {"label": "Alvarado, Luz", "method": "index"}
    assert co.student_key("unknown.docx", None, idx)["method"] == "file"


def test_build_cohort_uses_index_html(tmp_path, monkeypatch):
    out, manifest = _cohort_fixture(tmp_path)
    subs = tmp_path / "subs"
    subs.mkdir()
    for f in manifest["files"]:
        f["name"] = f["stem"] + ".docx"
    (subs / "index.html").write_text(
        "<table><tr><td>Alpha, Ann</td><td><a href='a_redline.docx'>r</a>"
        "<a href='a_memo.docx'>m</a></td></tr>"
        "<tr><td>Beta, Bo</td><td><a href='b_redline.docx'>r</a></td></tr></table>")
    manifest["folder"] = str(subs)
    c = co.build_cohort(out, manifest, None, le.EvalOptions(), with_brief=False)
    assert [(s["label"], s["method"], len(s["files"])) for s in c["students"]] == [
        ("Alpha, Ann", "index", 2), ("Beta, Bo", "index", 1)]
    assert c["aggregate"]["index"] == {"found": True, "names": 2, "matched_files": 3}


def test_recursive_listing_and_collision_safe_stems(tmp_path, monkeypatch):
    (tmp_path / "Park, Kim").mkdir()
    (tmp_path / "Park, Kim" / "memo.txt").write_text("123 F.3d 456")
    (tmp_path / "Lee, Sam").mkdir()
    (tmp_path / "Lee, Sam" / "memo.txt").write_text("999 U.S. 1")
    (tmp_path / "_notes").mkdir()
    (tmp_path / "_notes" / "x.txt").write_text("skip")
    (tmp_path / "old_results").mkdir()
    (tmp_path / "old_results" / "_run.json").write_text("{}")
    (tmp_path / "old_results" / "memo.txt").write_text("skip")
    (tmp_path / "index.html").write_text("<p>not a submission</p>")
    files = le.list_submission_files(tmp_path)
    assert [le.submission_ids(tmp_path, p) for p in files] == [
        ("Lee, Sam/memo.txt", "Lee, Sam__memo"), ("Park, Kim/memo.txt", "Park, Kim__memo")]
    assert len(le.list_submission_files(tmp_path, recursive=False)) == 0
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    m = le.run_folder(tmp_path, None, le.EvalOptions(check_history=False), tmp_path / "out")
    assert [f["name"] for f in m["files"]] == ["Lee, Sam/memo.txt", "Park, Kim/memo.txt"]
    assert (tmp_path / "out" / "Lee, Sam__memo.json").exists()
    assert (tmp_path / "out" / "Park, Kim__memo.md").exists()
    rec = json.loads((tmp_path / "out" / "Park, Kim__memo.json").read_text())
    assert rec["file"] == "Park, Kim/memo.txt" and rec["stem"] == "Park, Kim__memo"


def test_detect_assignment_title_and_match(tmp_path, monkeypatch):
    d = tmp_path / "Redline the AI-Drafted Memo Find the Flaws Download Sep 4, 2026 738 AM"
    d.mkdir()
    det = le.detect_assignment_title(d)
    assert det == {"title": "Redline the AI-Drafted Memo Find the Flaws", "source": "folder name"}
    (d / "index.html").write_text(D2L_INDEX)      # <h1>Redline the AI-Drafted Memo</h1>
    det = le.detect_assignment_title(d)
    assert det == {"title": "Redline the AI-Drafted Memo", "source": "index.html heading"}
    (d / "index.html").write_text("<html><head><title>Assignment Submissions</title></head>"
                                  "<body><h1>Submissions: Memo 1 – Rodriguez</h1></body></html>")
    assert le.detect_assignment_title(d)["title"] == "Memo 1 – Rodriguez"
    packs = [{"path": "/p/ia391.json", "assignment_id": 391, "has_key": True, "has_rubric": True,
              "title": "Redline the AI-Drafted Memo: Find the Flaws"},
             {"path": "/p/ia198.json", "assignment_id": 198, "has_key": False, "has_rubric": True,
              "title": "Apply AI Tools to Draft a Legal Memo with Ethical Review"}]
    m = le.match_pack("Redline the AI-Drafted Memo Find the Flaws", packs)
    assert m["confident"] and m["best"]["assignment_id"] == 391 and m["score"] == 1.0
    m = le.match_pack("Redline the AI-Drafted Memo", packs)
    assert m["confident"] and m["best"]["assignment_id"] == 391        # containment
    m = le.match_pack("Contract drafting exercise", packs)
    assert m["best"] is None and not m["confident"]
    assert le.match_pack("anything", [])["best"] is None


def test_union_status():
    u = co._union_status
    assert u(["fixed_silently", "identified_not_fixed"]) == "fixed_and_explained"
    assert u(["fixed_silently", "missed"]) == "fixed_silently"
    assert u(["identified_not_fixed", "unclear"]) == "identified_not_fixed"
    assert u(["unclear", "missed"]) == "missed"
    assert u(["unclear"]) == "unclear"


def _cohort_fixture(tmp_path):
    """Two students: A submitted a redline + memo, B one file; three planted
    errors; a shared bad citation."""
    out = tmp_path / "run"
    out.mkdir()
    files = [
        ("12345-1 - Alpha, Ann - Sep 3, 2026 1100 AM - redline.docx", "a_redline",
         {"E1": "fixed_silently", "E2": "missed", "E3": "fixed_silently"}, ["469 Pa. 46"]),
        ("12345-2 - Alpha, Ann - Sep 3, 2026 1101 AM - memo.pdf", "a_memo",
         {"E1": "identified_not_fixed", "E2": "missed", "E3": "unclear"}, []),
        ("22222-3 - Beta, Bo - Sep 3, 2026 1102 AM - redline.docx", "b_redline",
         {"E1": "fixed_and_explained", "E2": "missed", "E3": "missed"}, ["469 Pa. 46", "1 U.S. 1"]),
    ]
    manifest = {"files": [], "pack": None, "status": "done"}
    for name, stem, statuses, bad in files:
        rec = {"file": name, "stem": stem,
               "planted": {"errors": [{"id": e, "status": s, "evidence": f"ev-{e}", "note": ""}
                                      for e, s in statuses.items()],
                           "additional_issues": [{"description": "conclusory", "assessment": "genuine_issue"}]},
               "source": {"findings": [{"citation": c, "status": "found_unrelated",
                                        "case_name": "Duprey", "note": "n", "history": None}
                                       for c in bad] + [{"citation": "503 Pa. 178", "status": "found_relevant",
                                                         "history": {"status": "skipped"}}]},
               "result": {"rubric": [{"criterion": "Detection", "suggested_level": "Proficient"}],
                          "focus_for_instructor": ["check burden of proof"], "integrity_note": ""}}
        (out / f"{stem}.json").write_text(json.dumps(rec))
        manifest["files"].append({"name": name, "stem": stem, "status": "ok",
                                  "levels": {"Detection": "Proficient"},
                                  "flags": {"citations": len(bad) + 1}})
    return out, manifest


def test_group_students_and_aggregate(tmp_path, monkeypatch):
    out, manifest = _cohort_fixture(tmp_path)
    seen = {}

    def fake_chat_json(prompt, **kw):
        seen["prompt"] = prompt
        return {"themes": [{"title": "Burden of proof", "prevalence": "missed by 2 of 2",
                            "what_to_teach": "Civil: preponderance.", "in_class_move": "poll"}],
                "strengths": ["Everyone fixed E1"],
                "announcement": "Hello class — nice work on ...", "instructor_notes": "check E3"}

    monkeypatch.setattr(co, "chat_json", fake_chat_json)
    c = co.build_cohort(out, manifest, KEY_PACK, le.EvalOptions())
    students = c["students"]
    assert [s["label"] for s in students] == ["Alpha, Ann", "Beta, Bo"]
    ann = students[0]
    assert len(ann["files"]) == 2 and ann["method"] == "d2l_prefix"
    st = {p["id"]: p["status"] for p in ann["planted"]}
    assert st == {"E1": "fixed_and_explained",    # fixed in redline + explained in memo
                  "E2": "missed", "E3": "fixed_silently"}
    assert ann["flags"]["key_detected"] == 2 and ann["flags"]["key_total"] == 3
    assert ann["flags"]["problems"] == 1          # de-duplicated across files
    agg = c["aggregate"]
    assert agg["students"] == 2 and agg["files"] == 3
    assert agg["heat"][0]["id"] == "E2" and agg["heat"][0]["missed"] == 2
    assert agg["heat"][0]["cells"] == {"Alpha, Ann": "missed", "Beta, Bo": "missed"}
    assert agg["citation_problems"][0] == {
        "citation": "469 Pa. 46", "students": 2, "status": "found_unrelated",
        "label": le.SOURCE_LABELS["found_unrelated"], "case_name": "Duprey", "note": "n"}
    assert agg["rubric_levels"] == {"Detection": {"Proficient": 3}}
    assert agg["extra_issues"] == [{"issue": "conclusory", "students": 3}]
    # brief prompt carries aggregates only — no student names
    assert "E2 [medium]" in seen["prompt"] and "missed by 2/2" in seen["prompt"]
    assert "Alpha" not in seen["prompt"] and "Beta" not in seen["prompt"]
    assert c["brief"]["themes"][0]["title"] == "Burden of proof"
    md = (out / "_next_class.md").read_text()
    assert "Planted errors × students" in md and "| E2 |" in md
    assert "Announcement draft" in md and "Hello class" in md
    assert (out / "_students.json").exists() and (out / "_cohort.json").exists()


def test_manual_pairing_overrides_automatic(tmp_path, monkeypatch):
    out, manifest = _cohort_fixture(tmp_path)
    # rename so nothing is auto-pairable, then group by hand
    for f in manifest["files"]:
        f["name"] = f["stem"] + ".docx"
    monkeypatch.setattr(co, "chat_json", lambda p, **k: {
        "themes": [], "strengths": [], "announcement": "", "instructor_notes": ""})
    auto = co.build_cohort(out, manifest, None, le.EvalOptions(), with_brief=False)
    assert [s["method"] for s in auto["students"]] == ["file"] * 3
    (out / co.PAIRING_FILE).write_text(json.dumps({
        "a_redline.docx": "Alpha, Ann", "a_memo.docx": "Alpha, Ann"}))
    manual = co.build_cohort(out, manifest, None, le.EvalOptions(), with_brief=False)
    assert [(s["label"], s["method"], len(s["files"])) for s in manual["students"]] == [
        ("Alpha, Ann", "manual", 2), ("b_redline", "file", 1)]
    st = {p["id"]: p["status"] for p in manual["students"][0]["planted"]}
    assert st["E1"] == "fixed_and_explained"          # union still applies


def test_build_cohort_survives_brief_failure(tmp_path, monkeypatch):
    out, manifest = _cohort_fixture(tmp_path)

    def boom(*a, **k):
        raise le.LocalLLMError("down")

    monkeypatch.setattr(co, "chat_json", boom)
    c = co.build_cohort(out, manifest, None, le.EvalOptions())
    assert c["brief"] is None and c["errors"] == ["brief failed: down"]
    assert len(c["students"]) == 2 and (out / "_next_class.md").exists()


def test_run_folder_builds_cohort(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    monkeypatch.setattr(le, "chat_json", lambda p, **k: {
        "content_assessment": "ok", "rubric": [], "focus_for_instructor": [],
        "integrity_note": ""})
    monkeypatch.setattr(co, "chat_json", lambda p, **k: {
        "themes": [], "strengths": [], "announcement": "Hi", "instructor_notes": ""})
    (tmp_path / "a.txt").write_text("123 F.3d 456")
    events = []
    m = le.run_folder(tmp_path, PACK, le.EvalOptions(check_history=False),
                      tmp_path / "out", progress=events.append)
    assert m["cohort"] == {"students": 1, "brief": True, "errors": []}
    assert [e["stage"] for e in events if e["stage"].startswith("cohort")] == [
        "cohort", "cohort_done"]
    assert (tmp_path / "out" / "_next_class.md").exists()


# ── calibration: override log, exemplars, agreement ─────────────────────────

from firstread import local_eval_calibration as cal  # noqa: E402


def _graded_record(levels, planted=None):
    return {"result": {"rubric": [{"criterion": c, "suggested_level": l}
                                  for c, l in levels.items()]},
            "planted": {"errors": [{"id": e, "status": s} for e, s in (planted or {}).items()]}}


def test_record_load_exemplars_and_stats(tmp_path):
    root = tmp_path / "res"
    out = root / "run1"
    out.mkdir(parents=True)
    rec = _graded_record({"Analysis": "Good", "Citations": "Poor"}, {"E1": "missed"})
    doc = cal.record_overrides(root, PACK, "run1", out, "alice", rec, {
        "rubric": {"Analysis": {"level": "Poor", "reason": "never applies the rule"},
                   "Citations": {"level": "Poor", "reason": ""},
                   "Bogus": {"level": "Good", "reason": "x"}},
        "planted": {"E1": {"status": "identified_not_fixed", "reason": "memo flags it"}}})
    assert doc["rubric"]["Analysis"] == {"model": "Good", "instructor": "Poor",
                                         "reason": "never applies the rule"}
    assert "Bogus" not in doc["rubric"]
    assert (out / "alice.overrides.json").exists()
    entries = cal.load_calibration(root, PACK["course_id"])
    assert len(entries) == 3
    assert not any("alice" in json.dumps(e) for e in entries)   # no stem, no text
    # re-saving the same file replaces, not duplicates
    cal.record_overrides(root, PACK, "run1", out, "alice", rec, {
        "rubric": {"Analysis": {"level": "Good", "reason": "on reflection fine"}}})
    entries = cal.load_calibration(root, PACK["course_id"])
    an = [e for e in entries if e["key"] == "Analysis"]
    assert len(an) == 1 and an[0]["instructor"] == "Good"
    # a second file, different assignment, disagreement with reason
    other = {**PACK, "assignment_id": 8, "title": "Memo 2"}
    cal.record_overrides(root, other, "run2", out, "bob",
                         _graded_record({"Analysis": "Good"}), {
                             "rubric": {"Analysis": {"level": "Poor", "reason": "conclusory"}}})
    entries = cal.load_calibration(root, PACK["course_id"])
    ex = cal.exemplars_for(entries, PACK["assignment_id"], "Analysis")
    assert ex[0]["assignment_id"] == 7                      # same assignment first
    block = cal.calibration_block(root, PACK, ["Analysis", "Citations", "Nope"])
    assert block.startswith("INSTRUCTOR CALIBRATION")
    assert "confirmed 'Good' — on reflection fine" in block
    assert "model suggested 'Good', instructor set 'Poor' — conclusory" in block
    assert "Nope" not in block
    assert cal.calibration_block(root, {**PACK, "course_id": 999}, ["Analysis"]) == ""
    stats = cal.agreement_stats(entries, ["Poor", "Good"])
    a = stats["rubric:Analysis"]
    assert (a["n"], a["exact"], a["adjacent"]) == (2, 1, 1)
    assert a["within_one_rate"] == 1.0 and a["tendency"] == "model reads higher"
    assert a["recent_disagreements"][0]["reason"] == "conclusory"
    assert stats["planted:E1"]["exact_rate"] == 0.0
    assert cal.list_courses(root)[0]["decisions"] == 4


def test_grade_injects_calibration(tmp_path, monkeypatch):
    seen = {}

    def fake(p, **k):
        seen["p"] = p
        return {"content_assessment": "", "rubric": [],
                "focus_for_instructor": [], "integrity_note": ""}

    monkeypatch.setattr(le, "chat_json", fake)
    le.grade("t", PACK, "s", le.EvalOptions(), calibration_text="INSTRUCTOR CALIBRATION — x")
    assert "INSTRUCTOR CALIBRATION — x" in seen["p"]
    assert seen["p"].index("INSTRUCTOR CALIBRATION") < seen["p"].index("SOURCE CHECK")


def test_run_folder_uses_calibration_root(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    root = tmp_path / "res"
    (root / "prior").mkdir(parents=True)
    cal.record_overrides(root, PACK, "prior", root / "prior", "x",
                         _graded_record({"Analysis": "Good"}),
                         {"rubric": {"Analysis": {"level": "Poor", "reason": "conclusory"}}})
    prompts = []
    monkeypatch.setattr(le, "chat_json", lambda p, **k: (prompts.append(p) or {
        "content_assessment": "", "rubric": [], "focus_for_instructor": [], "integrity_note": ""}))
    (tmp_path / "a.txt").write_text("123 F.3d 456")
    m = le.run_folder(tmp_path, PACK, le.EvalOptions(check_citations=False), root / "run1")
    assert m["files"][0]["status"] == "ok"
    assert any("instructor set 'Poor' — conclusory" in p for p in prompts)
    prompts.clear()
    le.run_folder(tmp_path, PACK, le.EvalOptions(check_citations=False, use_calibration=False),
                  root / "run2")
    assert not any("INSTRUCTOR CALIBRATION" in p for p in prompts)


# ── expected moves (story 276) ───────────────────────────────────────────────

from firstread import expected_moves as em  # noqa: E402


def test_moves_text_roundtrip_and_sidecar(tmp_path):
    text = ("# comment\n"
            "M1 | high | States the elements of negligence | looks like: all four named\n"
            "medium | Applies the rule to the facts\n"
            "Addresses the counterargument | looks like: comparative negligence answered\n"
            "\n")
    moves = em.parse_moves_text(text)
    assert [(m["id"], m["severity"], m["move"]) for m in moves] == [
        ("M1", "high", "States the elements of negligence"),
        ("M2", "medium", "Applies the rule to the facts"),
        ("M3", "medium", "Addresses the counterargument")]
    assert moves[0]["looks_like"] == "all four named"
    assert moves[2]["looks_like"] == "comparative negligence answered"
    again = em.parse_moves_text(em.moves_to_text(moves))
    assert again == moves
    pack_path = tmp_path / "ia7.json"
    pack_path.write_text(json.dumps(PACK))
    assert le.load_pack(pack_path).get("expected_moves") is None
    em.save_moves(pack_path, moves)
    assert (tmp_path / "ia7.moves.json").exists()
    loaded = le.load_pack(pack_path)
    assert [m["id"] for m in loaded["expected_moves"]["moves"]] == ["M1", "M2", "M3"]
    assert loaded["expected_moves"]["drafted_by"] == "instructor"


def test_moves_prompt_schema_tidy_summary():
    doc = {"moves": em.parse_moves_text(
        "M1 | high | States the rule\nM2 | medium | Applies it\nM3 | low | Polishes")}
    prompt = em.build_moves_prompt(doc, "the memo", assignment_ctx="ctx")
    for needle in ("[M1] severity=high — States the rule", "ctx", "the memo",
                   "present_weak", "naming a rule is not applying it"):
        assert needle in prompt
    assert em.moves_schema(["M1"])["properties"]["moves"]["items"]["properties"]["id"]["enum"] == ["M1"]
    det = em.tidy_moves({"moves": [
        {"id": "M2", "status": "present_weak", "evidence": "e", "note": "thin"},
        {"id": "M2", "status": "missing", "evidence": "", "note": "dup"},
        {"id": "M3", "status": "not_applicable", "evidence": "", "note": "n/a"},
        {"id": "M9", "status": "missing", "evidence": "", "note": ""}],
        "summary": "s"}, doc)
    assert [(r["id"], r["status"]) for r in det["moves"]] == [
        ("M1", "missing"), ("M2", "present_weak"), ("M3", "not_applicable")]
    assert (det["total"], det["present"], det["strong"],
            det["high_total"], det["high_present"]) == (2, 1, 0, 1, 0)
    txt = em.moves_summary_text(det)
    assert txt.startswith("EXPECTED MOVES CHECK: 1 of 2 moves present (0 strong; 0/1 high-severity)")
    assert "M1 (high) States the rule: MISSING" in txt
    draft = em.draft_moves_prompt(PACK)
    assert "ctx" in draft and "Criterion: Analysis" in draft


def test_evaluate_file_with_expected_moves(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    pack = {**PACK, "expected_moves": {"moves": em.parse_moves_text(
        "M1 | high | States the rule\nM2 | medium | Applies it")}}
    prompts = []

    def fake_chat_json(prompt, **kw):
        prompts.append(prompt)
        if "EXPECTED MOVES:" in prompt:
            return {"moves": [{"id": "M1", "status": "present_strong", "evidence": "q", "note": ""}],
                    "summary": "rule stated, never applied"}
        assert "EXPECTED MOVES CHECK: 1 of 2" in prompt     # fed to the rubric read
        return {"content_assessment": "ok", "rubric": [], "focus_for_instructor": [],
                "integrity_note": ""}

    monkeypatch.setattr(le, "chat_json", fake_chat_json)
    (tmp_path / "a.txt").write_text("memo")
    m = le.run_folder(tmp_path, pack, le.EvalOptions(check_citations=False), tmp_path / "out")
    f = m["files"][0]
    assert f["flags"]["moves_total"] == 2 and f["flags"]["moves_present"] == 1
    rec = json.loads((tmp_path / "out" / "a.json").read_text())
    assert [(r["id"], r["status"]) for r in rec["moves"]["moves"]] == [
        ("M1", "present_strong"), ("M2", "missing")]
    md = (tmp_path / "out" / "a.md").read_text()
    assert "Expected moves: 1 of 2 present" in md and "MISSING" in md
    assert "Moves |" in (tmp_path / "out" / "_summary.md").read_text()


def test_draft_expected_moves(monkeypatch):
    monkeypatch.setattr(le, "chat_json", lambda p, **k: {"moves": [
        {"move": "States the rule", "severity": "high", "looks_like": "x"},
        {"move": "", "severity": "low", "looks_like": ""},
        {"move": "Applies it", "severity": "bogus", "looks_like": "y"}]})
    moves = le.draft_expected_moves(PACK, le.EvalOptions())
    assert [(m["id"], m["severity"], m["move"]) for m in moves] == [
        ("M1", "high", "States the rule"), ("M2", "medium", "Applies it")]


# ── calibration study tooling (story 278) ────────────────────────────────────


# ── module context: what the student could have read ────────────────────────

from firstread import local_eval_context as mctx  # noqa: E402


def test_readings_become_module_context(tmp_path):
    (tmp_path / "readings").mkdir()
    (tmp_path / "readings" / "week2.md").write_text("duty breach causation damages")
    pack = {"title": "Memo 1", "module_title": "Torts", "readings": [
        {"title": "Week 2 notes", "file": "readings/week2.md"},
        {"title": "Handout", "text": "an offer is a manifestation of willingness"},
        "a bare string reading",
        {"title": "Missing", "file": "readings/nope.md"}]}
    mc = mctx.readings_to_module_context(pack, tmp_path)
    assert [i["title"] for i in mc["items"]] == ["Week 2 notes", "Handout", "Reading 3"]
    assert all(not i["excerpted"] for i in mc["items"])
    block = mctx.module_context_text({"module_context": mc})
    assert block.startswith("WHAT THE STUDENT COULD HAVE READ — Torts")
    assert "duty breach causation damages" in block
    assert mctx.readings_to_module_context({"readings": []}) is None


def test_readings_over_budget_keep_head_and_tail():
    filler = "Procedural history and unrelated matters. " * 400        # ~17k
    big = "THE RULE: duty, breach, causation, damages. " + filler + " IN SUM: four elements."
    mc = mctx.readings_to_module_context(
        {"readings": [{"title": "Big", "text": big}, {"title": "Small", "text": "short " * 50}]},
        budget_chars=8000)
    sec = mc["items"][0]
    assert sec["excerpted"] and sec["chars"] < sec["full_chars"]
    assert sec["text"].startswith("THE RULE") and sec["text"].rstrip().endswith("four elements.")
    assert "[... middle of reading omitted ...]" in sec["text"]
    assert not mc["items"][1]["excerpted"]                # under its minimum share


def test_hand_written_pack_is_normalized(tmp_path):
    raw = {"title": "Negligence memo", "course_title": "LGST 101",
           "task": "Write a predictive memo.", "criteria": "Rule, application, conclusion.",
           "objectives": ["Apply the elements of negligence"],
           "rubric": {"criteria": [{"name": "Analysis", "levels": {
               "Exemplary": "a", "Proficient": "b", "Developing": "c", "Beginning": "d"}}]},
           "readings": [{"title": "Notes", "text": "duty breach"}]}
    path = tmp_path / "negligence-memo.json"
    path.write_text(json.dumps(raw))
    pack = le.load_pack(path)
    assert pack["assignment_id"] == "negligence-memo"
    assert pack["course_id"] == "lgst-101"
    assert pack["rubric_v2"]["groups"][0]["criteria"][0]["name"] == "Analysis"
    assert pack["level_names"] == ["Beginning", "Developing", "Proficient", "Exemplary"]
    assert pack["assignment_ctx"].startswith("ASSIGNMENT: Negligence memo")
    assert "Task: Write a predictive memo." in pack["assignment_ctx"]
    assert "- Apply the elements of negligence" in pack["assignment_ctx"]
    assert pack["module_context"]["items"][0]["text"] == "duty breach"


def test_example_pack_in_repo_loads():
    pack = le.load_pack(REPO / "packs" / "example-memo.json")
    assert pack["rubric_v2"] and pack["level_names"] and pack["assignment_ctx"]
    assert pack["module_context"]["items"]


def test_module_context_text_and_digest_preference():
    pack = {"module_context": {"module": {"module_number": 3, "module_title": "Torts", "weeks": "Weeks 5-6"},
                               "items": [{"kind": "textbook_section", "title": "Negligence",
                                          "text": "full text here", "excerpted": True},
                                         {"kind": "lecture", "title": "Lecture 5",
                                          "text": "outline", "digest": "- key point"}]}}
    block = mctx.module_context_text(pack)
    assert block.startswith("WHAT THE STUDENT COULD HAVE READ — Module 3: Torts (Weeks 5-6)")
    assert "### textbook section: Negligence (excerpts)\nfull text here" in block
    assert "### lecture outline: Lecture 5 (digest)\n- key point" in block
    assert mctx.module_context_text({}) == ""
    assert mctx.module_context_text(pack, max_chars=80).endswith("[... module context truncated ...]")
    p = mctx.digest_prompt({"title": "Negligence", "text": "body"}, "Memo 1")
    assert "READING: Negligence" in p and "Memo 1" in p


def test_context_window_sizing():
    assert le.context_window_for("x" * 1000, 16384) == 16384
    assert le.context_window_for("x" * 100000, 16384) == 40960     # ~31k tokens + reply → 5×8k


def test_evaluate_file_carries_reading_block_to_every_prompt(tmp_path, monkeypatch):
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    pack = {**KEY_PACK, "expected_moves": {"moves": em.parse_moves_text("M1 | high | States the rule")},
            "module_context": {"module": {"module_title": "Torts"}, "items": [
                {"kind": "textbook_section", "title": "Negligence", "text": "DUTY BREACH CAUSATION"}]}}
    prompts = []

    def fake(prompt, **kw):
        prompts.append((prompt, kw["num_ctx"]))
        if "ANSWER KEY — PLANTED ERRORS" in prompt:
            return {"errors": [], "additional_issues": [], "summary": ""}
        if "EXPECTED MOVES:" in prompt:
            return {"moves": [], "summary": ""}
        return {"content_assessment": "", "rubric": [], "focus_for_instructor": [], "integrity_note": ""}

    monkeypatch.setattr(le, "chat_json", fake)
    (tmp_path / "a.txt").write_text("memo")
    m = le.run_folder(tmp_path, pack, le.EvalOptions(check_citations=False), tmp_path / "out")
    assert m["files"][0]["status"] == "ok"
    assert len(prompts) == 3
    assert all("WHAT THE STUDENT COULD HAVE READ — Torts" in p and "DUTY BREACH CAUSATION" in p
               for p, _ in prompts)
    assert all(n >= 16384 for _, n in prompts)
    rec = json.loads((tmp_path / "out" / "a.json").read_text())
    assert rec["timings"]["module_context_chars"] > 0


def test_digest_module_context_writes_pack(tmp_path, monkeypatch):
    pack_path = tmp_path / "ia7.json"
    pack_path.write_text(json.dumps({**PACK, "module_context": {"module": {}, "items": [
        {"kind": "textbook_section", "title": "A", "text": "long text"},
        {"kind": "lecture", "title": "B", "text": "outline"}]}}))
    def fake(p, **k):
        title = p.split("READING: ")[1].split("\n")[0]
        if title == "B":                       # degenerate reply
            return {"sections": [{"heading": title, "bullets": []}]}
        return {"sections": [{"heading": "Rules", "bullets": [f"point about {title}"]},
                             {"heading": "", "bullets": ["second point", " "]}]}

    monkeypatch.setattr(le, "chat_json", fake)
    mc = le.digest_module_context(pack_path, le.EvalOptions())
    saved = json.loads(pack_path.read_text())["module_context"]["items"]
    assert saved[0]["digest"] == "## Rules\n- point about A\n\n- second point"
    assert saved[0]["digest_model"] == "gemma4:26b"
    assert "digest" not in saved[1] and saved[1]["digest_error"].startswith("digest too short")
    assert mc["digest_rejected"] == ["B"] and mc["digest_chars"] == len(saved[0]["digest"])
    block = mctx.module_context_text({"module_context": {"module": {}, "items": saved}})
    assert "(digest)\n## Rules" in block and "B (full text)\noutline" in block
    assert mctx.digest_acceptable("x" * 700, "y" * 5000) and not mctx.digest_acceptable("x" * 300, "y" * 5000)
    assert mctx.digest_acceptable("x" * 60, "y" * 100)   # tiny source: half is enough


# ── UI ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def ui(tmp_path, monkeypatch):
    from firstread import ui as mod
    monkeypatch.setattr(mod, "RESULTS_ROOT", tmp_path / "res")
    monkeypatch.setattr(mod, "list_models", lambda host: ["gemma4:26b"])
    mod.RUNS.clear()
    mod._ACTIVE["run"] = None
    return mod


def test_ui_index_renders(ui):
    r = ui.app.test_client().get("/")
    assert r.status_code == 200
    assert b"Evaluate folder" in r.data
    assert b"gemma4:26b" in r.data


def test_ui_start_validates_folder(ui, tmp_path):
    c = ui.app.test_client()
    r = c.post("/start", data={"folder": str(tmp_path / "nope"),
                               "check_citations": "on"})
    assert r.status_code == 400 and b"Not a folder" in r.data
    empty = tmp_path / "empty"
    empty.mkdir()
    r = c.post("/start", data={"folder": str(empty), "check_citations": "on"})
    assert r.status_code == 400 and b"No docx" in r.data


def test_ui_file_page_reads_older_runs_unverifiable(ui):
    """A run made before "unverifiable" existed stored statutes / WL numbers
    as not_found and counted them as problems; the file page recounts."""
    run = ui._results_root() / "old-run"
    run.mkdir(parents=True)
    (run / "_run.json").write_text(json.dumps({
        "run_id": "old-run", "status": "done", "files": [], "started_at": "t"}))
    (run / "a.json").write_text(json.dumps({
        "file": "a.txt", "stem": "a", "model": "m", "preliminary": True,
        "flags": {"citations": 2, "problems": 2, "history_alerts": 0,
                  "unchecked": 0},
        "source": {"checked": True, "count": 2, "findings": [
            {"citation": "42 Pa. Cons. Stat. § 7102", "status": "not_found",
             "proposition": "", "note": "", "history": None},
            {"citation": "12 A.3d 99", "status": "not_found",
             "proposition": "", "note": "", "history": None}]},
        "result": None, "review": None, "errors": [], "redactions": {},
        "timings": {}, "evaluated_at": "now"}))
    page = ui.app.test_client().get("/run/old-run/file/a").data.decode()
    assert "1 problem(s)" in page and "1 couldn't verify" in page
    assert "could not be verified automatically" in page
    assert page.count("possibly inaccurate or fabricated") == 1


def test_ui_start_run_and_read_back(ui, tmp_path, monkeypatch):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "a.txt").write_text("x")
    calls = {}

    def fake_run_folder(folder, pack, opts, outdir, progress=None,
                        should_stop=None, run_id="", pack_path="", files=None):
        calls["opts"] = opts
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        progress({"stage": "file", "file": "a.txt", "index": 1, "total": 1,
                  "message": "starting"})
        rec = {"file": "a.txt", "stem": "a", "model": opts.model,
               "preliminary": True, "flags": {"citations": 1, "problems": 1,
                                              "history_alerts": 0,
                                              "unchecked": 0},
               "source": {"checked": True, "count": 1, "findings": [
                   {"citation": "1 U.S. 1", "status": "not_found",
                    "proposition": "p", "type": "case", "case_name": "",
                    "url": "", "note": "no hits", "history": None}],
                   "emissions": {"courtlistener": ["1 U.S. 1"]}},
               "result": None, "review": None, "errors": [],
               "redactions": {}, "timings": {"total_s": 1.0},
               "chars_in": 1, "evaluated_at": "now"}
        (outdir / "a.json").write_text(json.dumps(rec))
        (outdir / "a.md").write_text("# a\n")
        progress({"stage": "file_done", "file": "a.txt", "status": "ok",
                  "flags": rec["flags"], "levels": {}, "message": "done"})
        files = [{"name": "a.txt", "stem": "a", "status": "ok",
                  "flags": rec["flags"], "levels": {}}]
        (outdir / "_run.json").write_text(json.dumps({
            "run_id": run_id, "folder": str(folder), "status": "done",
            "model": opts.model, "files": files, "started_at": "t"}))
        return {"status": "done", "files": files}

    monkeypatch.setattr(ui, "run_folder", fake_run_folder)
    c = ui.app.test_client()
    r = c.post("/start", data={"folder": str(sub), "pack": "none",
                               "model": "gemma4:26b",
                               "check_citations": "on", "check_history": "on",
                               "max_history_checks": "2"})
    assert r.status_code == 302
    run_id = r.headers["Location"].rstrip("/").rsplit("/", 1)[-1]

    for _ in range(100):
        s = c.get(f"/run/{run_id}/status").get_json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert s["status"] == "done", s
    assert s["files"][0]["flags"]["problems"] == 1
    assert calls["opts"].check_history and calls["opts"].max_history_checks == 2
    assert not calls["opts"].with_review

    assert c.get(f"/run/{run_id}").status_code == 200
    page = c.get(f"/run/{run_id}/file/a")
    assert page.status_code == 200 and b"1 U.S. 1" in page.data
    assert b"possibly inaccurate or fabricated" in page.data
    assert c.get(f"/run/{run_id}/md/a").data.startswith(b"# a")
    assert c.get(f"/run/{run_id}/file/../a").status_code == 404
    assert run_id.encode() in c.get("/runs").data

    # after a restart the run is still readable from disk
    ui.RUNS.clear()
    assert c.get(f"/run/{run_id}").status_code == 200
    assert c.get("/").status_code == 200      # no active run → form enabled


def test_ui_results_root_is_configurable_and_remembered(ui, tmp_path, monkeypatch):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "a.txt").write_text("x")
    tenant = tmp_path / "OneDrive" / "reviews"      # does not exist yet

    def fake_run_folder(folder, pack, opts, outdir, progress=None,
                        should_stop=None, run_id="", pack_path="", files=None):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "_run.json").write_text(json.dumps({
            "run_id": run_id, "folder": str(folder), "status": "done",
            "model": opts.model, "files": [], "started_at": "t"}))
        return {"status": "done", "files": []}

    monkeypatch.setattr(ui, "run_folder", fake_run_folder)
    c = ui.app.test_client()
    r = c.post("/start", data={"folder": str(sub), "check_citations": "on",
                               "pack": "none", "results_root": str(tenant)})
    assert r.status_code == 302
    run_id = r.headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    for _ in range(100):
        if (tenant / run_id / "_run.json").exists():
            break
        time.sleep(0.02)
    assert (tenant / run_id / "_run.json").exists()       # landed in the tenant
    assert ui._results_root() == tenant                   # remembered
    assert str(tenant).encode() in c.get("/").data        # pre-filled next time
    ui.RUNS.clear()
    assert c.get(f"/run/{run_id}").status_code == 200     # routes follow it
    assert run_id.encode() in c.get("/runs").data
    # refusing the submissions folder itself
    r = c.post("/start", data={"folder": str(sub), "check_citations": "on",
                               "pack": "none", "results_root": str(sub)})
    assert r.status_code == 400 and b"must not be the submissions folder" in r.data


def test_ui_class_view_and_student_page(ui, tmp_path, monkeypatch):
    out, manifest = _cohort_fixture(tmp_path)
    run_id = "cohort-run"
    d = ui.RESULTS_ROOT / run_id
    d.mkdir(parents=True)
    for p in out.iterdir():
        (d / p.name).write_text(p.read_text())
    manifest.update({"run_id": run_id, "folder": "/x", "model": "m", "started_at": "t"})
    (d / "_run.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(co, "chat_json", lambda p, **k: {
        "themes": [{"title": "Burden", "prevalence": "2 of 2", "what_to_teach": "t",
                    "in_class_move": "m"}], "strengths": [],
        "announcement": "Hello class", "instructor_notes": ""})
    co.build_cohort(d, manifest, None, le.EvalOptions())
    c = ui.app.test_client()
    page = c.get(f"/run/{run_id}").data
    assert b"Class summary" in page and b"By student" in page
    assert b"Alpha, Ann" in page and b"Hello class" in page
    assert b"Planted errors" in page and b"469 Pa. 46" in page
    # the collapsed summary comes before the submissions
    summary = page.index(b'<section id="class" aria-label="Class summary" hidden>')
    assert summary < page.index(b"<h2>Submissions</h2>")
    from urllib.parse import quote
    sp = c.get(f"/run/{run_id}/student/{quote('Alpha, Ann')}").data
    assert b"paired by d2l_prefix" in sp and b"found, explained and fixed" in sp
    assert c.get(f"/run/{run_id}/student/Nobody").status_code == 404
    assert c.get(f"/run/{run_id}/next-class").data.startswith(b"# ")
    # manual regroup through the form: move Bo's file under Ann, then undo
    name_b = manifest["files"][2]["name"]
    r = c.post(f"/run/{run_id}/pairing", data={f"label:{name_b}": "Alpha, Ann",
                                                "label:not-a-file": "x"})
    assert r.status_code == 302
    pairing = json.loads((d / co.PAIRING_FILE).read_text())
    assert pairing == {name_b: "Alpha, Ann"}
    students = json.loads((d / "_students.json").read_text())
    assert [(s["label"], len(s["files"])) for s in students] == [("Alpha, Ann", 3)]
    c.post(f"/run/{run_id}/pairing", data={f"label:{name_b}": ""})
    students = json.loads((d / "_students.json").read_text())
    assert [s["label"] for s in students] == ["Alpha, Ann", "Beta, Bo"]


def test_ui_decisions_form_and_calibration_page(ui, tmp_path):
    run_id = "cal-run"
    d = ui.RESULTS_ROOT / run_id
    d.mkdir(parents=True)
    pack_path = tmp_path / "ia7.json"
    pack_path.write_text(json.dumps(PACK))
    (d / "_run.json").write_text(json.dumps({
        "run_id": run_id, "folder": "/x", "status": "done", "model": "m",
        "pack": str(pack_path), "started_at": "t",
        "files": [{"name": "a.txt", "stem": "a", "status": "ok"}]}))
    rec = {"file": "a.txt", "stem": "a", "model": "m", "flags": {}, "source": {},
           "result": {"content_assessment": "ok",
                      "rubric": [{"criterion": "Analysis", "suggested_level": "Good",
                                  "rationale": "r", "instructor_check": "c"}],
                      "focus_for_instructor": [], "integrity_note": ""},
           "planted": None, "review": None, "errors": [], "redactions": {},
           "timings": {}, "chars_in": 1, "evaluated_at": ""}
    (d / "a.json").write_text(json.dumps(rec))
    c = ui.app.test_client()
    page = c.get(f"/run/{run_id}/file/a").data
    assert b"Your decisions" in page and b'name="level:Analysis"' in page
    r = c.post(f"/run/{run_id}/file/a/overrides", data={
        "level:Analysis": "Poor", "reason:Analysis": "conclusory"})
    assert r.status_code == 302
    page = c.get(f"/run/{run_id}/file/a").data
    assert b"Last saved" in page and b"conclusory" in page and b"0/1 exact" in page
    cp = c.get("/calibration").data
    assert b"course 1" in cp and b"Analysis" in cp and b"conclusory" in cp
    assert b"model reads higher" not in cp      # tendency needs 3+ decisions
    assert c.get("/calibration?course=1").status_code == 200


def test_ui_packs_and_moves_editor(ui, tmp_path, monkeypatch):
    packs_dir = tmp_path / "packs"
    packs_dir.mkdir()
    (packs_dir / "ia7.json").write_text(json.dumps(PACK))
    def fake_discover():
        pk = json.loads((packs_dir / "ia7.json").read_text())
        mc = pk.get("module_context") or {}
        return [{"path": str(packs_dir / "ia7.json"), "assignment_id": 7, "title": "Memo 1",
                 "course_title": "C", "has_rubric": True, "has_key": False,
                 "has_moves": (packs_dir / "ia7.moves.json").exists(), "exported_at": "",
                 "readings": len(mc.get("items") or []),
                 "reading_chars": mc.get("total_chars", 0),
                 "digested": bool(mc.get("digest_chars"))}]

    monkeypatch.setattr(ui, "discover_packs", fake_discover)
    c = ui.app.test_client()
    assert b"Memo 1" in c.get("/packs").data
    page = c.get("/packs/7/moves").data
    assert b"Expected moves" in page and b"Draft with the local model" in page
    # draft: shown in the editor, not saved
    monkeypatch.setattr(ui, "_model_status", lambda: (["gemma4:26b"], None))
    import firstread.local_eval as le_mod
    monkeypatch.setattr(le_mod, "chat_json", lambda p, **k: {"moves": [
        {"move": "States the rule", "severity": "high", "looks_like": "x"}]})
    r = c.post("/packs/7/moves", data={"action": "draft", "moves_text": ""})
    assert r.status_code == 200 and b"M1 | high | States the rule" in r.data
    assert b"draft from the local model" in r.data
    assert not (packs_dir / "ia7.moves.json").exists()
    # save
    r = c.post("/packs/7/moves", data={"action": "save",
                                       "moves_text": "M1 | high | States the rule\nApplies it"})
    assert r.status_code == 302
    doc = json.loads((packs_dir / "ia7.moves.json").read_text())
    assert [m["move"] for m in doc["moves"]] == ["States the rule", "Applies it"]
    assert b"Saved" in c.get("/packs/7/moves").data
    # blank removes
    c.post("/packs/7/moves", data={"action": "save", "moves_text": ""})
    assert not (packs_dir / "ia7.moves.json").exists()
    assert c.get("/packs/999/moves").status_code == 404


def test_ui_detects_assignment_and_uses_pack(ui, tmp_path, monkeypatch):
    packs_dir = tmp_path / "packs"
    packs_dir.mkdir()
    pack_path = packs_dir / "ia391.json"
    pack_path.write_text(json.dumps({**PACK, "assignment_id": 391,
                                     "title": "Redline the AI-Drafted Memo: Find the Flaws"}))
    monkeypatch.setattr(ui, "discover_packs", lambda: [{
        "path": str(pack_path), "assignment_id": 391, "title": "Redline the AI-Drafted Memo: Find the Flaws",
        "course_title": "BUSL 101", "has_rubric": True, "has_key": True, "has_moves": False, "exported_at": ""}])
    import firstread.local_eval as le_mod
    monkeypatch.setattr(le_mod, "discover_packs", ui.discover_packs)
    sub = tmp_path / "Redline the AI-Drafted Memo Find the Flaws Download Sep 4, 2026 738 AM"
    sub.mkdir()
    (sub / "a.txt").write_text("x")
    c = ui.app.test_client()
    j = c.get("/detect?folder=" + str(sub)).get_json()
    assert j["title"] == "Redline the AI-Drafted Memo Find the Flaws"
    assert j["match"]["assignment_id"] == 391 and j["match"]["confident"] and j["files"] == 1
    assert c.get("/detect?folder=/nope").status_code == 400
    seen = {}

    def fake_run_folder(folder, pack, opts, outdir, progress=None, should_stop=None,
                        run_id="", pack_path="", files=None):
        seen["pack"] = pack and pack["assignment_id"]
        Path(outdir).mkdir(parents=True, exist_ok=True)
        (Path(outdir) / "_run.json").write_text(json.dumps({"run_id": run_id, "folder": str(folder),
                                                             "status": "done", "files": [], "started_at": "t"}))
        return {"status": "done", "files": []}

    monkeypatch.setattr(ui, "run_folder", fake_run_folder)
    # pack left blank → detected and used
    r = c.post("/start", data={"folder": str(sub), "pack": "", "check_citations": "on"})
    assert r.status_code == 302
    run_id = r.headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    for _ in range(100):
        if seen.get("pack"):
            break
        time.sleep(0.02)
    assert seen["pack"] == 391
    assert c.get(f"/run/{run_id}/status").get_json()["pack_auto"]["title"].startswith("Redline")
    # explicit citations-only still works
    seen.clear()
    ui._ACTIVE["run"] = None
    r = c.post("/start", data={"folder": str(sub), "pack": "none", "check_citations": "on"})
    assert r.status_code == 302
    for _ in range(100):
        if "pack" in seen:
            break
        time.sleep(0.02)
    assert seen["pack"] is None
    # unrecognised folder with no pack → asked to choose
    ui._ACTIVE["run"] = None
    other = tmp_path / "Contract drafting exercise"
    other.mkdir()
    (other / "a.txt").write_text("x")
    r = c.post("/start", data={"folder": str(other), "pack": "", "check_citations": "on"})
    assert r.status_code == 400 and b"example-memo.json" in r.data


def test_ui_file_links_with_real_world_names(ui, tmp_path):
    """Submission stems carry spaces, commas, parentheses and dashes; the
    result routes must serve them and still refuse path escapes."""
    from urllib.parse import quote
    run_id = "20260903-1-Redline the Memo (draft)"
    stem = "Rivera, Jordan - Memo 1 (2)"
    d = ui.RESULTS_ROOT / run_id
    d.mkdir(parents=True)
    (d / "_run.json").write_text(json.dumps({
        "run_id": run_id, "folder": "/x", "status": "done", "model": "m",
        "started_at": "t",
        "files": [{"name": stem + ".docx", "stem": stem, "status": "ok",
                   "flags": {}, "levels": {}}]}))
    (d / f"{stem}.json").write_text(json.dumps({
        "file": stem + ".docx", "stem": stem, "model": "m", "flags": {},
        "source": {}, "result": None, "review": None, "errors": [],
        "redactions": {}, "timings": {}, "chars_in": 0, "evaluated_at": ""}))
    (d / f"{stem}.md").write_text("# ok\n")
    (d / f"{stem}.txt").write_text("seen\n")
    c = ui.app.test_client()
    assert c.get(f"/run/{quote(run_id)}").status_code == 200
    assert c.get(f"/run/{quote(run_id)}/file/{quote(stem)}").status_code == 200
    assert c.get(f"/run/{quote(run_id)}/md/{quote(stem)}").data == b"# ok\n"
    assert c.get(f"/run/{quote(run_id)}/txt/text/{quote(stem)}").data == b"seen\n"
    assert c.get(f"/run/{quote(run_id)}/txt/redline/{quote(stem)}").status_code == 404
    assert c.get(f"/run/{quote(run_id)}/txt/other/{quote(stem)}").status_code == 404
    # escapes still refused
    outside = tmp_path / "secret.json"
    outside.write_text("{}")
    assert c.get(f"/run/{quote(run_id)}/file/{quote('../../secret')}").status_code == 404
    assert c.get(f"/run/{quote('..')}/file/x").status_code == 404
    assert c.get(f"/run/{quote(run_id)}/file/{quote('..%2Fsecret')}").status_code == 404


# ── roster derived from the folder itself (2026-09-12 header-name leak) ──────

def test_identities_from_filename_reads_the_brightspace_prefix():
    d2l = "12345-67890 - Rivera, Jordan - Sep 3, 2026 1128 PM - Redline memo (2).docx"
    assert le.identities_from_filename(d2l) == [
        {"sub": "d2l:12345", "name": "Rivera, Jordan", "email": ""}]
    assert le.identities_from_filename("Untitled document.docx") == []


def test_merge_identities_dedupes_by_name_and_keeps_the_roster_entry():
    roster = [{"sub": "s1", "name": "Jordan Rivera", "email": "j@example.edu"}]
    derived = [{"sub": "d2l:1", "name": "Rivera, Jordan", "email": ""},
               {"sub": "d2l:2", "name": "Park, Kim", "email": ""}]
    merged = le.merge_identities(roster, derived)
    assert [m["sub"] for m in merged] == ["s1", "d2l:2"]


def test_folder_identities_unions_roster_index_and_file_names(tmp_path):
    (tmp_path / "index.html").write_text(D2L_INDEX)
    files = [tmp_path / "12345-1 - Rivera, Jordan - Sep 3, 2026 1128 PM - memo.docx",
             tmp_path / "12345-2 - Rivera, Jordan - Sep 3, 2026 1130 PM - cover.docx",
             tmp_path / "99-3 - Nguyen, Bao - Sep 3, 2026 1130 PM - memo.pdf",
             tmp_path / "loose.pdf"]
    idents, sources = le.folder_identities(
        tmp_path, files, [{"sub": "r", "name": "Extra Person", "email": ""}])
    names = {i["name"] for i in idents}
    assert {"Extra Person", "Alvarado, Luz", "Park, Kim", "Rivera, Jordan",
            "Nguyen, Bao"} <= names
    assert sources == {"roster_file": 1, "index_html": 3, "file_names": 2,
                       "total": len(idents)}


def test_run_folder_scrubs_header_names_from_file_prefixes_without_a_roster(
        tmp_path, monkeypatch):
    """The bug: a Brightspace download run with no roster sent the student's
    name (memo FROM: line + bare header) to the model. The file name knows
    who it is, so the pipeline must use it."""
    _stub_verifier(monkeypatch)
    monkeypatch.setattr(le, "list_models", lambda host: ["gemma4:26b"])
    seen = []

    def fake_chat_json(prompt, **kw):
        seen.append(prompt)
        return {"content_assessment": "ok", "rubric": [],
                "focus_for_instructor": [], "integrity_note": "n"}

    monkeypatch.setattr(le, "chat_json", fake_chat_json)
    sub = tmp_path / "subs"
    sub.mkdir()
    a = sub / "111-1 - Rivera, Jordan - Sep 3, 2026 1128 PM - memo.txt"
    b = sub / "222-2 - Park, Kim - Sep 3, 2026 1130 PM - memo.txt"
    a.write_text("TO: Professor Faith\nFROM: Jordan Rivera\nRE: Acme\n\n"
                 "As Kim Park argued in class, the covenant fails.")
    b.write_text("Kim Park\nBUSL 101\nProfessor Faith\n\nI disagree with Jordan.")
    out = tmp_path / "out"
    m = le.run_folder(sub, PACK, le.EvalOptions(check_citations=False), out)
    assert m["status"] == "done"
    assert m["roster"] == {"roster_file": 0, "index_html": 0,
                           "file_names": 2, "total": 2}
    assert m["options"]["identities"] == 2
    for p in seen:                       # nothing the model saw carried a name
        for name in ("Jordan", "Rivera", "Kim Park"):
            assert name not in p, name
    for stem in ("111-1 - Rivera, Jordan - Sep 3, 2026 1128 PM - memo",
                 "222-2 - Park, Kim - Sep 3, 2026 1130 PM - memo"):
        saved = (out / f"{stem}.txt").read_text()
        assert "Jordan" not in saved and "Kim Park" not in saved
        assert "[name redacted]" in saved or "Student " in saved
    # The roster block is counts only — the file names themselves carry
    # the student's name by Brightspace's design, as they always have.
    assert "Jordan" not in json.dumps(m["roster"] | m["options"])


# ── New Run pickers: active courses first, everything A to Z ────────────────


def test_local_packs_sort_by_course_then_title(tmp_path, monkeypatch):
    import firstread.local_eval as le
    for aid, title, course in ((5, "week 3 Memo", "Torts"), (9, "Brief", "torts"),
                               (2, "Zoning note", "Property"), (7, "Adverse possession", "Property")):
        (tmp_path / f"ia{aid}.json").write_text(json.dumps(
            {"assignment_id": aid, "title": title, "course_title": course}))
    monkeypatch.setattr(le, "pack_dirs", lambda: [tmp_path])
    assert [p["assignment_id"] for p in le.discover_packs()] == [7, 2, 9, 5]


def test_ui_new_run_groups_packs_by_course(ui, monkeypatch):
    pack = lambda aid, title, course: {"path": f"/p/ia{aid}.json", "assignment_id": aid,  # noqa: E731
                                       "title": title, "course_title": course,
                                       "has_key": False, "readings": 0, "has_moves": False,
                                       "has_rubric": True}
    monkeypatch.setattr(ui, "discover_packs", lambda: [
        pack(7, "Adverse possession", "Property"), pack(2, "Zoning note", "Property"),
        pack(9, "Brief", "Torts")])
    page = ui.app.test_client().get("/").data.decode()
    assert page.index('<optgroup label="Property">') < page.index("Adverse possession") \
        < page.index("Zoning note") < page.index('<optgroup label="Torts">') \
        < page.index("Brief")
