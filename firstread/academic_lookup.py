"""Verify academic sources against Semantic Scholar, OpenAlex and doi.org
(Story 1028; spec doc/LOCAL_EVAL_ACADEMIC_SOURCES_SPEC.md §2–4).

Input: sources from utils/academic_citations.extract_sources (already-
scrubbed text). Output: finding rows for utils/local_eval.check_citations.

What leaves the machine (spec §4.1) — and nothing else:

* api.semanticscholar.org: DOI / arXiv / PMID ids (one batch POST per
  paper) and a source's title (``query=``);
* api.openalex.org: a DOI, or a title with a publication-year range;
* doi.org: one DOI, to ask whether it is registered.

Author names are compared locally against the returned records and never
sent. A field holding a pii_gate pseudonym or ``[… redacted]`` marker is
never sent. Personal communications, course materials, web pages and news
are never looked up, and no cited URL is fetched. Every value sent is
recorded in the returned ``emissions``.

Verdicts (spec §2.4): found · found_mismatch · found_conflated · not_found ·
ambiguous · unverifiable · unknown. "Possibly fabricated" is only ever a
signal to check by hand — never a finding about who or what wrote a paper.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional

import requests

from firstread import semantic_scholar_http as ssh
from firstread.academic_citations import (INDEXABLE_KINDS, NEVER_LOOKUP_KINDS,
                                      REDACTION_RE, fold, norm_title, normalize_doi,
                                      source_key)

log = logging.getLogger(__name__)

SS_FIELDS = "title,authors,year,venue,journal,externalIds,publicationTypes,abstract,url"
OPENALEX_WORKS = "https://api.openalex.org/works"
DOI_HANDLE_API = "https://doi.org/api/handles/"
CACHE_FILE = "_scholarly_cache.json"
CACHE_TTL_FOUND = 30 * 86400
CACHE_TTL_NOT_FOUND = 7 * 86400
SS_SKIP_AFTER = 3          # consecutive unavailable calls → skip SS for the run
QUERY_TITLE_MAX = 300

# acceptance (spec §2.3)
ACCEPT_TITLE = 0.92
ACCEPT_TITLE_WITH_SUPPORT = 0.80
NEAR_MISS_LOW = 0.60
AMBIGUITY_MARGIN = 0.02
OTHER_WORK_BELOW = 0.50
COSMETIC_TITLE = 0.97

VERDICT_LABELS = {
    "found": "verified: the source exists and the reference matches the indexed record",
    "found_mismatch": "exists, but the reference gets details wrong",
    "found_conflated": "a real work, cited with the wrong authors or another work's DOI "
                       "— possibly fabricated or conflated; check by hand",
    "not_found": "not found in Semantic Scholar or OpenAlex — possibly inaccurate or "
                 "fabricated; check by hand",
    "ambiguous": "several possible matches — confirm which work is meant",
    "unverifiable": "not indexed by these services — check by hand",
    "unknown": "not checked (rate limit, outage or daily budget) — not evidence of anything",
}

SIGNAL_NOTES = {
    "doi_unregistered": "Its DOI is not registered at doi.org.",
    "doi_other_work": "Its DOI belongs to a different work",
    "authors_disjoint": "None of the cited authors are among the indexed authors.",
    "indexable_not_found": "A journal citation with volume and pages that neither index has.",
    "future_year": "The cited year is later than today.",
}

UNVERIFIABLE_REASONS = {
    "personal_communication": "Personal communication or interview — never looked up; "
                              "it can't be checked against an index.",
    "course_material": "Course material (lecture notes, an LMS page) — never looked up.",
    "web": "Web page — not looked up; open the link to check it.",
    "news": "News article — not looked up; open the link or search the outlet.",
    "legal": "A case, statute or other legal authority, so the academic indexes "
             "weren't asked about it. No reporter or code citation could be read "
             "from this entry, so it wasn't looked up as one either — check by hand.",
    "redacted": "Part of this reference was redacted as a possible student name — "
                "check by hand.",
    "unreadable": "Couldn't read this reference's title — check by hand.",
    "not_indexed": "Not found in Semantic Scholar or OpenAlex; books, reports and "
                   "similar sources often aren't indexed — check by hand.",
}


# ── comparison (spec §2.2) ───────────────────────────────────────────────────

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0


def _main(t: str) -> str:
    return re.split(r"\s*(?::| — | - )\s*", t, maxsplit=1)[0]


def title_similarity(cited: str, indexed: str) -> float:
    """max(full vs full, main title vs main title): a missing subtitle is
    tolerated when both main titles have at least 3 words."""
    full = _ratio(norm_title(cited), norm_title(indexed))
    mc, mi = norm_title(_main(cited)), norm_title(_main(indexed))
    if len(mc.split()) >= 3 and len(mi.split()) >= 3:
        return max(full, _ratio(mc, mi))
    return full


def indexed_surnames(authors: list[str]) -> list[str]:
    """Folded surname forms of indexed author names, in order: the last
    token, plus particle / hyphen variants ('van der Berg' → 'berg', 'van
    der berg'; 'Smith-Jones' → 'smith-jones', 'smith', 'jones')."""
    out = []
    for name in authors or []:
        toks = fold(name).replace(",", " ").split()
        if not toks:
            out.append("")
            continue
        out.append(toks[-1])
    return out


def _surname_forms(s: str) -> set[str]:
    s = fold(s)
    forms = {s, s.split()[-1] if s.split() else s}
    for f in list(forms):
        forms |= set(f.split("-"))
    return {f for f in forms if f}


def _redacted_name(s: str) -> bool:
    """A cited surname pii_gate replaced (folded: 'student a12')."""
    return bool(REDACTION_RE.search(s or "") or re.match(r"student [a-z]\d{2}\b", s or "", re.I))


def author_agreement(cited: list[str], indexed_authors: list[str]) -> Optional[bool]:
    """True: a cited surname is among the indexed authors. False: none is.
    None: nothing to compare, or a cited name was redacted (so a scrubbed
    surname can never produce a false "conflated")."""
    cited = [c for c in (cited or []) if c]
    if not cited or not indexed_authors:
        return None
    if any(_redacted_name(c) for c in cited):
        return None
    idx = set()
    for name in indexed_authors:
        toks = fold(name).replace(",", " ").split()
        if toks:
            idx |= _surname_forms(toks[-1])
            idx |= _surname_forms(" ".join(toks[1:])) if len(toks) > 2 else set()
    return any(_surname_forms(c) & idx for c in cited)


def _abbreviates(t: str, word: str) -> bool:
    """'psychol' ~ 'psychologist', 'pa' ~ 'pennsylvania': same first letter,
    and the abbreviation's letters appear in order."""
    if not word.startswith(t[:1]):
        return False
    it = iter(word)
    return all(ch in it for ch in t)


def venue_matches(cited: str, indexed: str) -> bool:
    """Each cited token abbreviates an indexed token, in order
    ('Am. Psychol.' ~ 'American Psychologist', 'U. Pa. L. Rev.' ~
    'University of Pennsylvania Law Review')."""
    ct = [t for t in re.split(r"[^a-z0-9]+", fold(cited)) if t and t not in ("of", "the", "and")]
    it = [t for t in re.split(r"[^a-z0-9]+", fold(indexed)) if t and t not in ("of", "the", "and")]
    if not ct or not it:
        return True
    j = 0
    for t in ct:
        while j < len(it) and not _abbreviates(t, it[j]):
            j += 1
        if j == len(it):
            return False
        j += 1
    return True


def _year_int(y) -> Optional[int]:
    try:
        return int(str(y)[:4])
    except (TypeError, ValueError):
        return None


# ── normalised candidate records ─────────────────────────────────────────────

def _clean_num(s) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s


def from_semantic_scholar(p: dict) -> dict:
    j = p.get("journal") or {}
    ext = p.get("externalIds") or {}
    pid = p.get("paperId") or ""
    vol = _clean_num(j.get("volume")).split(" ")[0] if j.get("volume") else ""
    return {
        "title": p.get("title") or "",
        "authors": [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")],
        "year": p.get("year"),
        "venue": p.get("venue") or j.get("name") or "",
        "doi": normalize_doi(ext.get("DOI") or ""),
        "paper_id": pid,
        "url": f"https://www.semanticscholar.org/paper/{pid}" if pid else (p.get("url") or ""),
        "abstract": (p.get("abstract") or "")[:3000],
        "publication_types": p.get("publicationTypes") or [],
        "volume": vol,
        "pages": _clean_num(j.get("pages")).replace(" ", ""),
        "source": "semantic_scholar",
    }


def from_openalex(w: dict) -> dict:
    abstract = ""
    inv = w.get("abstract_inverted_index")
    if isinstance(inv, dict):
        pos = sorted((p, word) for word, ps in inv.items() for p in ps)
        abstract = " ".join(word for _, word in pos)
    src = ((w.get("primary_location") or {}).get("source") or {})
    biblio = w.get("biblio") or {}
    pages = "-".join(x for x in (biblio.get("first_page"), biblio.get("last_page")) if x)
    doi = normalize_doi(w.get("doi") or "")
    typ = w.get("type") or ""
    return {
        "title": w.get("title") or w.get("display_name") or "",
        "authors": [a.get("author", {}).get("display_name", "")
                    for a in (w.get("authorships") or []) if a.get("author")],
        "year": w.get("publication_year"),
        "venue": src.get("display_name") or "",
        "doi": doi,
        "paper_id": w.get("id") or "",
        "url": f"https://doi.org/{doi}" if doi else (w.get("id") or ""),
        "abstract": abstract[:3000],
        "publication_types": {"article": ["JournalArticle"], "book": ["Book"],
                              "book-chapter": ["BookSection"],
                              "preprint": ["Preprint"]}.get(typ, [typ] if typ else []),
        "volume": _clean_num(biblio.get("volume")),
        "pages": pages,
        "source": "openalex",
    }


# ── cache (spec §3.5) ────────────────────────────────────────────────────────

class ScholarCache:
    """Public bibliographic records keyed by identifier or normalised
    title — never a file name, student label or proposition. In memory for
    a run; persisted to ``path`` (the results root's _scholarly_cache.json)
    when given. ``refresh`` ignores what is stored (but still writes)."""

    def __init__(self, path: Optional[Path] = None, refresh: bool = False):
        self.path = Path(path) if path else None
        self.refresh = refresh
        self.data: dict[str, dict] = {}
        self.dirty = False
        if self.path and self.path.is_file() and not refresh:
            try:
                raw = json.loads(self.path.read_text())
                if isinstance(raw, dict):
                    self.data = {k: v for k, v in raw.get("entries", {}).items()
                                 if isinstance(v, dict)}
            except Exception:  # noqa: BLE001 — a bad cache is an empty cache
                log.warning("scholarly cache unreadable; starting empty", exc_info=True)

    def get(self, key: str) -> Optional[dict]:
        e = self.data.get(key)
        if not e:
            return None
        ttl = CACHE_TTL_FOUND if e.get("record") is not None else CACHE_TTL_NOT_FOUND
        if time.time() - float(e.get("fetched_at", 0)) > ttl:
            return None
        return e

    def put(self, key: str, record: Optional[dict], extra: Optional[dict] = None):
        self.data[key] = {"record": record, "fetched_at": time.time(), **(extra or {})}
        self.dirty = True

    def save(self):
        if not (self.path and self.dirty):
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 1, "entries": self.data}))
            tmp.replace(self.path)
            self.dirty = False
        except Exception:  # noqa: BLE001 — caching is a nicety
            log.warning("could not write the scholarly cache", exc_info=True)


# ── the checker ──────────────────────────────────────────────────────────────

@dataclass
class SourceOptions:
    max_sources: int = 40
    max_support_reads: int = 10
    openalex: bool = True
    doi_registry: bool = True


@dataclass
class _Ctx:
    """Per-paper bookkeeping."""
    emissions: dict = field(default_factory=lambda: {
        "semantic_scholar": [], "openalex": [], "doi_org": []})
    cache_hits: int = 0
    support_reads: int = 0


class SourceChecker:
    """One per run: holds the cache and the Semantic Scholar circuit
    breaker, so a run that hits a rate-limiting wall stops knocking."""

    def __init__(self, options: Optional[SourceOptions] = None,
                 cache: Optional[ScholarCache] = None):
        self.opts = options or SourceOptions()
        self.cache = cache or ScholarCache()
        self.ss_fail_streak = 0
        self.ss_skipped = False
        self.ss_skipped_after = ""
        self.ss_skipped_sources = 0
        self.openalex_unavailable = ""
        self._oa_session: Optional[requests.Session] = None
        self._doi_session: Optional[requests.Session] = None
        self.current_file = ""

    # -- status for the manifest / UI
    def status(self) -> dict:
        return {"ss_key": bool(ssh.api_key()),
                "openalex_key": bool((os.getenv("OPENALEX_API_KEY") or "").strip()),
                "ss_skipped": self.ss_skipped,
                "ss_skipped_after": self.ss_skipped_after,
                "ss_skipped_sources": self.ss_skipped_sources}

    # -- Semantic Scholar with the circuit breaker
    def _ss_ok(self):
        self.ss_fail_streak = 0

    def _ss_failed(self, e: Exception):
        self.ss_fail_streak += 1
        log.warning("Semantic Scholar unavailable (%s in a row): %s", self.ss_fail_streak, e)
        if self.ss_fail_streak >= SS_SKIP_AFTER and not self.ss_skipped:
            self.ss_skipped = True
            self.ss_skipped_after = self.current_file
            log.warning("Semantic Scholar skipped for the rest of this run")

    def _ss_batch(self, ids: list[str], ctx: _Ctx) -> Optional[list]:
        if self.ss_skipped or not ids:
            return None
        ctx.emissions["semantic_scholar"] += ids
        try:
            out = ssh.post_batch(ids, SS_FIELDS)
        except ssh.SemanticScholarUnavailable as e:
            self._ss_failed(e)
            return None
        self._ss_ok()
        return out

    def _ss_title(self, title: str, ctx: _Ctx, *, search: bool = False) -> Optional[list]:
        """Candidates for a title: ``/paper/search/match`` (one best) or
        ``/paper/search`` (top 10). None = Semantic Scholar unavailable."""
        if self.ss_skipped:
            return None
        ctx.emissions["semantic_scholar"].append(f"title: {title}")
        try:
            if search:
                resp = ssh.get("/paper/search", {"query": title, "fields": SS_FIELDS, "limit": 10})
            else:
                resp = ssh.get("/paper/search/match", {"query": title, "fields": SS_FIELDS})
        except ssh.SemanticScholarUnavailable as e:
            self._ss_failed(e)
            return None
        if resp.status_code == 404:
            self._ss_ok()
            return []
        if not resp.ok:
            self._ss_failed(RuntimeError(f"HTTP {resp.status_code}"))
            return None
        self._ss_ok()
        try:
            return [from_semantic_scholar(p) for p in (resp.json().get("data") or [])
                    if isinstance(p, dict)]
        except ValueError:
            return None

    # -- OpenAlex
    def _oa(self) -> requests.Session:
        if self._oa_session is None:
            from requests.adapters import HTTPAdapter
            from firstread.openalex_http import retry_policy
            s = requests.Session()
            s.headers["User-Agent"] = ssh.USER_AGENT
            s.mount("https://", HTTPAdapter(max_retries=retry_policy()))
            self._oa_session = s
        return self._oa_session

    def _openalex(self, ctx: _Ctx, *, doi: str = "", title: str = "",
                  year: Optional[int] = None) -> Optional[list]:
        """Candidates from OpenAlex. None = not searched (off, or budget)."""
        if not self.opts.openalex or self.openalex_unavailable:
            return None
        from firstread.openalex_http import OpenAlexBudgetExhausted, openalex_get
        try:
            if doi:
                ctx.emissions["openalex"].append(f"doi: {doi}")
                resp = openalex_get(self._oa(), f"{OPENALEX_WORKS}/doi:{doi}", timeout=15)
                if resp.status_code == 404:
                    return []
                return [from_openalex(resp.json())] if resp.ok else None
            params = {"search": title, "per_page": 10}
            if year:
                params["filter"] = f"publication_year:{year - 1}-{year + 1}"
            ctx.emissions["openalex"].append(
                f"title: {title}" + (f" ({year - 1}–{year + 1})" if year else ""))
            resp = openalex_get(self._oa(), OPENALEX_WORKS, params=params, timeout=15)
            if not resp.ok:
                return None
            return [from_openalex(w) for w in (resp.json().get("results") or [])]
        except OpenAlexBudgetExhausted as e:
            self.openalex_unavailable = str(e)
            return None
        except (requests.RequestException, ValueError) as e:
            log.warning("OpenAlex lookup failed: %s", e)
            return None

    # -- doi.org
    def _doi_registered(self, doi: str, ctx: _Ctx) -> Optional[bool]:
        if not self.opts.doi_registry:
            return None
        key = f"doireg:{doi}"
        if (hit := self.cache.get(key)) is not None:
            ctx.cache_hits += 1
            return hit.get("registered")
        if self._doi_session is None:
            self._doi_session = requests.Session()
            self._doi_session.headers["User-Agent"] = ssh.USER_AGENT
        ctx.emissions["doi_org"].append(doi)
        try:
            resp = self._doi_session.get(DOI_HANDLE_API + doi, timeout=ssh.TIMEOUT)
        except requests.RequestException as e:
            log.warning("doi.org check failed: %s", e)
            return None
        code = None
        try:
            code = (resp.json() or {}).get("responseCode")
        except ValueError:
            pass
        if resp.status_code == 200 and code == 1:
            registered = True
        elif resp.status_code == 404 or code == 100:
            registered = False
        else:
            return None
        self.cache.put(key, {"doi": doi} if registered else None, {"registered": registered})
        return registered

    # -- one paper
    def check(self, sources: list[dict], *, judge=None,
              progress: Optional[Callable[[dict], None]] = None,
              file_label: str = "") -> tuple[list[dict], dict, dict]:
        """``(rows, emissions, meta)`` for one paper's sources."""
        self.current_file = file_label
        ctx = _Ctx()
        capped = len(sources) > self.opts.max_sources
        sources = sources[:self.opts.max_sources]
        id_records = self._identifier_records(sources, ctx)
        rows = []
        for i, src in enumerate(sources, 1):
            if progress:
                try:
                    progress({"stage": "sources", "index": i, "total": len(sources),
                              "message": f"{(src.get('title') or src.get('raw') or '')[:60]}: looking up"})
                except Exception:  # noqa: BLE001
                    pass
            try:
                row = self._check_one(src, id_records, ctx)
            except Exception as e:  # noqa: BLE001 — one source never stops a paper
                log.warning("source check failed", exc_info=True)
                row = _base_row(src)
                row.update(status="unknown", note=f"check failed: {str(e)[:200]}")
            abstract = row.pop("_abstract", "")
            if judge is not None and row["status"] in ("found", "found_mismatch"):
                self._support(row, abstract, judge, ctx)
            rows.append(row)
        self.cache.save()
        meta = {"sources_count": len(rows), "sources_capped": capped,
                "cache_hits": ctx.cache_hits, "scholarly": self.status()}
        return rows, ctx.emissions, meta

    def _identifier_records(self, sources, ctx) -> dict:
        """key → normalised record (or None = unknown id) from the cache and
        one batch call. Missing keys = not looked up (SS unavailable)."""
        out: dict[str, Optional[dict]] = {}
        want = []
        for s in sources:
            if s.get("source_kind") in NEVER_LOOKUP_KINDS:
                continue
            for sid, key in _ids(s):
                if key in out or sid in want:
                    continue
                if (hit := self.cache.get(key)) is not None:
                    ctx.cache_hits += 1
                    out[key] = hit.get("record")
                else:
                    want.append(sid)
        recs = self._ss_batch(want, ctx)
        if recs is not None:
            for sid, rec in zip(want, recs):
                key = _key_for_id(sid)
                norm = from_semantic_scholar(rec) if rec else None
                out[key] = norm
                if norm:
                    self.cache.put(key, norm)
                # an unknown id is not cached yet: doi.org / OpenAlex decide
        return out

    def _check_one(self, src: dict, id_records: dict, ctx: _Ctx) -> dict:
        row = _base_row(src)
        kind = src.get("source_kind") or "other"
        if kind in NEVER_LOOKUP_KINDS:
            return _unverifiable(row, kind)
        redacted = bool(REDACTION_RE.search(src.get("title") or "") or
                        REDACTION_RE.search(src.get("venue") or ""))
        if redacted:
            # never send it, and never compare it: the scrubber changed it
            src = dict(src, title="", venue="")
        title = _query_title(src.get("title") or "")
        ids = _ids(src)
        if not ids and not title:
            return _unverifiable(row, "redacted" if redacted else "unreadable")
        if (not ids and not src.get("venue")
                and any(_redacted_name(a) for a in src.get("author_surnames") or [])):
            # an author the scrubber took for a classmate, and nothing that
            # says it was published: likely a student's own work — not sent
            return _unverifiable(row, "redacted")
        year = _year_int(src.get("year"))
        signals: list[str] = []
        unavailable: list[str] = []
        other_work: Optional[dict] = None
        doi_registered = False

        # 1. identifiers
        for sid, key in ids:
            rec = id_records.get(key)
            if key not in id_records:
                unavailable.append("Semantic Scholar")
            via = "doi" if sid.startswith("DOI:") else sid.split(":")[0].lower()
            if rec is None and sid.startswith("DOI:"):
                doi = sid[4:]
                registered = self._doi_registered(doi, ctx)
                if registered is False:
                    signals.append("doi_unregistered")
                    continue
                doi_registered = doi_registered or bool(registered)
                oa = self._openalex(ctx, doi=doi)
                if oa is None:
                    if self.opts.openalex:
                        unavailable.append("OpenAlex")
                elif oa:
                    rec, via = oa[0], "openalex_doi"
                    self.cache.put(key, rec)
            if rec is None:
                continue
            if not src.get("title"):
                return self._decide(row, src, rec, 1.0, via, signals)
            sim = title_similarity(src["title"], rec["title"])
            if sim >= ACCEPT_TITLE_WITH_SUPPORT:
                return self._decide(row, src, rec, sim, via, signals)
            if sim < OTHER_WORK_BELOW:
                other_work = rec
                signals.append("doi_other_work")
                continue
            # 0.50–0.80 on the cited identifier: the work, with its title wrong
            return self._decide(row, src, rec, sim, via, signals, title_major=True)

        # 2. title
        if title:
            cands, via = self._title_candidates(title, year, ctx, unavailable)
            best, ambiguous = _choose(src, cands)
            if ambiguous:
                row.update(status="ambiguous", fabrication_signals=signals,
                           note="Several works match: " + "; ".join(
                               f"{c['title']} ({c.get('year') or 'n.d.'})" for c in ambiguous[:3]))
                return row
            if best:
                rec, sim = best
                return self._decide(row, src, rec, sim, via.get(id(rec), "title_match"), signals,
                                    other_work=other_work)

        # 3. nothing accepted
        if doi_registered and not signals:
            _unverifiable(row, "not_indexed")
            row["note"] = ("Its DOI is registered at doi.org, but neither Semantic "
                           "Scholar nor OpenAlex has the work — check by hand.")
            return row
        return self._not_found(row, src, signals, unavailable, other_work)

    def _title_candidates(self, title, year, ctx, unavailable) -> tuple[list, dict]:
        key = f"title:{norm_title(title)}|{year or ''}"
        if (hit := self.cache.get(key)) is not None:
            ctx.cache_hits += 1
            recs = hit.get("records") or ([hit["record"]] if hit.get("record") else [])
            return recs, {id(r): "cache" for r in recs}
        via: dict[int, str] = {}
        cands: list[dict] = []
        ss = self._ss_title(title, ctx)
        if ss is None:
            unavailable.append("Semantic Scholar")
        else:
            for r in ss:
                via[id(r)] = "title_match"
            cands += ss
            best = max((title_similarity(title, c["title"]) for c in ss), default=0.0)
            if NEAR_MISS_LOW <= best < ACCEPT_TITLE_WITH_SUPPORT:
                more = self._ss_title(title, ctx, search=True)
                if more:
                    for r in more:
                        via[id(r)] = "title_search"
                    cands += more
        accepted = [c for c in cands if title_similarity(title, c["title"]) >= ACCEPT_TITLE_WITH_SUPPORT]
        if not accepted:
            oa = self._openalex(ctx, title=title, year=year)
            if oa is None:
                if self.opts.openalex:
                    unavailable.append("OpenAlex")
            else:
                for r in oa:
                    via[id(r)] = "openalex_title"
                cands += oa
        good = [c for c in cands if title_similarity(title, c["title"]) >= ACCEPT_TITLE_WITH_SUPPORT]
        searched_all = "Semantic Scholar" not in unavailable and (
            "OpenAlex" not in unavailable or bool(good))
        if good:
            self.cache.put(key, good[0], {"records": good[:5]})
        elif searched_all:
            self.cache.put(key, None, {"records": []})
        return cands, via

    def _decide(self, row, src, rec, sim, via, signals, *, title_major=False,
                other_work=None) -> dict:
        agree = author_agreement(src.get("author_surnames") or [], rec["authors"])
        _fill_indexed(row, rec, via, sim, agree, src)
        if other_work is not None:
            signals = [s for s in signals if s != "doi_other_work"] + ["doi_other_work"]
            row.update(status="found_conflated", fabrication_signals=signals,
                       note=(f"{SIGNAL_NOTES['doi_other_work']}: “{other_work['title']}”. "
                             f"The title matches “{rec['title']}”."))
            return row
        if agree is False:
            row.update(status="found_conflated",
                       fabrication_signals=signals + ["authors_disjoint"],
                       note=(f"{SIGNAL_NOTES['authors_disjoint']} Indexed authors: "
                             f"{', '.join(rec['authors'][:4])}"
                             + (" et al." if len(rec["authors"]) > 4 else "") + "."))
            return row
        diffs = differences(src, rec, sim, title_major=title_major)
        row["differences"] = diffs
        row["fabrication_signals"] = signals
        if diffs:
            row["status"] = "found_mismatch"
            row["note"] = " ".join(_diff_sentence(d) for d in diffs)
        else:
            row["status"] = "found"
        return row

    def _not_found(self, row, src, signals, unavailable, other_work) -> dict:
        kind = src.get("source_kind") or "other"
        year = _year_int(src.get("year"))
        if year and year > datetime.now(timezone.utc).year:
            signals.append("future_year")
        if kind in ("journal_article", "law_review") and src.get("volume") and src.get("pages"):
            signals.append("indexable_not_found")
        if other_work is not None and "doi_other_work" in signals:
            signals_note = (f"{SIGNAL_NOTES['doi_other_work']}: “{other_work['title']}”. ")
        else:
            signals_note = ""
        strong = "doi_unregistered" in signals or "doi_other_work" in signals
        if unavailable and not strong:
            services = " and ".join(sorted(set(unavailable)))
            if "Semantic Scholar" in unavailable and self.ss_skipped:
                self.ss_skipped_sources += 1
            why = self.openalex_unavailable if "OpenAlex" in unavailable and self.openalex_unavailable else ""
            row.update(status="unknown", fabrication_signals=signals,
                       note=(f"{services} could not be searched"
                             + (f" ({why})" if why else "") + " — not checked, not missing."))
            return row
        if kind in INDEXABLE_KINDS or strong:
            row.update(status="not_found", fabrication_signals=signals,
                       note=(signals_note + " ".join(SIGNAL_NOTES[s] for s in signals
                                                     if s in SIGNAL_NOTES and s != "doi_other_work")).strip())
            return row
        _unverifiable(row, "not_indexed")
        row["fabrication_signals"] = signals
        return row

    def _support(self, row, abstract, judge, ctx):
        """The abstract read (spec §2.5): its own field, never a problem."""
        if not (abstract and row.get("proposition")):
            return
        if ctx.support_reads >= self.opts.max_support_reads:
            return
        ctx.support_reads += 1
        from firstread.citation_verifier import check_relevance
        try:
            rel = check_relevance(row["proposition"], row["case_name"], abstract,
                                  citation_type="statute", llm=judge) or {}
        except Exception:  # noqa: BLE001
            rel = {"status": "unknown"}
        st = rel.get("status") or "unknown"
        note = " ".join(x for x in (rel.get("summary"), rel.get("confidence_note")) if x).strip()
        if st in ("found_unrelated", "found_miscited"):
            note = ("The abstract doesn't obviously support this point — read the source. "
                    + note).strip()
        row["support"] = {"status": st, "basis": "abstract", "note": note}


# ── helpers ──────────────────────────────────────────────────────────────────

def _ids(src: dict) -> list[tuple[str, str]]:
    """(Semantic Scholar id, cache key) pairs for a source's identifiers."""
    out = []
    if src.get("doi"):
        out.append((f"DOI:{src['doi']}", f"doi:{src['doi']}"))
    if src.get("arxiv"):
        out.append((f"ARXIV:{src['arxiv']}", f"arxiv:{src['arxiv']}"))
    if src.get("pmid"):
        out.append((f"PMID:{src['pmid']}", f"pmid:{src['pmid']}"))
    return out


def _key_for_id(sid: str) -> str:
    kind, _, val = sid.partition(":")
    return f"{kind.lower()}:{val}"


def _query_title(title: str) -> str:
    t = re.sub(r"\s+", " ", title or "").strip().strip("\"'")
    return t[:QUERY_TITLE_MAX]


def _choose(src: dict, cands: list[dict]):
    """(best (record, similarity) or None, ambiguous list or None) under the
    acceptance rule (spec §2.3)."""
    cited = src.get("title") or ""
    year = _year_int(src.get("year"))
    passing = []
    for c in cands:
        sim = title_similarity(cited, c["title"])
        agree = author_agreement(src.get("author_surnames") or [], c["authors"])
        cy = _year_int(c.get("year"))
        year_ok = bool(year and cy and abs(year - cy) <= 1)
        if sim >= ACCEPT_TITLE or (sim >= ACCEPT_TITLE_WITH_SUPPORT and (agree or year_ok)):
            passing.append((c, sim, bool(agree) and year_ok))
    if not passing:
        return None, None
    # one work in several versions (conference + journal): same title, shared authors
    works: list[list] = []
    for p in passing:
        for w in works:
            c0 = w[0][0]
            same = ((p[0]["doi"] and p[0]["doi"] == c0["doi"]) or
                    (p[0]["paper_id"] and p[0]["paper_id"] == c0["paper_id"]) or
                    (norm_title(p[0]["title"]) == norm_title(c0["title"]) and
                     author_agreement(indexed_surnames(p[0]["authors"]), c0["authors"]) is not False))
            if same:
                w.append(p)
                break
        else:
            works.append([p])

    def best_of(w):   # the version whose year / venue matches the cite
        return max(w, key=lambda p: (p[2], p[1],
                                     venue_matches(src.get("venue") or "", p[0].get("venue") or "")))

    tops = sorted((best_of(w) for w in works), key=lambda p: (p[2], p[1]), reverse=True)
    if len(tops) > 1 and tops[0][1] - tops[1][1] <= AMBIGUITY_MARGIN and \
            not (tops[0][2] and not tops[1][2]):
        return None, [t[0] for t in tops]
    return (tops[0][0], tops[0][1]), None


def differences(src: dict, rec: dict, sim: float, *, title_major: bool = False) -> list[dict]:
    """What the reference gets wrong against the indexed record (spec §2.4.1)."""
    out = []
    cy, iy = _year_int(src.get("year")), _year_int(rec.get("year"))
    if cy and iy and cy != iy:
        out.append({"field": "year", "cited": str(cy), "indexed": str(iy),
                    "severity": "minor" if abs(cy - iy) == 1 else "major"})
    cited_names = src.get("author_surnames") or []
    idx = indexed_surnames(rec.get("authors") or [])
    if cited_names and idx and not _redacted_name(cited_names[0]):
        if not (_surname_forms(cited_names[0]) & _surname_forms(idx[0])):
            out.append({"field": "authors", "cited": cited_names[0].title(),
                        "indexed": (rec["authors"][0] if rec.get("authors") else ""),
                        "severity": "minor"})
    if src.get("title") and sim < COSMETIC_TITLE:
        out.append({"field": "title", "cited": src["title"], "indexed": rec.get("title", ""),
                    "severity": "major" if title_major else "minor"})
    if src.get("venue") and rec.get("venue") and not venue_matches(src["venue"], rec["venue"]) \
            and not REDACTION_RE.search(src["venue"]):
        out.append({"field": "venue", "cited": src["venue"], "indexed": rec["venue"],
                    "severity": "minor"})
    for f in ("volume", "pages"):
        c, i = str(src.get(f) or ""), str(rec.get(f) or "")
        cn, inn = re.sub(r"[^\d]", "-", c).strip("-"), re.sub(r"[^\d]", "-", i).strip("-")
        if f == "pages":
            cn, inn = cn.split("-")[0], inn.split("-")[0]
        if cn and inn and cn.isdigit() and inn.isdigit() and cn != inn:
            out.append({"field": f, "cited": c, "indexed": i, "severity": "minor"})
    types = set(rec.get("publication_types") or [])
    kind = src.get("source_kind")
    if kind in ("book", "book_chapter") and "JournalArticle" in types and not types & {"Book", "BookSection"}:
        out.append({"field": "kind", "cited": kind.replace("_", " "), "indexed": "journal article",
                    "severity": "minor"})
    elif kind in ("journal_article", "law_review") and types and types <= {"Book", "BookSection"}:
        out.append({"field": "kind", "cited": "journal article", "indexed": "book",
                    "severity": "minor"})
    out.sort(key=lambda d: d["severity"] != "major")
    return out


def _diff_sentence(d: dict) -> str:
    label = {"year": "year", "authors": "first author", "title": "title", "venue": "venue",
             "volume": "volume", "pages": "first page", "kind": "kind of source"}[d["field"]]
    return f"Cited {label} {d['cited']!s}; indexed {d['indexed']!s}."


def _base_row(src: dict) -> dict:
    cited = {k: src.get(k) for k in ("title", "author_surnames", "year", "venue", "volume",
                                     "issue", "pages", "doi", "arxiv", "pmid", "url")}
    citation = (src.get("raw") or "").strip()
    if src.get("origin") == "identifier" or not citation:
        citation = (f"doi:{src['doi']}" if src.get("doi") else
                    f"arXiv:{src['arxiv']}" if src.get("arxiv") else
                    f"PMID:{src['pmid']}" if src.get("pmid") else citation)
    return {"citation": citation, "type": "academic",
            "source_kind": src.get("source_kind") or "other", "style": src.get("style") or "unknown",
            "origin": src.get("origin"), "in_text": list(src.get("in_text") or [])[:5],
            "proposition": (src.get("proposition") or "")[:300],
            "status": "unknown", "case_name": "", "url": "", "court": "", "date_filed": "",
            "note": "", "cited": cited, "indexed": None, "match": None,
            "differences": [], "fabrication_signals": [], "support": None, "history": None,
            "key": source_key(src)}


def _unverifiable(row: dict, reason: str) -> dict:
    row.update(status="unverifiable", unverifiable_reason=reason,
               note=UNVERIFIABLE_REASONS[reason])
    if reason == "web" and (row.get("cited") or {}).get("url"):
        row["url"] = row["cited"]["url"]
    return row


def _fill_indexed(row, rec, via, sim, agree, src):
    cy, iy = _year_int(src.get("year")), _year_int(rec.get("year"))
    row["indexed"] = {k: rec.get(k) for k in ("title", "authors", "year", "venue", "doi",
                                              "paper_id", "source")}
    row["indexed"]["authors"] = (rec.get("authors") or [])[:8]
    row["match"] = {"via": via, "title_similarity": round(sim, 3), "author_agrees": agree,
                    "year_delta": (iy - cy) if (cy and iy) else None,
                    "cached": via == "cache"}
    row["case_name"] = rec.get("title") or ""
    row["url"] = rec.get("url") or (f"https://doi.org/{rec['doi']}" if rec.get("doi") else "")
    row["court"] = rec.get("venue") or ""
    row["date_filed"] = str(rec.get("year") or "")
    row["_abstract"] = rec.get("abstract") or ""   # for the support read only
