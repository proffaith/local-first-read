"""
Citation verification — extract legal citations from text and verify them
against CourtListener (cases) and Cornell LII (statutes) to catch fabricated
or mis-described authorities.

Snapshot of ProfFaith's ``utils/citation_verifier.py``, cut down to what the
local first read calls. In ProfFaith the two judgment calls here (does the
case say what the student says; how did a higher court treat it) can run on
Anthropic. In this snapshot they run only through an injected ``llm``
callable, which ``firstread.local_eval.LocalJudge`` points at a localhost
Ollama model, so no code path here sends student text to a cloud model.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from firstread.rate_limited_http import backoff_only_retry, rate_limited_get

log = logging.getLogger(__name__)

# LLM seam for the two judgment calls in this module (relevance +
# disposition). ``llm(prompt, kind) -> str`` must return text in the
# ``KEY: value`` line format the parsers below read; ``kind`` is
# ``"relevance"`` or ``"disposition"``. firstread.local_eval.LocalJudge runs
# the prompts on a localhost model so student text never leaves the machine.
LLMFn = Callable[[str, str], str]


def _clean_format_line(line: str) -> str:
    """Normalize one response line so the ``KEY: value`` parsers below also
    accept markdown-decorated output (``**KEY:** value``, ``- KEY: value``)
    that smaller local models tend to emit."""
    return line.strip().strip("*-#> ").replace("**", "").strip()

# ---------------------------------------------------------------------------
# CourtListener config
# ---------------------------------------------------------------------------
CL_API_V4 = "https://www.courtlistener.com/api/rest/v4/search/"
CL_CLUSTERS_V4 = "https://www.courtlistener.com/api/rest/v4/clusters/"
CL_OPINIONS_V4 = "https://www.courtlistener.com/api/rest/v4/opinions/"


def _cl_headers() -> dict:
    token = os.getenv("COURT_LISTENER_API_KEY")
    if token:
        return {"Authorization": f"Token {token}"}
    return {}


_cl_session = None


def _get_cl_session() -> requests.Session:
    """Shared CourtListener session with automatic retry/back-off.

    CourtListener occasionally drops slow connections (read timeout) or
    returns transient 5xx under load.  A single blip should not silently
    drop a citation from a discovery run, so we retry idempotent GETs a couple
    of times before giving up.  Read timeouts on GET are retried automatically
    because GET is in `allowed_methods` and the read budget tracks `total`.

    429 is not retried here and Retry-After is never slept on: DRF throttling
    asks for up to an hour.  Always GET through `_cl_get` (see
    utils/rate_limited_http).
    """
    global _cl_session
    if _cl_session is None:
        from requests.adapters import HTTPAdapter
        _cl_session = requests.Session()
        retry = backoff_only_retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,  # let existing resp.ok checks handle final 4xx/5xx
        )
        _cl_session.mount("https://", HTTPAdapter(max_retries=retry))
    return _cl_session


def _cl_get(url: str, **kwargs) -> requests.Response:
    """GET via the CourtListener session; a long 429 comes back at once for
    the caller's ``resp.ok`` check instead of sleeping out Retry-After."""
    return rate_limited_get(_get_cl_session(), url, service="CourtListener",
                            **kwargs)


# ---------------------------------------------------------------------------
# Cornell LII config (statutory lookups)
# ---------------------------------------------------------------------------
CORNELL_USC_URL = "https://www.law.cornell.edu/uscode/text/{title}/{section}"
CORNELL_CFR_URL = "https://www.law.cornell.edu/cfr/text/{title}/{section}"
CORNELL_RATE_LIMIT = 1.5  # seconds between requests


# ---------------------------------------------------------------------------
# Text normalisation — clean PDF / Word artefacts before eyecite parsing
# ---------------------------------------------------------------------------

import unicodedata as _unicodedata

# Characters that look like a regular space but silently break eyecite
_SPACE_CHARS = {
    "\u00a0",   # non-breaking space  (very common in PDFs)
    "\u202f",   # narrow no-break space
    "\u2007",   # figure space
    "\u2009",   # thin space
    "\u200a",   # hair space
    "\u2002",   # en space
    "\u2003",   # em space
    "\u2004",   # three-per-em space
    "\u2005",   # four-per-em space
    "\u2006",   # six-per-em space
    "\u2008",   # punctuation space
    "\u205f",   # medium mathematical space
    "\u3000",   # ideographic space
}

# Zero-width characters that invisibly break tokenisation
_ZEROWIDTH_CHARS = {
    "\u200b",   # zero-width space
    "\u200c",   # zero-width non-joiner
    "\u200d",   # zero-width joiner
    "\ufeff",   # BOM / zero-width no-break space
    "\u2060",   # word joiner
}

# Dash / hyphen variants → standard ASCII hyphen-minus
_DASH_MAP = {
    "\u2010": "-",   # hyphen
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash (keep for ranges but normalise)
    "\u2014": "-",   # em dash
    "\u2015": "-",   # horizontal bar
    "\ufe58": "-",   # small em dash
    "\ufe63": "-",   # small hyphen-minus
    "\uff0d": "-",   # fullwidth hyphen-minus
}

# Quote / apostrophe variants → ASCII equivalents
_QUOTE_MAP = {
    "\u2018": "'",   # left single quotation mark
    "\u2019": "'",   # right single quotation mark  (Word "smart quote")
    "\u201a": "'",   # single low-9 quotation mark
    "\u201b": "'",   # single high-reversed-9 quotation mark
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u201e": '"',   # double low-9 quotation mark
    "\u201f": '"',   # double high-reversed-9 quotation mark
    "\u2032": "'",   # prime (sometimes used for feet / apostrophe)
    "\u2033": '"',   # double prime
}

# Build a single translation table for fast str.translate()
_NORM_TABLE = str.maketrans(
    {**{ch: " " for ch in _SPACE_CHARS},
     **{ch: ""  for ch in _ZEROWIDTH_CHARS},
     **_DASH_MAP,
     **_QUOTE_MAP}
)


def _normalize_legal_text(text: str) -> str:
    """Clean common PDF / Word artefacts so eyecite can parse reliably.

    Handles: non-breaking & zero-width spaces, smart quotes / apostrophes,
    dash variants, Unicode normalisation (NFC).  Returns a cleaned copy;
    the original is not modified.
    """
    text = _unicodedata.normalize("NFC", text)
    text = text.translate(_NORM_TABLE)
    # Collapse runs of multiple spaces that might result from stripping
    text = re.sub(r"  +", " ", text)
    return text


# ---------------------------------------------------------------------------
# Citation extraction via eyecite
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> List[Dict[str, Any]]:
    """Parse *text* and return a list of citation dicts.

    Each dict contains:
        citation_type   – "case" or "statute"
        citation_text   – raw string (e.g. "347 U.S. 483" or "42 U.S.C. § 1983")
        proposition     – the sentence(s) surrounding the citation
        start, end      – character offsets in *text*

    Case citations also include:
        volume, reporter, page, plaintiff, defendant, year, court

    Statutory citations also include:
        title, reporter, section, subject
    """
    from eyecite import get_citations
    from eyecite.models import FullCaseCitation, FullLawCitation

    # Normalise PDF / Word artefacts before eyecite sees the text
    text = _normalize_legal_text(text)

    raw_cites = get_citations(text)
    results: list[dict] = []
    seen: set[str] = set()

    for cite in raw_cites:
        token = cite.token
        cite_str = token.data
        # Deduplicate identical citation strings
        if cite_str in seen:
            continue

        groups = token.groups or {}
        meta = cite.metadata

        if isinstance(cite, FullCaseCitation):
            seen.add(cite_str)
            proposition = _extract_proposition(text, token.start, token.end)
            results.append({
                "citation_type": "case",
                "citation_text": cite_str,
                "volume": groups.get("volume", ""),
                "reporter": groups.get("reporter", ""),
                "page": groups.get("page", ""),
                "plaintiff": getattr(meta, "plaintiff", "") or "",
                "defendant": getattr(meta, "defendant", "") or "",
                "year": getattr(meta, "year", "") or "",
                "court": getattr(meta, "court", "") or "",
                "proposition": proposition,
                "start": token.start,
                "end": token.end,
            })

        elif isinstance(cite, FullLawCitation):
            seen.add(cite_str)
            proposition = _extract_proposition(text, token.start, token.end)
            results.append({
                "citation_type": "statute",
                "citation_text": cite_str,
                "title": groups.get("title", "") or groups.get("chapter", ""),
                "reporter": groups.get("reporter", ""),
                "section": groups.get("section", ""),
                "subject": groups.get("subject", ""),
                "proposition": proposition,
                "start": token.start,
                "end": token.end,
            })

    return results


# ---------------------------------------------------------------------------
# Academic / law review citation extraction (regex-based, not eyecite)
# ---------------------------------------------------------------------------

# Bluebook / ALWD format:  Author, Title, Vol. Journal Page (Year)
# e.g. "Sunstein, Beyond the Precautionary Principle, 151 U. Pa. L. Rev. 1003 (2003)"
# or   "Tribe, American Constitutional Law, 23 Harv. C.R.-C.L. L. Rev. 1 (1988)"
_ACADEMIC_CITE_RE = re.compile(
    r'(?P<author>[A-Z][a-zA-Z\'\-]+(?:\s+(?:&|and)\s+[A-Z][a-zA-Z\'\-]+)?'  # Author(s)
    r'(?:,\s*[A-Z][a-zA-Z\.\s]*?)?)'                        # optional first-name / initials
    r',\s+'
    r'(?P<title>[A-Z][^,]{5,120})'                           # Title (starts uppercase, 5-120 chars)
    r',\s+'
    r'(?P<volume>\d{1,4})\s+'                                 # Volume number
    r'(?P<journal>[A-Z][A-Za-z\.\s&\-]+?)\s+'                # Journal abbreviation
    r'(?P<page>\d{1,5})'                                      # Start page
    r'(?:\s*[-–]\s*\d{1,5})?'                                # optional end page
    r'\s*\((?P<year>\d{4})\)',                                # Year in parens
)


# Pattern matching "Id." or "Id. at NNN" — the standard legal shorthand
# meaning "same case as the immediately preceding citation".
_ID_CITE_RE = re.compile(
    r'\bId\.\s*(?:at\s+\d+(?:\s*[-–]\s*\d+)?\.?)?',
)


def _extend_through_id_sentences(text: str, sent_end: int,
                                  max_extend: int = 3000) -> int:
    """Scan forward from *sent_end* and include subsequent sentences that
    contain ``Id.`` citations (meaning they still describe the same case).

    Stops when:
      - a sentence does NOT contain an Id. reference
      - a paragraph boundary is reached
      - a new full citation (volume + reporter) appears
      - *max_extend* additional characters have been consumed
    """
    from eyecite import get_citations
    from eyecite.models import FullCaseCitation

    limit = min(sent_end + max_extend, len(text))
    pos = sent_end

    while pos < limit:
        # Skip whitespace
        while pos < limit and text[pos] in ' \t\r':
            pos += 1

        # Paragraph break → stop
        if pos < limit and text[pos] == '\n':
            # Check for double newline or newline+indent (paragraph boundary)
            if pos + 1 < limit and text[pos + 1] == '\n':
                break
            # Single newline + 4+ spaces indent → paragraph boundary
            leading = 0
            k = pos + 1
            while k < limit and text[k] in ' \t':
                leading += 1
                k += 1
            if leading >= 4:
                break
            pos += 1
            continue

        # Find the end of the next sentence
        next_sent_end = pos
        for i in range(pos, limit):
            if text[i] == '\n':
                next_sent_end = i
                break
            if text[i] == '.' and i + 1 < len(text):
                # Check if this period ends a sentence (not an abbreviation)
                # A period followed by a space and uppercase = sentence end
                # But "Id." should not be treated as sentence end here
                if i + 2 < len(text) and text[i + 1] in ' \t' and text[i + 2].isupper():
                    # Check if this is "Id." — if so, it's not a sentence end
                    j = i - 1
                    while j >= pos and text[j].isalpha():
                        j -= 1
                    word = text[j + 1:i].lower()
                    if word in ('id',):
                        continue
                    # Check for other common abbreviations
                    if word in ('v', 'vs', 'no', 'nos', 'dr', 'mr', 'ms',
                                'mrs', 'jr', 'sr', 'st', 'corp', 'ltd',
                                'inc', 'co', 'cf', 'eg', 'ie'):
                        continue
                    next_sent_end = i + 1
                    break
        else:
            next_sent_end = limit

        if next_sent_end <= pos:
            break

        sentence = text[pos:next_sent_end]

        # Does this sentence contain an Id. citation?
        if not _ID_CITE_RE.search(sentence):
            break

        # Does it also contain a NEW full citation (not just Id.)?
        # If so, the author has moved on to a different case — stop.
        try:
            new_cites = list(get_citations(sentence))
            has_full = any(isinstance(c, FullCaseCitation) for c in new_cites)
            if has_full:
                break
        except Exception:
            pass

        # Include this Id. sentence
        pos = next_sent_end

    return pos


def _extract_proposition(text: str, start: int, end: int) -> str:
    """Return the citation sentence *plus* up to two preceding sentences,
    strictly within the same paragraph.

    Legal memos cite cases at the **end** of a claim, so the proposition
    the case supports is in the sentence(s) *before* the citation — e.g.:

        "Non-compete clauses must be reasonable in scope and duration.
         Food Fair v. Greeley, 264 Md. 105 (1972)."

    A naive single-sentence grab would only capture the citation itself.
    We walk backwards through sentence boundaries (period / ! / ? followed
    by a space and an uppercase letter) which avoids false splits on legal
    abbreviations like "Md.", "Inc.", "App.", and "v."

    Paragraph boundaries (double newline) are treated as a hard stop.
    A new paragraph typically introduces a new case or topic, so crossing
    that boundary would pull in context about an unrelated citation.
    """
    # --- walk forwards to find end of the citation's sentence ---
    sent_end = end
    for i in range(end, min(end + 300, len(text))):
        if text[i] in ".\n":
            sent_end = i + 1
            break
    else:
        sent_end = min(end + 300, len(text))

    # --- extend forward through "Id." continuation sentences ---
    # In legal writing, the full citation appears at first mention and then
    # "Id." (or "Id. at NNN") is used for subsequent references to the same
    # case.  These Id. sentences are still describing the *same* case and
    # should be included in the proposition so the LLM sees the full
    # characterization the author is making.
    sent_end = _extend_through_id_sentences(text, sent_end)

    # --- walk backwards to capture ~2 preceding sentences ---
    # Paragraph breaks are always a hard stop.
    boundaries = 0
    context_start = max(start - 800, 0)       # safe fallback
    for i in range(start - 1, max(start - 800, -1), -1):
        if i < 0:
            context_start = 0
            break
        # Paragraph break — always a hard stop.  A new paragraph in a
        # legal memo typically introduces a different case or argument,
        # so we must not pull that context into this citation's
        # proposition.
        if text[i] == '\n':
            # Double newline: always a paragraph break (DOCX extraction
            # joins paragraphs with \n\n)
            if i > 0 and text[i - 1] == '\n':
                context_start = i + 1
                break
            # Single newline followed by significant indentation (4+ spaces):
            # common in pasted/formatted text where paragraphs are indented.
            # This catches the "newline + indent" paragraph style without
            # false-triggering on simple line wraps (which have 0-1 spaces).
            leading_ws = 0
            k = i + 1
            while k < len(text) and text[k] in ' \t':
                leading_ws += 1
                k += 1
            if leading_ws >= 4:
                context_start = i + 1
                break
        # Sentence-ending punctuation followed by space + uppercase letter.
        # This skips abbreviations like "Md. App." (lowercase / non-alpha
        # follows the period) and only splits on real sentence endings.
        # Additionally skip "v." (versus) and other common legal/title
        # abbreviations that precede uppercase names.
        if (text[i] in '.!?'
                and i + 2 < len(text)
                and text[i + 1] in ' \t'
                and text[i + 2].isupper()):
            # Check for common abbreviations before this period
            # (walk back to extract the preceding "word")
            abbrev_end = i
            j = i - 1
            while j >= 0 and text[j].isalpha():
                j -= 1
            word_before = text[j + 1:abbrev_end].lower()
            if word_before in (
                'v', 'vs',                        # versus
                'no', 'nos',                      # number
                'dr', 'mr', 'ms', 'mrs', 'jr', 'sr',  # titles
                'st', 'corp', 'ltd', 'inc', 'co',      # entity types
                'id', 'cf', 'eg', 'ie',           # citation signals
            ):
                continue  # skip — this period is an abbreviation, not a sentence end
            boundaries += 1
            if boundaries >= 3:          # 2 preceding sentences + citation sentence
                context_start = i + 2    # start after ". "
                break

    return text[context_start:sent_end].strip()


# ---------------------------------------------------------------------------
# CourtListener lookup
# ---------------------------------------------------------------------------

def _reporter_variants(reporter: str) -> List[str]:
    """Generate alternative reporter abbreviation forms for CourtListener retry.

    CourtListener's citation index is sensitive to spacing and period placement.
    For example, "Md. App." may be indexed as "Md.App." or vice-versa.
    """
    variants: list[str] = []
    if not reporter:
        return variants

    # Collapse inner spaces around periods:  "Md. App." → "Md.App."
    collapsed = re.sub(r'\.\s+', '.', reporter)
    if collapsed != reporter:
        variants.append(collapsed)

    # Expand collapsed periods back: "Md.App." → "Md. App."
    expanded = re.sub(r'\.(?=[A-Za-z])', '. ', reporter)
    if expanded != reporter and expanded not in variants:
        variants.append(expanded)

    # Strip trailing period: "Md. App." → "Md. App"
    if reporter.endswith('.'):
        stripped = reporter[:-1]
        if stripped not in variants:
            variants.append(stripped)

    # Add dotted forms for bare reporter abbreviations such as "US" -> "U.S."
    # and "S Ct" -> "S. Ct.".
    tokens = [part for part in reporter.split() if part]
    dotted_tokens: list[str] = []
    changed = False
    for token in tokens:
        if "." in token:
            dotted_tokens.append(token)
            continue
        if token.isupper() and len(token) <= 3:
            dotted_tokens.append(".".join(token) + ".")
            changed = True
            continue
        if token[0:1].isupper() and token[1:].islower() and len(token) <= 3:
            dotted_tokens.append(f"{token}.")
            changed = True
            continue
        dotted_tokens.append(token)
    if changed:
        dotted = " ".join(dotted_tokens)
        if dotted not in variants and dotted != reporter:
            variants.append(dotted)

    return variants


def _extract_parties_from_context(text: str) -> tuple[str, str]:
    """Extract plaintiff/defendant from surrounding text by finding an
    ``X v. Y`` pattern.

    Useful when eyecite doesn't capture party names — for example, when a
    LEXIS citation appears in a separate sentence from the case caption::

        "Lord v. Hannon Armstrong Capital, LLC, where Lord signed ...
         2015 Md. App. LEXIS 201."

    Returns (plaintiff, defendant) or ("", "").
    """
    # Match "Capitalized-Word(s) v. Capitalized-Word(s)" with optional
    # legal entity suffixes (Inc., LLC, Corp., etc.)
    m = re.search(
        r'([A-Z][A-Za-z\'\-]+(?:\s+(?:[A-Z&][A-Za-z\'\-]*|of|the|in|for))*'
        r'(?:,?\s*(?:Inc\.?|LLC|L\.L\.C\.?|Corp\.?|Co\.?|Ltd\.?))?)'
        r'\s+v\.\s+'
        r'([A-Z][A-Za-z\'\-]+(?:\s+(?:[A-Z&][A-Za-z\'\-]*|of|the|in|for))*'
        r'(?:,?\s*(?:Inc\.?|LLC|L\.L\.C\.?|Corp\.?|Co\.?|Ltd\.?))?)',
        text,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", ""


_CASE_TOKEN_STOPWORDS = {
    "the", "and", "for", "of", "in", "on", "at", "to", "by", "et", "al",
    "inc", "llc", "corp", "co", "ltd", "company", "corporation", "assoc",
    "assocs", "associates", "department", "board", "commission",
}


def _normalize_case_caption(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\bversus\b|\bvs\.?\b", " v ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_case_caption(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    parts = re.split(r"\s+v(?:\.|s\.?)?\s+", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", ""


def _caption_tokens(text: str) -> set[str]:
    normalized = _normalize_case_caption(text)
    tokens = []
    for token in normalized.split():
        if token in _CASE_TOKEN_STOPWORDS:
            continue
        if len(token) <= 2 and token != "v":
            continue
        tokens.append(token)
    return set(tokens)


def _token_overlap(expected: str, actual: str) -> float:
    expected_tokens = _caption_tokens(expected)
    if not expected_tokens:
        return 0.0
    actual_tokens = _caption_tokens(actual)
    if not actual_tokens:
        return 0.0
    return len(expected_tokens & actual_tokens) / max(1, len(expected_tokens))


def _candidate_preview_from_hit(hit: dict) -> dict:
    cl_url = _build_cl_opinion_url(hit)
    return {
        "case_name": hit.get("caseName", "") or hit.get("case_name", ""),
        "date_filed": hit.get("dateFiled", "") or hit.get("date_filed", ""),
        "court": hit.get("court", "") or hit.get("court_citation_string", ""),
        "courtlistener_url": cl_url,
    }


def _build_lookup_result_from_hit(hit: dict, headers: dict) -> Dict[str, Any]:
    cluster_id = hit.get("cluster_id") or _extract_cluster_id(hit)
    opinion_id = hit.get("id") or hit.get("opinion_id") or ""
    court_slug = hit.get("court_id", "") or hit.get("court", "")
    opinion_snippet = _get_opinion_snippet(hit, cluster_id, headers)

    cl_url = _build_cl_opinion_url(hit, cluster_id)

    return {
        "found": True,
        "match_status": "matched",
        "case_name": hit.get("caseName", "") or hit.get("case_name", ""),
        "court": hit.get("court", "") or hit.get("court_citation_string", ""),
        "court_id": str(court_slug),
        "cluster_id": str(cluster_id) if cluster_id else "",
        "opinion_id": str(opinion_id) if opinion_id else "",
        "date_filed": hit.get("dateFiled", "") or hit.get("date_filed", ""),
        "courtlistener_url": cl_url,
        "opinion_snippet": opinion_snippet,
    }


def _score_cl_candidate(cite: Dict[str, Any], hit: dict, query_mode: str) -> tuple[int, list[str]]:
    plaintiff = (cite.get("plaintiff") or "").strip()
    defendant = (cite.get("defendant") or "").strip()
    year = str(cite.get("year") or "").strip()
    candidate_name = hit.get("caseName", "") or hit.get("case_name", "") or ""
    candidate_date = hit.get("dateFiled", "") or hit.get("date_filed", "") or ""
    expected_caption = ""
    if plaintiff and defendant:
        expected_caption = f"{plaintiff} v. {defendant}"
    else:
        expected_caption = plaintiff or defendant
    candidate_plaintiff, candidate_defendant = _split_case_caption(candidate_name)
    score = {
        "citation": 3,
        "citation_text": 3,
        "caption_year": 2,
        "caption_text": 2,
        "caption_relaxed": 1,
        "caption_vs": 1,
        "defendant": 0,
        "plaintiff": 0,
    }.get(query_mode, 0)
    reasons: list[str] = [query_mode]

    normalized_expected = _normalize_case_caption(expected_caption)
    normalized_candidate = _normalize_case_caption(candidate_name)
    if normalized_expected and normalized_candidate:
        if normalized_expected == normalized_candidate:
            score += 8
            reasons.append("caption_exact")
        elif normalized_expected in normalized_candidate or normalized_candidate in normalized_expected:
            score += 6
            reasons.append("caption_contains")

    plaintiff_overlap = _token_overlap(plaintiff, candidate_plaintiff or candidate_name)
    defendant_overlap = _token_overlap(defendant, candidate_defendant or candidate_name)

    if plaintiff_overlap > 0:
        score += round(plaintiff_overlap * 4)
        reasons.append(f"plaintiff:{plaintiff_overlap:.2f}")
    if defendant_overlap > 0:
        score += round(defendant_overlap * 5)
        reasons.append(f"defendant:{defendant_overlap:.2f}")
    if plaintiff and defendant and plaintiff_overlap >= 0.5 and defendant_overlap >= 0.5:
        score += 2
        reasons.append("both_parties")
    if plaintiff and defendant:
        if plaintiff_overlap == 0 and defendant_overlap >= 0.6:
            score -= 2
            reasons.append("missing_plaintiff")
        if defendant_overlap == 0 and plaintiff_overlap >= 0.6:
            score -= 3
            reasons.append("missing_defendant")

    if year and candidate_date[:4].isdigit():
        candidate_year = candidate_date[:4]
        if candidate_year == year:
            score += 3
            reasons.append("year_exact")
        elif abs(int(candidate_year) - int(year)) == 1:
            score += 1
            reasons.append("year_close")
        else:
            score -= 3
            reasons.append("year_mismatch")

    if query_mode == "citation" and not plaintiff and not defendant:
        score += 2
        reasons.append("citation_only")

    return score, reasons


def _min_confident_match_score(query_mode: str) -> int:
    return {
        "citation": 5,
        "citation_text": 5,
        "caption_year": 7,
        "caption_text": 6,
        "caption_relaxed": 6,
        "caption_vs": 6,
        "defendant": 5,
        "plaintiff": 5,
    }.get(query_mode, 6)


def _format_ambiguous_match_note(candidates: list[dict]) -> str:
    previews = []
    for item in candidates[:3]:
        label = item.get("case_name") or "Unnamed case"
        year = (item.get("date_filed") or "")[:4]
        if year:
            label += f" ({year})"
        previews.append(label)
    preview_text = "; ".join(previews) if previews else "multiple possible matches"
    return (
        "CourtListener returned multiple plausible matches for this citation: "
        f"{preview_text}. Review the case caption, reporter, or year."
    )


def _select_best_cl_hit(cite: Dict[str, Any], hits: List[dict], query_mode: str) -> dict:
    if not hits:
        return {"status": "none"}

    deduped_hits: list[dict] = []
    seen_keys: set[str] = set()
    for hit in hits:
        cluster_id = str(hit.get("cluster_id") or _extract_cluster_id(hit) or "").strip()
        case_name = (hit.get("caseName", "") or hit.get("case_name", "") or "").strip().lower()
        date_filed = (hit.get("dateFiled", "") or hit.get("date_filed", "") or "").strip()
        dedupe_key = cluster_id or f"{case_name}|{date_filed}"
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_hits.append(hit)

    scored = []
    for hit in deduped_hits:
        score, reasons = _score_cl_candidate(cite, hit, query_mode)
        scored.append({
            "hit": hit,
            "score": score,
            "reasons": reasons,
            "preview": _candidate_preview_from_hit(hit),
        })

    scored.sort(key=lambda row: (row["score"], row["preview"].get("date_filed", "")), reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    min_score = _min_confident_match_score(query_mode)

    if best["score"] < min_score:
        return {"status": "none"}

    if second and second["score"] >= min_score and (best["score"] - second["score"]) <= 1:
        return {
            "status": "ambiguous",
            "score": best["score"],
            "candidates": [row["preview"] for row in scored[:3]],
        }

    return {
        "status": "matched",
        "score": best["score"],
        "hit": best["hit"],
        "reasons": best["reasons"],
    }


def _run_cl_lookup_strategy(
    cite: Dict[str, Any],
    *,
    query: str,
    headers: dict,
    query_mode: str,
) -> dict:
    hits = _cl_search_candidates(query, headers)
    if not hits:
        return {"status": "none"}

    selection = _select_best_cl_hit(cite, hits, query_mode)
    if selection["status"] == "matched":
        return {
            "status": "matched",
            "score": selection["score"],
            "result": _build_lookup_result_from_hit(selection["hit"], headers),
        }
    if selection["status"] == "ambiguous":
        return {
            "status": "ambiguous",
            "score": selection["score"],
            "result": {
                "found": False,
                "match_status": "ambiguous",
                "confidence_note": _format_ambiguous_match_note(selection["candidates"]),
                "candidate_matches": selection["candidates"],
                "case_name": "",
                "court": "",
                "date_filed": "",
                "courtlistener_url": (selection["candidates"][0] if selection["candidates"] else {}).get("courtlistener_url", ""),
                "opinion_snippet": "",
            },
        }
    return {"status": "none"}


def lookup_citation(cite: Dict[str, Any], *, rate_limit: float = 1.0) -> Dict[str, Any]:
    """Search CourtListener for a citation. Returns enriched dict with lookup results.

    Tries (1) citation string search, (1b) reporter variant retries,
    then (2) party name + year fallback (with context-based party name
    extraction when eyecite didn't provide them).
    """
    headers = _cl_headers()
    volume = cite.get("volume", "")
    reporter = cite.get("reporter", "")
    page = cite.get("page", "")

    result: Dict[str, Any] = {
        "found": False,
        "match_status": "not_found",
        "case_name": "",
        "court": "",
        "date_filed": "",
        "courtlistener_url": "",
        "opinion_snippet": "",
        "confidence_note": "",
        "candidate_matches": [],
    }
    best_ambiguous: Dict[str, Any] | None = None

    # Strategy 1: search by citation string (most reliable)
    if volume and reporter and page:
        query = f'citation:("{volume} {reporter} {page}")'
        outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="citation")
        if outcome["status"] == "matched":
            result.update(outcome["result"])
            return result
        if outcome["status"] == "ambiguous":
            best_ambiguous = outcome
        time.sleep(rate_limit)

        # Strategy 1b: retry with reporter abbreviation variants
        for alt_reporter in _reporter_variants(reporter):
            query = f'citation:("{volume} {alt_reporter} {page}")'
            outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="citation")
            if outcome["status"] == "matched":
                log.info("Found citation with alt reporter: %s → %s",
                         reporter, alt_reporter)
                result.update(outcome["result"])
                return result
            if outcome["status"] == "ambiguous" and (
                not best_ambiguous or outcome["score"] > best_ambiguous["score"]
            ):
                best_ambiguous = outcome
            time.sleep(rate_limit)

        # Strategy 1c: fall back to broader text search on the citation string.
        # CourtListener's frontend sometimes finds these even when citation:(...)
        # does not.
        broad_queries: list[str] = []
        seen_broad_queries: set[str] = set()
        raw_citation_text = str(cite.get("citation_text") or "").strip()
        canonical_citation_text = " ".join(
            part.strip()
            for part in (str(volume), str(reporter), str(page))
            if part and part.strip()
        )
        broad_candidates = [raw_citation_text, canonical_citation_text]
        broad_candidates.extend(
            f"{volume} {alt_reporter} {page}".strip()
            for alt_reporter in _reporter_variants(reporter)
            if alt_reporter
        )
        for candidate in broad_candidates:
            candidate = candidate.strip()
            if not candidate or candidate in seen_broad_queries:
                continue
            seen_broad_queries.add(candidate)
            broad_queries.append(f'"{candidate}"')

        for query in broad_queries:
            outcome = _run_cl_lookup_strategy(
                cite,
                query=query,
                headers=headers,
                query_mode="citation_text",
            )
            if outcome["status"] == "matched":
                log.info("Found citation via broad text search: %s", query)
                result.update(outcome["result"])
                return result
            if outcome["status"] == "ambiguous" and (
                not best_ambiguous or outcome["score"] > best_ambiguous["score"]
            ):
                best_ambiguous = outcome
            time.sleep(rate_limit)

    # Strategy 1d: CourtListener citation= search parameter.
    # CL's frontend uses a separate `citation` parameter that searches
    # across the citation index differently from the `citation:()` query
    # operator.  The actual case often appears in the results alongside
    # cases that *cite* it.  We scan the top results and match on the
    # citation string appearing in the result's own citation list.
    if volume and reporter and page:
        cite_str_for_param = f"{volume} {reporter} {page}"
        try:
            params = {
                "q": cite_str_for_param,
                "citation": cite_str_for_param,
                "type": "o",
                "stat_Published": "on",
                "order_by": "score desc",
                "format": "json",
            }
            resp = _cl_get(CL_API_V4, params=params, headers=headers, timeout=15)
            if resp.ok:
                hits = resp.json().get("results", [])
                # Look through more results than usual — the target case
                # may not be first (it's mixed with citing cases)
                for hit in hits[:20]:
                    # Check if this hit's own citations list contains our target.
                    # CL search results include a `citation` field with the
                    # case's own citations.
                    hit_cites = hit.get("citation") or []
                    if isinstance(hit_cites, list):
                        for hc in hit_cites:
                            if (str(volume) in str(hc)
                                    and str(page) in str(hc)
                                    and any(rp in str(hc) for rp in [reporter, reporter.replace(".", "")])):
                                # This hit IS the case we're looking for
                                log.info("Found via citation= param: %s → %s",
                                         cite_str_for_param, hit.get("caseName", "?"))
                                result.update(_build_lookup_result_from_hit(hit, headers))
                                return result
        except Exception as e:
            log.warning("Strategy 1d (citation= param) failed: %s", e)
        time.sleep(rate_limit)

    # Strategy 2: party names + year
    plaintiff = cite.get("plaintiff", "")
    defendant = cite.get("defendant", "")
    year = cite.get("year", "")

    # Fallback: extract party names from the proposition text when eyecite
    # didn't provide them (common with LEXIS citations where the case
    # caption appears in a different sentence from the citation itself).
    if not plaintiff or not defendant:
        proposition = cite.get("proposition", "")
        if proposition:
            ctx_p, ctx_d = _extract_parties_from_context(proposition)
            if ctx_p and not plaintiff:
                plaintiff = ctx_p
                log.info("Extracted plaintiff from context: %s", plaintiff)
            if ctx_d and not defendant:
                defendant = ctx_d
                log.info("Extracted defendant from context: %s", defendant)
    if plaintiff and defendant:
        parties = f"{plaintiff} v. {defendant}"
        query = f'caseName:("{parties}")'
        if year:
            query += f" filed_after:{year}-01-01 filed_before:{year}-12-31"
        outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="caption_year")
        if outcome["status"] == "matched":
            result.update(outcome["result"])
            return result
        if outcome["status"] == "ambiguous" and (
            not best_ambiguous or outcome["score"] > best_ambiguous["score"]
        ):
            best_ambiguous = outcome
        time.sleep(rate_limit)

        # Strategy 2b: broader text search on party-name hints. This catches
        # official captions that include extra entity words not present in the
        # textbook, such as "Meritor Savings Bank, FSB v. Vinson".
        query = f'"{plaintiff}" "{defendant}"'
        if year:
            query += f" filed_after:{year}-01-01 filed_before:{year}-12-31"
        outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="caption_text")
        if outcome["status"] == "matched":
            log.info("Found via caption text search: %s v. %s", plaintiff, defendant)
            result.update(outcome["result"])
            return result
        if outcome["status"] == "ambiguous" and (
            not best_ambiguous or outcome["score"] > best_ambiguous["score"]
        ):
            best_ambiguous = outcome
        time.sleep(rate_limit)

        # Strategy 2c: party names with year as a TEXT term instead of a
        # date-range filter.  CourtListener's filed_after/filed_before
        # filters fail for some older SCOTUS opinions due to metadata gaps,
        # but the year as plain text matches opinions that mention it.
        if year:
            query = f'"{plaintiff}" "{defendant}" {year}'
            outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="caption_text")
            if outcome["status"] == "matched":
                log.info("Found via caption text + year-as-text: %s v. %s (%s)",
                         plaintiff, defendant, year)
                result.update(outcome["result"])
                return result
            if outcome["status"] == "ambiguous" and (
                not best_ambiguous or outcome["score"] > best_ambiguous["score"]
            ):
                best_ambiguous = outcome
            time.sleep(rate_limit)

    # Strategy 3: broader case-name search (relaxed constraints)
    # When a citation string can't be found (e.g. malformed page number),
    # try searching by party names without the year constraint, or by
    # individual party names if the combined search didn't work.
    if plaintiff and defendant:
        # 3a: combined names without year filter
        if year:
            query = f'caseName:("{plaintiff} v. {defendant}")'
            outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="caption_relaxed")
            if outcome["status"] == "matched":
                log.info("Found via case-name search (no year filter): %s v. %s",
                         plaintiff, defendant)
                result.update(outcome["result"])
                return result
            if outcome["status"] == "ambiguous" and (
                not best_ambiguous or outcome["score"] > best_ambiguous["score"]
            ):
                best_ambiguous = outcome
            time.sleep(rate_limit)

        # 3b: try with "vs" instead of "v."
        query = f'caseName:("{plaintiff} vs {defendant}")'
        outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="caption_vs")
        if outcome["status"] == "matched":
            log.info("Found via case-name search (vs variant): %s vs %s",
                     plaintiff, defendant)
            result.update(outcome["result"])
            return result
        if outcome["status"] == "ambiguous" and (
            not best_ambiguous or outcome["score"] > best_ambiguous["score"]
        ):
            best_ambiguous = outcome
        time.sleep(rate_limit)

    # 3c: search by defendant name alone (often more distinctive)
    if defendant and len(defendant) > 3:
        query = f'caseName:("{defendant}")'
        if year:
            query += f" filed_after:{year}-01-01 filed_before:{year}-12-31"
        if volume and reporter:
            query += f' citation:("{volume} {reporter}")'
        outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="defendant")
        if outcome["status"] == "matched":
            log.info("Found via defendant-only search: %s", defendant)
            result.update(outcome["result"])
            return result
        if outcome["status"] == "ambiguous" and (
            not best_ambiguous or outcome["score"] > best_ambiguous["score"]
        ):
            best_ambiguous = outcome
        time.sleep(rate_limit)

    # 3d: search by plaintiff name alone
    if plaintiff and len(plaintiff) > 3:
        query = f'caseName:("{plaintiff}")'
        if year:
            query += f" filed_after:{year}-01-01 filed_before:{year}-12-31"
        if volume and reporter:
            query += f' citation:("{volume} {reporter}")'
        outcome = _run_cl_lookup_strategy(cite, query=query, headers=headers, query_mode="plaintiff")
        if outcome["status"] == "matched":
            log.info("Found via plaintiff-only search: %s", plaintiff)
            result.update(outcome["result"])
            return result
        if outcome["status"] == "ambiguous" and (
            not best_ambiguous or outcome["score"] > best_ambiguous["score"]
        ):
            best_ambiguous = outcome

    if best_ambiguous:
        result.update(best_ambiguous["result"])
    return result


# ---------------------------------------------------------------------------
# Statutory lookup via Cornell LII
# ---------------------------------------------------------------------------

def statute_range_start(cite: Dict[str, Any]) -> str:
    """First section of a multi-section cite ("9 U.S.C. §§ 9-10" → "9"),
    else "". Cornell LII has a page per section, never per range, so a
    range always 404s as written.

    Only a "§§" cite is read as a range: hyphens are also part of real
    section numbers (42 U.S.C. § 2000e-2, 15 U.S.C. § 78j-1), so "A-B" is
    split only when both sides are plain numbers and B > A. Not used by
    ``lookup_statute`` itself — callers opt in.
    """
    text = cite.get("citation_text", "") or ""
    section = (cite.get("section", "") or "").strip()
    if "§§" not in text or not section:
        return ""
    first = re.split(r"\s*(?:,|;|–|—|\bto\b|\bthrough\b|\bet seq\b)\s*",
                     section, maxsplit=1)[0].strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", first)
    if m and int(m.group(2)) > int(m.group(1)):
        first = m.group(1)
    return first if first and first != section else ""


def lookup_statute(cite: Dict[str, Any], *, rate_limit: float = 1.5) -> Dict[str, Any]:
    """Check Cornell LII for a statutory citation.

    Supports U.S.C. and C.F.R. citations. Other reporters (state codes)
    return not_found with a note that the reporter is unsupported.

    Returns:
        found         – bool
        statute_name  – descriptive name from page title
        source_url    – Cornell LII URL
        statute_text  – extracted text for relevance check
    """
    reporter = cite.get("reporter", "")
    title = cite.get("title", "")
    section = cite.get("section", "")

    result: Dict[str, Any] = {
        "found": False,
        "statute_name": "",
        "source_url": "",
        "statute_text": "",
    }

    if not section:
        return result

    # Strip subsection parentheticals for URL: "1983(a)(1)" → "1983"
    base_section = re.sub(r'\(.*$', '', section).strip()

    # Determine URL pattern from reporter
    reporter_upper = reporter.upper().replace(" ", "").replace(".", "")
    if "USC" in reporter_upper:
        if not title:
            return result
        url = CORNELL_USC_URL.format(title=title, section=base_section)
    elif "CFR" in reporter_upper:
        if not title:
            return result
        url = CORNELL_CFR_URL.format(title=title, section=base_section)
    else:
        result["statute_name"] = cite.get("citation_text", "")
        return result

    # Fetch from Cornell LII
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "local-first-read/1.0 (educational tool)",
        })
        time.sleep(rate_limit)

        if resp.status_code == 404:
            return result
        if not resp.ok:
            log.warning("Cornell LII request failed %s: %s", resp.status_code, url)
            return result

        html = resp.text
        statute_name = _extract_cornell_title(html) or cite.get("citation_text", "")
        statute_text = _extract_cornell_text(html)

        result["found"] = True
        result["statute_name"] = statute_name
        result["source_url"] = url
        result["statute_text"] = statute_text
        return result

    except Exception:
        log.exception("Cornell LII fetch error: %s", url)
        return result


def _extract_cornell_title(html: str) -> str:
    """Extract the page title from Cornell LII HTML."""
    m = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        # Remove " | LII / Legal Information Institute" suffix
        title = re.sub(r'\s*\|.*$', '', title)
        # Decode common HTML entities
        title = title.replace('&sect;', '§').replace('&amp;', '&')
        title = title.replace('&mdash;', '—').replace('&ndash;', '–')
        return title
    return ""


def _extract_cornell_text(html: str, budget: int = 5000) -> str:
    """Extract statute text from Cornell LII HTML page.

    Strips HTML tags and returns up to *budget* characters of the main
    content area for the LLM relevance check.  Cornell uses a
    ``tab_default_1`` div for the primary statutory text on both USC
    and CFR pages.
    """
    # Try to isolate the main content area (tab_default_1 on Cornell)
    content = html
    m = re.search(
        r'id="tab_default_1"[^>]*>([\s\S]*?)'
        r'(?:<div[^>]*id="tab_default_2|</body>|$)',
        html,
    )
    if m and len(m.group(1)) > 50:
        content = m.group(1)

    # Strip HTML tags
    clean = re.sub(r'<[^>]+>', '', content)
    # Decode common HTML entities
    clean = clean.replace('&sect;', '§').replace('&amp;', '&')
    clean = clean.replace('&mdash;', '—').replace('&ndash;', '–')
    clean = clean.replace('&nbsp;', ' ').replace('&quot;', '"')
    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:budget]


def _cl_search_candidates(query: str, headers: dict, *, limit: int = 5) -> List[dict]:
    """Execute a CourtListener v4 search and return raw candidate hits."""
    params = {"q": query, "type": "o", "format": "json"}
    try:
        resp = _cl_get(CL_API_V4, params=params, headers=headers, timeout=15)
        if not resp.ok:
            log.warning("CL search failed %s: %s", resp.status_code, query)
            return []

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return []
        return results[:limit]
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        # Transient upstream blip — already retried by the session. Log a
        # concise line rather than a full traceback; the caller falls through
        # to its next lookup strategy (or skips this citation).
        log.warning("CL search unreachable (after retries) for query %s: %s", query, e)
        return []
    except Exception:
        log.exception("CL search error for query: %s", query)
        return []


def _extract_cluster_id(hit: dict) -> Optional[str]:
    """Try to extract cluster_id from various CL response formats."""
    # v4 search results sometimes use 'cluster_id' directly
    cid = hit.get("cluster_id")
    if cid:
        return str(cid)
    # Or the cluster URL
    cluster_url = hit.get("cluster") or ""
    if cluster_url:
        parts = cluster_url.rstrip("/").split("/")
        if parts:
            return parts[-1]
    return None


def _build_cl_opinion_url(hit: dict, cluster_id: str | None = None) -> str:
    """Build a CourtListener opinion URL from a search-API hit dict.

    CourtListener requires the case-name slug in the URL path; without it the
    server returns 404.  The v4 search API conveniently includes an
    ``absolute_url`` field (e.g. ``/opinion/12345/doe-v-roe/``) that we
    prefer.  Falls back to constructing from *cluster_id* when no slug is
    available (rare — only when the hit comes from a non-search endpoint).
    """
    abs_url = (hit.get("absolute_url") or "").strip()
    if abs_url:
        # absolute_url is a path like "/opinion/12345/doe-v-roe/"
        return f"https://www.courtlistener.com{abs_url}"
    cid = cluster_id or _extract_cluster_id(hit)
    if cid:
        # Slug-less fallback — may 404 on CL front-end but is still a
        # reasonable best-effort link.
        return f"https://www.courtlistener.com/opinion/{cid}/"
    return ""


_OPINION_TEXT_FIELDS = (
    "plain_text", "html_with_citations", "html",
    "html_columbia", "html_lawbox", "xml_harvard",
)


def _clean_snippet(val: str) -> str:
    """Strip HTML/XML tags and keep up to 60 000 chars of opinion text.

    We store a generous buffer here because `_build_relevance_excerpt`
    will later select the most relevant portions (beginning, end, and
    keyword-matched sections) to fit within the LLM's prompt budget.
    The full text is only held in memory during the Celery task — it is
    not persisted to the database.
    """
    clean = re.sub(r"<[^>]+>", "", val)
    return clean.strip()[:60000]


def _get_opinion_snippet(hit: dict, cluster_id: Optional[str],
                         headers: dict) -> str:
    """Extract a useful text snippet from the search hit, opinion endpoint,
    or cluster sub-opinions (fallback).

    Tries three strategies in order:
      1. Inline text from the search-hit dict
      2. Opinion REST endpoint (by opinion id from the hit)
      3. Cluster REST endpoint → iterate sub_opinions for text
    """

    # --- Strategy 1: inline text from search hit ---
    for field in ("snippet", "text", "plain_text", "html"):
        val = hit.get(field, "")
        if val and len(val) > 50:
            return _clean_snippet(val)

    # --- Strategy 2: opinion REST endpoint ---
    opinion_id = hit.get("id") or hit.get("opinion_id")
    if opinion_id:
        snippet = _fetch_opinion_text(opinion_id, headers)
        if snippet:
            return snippet

    # --- Strategy 3: cluster → sub_opinions ---
    if cluster_id:
        snippet = _fetch_text_via_cluster(cluster_id, headers)
        if snippet:
            return snippet

    return ""


def _fetch_opinion_text(opinion_id, headers: dict) -> str:
    """Fetch a single opinion by id and return its text, or ''."""
    try:
        resp = requests.get(
            f"{CL_OPINIONS_V4}{opinion_id}/",
            headers=headers,
            params={"format": "json"},
            timeout=15,
        )
        if resp.ok:
            odata = resp.json()
            for field in _OPINION_TEXT_FIELDS:
                val = odata.get(field, "")
                if val and len(val) > 50:
                    return _clean_snippet(val)
    except Exception:
        log.warning("Failed to fetch opinion text for id=%s", opinion_id)
    return ""


def _fetch_text_via_cluster(cluster_id: str, headers: dict) -> str:
    """Fetch the cluster, iterate its sub_opinions, and return the first
    opinion text we find.  This is the most reliable fallback because
    CourtListener always populates sub_opinions on the cluster object."""
    try:
        resp = requests.get(
            f"{CL_CLUSTERS_V4}{cluster_id}/",
            headers=headers,
            params={"format": "json"},
            timeout=15,
        )
        if not resp.ok:
            log.warning("CL cluster fetch failed %s for cluster %s",
                        resp.status_code, cluster_id)
            return ""

        cluster_data = resp.json()
        sub_opinion_urls = cluster_data.get("sub_opinions", [])
        if not sub_opinion_urls:
            log.info("Cluster %s has no sub_opinions listed", cluster_id)
            return ""

        for op_url in sub_opinion_urls[:3]:  # limit to first 3 to avoid delays
            try:
                op_resp = requests.get(
                    op_url,
                    headers=headers,
                    params={"format": "json"},
                    timeout=15,
                )
                if not op_resp.ok:
                    continue
                op_data = op_resp.json()
                for field in _OPINION_TEXT_FIELDS:
                    val = op_data.get(field, "")
                    if val and len(val) > 50:
                        return _clean_snippet(val)
            except Exception:
                log.warning("Failed to fetch sub-opinion from %s", op_url)
                continue

    except Exception:
        log.exception("Failed to fetch cluster %s for opinion text", cluster_id)

    return ""


def _build_relevance_excerpt(opinion_text: str, proposition: str,
                             budget: int = 5000) -> str:
    """Build a smart excerpt from a (potentially long) opinion for the
    relevance-check LLM.

    Court opinions typically follow: caption → procedural history → facts →
    analysis → holding/conclusion.  A naive head-truncation captures the
    preamble but misses the holding.  This function allocates the character
    budget across three zones:

        1. HEAD  – first ~30% (case identification, parties, posture)
        2. TAIL  – last  ~30% (holding, conclusion, disposition)
        3. KEYWORD MATCHES – ~40% around sentences that share key terms
           with the brief's proposition

    If the opinion fits within the budget, it is returned as-is.
    """
    if len(opinion_text) <= budget:
        return opinion_text

    head_budget = int(budget * 0.30)
    tail_budget = int(budget * 0.30)
    kw_budget = budget - head_budget - tail_budget  # ~40%

    head = opinion_text[:head_budget]
    tail = opinion_text[-tail_budget:]

    # --- keyword-matched middle section ---
    # Extract significant words from the proposition (4+ chars, not common)
    stop_words = {
        'that', 'this', 'with', 'from', 'were', 'been', 'have', 'which',
        'their', 'would', 'could', 'should', 'about', 'other', 'there',
        'when', 'upon', 'such', 'also', 'into', 'than', 'them', 'then',
        'being', 'does', 'each', 'between', 'because', 'court', 'case',
        'held', 'stated', 'however', 'therefore', 'appellee', 'appellant',
    }
    prop_words = set()
    for w in re.findall(r'[A-Za-z]{4,}', proposition.lower()):
        if w not in stop_words:
            prop_words.add(w)

    # Find sentences in the middle zone that contain proposition keywords
    middle_start = head_budget
    middle_end = len(opinion_text) - tail_budget
    middle = opinion_text[middle_start:middle_end]

    if prop_words and middle:
        # Split middle into rough sentence chunks and score them
        # Use a simple split on ". " that won't break on abbreviations
        chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', middle)
        scored: list[tuple[int, int, str]] = []  # (score, position, chunk)
        pos = 0
        for chunk in chunks:
            score = sum(1 for w in re.findall(r'[A-Za-z]{4,}', chunk.lower())
                        if w in prop_words)
            scored.append((score, pos, chunk))
            pos += len(chunk) + 1

        # Sort by score desc, pick top chunks that fit in budget
        scored.sort(key=lambda x: -x[0])
        kw_parts: list[tuple[int, str]] = []  # (position, text)
        used = 0
        for score, position, chunk in scored:
            if score == 0:
                break
            if used + len(chunk) > kw_budget:
                continue
            kw_parts.append((position, chunk))
            used += len(chunk)

        # Sort selected chunks by their original position for readability
        kw_parts.sort(key=lambda x: x[0])
        kw_section = " [...] ".join(part for _, part in kw_parts)
    else:
        kw_section = ""

    # --- assemble ---
    parts = [head.rstrip()]
    if kw_section:
        parts.append("\n[...]\n" + kw_section.strip())
    parts.append("\n[...]\n" + tail.lstrip())

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Relevance check via Anthropic
# ---------------------------------------------------------------------------

def _research_grounding_block(research_context: str) -> str:
    """Wrap an LO research readme as an authoritative grounding block for the
    relevance prompts. Empty string in → empty string out (no grounding)."""
    if not research_context:
        return ""
    return (
        "\nAUTHORITATIVE RESEARCH CONTEXT — the instructor's briefing on exactly "
        "what this part of the curriculum teaches. Treat it as the source of "
        "truth when judging relevance: an authority engages this objective only "
        "if it speaks to the doctrine described here, and the \"Out of Scope\" "
        "boundaries are decisive for rejecting false positives.\n"
        f"{research_context[:4000]}\n"
    )


def check_relevance(proposition: str, case_name: str,
                    opinion_snippet: str,
                    citation_type: str = "case",
                    research_context: str = "",
                    llm: Optional[LLMFn] = None) -> Dict[str, str]:
    """Ask the injected ``llm`` whether the cited authority is topically relevant
    and whether the brief accurately characterizes it.

    Works for both case law and statutory citations. Pass ``llm`` to run the
    same prompt through another model (see ``LLMFn``); a failure on that
    path returns status ``unknown`` rather than the production fail-open
    ``found_relevant``, because a local run has no second line of defence.

    Returns:
        {"status": "found_relevant" | "found_miscited" | "found_unrelated",
         "summary": "...",
         "confidence_note": "..."}

    Status meanings:
        found_relevant    – authority is topically relevant AND the brief's
                            characterization is accurate (or close enough)
        found_miscited    – authority is topically relevant but the brief
                            mischaracterizes the content
        found_unrelated   – authority deals with a completely different area
                            of law and does not support the proposition
    """
    is_statute = (citation_type == "statute")
    source_label = "Statute" if is_statute else "Case"

    if not opinion_snippet or not proposition:
        return {
            "status": "found_relevant",  # benefit of the doubt when no text
            "summary": f"{source_label} found: {case_name}. No text available to verify relevance.",
            "confidence_note": f"Could not retrieve {source_label.lower()} text for relevance check.",
        }

    if llm is None:
        raise ValueError("check_relevance needs an llm callable (see LocalJudge)")

    # Build a smart excerpt that covers the beginning, end, and
    # keyword-relevant sections — not just the first N chars.
    excerpt = _build_relevance_excerpt(opinion_snippet, proposition, budget=5000)

    research_block = _research_grounding_block(research_context)

    if is_statute:
        prompt = f"""A legal brief cites the statutory provision "{case_name}".
Below is the passage from the brief where the citation appears. The legal
claim the author is making is typically in the sentence(s) BEFORE the
citation.

PASSAGE FROM BRIEF:
{proposition[:1500]}
{research_block}
Here is the text of the statutory provision:

STATUTE TEXT:
{excerpt}

Answer these TWO separate questions:

QUESTION 1 — TOPICAL RELEVANCE:
Is this statute about the same general area of law as the proposition the
brief is arguing? Only answer NO if the statute is about a completely
unrelated legal topic.

QUESTION 2 — CHARACTERIZATION ACCURACY:
Does the brief accurately describe what this statute provides or requires?
Answer YES if the statute supports the brief's characterization.
Answer CANNOT DETERMINE if the statute text is insufficient to verify.
Answer NO only if the statute clearly contradicts the brief's claims.

Respond in this exact format:
TOPICALLY_RELEVANT: YES or NO
ACCURATELY_DESCRIBED: YES or NO or CANNOT_DETERMINE
SUMMARY: [1-2 sentences summarizing what this statute provides]
NOTE: [If the brief mischaracterizes the statute, explain the specific contradiction. Otherwise "N/A"]"""
    else:
        prompt = f"""A legal brief cites the case "{case_name}". Below is the passage
from the brief where the citation appears. The legal claim the author is
making is typically in the sentence(s) BEFORE the citation — the citation
itself may just be a case name + reporter reference.

PASSAGE FROM BRIEF:
{proposition[:1500]}
{research_block}
Here is an excerpt from the actual court opinion. If the opinion was long,
sections have been selected from the beginning, end, and keyword-relevant
portions, with "[...]" indicating omitted text.

OPINION EXCERPT:
{excerpt}

Answer these TWO separate questions:

QUESTION 1 — TOPICAL RELEVANCE:
Is this case about the same general area of law as the proposition the
brief is arguing? For example, if the brief discusses non-compete
agreements and the case is about employment restrictive covenants, that
IS topically relevant even if the specific holding differs. Only answer
NO if the case is about a completely unrelated legal topic.

QUESTION 2 — CHARACTERIZATION ACCURACY:
Does the brief accurately describe this case's holding, facts, and
procedural posture?

IMPORTANT: The opinion excerpt above may be incomplete — it is only a
portion of the full opinion. You should ONLY answer NO if the excerpt
ACTIVELY CONTRADICTS what the brief claims (e.g. the brief says the
court ruled for the plaintiff but the excerpt shows the court ruled
for the defendant, or the brief gets the parties' roles reversed).

Do NOT answer NO simply because the excerpt does not mention or confirm
something the brief claims. If the brief discusses an aspect of the
holding that is not covered in the excerpt, that is consistent with the
excerpt being truncated — answer CANNOT DETERMINE in that situation.

Answer YES if the excerpt supports the brief's characterization.
Answer CANNOT DETERMINE if the excerpt is silent on the brief's claims
or does not contain enough information to verify the characterization.
Answer NO only if the excerpt clearly contradicts the brief's claims.

Respond in this exact format:
TOPICALLY_RELEVANT: YES or NO
ACCURATELY_DESCRIBED: YES or NO or CANNOT_DETERMINE
SUMMARY: [1-2 sentences summarizing what this case is actually about]
NOTE: [If the brief mischaracterizes the case, explain the specific contradiction from the excerpt. Otherwise "N/A"]"""

    try:
        text = llm(prompt, "relevance")
        return _parse_relevance_response(text)
    except Exception:
        log.exception("Relevance check failed for %s", case_name)
        return {
            "status": "unknown",
            "summary": f"{source_label} found: {case_name}. Local relevance check failed.",
            "confidence_note": "Local LLM relevance check encountered an error — verify manually.",
        }


def _parse_relevance_response(text: str) -> Dict[str, str]:
    """Parse the structured LLM response into status/summary/note.

    Two-axis evaluation:
        topically_relevant + accurately_described → found_relevant
        topically_relevant + NOT accurately_described → found_miscited
        NOT topically_relevant (either accuracy) → found_unrelated
    """
    topically_relevant = True   # default to benefit of the doubt
    accurately_described = True
    summary = ""
    note = ""

    for line in text.splitlines():
        line = _clean_format_line(line)
        upper = line.upper()
        if upper.startswith("TOPICALLY_RELEVANT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val == "NO":
                topically_relevant = False
        elif upper.startswith("ACCURATELY_DESCRIBED:"):
            val = line.split(":", 1)[1].strip().upper()
            # Must match exactly "NO" — not "CANNOT_DETERMINE" which
            # contains "NO" as a substring in "CANNOT"
            if val == "NO":
                accurately_described = False
            # CANNOT_DETERMINE → treat as accurate (benefit of the doubt)
        # Legacy format support: if the LLM falls back to old format
        elif upper.startswith("RELEVANT:"):
            val = line.split(":", 1)[1].strip().upper()
            if val == "NO":
                topically_relevant = False
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif upper.startswith("NOTE:"):
            note = line.split(":", 1)[1].strip()

    # Determine status from the two axes
    if not topically_relevant:
        status = "found_unrelated"
    elif not accurately_described:
        status = "found_miscited"
    else:
        status = "found_relevant"

    return {
        "status": status,
        "summary": summary or "Case found.",
        "confidence_note": note if note and note.upper() != "N/A" else "",
    }


# ---------------------------------------------------------------------------
# Subsequent-history (lightweight Shepardization)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Backup caption search for subsequent history
# ---------------------------------------------------------------------------

def _extract_caption_search_terms(case_name: str) -> str:
    """Extract distinctive party-name keywords from a case caption.

    Returns a CL-compatible query fragment like ``"Safadi" "McColley"``.
    Strips common legal titles, org suffixes, and noise words so the search
    matches appellate decisions that may use slightly different caption forms.
    """
    if not case_name:
        return ""

    # Split on " v. " / " vs. " / " v "
    parts = re.split(r'\s+v\.?\s+', case_name, maxsplit=1, flags=re.IGNORECASE)

    _SUFFIX_RE = re.compile(
        r',?\s*(?:Inc\.?|LLC|L\.?L\.?C\.?|Corp\.?|Co\.?|Ltd\.?|'
        r'et\s+al\.?|&\s*Assocs?\.?|&\s*Associates?|'
        r'Stores?|Company|Corporation)\s*$',
        re.IGNORECASE,
    )
    _PREFIX_RE = re.compile(
        r'^(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?|The)\s+',
        re.IGNORECASE,
    )

    terms: list[str] = []
    for part in parts:
        cleaned = part.strip()
        # Loop suffix stripping — handles stacked suffixes like ", Inc." + "& Assocs."
        while True:
            new = _SUFFIX_RE.sub('', cleaned).strip()
            if new == cleaned:
                break
            cleaned = new
        # Strip title prefixes
        cleaned = _PREFIX_RE.sub('', cleaned).strip()
        if cleaned and len(cleaned) > 1:
            terms.append(f'"{cleaned}"')

    return " ".join(terms)


def _backup_caption_search(
    case_name: str,
    date_filed: str,
    cited_court_id: str,
    headers: dict,
    max_results: int = 10,
) -> list[dict]:
    """Search CL for appellate decisions with a similar case caption filed
    after *date_filed*, filtering for courts above *cited_court_id*.

    This catches situations where CourtListener has the appellate decision
    indexed but has not linked it via ``cites:()``.
    """
    from firstread.court_hierarchy import is_higher_court

    caption_q = _extract_caption_search_terms(case_name)
    if not caption_q:
        return []

    query = f"caseName:({caption_q})"
    if date_filed:
        query += f" filed_after:{date_filed}"

    params = {
        "q": query,
        "type": "o",
        "order_by": "dateFiled asc",
        "format": "json",
    }
    try:
        resp = _cl_get(CL_API_V4, params=params, headers=headers, timeout=20)
        if not resp.ok:
            log.warning("Backup caption search failed %s: %s",
                        resp.status_code, query)
            return []
        data = resp.json()
    except Exception:
        log.exception("Backup caption search error: %s", query)
        return []

    higher_hits: list[dict] = []
    for hit in data.get("results", [])[:max_results]:
        hit_court = str(hit.get("court_id", "") or hit.get("court", ""))
        if is_higher_court(hit_court, cited_court_id):
            higher_hits.append(hit)

    return higher_hits


# Disposition severity — higher = worse for the cited case
_DISPOSITION_SEVERITY = {
    "discussed": 0,
    "affirmed": 1,
    "distinguished": 2,
    "remanded": 3,
    "vacated": 4,
    "reversed": 5,
    "other": 1,
}


def check_subsequent_history(
    cluster_id: str,
    cited_court_id: str,
    case_name: str,
    *,
    date_filed: str = "",
    max_citing: int = 20,
    max_llm_checks: int = 5,
    rate_limit: float = 1.0,
    llm: Optional[LLMFn] = None,
) -> Dict[str, Any]:
    """Look up forward citations and check for higher-court dispositions.

    ``llm`` (see ``LLMFn``) routes the per-citing-case disposition prompt
    through another model instead of Anthropic; CourtListener traffic is
    unchanged either way.

    When the ``cites:()`` search yields no higher-court hits (common for
    trial-court decisions), a backup caption-based search is attempted:
    look for appellate decisions with the same party names, filed after
    *date_filed*, in a higher court within the same court system.

    Returns dict with:
        status          – none_found | good_standing | caution | reversed |
                          vacated | error | skipped
        summary         – human-readable one-liner
        citing_cases    – list of dicts for CitationSubsequentHistory rows
    """
    from firstread.court_hierarchy import is_higher_court, should_skip_history

    empty = {"status": "none_found", "summary": "", "citing_cases": []}

    if should_skip_history(cited_court_id):
        return {"status": "skipped",
                "summary": "History check skipped (top-level court).",
                "citing_cases": []}

    # --- Query CL for opinions that cite this cluster ---
    headers = _cl_headers()
    try:
        params = {
            "q": f"cites:({cluster_id})",
            "type": "o",
            "order_by": "dateFiled desc",
            "format": "json",
        }
        resp = _cl_get(CL_API_V4, params=params, headers=headers, timeout=20)
        if not resp.ok:
            log.warning("CL cites search failed %s for cluster %s",
                        resp.status_code, cluster_id)
            return {**empty, "status": "error",
                    "summary": "CourtListener cited-by search failed."}
        data = resp.json()
    except Exception:
        log.exception("CL cites search error for cluster %s", cluster_id)
        return {**empty, "status": "error",
                "summary": "Error during cited-by search."}

    results = data.get("results", [])

    # --- Filter to higher-court opinions ---
    higher_court_hits = []
    for hit in results[:max_citing]:
        citing_court = hit.get("court_id", "") or hit.get("court", "")
        if is_higher_court(str(citing_court), cited_court_id):
            higher_court_hits.append(hit)

    # --- Backup: caption-based search for appellate decisions ---
    # Trial court decisions often lack cited-by data in CL.  Search for
    # higher-court cases with the same party names filed after this case.
    if not higher_court_hits and case_name:
        backup_hits = _backup_caption_search(
            case_name, date_filed, cited_court_id, headers,
        )
        if backup_hits:
            log.info("Backup caption search found %d higher-court hit(s) "
                     "for %s", len(backup_hits), case_name)
            higher_court_hits = backup_hits

    if not higher_court_hits:
        return empty

    # --- Analyse disposition for each higher-court opinion ---
    citing_cases: list[dict] = []
    worst_disposition = "affirmed"
    worst_severity = 0

    for i, hit in enumerate(higher_court_hits[:max_llm_checks]):
        if i > 0:
            time.sleep(rate_limit)

        citing_cid = hit.get("cluster_id") or _extract_cluster_id(hit)
        citing_oid = hit.get("id") or hit.get("opinion_id")
        citing_name = hit.get("caseName", "") or hit.get("case_name", "")
        citing_court_display = hit.get("court", "") or hit.get("court_citation_string", "")
        citing_court_slug = str(hit.get("court_id", "") or hit.get("court", ""))
        citing_date = hit.get("dateFiled", "") or hit.get("date_filed", "")

        cl_url = _build_cl_opinion_url(hit, citing_cid)

        # Fetch opinion text
        snippet = ""
        if citing_oid:
            snippet = _fetch_opinion_text(citing_oid, headers)
        if not snippet and citing_cid:
            snippet = _fetch_text_via_cluster(str(citing_cid), headers)

        # LLM disposition analysis (or fallback if no text)
        if snippet:
            disp = _analyze_disposition(case_name, citing_name, snippet, llm=llm)
        else:
            disp = {"disposition": "discussed",
                    "note": "No opinion text available for disposition analysis."}

        disposition = disp["disposition"]
        severity = _DISPOSITION_SEVERITY.get(disposition, 1)
        if severity > worst_severity:
            worst_severity = severity
            worst_disposition = disposition

        citing_cases.append({
            "citing_cluster_id": str(citing_cid) if citing_cid else "",
            "citing_case_name": citing_name,
            "citing_court": citing_court_display,
            "citing_court_id": citing_court_slug,
            "citing_date_filed": citing_date,
            "disposition": disposition,
            "disposition_note": disp.get("note", ""),
            "courtlistener_url": cl_url,
        })

    # --- Determine overall status ---
    if worst_disposition in ("reversed",):
        status = "reversed"
    elif worst_disposition in ("vacated",):
        status = "vacated"
    elif worst_disposition in ("remanded", "distinguished"):
        status = "caution"
    else:
        status = "good_standing"

    # Build summary from the most severe citing case
    worst_case = max(citing_cases,
                     key=lambda c: _DISPOSITION_SEVERITY.get(c["disposition"], 0))
    year = worst_case["citing_date_filed"][:4] if worst_case["citing_date_filed"] else ""
    summary_parts = [worst_case["disposition"].title()]
    if worst_case["citing_court"]:
        summary_parts.append(f"by {worst_case['citing_court']}")
    if worst_case["citing_case_name"]:
        summary_parts.append(f"in {worst_case['citing_case_name']}")
    if year:
        summary_parts.append(f"({year})")
    summary = " ".join(summary_parts)

    return {"status": status, "summary": summary, "citing_cases": citing_cases}


def _analyze_disposition(
    cited_case_name: str,
    citing_case_name: str,
    citing_opinion_text: str,
    llm: Optional[LLMFn] = None,
) -> Dict[str, str]:
    """Ask the injected ``llm`` how a higher court
    treated the cited case.

    Returns {"disposition": "...", "note": "..."}.
    """
    if llm is None:
        raise ValueError("_analyze_disposition needs an llm callable (see LocalJudge)")

    # Head-truncating the citing opinion routinely misses the passage that
    # actually discusses the cited case (observed: every disposition note
    # read "the excerpt does not contain the citation"). Pull the head, the
    # tail, and the passages that mention the cited parties instead.
    excerpt = _build_relevance_excerpt(citing_opinion_text, cited_case_name,
                                       budget=3000)
    prompt = f"""The case "{cited_case_name}" was cited by a higher court in
"{citing_case_name}". Below is an excerpt from the higher court's opinion;
sections have been selected from its beginning, its end, and passages that
mention the cited case, with "[...]" marking omitted text.

Determine how the higher court treated the lower court's decision in
"{cited_case_name}".

OPINION EXCERPT:
{excerpt}

Classify the treatment as exactly ONE of:
- AFFIRMED: The higher court upheld the lower court's decision
- REVERSED: The higher court overturned the lower court's decision
- VACATED: The higher court voided the lower court's decision
- REMANDED: The higher court sent the case back for further proceedings
- DISTINGUISHED: The higher court noted differences making the cited case inapplicable
- DISCUSSED: The higher court merely discussed or cited the case without acting on it
- OTHER: None of the above

Respond in this exact format (two lines only):
DISPOSITION: [one word from above]
NOTE: [one sentence explaining the treatment]"""

    try:
        text = llm(prompt, "disposition")
        return _parse_disposition_response(text)
    except Exception:
        log.exception("Disposition analysis failed for %s → %s",
                      cited_case_name, citing_case_name)
        return {"disposition": "other",
                "note": "Disposition analysis encountered an error."}


def _parse_disposition_response(text: str) -> Dict[str, str]:
    """Parse the structured LLM disposition response."""
    disposition = "other"
    note = ""

    valid = {"affirmed", "reversed", "vacated", "remanded",
             "distinguished", "discussed", "other"}

    for line in text.splitlines():
        line = _clean_format_line(line)
        upper = line.upper()
        if upper.startswith("DISPOSITION:"):
            val = line.split(":", 1)[1].strip().lower()
            # Handle multi-word values like "reversed and remanded"
            if "reverse" in val:
                disposition = "reversed"
            elif "vacate" in val:
                disposition = "vacated"
            elif "remand" in val:
                disposition = "remanded"
            elif "affirm" in val:
                disposition = "affirmed"
            elif "distinguish" in val:
                disposition = "distinguished"
            elif "discuss" in val:
                disposition = "discussed"
            elif val in valid:
                disposition = val
        elif upper.startswith("NOTE:"):
            note = line.split(":", 1)[1].strip()

    return {"disposition": disposition, "note": note}


# ---------------------------------------------------------------------------
# Topic-based citing opinion discovery
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Secondary-source (academic) discovery by case
# ---------------------------------------------------------------------------


