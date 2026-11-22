"""Academic-source citations in a paper (Story 1028): reference lists,
footnote citations and bare identifiers, parsed into fields and linked to the
sentence that cites them. Local and pure: no network, no database.

Used by utils/local_eval.check_citations on already-scrubbed text, beside
eyecite's case / statute extraction. Offsets are into
``citation_verifier._normalize_legal_text(text)``, the same text eyecite
sees, so an eyecite row inside an academic row's span can be dropped (a
law-review cite eyecite reads as a "case").

Three detectors (spec §1.2):

* the reference list under a References / Works Cited / Bibliography heading
  (APA 7, MLA 9, Chicago in both forms, numbered lists);
* full citations in the ``Notes`` section ``_docx_paragraphs(include_notes=True)``
  appends, and Bluebook journal citations anywhere; ``Id.`` / ``supra``
  resolve to the earlier note;
* bare DOIs, doi.org URLs, arXiv, SSRN and PubMed ids.

Each source gets a ``source_kind`` that decides whether it is looked up and
what "not found" means (utils/academic_lookup). Personal communications and
course materials are never looked up.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, Optional

MAX_ENTRIES = 120
MAX_ENTRY_CHARS = 600
RAW_KEEP = 400

# Kinds the indexes should carry: "not found" is a problem only for these.
INDEXABLE_KINDS = {"journal_article", "law_review", "conference_paper", "preprint"}
# Kinds never sent anywhere: not indexed, and personal ones name private people.
# "legal": a case, statute or rule listed among the references. The case /
# statute check owns those (CourtListener, Cornell LII); an academic index
# is never asked about one.
NEVER_LOOKUP_KINDS = {"web", "news", "personal_communication", "course_material", "legal"}

# pii_gate output: a pseudonym ("Student A12") or a "[… redacted]" marker.
REDACTION_RE = re.compile(r"\bStudent [A-Z]\d{2}\b|\[[a-z ]{2,30} redacted\]")

_HEADING_RE = re.compile(
    r"^[#*_\s]*(references|reference list|works cited|works consulted|"
    r"bibliography|sources|sources cited|cited sources|literature cited)"
    r"[*_\s]*:?\s*$", re.I | re.M)
_SECTION_END_RE = re.compile(r"^\s*(?:(?:appendix|exhibit|attachment)\b.*|Notes)\s*$",
                             re.I | re.M)
_NOTES_HEADING_RE = re.compile(r"^Notes\s*$", re.M)
_NOTE_LINE_RE = re.compile(r"^\[\^(e?\d+)\]\s+(.+)$", re.M)
_NOTE_MARKER_RE = re.compile(r"\[\^(e?\d+)\]")
_NUMBER_PREFIX_RE = re.compile(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})\.)\s+")
_ENTRY_START_RE = re.compile(
    r"^(?:\[\d{1,3}\]|\d{1,3}\.)\s|^[A-Z][\w'\-]+(?: [A-Z][\w'\-]+)?, (?:[A-Z]\.|[A-Z][a-z]+)")
# Inside one long paragraph (PDF text without blank lines): a new entry
# starts after a full stop with "Surname, I." / "Surname, First" / "[n]".
_INLINE_SPLIT_RE = re.compile(
    r"(?<=[.\]])\s+(?=(?:\[\d{1,3}\]\s*)?[A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)?, "
    r"(?:[A-Z]\.|[A-Z][a-z]+[,.]))")

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")
_ARXIV_RE = re.compile(r"(?:\barXiv:\s?|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_SSRN_RE = re.compile(r"ssrn\.com/(?:abstract=|sol3/papers\.cfm\?abstract_id=)(\d{4,9})", re.I)
_PMID_RE = re.compile(r"\bPMID:?\s?(\d{6,9})\b")
_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")

_SIGNAL_RE = re.compile(
    r"^(?:see(?: also| generally)?|cf\.|but see|but cf\.|accord|compare|"
    r"e\.g\.|see, e\.g\.)[,]?\s+", re.I)
_CASE_OR_STATUTE_RE = re.compile(r"\sv\.?\s|§|\bU\.S\.C\b|\bC\.F\.R\b|\bPub\. L\.")

# A reference entry that is a legal authority, not a publication.
_LEGAL_ENTRY_RE = re.compile(
    r"\sv\.?\s|§|\bU\.\s?S\.\s?C\b|\bC\.\s?F\.\s?R\b|\bPub\. L\.|\bStat\.\s\d|"
    r"\bConst\.|\bFed\. R\. |\bRestatement\b|"
    r"^(?:In re|Ex parte|In the Matter of|Matter of|Estate of)\b")
_CASE_TITLE_RE = re.compile(r"\sv\.?(?:\s|$)")
_COURT_RE = re.compile(
    r"\bSupreme Court\b|\bCourt of (?:Special )?Appeals\b|\b(?:Circuit|District) Court\b|"
    r"\bUnited States Reports\b|\bAppellate Court\b", re.I)

# Bluebook journal article: Author, Title, 151 U. Pa. L. Rev. 1003, 1010 (2003)
_BLUEBOOK_ARTICLE_RE = re.compile(
    r"(?P<author>[A-Z][A-Za-z.'\- ]{1,60}?(?:(?:,| &| and) [A-Z][A-Za-z.'\- ]{1,60}?)*), "
    r"(?P<title>[A-Z][^,()]{4,200}?), "
    r"(?P<volume>\d{1,4}) (?P<journal>[A-Z][A-Za-z.&'\- ]{1,80}?) (?P<page>\d{1,5})"
    r"(?:, \d{1,5}(?:[-–]\d{1,5})?)?(?: n\.\s?\d+)? \((?P<year>\d{4})\)")
# Bluebook book: Author, Title 45 (2d ed. 2010)
_BLUEBOOK_BOOK_RE = re.compile(
    r"^(?P<author>[A-Z][A-Za-z.'\- ]{1,60}?(?:(?:,| &| and) [A-Z][A-Za-z.'\- ]{1,60}?)*), "
    r"(?P<title>[A-Z][^,()]{4,200}?)(?: \d{1,5}(?:[-–]\d{1,5})?)? "
    r"\((?:[^()]*?ed\.,? )?(?P<year>\d{4})\)")
# Chicago note, article: Author, "Title," Journal 12, no. 3 (2021): 45-67.
_CHICAGO_NOTE_ARTICLE_RE = re.compile(
    r"^(?P<author>[^,\"]{2,80}(?:, (?:and )?[^,\"]{2,80})*), \"(?P<title>[^\"]{4,250}?),?\" "
    r"(?P<journal>[^0-9\"]{2,120}?) (?P<volume>\d{1,4})(?:, no\. (?P<issue>[\w-]+))? "
    r"\((?:[A-Za-z]+ )?(?P<year>\d{4})\)(?:: (?P<pages>[\d\-–]+))?")
# Chicago note, book: Author, Title (Place: Publisher, Year), page.
_CHICAGO_NOTE_BOOK_RE = re.compile(
    r"^(?P<author>[^,\"]{2,80}(?:, (?:and )?[^,\"]{2,80})*), (?P<title>[A-Z][^(\"]{4,250}?) "
    r"\((?:[^:()]{2,60}: )?(?P<publisher>[^,()]{2,80}), (?P<year>\d{4})\)")
_SHORT_ID_RE = re.compile(r"^(?:Id\.|Ibid\.?)", re.I)
_SUPRA_RE = re.compile(r"supra note (\d+)", re.I)

# APA: Authors. (2021). Title. Venue, 12(3), 45-67.
_APA_RE = re.compile(
    r"^(?P<authors>.+?)\s*\((?P<year>\d{4}[a-z]?|n\.\s?d\.|in press)(?:,[^)]*)?\)\.\s*(?P<rest>.+)$",
    re.I)
# Chicago author-date: Surname, First. 2021. "Title." Journal 12 (3): 45-67.
_CHICAGO_AD_RE = re.compile(
    r"^(?P<authors>[^\"]+?)\.\s+(?P<year>\d{4})[a-z]?\.\s+(?P<rest>.+)$")
# MLA / Chicago bibliography: Authors. "Title." Rest
_QUOTED_TITLE_RE = re.compile(r"^(?P<authors>[^\"]+?)\.\s*\"(?P<title>[^\"]{3,300}?)[.,?!]?\"\s*(?P<rest>.*)$")
_VOL_ISSUE_PAGES_RE = re.compile(
    r"(?P<volume>\d{1,4})\s*(?:\((?P<issue>[^)]{1,12})\))?\s*[,:]\s*(?P<pages>e?\d+(?:\s?[-–]\s?\d+)?)")
_MLA_VOL_RE = re.compile(r"\bvol\.\s*(\d+)", re.I)
_MLA_NO_RE = re.compile(r"\bno\.\s*([\w-]+)", re.I)
_PAGES_RE = re.compile(r"\bpp?\.\s*(\d+(?:\s?[-–]\s?\d+)?)")
_COLON_PAGES_RE = re.compile(r":\s*(\d+(?:\s?[-–]\s?\d+)?)")
_CHICAGO_VOL_RE = re.compile(r"\s(\d{1,4})(?:,\s*no\.\s*([\w-]+))?\s*\((?:[A-Za-z]+\s)?\d{4}\)")
_EDITION_RE = re.compile(r"\((?:\d+\w{1,2}|rev\.|[A-Z][a-z]+)\s+ed\.\)|\b\d+\w{1,2} ed\.", re.I)
_CHAPTER_RE = re.compile(r"\bIn\s+(?:[A-Z][^()]*?\((?:Eds?|Ed)\.\),\s*)?(?P<book>[^()]+?)\s*\(pp?\.", re.I)

_KIND_PATTERNS = [
    ("personal_communication", re.compile(
        r"personal communication|\binterview(?:ed)?\b|\be-?mail\b|\btext message\b", re.I)),
    ("course_material", re.compile(
        r"\blecture\b|class notes|course (?:notes|materials?|packet)|syllabus|"
        r"brightspace|\bd2l\b|canvas|blackboard|moodle|powerpoint slides", re.I)),
    ("preprint", re.compile(r"\barxiv\b|\bssrn\b|preprint|working paper|biorxiv|medrxiv", re.I)),
    ("conference_paper", re.compile(r"\bproceedings\b|\bconference\b|\bsymposium\b|\bworkshop\b", re.I)),
    ("law_review", re.compile(r"\bL\.\s?Rev\.|\bL\.\s?J\.|law review|law journal", re.I)),
    ("news", re.compile(
        r"new york times|washington post|wall street journal|\bnytimes\b|"
        r"\bthe guardian\b|\breuters\b|associated press|\bnpr\b|\bcnn\b|\bbbc\b|"
        r"\bnewsweek\b|\btime magazine\b|\bforbes\b|\bbloomberg\b|\bthe atlantic\b|"
        r"\bnewspaper\b|\bbaltimore sun\b|usa today", re.I)),
    ("report", re.compile(
        r"\b(?:report|white paper|technical paper|fact sheet|policy brief)\b|"
        r"\b(?:department|bureau|commission|agency|office) of\b|"
        r"\bU\.S\. (?:Department|Government)|\bGAO\b|\bCongressional Research Service\b", re.I)),
]
_PERSONAL_URL_RE = re.compile(
    r"docs\.google\.|drive\.google\.|onedrive\.|1drv\.ms|sharepoint\.com|dropbox\.com|"
    r"icloud\.com|\bbox\.com|notion\.so|evernote\.com", re.I)
_JOURNAL_VENUE_RE = re.compile(
    r"journal|review|quarterly|annals|bulletin|studies|psycholog|research|"
    r"science|letters|proceedings|transactions|\bJ\.", re.I)


# ── small helpers ────────────────────────────────────────────────────────────

def fold(s: str) -> str:
    """ASCII-fold and lowercase (Müller → muller)."""
    return (unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore")
            .decode("ascii").lower().strip())


def norm_title(s: str) -> str:
    """The comparison form of a title: folded, & → and, punctuation and a
    leading article dropped, whitespace collapsed."""
    t = fold(s).replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"^(?:the|a|an) ", "", t)


def appears_in(title: str, raw: str, threshold: float = 0.9) -> bool:
    """Whether ``title`` is (almost) a substring of ``raw`` — the check that
    a model-parsed title was read off the entry, not invented."""
    nt, nr = norm_title(title), norm_title(raw)
    if not nt:
        return False
    if nt in nr:
        return True
    blocks = SequenceMatcher(None, nt, nr, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(nt) >= threshold


def normalize_doi(doi: str) -> str:
    """Lower-case DOI without a resolver prefix, trailing punctuation or an
    unbalanced closing parenthesis."""
    d = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", (doi or "").strip(), flags=re.I)
    d = re.sub(r"^doi:\s*", "", d, flags=re.I)
    d = d.rstrip(".,;:")
    while d.endswith(")") and d.count(")") > d.count("("):
        d = d[:-1].rstrip(".,;:")
    return d.lower()


def _clean_year(y) -> Optional[str]:
    m = _YEAR_RE.search(str(y or ""))
    return m.group(1) if m else None


_PARTICLES = {"van", "von", "de", "der", "den", "del", "della", "la", "le", "da", "di", "du", "st"}


def _surname_of_first_last(name: str) -> str:
    """'Cass R. Sunstein' → 'sunstein'; 'Ludwig van Beethoven' → 'van beethoven'."""
    toks = [t for t in re.split(r"\s+", name.strip().strip(".,")) if t]
    toks = [t for t in toks if not re.fullmatch(r"(?:[A-Z]\.)+|Jr\.?|Sr\.?|II|III|IV", t)]
    if not toks:
        return ""
    sur = [toks[-1]]
    i = len(toks) - 2
    while i >= 0 and toks[i].lower() in _PARTICLES:
        sur.insert(0, toks[i])
        i -= 1
    return fold(" ".join(sur))


def _surnames_apa(authors: str) -> list[str]:
    """'Smith, J. A., Lee, K., & García, M.' → ['smith', 'lee', 'garcia']."""
    out = []
    for m in re.finditer(r"((?:(?:van|von|de|der|del|la|le|da|di|du) )?[A-Z][\w'\-]+(?: [A-Z][\w'\-]+)?),\s*(?:[A-Z]\.|[A-Z]-[A-Z]\.)",
                         authors):
        out.append(fold(m.group(1)))
    return out


def _surnames_mla(authors: str) -> list[str]:
    """MLA / Chicago bibliography: first author surname-first, the rest
    first-last. 'Smith, Jane, and John Lee' → ['smith', 'lee']."""
    a = re.sub(r",?\s*et al\.?$", "", authors.strip().rstrip("."))
    parts = re.split(r",\s*and\s+|\s+and\s+|,\s*(?=[A-Z][a-z]+ [A-Z])", a)
    out = []
    if parts:
        first = parts[0]
        out.append(fold(first.split(",")[0]) if "," in first
                   else _surname_of_first_last(first))
        out += [_surname_of_first_last(p) for p in parts[1:]]
    return [s for s in out if s]


def _surnames_first_last(authors: str) -> list[str]:
    a = re.sub(r",?\s*et al\.?$", "", authors.strip().rstrip("."))
    parts = re.split(r",\s*(?:and|&)\s+|\s+(?:and|&)\s+|,\s+", a)
    return [s for s in (_surname_of_first_last(p) for p in parts) if s]


def _split_title_rest(rest: str) -> tuple[str, str]:
    """APA's 'Title. Venue, 12(3), 45.' → ('Title', 'Venue, 12(3), 45.').
    The title ends at the first '. ', '? ' or '! ' not right after a lone
    capital (so 'U.S.' and initials don't end it)."""
    for m in re.finditer(r"[.?!](?=\s+|$)", rest):
        i = m.start()
        before = rest[max(0, i - 2):i]
        if rest[i] == "." and re.fullmatch(r"(?:^|[\s.])[A-Z]", before.rjust(2)):
            continue
        title = rest[:i + (1 if rest[i] in "?!" else 0)].strip()
        return title, rest[m.end():].strip()
    return rest.strip().rstrip("."), ""


def _venue_fields(rest: str, fields: dict):
    """Venue, volume, issue, pages from the part after the title."""
    rest = _URL_RE.sub("", rest).strip()
    rest = re.sub(r"\bdoi:\s*\S+", "", rest, flags=re.I)
    rest = _PMID_RE.sub("", _ARXIV_RE.sub("", rest)).strip(" .")
    if not rest:
        return
    pages = _PAGES_RE.search(rest) or _COLON_PAGES_RE.search(rest)
    vol, no = _MLA_VOL_RE.search(rest), _MLA_NO_RE.search(rest)
    cm = _CHICAGO_VOL_RE.search(rest)
    m = _VOL_ISSUE_PAGES_RE.search(rest)
    if vol:                                   # MLA: Journal, vol. 12, no. 3, 2021, pp.
        fields["venue"] = rest.split(",")[0].strip(" .")
        fields["volume"] = vol.group(1)
        fields["issue"] = no.group(1) if no else ""
    elif cm:                                  # Chicago: Journal 12, no. 3 (2021): 45
        fields["venue"] = rest[:cm.start()].strip(" ,.")
        fields["volume"], fields["issue"] = cm.group(1), cm.group(2) or ""
    elif m:                                   # APA 12(3), 45  ·  Chicago a-d 12 (3): 45
        fields["venue"] = rest[:m.start()].strip(" ,.")
        fields.update(volume=m.group("volume"), issue=m.group("issue") or "",
                      pages=re.sub(r"\s", "", m.group("pages")))
        return
    else:
        first = re.split(r"\.\s", rest)[0].strip(" .")
        if _EDITION_RE.search(rest) or not _JOURNAL_VENUE_RE.search(first):
            fields["publisher"] = first
        else:
            fields["venue"] = first.split(",")[0].strip()
    if pages:
        fields["pages"] = re.sub(r"\s", "", pages.group(1))


def _empty_fields(raw: str) -> dict:
    return {"raw": raw[:RAW_KEEP], "title": "", "author_surnames": [], "year": None,
            "venue": "", "volume": "", "issue": "", "pages": "", "publisher": "",
            "doi": "", "arxiv": "", "pmid": "", "url": "", "style": "unknown",
            "source_kind": "other", "parsed": False}


def _identifiers(raw: str, fields: dict):
    m = _DOI_RE.search(raw)
    if m:
        fields["doi"] = normalize_doi(m.group(1))
    s = _SSRN_RE.search(raw)
    if s and not fields["doi"]:
        fields["doi"] = f"10.2139/ssrn.{s.group(1)}"
    a = _ARXIV_RE.search(raw)
    if a:
        fields["arxiv"] = a.group(1)
    p = _PMID_RE.search(raw)
    if p:
        fields["pmid"] = p.group(1)
    for u in _URL_RE.findall(raw):
        u = u.rstrip(".,;)")
        if not re.search(r"doi\.org/|arxiv\.org/|ssrn\.com/|ncbi\.nlm\.nih\.gov/pubmed", u, re.I):
            fields["url"] = u
            break


def classify(fields: dict) -> str:
    """The source kind (spec §1.3) from the parsed fields and the raw entry."""
    raw = fields.get("raw", "")
    for kind, rx in _KIND_PATTERNS[:2]:          # personal / course material first
        if rx.search(raw):
            return kind
    if fields.get("url") and _PERSONAL_URL_RE.search(fields["url"]):
        return "web"                             # a student's own doc: never looked up
    if fields.get("arxiv") or (fields.get("doi") or "").startswith("10.2139/ssrn"):
        return "preprint"
    if fields.get("style") == "bluebook" and fields.get("volume"):
        return "law_review" if _KIND_PATTERNS[4][1].search(raw) else "journal_article"
    for kind, rx in _KIND_PATTERNS[2:]:
        if rx.search(raw):
            return kind
    if _CHAPTER_RE.search(raw):
        return "book_chapter"
    if fields.get("volume") and (fields.get("pages") or fields.get("venue")):
        return "journal_article"
    if fields.get("venue") and _JOURNAL_VENUE_RE.search(fields["venue"]):
        return "journal_article"
    if fields.get("doi"):
        return "journal_article"
    if fields.get("url") and not fields.get("venue"):
        return "web"
    if fields.get("publisher") or _EDITION_RE.search(raw):
        return "book"
    return "other"


# ── parsing one entry ────────────────────────────────────────────────────────

def parse_entry(raw: str) -> dict:
    """Fields of one reference entry or full note citation (spec §1.3).
    ``parsed`` is False when no title could be read."""
    raw = re.sub(r"\s+", " ", raw or "").strip()
    fields = _empty_fields(raw)
    text = _NUMBER_PREFIX_RE.sub("", raw)
    text = _SIGNAL_RE.sub("", text)
    _identifiers(text, fields)
    body = _URL_RE.sub("", text).strip()
    body = re.sub(r"\s*\bdoi:\s*\S+", "", body, flags=re.I).strip()

    m = _BLUEBOOK_ARTICLE_RE.search(body)
    if m and not _CASE_OR_STATUTE_RE.search(m.group(0)):
        fields.update(style="bluebook", title=m.group("title").strip(),
                      author_surnames=_surnames_first_last(m.group("author")),
                      year=m.group("year"), venue=m.group("journal").strip(),
                      volume=m.group("volume"), pages=m.group("page"))
    elif (m := _CHICAGO_NOTE_ARTICLE_RE.match(body)):
        fields.update(style="chicago", title=m.group("title").strip(" ,"),
                      author_surnames=_surnames_first_last(m.group("author")),
                      year=m.group("year"), venue=m.group("journal").strip(" ,"),
                      volume=m.group("volume"), issue=m.group("issue") or "",
                      pages=(m.group("pages") or "").replace("–", "-"))
    elif (m := _APA_RE.match(body)):
        title, rest = _split_title_rest(m.group("rest"))
        fields.update(style="apa", title=title, year=_clean_year(m.group("year")),
                      author_surnames=_surnames_apa(m.group("authors")))
        chap = _CHAPTER_RE.search(rest)
        if chap:
            fields["venue"] = chap.group("book").strip(" ,.")
            tail = rest[chap.end():]
            pm = re.search(r"(\d+\s?[-–]\s?\d+)\)\.\s*([^.]+)", tail)
            if pm:
                fields["pages"] = re.sub(r"\s", "", pm.group(1))
                fields["publisher"] = pm.group(2).strip()
        else:
            _venue_fields(rest, fields)
    elif (m := _CHICAGO_AD_RE.match(body)) and m.group("rest").startswith('"'):
        q = re.match(r"\"(?P<title>[^\"]{3,300}?)[.,?!]?\"\s*(?P<rest>.*)$", m.group("rest"))
        if q:
            fields.update(style="chicago", title=q.group("title"), year=m.group("year"),
                          author_surnames=_surnames_mla(m.group("authors")))
            _venue_fields(q.group("rest"), fields)
    elif (m := _QUOTED_TITLE_RE.match(body)):
        rest = m.group("rest")
        style = "chicago" if re.search(r"\(\d{4}\)", rest) else "mla"
        fields.update(style=style, title=m.group("title").strip(),
                      author_surnames=_surnames_mla(m.group("authors")),
                      year=_clean_year(_YEAR_RE.findall(rest)[-1] if _YEAR_RE.findall(rest) else None))
        _venue_fields(rest, fields)
    elif (m := _CHICAGO_NOTE_BOOK_RE.match(body)) and not _CASE_OR_STATUTE_RE.search(body):
        fields.update(style="chicago", title=m.group("title").strip(" ,"),
                      author_surnames=_surnames_first_last(m.group("author")),
                      year=m.group("year"), publisher=m.group("publisher").strip())
    elif (m := _BLUEBOOK_BOOK_RE.match(body)) and not _CASE_OR_STATUTE_RE.search(body):
        fields.update(style="bluebook", title=m.group("title").strip(),
                      author_surnames=_surnames_first_last(m.group("author")),
                      year=m.group("year"))
        if _EDITION_RE.search(body):
            fields["publisher"] = "(edition)"
    else:
        # MLA / Chicago book: Surname, First. Title. Publisher, 2021.
        bm = re.match(r"^(?P<authors>[A-Z][^.]{1,80}?)\.\s+(?P<title>[A-Z][^.]{3,250}?)\.\s+(?P<rest>.+)$", body)
        if bm and _YEAR_RE.search(bm.group("rest")) and "," in bm.group("authors"):
            fields.update(style="mla", title=bm.group("title").strip(),
                          author_surnames=_surnames_mla(bm.group("authors")),
                          year=_clean_year(_YEAR_RE.findall(bm.group("rest"))[-1]))
            rest = bm.group("rest")
            pub = re.match(r"^(?:[^:,.]{2,40}:\s*)?(?P<pub>[^,.]{2,80}),\s*\d{4}", rest)
            if pub:
                fields["publisher"] = pub.group("pub").strip()
            else:
                _venue_fields(rest, fields)
    title = fields["title"].strip().strip('"').strip()
    # trailing descriptors: "(7th ed.)", "[Lecture notes]", "(GAO-20-123)"
    while (d := re.search(r"\s*[(\[][^()\[\]]{1,40}[)\]]\s*$", title)) and d.start() > 3:
        title = title[:d.start()].rstrip(" .")
    fields["title"] = title
    fields["parsed"] = bool(fields["title"])
    fields["source_kind"] = classify(fields)
    return fields


# ── detectors ────────────────────────────────────────────────────────────────

def reference_section(text: str) -> Optional[tuple[int, int]]:
    """(start, end) of the reference list body (after its heading)."""
    heads = list(_HEADING_RE.finditer(text))
    if not heads:
        return None
    h = heads[-1]
    end_m = _SECTION_END_RE.search(text, h.end())
    end = end_m.start() if end_m else len(text)
    return (h.end(), end) if end > h.end() else None


def _split_entries(text: str, start: int, end: int) -> list[tuple[str, int, int]]:
    """(raw, start, end) per entry inside the reference list."""
    out = []
    pos = start
    for chunk in re.split(r"(\n\s*\n)", text[start:end]):
        cstart, pos = pos, pos + len(chunk)
        if not chunk.strip() or re.fullmatch(r"\n\s*\n", chunk):
            continue
        # a wrapped PDF list: lines that don't start an entry continue one
        lines = chunk.split("\n")
        pieces, cur, cur_start, lpos = [], "", cstart, cstart
        for ln in lines:
            if cur and _ENTRY_START_RE.match(ln.strip()):
                pieces.append((cur, cur_start))
                cur, cur_start = ln, lpos
            else:
                cur = f"{cur} {ln}" if cur else ln
                if not cur.strip():
                    cur_start = lpos
            lpos += len(ln) + 1
        if cur.strip():
            pieces.append((cur, cur_start))
        for piece, pstart in pieces:
            # one long paragraph holding several entries
            sub = _INLINE_SPLIT_RE.split(piece) if len(piece) > MAX_ENTRY_CHARS or \
                len(re.findall(r"\(\d{4}[a-z]?\)", piece)) > 1 else [piece]
            spos = pstart
            for s in sub:
                s_clean = s.strip()
                if s_clean:
                    idx = text.find(s_clean[:40], spos, end)
                    sstart = idx if idx >= 0 else spos
                    out.append((s_clean, sstart, sstart + len(s_clean)))
                    spos = sstart + len(s_clean)
    return out[:MAX_ENTRIES]


def _notes(text: str) -> list[tuple[str, str, int, int]]:
    """(note id, note text, start, end) for each ``[^n]`` line after Notes."""
    heads = list(_NOTES_HEADING_RE.finditer(text))
    if not heads:
        return []
    start = heads[-1].end()
    return [(m.group(1), m.group(2), m.start(), m.end())
            for m in _NOTE_LINE_RE.finditer(text, start)]


def _row(fields: dict, origin: str, start: int, end: int) -> dict:
    return dict(fields, origin=origin, start=start, end=end, in_text=[],
                proposition="", note_ids=[], number=None)


def source_key(r: dict) -> str:
    if r.get("doi"):
        return f"doi:{r['doi']}"
    if r.get("arxiv"):
        return f"arxiv:{r['arxiv']}"
    if r.get("pmid"):
        return f"pmid:{r['pmid']}"
    return f"title:{norm_title(r.get('title', ''))}|{r.get('year') or ''}"


# ── legal authorities among the sources ─────────────────────────────────────

def _legal_cites_in(src: dict, cites: list[dict]) -> list[dict]:
    """The eyecite case / statute citations this source row is about: inside
    its span, or (eyecite keeps one row per citation string, so a case cited
    in the body and listed again in the references has no second row) the
    same citation string in the entry."""
    raw = src.get("raw") or ""
    out = []
    for c in cites:
        ctext = re.sub(r"\s+", " ", c.get("citation_text") or "").strip()
        if _within(c, src) or (ctext and ctext in raw):
            out.append(c)
    return out


def _within(cite: dict, src: dict) -> bool:
    s, e = cite.get("start"), cite.get("end")
    if s is None or e is None:
        return False
    spans = src.get("spans") or [(src["start"], src["end"])]
    return any(a <= s and e <= b for a, b in spans)


def is_legal_authority(src: dict, cites_in: list[dict]) -> bool:
    """True when a source row is a case, statute or rule rather than a
    publication. ``cites_in`` is ``_legal_cites_in(src, cites)``. An article
    or book that names a case in its title stays academic."""
    if src.get("doi") or src.get("arxiv") or src.get("pmid"):
        return False
    if src.get("source_kind") in ("personal_communication", "course_material"):
        return False
    raw = src.get("raw") or ""
    if not src.get("parsed"):
        return bool(cites_in) or bool(_LEGAL_ENTRY_RE.search(_NUMBER_PREFIX_RE.sub("", raw)))
    published = bool(src.get("author_surnames")) and bool(
        src.get("venue") or src.get("volume") or src.get("publisher"))
    if cites_in and not published:
        return True                # "Miranda v. Arizona. (1966). 384 U.S. 436."
    # MLA: "United States, Supreme Court. Miranda v. Arizona. ... United States Reports, vol. 384"
    return bool(_CASE_TITLE_RE.search(src.get("title") or "") and _COURT_RE.search(raw)
                and not _JOURNAL_VENUE_RE.search(src.get("venue") or ""))


def _named_in_body(body: str, raw: str) -> str:
    """The body sentence that names a reference-list authority ("Brown v.
    Board of Education", "In re Gault"), or "" when none does."""
    name = re.split(r",\s*\d|\.\s*\(?\d|\s\(\d{4}", _NUMBER_PREFIX_RE.sub("", raw), maxsplit=1)[0]
    name = name.strip(" .,")
    if len(name) < 6:
        return ""
    m = re.search(re.escape(name), body, re.I)
    return _sentence(body, m.start(), m.end())[:600] if m else ""


def claim_legal(text: str, out: dict, cites: list[dict]) -> int:
    """Take the legal authorities out of ``extract_sources`` output so that
    no authority is checked twice, or against the wrong index. ``cites`` are
    ``citation_verifier.extract_citations`` rows for the same text.

    * A source the case / statute check already has a citation for is
      removed: its CourtListener / Cornell LII row is the one row for it.
    * A legal authority eyecite read no citation from ("Miranda v. Arizona
      (1966).") stays, as ``source_kind`` "legal": reported, never looked up.

    A citation that appears only inside a reference entry has no claim
    around it, so its ``proposition`` becomes the body sentence that names
    the authority (or "" — then only its existence is checked). Mutates
    ``out`` and ``cites``; returns how many sources were removed."""
    from firstread.citation_verifier import _normalize_legal_text
    text = _normalize_legal_text(text or "")
    sec = reference_section(text)
    body = text[:sec[0]] if sec else text
    kept, removed = [], []
    for src in out.get("sources") or []:
        inside = _legal_cites_in(src, cites)
        if not is_legal_authority(src, inside):
            kept.append(src)
            continue
        if not inside:
            src["source_kind"] = "legal"
            kept.append(src)
            continue
        removed.append(src)
        if src.get("origin") == "reference_list":
            prop = None
            for c in inside:
                if _within(c, src):
                    if prop is None:
                        prop = _named_in_body(body, src.get("raw") or "")
                    c["proposition"] = prop
    out["sources"] = kept
    form = out.get("citation_form") or {}
    if removed and form.get("uncited_entry"):
        gone = {(r.get("title") or (r.get("raw") or "")[:80]) for r in removed}
        form["uncited_entry"] = [u for u in form["uncited_entry"] if u not in gone]
    return len(removed)


# In-text forms (spec §1.4)
_NAME = r"[A-Z][\w'\-]+(?: [A-Z][\w'\-]+)?"
_AD_ITEM_RE = re.compile(
    rf"(?:see |e\.g\., |cf\. )?(?P<name>{_NAME})(?: (?:&|and) {_NAME}| et al\.)?,? "
    r"(?P<year>\d{4})[a-z]?(?:, (?:p|pp)\. ?[\d\-–]+)?$")
_NARRATIVE_RE = re.compile(rf"(?P<name>{_NAME})(?: (?:&|and) {_NAME}| et al\.)? \((?P<year>\d{{4}})[a-z]?\)")
_PAREN_RE = re.compile(r"\(([^()]{4,240})\)")
_MLA_ITEM_RE = re.compile(rf"^(?P<name>{_NAME})(?: and {_NAME}| et al\.)? (?P<page>\d+(?:[-–]\d+)?)$")
_NUMBERED_RE = re.compile(r"(?<!\^)\[(\d{1,3}(?:\s?[,–-]\s?\d{1,3})*)\]")


def _in_text_cites(text: str, exclude: list[tuple[int, int]]) -> list[dict]:
    """Every in-text citation outside the excluded spans (reference list,
    notes): ``{form, name, year, page, number, text, start, end}``."""
    def excluded(i):
        return any(a <= i < b for a, b in exclude)

    found = []
    for pm in _PAREN_RE.finditer(text):
        if excluded(pm.start()):
            continue
        for item in pm.group(1).split(";"):
            item = item.strip()
            ad = _AD_ITEM_RE.match(item)
            if ad:
                found.append({"form": "author_date", "name": fold(ad.group("name")),
                              "year": ad.group("year"), "text": f"({item})",
                              "start": pm.start(), "end": pm.end()})
                continue
            ml = _MLA_ITEM_RE.match(item)
            if ml:
                found.append({"form": "mla", "name": fold(ml.group("name")), "year": None,
                              "text": f"({item})", "start": pm.start(), "end": pm.end()})
    for nm in _NARRATIVE_RE.finditer(text):
        if not excluded(nm.start()):
            found.append({"form": "author_date", "name": fold(nm.group("name")),
                          "year": nm.group("year"), "text": nm.group(0),
                          "start": nm.start(), "end": nm.end()})
    for num in _NUMBERED_RE.finditer(text):
        if excluded(num.start()):
            continue
        for part in re.split(r"\s?,\s?", num.group(1)):
            rng = re.split(r"\s?[–-]\s?", part)
            lo = int(rng[0])
            hi = int(rng[-1]) if len(rng) > 1 else lo
            for n in range(lo, min(hi, lo + 20) + 1):
                found.append({"form": "numbered", "number": n, "name": "", "year": None,
                              "text": num.group(0), "start": num.start(), "end": num.end()})
    found.sort(key=lambda c: c["start"])
    return found


def _proposition(text: str, start: int, end: int) -> str:
    """The case-citation rule: the citing sentence plus up to two before it
    (a case cite ends the claim it supports). Used for inline Bluebook
    citations and bare identifiers."""
    from firstread.citation_verifier import _extract_proposition
    try:
        return _extract_proposition(text, start, end)
    except Exception:  # noqa: BLE001 — a proposition is a nicety, never a failure
        return ""


_SENTENCE_END_RE = re.compile(r"[.!?](?:\[\^e?\d+\])?(?=[\"')\]]*(?:\s|$))")


def _sentence(text: str, start: int, end: int) -> str:
    """The one sentence holding an in-text citation or a note marker: an
    author-date, MLA or numbered cite sits inside the claim it supports, so
    the sentences before it belong to other sources."""
    para_start = text.rfind("\n\n", 0, start)
    para_start = 0 if para_start < 0 else para_start + 2
    para_end = text.find("\n\n", end)
    para_end = len(text) if para_end < 0 else para_end
    tail = re.search(r"[.!?][\"')\]]*$", text[para_start:start])
    if tail:                               # "…has limits.[^1]": the marker closes it
        back_limit, stop = para_start + tail.start(), end
    else:
        back_limit = start
        m = _SENTENCE_END_RE.search(text, end, para_end)
        stop = m.end() if m else para_end
    begin = para_start
    for m in _SENTENCE_END_RE.finditer(text, para_start, back_limit):
        before = text[max(para_start, m.start() - 3):m.start()]
        if re.search(r"(?:\b[A-Z]|\bal|\bp|\bpp|\bv|\bed|\bvol|\bno)$", before):
            continue                       # "et al.", "p.", initials: not an end
        begin = m.end()
    return re.sub(r"\s+", " ", text[begin:stop]).strip()


def extract_sources(text: str) -> dict:
    """``{"sources": [...], "citation_form": {...}}`` for (scrubbed) ``text``.

    Each source: the parse_entry fields plus ``origin`` (reference_list |
    note | inline | identifier), ``start``/``end`` (offsets into the
    normalised text), ``in_text`` (the in-text citations that use it),
    ``proposition`` and ``note_ids``."""
    from firstread.citation_verifier import _normalize_legal_text
    text = _normalize_legal_text(text or "")
    rows: list[dict] = []
    exclude: list[tuple[int, int]] = []

    # (a) the reference list
    sec = reference_section(text)
    if sec:
        exclude.append(sec)
        for i, (raw, s, e) in enumerate(_split_entries(text, *sec), 1):
            if len(raw) > MAX_ENTRY_CHARS:
                continue
            row = _row(parse_entry(raw), "reference_list", s, e)
            num = _NUMBER_PREFIX_RE.match(raw)
            row["number"] = int(num.group(1) or num.group(2)) if num else i
            rows.append(row)

    # (b) notes: full citations, Id. / supra resolve to an earlier note
    notes = _notes(text)
    if notes:
        exclude.append((notes[0][2], len(text)))
    by_note: dict[str, dict] = {}
    prev = None
    for nid, ntext, s, e in notes:
        for part in re.split(r";\s+", ntext):
            part = part.strip()
            target = None
            if _SHORT_ID_RE.match(part):
                target = prev
            elif (sm := _SUPRA_RE.search(part)):
                target = by_note.get(sm.group(1))
            if target is not None:
                target["note_ids"].append(nid)
                by_note.setdefault(nid, target)
                continue
            if _CASE_OR_STATUTE_RE.search(part):
                continue
            fields = parse_entry(part)
            if not fields["parsed"] and not (fields["doi"] or fields["arxiv"]):
                continue
            row = _row(fields, "note", s, e)
            row["note_ids"].append(nid)
            rows.append(row)
            by_note.setdefault(nid, row)
            prev = row

    # Bluebook journal citations anywhere else (e.g. PDF footnotes)
    for m in _BLUEBOOK_ARTICLE_RE.finditer(text):
        if any(a <= m.start() < b for a, b in exclude) or _CASE_OR_STATUTE_RE.search(m.group(0)):
            continue
        rows.append(_row(parse_entry(m.group(0)), "inline", m.start(), m.end()))

    # (c) bare identifiers
    have = {source_key(r) for r in rows}
    for rx, kind in ((_DOI_RE, "doi"), (_SSRN_RE, "ssrn"), (_ARXIV_RE, "arxiv"), (_PMID_RE, "pmid")):
        for m in rx.finditer(text):
            fields = _empty_fields(m.group(0))
            if kind == "doi":
                fields["doi"] = normalize_doi(m.group(1))
            elif kind == "ssrn":
                fields["doi"] = f"10.2139/ssrn.{m.group(1)}"
            elif kind == "arxiv":
                fields["arxiv"] = m.group(1)
            else:
                fields["pmid"] = m.group(1)
            key = source_key(fields)
            inside = any(r["start"] <= m.start() < r["end"] for r in rows)
            if key in have or inside:
                continue
            fields["source_kind"] = "preprint" if kind in ("ssrn", "arxiv") else "journal_article"
            have.add(key)
            rows.append(_row(fields, "identifier", m.start(), m.end()))

    rows = _dedupe(rows)

    # in-text linkage → proposition
    cites = _in_text_cites(text, exclude)
    used: set[int] = set()
    for c in cites:
        idx = _match_in_text(c, rows)
        if idx is None:
            continue
        r = rows[idx]
        used.add(idx)
        if c["text"] not in r["in_text"]:
            r["in_text"].append(c["text"])
        if not r["proposition"]:
            r["proposition"] = _sentence(text, c["start"], c["end"])
    for r in rows:
        if r["proposition"]:
            continue
        for nid in r["note_ids"]:
            mk = re.search(rf"\[\^{re.escape(nid)}\]", text)
            if mk and not any(a <= mk.start() < b for a, b in exclude):
                r["proposition"] = _sentence(text, mk.start(), mk.end())
                r["in_text"].append(mk.group(0))
                break
        if not r["proposition"] and r["origin"] in ("inline", "identifier"):
            r["proposition"] = _proposition(text, r["start"], r["end"])

    form = {"in_text_missing": [], "uncited_entry": []}
    ref_rows = [i for i, r in enumerate(rows) if r["origin"] == "reference_list"]
    if ref_rows:
        for c in cites:
            if c["form"] == "author_date" and _match_in_text(c, rows) is None:
                if c["text"] not in form["in_text_missing"]:
                    form["in_text_missing"].append(c["text"])
        if any(c["form"] in ("author_date", "mla", "numbered") for c in cites):
            form["uncited_entry"] = [rows[i]["title"] or rows[i]["raw"][:80]
                                     for i in ref_rows if i not in used][:20]
        form["in_text_missing"] = form["in_text_missing"][:20]
    for r in rows:
        r["proposition"] = (r["proposition"] or "")[:600]
    return {"sources": rows, "citation_form": form}


def _match_in_text(c: dict, rows: list[dict]) -> Optional[int]:
    """Index of the source an in-text citation points to, else None."""
    if c["form"] == "numbered":
        for i, r in enumerate(rows):
            if r["origin"] == "reference_list" and r.get("number") == c["number"]:
                return i
        return None
    cited = c["name"].split()[-1] if c["name"] else ""
    for i, r in enumerate(rows):
        names = r.get("author_surnames") or []
        if not names or not cited:
            continue
        if cited != names[0].split()[-1] and c["name"] != names[0]:
            continue
        if c["form"] == "mla" or c.get("year") == (r.get("year") or ""):
            return i
    return None


def _dedupe(rows: list[dict]) -> list[dict]:
    """One row per source: same DOI / arXiv / PMID, or same normalised title +
    year. The reference-list row (most fields) is kept; the others' note ids
    and spans fold into it."""
    order = {"reference_list": 0, "note": 1, "inline": 2, "identifier": 3}
    out: list[dict] = []
    keys: dict[str, int] = {}
    for r in sorted(rows, key=lambda r: (order[r["origin"]], r["start"])):
        ks = [source_key(r)]
        if r.get("title"):
            ks.append(f"title:{norm_title(r['title'])}|{r.get('year') or ''}")
        hit = next((keys[k] for k in ks if k in keys and k != "title:|"), None)
        if hit is None:
            for k in ks:
                keys.setdefault(k, len(out))
            r["spans"] = [(r["start"], r["end"])]
            out.append(r)
            continue
        keep = out[hit]
        keep["note_ids"] += [n for n in r["note_ids"] if n not in keep["note_ids"]]
        keep["spans"].append((r["start"], r["end"]))
        for f in ("doi", "arxiv", "pmid", "venue", "volume", "pages"):
            if not keep.get(f) and r.get(f):
                keep[f] = r[f]
    out.sort(key=lambda r: r["start"])
    return out


def fill_unparsed(sources: list[dict], parse_fn: Callable[[list[str]], list[dict]]) -> int:
    """Local-model fallback (spec §1.3) for entries the heuristics couldn't
    read. ``parse_fn(raws) -> [{index, title, authors, year, venue, kind}]``
    runs on the localhost model. A returned title is kept only when it was
    read off the entry (``appears_in``). Returns how many were filled."""
    todo = [r for r in sources if not r["parsed"] and r["origin"] == "reference_list"
            and r.get("source_kind") != "legal"]
    if not todo:
        return 0
    try:
        parsed = parse_fn([r["raw"] for r in todo]) or []
    except Exception:  # noqa: BLE001 — the fallback must never stop a run
        return 0
    filled = 0
    for item in parsed:
        try:
            r = todo[int(item.get("index"))]
        except (TypeError, ValueError, IndexError):
            continue
        title = (item.get("title") or "").strip()
        if not title or not appears_in(title, r["raw"]):
            continue
        r["title"] = title
        if not r["author_surnames"]:
            r["author_surnames"] = [s for s in (
                fold(a.split(",")[0]) if "," in a else _surname_of_first_last(a)
                for a in (item.get("authors") or [])[:6]) if s]
        r["year"] = r["year"] or _clean_year(item.get("year"))
        r["venue"] = r["venue"] or (item.get("venue") or "")
        r["parsed"] = True
        r["style"] = r["style"] if r["style"] != "unknown" else "model"
        kind = item.get("kind") or ""
        r["source_kind"] = kind if kind in SOURCE_KINDS else classify(r)
        filled += 1
    return filled


SOURCE_KINDS = ["journal_article", "law_review", "conference_paper", "preprint", "book",
                "book_chapter", "report", "web", "news", "personal_communication",
                "course_material", "legal", "other"]

PARSE_SCHEMA = {
    "type": "object",
    "properties": {"entries": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "title": {"type": "string"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "year": {"type": "string"},
            "venue": {"type": "string"},
            "kind": {"type": "string", "enum": SOURCE_KINDS},
        },
        "required": ["index", "title", "authors", "year", "venue", "kind"],
    }}},
    "required": ["entries"],
}


def parse_prompt(raws: list[str]) -> str:
    lines = "\n".join(f"[{i}] {r}" for i, r in enumerate(raws))
    return ("Each numbered line below is one entry from a student paper's "
            "reference list. For each entry, copy its TITLE exactly as written "
            "(do not correct or complete it), list the authors as written, the "
            "year (empty if none), the journal / book / venue (empty if none), "
            "and the kind of source (legal = a court case, statute, regulation, "
            "court rule or constitution). Never invent a field that is not in the "
            "entry.\n\n" + lines)
