"""
PII scrubber: replaces student names, emails, ids, phone numbers and similar
identifiers with deterministic pseudonyms before any text is used.

Snapshot of ProfFaith's ``utils/pii_gate.py``. In ProfFaith it guards every
outbound emission and writes an audit row to the database; in this local tool
the model runs on localhost, so the scrubber's job is narrower: the model and
the saved reports only ever see the pseudonymised text, and the run's
``_emissions.jsonl`` records exactly what went out to the research APIs
(citation identifiers only).

- Identities are replaced with DETERMINISTIC pseudonyms per (context, sub) so
  cross-references stay legible ("Student A37 replied to Student B04"). The
  map is a keyed hash (PII_PSEUDONYM_SECRET); nothing is persisted.
- Names come from a roster file, the LMS bulk-download filename prefixes and
  the download's index.html; the header-block and "my name is" rules catch
  names the roster misses. No scrubber is perfect: read the saved
  ``<stem>.txt`` ("what the model saw") when it matters.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# How long the scrubbed content copy is retained for leak-validation before
# being purged from the audit row (the row itself survives, content-free).
CONTENT_TTL_HOURS = int(os.environ.get("PII_AUDIT_CONTENT_TTL_HOURS", "24"))

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# ── Financial and government identifiers ─────────────────────────────────────
# Added 2026-07-30 after an e-discovery corpus review surfaced these shapes
# surviving the gate untouched: 337 card-shaped strings passed through with 0%
# recall, and wire instructions ("account 4738299011 ... routing 021000021")
# fired nothing. Coursework was never expected to carry them, but students paste
# record dumps, and none of it should reach an LLM if it can be avoided.
#
# Cards are validated by Luhn before redaction. Shape alone is not enough: a
# 16-digit run may be a docket, deal or contract number, and only the checksum
# separates those from a real card.
_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")

# Account/routing stay label-anchored, consistent with _LABELLED_FIELDS above:
# a bare 9-17 digit run is far more often a citation or record number.
_ACCOUNT_RE = re.compile(
    r"(?i)\b(?:bank\s+)?acc(?:oun)?t\.?\s*(?:#|no\.?|number|is)?\s*[:#]?\s*(\d{7,17})\b")
_ROUTING_RE = re.compile(
    r"(?i)\b(?:routing|aba|transit)\s*(?:#|no\.?|number)?\s*[:#]?\s*(\d{9})\b")
_PASSPORT_RE = re.compile(
    r"(?i)\bpassport\s*(?:#|no\.?|number)?\s*[:#]?\s*([A-Z0-9]{6,10})\b")
# The value must contain a digit, so "DL Piper" and similar never match.
_DL_RE = re.compile(
    r"(?i)\b(?:driver'?s?\s+licen[cs]e|dl|dln)\s*(?:#|no\.?|number)?\s*[:#]?\s*"
    r"((?=[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{5,15})\b")
# SSN written without separators, only where a label makes the reading certain.
_SSN_BARE_RE = re.compile(
    r"(?i)\b(?:ssn|soc(?:ial)?\.?\s*sec(?:urity)?\.?)\s*"
    r"(?:#|no\.?|number)?\s*[:#]?\s*(\d{9})\b")


def _luhn(digits: str) -> bool:
    """Luhn checksum — separates payment cards from lookalike record numbers."""
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0
_ID_LINE_RE = re.compile(
    r"(?im)^\s*(student\s*(?:id|number)|id\s*(?:no\.?|number))\s*[:#]\s*\S+.*$")

# ── Label-anchored sensitive fields ──────────────────────────────────────────
# These categories are matched ONLY when introduced by their label. That is a
# deliberate constraint, not laziness: legal writing is saturated with things
# that look like PII out of context. A bare-date rule would redact "decided
# June 12, 2025"; a bare-address rule would eat a property description; a bare
# number rule would mangle "606 U.S. 942". Over-redaction destroys the very
# feedback these features exist to give, so every rule here needs its label.
#
# Consequence worth stating plainly: PII mentioned in free prose with no label
# ("she was born in April 2004") is NOT caught. Labelled fields — the cover
# sheets, intake forms and record dumps students actually paste — are.
_LABELLED_FIELDS = {
    "dob": r"d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date|birthday",
    "ssn_like": r"ssn|social\s+security(?:\s+(?:no\.?|number|#))?",
    "student_ids": r"student\s*(?:id|no\.?|number|#)|(?:^|\s)id\s*(?:no\.?|number|#)",
    "addresses": r"(?:home|mailing|street|residential)?\s*address",
    "birthplace": r"place\s+of\s+birth|birth\s*place|born\s+in",
    "maiden_names": r"(?:mother'?s?\s+)?maiden\s+name",
    "gpa": r"g\.?p\.?a\.?|grade\s+point\s+average|cumulative\s+average",
    "disability": r"iep|504\s*plan|accommodation(?:\s+file)?|disability|"
                  r"special\s+education",
    "discipline": r"disciplinary(?:\s+(?:record|referral|action))?|"
                  r"referral\s+date|suspension|expulsion",
    "guardians": r"(?:parent|guardian|mother|father|emergency\s+contact)"
                 r"(?:'?s)?(?:\s+(?:name|email|phone|contact))?",
}
# Two shapes, because student PII arrives in two shapes.
#
# (a) FIELD-LIST form — "DOB: 04/17/2004" on its own line, as on a cover sheet
#     or record dump. The whole line goes; there is nothing else on it.
_LABELLED_LINE_RE = {
    cat: re.compile(rf"(?im)^\s*(?:{pat})\s*[:#=–-]\s*.*$")
    for cat, pat in _LABELLED_FIELDS.items()
}

# (b) INLINE form — "…(student ID M00432198, DOB 04/17/2004), who resides at…".
#     Here the label sits mid-sentence, so only the VALUE may be redacted;
#     taking the rest of the line would eat the student's actual argument.
#     Observed live 2026-07-22: a worst-case draft phrased every identifier this
#     way, and a line-anchored rule caught none of them.
_DATE_VALUE = (r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
               r"|(?:January|February|March|April|May|June|July|August|September"
               r"|October|November|December)\s+\d{1,2},?\s+\d{4}")

# Date-bearing labels: redact a date appearing shortly after the label. The
# {0,40} window keeps it in the same clause, so "decided June 12, 2025" three
# sentences later is untouched.
_INLINE_DATE_RE = {
    cat: re.compile(rf"(?i)(?:{pat})[^.\n]{{0,40}}?({_DATE_VALUE})")
    for cat, pat in {
        "dob": _LABELLED_FIELDS["dob"],
        "discipline": _LABELLED_FIELDS["discipline"] + r"|recorded|dated",
    }.items()
}

# Numeric-value labels: "a term GPA of 1.87".
_INLINE_GPA_RE = re.compile(
    rf"(?i)(?:{_LABELLED_FIELDS['gpa']})[^.\n]{{0,20}}?\b(\d\.\d{{1,2}})\b")

# Title-cased value labels: "place of birth Baltimore, MD", "maiden name Delacroix".
_INLINE_NAMEISH_RE = {
    cat: re.compile(rf"(?i)(?:{pat})\s*(?:[:#=–-]|\bis\b|\bof\b)?\s*"
                    rf"([A-Z][\w'-]+(?:[,\s]+[A-Z][\w'-]+){{0,3}})")
    for cat, pat in {
        "birthplace": _LABELLED_FIELDS["birthplace"],
        "maiden_names": _LABELLED_FIELDS["maiden_names"],
    }.items()
}

# Date of birth also appears as a narrative ("born on April 17, 2004"). Anchored
# on the birth verb so ordinary case-decision dates are untouched.
_DOB_NARRATIVE_RE = re.compile(
    r"(?i)\bborn\s+(?:on\s+)?"
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)\s+\d{1,2},?\s+\d{4})")

# Institution-formatted student IDs (e.g. M00432198). Configurable because the
# shape is institution-specific; the default is a single letter + 6-10 digits,
# which does not collide with reporter citations ("606 U.S. 942").
_STUDENT_ID_RE = re.compile(
    os.environ.get("PII_STUDENT_ID_PATTERN", r"\b[A-Za-z]\d{6,10}\b"))

# Case-file style references: IEP-2025-0417, #IEP-2025-0417, 504-2025-11.
_CASE_REF_RE = re.compile(
    r"(?i)#?\b(?:iep|504|acc(?:om)?)[-/]\d{2,4}[-/]\d{2,6}\b")

# US street addresses. Requires a street-type suffix so it cannot match a
# citation or an ordinary numeric phrase.
_ADDRESS_RE = re.compile(
    r"(?i)\b\d{1,6}\s+[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3}\s+"
    r"(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd|court|ct"
    r"|circle|cir|place|pl|way|terrace|ter|parkway|pkwy|highway|hwy)\b"
    r"(?:[,\s]+(?:apt\.?|apartment|unit|suite|ste\.?|#)\s*[\w-]+)?"
    r"(?:[,\s]+[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)?"
    r"(?:[,\s]+[A-Z]{2})?(?:\s+\d{5}(?:-\d{4})?)?")
_NAME_HEADER_RE = re.compile(
    r"(?im)^\s*(name|student|author|submitted\s+by|prepared\s+by)\s*:\s*.+$")
# Legal-memo heading ("FROM: Jordan Rivera") and signature-style "By: …" lines.
# The label is kept so the model still sees a memo header; only the value goes.
# A quoted e-mail's "From:" line is over-redacted too — harmless.
_FROM_HEADER_RE = re.compile(r"(?im)^(\s*(?:from|by)\s*:\s*)(.+)$")

# ── Bare student header block ────────────────────────────────────────────────
# Student papers open with an unlabelled block — name / course / instructor /
# date — that no label-anchored rule can see:
#
#     Jordan Rivera
#     BUSL 101 – Business Law
#     Professor Faith
#     September 5, 2026
#
# Live 2026-09-12: a memo submitted without a roster reached the local model
# with the student's name intact in exactly this shape (and in "FROM:").
# The rule below is deliberately narrow so legal content is never eaten: it
# only looks at the first _HEADER_LINES non-blank lines, only when that window
# also carries a course code, an instructor line or a memo TO:/RE: line, and
# only redacts lines that are nothing but 2-4 capitalised name tokens.
_HEADER_LINES = 12
_HEADER_TRIGGER_RE = re.compile(
    r"(?im)^\s*(?:professor|prof\.?|instructor|dr\.?|to|re|subject)\b"
    r"|\b[A-Z]{2,5}\s?-?\d{3}[A-Z]?\b")
_HEADER_NAME_LINE_RE = re.compile(
    r"^\s*[A-Z][a-z'\-]+(?:\s+(?:[A-Z]\.|[A-Z][a-z'\-]+)){1,3}"
    r"(?:,?\s+(?:Jr|Sr|II|III|IV)\.?)?\s*$")
# Lines that look like a name but are headings, courts or people who are
# not the student. Prefix-matched, case-insensitive.
_HEADER_SKIP_PREFIXES = (
    "professor", "prof", "instructor", "dr", "judge", "justice", "hon",
    "question presented", "questions presented", "brief answer",
    "short answer", "statement of facts", "table of contents",
    "table of authorities", "legal memorandum", "memorandum of law",
    "office memorandum", "legal analysis", "issue statement",
)


def _scrub_header_block(text: str) -> tuple[str, int]:
    """Redact bare name lines in a student header block (see above)."""
    lines = text.split("\n")
    window, seen = [], 0
    for i, ln in enumerate(lines):
        if ln.strip():
            window.append(i)
            seen += 1
            if seen >= _HEADER_LINES:
                break
    block = "\n".join(lines[i] for i in window)
    if not _HEADER_TRIGGER_RE.search(block):
        return text, 0
    n = 0
    for i in window:
        ln = lines[i]
        if not _HEADER_NAME_LINE_RE.match(ln):
            continue
        low = ln.strip().lower().rstrip(".:")
        if low.startswith(_HEADER_SKIP_PREFIXES):
            continue
        lines[i] = "[name redacted]"
        n += 1
    return ("\n".join(lines), n) if n else (text, 0)

# First-person self-identification ("I, Jane Q. Doe, argue…"). Safe because
# case/party names are never written in this first-person frame; only the
# name between "I," and the next comma is redacted.
_SELF_ID_RE = re.compile(
    r"\bI,\s+([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,3}),")
# Conversational self-introduction ("I am John Smith", "I'm Jane Doe",
# "my name is John Q. Public") — surfaced by the interview/negotiation chat
# tools, where students introduce themselves to the AI character. Requires
# 2-4 capitalized words so single-word role personas ("I am Jamie") and
# ordinary sentences don't match; an occasional Title-Case false positive
# only over-redacts, never leaks.
_SELF_INTRO_RE = re.compile(
    r"\b(I\s+am|I'm|[Mm]y\s+name\s+is)\s+"
    r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3})\b")
# "My name is Sadieanne" — a single given name after the explicit
# "my name is" frame is still a self-identification (discussion-board
# introductions, 2026-09-12: the roster knew the short form only, so the
# long form walked through). "I am"/"I'm" stay 2+ words for personas.
_SELF_NAME_RE = re.compile(
    r"\b([Mm]y\s+name\s+is)\s+([A-Z][A-Za-z'\-]+)\b(?!\s+[A-Z])")
# Closing signature blocks ("Sincerely,\n Jane Doe").
_SIGNOFF_RE = re.compile(
    r"(?im)^\s*(sincerely|respectfully(?:\s+submitted)?|regards|"
    r"best(?:\s+regards)?|thank\s+you)\s*,?\s*\n+\s*"
    r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3})\s*$")

# ── Identity-name normalisation ──────────────────────────────────────────────
# LMS display names are decorated in ways that defeat naive matching. Observed
# live: a Moodle account named "Jane Doe_Student" failed to redact "Jane Doe"
# in a draft, because the full-name form never matched and splitting on
# whitespace yielded ["Jane", "Doe_Student"] — "Jane" fell under the length
# guard, and "_" is a regex word character so \bFaith_Student\b never isolates
# "Faith". Both halves missed and the student's real name reached the LLM.
#
# Role words an institution may append/prepend to a display name. Dropped
# before matching so they neither anchor a match nor get redacted themselves.
_ROLE_WORDS = frozenset({
    "student", "learner", "pupil", "teacher", "instructor", "faculty", "prof",
    "professor", "ta", "tutor", "admin", "administrator", "staff", "guest",
    "test", "testing", "demo", "sample", "trial", "user", "account",
})

# Short tokens that are real names but also common English words. Matched
# case-sensitively (below) so "Art"/"Will"/"Grace" don't eat ordinary prose.
_SHORT_NAME_MIN = 3

def _name_variants(raw: str) -> tuple[list[str], list[str]]:
    """Split a display name into (full-name forms, individual name parts).

    Handles the decorations that broke live: role suffixes ("Jane Doe_Student",
    "Jane Doe (Student)"), inverted order ("FAITH, TIMOTHY"), and non-space
    separators (underscore, dot, hyphen-as-joiner).
    """
    raw = (raw or "").strip()
    if not raw:
        return [], []

    # Drop bracketed/parenthesised decorations: "Jane Doe (Student)".
    base = re.sub(r"[(\[{][^)\]}]*[)\]}]", " ", raw)

    # Split into word tokens on anything that isn't a name character. This is
    # what rescues "Faith_Student" -> ["Faith", "Student"].
    tokens = [t for t in re.split(r"[^A-Za-z'\-]+", base) if t]
    # Drop role words and lone initials.
    tokens = [t for t in tokens
              if t.lower().strip(".'-") not in _ROLE_WORDS and len(t.strip(".'-")) > 1]
    if not tokens:
        return [], []

    # "FAITH, TIMOTHY" -> "TIMOTHY FAITH". Only when the comma genuinely
    # separates two name groups, not a suffix like "Doe, Jr.".
    if "," in base:
        left, _, right = base.partition(",")
        lt = [t for t in re.split(r"[^A-Za-z'\-]+", left) if t
              and t.lower() not in _ROLE_WORDS and len(t) > 1]
        rt = [t for t in re.split(r"[^A-Za-z'\-]+", right) if t
              and t.lower() not in _ROLE_WORDS and len(t) > 1]
        if lt and rt:
            tokens = rt + lt

    fulls = []
    if len(tokens) > 1:
        fulls.append(" ".join(tokens))          # "Jane Doe"
        fulls.append(f"{tokens[-1]}, {tokens[0]}")  # "Doe, Jane"
    # Longest first so the full form wins before parts fragment it.
    fulls = sorted({f for f in fulls if len(f) >= 4}, key=len, reverse=True)
    return fulls, tokens


_warned_default_secret = False


def _secret() -> bytes:
    """HMAC key for pseudonyms/watermarks/opaque tags.

    The privacy policy's "pseudonyms are not reversible" claim is only true
    when this is a real secret: the final fallback is a source-visible
    constant, kept so tests and dev shells run — but a deployment landing on
    it produces publicly recomputable pseudonyms, so warn loudly once.
    """
    val = os.environ.get("PII_PSEUDONYM_SECRET")
    if not val:
        global _warned_default_secret
        if not _warned_default_secret:
            _warned_default_secret = True
            import logging
            logging.getLogger(__name__).warning(
                "pii_gate: PII_PSEUDONYM_SECRET is not set — pseudonyms are "
                "keyed on a PUBLIC constant and anyone with the roster can "
                "recompute them. Set PII_PSEUDONYM_SECRET in your .env.")
        val = "local-first-read-pii-gate"
    return val.encode("utf-8")


def pseudonym_for(context_id: str, sub: str) -> str:
    """Deterministic per-(course context, LMS sub) pseudonym, e.g. 'Student A37'.

    Pure keyed hash — the same student in the same course always maps to the
    same pseudonym, without any stored mapping to purge or leak.
    """
    digest = hmac.new(_secret(), f"{context_id}:{sub}".encode("utf-8"),
                      hashlib.sha256).digest()
    letter = chr(ord("A") + digest[0] % 26)
    number = (digest[1] * 256 + digest[2]) % 100
    return f"Student {letter}{number:02d}"


def opaque_ref(purpose: str, context_id: str, sub: str) -> str:
    """Deterministic 16-hex opaque handle for one student in one course, for a
    form to name a roster row without carrying the LMS sub (e.g. the study
    coach's instructor override). Collision-free in practice, unlike the
    display pseudonym; the server re-derives it over the roster to resolve it.
    """
    return hmac.new(_secret(), f"{purpose}:{context_id}:{sub}".encode("utf-8"),
                    hashlib.sha256).hexdigest()[:16]


def watermark_tag(context_id: str, sub: str) -> str:
    """Deterministic pseudonymous tag for reader-facing watermarks,
    e.g. 'PF-4A7C21E9'.

    Replaces embedding the reader's email/username in licensed-copy
    watermarks (CLAIM-EMI-013): the operator can still trace a leaked copy
    back to its licensee by recomputing the tag over known readers, but the
    document itself carries no PII. Pure keyed hash — nothing persisted.
    """
    digest = hmac.new(_secret(), f"wm:{context_id}:{sub}".encode("utf-8"),
                      hashlib.sha256).hexdigest()
    return f"PF-{digest[:8].upper()}"


def scrub(text: str, identities: list | None = None,
          context_id: str = "") -> tuple[str, dict]:
    """Redact PII from student-work text before any external emission.

    ``identities``: known people, each ``{"sub": ..., "name": ..., "email": ...}``
    (name/email optional — launches configured claim-free still pseudonymize
    generic patterns). Known identities are replaced with their deterministic
    pseudonym; a generic pattern pass then catches self-identification the
    roster doesn't know about.

    Returns ``(clean_text, report)`` — the report holds redaction COUNTS by
    category, never the redacted values.
    """
    # Existing keys are preserved verbatim — callers and tests assert on them.
    # New categories are additive.
    report = {"identity_names": 0, "identity_emails": 0, "generic_emails": 0,
              "phones": 0, "ssn_like": 0, "id_lines": 0, "name_headers": 0,
              "self_id": 0, "signoffs": 0, "cards": 0, "bank_accounts": 0,
              "routing_numbers": 0, "passports": 0, "drivers_licenses": 0}
    report.update({cat: 0 for cat in _LABELLED_FIELDS})
    report["addresses"] = 0
    clean = text or ""

    for ident in identities or []:
        pseudo = pseudonym_for(context_id, str(ident.get("sub", "")))
        raw_name = (ident.get("name") or "").strip()
        fulls, parts = _name_variants(raw_name)

        # Match the raw display name too — it may be exactly how the student
        # typed it — then the normalised full forms, longest first.
        for form in ([raw_name] if len(raw_name) >= 4 else []) + fulls:
            pattern = re.compile(rf"\b{re.escape(form)}\b", re.IGNORECASE)
            clean, n = pattern.subn(pseudo, clean)
            report["identity_names"] += n

        # Then bare first/last mentions. Names of 3 chars ("Ana", "Wei", "Bo")
        # are real and must be caught, but are matched CASE-SENSITIVELY so an
        # ordinary lowercase word is never eaten; 4+ stays case-insensitive.
        for part in parts:
            part = part.strip(".'-")
            if len(part) < _SHORT_NAME_MIN:
                continue
            flags = 0 if len(part) < 4 else re.IGNORECASE
            part_re = re.compile(rf"\b{re.escape(part)}\b", flags)
            clean, n = part_re.subn(pseudo, clean)
            report["identity_names"] += n
        email = (ident.get("email") or "").strip()
        if email:
            email_re = re.compile(re.escape(email), re.IGNORECASE)
            clean, n = email_re.subn(f"{pseudo} (email redacted)", clean)
            report["identity_emails"] += n

    # Collapse stutter from adjacent part-replacements ("Jane Doakes" →
    # "Student B14 Student B14" when the full-name form didn't match).
    clean = re.sub(r"(Student [A-Z]\d{2})([,\s]+\1)+", r"\1", clean)

    clean, n = _NAME_HEADER_RE.subn("[header redacted]", clean)
    report["name_headers"] = n
    clean, n = _FROM_HEADER_RE.subn(lambda m: f"{m.group(1)}[name redacted]", clean)
    report["name_headers"] += n
    clean, n = _scrub_header_block(clean)
    report["name_headers"] += n
    clean, n = _SELF_ID_RE.subn("I, [name redacted],", clean)
    report["self_id"] = n
    clean, n = _SELF_INTRO_RE.subn(lambda m: f"{m.group(1)} [name redacted]", clean)
    report["self_id"] += n
    clean, n = _SELF_NAME_RE.subn(lambda m: f"{m.group(1)} [name redacted]", clean)
    report["self_id"] += n
    clean, n = _SIGNOFF_RE.subn(lambda m: f"{m.group(1)},\n[name redacted]", clean)
    report["signoffs"] = n
    clean, n = _ID_LINE_RE.subn("[id redacted]", clean)
    report["id_lines"] = n
    clean, n = _EMAIL_RE.subn("[email redacted]", clean)
    report["generic_emails"] = n
    clean, n = _PHONE_RE.subn("[phone redacted]", clean)
    report["phones"] = n
    clean, n = _SSN_RE.subn("[ssn redacted]", clean)
    report["ssn_like"] = n

    # ── Financial / government identifiers ───────────────────────────────────
    # Cards run before the label-anchored rules so a card written inside a
    # labelled line is redacted by checksum rather than swallowed whole.
    def _card(m):
        digits = re.sub(r"[ -]", "", m.group(0))
        if len(digits) == 16 and _luhn(digits):
            report["cards"] += 1
            return "[card redacted]"
        return m.group(0)

    clean = _CARD_RE.sub(_card, clean)

    def _value_only(cat, key):
        """Redact only the captured value, leaving the sentence intact."""
        def repl(m):
            report[key] = report.get(key, 0) + 1
            return m.group(0).replace(m.group(1), f"[{cat} redacted]")
        return repl

    clean = _SSN_BARE_RE.sub(_value_only("ssn", "ssn_like"), clean)
    clean = _ACCOUNT_RE.sub(_value_only("account", "bank_accounts"), clean)
    clean = _ROUTING_RE.sub(_value_only("routing", "routing_numbers"), clean)
    clean = _PASSPORT_RE.sub(_value_only("passport", "passports"), clean)
    clean = _DL_RE.sub(_value_only("licence", "drivers_licenses"), clean)

    # ── Label-anchored fields (see _LABELLED_FIELDS for why labels are required) ──
    # Field-list lines first (whole line), then inline mentions (value only).
    for cat, rx in _LABELLED_LINE_RE.items():
        clean, n = rx.subn(f"[{cat} redacted]", clean)
        report[cat] = report.get(cat, 0) + n

    # Inline: keep the surrounding sentence, redact only the value.
    def _keep_label(cat):
        return lambda m: m.group(0).replace(m.group(1), f"[{cat} redacted]")

    for cat, rx in _INLINE_DATE_RE.items():
        clean, n = rx.subn(_keep_label(cat), clean)
        report[cat] = report.get(cat, 0) + n

    clean, n = _INLINE_GPA_RE.subn(_keep_label("gpa"), clean)
    report["gpa"] = report.get("gpa", 0) + n

    for cat, rx in _INLINE_NAMEISH_RE.items():
        clean, n = rx.subn(_keep_label(cat), clean)
        report[cat] = report.get(cat, 0) + n

    clean, n = _DOB_NARRATIVE_RE.subn("[dob redacted]", clean)
    report["dob"] = report.get("dob", 0) + n

    clean, n = _CASE_REF_RE.subn("[disability redacted]", clean)
    report["disability"] = report.get("disability", 0) + n

    clean, n = _ADDRESS_RE.subn("[address redacted]", clean)
    report["addresses"] = report.get("addresses", 0) + n

    clean, n = _STUDENT_ID_RE.subn("[id redacted]", clean)
    report["student_ids"] = report.get("student_ids", 0) + n

    return clean, report


def pseudonymize_identity(text: str, name: str, *, context_id: str = "",
                          sub: str = "") -> tuple[str, str, int]:
    """Replace every variant of ONE person's name with their deterministic
    pseudonym, leaving everything else untouched.

    Narrower than scrub() on purpose: the reference-letter generator sends
    instructor-AUTHORED text in which labelled fields (a grade, a GPA) are
    included deliberately — only the subject's identity must not leave.
    Running the full scrub there would redact content the instructor chose
    to send. Uses the same name-variant matching as scrub()'s identity pass.

    Returns ``(clean_text, pseudonym, replacements)``.
    """
    pseudo = pseudonym_for(context_id, sub or (name or "").strip().lower())
    clean = text or ""
    count = 0
    raw = (name or "").strip()
    fulls, parts = _name_variants(raw)
    for form in ([raw] if len(raw) >= 4 else []) + fulls:
        pattern = re.compile(rf"\b{re.escape(form)}\b", re.IGNORECASE)
        clean, n = pattern.subn(pseudo, clean)
        count += n
    for part in parts:
        part = part.strip(".'-")
        if len(part) < _SHORT_NAME_MIN:
            continue
        flags = 0 if len(part) < 4 else re.IGNORECASE
        clean, n = re.compile(rf"\b{re.escape(part)}\b", flags).subn(pseudo, clean)
        count += n
    clean = re.sub(r"(Student [A-Z]\d{2})([,\s]+\1)+", r"\1", clean)
    return clean, pseudo, count
