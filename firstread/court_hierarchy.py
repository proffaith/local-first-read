"""Court hierarchy utilities for subsequent-history (Shepardization) checks.

Loads courts.json once at import and provides helpers to determine whether
one court is hierarchically above another in the same jurisdiction.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load courts.json
# ---------------------------------------------------------------------------
_COURTS_PATH = os.path.join(os.path.dirname(__file__), "courts.json")

COURT_BY_ID: Dict[str, dict] = {}

try:
    with open(_COURTS_PATH, "r", encoding="utf-8") as f:
        _courts_list = json.load(f)
    for c in _courts_list:
        cid = c.get("id", "")
        if cid:
            COURT_BY_ID[cid] = c
except Exception:
    log.warning("Failed to load courts.json — court hierarchy checks disabled")

# ---------------------------------------------------------------------------
# Federal hierarchy (hardcoded — appeal_to chains are incomplete in courts.json)
# ---------------------------------------------------------------------------
# Rank: higher number = higher court
_FEDERAL_RANK: Dict[str, int] = {"scotus": 10}

# Circuit courts of appeals
for _i in range(1, 12):
    _FEDERAL_RANK[f"ca{_i}"] = 5
_FEDERAL_RANK["cadc"] = 5
_FEDERAL_RANK["cafc"] = 5

# Well-known federal district court slugs follow the pattern
# ending in "d" (e.g., "mdd", "nyed", "cacd").  We don't enumerate
# them all — we detect them via appeal_to pointing at a circuit.

# Bankruptcy courts (e.g., "almb", "alnb") rank below districts.
# Detected by name heuristic.

_LEVEL_RANK = {
    "colr": 4,   # Court of Last Resort (state supreme courts)
    "iac": 3,    # Intermediate Appellate Court
    "gjc": 2,    # General Jurisdiction Court (trial)
    "ljc": 1,    # Limited Jurisdiction Court
}

# Overrides for courts whose vendored courts.json entry has level:"" AND a
# name the heuristics misread. Maryland renamed both appellate courts in
# Dec 2022 — "md" (Court of Appeals, now Supreme Court of Maryland) is the
# court of last resort, "mdctspecapp" (Court of Special Appeals, now
# Appellate Court of Maryland) is intermediate — but the name heuristic
# ranks both as "appeal" = 3, so history checks miss md → mdctspecapp.
_LEVEL_OVERRIDES: Dict[str, str] = {
    "md": "colr",
    "mdctspecapp": "iac",
}


def _court_rank(court_id: str) -> int:
    """Return a numeric rank for a court (higher = higher court)."""
    # Hardcoded federal courts
    if court_id in _FEDERAL_RANK:
        return _FEDERAL_RANK[court_id]

    if court_id in _LEVEL_OVERRIDES:
        return _LEVEL_RANK[_LEVEL_OVERRIDES[court_id]]

    info = COURT_BY_ID.get(court_id)
    if not info:
        return 0  # unknown

    # Level field (when populated)
    level = (info.get("level") or "").lower()
    if level in _LEVEL_RANK:
        return _LEVEL_RANK[level]

    # Infer from appeal_to — if a court appeals to a circuit, it's a district
    appeal_to = info.get("appeal_to") or ""
    if appeal_to in _FEDERAL_RANK and _FEDERAL_RANK.get(appeal_to, 0) >= 5:
        return 3  # district-level (appeals to circuit)

    # Name heuristics
    name = (info.get("name") or "").lower()
    if "supreme" in name:
        return 4
    if "bankruptcy" in name:
        return 2
    if "district" in name:
        return 3
    if any(kw in name for kw in ("appeal", "appellate", "circuit")):
        return 3  # intermediate appellate

    # Type field fallback
    court_type = (info.get("type") or "").lower()
    if court_type == "appellate":
        return 3
    if court_type == "trial":
        return 2

    return 0


def _court_system(court_id: str) -> str:
    """Return 'federal', 'state', or '' for a court."""
    if court_id in _FEDERAL_RANK:
        return "federal"
    info = COURT_BY_ID.get(court_id, {})
    return (info.get("system") or "").lower()


def _court_location(court_id: str) -> str:
    """Return the state/location string for a court, or ''."""
    info = COURT_BY_ID.get(court_id, {})
    return (info.get("location") or "").lower()


def _appeals_to(court_id: str) -> Optional[str]:
    """Return the court_id this court appeals to, or None."""
    info = COURT_BY_ID.get(court_id, {})
    return info.get("appeal_to") or None


def is_higher_court(citing_court_id: str, cited_court_id: str) -> bool:
    """Return True if *citing_court_id* is hierarchically above *cited_court_id*.

    Checks:
    1. Direct appeal_to chain (cited → citing)
    2. Same system (federal/state) and same jurisdiction/location
    3. Rank comparison (citing rank > cited rank)
    """
    if not citing_court_id or not cited_court_id:
        return False
    if citing_court_id == cited_court_id:
        return False

    citing_rank = _court_rank(citing_court_id)
    cited_rank = _court_rank(cited_court_id)

    # Must be strictly higher
    if citing_rank <= cited_rank:
        return False

    # Check appeal_to chain (up to 3 levels: district → circuit → scotus)
    cur = cited_court_id
    for _ in range(3):
        parent = _appeals_to(cur)
        if not parent:
            break
        if parent == citing_court_id:
            return True  # direct appellate relationship
        cur = parent

    # SCOTUS is above everything
    if citing_court_id == "scotus":
        return True

    # Same-system check
    citing_sys = _court_system(citing_court_id)
    cited_sys = _court_system(cited_court_id)

    if citing_sys and cited_sys and citing_sys != cited_sys:
        return False  # don't compare across federal/state

    # For state courts, require same location/jurisdiction
    if cited_sys == "state":
        citing_loc = _court_location(citing_court_id)
        cited_loc = _court_location(cited_court_id)
        if citing_loc and cited_loc and citing_loc != cited_loc:
            return False

    return True


def should_skip_history(court_id: str) -> bool:
    """Return True if subsequent-history checking should be skipped for this court.

    Skips SCOTUS (no higher court exists) and state supreme courts that have
    no appeal_to (already the court of last resort in their system).
    """
    if court_id == "scotus":
        return True

    info = COURT_BY_ID.get(court_id, {})
    level = (_LEVEL_OVERRIDES.get(court_id) or info.get("level") or "").lower()
    system = (info.get("system") or "").lower()

    # State courts of last resort
    if level == "colr":
        return True

    # State supreme courts identified by name when level is empty
    if system == "state" and not _appeals_to(court_id):
        name = (info.get("name") or "").lower()
        if "supreme" in name:
            return True

    return False
