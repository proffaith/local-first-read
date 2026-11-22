"""
Rubric JSON schema (v1 -> v2) and normalization helpers.

v1 (legacy, single-group, fixed 4-level set):
    {
        "criteria": [
            {
                "name": "Legal Analysis",
                "weight": 30,
                "levels": {
                    "Exemplary":  "...",
                    "Proficient": "...",
                    "Developing": "...",
                    "Beginning":  "..."
                }
            },
            ...
        ]
    }

v2 (multi-group, custom levels per group; mirrors Brightspace's <criteria_group>):
    {
        "version": 2,
        "groups": [
            {
                "name": "Cultural Background & Accuracy",
                "levels": [
                    {"name": "Beginning",  "value": 9.0},
                    {"name": "Developing", "value": 13.0},
                    {"name": "Proficient", "value": 17.0},
                    {"name": "Exemplary",  "value": 20.0}
                ],
                "criteria": [
                    {
                        "name": "Cultural Background & Accuracy",
                        "cells": {
                            "Beginning":  "Minimal or inaccurate content",
                            "Developing": "Superficial or partially incorrect",
                            "Proficient": "Mostly accurate with minor gaps",
                            "Exemplary":  "Rich, accurate, well-synthesized"
                        }
                    }
                ]
            }
        ]
    }

Stored v1 rubrics auto-promote on read; nothing in the DB needs backfilling.
A v1 rubric promoted to v2 produces byte-identical D2L XML to the legacy path.
"""

from __future__ import annotations

from typing import Any, Dict, List


RUBRIC_JSON_VERSION = 2

DEFAULT_LEVEL_NAMES: List[str] = ["Beginning", "Developing", "Proficient", "Exemplary"]

DEFAULT_LEVELS: List[Dict[str, Any]] = [
    {"name": "Beginning",  "value": 1.0},
    {"name": "Developing", "value": 2.0},
    {"name": "Proficient", "value": 3.0},
    {"name": "Exemplary",  "value": 4.0},
]


def is_valid_rubric_json(rubric_json: Any) -> bool:
    """True if the input is a non-empty rubric in either v1 or v2 shape."""
    if not isinstance(rubric_json, dict):
        return False
    groups = rubric_json.get("groups")
    if isinstance(groups, list) and groups:
        return any(
            isinstance(g, dict) and (g.get("criteria") or [])
            for g in groups
        )
    criteria = rubric_json.get("criteria")
    return isinstance(criteria, list) and len(criteria) > 0


def normalize_rubric_json(rubric_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Promote any rubric_json to v2 with light validation.
    Raises ValueError if neither shape can be interpreted.
    """
    if not isinstance(rubric_json, dict):
        raise ValueError("rubric_json must be a dict")

    if rubric_json.get("version") == RUBRIC_JSON_VERSION or "groups" in rubric_json:
        return _coerce_v2(rubric_json)

    if "criteria" in rubric_json:
        return _v1_to_v2(rubric_json)

    raise ValueError("rubric_json has neither 'groups' (v2) nor 'criteria' (v1)")


def total_max_points(rubric_json_v2: Dict[str, Any]) -> float:
    """Sum of (max level_value * criterion count) across all groups."""
    total = 0.0
    for g in rubric_json_v2.get("groups") or []:
        levels = g.get("levels") or []
        criteria = g.get("criteria") or []
        if not levels or not criteria:
            continue
        max_level = max((float(lvl.get("value", 0.0)) for lvl in levels), default=0.0)
        total += max_level * len(criteria)
    return total


def _v1_to_v2(rubric_json: Dict[str, Any]) -> Dict[str, Any]:
    criteria_v1 = rubric_json.get("criteria") or []
    if not isinstance(criteria_v1, list):
        raise ValueError("v1 rubric_json 'criteria' must be a list")

    criteria_v2: List[Dict[str, Any]] = []
    for c in criteria_v1:
        if not isinstance(c, dict):
            continue
        levels_dict = c.get("levels") or {}
        cells = {name: str(levels_dict.get(name, "") or "") for name in DEFAULT_LEVEL_NAMES}
        criteria_v2.append({
            "name": str(c.get("name") or "").strip(),
            "cells": cells,
        })

    return {
        "version": RUBRIC_JSON_VERSION,
        "groups": [
            {
                "name": "Criteria",
                "levels": [dict(lvl) for lvl in DEFAULT_LEVELS],
                "criteria": criteria_v2,
            }
        ],
    }


def _coerce_v2(rubric_json: Dict[str, Any]) -> Dict[str, Any]:
    """Light validation + defaulting on a v2 rubric. Returns a fresh dict."""
    groups_in = rubric_json.get("groups") or []
    if not isinstance(groups_in, list):
        raise ValueError("v2 rubric_json 'groups' must be a list")

    groups_out: List[Dict[str, Any]] = []
    for gi, g in enumerate(groups_in):
        if not isinstance(g, dict):
            continue

        levels_in = g.get("levels")
        if not isinstance(levels_in, list) or not levels_in:
            levels_in = DEFAULT_LEVELS

        levels_out: List[Dict[str, Any]] = []
        for lvl in levels_in:
            if not isinstance(lvl, dict):
                continue
            name = str(lvl.get("name") or "").strip()
            try:
                value = float(lvl.get("value", 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if name:
                levels_out.append({"name": name, "value": value})

        if not levels_out:
            levels_out = [dict(lvl) for lvl in DEFAULT_LEVELS]

        criteria_in = g.get("criteria") or []
        criteria_out: List[Dict[str, Any]] = []
        for c in criteria_in:
            if not isinstance(c, dict):
                continue
            cells_in = c.get("cells") or {}
            cells_out = {
                lvl["name"]: str(cells_in.get(lvl["name"], "") or "")
                for lvl in levels_out
            }
            criteria_out.append({
                "name": str(c.get("name") or "").strip(),
                "cells": cells_out,
            })

        groups_out.append({
            "name": str(g.get("name") or f"Group {gi + 1}").strip(),
            "levels": levels_out,
            "criteria": criteria_out,
        })

    if not any(g["criteria"] for g in groups_out):
        raise ValueError("v2 rubric_json must have at least one group with at least one criterion")

    return {"version": RUBRIC_JSON_VERSION, "groups": groups_out}
