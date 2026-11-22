"""Semantic Scholar HTTP for the local evaluation tool's academic-source
check (Story 1028, utils/academic_lookup).

Semantic Scholar's API page: unauthenticated requests share "1000 requests
per second ... among all unauthenticated users", and "the introductory rate
limit for an API key is 1 RPS on all endpoints". In practice the shared pool
is saturated (2026-11-06: 3 of 5 spaced keyless probes got HTTP 429, with no
Retry-After header). So:

* ``RATE_KEYED`` paces at 1.05 s, not the 10/s production's
  citation_verifier used to assume (it now imports this constant).
* A 429 goes through ``rate_limited_get`` (one wait of ≤ 5 s), then two more
  tries after ``BACKOFF`` waits, about 15 s in all. A Retry-After over
  ``LONG_RETRY_AFTER`` ends the call at once. Never an unbounded sleep —
  see utils/rate_limited_http and the OpenAlex 22-hour hang it describes.
* What can't be fetched raises ``SemanticScholarUnavailable``; the caller
  marks the source "not checked", never "not found".

``SEMANTIC_SCHOLAR_API_KEY``, when set, is sent as ``x-api-key`` per request
(to api.semanticscholar.org only) and never logged or written to disk.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from firstread.rate_limited_http import backoff_only_retry, rate_limited_get, retry_after_seconds

log = logging.getLogger(__name__)

API_BASE = "https://api.semanticscholar.org/graph/v1"
RATE_KEYED = 1.05      # introductory key: 1 request/s on all endpoints
RATE_ANON = 1.1        # the shared keyless pool
BACKOFF = (4.0, 10.0)  # extra waits after rate_limited_get's own short retry
LONG_RETRY_AFTER = 60.0
TIMEOUT = (5, 15)      # connect, read
# Research APIs ask callers to say who they are. Set FIRSTREAD_CONTACT_EMAIL
# to your own address; it is never filled in for you.
_CONTACT = (os.getenv("FIRSTREAD_CONTACT_EMAIL") or "").strip()
USER_AGENT = (f"local-first-read/1.0 (mailto:{_CONTACT})" if _CONTACT
              else "local-first-read/1.0")
BATCH_MAX = 500

_session: Optional[requests.Session] = None
_last_call = 0.0


class SemanticScholarUnavailable(RuntimeError):
    """A call could not be made or finished (rate limit, timeout, outage).
    ``str(exc)`` is instructor-facing."""


def api_key() -> str:
    return (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "").strip()


def auth_headers() -> dict:
    key = api_key()
    return {"x-api-key": key} if key else {}


def rate() -> float:
    return RATE_KEYED if api_key() else RATE_ANON


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        s.mount("https://", HTTPAdapter(max_retries=backoff_only_retry(
            total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset({"GET", "POST"}),  # the batch POST is a read
            raise_on_status=False)))
        _session = s
    return _session


def pace():
    """Sleep so calls in this process stay ``rate()`` seconds apart."""
    global _last_call
    wait = _last_call + rate() - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _call(method: str, url: str, **kwargs) -> requests.Response:
    """One logical request: pacing, ≤ 3 tries on 429 / timeouts, never a
    long sleep. Returns any non-429 response (the caller checks status)."""
    kwargs.setdefault("timeout", TIMEOUT)
    headers = {**auth_headers(), **kwargs.pop("headers", {})}
    last = "Semantic Scholar is rate-limiting (HTTP 429)"
    for extra in (0.0,) + BACKOFF:
        if extra:
            time.sleep(extra)
        pace()
        try:
            if method == "GET":
                resp = rate_limited_get(session(), url, service="Semantic Scholar",
                                        headers=headers, **kwargs)
            else:
                resp = session().post(url, headers=headers, **kwargs)
        except requests.RequestException as e:
            last = f"Semantic Scholar could not be reached ({type(e).__name__})"
            log.warning("%s: %s", last, e)
            continue
        if resp.status_code != 429:
            return resp
        wait = retry_after_seconds(resp)
        if wait is not None and wait > LONG_RETRY_AFTER:
            raise SemanticScholarUnavailable(
                f"Semantic Scholar is rate-limiting and asks to wait {wait:.0f} s")
    raise SemanticScholarUnavailable(last)


def get(path: str, params: dict) -> requests.Response:
    return _call("GET", f"{API_BASE}{path}", params=params)


def post_batch(ids: list[str], fields: str) -> list[Optional[dict]]:
    """``POST /paper/batch``: one record (or None) per id, in order."""
    out: list[Optional[dict]] = []
    for i in range(0, len(ids), BATCH_MAX):
        chunk = ids[i:i + BATCH_MAX]
        resp = _call("POST", f"{API_BASE}/paper/batch", params={"fields": fields},
                     json={"ids": chunk})
        if resp.status_code == 400:      # every id unknown / malformed
            out += [None] * len(chunk)
            continue
        if not resp.ok:
            raise SemanticScholarUnavailable(
                f"Semantic Scholar batch lookup returned HTTP {resp.status_code}")
        data = resp.json()
        if not isinstance(data, list) or len(data) != len(chunk):
            raise SemanticScholarUnavailable("Semantic Scholar batch reply had the wrong shape")
        out += [d if isinstance(d, dict) else None for d in data]
    return out
