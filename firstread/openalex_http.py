"""Shared HTTP plumbing for OpenAlex (utils/scholar_corpus, utils/citation_verifier).

OpenAlex meters its API by a daily budget. Without a key, every caller on the
same IP shares one small budget (Heroku dynos share outbound IPs with other
apps, so it can be spent before we make a call). Once it is gone, OpenAlex
answers HTTP 429 with ``Retry-After`` set to the seconds until midnight UTC —
about a day. urllib3's ``Retry`` honours that header by default and SLEEPS for
it, so a session that lists 429 in ``status_forcelist`` hangs its worker for
~22 hours (seen 2026-10-14). Three rules here prevent that:

* ``retry_policy()`` retries only transient 5xx and never honours Retry-After.
* ``openalex_get()`` handles 429 itself: a per-second 429 (Retry-After ≤ 5 s,
  or none) is waited out once; a budget 429 ("Insufficient budget" in the
  body, or Retry-After > 60 s) raises ``OpenAlexBudgetExhausted`` at once,
  logs a WARNING, and latches so later calls in this process skip the network
  until the budget resets.
* ``OPENALEX_API_KEY``, when set, is sent as ``Authorization: Bearer`` on
  OpenAlex requests only — never on the session, because the scholar session
  also downloads PDFs from publisher hosts. A free key is a 10× budget of our
  own (https://help.openalex.org/api/authentication/).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# A 429 that asks for at most this long is the per-second limit: wait, retry once.
SHORT_WAIT_MAX = 5.0
# A Retry-After longer than this is the daily budget, not the per-second limit.
BUDGET_RETRY_AFTER = 60.0

BUDGET_MESSAGE = ("OpenAlex's daily request budget is used up; "
                  "try again after midnight UTC, when it resets.")

# Epoch seconds until which this process treats the budget as spent.
_budget_reset_at = 0.0


class OpenAlexBudgetExhausted(RuntimeError):
    """OpenAlex's daily budget (keyless shared-IP or per-key) is spent.
    ``str(exc)`` is instructor-facing."""

    def __init__(self, reset_at: float | None = None):
        super().__init__(BUDGET_MESSAGE)
        self.reset_at = reset_at


def retry_policy() -> Retry:
    """Retry transient 5xx with backoff. 429 is not retried here and no
    Retry-After header is ever slept on — see ``openalex_get``."""
    return Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503],
                 respect_retry_after_header=False)


def auth_headers() -> dict:
    """``Authorization: Bearer`` for ``OPENALEX_API_KEY``, else {}. Pass per
    request to api.openalex.org only."""
    key = (os.getenv("OPENALEX_API_KEY") or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def budget_exhausted() -> bool:
    """True while an earlier budget 429 in this process has not yet reset."""
    return time.time() < _budget_reset_at


def _header_seconds(resp: requests.Response, name: str) -> float | None:
    raw = (resp.headers.get(name) or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:  # Retry-After may also be an HTTP-date
        when = parsedate_to_datetime(raw)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def _retry_after(resp: requests.Response) -> float | None:
    wait = _header_seconds(resp, "Retry-After")
    if wait is not None:
        return wait
    try:
        body_wait = (resp.json() or {}).get("retryAfter")
        return max(0.0, float(body_wait)) if body_wait is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def _is_budget_429(resp: requests.Response) -> bool:
    if resp.status_code != 429:
        return False
    if "insufficient budget" in (resp.text or "").lower():
        return True
    wait = _retry_after(resp)
    return wait is not None and wait > BUDGET_RETRY_AFTER


def _seconds_to_utc_midnight() -> float:
    now = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                 microsecond=0)
    return (midnight - now).total_seconds()


def _trip_budget(resp: requests.Response) -> OpenAlexBudgetExhausted:
    global _budget_reset_at
    wait = (_retry_after(resp) or _header_seconds(resp, "X-RateLimit-Reset")
            or _seconds_to_utc_midnight())
    wait = min(wait, 26 * 3600)
    _budget_reset_at = time.time() + wait
    keyed = bool(auth_headers())
    log.warning(
        "OpenAlex daily budget exhausted (HTTP 429, %s): skipping OpenAlex in "
        "this process for %.1f h, until %s UTC.%s",
        "API key" if keyed else "keyless budget shared by every caller on this IP",
        wait / 3600,
        datetime.fromtimestamp(_budget_reset_at, timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "" if keyed else " Set OPENALEX_API_KEY for a budget of our own.")
    return OpenAlexBudgetExhausted(_budget_reset_at)


def openalex_get(sess: requests.Session, url: str, *, params: dict | None = None,
                 timeout: float = 25) -> requests.Response:
    """GET an OpenAlex endpoint; callers check ``resp.ok`` as before.

    Raises ``OpenAlexBudgetExhausted`` (without sleeping) when the daily
    budget is spent, now or earlier in this process. A per-second 429 is
    waited out once (≤ ``SHORT_WAIT_MAX``); any other 429 is returned."""
    if budget_exhausted():
        log.debug("OpenAlex budget latched until %s; skipping %s",
                  _budget_reset_at, url)
        raise OpenAlexBudgetExhausted(_budget_reset_at)
    headers = auth_headers()
    resp = sess.get(url, params=params, headers=headers, timeout=timeout)
    if resp.status_code == 429 and not _is_budget_429(resp):
        wait = _retry_after(resp)
        if wait is not None and wait > SHORT_WAIT_MAX:
            log.warning("OpenAlex rate-limited (HTTP 429, Retry-After %.0f s); "
                        "not waiting", wait)
            return resp
        time.sleep(1.0 if wait is None else wait)
        resp = sess.get(url, params=params, headers=headers, timeout=timeout)
    if resp.status_code == 429:
        if _is_budget_429(resp):
            raise _trip_budget(resp)
        log.warning("OpenAlex still rate-limited (HTTP 429) after a short wait")
    return resp
