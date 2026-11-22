"""429 / Retry-After handling for retrying ``requests`` sessions
(Semantic Scholar and CourtListener in utils/citation_verifier).

urllib3's ``Retry`` honours ``Retry-After`` by default and SLEEPS for the
whole value. It does so for 413/429/503 whether or not the status is in
``status_forcelist`` — removing 429 from the list is not enough. CourtListener
throttles with ``Retry-After`` up to an hour, and one of these sessions serves
a synchronous web request (routes/citation_verify academic search), where any
wait past Heroku's 30 s router limit is an H12. The same mechanism hung an
OpenAlex worker for ~22 h on 2026-10-14 (see utils/openalex_http). Two rules:

* ``backoff_only_retry()`` retries transient 5xx/read errors with backoff,
  never 429, and never honours Retry-After.
* ``rate_limited_get()`` handles 429 itself: a short Retry-After (≤ 5 s, or
  none) is waited out once; anything longer is returned at once with a
  WARNING naming the service. Callers check ``resp.ok`` exactly as before,
  so their not-found / error results are unchanged.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# A 429 that asks for at most this long is waited out, then retried once.
SHORT_WAIT_MAX = 5.0


def backoff_only_retry(**kwargs) -> Retry:
    """``Retry(**kwargs)`` that never sleeps on Retry-After. 429 must not be
    in ``status_forcelist`` — route 429s through ``rate_limited_get``."""
    if 429 in (kwargs.get("status_forcelist") or ()):
        raise ValueError("429 belongs to rate_limited_get(), not status_forcelist")
    return Retry(respect_retry_after_header=False, **kwargs)


def retry_after_seconds(resp: requests.Response) -> float | None:
    """Seconds the server asked us to wait (delta-seconds or HTTP-date), or
    None when there is no usable Retry-After header."""
    raw = (resp.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def rate_limited_get(sess: requests.Session, url: str, *, service: str,
                     **kwargs) -> requests.Response:
    """``sess.get(url, **kwargs)`` that never waits more than
    ``SHORT_WAIT_MAX`` on a 429. ``service`` names the API in the WARNING."""
    resp = sess.get(url, **kwargs)
    if resp.status_code != 429:
        return resp
    wait = retry_after_seconds(resp)
    if wait is not None and wait > SHORT_WAIT_MAX:
        log.warning("%s rate-limited (HTTP 429, Retry-After %.0f s); "
                    "not waiting", service, wait)
        return resp
    time.sleep(1.0 if wait is None else wait)
    resp = sess.get(url, **kwargs)
    if resp.status_code == 429:
        log.warning("%s still rate-limited (HTTP 429) after a short wait; "
                    "giving up", service)
    return resp
