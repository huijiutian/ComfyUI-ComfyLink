"""Best-effort "a newer plugin is available" check for the ComfyUI panel.

The panel polls /comfylink/status every 3s, so this must NEVER hit the relay on
every poll. We fetch the public, unauthenticated GET {relay}/v1/versions at most
once per _TTL, cache the parsed result, and NEGATIVELY cache failures (a failed
refresh still advances the timestamp, so an offline / old / flaky relay isn't
hammered every 3s). Any failure — offline, old relay 404, bad JSON, timeout —
degrades silently to "no update info": the panel then simply shows nothing extra.

Version comparison mirrors the app's semver semantics (app/lib/core/version/
semver.dart): tolerant of a leading v and missing segments, but ANYTHING it can't
parse compares as "not older" so we never nag on a version string we don't
understand.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import aiohttp

from .version import __version__

# Refresh at most once per half hour (applies to BOTH success and failure).
_TTL = 30 * 60.0
# Short per-request timeout: the status poll must stay snappy even on the one
# poll (every 30 min) that actually reaches out to the relay.
_TIMEOUT = aiohttp.ClientTimeout(total=4)

# The "no update info" result — also the shape callers get on any failure.
_EMPTY: dict = {"available": False, "latest": "", "below_min": False, "url": ""}

# Module-level negative cache. `_last_check` is a time.monotonic() stamp of the
# last ATTEMPT (success or failure); `_cached` is the last computed result.
_last_check: Optional[float] = None
_cached: dict = dict(_EMPTY)

_NUM = re.compile(r"^\d+$")


def _parse(raw) -> Optional[tuple]:
    """Parse an "x.y.z" version into an int tuple, or None if unparseable.

    Tolerant like the app's semver: strips a leading v/V and any -pre / +build
    suffix, pads missing trailing segments with 0 ("1.0" -> (1, 0, 0)). Empty,
    non-numeric segments, or more than 3 segments -> None.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    if s[:1] in ("v", "V"):
        s = s[1:]
    cut = min((i for i in (s.find("-"), s.find("+")) if i >= 0), default=-1)
    if cut >= 0:
        s = s[:cut]
    if not s:
        return None
    parts = s.split(".")
    if len(parts) > 3:
        return None
    out = [0, 0, 0]
    for i, p in enumerate(parts):
        if not _NUM.match(p):
            return None
        out[i] = int(p)
    return tuple(out)


def _is_older(current: str, other: str) -> bool:
    """True if `current` is strictly older than `other`.

    Either side unparseable -> False (conservative: never nag on a version we
    can't compare). Mirrors semver.dart's isOlder.
    """
    a = _parse(current)
    b = _parse(other)
    if a is None or b is None:
        return False
    return a < b


async def _fetch(base_url: str) -> dict:
    """Hit /v1/versions and compute the update result vs the running version.

    Raises on any transport / HTTP / JSON problem — the caller swallows it.
    """
    url = base_url.rstrip("/") + "/v1/versions"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=_TIMEOUT) as r:
            if r.status != 200:
                raise RuntimeError(f"versions HTTP {r.status}")
            data = await r.json()
    plugin = (data or {}).get("plugin") or {}
    latest = str(plugin.get("latest") or "")
    minimum = str(plugin.get("min") or "")
    upurl = str(plugin.get("url") or "")
    return {
        "available": _is_older(__version__, latest),
        "latest": latest,
        "below_min": _is_older(__version__, minimum),
        "url": upurl,
    }


async def get_update_info(base_url: str) -> dict:
    """Cached update result, refreshing at most once per _TTL.

    Best-effort: any failure is swallowed and the timestamp is STILL advanced
    (negative cache), so a failing relay isn't retried on every 3s status poll.
    Returns a fresh dict each call (safe for the caller to mutate / serialize).
    """
    global _last_check, _cached
    now = time.monotonic()
    if _last_check is not None and (now - _last_check) < _TTL:
        return dict(_cached)
    # Advance BEFORE the await so overlapping polls don't stampede the relay.
    _last_check = now
    try:
        _cached = await _fetch(base_url)
    except Exception:  # noqa: BLE001 - any failure => no update info, never raise
        _cached = dict(_EMPTY)
    return dict(_cached)
