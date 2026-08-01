"""Async client for the comfylink-relay HTTP API."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import socket
from typing import Optional
from urllib.parse import urlsplit

import aiohttp

from .version import __commit__, __version__

log = logging.getLogger("comfylink.relay")

# Claim is a server-held long-poll (~28s); allow margin over it.
CLAIM_TIMEOUT = aiohttp.ClientTimeout(total=45)
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)

# progress(wait=True) 也是服务端持有的长轮询:中继会把请求挂起最多 ~25s 等取消。
# 客户端超时必须**大于**那个窗口,否则长轮询会被我们自己掐断(每 30s 一次
# TimeoutError,取消也就等不到了)。只给这一个请求放宽,DEFAULT_TIMEOUT 保持 30s
# 不动 —— 其它调用都是普通的一来一回,给它们放宽只会让真故障拖更久才暴露。
PROGRESS_WAIT_TIMEOUT = aiohttp.ClientTimeout(total=45)

# ── Delivery retry (ride a short relay/network outage, e.g. a relay redeploy) ──
# Only the DELIVERY-critical, one-shot calls (result / sign_upload / put_object)
# use this: a job whose render finishes during the deploy window would otherwise
# lose its result (the rendered image is safe on disk, but never reaches the app,
# so the user must regenerate). We retry ONLY infrastructure blips — connection
# errors, timeouts, and 502/503/504 — with capped exponential backoff up to
# _RETRY_MAX_ELAPSED of cumulative wait. That budget sits comfortably under the
# relay reaper's 5-minute job-stale threshold, so a late-but-delivered result is
# never mistaken for a dead job. Permanent errors (4xx auth/revoke, SSRF, R2 4xx)
# propagate IMMEDIATELY so a real failure or an unpair is never masked as a retry.
_RETRY_MAX_ELAPSED = 90.0   # seconds of cumulative backoff before giving up
_RETRY_BASE = 1.0           # first backoff
_RETRY_CAP = 15.0           # per-attempt backoff ceiling
_RETRY_STATUSES = frozenset({502, 503, 504})  # transient proxy/relay; 4xx never retried

# Cap on object-storage downloads (input images). Bounds memory so a malicious
# or buggy relay can't point get_object at a huge/endless body and OOM the host.
MAX_OBJECT_BYTES = 64 * 1024 * 1024  # 64 MiB
_DOWNLOAD_CHUNK = 256 * 1024


class RelayError(RuntimeError):
    """A non-2xx relay response (or a locally-refused request when status == 0).

    `code` carries the relay's machine-readable ``error_code`` when the error
    body was JSON and contained one (e.g. "plugin_too_old"); it is "" otherwise.
    `payload` is the parsed JSON error body (``{}`` when the body wasn't a JSON
    object) so callers can read companion fields such as ``min_version`` /
    ``update_url`` without re-reading the response. Both are keyword-only with
    safe defaults so every existing ``RelayError(msg, status)`` call site is
    unaffected.
    """

    def __init__(self, message: str, status: int = 0, *, code: str = "",
                 payload: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.payload = payload or {}


def _insecure_allowed() -> bool:
    """Dev/integration escape hatch.

    The throwaway integration harness runs a fake ComfyUI + fake R2 on
    127.0.0.1, which the SSRF block below would otherwise reject. Setting
    COMFYLINK_ALLOW_INSECURE=1 skips the private-IP check and allows http so
    that harness keeps working. OFF by default => production is locked down.
    """
    return os.environ.get("COMFYLINK_ALLOW_INSECURE", "").strip() in ("1", "true", "yes")


def _ip_is_blocked(ip: str) -> bool:
    """True if `ip` is loopback/private/link-local/reserved — i.e. an SSRF
    target we must never fetch from (cloud metadata 169.254.169.254, localhost
    services, internal hosts). Pure + separately testable (no DNS)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        # Not parseable as an IP -> treat as unsafe rather than fail open.
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
        # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) — classify by the embedded v4.
        or (getattr(addr, "ipv4_mapped", None) is not None and _ip_is_blocked(str(addr.ipv4_mapped)))
    )


def _validate_url(url: str) -> None:
    """SSRF guard for relay-supplied (presigned R2) URLs.

    The relay hands us arbitrary GET/PUT URLs; a compromised relay or a crafted
    job payload could aim them at internal services or the cloud metadata
    endpoint. We require https and reject any URL whose host resolves to a
    private/loopback/link-local/reserved address.

    Note on host allowlisting: we deliberately do NOT pin an R2 host suffix.
    The plugin doesn't know the bucket/account host statically (it's chosen by
    the relay at sign time), so a static allowlist would be brittle. The
    resolved-IP block is the must-have defense and is sufficient to stop the
    metadata endpoint and localhost. Raises RelayError on rejection.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname

    if _insecure_allowed():
        # Dev harness: allow http + localhost, skip the private-IP block.
        if scheme not in ("http", "https"):
            raise RelayError(f"refusing non-http(s) URL: {scheme or '<none>'}")
        return

    if scheme != "https":
        raise RelayError(f"refusing non-https URL (scheme {scheme or '<none>'})")
    if not host:
        raise RelayError("refusing URL with no host")

    # Resolve and check EVERY returned address (defeats DNS that returns both a
    # public and a private record, and IPv6/IPv4 splits).
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise RelayError(f"cannot resolve host {host!r}: {e}")
    for info in infos:
        ip = info[4][0]
        if _ip_is_blocked(ip):
            raise RelayError(f"refusing URL to non-public address ({host} -> {ip})")


async def redeem_pair_code(
    session: aiohttp.ClientSession, base_url: str, code: str, name: str
) -> tuple[str, str]:
    """Exchange a one-time pairing code for a device token. Unauthenticated.

    Returns (device_token, device_id). Raises RelayError on failure.
    """
    url = base_url.rstrip("/") + "/v1/devices/claim"
    async with session.post(url, json={"code": code, "name": name},
                            timeout=DEFAULT_TIMEOUT) as r:
        if r.status != 200:
            raise RelayError(f"pairing failed: {await r.text()}", r.status)
        d = await r.json()
    return d["device_token"], d.get("device_id", "")


class RelayClient:
    # `auth` is any object exposing `async token() -> str` (see TokenAuth).
    def __init__(self, session: aiohttp.ClientSession, base_url: str, auth):
        self._session = session
        self._base = base_url.rstrip("/")
        self._auth = auth

    async def _headers(self) -> dict:
        return {"Authorization": f"Bearer {await self._auth.token()}"}

    async def _deliver(self, what: str, thunk):
        """Run a delivery-critical call, retrying across a short relay/network
        outage. `thunk` is a zero-arg async factory (called fresh each attempt);
        its return value is passed straight back on success.

        Retries ONLY transient infrastructure failures — connection errors,
        timeouts, and 502/503/504 — with capped exponential backoff, up to
        _RETRY_MAX_ELAPSED of cumulative sleep. A non-transient RelayError (any
        other status, incl. 4xx auth/revoke and 0 = SSRF/validation) propagates
        immediately. Retrying is safe: the relay's result write is terminal-
        guarded (a re-POST is a benign no-op, no duplicate job_event) and a
        presigned sign/PUT is repeatable.
        """
        slept = 0.0
        delay = _RETRY_BASE
        attempt = 0
        while True:
            attempt += 1
            try:
                return await thunk()
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                err = e
            except RelayError as e:
                if e.status not in _RETRY_STATUSES:
                    raise
                err = e
            if slept >= _RETRY_MAX_ELAPSED:
                log.warning("%s: giving up after %d attempt(s): %s", what, attempt, err)
                raise err
            sleep_for = min(delay, _RETRY_CAP, _RETRY_MAX_ELAPSED - slept)
            log.warning("%s: transient failure (attempt %d), retrying in %.0fs: %s",
                        what, attempt, sleep_for, err)
            await asyncio.sleep(sleep_for)
            slept += sleep_for
            delay = min(delay * 2, _RETRY_CAP)

    async def register(self, backend_id: str, name: str) -> dict:
        # version/commit let the app tell the user their plugin is out of date.
        # Optional server-side, so older relays simply ignore them.
        return await self._json("POST", "/v1/backends/register",
                                {"backend_id": backend_id, "name": name,
                                 "version": __version__, "commit": __commit__})

    async def sign_object_info(self, backend_id: str) -> tuple[str, str]:
        """Request a presigned PUT URL for this backend's object_info snapshot.

        Returns (key, url). The relay returns 503 if R2 isn't configured.
        """
        d = await self._json("POST", "/v1/backends/object-info/sign",
                             {"backend_id": backend_id})
        return d["key"], d["url"]

    async def upload_object_info(self, backend_id: str, object_info: dict) -> None:
        """Snapshot object_info to R2: sign a PUT, then upload the JSON to it.

        The blob no longer goes through the relay (egress fix) — we ship it
        straight to object storage via a presigned PUT. The SSRF guard in
        put_object passes for the https R2 URL the relay hands back.
        """
        _key, url = await self.sign_object_info(backend_id)
        data = json.dumps(object_info).encode()
        await self.put_object(url, data, "application/json")

    async def sign_put_loras(self, backend_id: str) -> tuple[str, str]:
        """Presigned PUT for this backend's LoRA inventory manifest.

        Same two-step shape as sign_object_info. A relay that predates this
        feature simply has no such route and answers 404 — the caller treats that
        as "this relay can't take an inventory yet" and moves on.
        """
        d = await self._json("POST", "/v1/backends/loras/sign-put",
                             {"backend_id": backend_id})
        return d["key"], d["url"]

    async def upload_loras(self, backend_id: str, manifest: dict) -> None:
        """Ship the LoRA inventory to R2: sign a PUT, then upload the JSON.

        ``retry=False`` on purpose. The delivery retry (see _deliver) exists so a
        FINISHED job's result survives a relay redeploy — losing it costs the
        user a regeneration. An inventory upload is not on that path: it is
        refreshed on a timer anyway, so burning 90s of backoff here would only
        keep a background task alive longer for no benefit.
        """
        _key, url = await self.sign_put_loras(backend_id)
        data = json.dumps(manifest).encode()
        await self.put_object(url, data, "application/json", retry=False)

    async def sign_put_workflow(self, backend_id: str, artifact: str,
                                workflow_id: Optional[str] = None) -> tuple[str, str]:
        """Request a presigned PUT URL for a workflow manifest or blob.

        artifact is "manifest" or "blob"; blob requires workflow_id. Returns
        (key, url). The relay returns 503 if R2 isn't configured.
        """
        body: dict = {"backend_id": backend_id, "artifact": artifact}
        if workflow_id is not None:
            body["workflow_id"] = workflow_id
        d = await self._json("POST", "/v1/backends/workflows/sign-put", body)
        return d["key"], d["url"]

    async def heartbeat(self, backend_id: str,
                        object_info_hash: str = "",
                        object_info_synced_at: float = 0.0) -> dict:
        """Mark this backend alive. Returns the relay's response body.

        The body is the plugin's ONLY inbound channel while idle, so it doubles
        as the relay's way to ask for something. Today that is
        ``loras_requested_at`` (worker._maybe_scan_models) and
        ``object_info_requested_at`` (worker._maybe_refresh_object_info) — both
        are "the user pressed refresh in the app at this timestamp". An older
        relay returns ``{}`` and the caller simply finds nothing to do, so
        reading the body is always safe.

        Version/commit go out on every beat so a `git pull` + ComfyUI restart
        refreshes what the relay/app see without waiting for a re-register.

        RECEIPT for the snapshot upload. The snapshot itself goes straight to R2
        over a presigned PUT, so the relay never sees it land and cannot tell the
        app "the new one is up". The beat carries the CONTENT fingerprint of
        whatever we last shipped (Pairing.object_info_hash) and the relay stores
        it, which is what lets the app wait for a refresh to actually complete.

        A FINGERPRINT, not a timestamp, on purpose: re-uploading a byte-identical
        snapshot leaves it unchanged, so the app can never be fooled into
        reporting "refreshed" for a beat that moved nothing.

        Sent only when we HAVE one. Empty means "keep what you have" relay-side,
        and omitting the key says exactly that — a backend that has never
        captured a snapshot must not put anything in that column, least of all
        some other identifier that happened to be at hand.

        ``object_info_synced_at`` is the OTHER half of that receipt, and it
        answers a different question. The fingerprint says WHAT we have; this says
        WHICH REFRESH REQUEST we have already served (it echoes back the
        ``object_info_requested_at`` the relay sent us — see
        worker._refresh_object_info). Both are needed because "the refresh
        finished" and "the content changed" are NOT the same event: a machine
        whose node set did not change since the last capture uploads nothing and
        keeps the very same fingerprint, yet the user's refresh was still served.
        Reporting only the fingerprint left the app waiting for a change that was
        never going to come.

        Same "empty = keep what you have" rule as the hash: 0 (never served a
        request) omits the key rather than writing a zero over the relay's value.
        """
        body = {"backend_id": backend_id,
                "version": __version__, "commit": __commit__}
        if object_info_hash:
            body["object_info_hash"] = object_info_hash
        if object_info_synced_at and object_info_synced_at > 0:
            body["object_info_synced_at"] = object_info_synced_at
        return await self._json("POST", "/v1/backends/heartbeat", body)

    async def abandon_jobs(self, backend_id: str) -> int:
        """Fail every still-claimed/running job left on this backend.

        Called once right after the worker's first successful register on a fresh
        process: ComfyUI's queue is in-memory, so a just-started process is by
        definition running zero jobs. Any job the relay still has as
        claimed/running is therefore a zombie from a previous run that was killed
        mid-job (the relay's reaper can't catch it — we immediately re-register
        and heartbeat, so its 'job stale AND backend offline' double-check never
        holds). The relay marks them failed and returns how many it cleared.
        """
        d = await self._json(
            "POST", f"/v1/backends/{backend_id}/jobs/abandon", {}
        )
        return int(d.get("abandoned", 0))

    async def claim(self, backend_id: str) -> Optional[dict]:
        """Long-poll for a job. Returns the job, or None on a 204 timeout."""
        async with self._session.get(
            self._base + "/v1/jobs/claim",
            params={"backend_id": backend_id},
            headers=await self._headers(),
            timeout=CLAIM_TIMEOUT,
        ) as r:
            if r.status == 204:
                return None
            await _check(r)
            return await r.json()

    async def progress(self, job_id: str, status: str, value: int, maximum: int,
                       wait: bool = False) -> dict:
        """Report progress. Returns {"cancel": bool}.

        ``wait=True`` 打开中继的「带取消等待的长轮询」:中继照常推进这个 job 的
        心跳(updated_at,reaper 据此判活),然后把请求挂起最多 ~25s,取消一被置位
        就立刻返回 —— 生成期的心跳因此同时就是取消通道,取消延迟从「最长一个心跳
        间隔」压到秒级。请求超时随之放宽到 PROGRESS_WAIT_TIMEOUT。

        ``wait`` 字段只在 True 时才写进 body:不带这个字段 = 老行为(中继立刻回),
        对老中继(不认识 wait)也天然兼容 —— 它会忽略这个字段照旧立刻返回。
        """
        body: dict = {"status": status, "progress": value, "max": maximum}
        if wait:
            body["wait"] = True
        return await self._json("POST", f"/v1/jobs/{job_id}/progress", body,
                                timeout=PROGRESS_WAIT_TIMEOUT if wait else None)

    async def result(self, job_id: str, status: str, images: list[dict], error: str = "",
                     error_code: str = "", total_bytes: int = 0) -> None:
        body: dict = {"status": status, "images": images, "error": error,
                      "total_bytes": total_bytes}
        if error_code:
            body["error_code"] = error_code
        # Delivery-critical: retry across a short relay outage so a finished job's
        # result isn't lost to a redeploy window. Idempotent (terminal-guarded).
        await self._deliver(
            f"result[{job_id}]",
            lambda: self._json("POST", f"/v1/jobs/{job_id}/result", body),
        )

    async def sign_upload(self, job_id: str, kind: str, filename: str, content_type: str) -> tuple[str, str]:
        """Request a presigned PUT URL. Returns (r2_key, url)."""
        d = await self._deliver(
            f"sign_upload[{job_id}]",
            lambda: self._json("POST", "/v1/uploads/sign",
                               {"job_id": job_id, "kind": kind, "filename": filename,
                                "content_type": content_type}),
        )
        return d["r2_key"], d["url"]

    async def put_object(self, url: str, data: bytes, content_type: str,
                         retry: bool = True) -> None:
        """Upload bytes to object storage via a presigned PUT URL (no auth header).

        ``retry`` defaults to True (the historical behaviour — every existing
        caller is on the delivery path). Pass False for background, re-runnable
        uploads such as the LoRA inventory, where riding out an outage buys
        nothing and only prolongs a best-effort task.
        """
        _validate_url(url)  # SSRF guard: relay-supplied URL (permanent — outside retry).

        async def _put():
            async with self._session.put(url, data=data,
                                         headers={"Content-Type": content_type}) as r:
                if r.status >= 300:
                    # Carry the status so a transient R2 5xx is retried while a 4xx
                    # (expired/forbidden presign) fails fast.
                    raise RelayError(f"storage PUT {r.status}: {await r.text()}", r.status)

        if not retry:
            await _put()
            return
        # A presigned PUT is idempotent (same key, overwrite), so retrying a blip
        # is safe. Upload is the other half of delivery — ride the outage here too.
        await self._deliver("put_object", _put)

    async def get_object(self, url: str, max_bytes: int = MAX_OBJECT_BYTES) -> bytes:
        _validate_url(url)  # SSRF guard: relay-supplied URL.
        async with self._session.get(url) as r:
            r.raise_for_status()
            # Reject up front if the server advertises an oversized body...
            clen = r.content_length
            if clen is not None and clen > max_bytes:
                raise RelayError(f"object too large: {clen} > {max_bytes} bytes")
            # ...and enforce while streaming in case Content-Length lies/absent.
            buf = bytearray()
            async for chunk in r.content.iter_chunked(_DOWNLOAD_CHUNK):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise RelayError(f"object exceeded {max_bytes} bytes")
            return bytes(buf)

    async def _json(self, method: str, path: str, body: dict,
                    timeout: Optional[aiohttp.ClientTimeout] = None) -> dict:
        """One JSON round-trip. ``timeout`` defaults to DEFAULT_TIMEOUT; only the
        server-held long-poll (progress wait=True) passes its own, longer one."""
        async with self._session.request(
            method, self._base + path, json=body,
            headers=await self._headers(), timeout=timeout or DEFAULT_TIMEOUT,
        ) as r:
            await _check(r)
            if r.content_length == 0:
                return {}
            ctype = r.headers.get("Content-Type", "")
            return await r.json() if "json" in ctype else {}


def _error_payload(body: str) -> dict:
    """Best-effort parse of a relay error body into a dict. NEVER raises.

    A non-JSON body, an empty body, or JSON that isn't an object all degrade to
    ``{}`` (=> RelayError.code == ""). We deliberately swallow everything: this
    runs on the failure path, so a surprise here must not replace the real error
    with a parse error.
    """
    try:
        data = json.loads(body)
    except Exception:  # noqa: BLE001 - any parse failure => no structured info
        return {}
    return data if isinstance(data, dict) else {}


async def _check(r: aiohttp.ClientResponse) -> None:
    if r.status >= 300:
        # Read the body ONCE: it is both the human message and the source of the
        # machine-readable error_code (aiohttp won't let us read it twice).
        body = await r.text()
        payload = _error_payload(body)
        code = payload.get("error_code")
        raise RelayError(
            f"relay {r.request_info.method} {r.url.path} -> {r.status}: {body}",
            r.status,
            code=code if isinstance(code, str) else "",
            payload=payload,
        )
