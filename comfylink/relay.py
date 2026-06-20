"""Async client for the comfylink-relay HTTP API."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from typing import Optional
from urllib.parse import urlsplit

import aiohttp

# Claim is a server-held long-poll (~28s); allow margin over it.
CLAIM_TIMEOUT = aiohttp.ClientTimeout(total=45)
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Cap on object-storage downloads (input images). Bounds memory so a malicious
# or buggy relay can't point get_object at a huge/endless body and OOM the host.
MAX_OBJECT_BYTES = 64 * 1024 * 1024  # 64 MiB
_DOWNLOAD_CHUNK = 256 * 1024


class RelayError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


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

    async def register(self, backend_id: str, name: str) -> dict:
        return await self._json("POST", "/v1/backends/register",
                                {"backend_id": backend_id, "name": name})

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

    async def heartbeat(self, backend_id: str) -> None:
        await self._json("POST", "/v1/backends/heartbeat", {"backend_id": backend_id})

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

    async def progress(self, job_id: str, status: str, value: int, maximum: int) -> dict:
        """Report progress. Returns {"cancel": bool}."""
        return await self._json("POST", f"/v1/jobs/{job_id}/progress",
                                {"status": status, "progress": value, "max": maximum})

    async def result(self, job_id: str, status: str, images: list[dict], error: str = "",
                     error_code: str = "", total_bytes: int = 0) -> None:
        body: dict = {"status": status, "images": images, "error": error,
                      "total_bytes": total_bytes}
        if error_code:
            body["error_code"] = error_code
        await self._json("POST", f"/v1/jobs/{job_id}/result", body)

    async def sign_upload(self, job_id: str, kind: str, filename: str, content_type: str) -> tuple[str, str]:
        """Request a presigned PUT URL. Returns (r2_key, url)."""
        d = await self._json("POST", "/v1/uploads/sign",
                             {"job_id": job_id, "kind": kind, "filename": filename,
                              "content_type": content_type})
        return d["r2_key"], d["url"]

    async def put_object(self, url: str, data: bytes, content_type: str) -> None:
        """Upload bytes to object storage via a presigned PUT URL (no auth header)."""
        _validate_url(url)  # SSRF guard: relay-supplied URL.
        async with self._session.put(url, data=data, headers={"Content-Type": content_type}) as r:
            if r.status >= 300:
                raise RelayError(f"storage PUT {r.status}: {await r.text()}")

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

    async def _json(self, method: str, path: str, body: dict) -> dict:
        async with self._session.request(
            method, self._base + path, json=body,
            headers=await self._headers(), timeout=DEFAULT_TIMEOUT,
        ) as r:
            await _check(r)
            if r.content_length == 0:
                return {}
            ctype = r.headers.get("Content-Type", "")
            return await r.json() if "json" in ctype else {}


async def _check(r: aiohttp.ClientResponse) -> None:
    if r.status >= 300:
        raise RelayError(
            f"relay {r.request_info.method} {r.url.path} -> {r.status}: {await r.text()}",
            r.status,
        )
