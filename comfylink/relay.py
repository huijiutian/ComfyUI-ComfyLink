"""Async client for the comfylink-relay HTTP API."""

from __future__ import annotations

from typing import Optional

import aiohttp

# Claim is a server-held long-poll (~28s); allow margin over it.
CLAIM_TIMEOUT = aiohttp.ClientTimeout(total=45)
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)


class RelayError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


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

    async def put_object_info(self, backend_id: str, object_info: dict) -> None:
        await self._json("POST", "/v1/backends/object-info",
                         {"backend_id": backend_id, "object_info": object_info})

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

    async def result(self, job_id: str, status: str, images: list[dict], error: str = "") -> None:
        await self._json("POST", f"/v1/jobs/{job_id}/result",
                         {"status": status, "images": images, "error": error})

    async def sign_upload(self, job_id: str, kind: str, filename: str, content_type: str) -> tuple[str, str]:
        """Request a presigned PUT URL. Returns (r2_key, url)."""
        d = await self._json("POST", "/v1/uploads/sign",
                             {"job_id": job_id, "kind": kind, "filename": filename,
                              "content_type": content_type})
        return d["r2_key"], d["url"]

    async def put_object(self, url: str, data: bytes, content_type: str) -> None:
        """Upload bytes to object storage via a presigned PUT URL (no auth header)."""
        async with self._session.put(url, data=data, headers={"Content-Type": content_type}) as r:
            if r.status >= 300:
                raise RelayError(f"storage PUT {r.status}: {await r.text()}")

    async def get_object(self, url: str) -> bytes:
        async with self._session.get(url) as r:
            r.raise_for_status()
            return await r.read()

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
