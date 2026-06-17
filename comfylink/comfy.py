"""Async client for the LOCAL ComfyUI HTTP/WS API (localhost).

This is the same public API the mobile app's DirectConnection uses — the plugin
just drives it locally, headless, on behalf of a relay job.
"""

from __future__ import annotations

import aiohttp


class ComfyClient:
    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        self._session = session
        self._base = base_url.rstrip("/")

    def _ws_url(self, client_id: str) -> str:
        scheme = "wss" if self._base.startswith("https") else "ws"
        host = self._base.split("://", 1)[1]
        return f"{scheme}://{host}/ws?clientId={client_id}"

    async def object_info(self) -> dict:
        async with self._session.get(self._base + "/object_info") as r:
            r.raise_for_status()
            return await r.json()

    async def submit(self, prompt: dict, client_id: str) -> str:
        body = {"prompt": prompt, "client_id": client_id}
        async with self._session.post(self._base + "/prompt", json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"/prompt {r.status}: {await r.text()}")
            data = await r.json()
        pid = data.get("prompt_id")
        if not pid:
            raise RuntimeError(f"/prompt returned no prompt_id: {data}")
        return pid

    def ws_connect(self, client_id: str):
        return self._session.ws_connect(self._ws_url(client_id), heartbeat=30)

    async def history(self, prompt_id: str) -> dict:
        async with self._session.get(self._base + f"/history/{prompt_id}") as r:
            r.raise_for_status()
            return await r.json()

    async def view(self, filename: str, subfolder: str, type_: str) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": type_}
        async with self._session.get(self._base + "/view", params=params) as r:
            r.raise_for_status()
            return await r.read()

    async def upload_image(self, name: str, data: bytes) -> dict:
        form = aiohttp.FormData()
        form.add_field("image", data, filename=name, content_type="application/octet-stream")
        form.add_field("overwrite", "true")
        async with self._session.post(self._base + "/upload/image", data=form) as r:
            r.raise_for_status()
            return await r.json()

    async def interrupt(self) -> None:
        async with self._session.post(self._base + "/interrupt") as r:
            await r.read()
