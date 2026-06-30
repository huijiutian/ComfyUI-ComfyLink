"""Async client for the LOCAL ComfyUI HTTP API (localhost).

This is the same public API the mobile app's DirectConnection uses — the plugin
just drives it locally, headless, on behalf of a relay job. We deliberately
drive it over the STABLE REST endpoints only (/prompt, /history, /queue,
/interrupt, /view): the websocket message format drifts across ComfyUI
releases, while these REST shapes have been stable for years, so the worker
stays version-independent.
"""

from __future__ import annotations

import aiohttp


class ComfyClient:
    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        self._session = session
        self._base = base_url.rstrip("/")

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

    async def history(self, prompt_id: str) -> dict:
        async with self._session.get(self._base + f"/history/{prompt_id}") as r:
            r.raise_for_status()
            return await r.json()

    async def queue(self) -> dict:
        """Current ComfyUI queue.

        Returns ``{"queue_running": [[number, "<pid>", ...], ...],
        "queue_pending": [...]}`` — the prompt id is at index 1 of each entry.
        Stable across ComfyUI versions; used to tell "still running/pending"
        from "interrupted/deleted" (present in neither history nor queue).
        """
        async with self._session.get(self._base + "/queue") as r:
            r.raise_for_status()
            return await r.json()

    async def queue_delete(self, ids: list[str]) -> None:
        """Remove specific PENDING prompts from the queue (precise, never global).

        POST /queue {"delete": [...]} drops not-yet-started items only; it never
        touches the prompt that is currently running. We use it to cancel our
        own queued job without interrupting a user's local generation.
        """
        async with self._session.post(self._base + "/queue", json={"delete": ids}) as r:
            await r.read()

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
