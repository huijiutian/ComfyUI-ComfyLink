"""Runtime status shared between the worker thread and the panel's HTTP routes.

Plain attributes guarded by a lock; the worker writes, the routes read.
Module singleton: [STATUS].
"""

from __future__ import annotations

import threading


class Status:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = "unpaired"  # unpaired | connecting | online | error
        self.node_count = 0
        self.error = ""
        self.active = False  # a job is currently running
        # Relay refused to serve this plugin version (403 error_code
        # "plugin_too_old"). Field names match the relay/app contract exactly —
        # one spelling across all three repos. Always present in snapshot() so
        # the panel can rely on them rather than probing for undefined.
        self.plugin_too_old = False
        self.plugin_min_version = ""
        self.plugin_update_url = ""
        # WebP conversion had to drop the prompt metadata (this env's Pillow
        # rejects the ``xmp=`` save argument): images still convert, but the
        # app can't read a prompt back out of them ("create preset from this
        # image" reports no generation info). Set by jobs.encode_output's
        # degrade branch; surfaced as a soft warning in the panel so the
        # condition is no longer symptomless (R-1.0.6-26 复诊). Sticky until
        # ComfyUI restarts — the env doesn't fix itself mid-run.
        self.webp_xmp_degraded = False

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "node_count": self.node_count,
                "error": self.error,
                "active": self.active,
                "plugin_too_old": self.plugin_too_old,
                "plugin_min_version": self.plugin_min_version,
                "plugin_update_url": self.plugin_update_url,
                "webp_xmp_degraded": self.webp_xmp_degraded,
            }


STATUS = Status()
