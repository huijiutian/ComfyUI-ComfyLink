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
            }


STATUS = Status()
