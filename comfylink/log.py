"""Shared logger for the plugin."""

import logging

log = logging.getLogger("comfylink")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[comfylink] %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False
