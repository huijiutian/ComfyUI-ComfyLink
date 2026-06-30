"""Single source of truth for the plugin version.

Keep this in sync with the `version` field in pyproject.toml (the Comfy
Registry reads pyproject; this constant feeds the runtime/panel).
"""

import subprocess
from pathlib import Path

__version__ = "0.1.0"


def _detect_commit() -> str:
    """Short git commit of the installed plugin (panel display, so the user can
    tell whether they `git pull`'d the latest). Best-effort + computed ONCE at
    import: any failure (no git, downloaded as zip, etc.) yields 'dev'."""
    try:
        root = Path(__file__).resolve().parent.parent  # plugin repo root
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "dev"
    except Exception:
        pass
    return "dev"


__commit__ = _detect_commit()
