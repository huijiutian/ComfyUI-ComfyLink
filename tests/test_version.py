"""Version source-of-truth checks — no aiohttp / ComfyUI / relay needed.

Asserts __version__ looks like a semver and that pyproject.toml's version
string matches it (so the registry manifest never drifts from the package).

Run:  python -m unittest discover -s tests
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.version import __version__  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPROJECT = os.path.join(_ROOT, "pyproject.toml")

# X.Y.Z with an optional pre-release/build suffix (e.g. 0.1.0, 1.2.3-rc1).
_SEMVER = re.compile(r"^\d+\.\d+\.\d+([-+.].+)?$")


def _pyproject_version() -> str:
    """Read [project].version from pyproject.toml.

    Uses tomllib on 3.11+; falls back to a line parse on 3.10.
    """
    try:
        import tomllib  # type: ignore
    except ModuleNotFoundError:
        with open(_PYPROJECT, encoding="utf-8") as f:
            for line in f:
                m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
                if m:
                    return m.group(1)
        raise AssertionError("no version line found in pyproject.toml")
    with open(_PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]["version"]


class TestVersion(unittest.TestCase):
    def test_version_is_nonempty_semver(self):
        self.assertTrue(__version__, "__version__ must be non-empty")
        self.assertRegex(__version__, _SEMVER)

    def test_pyproject_matches_package(self):
        self.assertEqual(_pyproject_version(), __version__)


if __name__ == "__main__":
    unittest.main()
