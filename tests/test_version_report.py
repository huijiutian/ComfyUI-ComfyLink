"""The plugin reports its version to the relay on register + every heartbeat.

The app uses these to tell the user "your ComfyUI plugin has a new version".
Heartbeat carries them too (not just register) so a `git pull` + ComfyUI restart
refreshes what the relay sees. Both fields are optional server-side, so an old
relay just ignores them — but the plugin must always send them.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.relay import RelayClient  # noqa: E402
from comfylink.version import __commit__, __version__  # noqa: E402


class TestVersionReporting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Bare RelayClient with _json stubbed — we only care about the body.
        self.relay = RelayClient.__new__(RelayClient)
        self.calls: list[tuple[str, str, dict]] = []

        async def fake_json(method, path, body):
            self.calls.append((method, path, body))
            return {}

        self.relay._json = fake_json

    async def test_register_sends_version_and_commit(self):
        await self.relay.register("b1", "my-box")
        self.assertEqual(self.calls, [(
            "POST", "/v1/backends/register",
            {"backend_id": "b1", "name": "my-box",
             "version": __version__, "commit": __commit__},
        )])

    async def test_heartbeat_sends_version_and_commit(self):
        await self.relay.heartbeat("b1")
        self.assertEqual(self.calls, [(
            "POST", "/v1/backends/heartbeat",
            {"backend_id": "b1", "version": __version__, "commit": __commit__},
        )])


if __name__ == "__main__":
    unittest.main()
