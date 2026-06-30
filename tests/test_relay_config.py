"""Relay address resolution (env > comfylink.json > default) + per-relay state."""

import json
import os
import tempfile
import unittest

from comfylink import config


class RelayConfigTest(unittest.TestCase):
    def setUp(self):
        # Save/clear the env vars these tests manipulate so they don't leak.
        self._env = {
            k: os.environ.get(k)
            for k in ("COMFYLINK_RELAY_URL", "COMFYLINK_CONFIG", "COMFYLINK_STATE")
        }
        for k in self._env:
            os.environ.pop(k, None)
        # _state_path reads these module globals; save to restore.
        self._relay, self._isdef = config.RELAY_URL, config.RELAY_IS_DEFAULT

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        config.RELAY_URL, config.RELAY_IS_DEFAULT = self._relay, self._isdef

    def _write_cfg(self, obj_or_text):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        if isinstance(obj_or_text, str):
            f.write(obj_or_text)
        else:
            json.dump(obj_or_text, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_resolution_order(self):
        # 3) neither env nor config → baked-in production default.
        os.environ["COMFYLINK_CONFIG"] = "/no/such/comfylink.json"
        self.assertEqual(config._load_relay_url(), config._DEFAULT_RELAY_URL)
        # 2) config file relay_url.
        os.environ["COMFYLINK_CONFIG"] = self._write_cfg(
            {"relay_url": "https://test-relay.example"}
        )
        self.assertEqual(config._load_relay_url(), "https://test-relay.example")
        # 1) env var wins over the config file.
        os.environ["COMFYLINK_RELAY_URL"] = "https://env-relay.example"
        self.assertEqual(config._load_relay_url(), "https://env-relay.example")

    def test_bad_config_falls_back_to_default(self):
        os.environ["COMFYLINK_CONFIG"] = self._write_cfg("{not valid json")
        self.assertEqual(config._load_relay_url(), config._DEFAULT_RELAY_URL)

    def test_state_path_keyed_per_relay(self):
        # Default relay keeps the original filename (back-compat with installs).
        config.RELAY_IS_DEFAULT = True
        self.assertEqual(config._state_path().name, "comfylink_state.json")
        # A non-default relay gets its own stable, distinct state file.
        config.RELAY_IS_DEFAULT = False
        config.RELAY_URL = "https://test-relay.example"
        a = config._state_path().name
        self.assertTrue(a.startswith("comfylink_state.") and a.endswith(".json"))
        self.assertNotEqual(a, "comfylink_state.json")
        self.assertEqual(a, config._state_path().name)  # stable for same relay
        config.RELAY_URL = "https://other-relay.example"
        self.assertNotEqual(a, config._state_path().name)  # distinct per relay


if __name__ == "__main__":
    unittest.main()
