"""Unit tests for the sprint1-security fixes — pure, no network.

  - relay._ip_is_blocked / relay._validate_url  (SSRF guard)
  - config.State.save  (0600 file permissions)

DNS is mocked (socket.getaddrinfo) so _validate_url is exercised without ever
touching the network. The is-this-IP-private classification is factored into
_ip_is_blocked and tested directly.

Run:  python -m unittest discover -s tests
"""

import os
import stat
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# relay.py imports aiohttp (bundled with ComfyUI at runtime). These tests only
# exercise pure functions and never touch the network, so if aiohttp isn't
# installed in the test env we stub a minimal placeholder to allow the import.
try:
    import aiohttp  # noqa: F401
except ImportError:  # pragma: no cover - env-dependent
    import types

    _stub = types.ModuleType("aiohttp")
    _stub.ClientTimeout = lambda **k: None
    _stub.ClientResponse = object
    _stub.ClientSession = object
    sys.modules["aiohttp"] = _stub

from comfylink import relay  # noqa: E402
from comfylink.relay import RelayError, _ip_is_blocked, _validate_url  # noqa: E402


def _fake_getaddrinfo(ip):
    """Return a getaddrinfo-shaped result resolving any host to `ip`."""
    def _inner(host, port, *a, **k):
        return [(0, 0, 0, "", (ip, port or 0))]
    return _inner


class TestIpIsBlocked(unittest.TestCase):
    def test_public_ips_allowed(self):
        for ip in ("8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"):
            self.assertFalse(_ip_is_blocked(ip), ip)

    def test_metadata_endpoint_blocked(self):
        self.assertTrue(_ip_is_blocked("169.254.169.254"))  # link-local

    def test_loopback_blocked(self):
        self.assertTrue(_ip_is_blocked("127.0.0.1"))
        self.assertTrue(_ip_is_blocked("::1"))

    def test_private_ranges_blocked(self):
        for ip in ("10.0.0.5", "172.16.3.4", "192.168.1.1", "fc00::1", "fd12::3"):
            self.assertTrue(_ip_is_blocked(ip), ip)

    def test_reserved_and_unspecified_blocked(self):
        self.assertTrue(_ip_is_blocked("0.0.0.0"))
        self.assertTrue(_ip_is_blocked("240.0.0.1"))  # reserved

    def test_ipv4_mapped_loopback_blocked(self):
        self.assertTrue(_ip_is_blocked("::ffff:127.0.0.1"))

    def test_garbage_treated_as_blocked(self):
        # Fail closed: anything we can't parse as an IP is unsafe.
        self.assertTrue(_ip_is_blocked("not-an-ip"))


class TestValidateUrl(unittest.TestCase):
    def setUp(self):
        # Ensure the dev escape hatch is OFF for the secure-path tests.
        self._prev = os.environ.pop("COMFYLINK_ALLOW_INSECURE", None)

    def tearDown(self):
        if self._prev is not None:
            os.environ["COMFYLINK_ALLOW_INSECURE"] = self._prev
        else:
            os.environ.pop("COMFYLINK_ALLOW_INSECURE", None)

    def test_https_public_host_allowed(self):
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("93.184.216.34")):
            # Should not raise.
            _validate_url("https://bucket.r2.cloudflarestorage.com/obj?sig=x")

    def test_http_denied_by_default(self):
        with self.assertRaises(RelayError):
            _validate_url("http://example.com/obj")

    def test_non_http_scheme_denied(self):
        for url in ("file:///etc/passwd", "ftp://host/x", "gopher://host/"):
            with self.assertRaises(RelayError):
                _validate_url(url)

    def test_metadata_ip_denied(self):
        # https URL whose host resolves to the cloud metadata endpoint.
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("169.254.169.254")):
            with self.assertRaises(RelayError):
                _validate_url("https://metadata.attacker.example/latest")

    def test_loopback_host_denied(self):
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("127.0.0.1")):
            with self.assertRaises(RelayError):
                _validate_url("https://localhost/obj")

    def test_normal_host_with_private_resolved_ip_denied(self):
        # A perfectly normal-looking public hostname that (via DNS rebinding or
        # a poisoned record) resolves to a private address must be rejected.
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("10.1.2.3")):
            with self.assertRaises(RelayError):
                _validate_url("https://cdn.example.com/obj")

    def test_multi_record_one_private_denied(self):
        # If ANY resolved address is private, reject.
        def _multi(host, port, *a, **k):
            return [
                (0, 0, 0, "", ("93.184.216.34", port or 0)),
                (0, 0, 0, "", ("127.0.0.1", port or 0)),
            ]
        with mock.patch.object(relay.socket, "getaddrinfo", _multi):
            with self.assertRaises(RelayError):
                _validate_url("https://cdn.example.com/obj")

    def test_insecure_flag_allows_http_localhost(self):
        os.environ["COMFYLINK_ALLOW_INSECURE"] = "1"
        # No DNS check happens under the flag; should not raise.
        _validate_url("http://127.0.0.1:9000/fake-r2/obj")
        _validate_url("http://localhost:8188/view")

    def test_insecure_flag_still_rejects_non_http_scheme(self):
        os.environ["COMFYLINK_ALLOW_INSECURE"] = "1"
        with self.assertRaises(RelayError):
            _validate_url("file:///etc/passwd")


class TestStatePermissions(unittest.TestCase):
    def test_save_writes_0600(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "state.json")
        os.environ["COMFYLINK_STATE"] = path
        try:
            import importlib

            import comfylink.config as cfg
            importlib.reload(cfg)
            st = cfg.State.load()
            st.add_pairing("clr_secret", "d1")  # appends + saves the token file
            self.assertTrue(os.path.isfile(path))
            if os.name == "posix":
                mode = stat.S_IMODE(os.stat(path).st_mode)
                self.assertEqual(mode, 0o600, oct(mode))
            # Round-trips regardless of platform.
            again = cfg.State.load()
            self.assertEqual(again.pairings[0].device_token, "clr_secret")
        finally:
            del os.environ["COMFYLINK_STATE"]

    def test_save_tightens_preexisting_world_readable_file(self):
        if os.name != "posix":
            self.skipTest("permission bits are POSIX-only")
        d = tempfile.mkdtemp()
        path = os.path.join(d, "state.json")
        # Simulate a file left world-readable by the old code.
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}")
        os.chmod(path, 0o644)
        os.environ["COMFYLINK_STATE"] = path
        try:
            import importlib

            import comfylink.config as cfg
            importlib.reload(cfg)
            st = cfg.State.load()
            st.add_pairing("clr_secret", "d1")  # appends + saves the token file
            mode = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode, 0o600, oct(mode))
        finally:
            del os.environ["COMFYLINK_STATE"]


if __name__ == "__main__":
    unittest.main()
