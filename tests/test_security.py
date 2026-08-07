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
from comfylink.relay import (  # noqa: E402
    RelayError,
    _host_is_trusted,
    _ip_is_blocked,
    _validate_url,
)


def _fake_getaddrinfo(ip):
    """Return a getaddrinfo-shaped result resolving any host to `ip`."""
    def _inner(host, port, *a, **k):
        return [(0, 0, 0, "", (ip, port or 0))]
    return _inner


def _boom_getaddrinfo(host, port, *a, **k):
    """A getaddrinfo that must never be called (proves the trusted-host
    short-circuit ran BEFORE any DNS work)."""
    raise AssertionError(f"getaddrinfo should not have been called for {host}")


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

    def test_ipv4_mapped_private_and_metadata_blocked(self):
        for ip in ("::ffff:10.0.0.5", "::ffff:192.168.1.1", "::ffff:169.254.169.254",
                   "::ffff:0.0.0.0"):
            self.assertTrue(_ip_is_blocked(ip), ip)

    def test_ipv4_mapped_public_allowed_on_every_python(self):
        # ⭐ Regression for the pre-gh-113171 Pythons ComfyUI_windows_portable
        # embeds (3.10.11 / 3.11.6-3.11.8): there, `::ffff:<public v4>` reports
        # is_private=True because the whole ::ffff:0:0/96 block was listed as
        # private, so a perfectly public address was refused. We now classify by
        # the EMBEDDED v4 first, which makes the verdict version-independent.
        for ip in ("::ffff:8.8.8.8", "::ffff:93.184.216.34", "::ffff:1.1.1.1"):
            self.assertFalse(_ip_is_blocked(ip), ip)

    def test_proxy_fake_ip_and_nat64_still_blocked_by_ip_rule(self):
        # These are the ranges that misfired against our own R2 in production.
        # The IP rule itself does NOT change — they stay blocked; the fix is the
        # host allowlist below, which never consults this rule for our own host.
        self.assertTrue(_ip_is_blocked("198.18.0.1"))     # Clash/mihomo fake-ip
        self.assertTrue(_ip_is_blocked("64:ff9b::808:808"))  # NAT64

    def test_garbage_treated_as_blocked(self):
        # Fail closed: anything we can't parse as an IP is unsafe.
        self.assertTrue(_ip_is_blocked("not-an-ip"))


class TestHostIsTrusted(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.pop(relay._TRUSTED_HOSTS_ENV, None)

    def tearDown(self):
        if self._prev is not None:
            os.environ[relay._TRUSTED_HOSTS_ENV] = self._prev
        else:
            os.environ.pop(relay._TRUSTED_HOSTS_ENV, None)

    def test_relay_r2_endpoint_trusted(self):
        # The shape the relay actually signs: account-level R2 endpoint,
        # path-style bucket (relay internal/r2/r2.go).
        self.assertTrue(_host_is_trusted("abc123.r2.cloudflarestorage.com"))
        self.assertTrue(_host_is_trusted("ABC123.R2.CloudflareStorage.com"))
        self.assertTrue(_host_is_trusted("abc123.r2.cloudflarestorage.com."))

    def test_lookalike_hosts_not_trusted(self):
        for host in (
            "r2.cloudflarestorage.com.attacker.example",   # suffix in the middle
            "notr2.cloudflarestorage.com.evil.example",
            "r2.cloudflarestorage.com",                    # apex, never signed
            "metadata.google.internal",
            "localhost",
            "169.254.169.254",
            "",
        ):
            self.assertFalse(_host_is_trusted(host), host)

    def test_env_adds_suffixes(self):
        os.environ[relay._TRUSTED_HOSTS_ENV] = " storage.example.com , "
        self.assertTrue(_host_is_trusted("acct.storage.example.com"))
        # Normalised to a leading dot => can't match a neighbouring domain.
        self.assertFalse(_host_is_trusted("notstorage.example.com"))
        self.assertFalse(_host_is_trusted("storage.example.com.evil.test"))

    def test_env_empty_is_a_no_op(self):
        os.environ[relay._TRUSTED_HOSTS_ENV] = " , ,, "
        self.assertFalse(_host_is_trusted("cdn.example.com"))
        self.assertTrue(_host_is_trusted("acct.r2.cloudflarestorage.com"))


class TestValidateUrl(unittest.TestCase):
    def setUp(self):
        # Ensure the dev escape hatch is OFF for the secure-path tests.
        self._prev = os.environ.pop("COMFYLINK_ALLOW_INSECURE", None)
        self._prev_trusted = os.environ.pop(relay._TRUSTED_HOSTS_ENV, None)

    def tearDown(self):
        if self._prev is not None:
            os.environ["COMFYLINK_ALLOW_INSECURE"] = self._prev
        else:
            os.environ.pop("COMFYLINK_ALLOW_INSECURE", None)
        if self._prev_trusted is not None:
            os.environ[relay._TRUSTED_HOSTS_ENV] = self._prev_trusted
        else:
            os.environ.pop(relay._TRUSTED_HOSTS_ENV, None)

    def test_https_public_host_allowed(self):
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("93.184.216.34")):
            # Should not raise.
            _validate_url("https://cdn.example.com/obj?sig=x")

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

    # ── The trusted-host bypass (2026-08-07 production fix) ──────────────────
    #
    # Each of these resolutions is a REAL failure mode observed on users'
    # machines: the render finished on disk, but the presigned PUT was refused
    # locally and the job was reported failed for nothing.

    def test_relay_r2_host_skips_ip_check_entirely(self):
        # getaddrinfo must not even be reached — no DNS, no verdict to misfire.
        with mock.patch.object(relay.socket, "getaddrinfo", _boom_getaddrinfo):
            _validate_url("https://acct.r2.cloudflarestorage.com/bucket/k?sig=x")

    def test_relay_r2_host_allowed_behind_proxy_fake_ip(self):
        # Clash / mihomo / sing-box fake-ip pool (198.18.0.0/15) — the #1 cause.
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("198.18.0.42")):
            _validate_url("https://acct.r2.cloudflarestorage.com/bucket/k?sig=x")

    def test_relay_r2_host_allowed_behind_nat64(self):
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("64:ff9b::b2b8:d822")):
            _validate_url("https://acct.r2.cloudflarestorage.com/bucket/k?sig=x")

    def test_relay_r2_host_allowed_with_poisoned_dns(self):
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("0.0.0.0")):
            _validate_url("https://acct.r2.cloudflarestorage.com/bucket/k?sig=x")

    def test_relay_r2_host_allowed_when_dns_fails_outright(self):
        # A proxy that answers nothing locally still lets the connection through.
        def _gaierror(host, port, *a, **k):
            raise relay.socket.gaierror("nodename nor servname provided")
        with mock.patch.object(relay.socket, "getaddrinfo", _gaierror):
            _validate_url("https://acct.r2.cloudflarestorage.com/bucket/k?sig=x")

    def test_trusted_host_still_requires_https(self):
        # ⛔ The allowlist NEVER relaxes the scheme — a presigned URL's signature
        # must not go out in plaintext, whatever the host.
        with self.assertRaises(RelayError):
            _validate_url("http://acct.r2.cloudflarestorage.com/bucket/k?sig=x")

    def test_allowlist_does_not_weaken_the_metadata_defence(self):
        # The whole point: everything NOT on the allowlist is judged exactly as
        # before, including a host that merely *contains* the trusted suffix.
        for host in ("169.254.169.254", "metadata.google.internal",
                     "r2.cloudflarestorage.com.attacker.example"):
            with mock.patch.object(relay.socket, "getaddrinfo",
                                   _fake_getaddrinfo("169.254.169.254")):
                with self.assertRaises(RelayError):
                    _validate_url(f"https://{host}/latest/meta-data/")

    def test_env_trusted_host_skips_ip_check(self):
        os.environ[relay._TRUSTED_HOSTS_ENV] = "storage.example.com"
        with mock.patch.object(relay.socket, "getaddrinfo", _boom_getaddrinfo):
            _validate_url("https://acct.storage.example.com/bucket/k?sig=x")

    def test_unknown_host_falls_back_to_ip_check(self):
        # An entry that doesn't match changes nothing — the guard is ADDITIVE,
        # so a wrong/typo'd suffix can never block an upload that works today.
        os.environ[relay._TRUSTED_HOSTS_ENV] = "typo.example.net"
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("93.184.216.34")):
            _validate_url("https://cdn.example.com/obj")
        with mock.patch.object(relay.socket, "getaddrinfo",
                               _fake_getaddrinfo("10.1.2.3")):
            with self.assertRaises(RelayError):
                _validate_url("https://cdn.example.com/obj")


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
