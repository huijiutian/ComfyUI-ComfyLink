"""Async client for the comfylink-relay HTTP API."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import socket
from typing import Optional
from urllib.parse import urlsplit

import aiohttp

from .version import __caps__, __commit__, __version__

log = logging.getLogger("comfylink.relay")

# Claim is a server-held long-poll (~28s); allow margin over it.
CLAIM_TIMEOUT = aiohttp.ClientTimeout(total=45)

# ── 空闲退避:中继建议「下次 claim 之前先睡一会儿」 ──────────────────────────
#
# 中继在 claim 的 204 响应头里捎带这个建议(只在「这台机器没活儿 + 这个账号的 App
# 也很久没露面」时才给)。⭐ 之所以让**中继**下发秒数而不是插件自己写死:插件推不
# 动更新,写死在这里的节奏发出去就再也调不动了;放在中继侧,改个环境变量就生效。
IDLE_SLEEP_HEADER = "X-ComfyLink-Idle-Sleep"

# ⛔ 本地保险丝。中继侧自己有上界,但那是**另一个进程里的假设**:万一它下发一个
# 离谱的值,用户的机器就那么久不干活了。⇒ 上界留一份在自己这边。
#
# ⚠️ **两个用途要两个上限,别共用一个** —— 它们约束的是完全不同的东西:
#
#   MAX_IDLE_SLEEP  claim 之间的那一觉 ⇒ 决定「一个新任务最坏多久才开始跑」。
#                   取「用户点了生成能忍多久」的量级,必须紧。
#   MAX_IDLE_BEAT   心跳间隔 ⇒ 决定「下行信号最坏多久到」。**有叫醒线兜着**
#                   (用户一回来 claim 会把心跳叫起来,见 claim 的 wake_beat),
#                   所以可以松得多。
#
# ⭐ 这两个曾经是同一个常量,后果是**中继把心跳退避调深根本不生效**、还没有任何
# 症状(下发 180 被本地夹回 120)。tests/test_idle_backoff.py 有一条钉着它们不同。
MAX_IDLE_SLEEP = 120
MAX_IDLE_BEAT = 600
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=30)

# progress(wait=True) 也是服务端持有的长轮询:中继会把请求挂起最多 ~25s 等取消。
# 客户端超时必须**大于**那个窗口,否则长轮询会被我们自己掐断(每 30s 一次
# TimeoutError,取消也就等不到了)。只给这一个请求放宽,DEFAULT_TIMEOUT 保持 30s
# 不动 —— 其它调用都是普通的一来一回,给它们放宽只会让真故障拖更久才暴露。
PROGRESS_WAIT_TIMEOUT = aiohttp.ClientTimeout(total=45)

# FOREIGN_QUEUE_UNKNOWN 是心跳里 `foreign_queue_depth` 的**「读不到」哨兵**。
#
# ⛔ 它必须是一个**负数**,不能是 0,也不能是 null:0 是一个完全合法的深度
# (「这台 ComfyUI 上没有别人的活」),而 null 在中继的整数字段上会静默变成 0。把
# 「没人占」和「不知道」混成一件事,正是将来写调度时最贵的一种错 —— 它会让一台我们
# 其实什么都不知道的机器看起来像最闲的那台。
FOREIGN_QUEUE_UNKNOWN = -1


def _wire_queue_depth(v) -> int:
    """把队列深度归一成上线用的 int(读不到 ⇒ FOREIGN_QUEUE_UNKNOWN)。

    ⛔ 守的是那条血教训的边界(见 heartbeat 的 docstring):一个 float / None /
    字符串跑到中继的整数字段上,会让**整个请求体**解析失败,这台机器的心跳从此
    永久死掉。这里宁可把认不出的值降级成「不知道」,也绝不让它出线。
    """
    if isinstance(v, bool):      # bool 是 int 的子类,别让 True 变成深度 1
        return FOREIGN_QUEUE_UNKNOWN
    try:
        n = int(v)
    except (TypeError, ValueError):
        return FOREIGN_QUEUE_UNKNOWN
    return n if n >= 0 else FOREIGN_QUEUE_UNKNOWN


# ── Delivery retry (ride a short relay/network outage, e.g. a relay redeploy) ──
# Only the DELIVERY-critical, one-shot calls (result / sign_upload / put_object)
# use this: a job whose render finishes during the deploy window would otherwise
# lose its result (the rendered image is safe on disk, but never reaches the app,
# so the user must regenerate). We retry ONLY infrastructure blips — connection
# errors, timeouts, and 502/503/504 — with capped exponential backoff up to
# _RETRY_MAX_ELAPSED of cumulative wait. That budget sits comfortably under the
# relay reaper's 5-minute job-stale threshold, so a late-but-delivered result is
# never mistaken for a dead job. Permanent errors (4xx auth/revoke, SSRF, R2 4xx)
# propagate IMMEDIATELY so a real failure or an unpair is never masked as a retry.
#
# ⚠️ _RETRY_MAX_ELAPSED counts SLEEP, not wall clock: every attempt can burn up to
# DEFAULT_TIMEOUT (30s) of its own before failing, so an outage in which requests
# HANG takes longer in real time than this number suggests. The shapes a Render
# restart actually produces — connection refused, 502/503 from the edge — fail
# fast, so in practice the wall clock ≈ this budget.
_RETRY_MAX_ELAPSED = 90.0   # seconds of cumulative backoff before giving up
_RETRY_BASE = 1.0           # first backoff
_RETRY_CAP = 15.0           # per-attempt backoff ceiling
_RETRY_STATUSES = frozenset({502, 503, 504})  # transient proxy/relay; 4xx never retried

# Exception shapes that mean "the transport broke", i.e. exactly the thing a relay
# restart does to an in-flight request. Kept as an explicit tuple rather than a
# broad `aiohttp.ClientError` catch, because ClientError also covers PERMANENT
# programming/URL faults (InvalidURL, ClientResponseError from raise_for_status)
# that must never be retried for 90 seconds.
#
# ⭐ ClientPayloadError is here for a reason, and it is NOT a ClientConnectionError
# (verified against aiohttp 3.14: its base is ClientError directly). It is what
# aiohttp raises when the response BODY is cut short — precisely what happens when
# the relay accepts a request, starts writing, and then the process goes away
# mid-deploy. Before it was listed here, that one shape escaped _deliver with zero
# retries, so a job whose render had already finished lost its result to a restart
# that every OTHER shape of the same outage rode out fine. A truncated response
# also means we cannot know whether the write landed — which is safe here for the
# same reason the rest of _deliver is: result is terminal-guarded server-side and a
# presigned PUT is idempotent, so re-sending is at worst a no-op.
_RETRY_EXCEPTIONS = (
    aiohttp.ClientConnectionError,   # refused / reset / server disconnected / connect timeout
    aiohttp.ClientPayloadError,      # response body truncated mid-flight (see above)
    asyncio.TimeoutError,            # our own total timeout elapsed
)

# Cap on object-storage downloads (input images). Bounds memory so a malicious
# or buggy relay can't point get_object at a huge/endless body and OOM the host.
MAX_OBJECT_BYTES = 64 * 1024 * 1024  # 64 MiB
_DOWNLOAD_CHUNK = 256 * 1024


class RelayError(RuntimeError):
    """A non-2xx relay response (or a locally-refused request when status == 0).

    `code` carries the relay's machine-readable ``error_code`` when the error
    body was JSON and contained one (e.g. "plugin_too_old"); it is "" otherwise.
    `payload` is the parsed JSON error body (``{}`` when the body wasn't a JSON
    object) so callers can read companion fields such as ``min_version`` /
    ``update_url`` without re-reading the response. Both are keyword-only with
    safe defaults so every existing ``RelayError(msg, status)`` call site is
    unaffected.
    """

    def __init__(self, message: str, status: int = 0, *, code: str = "",
                 payload: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.payload = payload or {}


def _insecure_allowed() -> bool:
    """Dev/integration escape hatch.

    The throwaway integration harness runs a fake ComfyUI + fake R2 on
    127.0.0.1, which the SSRF block below would otherwise reject. Setting
    COMFYLINK_ALLOW_INSECURE=1 skips the private-IP check and allows http so
    that harness keeps working. OFF by default => production is locked down.
    """
    return os.environ.get("COMFYLINK_ALLOW_INSECURE", "").strip() in ("1", "true", "yes")


# ── Trusted object-storage hosts (see _host_is_trusted) ──────────────────────
# The relay signs EVERY presigned URL we are ever handed with one S3 client,
# built from its R2_ENDPOINT env var (relay: cmd/relay/main.go r2.New(...) ->
# internal/r2/r2.go, UsePathStyle) — so every URL's host is exactly that
# endpoint's host, which production sets to the account-level R2 endpoint
# `https://<accountid>.r2.cloudflarestorage.com` (relay deploy/env.example,
# deploy/DEPLOY.md). Hence the suffix below.
_TRUSTED_HOST_SUFFIXES = (".r2.cloudflarestorage.com",)

# Escape hatch for the day the relay repoints R2_ENDPOINT at a custom S3 domain:
# comma-separated host suffixes, e.g. "storage.example.com". Additive only — an
# unknown/typo'd entry simply doesn't match and the URL falls back to the
# resolved-IP block below, i.e. exactly today's behaviour. It can never widen
# anything beyond https URLs pointing at the named hosts.
_TRUSTED_HOSTS_ENV = "COMFYLINK_TRUSTED_HOSTS"


def _trusted_suffixes() -> tuple[str, ...]:
    """Built-in trusted host suffixes plus any from COMFYLINK_TRUSTED_HOSTS.

    Read per call (not cached at import) so a user can set the env var and
    restart ComfyUI without any other ceremony. Entries are lower-cased and
    normalised to a leading dot so "example.com" can never match
    "notexample.com".
    """
    extra = []
    for raw in os.environ.get(_TRUSTED_HOSTS_ENV, "").split(","):
        h = raw.strip().lower().strip(".")
        if h:
            extra.append("." + h)
    return _TRUSTED_HOST_SUFFIXES + tuple(extra)


def _host_is_trusted(host: str) -> bool:
    """True if `host` is object storage the RELAY ITSELF signs URLs for.

    ⭐ Why this exists (2026-08-07, from production metrics): the resolved-IP
    block below rejects a URL when ANY address `getaddrinfo` returns looks
    non-public, and on a real user's machine that misfires against our own R2:

      * a local proxy/TUN (Clash / mihomo / sing-box) hands back a fake-ip from
        198.18.0.0/15 -> `is_private`, even though the connection works fine;
      * poisoned DNS answers 0.0.0.0 / 127.0.0.1;
      * a NAT64 AAAA record (64:ff9b::/96) -> `is_reserved`, which sinks a
        dual-stack machine even when its A record is perfectly public;
      * `::ffff:<public v4>` on a Python without the gh-113171 fix (every
        ComfyUI_windows_portable embed up to 3.11.8) -> `is_private`.
        (Neutralised at the root in _ip_is_blocked too, see there.)

    The user's render had already finished on their disk; only the RESULT
    UPLOAD was refused, so the job was reported failed for nothing.

    ⚠️ This is a NARROWING, not a widening. It only skips the resolved-IP block
    for https URLs whose host is the relay's own storage endpoint. The metadata
    endpoint (169.254.169.254), localhost and internal hostnames don't end in
    this suffix, so their defence is untouched. And a host under this suffix is
    by construction a public Cloudflare endpoint — it would have passed the IP
    check anyway on a machine with clean DNS, so nothing newly reachable is
    exposed here.
    """
    h = (host or "").strip().lower().rstrip(".")
    if not h:
        return False
    return any(h.endswith(sfx) for sfx in _trusted_suffixes())


def _ip_is_blocked(ip: str) -> bool:
    """True if `ip` is loopback/private/link-local/reserved — i.e. an SSRF
    target we must never fetch from (cloud metadata 169.254.169.254, localhost
    services, internal hosts). Pure + separately testable (no DNS)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        # Not parseable as an IP -> treat as unsafe rather than fail open.
        return True
    # IPv4-mapped IPv6 (::ffff:<v4>) — classify by the EMBEDDED v4, and do it
    # FIRST. This used to sit at the end of the boolean chain below, where it was
    # dead code (is_private already short-circuited) AND version-dependent:
    # before gh-113171 / CVE-2024-4032, Python listed the whole ::ffff:0:0/96
    # block in _private_networks, so ::ffff:<PUBLIC v4> came back is_private=True
    # and a perfectly good address was refused. ComfyUI_windows_portable ships
    # 3.10.11 / 3.11.6-3.11.8 — all of them pre-fix. Deciding on the embedded v4
    # up front makes the verdict identical on every Python version.
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        return _ip_is_blocked(str(mapped))
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _validate_url(url: str) -> None:
    """SSRF guard for relay-supplied (presigned R2) URLs.

    The relay hands us arbitrary GET/PUT URLs; a compromised relay or a crafted
    job payload could aim them at internal services or the cloud metadata
    endpoint. We require https and reject any URL whose host resolves to a
    private/loopback/link-local/reserved address.

    TWO gates, and the order matters. https is required ALWAYS — no host is ever
    trusted enough to send a presigned URL's signature over plaintext. Only then
    may a host on the relay's own storage endpoint (_host_is_trusted) skip the
    resolved-IP block, which on real users' machines rejected our own R2 (proxy
    fake-ip / NAT64 / poisoned DNS — see _host_is_trusted). Every other host is
    judged exactly as before. Raises RelayError on rejection.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname

    if _insecure_allowed():
        # Dev harness: allow http + localhost, skip the private-IP block.
        if scheme not in ("http", "https"):
            raise RelayError(f"refusing non-http(s) URL: {scheme or '<none>'}")
        return

    if scheme != "https":
        raise RelayError(f"refusing non-https URL (scheme {scheme or '<none>'})")
    if not host:
        raise RelayError("refusing URL with no host")

    # https + the relay's own object storage: the IP check can only hurt here
    # (it has no attack surface left to cover, and it demonstrably misfires).
    if _host_is_trusted(host):
        return

    # Resolve and check EVERY returned address (defeats DNS that returns both a
    # public and a private record, and IPv6/IPv4 splits).
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise RelayError(f"cannot resolve host {host!r}: {e}")
    for info in infos:
        ip = info[4][0]
        if _ip_is_blocked(ip):
            raise RelayError(f"refusing URL to non-public address ({host} -> {ip})")


async def redeem_pair_code(
    session: aiohttp.ClientSession, base_url: str, code: str, name: str
) -> tuple[str, str]:
    """Exchange a one-time pairing code for a device token. Unauthenticated.

    Returns (device_token, device_id). Raises RelayError on failure.
    """
    url = base_url.rstrip("/") + "/v1/devices/claim"
    async with session.post(url, json={"code": code, "name": name},
                            timeout=DEFAULT_TIMEOUT) as r:
        if r.status != 200:
            raise RelayError(f"pairing failed: {await r.text()}", r.status)
        d = await r.json()
    return d["device_token"], d.get("device_id", "")


class RelayClient:
    # `auth` is any object exposing `async token() -> str` (see TokenAuth).
    def __init__(self, session: aiohttp.ClientSession, base_url: str, auth):
        self._session = session
        self._base = base_url.rstrip("/")
        self._auth = auth

    async def _headers(self) -> dict:
        return {"Authorization": f"Bearer {await self._auth.token()}"}

    async def _deliver(self, what: str, thunk):
        """Run a delivery-critical call, retrying across a short relay/network
        outage. `thunk` is a zero-arg async factory (called fresh each attempt);
        its return value is passed straight back on success.

        Retries ONLY transient infrastructure failures — the transport shapes in
        _RETRY_EXCEPTIONS (connection errors, truncated response bodies, timeouts)
        and statuses 502/503/504 — with capped exponential backoff, up to
        _RETRY_MAX_ELAPSED of cumulative sleep. A non-transient RelayError (any
        other status, incl. 4xx auth/revoke and 0 = SSRF/validation) propagates
        immediately. Retrying is safe: the relay's result write is terminal-
        guarded (a re-POST is a benign no-op, no duplicate job_event) and a
        presigned sign/PUT is repeatable.
        """
        slept = 0.0
        delay = _RETRY_BASE
        attempt = 0
        while True:
            attempt += 1
            try:
                return await thunk()
            except _RETRY_EXCEPTIONS as e:
                err = e
            except RelayError as e:
                if e.status not in _RETRY_STATUSES:
                    raise
                err = e
            if slept >= _RETRY_MAX_ELAPSED:
                log.warning("%s: giving up after %d attempt(s): %s", what, attempt, err)
                raise err
            sleep_for = min(delay, _RETRY_CAP, _RETRY_MAX_ELAPSED - slept)
            log.warning("%s: transient failure (attempt %d), retrying in %.0fs: %s",
                        what, attempt, sleep_for, err)
            await asyncio.sleep(sleep_for)
            slept += sleep_for
            delay = min(delay * 2, _RETRY_CAP)

    async def register(self, backend_id: str, name: str) -> dict:
        # version/commit let the app tell the user their plugin is out of date.
        # Optional server-side, so older relays simply ignore them.
        # caps is the same deal (see version.__caps__): sent on BOTH register and
        # every heartbeat, unconditionally, so a git-pull + restart updates what
        # the app is allowed to offer without a re-pairing.
        return await self._json("POST", "/v1/backends/register",
                                {"backend_id": backend_id, "name": name,
                                 "version": __version__, "commit": __commit__,
                                 "caps": __caps__})

    async def sign_object_info(self, backend_id: str) -> tuple[str, str]:
        """Request a presigned PUT URL for this backend's object_info snapshot.

        Returns (key, url). The relay returns 503 if R2 isn't configured.
        """
        d = await self._json("POST", "/v1/backends/object-info/sign",
                             {"backend_id": backend_id})
        return d["key"], d["url"]

    async def upload_object_info(self, backend_id: str, object_info: dict) -> None:
        """Snapshot object_info to R2: sign a PUT, then upload the JSON to it.

        The blob no longer goes through the relay (egress fix) — we ship it
        straight to object storage via a presigned PUT. The SSRF guard in
        put_object passes for the https R2 URL the relay hands back.
        """
        _key, url = await self.sign_object_info(backend_id)
        data = json.dumps(object_info).encode()
        await self.put_object(url, data, "application/json")

    async def sign_put_loras(self, backend_id: str) -> tuple[str, str]:
        """Presigned PUT for this backend's LoRA inventory manifest.

        Same two-step shape as sign_object_info. A relay that predates this
        feature simply has no such route and answers 404 — the caller treats that
        as "this relay can't take an inventory yet" and moves on.
        """
        d = await self._json("POST", "/v1/backends/loras/sign-put",
                             {"backend_id": backend_id})
        return d["key"], d["url"]

    async def upload_loras(self, backend_id: str, manifest: dict) -> None:
        """Ship the LoRA inventory to R2: sign a PUT, then upload the JSON.

        ``retry=False`` on purpose. The delivery retry (see _deliver) exists so a
        FINISHED job's result survives a relay redeploy — losing it costs the
        user a regeneration. An inventory upload is not on that path: it is
        refreshed on a timer anyway, so burning 90s of backoff here would only
        keep a background task alive longer for no benefit.
        """
        _key, url = await self.sign_put_loras(backend_id)
        data = json.dumps(manifest).encode()
        await self.put_object(url, data, "application/json", retry=False)

    async def sign_put_workflow(self, backend_id: str, artifact: str,
                                workflow_id: Optional[str] = None) -> tuple[str, str]:
        """Request a presigned PUT URL for a workflow manifest or blob.

        artifact is "manifest" or "blob"; blob requires workflow_id. Returns
        (key, url). The relay returns 503 if R2 isn't configured.
        """
        body: dict = {"backend_id": backend_id, "artifact": artifact}
        if workflow_id is not None:
            body["workflow_id"] = workflow_id
        d = await self._json("POST", "/v1/backends/workflows/sign-put", body)
        return d["key"], d["url"]

    async def heartbeat(self, backend_id: str,
                        object_info_hash: str = "",
                        object_info_synced_at: float = 0.0,
                        foreign_queue_depth: int = FOREIGN_QUEUE_UNKNOWN,
                        loaded_checkpoint: str = "",
                        headers_out: Optional[dict] = None) -> dict:
        """Mark this backend alive. Returns the relay's response body.

        ``headers_out``(可选)收响应头,调用方用它读空闲退避的建议心跳间隔
        (``X-ComfyLink-Idle-Sleep``,见 _idle_sleep_hint)。不传 = 不关心。

        The body is the plugin's ONLY inbound channel while idle, so it doubles
        as the relay's way to ask for something. Today that is
        ``loras_requested_at`` (worker._maybe_scan_models) and
        ``object_info_requested_at`` (worker._maybe_refresh_object_info) — both
        are "the user pressed refresh in the app at this timestamp". An older
        relay returns ``{}`` and the caller simply finds nothing to do, so
        reading the body is always safe.

        Version/commit go out on every beat so a `git pull` + ComfyUI restart
        refreshes what the relay/app see without waiting for a re-register.

        RECEIPT for the snapshot upload. The snapshot itself goes straight to R2
        over a presigned PUT, so the relay never sees it land and cannot tell the
        app "the new one is up". The beat carries the CONTENT fingerprint of
        whatever we last shipped (Pairing.object_info_hash) and the relay stores
        it, which is what lets the app wait for a refresh to actually complete.

        A FINGERPRINT, not a timestamp, on purpose: re-uploading a byte-identical
        snapshot leaves it unchanged, so the app can never be fooled into
        reporting "refreshed" for a beat that moved nothing.

        Sent only when we HAVE one. Empty means "keep what you have" relay-side,
        and omitting the key says exactly that — a backend that has never
        captured a snapshot must not put anything in that column, least of all
        some other identifier that happened to be at hand.

        ``object_info_synced_at`` is the OTHER half of that receipt, and it
        answers a different question. The fingerprint says WHAT we have; this says
        WHICH REFRESH REQUEST we have already served (it echoes back the
        ``object_info_requested_at`` the relay sent us — see
        worker._refresh_object_info). Both are needed because "the refresh
        finished" and "the content changed" are NOT the same event: a machine
        whose node set did not change since the last capture uploads nothing and
        keeps the very same fingerprint, yet the user's refresh was still served.
        Reporting only the fingerprint left the app waiting for a change that was
        never going to come.

        Same "empty = keep what you have" rule as the hash: 0 (never served a
        request) omits the key rather than writing a zero over the relay's value.

        ⛔ IT GOES OUT AS AN ``int``, and that is not cosmetic. It is a unix
        SECOND — an integer by definition — but it is carried in
        ``Pairing.object_info_synced_at``, which is a Python ``float``, so
        ``json.dumps`` writes ``1785561497.0``. The relay's field is a Go
        ``int64``, and Go's json decoder treats a decimal point there as a HARD
        error for the WHOLE request body — not just that field. The observed
        production consequence (2026-08-01, on a user's machine):

            {"backend_id": "...", "object_info_synced_at": 1785561497}    -> 200
            {"backend_id": "...", "object_info_synced_at": 1785561497.0}  -> 400
                                                  {"error":"backend_id required"}

        i.e. the body failed to parse, every field came back zero-valued, and the
        relay answered with a message about a field we had in fact sent. The beat
        then failed FOREVER for that backend — no refresh signals, no last_seen —
        and only for backends that had ever served one refresh (0 omits the key,
        so a fresh pairing looked perfectly healthy). See kb 06-pitfalls.

        The relay now also tolerates the float shape, but the conversion stays
        HERE and is the real fix: the value is an integer, the wire format should
        say so, and plugins already installed on users' machines are the ones the
        relay has to be tolerant for.

        ``foreign_queue_depth`` / ``loaded_checkpoint`` are this machine's ComfyUI
        state (worker._ComfyProbe). See the comment on the body below for their
        wire contract; both are sent on EVERY beat, unconditionally.
        """
        # ⚠️ caps 和 version/commit 同规格:**每一拍都发,无条件**。中继每次心跳都
        # 用收到的值覆盖(空也写空),这样用户从新插件回退到老插件时能力会立刻消失;
        # 反过来说,这里一旦写成「有条件才带」,能力就会在 App 里闪断。见 version.__caps__。
        #
        # ⭐ foreign_queue_depth / loaded_checkpoint 走的是**同一条规矩**,而且理由
        # 更硬:中继每一拍都会用收到的值覆盖(读不到就覆盖成 NULL),所以这两个键必须
        # **无条件出现**。写成「取到了才带上」会让中继侧的值在两拍之间闪断/残留 ——
        # 一个残留下来的旧队列深度比没有值更坏,将来的调度会照着它派活。
        #
        # wire 契约(中继侧 api/backends.go 的 heartbeatReq 是同一份):
        #	foreign_queue_depth  int。**「别人占了这台机器多少」** —— 这台 ComfyUI
        #	                     队列里**不是我们提交的**那些 prompt 的条数。
        #	                     >=0 是真实条数(0 = 没有别人的活),**-1 = 读不到**。
        #	                     ⛔ 0 和「不知道」必须分得开:Go 那边 null 落到整数
        #	                     字段上就是 0,正好把两者混成一件事。
        #	                     ⚠️ 为什么不是队列**总深度**:中继自己就知道它派了
        #	                     多少活(它自己的 jobs 表里有),总深度里我们那部分
        #	                     对它信息量为零。它真正看不见的是用户自己占了多少。
        #	                     口径与近似程度见 worker._ComfyProbe。
        #	loaded_checkpoint    str。"" = 读不到 / 这台机器还没跑过东西。近似值,
        #	                     偏差见 jobs.last_checkpoint 上那段。
        #
        # ⛔ foreign_queue_depth 必须以**真正的 int** 出线,理由同上面那段水位线的血
        # 教训:一个 `3.0` 会让中继的整数字段解析**整个请求体**失败,这台机器的心跳
        # 从此永久死掉。_wire_queue_depth 就守在这个边界上。
        body = {"backend_id": backend_id,
                "version": __version__, "commit": __commit__,
                "caps": __caps__,
                "foreign_queue_depth": _wire_queue_depth(foreign_queue_depth),
                "loaded_checkpoint": str(loaded_checkpoint or "")}
        if object_info_hash:
            body["object_info_hash"] = object_info_hash
        if object_info_synced_at and object_info_synced_at > 0:
            # int() at the WIRE BOUNDARY, not at the call site: the value can come
            # straight out of comfylink_state.json, where every existing install
            # already has a float persisted. Normalising here covers those too.
            body["object_info_synced_at"] = int(object_info_synced_at)
        return await self._json("POST", "/v1/backends/heartbeat", body,
                                headers_out=headers_out)

    async def abandon_jobs(self, backend_id: str) -> int:
        """Fail every still-claimed/running job left on this backend.

        Called once right after the worker's first successful register on a fresh
        process: ComfyUI's queue is in-memory, so a just-started process is by
        definition running zero jobs. Any job the relay still has as
        claimed/running is therefore a zombie from a previous run that was killed
        mid-job (the relay's reaper can't catch it — we immediately re-register
        and heartbeat, so its 'job stale AND backend offline' double-check never
        holds). The relay marks them failed and returns how many it cleared.
        """
        d = await self._json(
            "POST", f"/v1/backends/{backend_id}/jobs/abandon", {}
        )
        return int(d.get("abandoned", 0))

    async def claim(self, backend_id: str,
                    stop: Optional[asyncio.Event] = None,
                    wake_beat: Optional[asyncio.Event] = None) -> Optional[dict]:
        """Long-poll for a job. Returns the job, or None on a 204 timeout.

        **空闲退避**:204 的响应头里可能带一个 ``X-ComfyLink-Idle-Sleep`` —— 中继
        在说「你这台机器没活儿,而且这个账号的 App 也很久没露面了,下一次 claim 之前
        先睡 N 秒」。没有这个头 = 没有建议 = 照旧立刻重来(老中继就是这样)。

        ⭐ **觉睡在这里,而不是甩给调用方**:claim 本来就是一次「等到有活儿为止」的
        长轮询,睡一会儿只是把这次等待延长 —— 调用方的循环形状一个字都不用改,也就
        不会有人漏掉它。

        ⛔ **必须等响应上下文退出之后再睡**:在 ``async with`` 里睡会一直占着这条
        连接(以及连接池的名额),白白把一次「省流量」变成一次「占资源」。

        ⛔ **中继给的值不无条件相信**:本地还有 ``MAX_IDLE_SLEEP`` 兜底。中继侧自己
        有上界,但那是**另一个进程里的假设** —— 万一它发疯下发一小时,用户的机器就
        一小时不领任务了。保险丝要留在自己这边。

        ``stop`` 传进来是为了让这一觉**可以被叫醒**:关插件的时候不该干等它睡完。

        ``wake_beat`` 是**反方向**的那条线:中继**不再**建议退避(用户回来了,或者这台
        机器有活儿了)时,把它置位,好让**可能正睡到几分钟一拍的心跳**立刻起来打一拍。
        ⭐ 心跳是下行信号(「刷新 LoRA / 重报 object_info」)的唯一通道,而它自己睡着
        时没人能叫它 —— 于是由 claim 代劳:claim 一直保持着较短的周期,用户一回来,
        它总是先知道的那一个。有了这条线,心跳才敢退得更深。
        """
        hint = 0
        async with self._session.get(
            self._base + "/v1/jobs/claim",
            params={"backend_id": backend_id},
            headers=await self._headers(),
            timeout=CLAIM_TIMEOUT,
        ) as r:
            if r.status != 204:
                await _check(r)
                job = await r.json()
                # ⭐ 领到活儿是「用户回来了」**最强**的信号,比下面那个「没有退避
                # 建议」还强 —— 漏掉它会留一个很难看的缝:空闲很久之后用户一打开
                # App 就提交,claim 直接返回 200,而心跳可能正睡在几分钟一拍上;
                # job 就在这个循环里内联跑,跑完之前不会再有 claim 去叫醒它。
                # ⛔ 而 UpdateProgress 只动 jobs 表、不刷 backends.last_seen ⇒
                # 出图期间 App 上这台机器会显示**离线**,恰是心跳退避那条
                # HasActiveJob 守卫要防的画面(守卫要等下一拍心跳才生效)。
                if wake_beat is not None:
                    wake_beat.set()
                return job
            hint = _idle_sleep_hint(r.headers)
        # ↑ 连接已归还,再睡。
        if hint:
            await _sleep_or_stop(hint, stop)
        elif wake_beat is not None:
            # 没有建议 = 该恢复常态了。⚠️ 这里**每一轮空闲 claim 都会置位**,所以
            # 「该不该被它叫醒」的判断留在心跳那边(只有它自己知道有没有在退避)——
            # 常态节奏下响应这个信号只会让心跳变密,那是反效果。
            wake_beat.set()
        return None

    async def progress(self, job_id: str, status: str, value: int, maximum: int,
                       wait: bool = False) -> dict:
        """Report progress. Returns {"cancel": bool}.

        ``wait=True`` 打开中继的「带取消等待的长轮询」:中继照常推进这个 job 的
        心跳(updated_at,reaper 据此判活),然后把请求挂起最多 ~25s,取消一被置位
        就立刻返回 —— 生成期的心跳因此同时就是取消通道,取消延迟从「最长一个心跳
        间隔」压到秒级。请求超时随之放宽到 PROGRESS_WAIT_TIMEOUT。

        ``wait`` 字段只在 True 时才写进 body:不带这个字段 = 老行为(中继立刻回),
        对老中继(不认识 wait)也天然兼容 —— 它会忽略这个字段照旧立刻返回。
        """
        body: dict = {"status": status, "progress": value, "max": maximum}
        if wait:
            body["wait"] = True
        return await self._json("POST", f"/v1/jobs/{job_id}/progress", body,
                                timeout=PROGRESS_WAIT_TIMEOUT if wait else None)

    async def result(self, job_id: str, status: str, images: list[dict], error: str = "",
                     error_code: str = "", total_bytes: int = 0) -> None:
        body: dict = {"status": status, "images": images, "error": error,
                      "total_bytes": total_bytes}
        if error_code:
            body["error_code"] = error_code
        # Delivery-critical: retry across a short relay outage so a finished job's
        # result isn't lost to a redeploy window. Idempotent (terminal-guarded).
        await self._deliver(
            f"result[{job_id}]",
            lambda: self._json("POST", f"/v1/jobs/{job_id}/result", body),
        )

    async def sign_upload(self, job_id: str, kind: str, filename: str, content_type: str) -> tuple[str, str]:
        """Request a presigned PUT URL. Returns (r2_key, url)."""
        d = await self._deliver(
            f"sign_upload[{job_id}]",
            lambda: self._json("POST", "/v1/uploads/sign",
                               {"job_id": job_id, "kind": kind, "filename": filename,
                                "content_type": content_type}),
        )
        return d["r2_key"], d["url"]

    async def put_object(self, url: str, data: bytes, content_type: str,
                         retry: bool = True) -> None:
        """Upload bytes to object storage via a presigned PUT URL (no auth header).

        ``retry`` defaults to True (the historical behaviour — every existing
        caller is on the delivery path). Pass False for background, re-runnable
        uploads such as the LoRA inventory, where riding out an outage buys
        nothing and only prolongs a best-effort task.
        """
        _validate_url(url)  # SSRF guard: relay-supplied URL (permanent — outside retry).

        async def _put():
            async with self._session.put(url, data=data,
                                         headers={"Content-Type": content_type}) as r:
                if r.status >= 300:
                    # Carry the status so a transient R2 5xx is retried while a 4xx
                    # (expired/forbidden presign) fails fast.
                    raise RelayError(f"storage PUT {r.status}: {await r.text()}", r.status)

        if not retry:
            await _put()
            return
        # A presigned PUT is idempotent (same key, overwrite), so retrying a blip
        # is safe. Upload is the other half of delivery — ride the outage here too.
        await self._deliver("put_object", _put)

    async def get_object(self, url: str, max_bytes: int = MAX_OBJECT_BYTES) -> bytes:
        _validate_url(url)  # SSRF guard: relay-supplied URL.
        async with self._session.get(url) as r:
            r.raise_for_status()
            # Reject up front if the server advertises an oversized body...
            clen = r.content_length
            if clen is not None and clen > max_bytes:
                raise RelayError(f"object too large: {clen} > {max_bytes} bytes")
            # ...and enforce while streaming in case Content-Length lies/absent.
            buf = bytearray()
            async for chunk in r.content.iter_chunked(_DOWNLOAD_CHUNK):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise RelayError(f"object exceeded {max_bytes} bytes")
            return bytes(buf)

    async def _json(self, method: str, path: str, body: dict,
                    timeout: Optional[aiohttp.ClientTimeout] = None,
                    headers_out: Optional[dict] = None) -> dict:
        """One JSON round-trip. ``timeout`` defaults to DEFAULT_TIMEOUT; only the
        server-held long-poll (progress wait=True) passes its own, longer one.

        ``headers_out`` 给调用方一个**显式**的口子去看响应头(心跳用它读空闲退避的
        建议)。⭐ 之所以是一个传进来的容器,而不是改返回类型、也不是往返回的 dict
        里塞私货:返回形状是很多调用方(和测试里的 mock)共用的契约,为了一个只有一
        处需要的东西去动它,代价落在所有人身上。
        """
        async with self._session.request(
            method, self._base + path, json=body,
            headers=await self._headers(), timeout=timeout or DEFAULT_TIMEOUT,
        ) as r:
            if headers_out is not None:
                headers_out.update(r.headers)
            await _check(r)
            if r.content_length == 0:
                return {}
            ctype = r.headers.get("Content-Type", "")
            return await r.json() if "json" in ctype else {}


def _error_payload(body: str) -> dict:
    """Best-effort parse of a relay error body into a dict. NEVER raises.

    A non-JSON body, an empty body, or JSON that isn't an object all degrade to
    ``{}`` (=> RelayError.code == ""). We deliberately swallow everything: this
    runs on the failure path, so a surprise here must not replace the real error
    with a parse error.
    """
    try:
        data = json.loads(body)
    except Exception:  # noqa: BLE001 - any parse failure => no structured info
        return {}
    return data if isinstance(data, dict) else {}


def _idle_sleep_hint(headers, cap: int = MAX_IDLE_SLEEP) -> int:
    """从响应头里读中继的退避建议,返回秒数;**任何说不准的情况一律返回 0**。

    ``cap`` 是本地保险丝,按用途给:claim 那一觉用默认的 MAX_IDLE_SLEEP,心跳间隔
    传 MAX_IDLE_BEAT(见那两个常量上面的注释 —— 共用一个会让中继调不动心跳)。

    0 = 没有建议 = 照旧立刻重来,也就是这个功能上线之前的行为。⛔ 这条「不确定就
    回 0」的纪律是整个机制的安全侧:少省一点流量只是钱,睡过头是用户点了生成却干等。

    老中继不发这个头 ⇒ 读不到 ⇒ 0 ⇒ 行为与改造前逐字节一致。
    """
    raw = ""
    try:
        raw = (headers.get(IDLE_SLEEP_HEADER) or "").strip()
        if not raw:
            # ⛔⛔ **HTTP 头名大小写不敏感,而 dict 敏感。** aiohttp 自己的 headers 是
            # CIMultiDict(不敏感),但心跳那条路把头 update 进了一个**普通 dict**,
            # 于是大小写就要命了 —— 而 Go 的 `w.Header().Set` 会把名字规范化成
            # `X-Comfylink-Idle-Sleep`(**`L` 变成小写 `l`**),和这里的常量并不逐字
            # 相同。⇒ 精确匹配不中时,退回一次大小写无关的扫描。
            # ⚠️ 这个坑不会有任何症状:退避只是"没生效",一切照常跑,流量一分不省。
            wanted = IDLE_SLEEP_HEADER.lower()
            for k, v in headers.items():
                if str(k).lower() == wanted:
                    raw = (v or "").strip()
                    break
    except Exception:  # noqa: BLE001 — headers 形状异常时不值得让 claim 挂掉
        return 0
    if not raw:
        return 0
    try:
        n = int(raw)
    except ValueError:
        return 0
    if n <= 0:
        return 0
    return min(n, cap)


async def _sleep_or_stop(seconds: int, stop: Optional[asyncio.Event],
                         wake: Optional[asyncio.Event] = None) -> None:
    """睡 ``seconds`` 秒,但 ``stop``(或 ``wake``)一被置位就立刻醒。

    * ``stop`` —— 关插件时不该干等它睡完。
    * ``wake`` —— 「有事了,别睡了」。心跳退避到几分钟一拍之后,唤醒它的通道就是
      claim(见 RelayClient.claim):用户一回来,claim 那边先知道,由它把心跳叫起来
      去取下行信号,而不是让用户干等一个退避周期。

    两者都为 None(测试/独立调用)时退化成普通 sleep。

    ⛔ **必须走 asyncio.sleep,不能用 wait_for(stop.wait(), timeout=…)。** 两者行为
    等价,但测试是靠 patch 掉 ``asyncio.sleep`` 来让这些循环飞转的(见
    tests/test_plugin_too_old.py 的 `patch.object(worker.asyncio, "sleep", …)`)——
    wait_for 的超时走的是事件循环的定时器,patch 不到它,于是整套测试会真的按秒睡,
    挂到超时。⭐ 这不只是测试的方便:**能被换掉的等待**是这类循环可测的前提。
    """
    events = [e for e in (stop, wake) if e is not None]
    if not events:
        await asyncio.sleep(seconds)
        return
    pending = {asyncio.ensure_future(asyncio.sleep(seconds))}
    pending |= {asyncio.ensure_future(e.wait()) for e in events}
    try:
        await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # 谁先醒都要把其余的收掉,否则会留下挂着的 task(以及一条
        # "Task was destroyed but it is pending" 的噪音)。
        for t in pending:
            t.cancel()


async def _check(r: aiohttp.ClientResponse) -> None:
    if r.status >= 300:
        # Read the body ONCE: it is both the human message and the source of the
        # machine-readable error_code (aiohttp won't let us read it twice).
        body = await r.text()
        payload = _error_payload(body)
        code = payload.get("error_code")
        raise RelayError(
            f"relay {r.request_info.method} {r.url.path} -> {r.status}: {body}",
            r.status,
            code=code if isinstance(code, str) else "",
            payload=payload,
        )
