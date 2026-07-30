# `ops/` — remote config the ComfyLink app reads at runtime

Three small static JSON files. The ComfyLink mobile app fetches them **anonymously
over `raw.githubusercontent.com`**, so editing one + pushing to `main` changes
behaviour for every installed app **with no app release and no relay redeploy**.
They live in this PUBLIC repo precisely because the app repo is private.

> ⚠️ **These files are the only live source.** There is no staging copy anywhere.
> If a doc or a sibling repo shows you "the same" JSON, that copy is not read by
> anything — change it and nothing happens, with no error. Change **these**.

| File | Drives | App-side reader |
|---|---|---|
| `status.json` | service-status banner **and** the in-app promo banner | `lib/core/status/` (`status_config.dart` holds the URL) |
| `versions.json` | update reminders / version floor for plugin + app | `lib/core/config/version_config.dart`; relay `GET /v1/versions` mirrors it as a fallback |
| `community.json` | feedback entry (Discord button + support email) | `lib/core/config/community_config.dart` |

Raw URLs (branch `main`):

```
https://raw.githubusercontent.com/huijiutian/ComfyUI-ComfyLink/main/ops/status.json
https://raw.githubusercontent.com/huijiutian/ComfyUI-ComfyLink/main/ops/versions.json
https://raw.githubusercontent.com/huijiutian/ComfyUI-ComfyLink/main/ops/community.json
```

The app can be pointed elsewhere per build with `--dart-define` (see the config
classes), but the shipped default is the URL above.

**Fail-safe by design**: a fetch error, timeout, non-200, malformed body, or a
missing field all degrade to *nothing shown*. These files can never break the
app — only add a notice.

---

## 1. `status.json`

Two independent features in one file: the outage banner (top-level fields) and
the promo banner (the `promo` block). **`promo.active` is not affected by the
top-level `active`, and vice versa.**

```json
{
  "active": false, "severity": "info", "message_en": "", "message_zh": "", "url": "",
  "promo": { "active": false, "message_en": "", "message_zh": "", "cta_en": "", "cta_zh": "" }
}
```

### Top level — outage banner

| field | type | meaning |
|---|---|---|
| `active` | bool | Show the banner only when **strictly** `true`. |
| `severity` | string | `info` \| `warn` \| `critical` → banner color/icon. Unknown → `info`. |
| `message_en` | string | English banner text. |
| `message_zh` | string | Chinese banner text. The app picks by locale, falling back to en. |
| `url` | string | Optional "learn more" link. Empty → no link. |

Dismissal is **session-only and keyed by content** — a persistent incident
reappears after an app restart, and editing the text/severity makes a
previously-dismissed banner reappear.

### `promo` — in-app promo banner

⚠️ **This is the promo channel.** Running a promotion means flipping
`promo.active` here — *not* enabling an introductory offer in the store back
office. (The old "banner follows the store's intro offer" code is gone.)

| field | type | meaning |
|---|---|---|
| `promo.active` | bool | Master switch, strictly `true` to show. |
| `promo.message_en` / `promo.message_zh` | string | Banner text, picked by locale. Empty for the current locale → not shown. |
| `promo.cta_en` / `promo.cta_zh` | string | Button label. Empty → the app's built-in default label. |

The banner also requires: the user is **signed in**, and is **not** already on a
paid tier. It auto-hides after ~5 s (or on manual dismiss) and is **not**
persisted — it shows again on the next cold start. Tapping it opens the
subscription page.

### Declare / clear an incident

1. Edit `ops/status.json`: `"active": true`, fill `message_en` + `message_zh`
   (optionally `severity` and a `url`).
2. `git commit && git push` to `main`.
3. Running apps pick it up on the next fetch: immediately on a cold start, or
   within ~10 min / on app-resume.

To clear: set `"active": false` (or empty the messages), commit, push.

Raw GitHub serves a CDN-cached copy — allow a minute or two to propagate. The app
sends `Cache-Control: no-cache` to minimise staleness.

Example of a live critical incident:

```json
{
  "active": true,
  "severity": "critical",
  "message_en": "Image generation is currently degraded. We are working on a fix.",
  "message_zh": "出图服务当前不稳定,我们正在修复。",
  "url": "https://comfylink.app/guide"
}
```

---

## 2. `versions.json`

Update reminders for both the plugin and the app. Field semantics
(`latest` / `min` / `remind`) are documented inline in the file's `_fields` block
and in the app repo's `docs/version-reminders.md` — **that doc is the contract,
don't restate it elsewhere.**

⛔ **Never raise `plugin.min` casually.** Released plugin 0.2.0 treats *any* 403
as "pairing revoked" and unpairs itself, and the version gate rejects with 403 —
so raising `min` unpairs exactly the users you were trying to notify. Keep it at
`0.0.0` unless you have verified the installed base. Bump `latest` only **after**
the new version is actually published and rolled out, never at build time.

The relay's `GET /v1/versions` serves a compiled-in fallback for clients that
cannot reach GitHub; keep the two in sync when you bump.

## 3. `community.json`

Feedback entry links. Empty `discord_url` hides the Discord button (the app then
offers the support email only) — fill it in and the button appears on the next
fetch. `email` is informational; the app has its own hardcoded fallback so a bad
value here cannot break the feedback entry.

---

## Before you push

If you are pushing as part of a plugin release, run `ops/scripts/plugin-preflight.sh`
(in the private ops repo): it warns when `status.json`'s `active` is `true` (i.e. an
outage banner is live network-wide) and checks `versions.json`'s `plugin.latest`
against the version in the code.
