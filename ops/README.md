# ops/status.json — ComfyLink app service-status banner

This file is the **live source** for the ComfyLink mobile app's service-status
banner. The app fetches it (raw GitHub) on startup and periodically; when
`active` is `true` it shows a dismissible banner on every screen — so the
operator can tell users "we know, we're fixing it" during an outage **without
shipping an app update**. It lives in this PUBLIC repo because the app reads it
anonymously over `raw.githubusercontent.com` (the app repo is private).

App URL: `https://raw.githubusercontent.com/huijiutian/ComfyUI-ComfyLink/main/ops/status.json`

## Declare an incident
1. Edit `ops/status.json`: set `"active": true`, fill `message_en` + `message_zh`
   (optionally `severity`: `info` | `warn` | `critical`, and a `url` for
   "learn more").
2. `git commit && git push` to `main`.
3. Running apps pick it up on next fetch (cold start, or app-resume / ~10 min).

## Clear it
Set `"active": false`, commit, push. The banner disappears on the next fetch.

Fail-safe: a fetch error, timeout, malformed body, or `active:false` all mean
**no banner** — this file can never break the app, only add a notice.
