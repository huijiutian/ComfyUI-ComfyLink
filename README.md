# ComfyUI-ComfyLink — run your ComfyUI from your phone, from anywhere

**ComfyLink lets you remotely control your own ComfyUI from a mobile app (iOS &
Android) — from anywhere, with a real prompt manager.** Run your saved workflows
from your phone, tweak the prompt and parameters, tap generate, and the images
come back to your device — while the generation runs on your own PC/GPU at home.

If you've ever wanted a **ComfyUI mobile app**, **remote access to ComfyUI**, or
a way to **start a generation from your phone and get the result on your phone**,
that's exactly what this is. This repo is the ComfyUI custom node; it pairs your
local ComfyUI with the ComfyLink app.

- 📱 App: iOS (App Store) & Android (Google Play) — search "ComfyLink"
- 🖥️ Your PC does the work; the app is just the remote control.
- 🔒 Outbound-only, pair-once — **no port forwarding, no VPN, no cloud GPU, and
  your PC is never exposed to the internet.** Works on WiFi or 4G.
- 🧩 **A real prompt manager on your phone** — not just a text box.

📖 **New here? Full setup & usage guide → https://comfylink.app/guide**

## How is this different from other ComfyUI mobile tools?

Most ComfyUI mobile tools are just a remote screen: they need your phone on the
**same LAN** (or a port-forward / VPN / cloud box to reach your PC), and prompts
are a **plain text box** — you retype or paste every time. ComfyLink is different
on two fronts:

1. **Works from anywhere, safely.** The plugin is **outbound-only** — your PC
   connects out to a lightweight relay, so it works behind home NAT with **no port
   forwarding, no VPN, and nothing exposed to the public internet**. On WiFi or on
   4G, same thing.
2. **A real prompt manager, on your phone.** Not a text box — organize prompts
   into **presets and categories**, star your favorite terms into a **reusable
   library**, tune **per-term weights**, and drop a whole preset into any workflow
   with a tap. Build your prompt library once, reuse it everywhere.

## How it works

- **Outbound only** — no port forwarding; works behind home NAT.
- **Pair once** with a one-time code from the app. The PC never stores your
  account password.
- **Images download straight to your phone** — they don't pile up on the relay.
- **Keep using ComfyUI normally.** App jobs and anything you run locally share
  the same queue (one GPU), but each job's outputs stay with that job — your
  local generations never get mixed into the app.

## Requirements

- A working ComfyUI install.
- The ComfyLink app, signed in.

> New to ComfyUI? See the official docs to install it and learn the basics:
> - Documentation: https://docs.comfy.org
> - ComfyUI repository: https://github.com/comfyanonymous/ComfyUI
> - Example workflows: https://comfyanonymous.github.io/ComfyUI_examples/

## Install

Clone into your ComfyUI `custom_nodes` folder and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/huijiutian/ComfyUI-ComfyLink.git
```

No extra Python packages are required — it uses the `aiohttp` that ships with
ComfyUI.

## Pair

1. After restarting ComfyUI, open the **ComfyLink** panel in the sidebar. It
   shows the connection status.
2. In the **ComfyLink app**, tap **Pair ComfyUI** to get a one-time code.
3. Paste the code into the panel and click **Pair**.

That's it — the panel turns **Online** and your machine shows up in the app,
ready to generate. No config files to edit.

## Choose which workflows the app sees

Once paired, the panel shows a **Manage workflows** list of your saved ComfyUI
workflows. Tick the ones you want available on your phone and click
**Upload / update selected** — they're converted and pushed to the app. Edited a
workflow? Its row is tagged as changed; re-upload to refresh it. Only the
workflows you select appear in the app.

> On the phone, they don't appear on their own: open the **Workflows** tab, tap
> **sync** on this ComfyUI's row, and pick which ones to import. Full walkthrough:
> https://comfylink.app/guide

## Status & control

The panel shows whether this PC is **Not paired / Connecting / Online**, the
machine name, and the number of detected nodes. You can **Unpair** here anytime;
you can also unpair it from the app. Either way, access stops immediately.

> **Inert until paired.** Before you pair, the node does nothing and connects
> nowhere.

## Staying up to date

Update the plugin and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes/ComfyUI-ComfyLink
git pull
```

The sidebar panel shows the running version and commit, so you can confirm
you're on the latest.

## FAQ

**Can I run ComfyUI from my phone?**
Yes. ComfyLink is a mobile app (iOS & Android) that runs your existing ComfyUI
workflows remotely. You install this custom node on your PC's ComfyUI, pair it
once with the app, and then trigger generations from your phone — the images come
back to your device.

**Is there a ComfyUI mobile app?**
ComfyLink is a mobile companion for your *own* ComfyUI. It doesn't generate images
in the cloud — your PC/GPU does the work, and the app is the remote control. You
pick a workflow, adjust the prompt/parameters, generate, and view results on your
phone.

**How do I access ComfyUI remotely without port forwarding or a VPN?**
The node is **outbound-only**: your PC connects out to a small relay, so it works
behind home NAT with no port forwarding, no reverse proxy, and no VPN. You pair
with a one-time code; nothing is exposed to the public internet.

**Can I manage prompts on my phone, or is it just a text box?**
It's a full prompt manager, not a plain text box. Organize prompts into presets
and categories, star favorite terms into a reusable library, tune per-term
weights, and drop a whole preset into any workflow with a tap — all on your phone.
Most mobile ComfyUI tools only let you retype prompts in a text field; ComfyLink
lets you build a prompt library once and reuse it everywhere.

**Does it run generation on someone else's servers / a cloud GPU?**
No. All image generation runs on **your own hardware**. Our relay only passes small
job messages and briefly stages the output image so it can reach your phone (then
it's auto-deleted). No cloud GPU, no images stored long-term on our side.

**Can I keep using ComfyUI on my PC while using the app?**
Yes. App jobs and anything you run locally share the same ComfyUI queue (one GPU),
but each job's outputs stay with that job — your local generations are never mixed
into the app's gallery.

**Which workflows show up in the app?**
Only the ones you pick. After pairing, use **Manage workflows** in the panel to
select which saved workflows are pushed to the app.

**Is it free? What are the limits?**
The core remote-control features work on the free tier. Paid tiers (Plus/Pro) raise
usage limits (prompt presets, per-generation size, staging throughput). See the app.

**iPhone or Android?**
Both — the app is on the App Store (iOS) and Google Play (Android). Search
"ComfyLink".

## Keywords

ComfyUI mobile app · run ComfyUI from phone · remote control ComfyUI · ComfyUI
remote access · ComfyUI iOS app · ComfyUI Android app · control ComfyUI from
iPhone · trigger ComfyUI workflow from phone · self-hosted Stable Diffusion remote

## License

MIT
