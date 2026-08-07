# ComfyUI-ComfyLink — run your ComfyUI from your phone, from anywhere

**ComfyLink lets you remotely control your own ComfyUI from a mobile app (iOS &
Android) — from anywhere, with a real prompt manager.** Run your saved workflows
from your phone, tweak the prompt and parameters, tap generate, and the results
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
on three fronts:

1. **Works from anywhere, safely.** The plugin is **outbound-only** — your PC
   connects out to a lightweight relay, so it works behind home NAT with **no port
   forwarding, no VPN, and nothing exposed to the public internet**. On WiFi or on
   4G, same thing.
2. **A real prompt manager, on your phone.** Not a text box — organize prompts
   into **presets and categories**, star your favorite terms into a **reusable
   library**, tune **per-term weights**, and drop a whole preset into any workflow
   with a tap. Build your prompt library once, reuse it everywhere.
3. **It knows what's installed on your machine.** On request, the plugin reports
   your **LoRA and checkpoint inventory** — including the trigger words model
   authors embed in their own files — so the app can offer real models and real
   trigger words instead of you typing filenames from memory.

## How it works

- **Outbound only** — no port forwarding; works behind home NAT.
- **Pair with a one-time code** from the app. The PC never stores your account
  password. More than one account can pair with the same ComfyUI.
- **Results are delivered to your phone** — images, and also animations and
  videos (`VHS_VideoCombine`, `SaveVideo`, animated WebP/GIF). Outputs are staged
  only briefly on the way to your device and are auto-cleared after ~2 days —
  they don't pile up on the relay.
- **Keep using ComfyUI normally.** App jobs and anything you run locally share
  the same queue (one GPU), but each job's outputs stay with that job — your
  local generations never get mixed into the app. Cancelling from the app is
  surgical: a job still queued is removed by name, and only a job that is
  actually running gets interrupted — **the plugin never fires a global
  interrupt that could kill the generation you started yourself**.

## Requirements

- A working ComfyUI install, recent enough to have the **new frontend sidebar**
  (the ComfyLink panel registers as a sidebar tab — without it, the panel won't
  appear and there is no other way to pair).
- **Python 3.10+**.
- The ComfyLink app, signed in.

> New to ComfyUI? See the official docs to install it and learn the basics:
> - Documentation: https://docs.comfy.org
> - ComfyUI repository: https://github.com/comfyanonymous/ComfyUI
> - Example workflows: https://comfyanonymous.github.io/ComfyUI_examples/

## Install

**With ComfyUI-Manager (easiest).** Open **Manager → Custom Nodes Manager**,
search for **ComfyLink**, click **Install**, then restart ComfyUI. Dependencies
are installed for you.

**Or clone it manually** into your ComfyUI `custom_nodes` folder and restart
ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/huijiutian/ComfyUI-ComfyLink.git
```

> Cloning tracks `main`, which is where development happens — you get changes
> before they're cut into a release. Installing through Manager keeps you on
> published versions.

### Dependencies

Both are normally already in your ComfyUI environment, and both are declared in
`requirements.txt` with loose `>=` bounds so pip won't downgrade what ComfyUI
ships:

- **`aiohttp`** — required. Used to talk to the relay.
- **`Pillow`** — optional but recommended. Only used for the app's *"convert
  outputs to WebP"* option, and for writing the prompt into the image (see
  [Round-trip](#round-trip-your-own-images-back-into-a-preset) below). Without a
  working Pillow the job still succeeds — you just get the original PNG instead
  of WebP. If Pillow is present but too old to write XMP, the panel says so.

## Pair

1. After restarting ComfyUI, open the **ComfyLink** panel in the sidebar. It
   shows the connection status.
2. In the **ComfyLink app**, tap **Pair ComfyUI** to get a one-time code.
3. Paste the code into the panel, optionally give this machine a **device name**,
   and click **Pair**.

That's it — the panel turns **Online** and your machine shows up in the app,
ready to generate. No config files to edit.

The pairing form stays available afterwards: **one ComfyUI can serve several
accounts** (yourself on two accounts, or a machine shared with someone you
trust). Each paired account is listed in the panel with its own **Unpair**
button, so you can revoke one without touching the others.

## Choose which workflows the app sees

Once paired, the panel shows a **Manage workflows** list of your saved ComfyUI
workflows. Tick the ones you want available on your phone, tick which **paired
accounts** to sync them to, then click **Upload / update selected** — they're
converted and pushed to the app. Each row shows when it was last uploaded, and
is tagged **changed** if you've edited the workflow since; re-upload to refresh
it. **Refresh list** re-reads your workflows from ComfyUI. Only the workflows you
select appear in the app, and uploading never deletes anything on the app side.

> On the phone, they don't appear on their own: open the **Workflows** tab, tap
> **sync** on this ComfyUI's row, and pick which ones to import. Full walkthrough:
> https://comfylink.app/guide

## Let the app see your models

The app can ask this machine two things, both **only when you tap refresh in the
app** — there is no background scanning:

- **Your model inventory.** The plugin lists your **LoRAs and checkpoints** and
  reads each file's embedded header: for LoRAs the trigger phrase the author put
  in the file plus the most frequent training tags; for checkpoints the model
  family (SDXL / Flux, eps vs v-pred, training resolution, and so on) so the app
  knows which prompt dialect fits. **No model file's content is ever read** — one
  `stat` plus the safetensors header, which is why a few hundred models scan
  almost instantly. Nothing is hashed and nothing is uploaded to a third party.
- **A fresh node/model snapshot** (`object_info`). This is where the app's model
  dropdowns come from, and it used to be captured only once at startup — so
  installing or deleting a model on your PC would leave the app choosing from a
  stale list and the job would fail. Now you can refresh it from the app, and
  the plugin reports back as soon as it's done.

Related: when a workflow references a model this machine no longer has, the
plugin parses ComfyUI's validation error and tells the app **which model is
missing by name**, instead of a wall of JSON.

## Prompts land in your ComfyUI log

Right before a job is submitted, the plugin prints the **final prompt text** to
your own ComfyUI console — per text input, labelled positive or negative, with
the node it belongs to. The roles are traced back from the sampler along the
graph, so prompts behind `FluxGuidance`, `ConditioningCombine` or text-concat
nodes are still located correctly.

This makes the phone-side prompt manager auditable: what you assembled on your
phone is printed on your machine, before the image exists. **It is logged
locally only and never sent to the relay.**

## Round-trip: your own images back into a preset

When WebP output is enabled, the plugin also writes the prompt into the image's
XMP as plain, directly readable fields (`comfylink:positive` /
`comfylink:negative`, plus a `comfylink:generator` marker) — alongside the full
workflow JSON. Drop one of your own images back into the app and *"build a preset
from this image"* reads the prompt straight off it, with no guessing about node
structure. If the prompt can't be determined, nothing is written rather than
something wrong.

## Status & control

The panel shows whether this PC is **Not paired / Connecting / Online** (plus
**· generating** while a job is on the GPU), the machine name, the number of
detected nodes, the list of paired accounts, and the running version and commit.
It also surfaces:

- a **soft update hint** when a newer plugin version is published;
- a **red banner** if this plugin is too old for the relay to accept jobs;
- an **amber warning** if Pillow on this machine can't write the prompt into
  WebP output.

You can **Unpair** any account here anytime; you can also unpair from the app.
Either way, access stops immediately.

> **Inert until paired.** Before you pair, the plugin takes no jobs and uploads
> nothing. (Opening the panel does make one small outbound request — a version
> check against the relay, cached for 30 minutes. It carries no credentials and
> none of your data.)

## Staying up to date

If you installed through ComfyUI-Manager, update it there. If you cloned it,
pull and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes/ComfyUI-ComfyLink
git pull
```

The sidebar panel shows the running version and commit, and hints when a newer
version is available, so you can confirm you're on the latest.

## FAQ

**Can I run ComfyUI from my phone?**
Yes. ComfyLink is a mobile app (iOS & Android) that runs your existing ComfyUI
workflows remotely. You install this custom node on your PC's ComfyUI, pair it
once with the app, and then trigger generations from your phone — the results come
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
job messages and briefly stages the output so it can reach your phone (auto-cleared
after ~2 days). No cloud GPU, no images kept long-term on our side.

**Can I keep using ComfyUI on my PC while using the app?**
Yes. App jobs and anything you run locally share the same ComfyUI queue (one GPU),
but each job's outputs stay with that job — your local generations are never mixed
into the app's gallery, and cancelling an app job never interrupts one of yours.

**Which workflows show up in the app?**
Only the ones you pick. After pairing, use **Manage workflows** in the panel to
select which saved workflows are pushed to the app, and to which paired accounts.

**Does it scan my model folders? What exactly gets sent?**
Only when you ask for it from the app — there is no background scanning. The
plugin sends **filenames, sizes and the metadata embedded in each safetensors
header** (trigger phrase, training tags, model family). It **never reads a model
file's content** and never hashes it.

**Can more than one person / account use the same ComfyUI?**
Yes. Pair each account from the panel; every account gets its own entry and its
own Unpair button. They share the machine's single GPU queue.

**Does it do video, or only images?**
Video and animation too — outputs from `VHS_VideoCombine`, native `SaveVideo`,
and animated WebP/GIF are delivered like any other result. Video is never
re-encoded to WebP.

**Is it free? What are the limits?**
The core remote-control features work on the free tier. Paid tiers (Plus/Pro) raise
usage limits (prompt presets, per-generation size, staging capacity). See the app
for the current numbers.

**iPhone or Android?**
Both — the app is on the App Store (iOS) and Google Play (Android). Search
"ComfyLink".

## Keywords

ComfyUI mobile app · run ComfyUI from phone · remote control ComfyUI · ComfyUI
remote access · ComfyUI iOS app · ComfyUI Android app · control ComfyUI from
iPhone · trigger ComfyUI workflow from phone · self-hosted Stable Diffusion remote

## License

MIT
