# ComfyUI-ComfyLink

Drive your own ComfyUI from your phone with the **ComfyLink** app.

This custom node connects your local ComfyUI to ComfyLink so the mobile app can
run your workflows remotely — pick a workflow, tweak parameters, generate, and
the images come back to your phone. Your PC does the work; the app is the remote.

## How it works

- **Outbound only** — no port forwarding; works behind home NAT.
- **Pair once** with a one-time code from the app. The PC never stores your
  account password.
- **Images go to storage and download straight to your phone** — they don't
  pile up on the relay.

## Requirements

- A working ComfyUI install.
- The ComfyLink app, signed in.

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
2. In the **ComfyLink app**, tap **Pair a new PC** to get a one-time code.
3. Paste the code into the panel and click **Pair**.

That's it — the panel turns **Online** and your machine shows up in the app,
ready to generate. No config files to edit.

## Status & control

The panel shows whether this PC is **Not paired / Connecting / Online**, the
machine name, and the number of detected nodes. You can **Unpair** here anytime;
you can also unpair it from the app. Either way, access stops immediately.

## Notes

- **Inert until paired.** Before you pair, the node does nothing and connects
  nowhere.
- A small `comfylink_state.json` is written next to the node to remember the
  pairing; it's local and git-ignored — don't commit it.

## Version

The current version is tracked in `comfylink/version.py` (`__version__`) and
mirrored in `pyproject.toml`, which follows the
[Comfy Registry](https://registry.comfy.org) conventions for publishing. The
sidebar panel shows the running version.

## License

MIT
