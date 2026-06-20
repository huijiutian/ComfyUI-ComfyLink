// ComfyLink — workflow catalog sync driver.
//
// The browser extension is the ONLY place that can convert ComfyUI's saved
// UI-graph workflows (nodes/links) into the API prompt format the App needs
// (graphToPrompt). So the browser drives sync: enumerate saved workflows,
// convert the new/changed ones offscreen, assemble a manifest, and POST both
// manifest + blobs to the Python plugin (POST /comfylink/sync). Python holds
// the device token and pushes everything to R2 — the browser NEVER sees the
// token.
//
// Contract: app/docs/workflow-sync.md (manifest schema lines 17-34, web
// extension behaviour lines 59-66).
//
// VERIFIED ComfyUI frontend API shapes (confirmed from official
// Comfy-Org/ComfyUI_frontend source on 2026-06-19):
//   - app.api.listUserDataFullInfo(dir) -> [{path,size,modified}], where
//     `path` is RELATIVE to `dir` (e.g. "subdir/My Workflow.json"). It does
//     GET /userdata?dir=...&recurse=true&split=false&full_info=true.
//   - app.api.getUserData(file) -> Promise<Response>, GET
//     /userdata/${encodeURIComponent(file)}. `file` is relative to the user
//     data root, so for workflows the path must be joined as
//     "workflows/" + path.  [NEEDS-VERIFICATION on a real ComfyUI: exact join
//     behaviour for the getUserData path — see report.]
//   - app.graphToPrompt(graph = this.rootGraph) -> {workflow, output} where
//     `output` is the API prompt { "<nodeId>": {class_type, inputs, _meta} }.
//     It accepts a graph argument, enabling offscreen conversion.
//   - LiteGraph is a global (window.LiteGraph). Offscreen conversion:
//       const g = new LiteGraph.LGraph(); g.configure(uiGraphJson);
//       const { output } = await app.graphToPrompt(g);
//
// Prefer `app.api`, fall back to the standalone `api` module.
import { app } from "../../scripts/app.js";
import { api as apiFallback } from "../../scripts/api.js";

// Single localStorage key for the last successfully-synced manifest. backendId
// isn't known in the browser; the browser is the single writer for v1, so a
// fixed key is acceptable (contract line 84).
const LAST_MANIFEST_KEY = "comfylink.lastManifest";

// Coalesce window for scheduleSync() (ms).
const DEBOUNCE_MS = 1500;
// Lightweight background rescan interval (ms).
const RESCAN_MS = 60000;

// ---- small helpers -------------------------------------------------------

function getApi() {
  // VERIFIED: api client is reachable as app.api; apiFallback is the
  // standalone scripts/api.js singleton.
  return app?.api ?? apiFallback;
}

function basename(path) {
  const i = path.lastIndexOf("/");
  return i >= 0 ? path.slice(i + 1) : path;
}

function nameOf(path) {
  // Display name = file basename without the trailing ".json".
  const b = basename(path);
  return b.toLowerCase().endsWith(".json") ? b.slice(0, -5) : b;
}

function fingerprintOf(entry) {
  // fingerprint = "<size>:<modified>" (contract line 27).
  return `${entry.size}:${entry.modified}`;
}

// workflowId = stable hex hash of the relative path. SHA-256 of the UTF-8
// path, hex digest, first 32 hex chars (16 bytes). The relay/App MUST agree
// on this hash(path)->hex mapping (contract line 15).
async function workflowId(path) {
  const bytes = new TextEncoder().encode(path);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const hex = Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return hex.slice(0, 32);
}

function loadLastManifest() {
  try {
    const raw = localStorage.getItem(LAST_MANIFEST_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

function saveLastManifest(manifest) {
  try {
    localStorage.setItem(LAST_MANIFEST_KEY, JSON.stringify(manifest));
  } catch (e) {
    console.warn("[ComfyLink] failed to persist last manifest", e);
  }
}

// Build id -> previous manifest entry, for diffing + carry-over of unchanged.
function indexById(manifest) {
  const map = new Map();
  if (manifest && Array.isArray(manifest.workflows)) {
    for (const w of manifest.workflows) {
      if (w && w.id) map.set(w.id, w);
    }
  }
  return map;
}

// ---- conversion ----------------------------------------------------------

// Convert one UI graph JSON to the API prompt.
// Primary path = offscreen LGraph: build a throwaway LGraph, configure it with
// the saved UI graph, and graphToPrompt(thatGraph). This does NOT disturb the
// user's live canvas. [NEEDS-VERIFICATION on a real ComfyUI: that offscreen
// graphToPrompt produces correct, non-empty output across frontend builds.]
//
// Fallback path = load/convert/restore on the LIVE graph: snapshot the current
// graph, app.loadGraphData(ui), graphToPrompt(), then RESTORE the snapshot in
// a finally so the user's canvas is always put back. Used ONLY if the offscreen
// path throws or yields empty output. [NEEDS-VERIFICATION: whether the fallback
// is ever required, and that loadGraphData round-trips cleanly.]
async function convertUiToApi(ui) {
  // --- primary: offscreen ---
  try {
    const LG = (typeof LiteGraph !== "undefined" && LiteGraph) || window.LiteGraph;
    if (LG && typeof LG.LGraph === "function") {
      const g = new LG.LGraph();
      g.configure(ui);
      const { output } = await app.graphToPrompt(g);
      if (output && Object.keys(output).length > 0) {
        return output;
      }
      // empty output -> fall through to fallback
    }
  } catch (e) {
    // swallow and try fallback
    console.warn("[ComfyLink] offscreen conversion failed, trying fallback", e);
  }

  // --- fallback: load / convert / restore on the live graph ---
  // Snapshot first so finally can always restore, even if conversion throws.
  let snapshot = null;
  try {
    snapshot = await app.graphToPrompt();
  } catch (e) {
    snapshot = null;
  }
  try {
    app.loadGraphData(ui);
    const { output } = await app.graphToPrompt();
    if (!output || Object.keys(output).length === 0) {
      throw new Error("empty conversion output");
    }
    return output;
  } finally {
    // Restore the user's canvas no matter what. snapshot.workflow is the UI
    // graph form that loadGraphData expects.
    try {
      if (snapshot && snapshot.workflow) {
        app.loadGraphData(snapshot.workflow);
      }
    } catch (e) {
      console.warn("[ComfyLink] failed to restore canvas after fallback", e);
    }
  }
}

// ---- core sync -----------------------------------------------------------

// Returns true if the environment can sync; logs a warn + returns false if a
// required API is missing (never throws out of the extension).
function capabilitiesOk() {
  const a = getApi();
  if (!a || typeof a.listUserDataFullInfo !== "function" || typeof a.getUserData !== "function") {
    console.warn("[ComfyLink] userdata API unavailable; skipping workflow sync.");
    return false;
  }
  if (typeof app.graphToPrompt !== "function") {
    console.warn("[ComfyLink] app.graphToPrompt unavailable; skipping workflow sync.");
    return false;
  }
  return true;
}

async function isPaired() {
  try {
    const r = await fetch(`/comfylink/status?_=${Date.now()}`, { cache: "no-store" });
    const s = await r.json();
    return !!s.paired;
  } catch (e) {
    return false;
  }
}

async function postSync(manifest, blobs) {
  const r = await fetch("/comfylink/sync", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ manifest, blobs }),
  });
  // Python returns { ok, uploaded } or { ok:false, error }.
  return r.json();
}

// One full sync pass. Returns silently on any guard miss; never throws.
async function runSync() {
  try {
    if (!(await isPaired())) return; // not paired -> skip (contract line 60)
    if (!capabilitiesOk()) return;

    const a = getApi();

    // 1. Enumerate ALL saved workflows (no filtering). path is relative to
    //    the "workflows" dir.
    let entries;
    try {
      entries = await a.listUserDataFullInfo("workflows");
    } catch (e) {
      console.warn("[ComfyLink] listUserDataFullInfo failed", e);
      return;
    }
    if (!Array.isArray(entries)) return;

    // 2. Diff vs last sync.
    const prev = loadLastManifest();
    const prevById = indexById(prev);

    const workflows = []; // manifest entries (ALL current workflows)
    const blobs = {}; // only NEW/CHANGED ready items get a blob

    for (const entry of entries) {
      const path = entry && entry.path;
      if (typeof path !== "string" || !path) continue;

      const id = await workflowId(path);
      const fingerprint = fingerprintOf(entry);
      const name = nameOf(path);
      const previous = prevById.get(id);

      // Unchanged: same fingerprint as last sync -> carry over the previous
      // manifest entry (status/name/node_count/error) and do NOT re-upload its
      // blob. Note: if it was previously status:"error", it stays error.
      if (previous && previous.fingerprint === fingerprint) {
        workflows.push({
          id,
          path,
          name: previous.name ?? name,
          fingerprint,
          status: previous.status === "error" ? "error" : "ready",
          ...(previous.error ? { error: previous.error } : {}),
          ...(previous.node_count != null ? { node_count: previous.node_count } : {}),
        });
        continue;
      }

      // New or changed: fetch + convert.
      let ui;
      try {
        // VERIFIED: getUserData path is relative to userdata root; join with
        // "workflows/". [NEEDS-VERIFICATION on real ComfyUI — see header.]
        const resp = await a.getUserData("workflows/" + path);
        if (!resp || !resp.ok) {
          workflows.push({ id, path, name, fingerprint, status: "error", error: "failed to read file" });
          continue;
        }
        ui = await resp.json();
      } catch (e) {
        workflows.push({ id, path, name, fingerprint, status: "error", error: "failed to read file" });
        continue;
      }

      try {
        const output = await convertUiToApi(ui);
        blobs[id] = output;
        workflows.push({
          id,
          path,
          name,
          fingerprint,
          status: "ready",
          node_count: Object.keys(output).length,
        });
      } catch (e) {
        // both primary + fallback failed -> error, not in blobs.
        workflows.push({
          id,
          path,
          name,
          fingerprint,
          status: "error",
          error: String((e && e.message) || e),
        });
      }
    }

    // 3. Assemble manifest (schema: contract lines 17-34). Deleted files
    //    simply don't appear in `workflows`.
    const manifest = {
      version: 1,
      updated_at: new Date().toISOString(),
      workflows,
    };

    // 4. POST to Python. Only overwrite the last manifest on success, so a
    //    failed sync retries new/changed items next time.
    const res = await postSync(manifest, blobs);
    if (res && res.ok) {
      saveLastManifest(manifest);
    } else {
      console.warn("[ComfyLink] sync upload failed", res && res.error);
    }
  } catch (e) {
    // Never let sync throw out of the extension.
    console.warn("[ComfyLink] workflow sync error", e);
  }
}

// ---- scheduling (debounce + single-flight) -------------------------------

let timer = null;
let running = false;
let queued = false;

// Coalesce calls within DEBOUNCE_MS and ensure only one sync runs at a time.
// If a trigger fires while a sync is running, queue exactly one re-run.
export function scheduleSync() {
  if (timer) clearTimeout(timer);
  timer = setTimeout(async () => {
    timer = null;
    if (running) {
      queued = true; // run once more after the current pass
      return;
    }
    running = true;
    try {
      await runSync();
    } finally {
      running = false;
      if (queued) {
        queued = false;
        scheduleSync();
      }
    }
  }, DEBOUNCE_MS);
}

// ---- triggers ------------------------------------------------------------

let installed = false;

// Wire up all sync triggers once. Safe to call multiple times.
//   - window focus           -> debounced rescan
//   - light interval (60s)   -> debounced rescan
//   - ComfyUI save event     -> debounced rescan [event name NEEDS-VERIFICATION]
// The focus + interval + post-pair triggers guarantee freshness regardless of
// whether the save event name is correct.
export function initSync() {
  if (installed) {
    // already installed; still kick a sync (e.g. panel re-opened).
    scheduleSync();
    return;
  }
  installed = true;

  // Background freshness.
  window.addEventListener("focus", scheduleSync);
  setInterval(scheduleSync, RESCAN_MS);

  // ComfyUI fires events on the api EventTarget. The exact "saved a workflow"
  // event name is uncertain across frontend versions, so we best-effort
  // subscribe to a few likely names. [NEEDS-VERIFICATION on real ComfyUI: the
  // actual save-workflow event name.] The focus/interval/post-pair triggers
  // cover us if none of these fire.
  const a = getApi();
  if (a && typeof a.addEventListener === "function") {
    for (const ev of ["graphChanged", "workflow_saved", "saved", "save"]) {
      try {
        a.addEventListener(ev, scheduleSync);
      } catch (e) {
        /* ignore unknown event names */
      }
    }
  }

  // Initial pass (will no-op if not paired / capabilities missing).
  scheduleSync();
}

// Call right after a successful pair so the first catalog lands immediately.
export function syncAfterPair() {
  scheduleSync();
}
