// ComfyLink — workflow catalog sync (manual, user-driven).
//
// The browser extension is the ONLY place that can convert ComfyUI's saved
// UI-graph workflows (nodes/links) into the API prompt format the App needs
// (graphToPrompt). So the browser drives sync: the user opens "Manage
// workflows", picks the ones to upload, and we convert the chosen ones on the
// spot, assemble a manifest, and POST both manifest + blobs to the Python
// plugin (POST /comfylink/sync). Python holds the device token and pushes
// everything to R2 — the browser NEVER sees the token.
//
// This is MANUAL by design. The previous version auto-synced ALL workflows in
// the background (focus / 60s interval / guessed save event / post-pair), which
// was unreliable (depended on the page being open + focused, the save event
// name was a guess, offscreen conversion is finicky) → users reported "my
// workflow updates never show up". Now upload is an explicit click: reliable,
// converted on the spot, failures visible immediately.
//
// IMPORTANT: the manifest we POST contains ONLY the workflows the user selected
// (not every workflow on disk). That is exactly what the App browses, so the
// App shows the user's chosen set. Already-uploaded items are default-checked in
// the UI so they stay in the manifest unless the user deliberately drops them.
//
// Contract: app/docs/workflow-sync.md (manifest schema lines 17-34).
//
// VERIFIED ComfyUI frontend API shapes (confirmed from official
// Comfy-Org/ComfyUI_frontend source on 2026-06-19):
//   - app.api.listUserDataFullInfo(dir) -> [{path,size,modified}], where
//     `path` is RELATIVE to `dir` (e.g. "subdir/My Workflow.json"). It does
//     GET /userdata?dir=...&recurse=true&split=false&full_info=true.
//   - app.api.getUserData(file) -> Promise<Response>, GET
//     /userdata/${encodeURIComponent(file)}. `file` is relative to the user
//     data root, so for workflows the path must be joined as
//     "workflows/" + path.
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

// localStorage key for the last successfully-uploaded manifest. One ComfyUI can
// now be paired to several accounts at once, and an upload pushes the SAME
// catalog to ALL of them — so the "uploaded" flags are a machine-level fact, not
// per-account. A single fixed key (the legacy unscoped one) holds it.
const LAST_MANIFEST_KEY = "comfylink.lastManifest";
function manifestKey() {
  return LAST_MANIFEST_KEY;
}

// Hard cap on how many workflows can be uploaded (anti-abuse). The plugin is the
// gate: the App mirrors the manifest 1:1, so blocking here keeps the synced set
// within the limit. Keep in sync with the App's AppLimits.workflowFloor.
const MAX_WORKFLOWS = 100;

// ---- small helpers -------------------------------------------------------

function getApi() {
  // api client is reachable as app.api; apiFallback is the standalone
  // scripts/api.js singleton.
  return app?.api ?? apiFallback;
}

function nameOf(path) {
  // Display name = the workflow's relative path minus the trailing ".json", so
  // workflows that share a filename across subdirs (sdxl/portrait.json vs
  // flux/portrait.json) stay distinguishable instead of both showing "portrait".
  return path.toLowerCase().endsWith(".json") ? path.slice(0, -5) : path;
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
    const raw = localStorage.getItem(manifestKey());
    if (!raw) return null;
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

function saveLastManifest(manifest) {
  try {
    localStorage.setItem(manifestKey(), JSON.stringify(manifest));
  } catch (e) {
    console.warn("[ComfyLink] failed to persist last manifest", e);
  }
}

// Map of uploaded workflow id -> { at, fingerprint } from the last manifest.
// Only successfully-uploaded (status "ready") entries count — an "error" entry
// was never really pushed. `at` = when it was uploaded; `fingerprint` lets the
// UI detect a workflow that changed on disk since its last upload.
function uploadedIndex(manifest) {
  const map = new Map();
  if (manifest && Array.isArray(manifest.workflows)) {
    for (const w of manifest.workflows) {
      if (w && w.id && w.status === "ready") {
        map.set(w.id, {
          at: w.uploaded_at || manifest.updated_at || "",
          fingerprint: w.fingerprint || "",
        });
      }
    }
  }
  return map;
}

// ---- conversion ----------------------------------------------------------

// Convert one UI graph JSON to the API prompt.
// Primary path = offscreen LGraph: build a throwaway LGraph, configure it with
// the saved UI graph, and graphToPrompt(thatGraph). This does NOT disturb the
// user's live canvas.
//
// Fallback path = load/convert/restore on the LIVE graph: snapshot the current
// graph, app.loadGraphData(ui), graphToPrompt(), then RESTORE the snapshot in
// a finally so the user's canvas is always put back. Used ONLY if the offscreen
// path throws or yields empty output.
export async function convertUiToApi(ui) {
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

// ---- capability / pairing / transport ------------------------------------

// Returns true if the environment can sync; logs a warn + returns false if a
// required API is missing.
function capabilitiesOk() {
  const a = getApi();
  if (!a || typeof a.listUserDataFullInfo !== "function" || typeof a.getUserData !== "function") {
    console.warn("[ComfyLink] userdata API unavailable; cannot sync workflows.");
    return false;
  }
  if (typeof app.graphToPrompt !== "function") {
    console.warn("[ComfyLink] app.graphToPrompt unavailable; cannot sync workflows.");
    return false;
  }
  return true;
}

// Returns { paired } from the local status endpoint. `paired` is true when this
// machine has at least one paired account. (Upload pushes to all of them.)
async function pairedStatus() {
  try {
    const r = await fetch(`/comfylink/status?_=${Date.now()}`, { cache: "no-store" });
    const s = await r.json();
    return { paired: !!s.paired };
  } catch (e) {
    return { paired: false };
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

// Enumerate the saved workflows on disk. `path` is relative to the "workflows"
// dir. Returns [] (never throws) if the API is missing or the call fails.
async function enumerateWorkflows() {
  if (!capabilitiesOk()) return [];
  const a = getApi();
  let entries;
  try {
    entries = await a.listUserDataFullInfo("workflows");
  } catch (e) {
    console.warn("[ComfyLink] listUserDataFullInfo failed", e);
    return [];
  }
  return Array.isArray(entries) ? entries : [];
}

// ---- public API (consumed by the panel UI) -------------------------------

// List the saved workflows for the management UI.
//   -> [{ path, name, fingerprint, uploaded, uploadedAt, changed }]
// `uploaded` = previously uploaded (by id = hash(path)); `uploadedAt` = ISO time
// of that upload; `changed` = uploaded but the file changed on disk since (its
// fingerprint differs) → the user should re-upload to refresh it. The UI default-
// checks uploaded ones. Sorted by path for stable display. Throws if the ComfyUI
// APIs are unavailable so the UI can surface a clear message.
export async function listWorkflows() {
  if (!capabilitiesOk()) {
    throw new Error("ComfyUI workflow APIs are unavailable");
  }
  const entries = await enumerateWorkflows();
  const uploaded = uploadedIndex(loadLastManifest());

  const out = [];
  for (const entry of entries) {
    const path = entry && entry.path;
    if (typeof path !== "string" || !path) continue;
    const id = await workflowId(path);
    const fp = fingerprintOf(entry);
    const up = uploaded.get(id);
    out.push({
      path,
      name: nameOf(path),
      fingerprint: fp,
      uploaded: !!up,
      uploadedAt: up ? up.at : "",
      changed: !!up && up.fingerprint !== "" && up.fingerprint !== fp,
    });
  }
  out.sort((x, y) => x.path.localeCompare(y.path));
  return out;
}

// Upload exactly the selected workflows.
//   paths: string[] of workflow-relative paths (from listWorkflows()).
//   -> { uploaded, errors: [{ path, error }] }
// For each selected path we read its saved UI graph, convert it on the spot,
// and collect a blob; conversion/read failures are recorded as status:"error"
// (no blob) and surfaced in `errors`. The manifest contains ONLY the selected
// items, so the App will browse exactly this set. On a successful POST the
// localStorage manifest is updated (drives the "uploaded" flag next time).
// Throws if not paired, capabilities are missing, or the POST itself fails.
export async function uploadSelected(paths) {
  if (!capabilitiesOk()) {
    throw new Error("ComfyUI workflow APIs are unavailable");
  }
  const { paired } = await pairedStatus();
  if (!paired) {
    throw new Error("This PC is not paired");
  }
  const selected = (Array.isArray(paths) ? paths : []).filter(
    (p) => typeof p === "string" && p
  );
  // Anti-abuse cap: block (don't upload) when over the limit. The App mirrors
  // the manifest 1:1, so this is where the 100-workflow ceiling is enforced.
  if (selected.length > MAX_WORKFLOWS) {
    throw new Error(
      `Over the limit: ${selected.length} selected, max ${MAX_WORKFLOWS}. Uncheck some and upload again.`
    );
  }
  const a = getApi();

  // Current fingerprints for every workflow on disk (so each manifest entry
  // carries the up-to-date <size>:<modified>).
  const byPath = new Map();
  for (const entry of await enumerateWorkflows()) {
    if (entry && typeof entry.path === "string") byPath.set(entry.path, entry);
  }

  const workflows = []; // manifest entries (ONLY the selected workflows)
  const blobs = {}; // id -> API prompt, only for the ones that converted
  const errors = []; // [{ path, error }] for read/convert failures
  const now = new Date().toISOString(); // upload time, stamped on ready entries

  for (const path of selected) {
    const id = await workflowId(path);
    const name = nameOf(path);
    const entry = byPath.get(path);
    const fingerprint = entry ? fingerprintOf(entry) : "";

    // 1. Read the saved UI graph.
    let ui;
    try {
      const resp = await a.getUserData("workflows/" + path);
      if (!resp || !resp.ok) throw new Error("failed to read file");
      ui = await resp.json();
    } catch (e) {
      const error = "failed to read file";
      workflows.push({ id, path, name, fingerprint, status: "error", error });
      errors.push({ path, error });
      continue;
    }

    // 2. Convert UI graph -> API prompt (offscreen, with live-graph fallback).
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
        uploaded_at: now,
      });
    } catch (e) {
      const error = String((e && e.message) || e);
      workflows.push({ id, path, name, fingerprint, status: "error", error });
      errors.push({ path, error });
    }
  }

  // 3. Assemble the manifest (selected items only) and POST to Python.
  const manifest = {
    version: 1,
    updated_at: new Date().toISOString(),
    workflows,
  };
  const res = await postSync(manifest, blobs);
  if (!res || !res.ok) {
    throw new Error((res && res.error) || "upload failed");
  }

  // 4. Persist on success only, so a failed POST doesn't poison the "uploaded"
  //    flags. Reflects exactly what every paired account's app now browses.
  saveLastManifest(manifest);

  return { uploaded: Object.keys(blobs).length, errors };
}
