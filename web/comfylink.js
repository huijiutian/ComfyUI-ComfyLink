// ComfyLink panel — pair this PC and show connection status.
// Registers a sidebar tab (current ComfyUI frontend); falls back to a toast if
// the sidebar API is unavailable.
import { app } from "../../scripts/app.js";
import { listWorkflows, uploadSelected } from "./sync.js";

const api = {
  async status() {
    // cache-buster + no-store: poll must never get a stale cached value.
    const r = await fetch(`/comfylink/status?_=${Date.now()}`, { cache: "no-store" });
    return r.json();
  },
  async pair(code, name) {
    const r = await fetch("/comfylink/pair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, name }),
    });
    return r.json();
  },
  async unpair() {
    const r = await fetch("/comfylink/unpair", { method: "POST" });
    return r.json();
  },
};

const STATE_LABEL = {
  unpaired: ["Not paired", "#9e9e9e"],
  connecting: ["Connecting…", "#ff9800"],
  online: ["Online", "#4caf50"],
  error: ["Error", "#f44336"],
};

function nameFromPath(path) {
  const i = path.lastIndexOf("/");
  const base = i >= 0 ? path.slice(i + 1) : path;
  return base.toLowerCase().endsWith(".json") ? base.slice(0, -5) : base;
}

function h(tag, props = {}, children = []) {
  const e = document.createElement(tag);
  Object.assign(e, props);
  if (props.style) e.setAttribute("style", props.style);
  for (const c of [].concat(children)) {
    if (c != null) e.append(c);
  }
  return e;
}

function buildPanel(root) {
  root.innerHTML = "";
  root.style.padding = "12px";
  root.style.fontSize = "13px";

  const dot = h("span", {
    style:
      "display:inline-block;width:10px;height:10px;border-radius:50%;background:#9e9e9e;margin-right:8px;",
  });
  const stateText = h("span", { textContent: "…" });
  const statusRow = h("div", { style: "margin-bottom:10px;font-weight:600;" }, [
    dot,
    stateText,
  ]);

  const detail = h("div", {
    style: "color:var(--descrip-text,#aaa);margin-bottom:14px;white-space:pre-wrap;",
  });

  // pairing form (shown when not paired)
  const nameInput = h("input", {
    type: "text",
    placeholder: "Device name",
    style: "width:100%;margin-bottom:8px;padding:6px;box-sizing:border-box;",
  });
  const codeInput = h("input", {
    type: "text",
    placeholder: "Pairing code (from the app)",
    style:
      "width:100%;margin-bottom:8px;padding:6px;box-sizing:border-box;text-transform:uppercase;",
  });
  const pairBtn = h("button", {
    textContent: "Pair",
    style: "width:100%;padding:8px;cursor:pointer;",
  });
  const pairForm = h("div", {}, [nameInput, codeInput, pairBtn]);

  // unpair (shown when paired)
  const unpairBtn = h("button", {
    textContent: "Unpair this PC",
    style: "width:100%;padding:8px;cursor:pointer;",
  });

  // --- workflow management (shown when paired) ----------------------------
  // Manual upload: pick which saved workflows to push to the app, convert them
  // on the spot, and POST manifest + blobs. No background auto-sync.
  const manageBtn = h("button", {
    textContent: "Manage workflows",
    style: "width:100%;padding:8px;margin-top:8px;cursor:pointer;",
  });

  // Collapsible management panel.
  const wfList = h("div", {
    style:
      "max-height:240px;overflow-y:auto;border:1px solid var(--border-color,#444);" +
      "border-radius:4px;padding:6px;margin:8px 0;",
  });
  const wfStatus = h("div", {
    style: "min-height:16px;margin-bottom:8px;color:var(--descrip-text,#aaa);font-size:12px;",
  });
  const uploadBtn = h("button", {
    textContent: "Upload / update selected",
    style: "width:100%;padding:8px;cursor:pointer;",
  });
  const reloadBtn = h("button", {
    textContent: "Refresh list",
    style: "width:100%;padding:6px;margin-bottom:8px;cursor:pointer;",
  });
  const managePanel = h(
    "div",
    { style: "display:none;margin-top:8px;" },
    [reloadBtn, wfList, wfStatus, uploadBtn]
  );

  const msg = h("div", { style: "margin-top:10px;min-height:18px;color:#f44336;" });

  // small, unobtrusive version line; filled from the status response.
  const versionLine = h("div", {
    style: "margin-top:14px;color:var(--descrip-text,#888);font-size:11px;opacity:0.7;",
    textContent: "ComfyLink",
  });

  root.append(
    statusRow,
    detail,
    pairForm,
    unpairBtn,
    manageBtn,
    managePanel,
    msg,
    versionLine
  );

  async function refresh() {
    let s;
    try {
      s = await api.status();
    } catch (e) {
      stateText.textContent = "Panel offline";
      return;
    }
    const [label, color] = STATE_LABEL[s.state] || ["Unknown", "#9e9e9e"];
    dot.style.background = color;
    stateText.textContent = `${label}${s.active ? " · generating" : ""}`;

    const lines = [`Name: ${s.backend_name || "-"}`];
    if (s.state === "online") lines.push(`Nodes: ${s.node_count}`);
    if (s.error) lines.push(`Note: ${s.error}`);
    detail.textContent = lines.join("\n");

    if (s.version) versionLine.textContent = `ComfyLink v${s.version}`;

    const paired = !!s.paired;
    pairForm.style.display = paired ? "none" : "block";
    unpairBtn.style.display = paired ? "block" : "none";
    manageBtn.style.display = paired ? "block" : "none";
    if (!paired) managePanel.style.display = "none"; // collapse when unpaired
    if (!nameInput.value && s.backend_name) nameInput.value = s.backend_name;
  }

  pairBtn.onclick = async () => {
    msg.style.color = "#f44336";
    msg.textContent = "";
    const code = codeInput.value.trim();
    if (!code) {
      msg.textContent = "Enter the pairing code from the app.";
      return;
    }
    pairBtn.disabled = true;
    try {
      const r = await api.pair(code, nameInput.value.trim());
      if (r.ok) {
        codeInput.value = "";
        msg.style.color = "#4caf50";
        msg.textContent = "Paired. Open “Manage workflows” to upload.";
      } else {
        msg.textContent = r.error || "Pairing failed.";
      }
    } catch (e) {
      msg.textContent = String(e);
    } finally {
      pairBtn.disabled = false;
      refresh();
    }
  };

  unpairBtn.onclick = async () => {
    await api.unpair();
    refresh();
  };

  // --- workflow management wiring -----------------------------------------
  // Render one checkbox row per saved workflow. Already-uploaded ones are
  // checked by default (so they stay in the manifest) and tagged "uploaded".
  function renderWorkflows(items) {
    wfList.innerHTML = "";
    if (!items.length) {
      wfList.append(
        h("div", {
          style: "color:var(--descrip-text,#aaa);font-size:12px;",
          textContent: "No saved workflows found.",
        })
      );
      return;
    }
    for (const wf of items) {
      const cb = h("input", { type: "checkbox", checked: wf.uploaded });
      cb.dataset.path = wf.path;
      const tag = wf.uploaded
        ? h("span", {
            style: "margin-left:6px;color:#4caf50;font-size:11px;",
            textContent: "uploaded",
          })
        : null;
      const label = h(
        "label",
        {
          style:
            "display:flex;align-items:center;gap:6px;padding:3px 0;cursor:pointer;font-size:12px;",
          title: wf.path,
        },
        [cb, h("span", { textContent: wf.name }), tag]
      );
      wfList.append(label);
    }
  }

  async function loadWorkflows() {
    wfStatus.style.color = "var(--descrip-text,#aaa)";
    wfStatus.textContent = "Loading…";
    reloadBtn.disabled = true;
    uploadBtn.disabled = true;
    try {
      const items = await listWorkflows();
      renderWorkflows(items);
      wfStatus.textContent = `${items.length} workflow(s) on this PC.`;
    } catch (e) {
      wfList.innerHTML = "";
      wfStatus.style.color = "#f44336";
      wfStatus.textContent = String((e && e.message) || e);
    } finally {
      reloadBtn.disabled = false;
      uploadBtn.disabled = false;
    }
  }

  manageBtn.onclick = () => {
    const open = managePanel.style.display !== "none";
    if (open) {
      managePanel.style.display = "none";
      return;
    }
    managePanel.style.display = "block";
    loadWorkflows();
  };

  reloadBtn.onclick = () => loadWorkflows();

  uploadBtn.onclick = async () => {
    const paths = Array.from(
      wfList.querySelectorAll("input[type=checkbox]")
    )
      .filter((c) => c.checked)
      .map((c) => c.dataset.path);
    if (!paths.length) {
      wfStatus.style.color = "#ff9800";
      wfStatus.textContent = "Select at least one workflow.";
      return;
    }
    uploadBtn.disabled = true;
    reloadBtn.disabled = true;
    wfStatus.style.color = "var(--descrip-text,#aaa)";
    wfStatus.textContent = `Uploading ${paths.length}…`;
    try {
      const { uploaded, errors } = await uploadSelected(paths);
      if (errors.length) {
        wfStatus.style.color = "#ff9800";
        const failed = errors.map((e) => nameFromPath(e.path)).join(", ");
        wfStatus.textContent = `Uploaded ${uploaded}; ${errors.length} failed: ${failed}`;
      } else {
        wfStatus.style.color = "#4caf50";
        wfStatus.textContent = `Uploaded ${uploaded} workflow(s).`;
      }
      // Reflect new "uploaded" tags / checkboxes.
      await loadWorkflows();
    } catch (e) {
      wfStatus.style.color = "#f44336";
      wfStatus.textContent = `Upload failed: ${(e && e.message) || e}`;
    } finally {
      uploadBtn.disabled = false;
      reloadBtn.disabled = false;
    }
  };

  refresh();
  const timer = setInterval(refresh, 3000);
  // best-effort cleanup if the node is removed from the DOM
  return () => clearInterval(timer);
}

app.registerExtension({
  name: "ComfyLink.Panel",
  async setup() {
    const reg = app.extensionManager?.registerSidebarTab;
    if (reg) {
      app.extensionManager.registerSidebarTab({
        id: "comfylink",
        icon: "pi pi-link",
        title: "ComfyLink",
        tooltip: "ComfyLink — pair & status",
        type: "custom",
        render: (el) => buildPanel(el),
      });
    } else {
      console.warn(
        "[ComfyLink] sidebar API unavailable; open /comfylink/status to check status."
      );
    }
    // Workflow upload is manual now: the panel's "Manage workflows" button lets
    // the user pick which workflows to convert + push. No background auto-sync.
  },
});
