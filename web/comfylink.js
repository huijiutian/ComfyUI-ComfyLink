// ComfyLink panel — pair this PC and show connection status.
// Registers a sidebar tab (current ComfyUI frontend); falls back to a toast if
// the sidebar API is unavailable.
import { app } from "../../scripts/app.js";

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

  const msg = h("div", { style: "margin-top:10px;min-height:18px;color:#f44336;" });

  root.append(statusRow, detail, pairForm, unpairBtn, msg);

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

    const paired = !!s.paired;
    pairForm.style.display = paired ? "none" : "block";
    unpairBtn.style.display = paired ? "block" : "none";
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
        msg.textContent = "Paired. Connecting…";
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
  },
});
