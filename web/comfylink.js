// ComfyLink panel — pair this PC (to one OR MORE accounts) and show status.
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
  // Unpair ONE account by its backend_id. Omitting it server-side unpairs all,
  // but the panel always targets a specific row.
  async unpair(backendId) {
    const r = await fetch("/comfylink/unpair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend_id: backendId }),
    });
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

// Sync-target selection persistence. We store the backend_ids the user has
// EXPLICITLY unchecked; anything NOT in this set is a target. This guarantees a
// newly-paired account defaults ON (it's absent from the "off" set), and a fresh
// install (no stored set) checks every account.
const SYNC_OFF_KEY = "comfylink.syncAccountsOff";
function loadSyncOff() {
  try {
    const raw = localStorage.getItem(SYNC_OFF_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(arr) ? arr : []);
  } catch (e) {
    return new Set();
  }
}
function saveSyncOff(set) {
  try {
    localStorage.setItem(SYNC_OFF_KEY, JSON.stringify([...set]));
  } catch (e) {
    console.warn("[ComfyLink] failed to persist sync-account selection", e);
  }
}

// Compact local "YYYY-MM-DD HH:MM" for an ISO upload timestamp ("" if invalid).
function fmtUploadedAt(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

// One paired account's display text: the email, or a placeholder while it is
// still unknown. The email is NOT known the instant a pairing appears — the
// worker learns it from the relay's register response a few seconds later, and
// it is deliberately never persisted, so it is unknown again after every ComfyUI
// restart until the worker re-registers. Any row that shows an account must
// therefore be able to go from placeholder -> email WITHOUT being rebuilt.
const ACCOUNT_PENDING = "pairing…";

// Paint an account's text into an EXISTING row: text, tooltip, and the muted
// italic styling used for the placeholder. Nothing else about the row is
// touched — no node is created, replaced or removed — so this is safe to call on
// every poll even while the user is interacting with the row.
// Early-outs when the text already matches (an email can never equal the
// placeholder), so a steady-state poll writes to the DOM zero times.
function paintAccount(span, titleEl, account) {
  const text = account || ACCOUNT_PENDING;
  if (span.textContent === text) return;
  span.textContent = text;
  titleEl.title = text;
  span.style.color = account ? "" : "var(--descrip-text,#aaa)";
  span.style.fontStyle = account ? "" : "italic";
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

  // HARD block bar: the relay has stopped serving this plugin version (403
  // plugin_too_old). Deliberately at the very top and far louder than the soft
  // amber `updateLine` hint below — nothing works until the user updates. A
  // single reused element toggled from s.plugin_too_old (never re-created by the
  // 3s poll).
  const tooOldBar = h("div", {
    style:
      "display:none;margin-bottom:10px;padding:8px 10px;border-radius:4px;" +
      "background:#b71c1c;color:#fff;font-size:12px;font-weight:700;" +
      "line-height:1.4;",
  });

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

  // --- paired accounts list (one row per account, each with its own Unpair) ---
  const accountsTitle = h("div", {
    style: "font-weight:600;margin-bottom:6px;display:none;",
    textContent: "Paired accounts",
  });
  const accountsList = h("div", { style: "margin-bottom:14px;" });
  // Which backend_ids the rendered rows currently are, plus the elements whose
  // text has to keep up with the status poll. Same contract as the sync-target
  // list below: rebuild rows only when the SET of accounts changes, repaint
  // their text on every poll.
  let accountRowKeys = "";
  const accountRows = new Map(); // backend_id -> { row, span }

  // --- pairing form: ALWAYS visible, used to ADD (append) more accounts -------
  const formTitle = h("div", {
    style: "font-weight:600;margin-bottom:6px;",
    textContent: "Pair an account",
  });
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
  const pairForm = h("div", { style: "margin-bottom:8px;" }, [
    formTitle,
    nameInput,
    codeInput,
    pairBtn,
  ]);

  // --- workflow management (shown when paired) ----------------------------
  // Manual upload: pick which saved workflows to push, convert them on the spot,
  // and POST manifest + blobs. The plugin fans the catalog out to every account
  // checked below. No background auto-sync.
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

  // Account picker: uploads target the CHECKED accounts (this ComfyUI can be
  // paired to several, and a single upload can push to any subset). One checkbox
  // row per pairing.
  const accountSelectTitle = h("div", {
    style: "font-weight:600;margin-bottom:4px;font-size:12px;",
    textContent: "Sync to accounts",
  });
  const accountsSyncList = h("div", {
    style: "max-height:120px;overflow-y:auto;margin-bottom:8px;",
  });
  // Tracks which backend_ids the checkbox list currently reflects, so the 3s
  // poll only rebuilds the rows when the SET of accounts actually changes (never
  // yanking the boxes out from under a mid-selection user). Everything that can
  // change WITHOUT the set changing (the account email) is repainted in place
  // instead — see syncAccountCheckboxes.
  let accountKeys = "";
  const syncRows = new Map(); // backend_id -> { label, span }
  const currentBackendIds = () =>
    Array.from(accountsSyncList.querySelectorAll("input[type=checkbox]"))
      .filter((c) => c.checked)
      .map((c) => c.dataset.backendId);

  const managePanel = h(
    "div",
    { style: "display:none;margin-top:8px;" },
    [accountSelectTitle, accountsSyncList, reloadBtn, wfList, wfStatus, uploadBtn]
  );

  const msg = h("div", { style: "margin-top:10px;min-height:18px;color:#f44336;" });

  // small, unobtrusive version line; filled from the status response.
  const versionLine = h("div", {
    style: "margin-top:14px;color:var(--descrip-text,#888);font-size:11px;opacity:0.7;",
    textContent: "ComfyLink",
  });

  // Soft "an update is available" hint, shown just under the version line. A
  // single reused element (never re-created per refresh) that we toggle + fill
  // from s.update; hidden entirely when there's nothing to say.
  const updateLine = h("div", {
    style:
      "display:none;margin-top:3px;color:var(--descrip-text,#888);" +
      "font-size:11px;opacity:0.75;",
  });

  // Test-relay warning (hidden on the production relay): makes it obvious when
  // the plugin is pointed at a non-default relay via comfylink.json.
  const relayWarn = h("div", {
    style:
      "display:none;margin-bottom:10px;padding:5px 8px;border-radius:4px;" +
      "background:#5d4037;color:#ffcc80;font-size:11px;",
  });

  root.append(
    tooOldBar,
    statusRow,
    relayWarn,
    detail,
    accountsTitle,
    accountsList,
    pairForm,
    manageBtn,
    managePanel,
    msg,
    versionLine,
    updateLine
  );

  // Render one row per paired account: email (or "pairing…" until it registers)
  // and a per-account Unpair button.
  //
  // Rows are rebuilt ONLY when the set of backend_ids changes; on every other
  // poll we just repaint the text. The rows carry state of their own — an Unpair
  // button that is disabled while its request is in flight, and keyboard focus —
  // and a blind rebuild every 3s would silently re-enable a button mid-unpair and
  // steal focus. The email still refreshes, because it is repainted below.
  function renderAccounts(pairings) {
    const items = Array.isArray(pairings) ? pairings : [];
    accountsTitle.style.display = items.length ? "block" : "none";
    const key = items.map((p) => p.backend_id).join("|");
    if (key !== accountRowKeys) {
      accountRowKeys = key;
      accountRows.clear();
      accountsList.innerHTML = "";
      for (const p of items) {
        const label = h("span", {
          style: "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
        });
        const btn = h("button", {
          textContent: "Unpair",
          style: "padding:3px 10px;cursor:pointer;font-size:11px;",
        });
        btn.onclick = async () => {
          btn.disabled = true;
          try {
            await api.unpair(p.backend_id);
          } catch (e) {
            /* refresh shows the resulting state regardless */
          } finally {
            // Re-enable explicitly: on success refresh() drops this row anyway,
            // but on failure the row survives and must stay clickable (the poll
            // no longer rebuilds it for us).
            btn.disabled = false;
            refresh();
          }
        };
        const row = h(
          "div",
          {
            style:
              "display:flex;align-items:center;gap:8px;padding:5px 0;" +
              "border-bottom:1px solid var(--border-color,#333);",
          },
          [label, btn]
        );
        accountRows.set(p.backend_id, { row, span: label });
        accountsList.append(row);
      }
    }
    // EVERY poll: the email arrives after the row does (and again after every
    // ComfyUI restart), so refresh the text whether or not we rebuilt.
    for (const p of items) {
      const r = accountRows.get(p.backend_id);
      if (r) paintAccount(r.span, r.span, p.account);
    }
  }

  // Build the sync-target checkbox list from the pairings. Only rebuild when the
  // SET of backend_ids changes (so the 3s poll never disturbs a user mid-check).
  // Each row's checked state comes from the persisted "off" set, so a newly-
  // paired account defaults ON and prior explicit unchecks are honored across
  // rebuilds. Toggling a box persists the change + reloads the workflow list.
  // Returns true when the list was rebuilt (caller may refresh the list).
  //
  // The rebuild guard deliberately keys on backend_ids ONLY — but that means the
  // account email, which lands a few seconds AFTER the pairing itself (and again
  // after every restart), changes without changing the key. So the text of the
  // existing rows is repainted on EVERY poll, below the guard. Repainting touches
  // only the <span>'s text/tooltip/colour: the checkbox element, its checked
  // state, its focus and its handler are never re-created, so a user mid-check is
  // still never disturbed.
  function syncAccountCheckboxes(pairings) {
    const items = Array.isArray(pairings) ? pairings : [];
    const key = items.map((p) => p.backend_id).join("|");
    const rebuilt = key !== accountKeys;
    if (rebuilt) {
      accountKeys = key;
      syncRows.clear();
      const off = loadSyncOff();
      accountsSyncList.innerHTML = "";
      for (const p of items) {
        const checked = !off.has(p.backend_id);
        const cb = h("input", { type: "checkbox", checked });
        cb.dataset.backendId = p.backend_id;
        cb.onchange = () => {
          const s = loadSyncOff();
          if (cb.checked) s.delete(p.backend_id);
          else s.add(p.backend_id);
          saveSyncOff(s);
          loadWorkflows();
        };
        const span = h("span", {
          style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;",
        });
        const label = h(
          "label",
          {
            style:
              "display:flex;align-items:center;gap:6px;padding:3px 0;cursor:pointer;font-size:12px;",
          },
          [cb, span]
        );
        syncRows.set(p.backend_id, { label, span });
        accountsSyncList.append(label);
      }
    }
    // EVERY poll, rebuilt or not: pull the (late-arriving) account email into the
    // rows that already exist. Without this the row stays on "pairing…" forever,
    // because the email never changes the rebuild key above.
    for (const p of items) {
      const r = syncRows.get(p.backend_id);
      if (r) paintAccount(r.span, r.label, p.account);
    }
    return rebuilt;
  }

  async function refresh() {
    let s;
    try {
      s = await api.status();
    } catch (e) {
      // The plugin's own HTTP route is unreachable (ComfyUI restarting, etc).
      // Grey the dot too, so a stale green "Online" dot can't outlive the text.
      dot.style.background = "#9e9e9e";
      stateText.textContent = "Panel offline";
      return;
    }
    // Hard block: relay refuses this plugin version. Missing field (older
    // plugin state / undefined) reads as false → bar stays hidden.
    if (s.plugin_too_old === true) {
      const min = s.plugin_min_version ? `v${s.plugin_min_version}` : "a newer version";
      tooOldBar.innerHTML = "";
      tooOldBar.append(
        `⛔ ComfyLink plugin too old — the relay has stopped serving this plugin. ` +
          `Update to ${min} or newer and restart ComfyUI. (Your pairing is kept.)`
      );
      if (s.plugin_update_url) {
        tooOldBar.append(
          h("a", {
            textContent: "Get the update",
            href: s.plugin_update_url,
            target: "_blank",
            rel: "noopener noreferrer",
            style: "color:#fff;text-decoration:underline;margin-left:6px;",
          })
        );
      }
      tooOldBar.style.display = "block";
    } else {
      tooOldBar.style.display = "none";
    }

    const [label, color] = STATE_LABEL[s.state] || ["Unknown", "#9e9e9e"];
    dot.style.background = color;
    stateText.textContent = `${label}${s.active ? " · generating" : ""}`;

    const lines = [`Name: ${s.backend_name || "-"}`];
    if (s.state === "online") lines.push(`Nodes: ${s.node_count}`);
    if (s.error) lines.push(`Note: ${s.error}`);
    detail.textContent = lines.join("\n");

    // Loud reminder when pointed at a non-default (test) relay.
    if (s.relay_is_default === false) {
      let host = s.relay_url || "";
      try {
        host = new URL(s.relay_url).host;
      } catch (e) {
        /* keep raw url if it doesn't parse */
      }
      relayWarn.textContent = `⚠ Test relay: ${host}`;
      relayWarn.style.display = "block";
    } else {
      relayWarn.style.display = "none";
    }

    if (s.version) {
      // Show "ComfyLink v0.1.0 · <commit>" so the user can tell if they pulled
      // the latest; hide the commit when unknown ("dev").
      const c = s.commit && s.commit !== "dev" ? ` · ${s.commit}` : "";
      versionLine.textContent = `ComfyLink v${s.version}${c}`;
    }

    // Soft update hint (never a popup, never blocks anything). Reuse updateLine:
    // fill + show when an update is available, otherwise hide.
    const up = s.update;
    if (up && up.available) {
      const latest = up.latest ? `v${up.latest}` : "a newer version";
      updateLine.innerHTML = "";
      if (up.below_min) {
        // Amber, slightly stronger wording — still only a hint.
        updateLine.style.color = "#ffb74d";
        updateLine.style.opacity = "0.95";
        updateLine.append(
          `Update recommended — your version may be out of date (${latest})`
        );
      } else {
        updateLine.style.color = "var(--descrip-text,#888)";
        updateLine.style.opacity = "0.75";
        updateLine.append(`Update available → ${latest}`);
      }
      if (up.url) {
        const link = h("a", {
          textContent: "  Get it",
          href: up.url,
          target: "_blank",
          rel: "noopener noreferrer",
          style: "color:inherit;text-decoration:underline;margin-left:4px;",
        });
        updateLine.append(link);
      }
      updateLine.style.display = "block";
    } else {
      updateLine.style.display = "none";
    }

    renderAccounts(s.pairings);
    // Keep the sync-target list in sync; if the set of accounts changed (e.g. one
    // was unpaired or newly paired) and the panel is open, reload so the markers
    // reflect the new checked set.
    const rebuilt = syncAccountCheckboxes(s.pairings);
    if (rebuilt && managePanel.style.display !== "none") loadWorkflows();

    const paired = !!s.paired;
    // The pair form is ALWAYS visible (add more accounts); workflow management
    // only makes sense once at least one account is paired.
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
        msg.textContent = "Paired. The account appears above once it connects.";
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
      // Tag: green "uploaded <date>" normally; amber "changed · uploaded <date>"
      // when the file changed on disk since (re-upload to refresh it).
      let tag = null;
      if (wf.uploaded) {
        const when = fmtUploadedAt(wf.uploadedAt);
        tag = wf.changed
          ? h("span", {
              style: "margin-left:6px;color:#ff9800;font-size:11px;",
              textContent: when ? `changed · uploaded ${when}` : "changed since upload",
            })
          : h("span", {
              style: "margin-left:6px;color:#4caf50;font-size:11px;",
              textContent: when ? `uploaded ${when}` : "uploaded",
            });
      }
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
      const items = await listWorkflows(currentBackendIds());
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
    const backendIds = currentBackendIds();
    if (!backendIds.length) {
      wfStatus.style.color = "#ff9800";
      wfStatus.textContent = "Check at least one account to sync to.";
      return;
    }
    uploadBtn.disabled = true;
    reloadBtn.disabled = true;
    wfStatus.style.color = "var(--descrip-text,#aaa)";
    wfStatus.textContent = `Uploading ${paths.length}…`;
    try {
      const { uploaded, errors, accounts } = await uploadSelected(paths, backendIds);
      // Number of accounts that actually synced OK (fall back to requested set).
      const n = (accounts && accounts.length) || backendIds.length;
      if (errors.length) {
        wfStatus.style.color = "#ff9800";
        const failed = errors.map((e) => nameFromPath(e.path)).join(", ");
        wfStatus.textContent = `Uploaded ${uploaded} to ${n} account(s); ${errors.length} failed: ${failed}`;
      } else {
        wfStatus.style.color = "#4caf50";
        wfStatus.textContent = `Uploaded ${uploaded} workflow(s) to ${n} account(s).`;
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
