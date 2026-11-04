(() => {
  "use strict";
  const API = window.CNR_ML_CONFIG.apiBase;
  const $ = (id) => document.getElementById(id);
  const store = { get: () => sessionStorage.getItem("cnrml_key"), set: (k) => sessionStorage.setItem("cnrml_key", k), clear: () => sessionStorage.removeItem("cnrml_key") };
  let me = null;

  // Safe DOM builder: text is always set via textContent / createTextNode.
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return el;
  }
  const fmt = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

  async function api(path, opts = {}) {
    let res;
    try {
      const headers = { Authorization: "Bearer " + store.get() };
      if (!(opts.body instanceof FormData)) headers["Content-Type"] = "application/json";
      res = await fetch(API + path, { ...opts, headers });
    } catch { throw new Error("Cannot reach the ML API. Check your connection and try again."); }
    if (res.status === 401) { signOut(); throw new Error("Your key was not accepted."); }
    if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b.detail || "Request failed (" + res.status + ")."); }
    return res.status === 204 ? null : res.json();
  }

  const PAGES = [
    { id: "overview", label: "Overview", perm: null, render: overview },
    { id: "models", label: "Models", perm: "read:catalog", render: models },
    { id: "experiments", label: "Experiments", perm: "read:experiments", render: experiments },
    { id: "datasets", label: "Datasets", perm: "read:datasets", render: datasets },
    { id: "audit", label: "Audit log", perm: "read:audit", render: audit },
    { id: "access", label: "Access", perm: "manage:keys", render: access },
  ];

  const empty = (title, body) => h("div", { class: "empty" }, h("b", null, title), h("p", null, body));

  async function overview() {
    const o = await api("/v1/overview");
    const c = o.counts;
    return [
      h("h1", null, "Overview"),
      h("div", { class: "notice" }, o.safety_notice),
      h("div", { class: "stats" }, [["Planned models", c.model_definitions], ["Datasets", c.datasets], ["Experiments", c.experiments], ["Model versions", c.model_versions], ["Deployed models", c.deployed_models]]
        .map(([l, n]) => h("div", { class: "stat" }, h("b", null, n), h("span", null, l)))),
      h("h2", null, "Build progress"),
      h("ol", { class: "track" }, o.phase.roadmap.map((p) =>
        h("li", { class: p.phase === o.phase.current ? "current" : "", "aria-current": p.phase === o.phase.current ? "step" : null },
          h("span", { class: "n" }, "Phase " + p.phase), h("span", { class: "t" }, p.name)))),
      c.datasets === 0 ? empty("No datasets yet", "Create one under Datasets and upload a file. Nothing has been trained and no model is available for predictions.") : null,
      h("p", { class: "muted" }, "API " + o.service.version + " · " + o.service.environment),
    ];
  }

  async function models() {
    const list = await api("/v1/catalog");
    return [h("h1", null, "Models"),
      h("p", null, "These are the prediction targets the laboratory is designed to support. A model becomes available only after evaluation, external validation and clinical-reviewer approval."),
      list.map((m) => h("div", { class: "card" },
        h("div", { class: "row" }, h("h3", null, m.name), h("span", { class: "badge " + m.status }, m.status)),
        h("p", null, m.description), h("small", null, "Task: " + m.task_type)))];
  }

  async function experiments() {
    const rows = await api("/v1/experiments");
    return [h("h1", null, "Experiments"), rows.length ? table(["ID", "Framework", "Status", "Started", "By"], rows.map((r) => [r.id, r.framework, r.status, fmt(r.created_at), r.created_by || "—"]))
      : empty("No experiments yet", "Training runs and their metrics will appear here once the first XGBoost model is trained (Phase 3).")];
  }

  const can = (perm) => me && me.permissions.includes(perm);
  const field = (label, el, id) => h("div", null, h("label", { for: id }, label), el);

  async function datasets(arg) {
    if (arg) return datasetDetail(Number(arg));
    const rows = await api("/v1/datasets");
    const out = [h("h1", null, "Datasets"),
      h("div", { class: "notice" }, "Uploaded files are analysed in memory. Only the de-identified table is stored; the original file is never kept. De-identified is not the same as anonymous: keep access restricted."),
      can("write:datasets") ? createDatasetForm() : null];
    out.push(rows.length ? table(["Name", "Sensitivity", "Versions", "Added", ""], rows.map((r) => [r.name, r.sensitivity, r.version_count, fmt(r.created_at),
      h("button", { class: "link", onclick: () => { location.hash = "datasets/" + r.id; } }, "Open")]))
      : empty("No datasets yet", can("write:datasets") ? "Create a dataset above, then upload a CSV or Excel file to it." : "A data scientist or admin can create the first dataset."));
    return out;
  }

  function createDatasetForm() {
    const name = h("input", { id: "dn", maxlength: 160 });
    const desc = h("input", { id: "dd", maxlength: 2000 });
    const sens = h("select", { id: "ds" }, ["restricted", "confidential", "internal"].map((x) => h("option", { value: x }, x)));
    const err = h("p", { class: "error", role: "alert" });
    const btn = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      try {
        const d = await api("/v1/datasets", { method: "POST", body: JSON.stringify({ name: name.value.trim(), description: desc.value.trim(), sensitivity: sens.value }) });
        location.hash = "datasets/" + d.id;
      } catch (e) { err.textContent = e.message; }
    } }, "Create dataset");
    return h("details", { class: "card" }, h("summary", null, "New dataset"),
      h("div", { class: "form3" }, field("Name", name, "dn"), field("Description", desc, "dd"), field("Sensitivity", sens, "ds")), btn, err);
  }

  async function datasetDetail(id) {
    const d = await api("/v1/datasets/" + id);
    const out = [h("p", null, h("button", { class: "link", onclick: () => { location.hash = "datasets"; } }, "← All datasets")),
      h("h1", null, d.name), h("p", null, d.description || "No description."),
      h("p", { class: "muted" }, "Sensitivity: " + d.sensitivity + " · created by " + (d.created_by || "—") + " on " + fmt(d.created_at))];
    if (can("write:datasets")) out.push(uploadWizard(d));
    out.push(h("h2", null, "Versions"));
    out.push(d.versions.length ? d.versions.map((v) => versionCard(d, v)) : empty("No versions yet", "Upload a file to create version 1."));
    return out;
  }

  function versionCard(d, v) {
    const r = v.report_json || {}, k = r.k_anonymity || {};
    const preview = h("div");
    const bits = [
      h("div", { class: "row" }, h("h3", null, "Version " + v.version), h("span", { class: "badge" }, v.deidentified ? "de-identified" : "not de-identified")),
      h("p", null, v.row_count.toLocaleString() + " rows × " + v.column_count + " columns · " + (v.file_format || "").toUpperCase() + " · by " + (v.created_by || "—") + " · " + fmt(v.created_at)),
      h("small", null, "Approval: " + (v.approval_ref || "—") + " · SHA-256 " + v.sha256.slice(0, 12) + "…"),
      reportBlock(r, k),
    ];
    if (can("read:dataset_rows")) bits.push(h("button", { onclick: async () => {
      try {
        const p = await api("/v1/datasets/" + d.id + "/versions/" + v.version + "/preview?rows=20");
        preview.replaceChildren(table(p.columns, p.rows.map((row) => row.map((c) => c == null ? "—" : String(c)))));
      } catch (e) { preview.replaceChildren(h("div", { class: "notice" }, e.message)); }
    } }, "Preview 20 rows"), preview);
    bits.push(h("details", null, h("summary", null, "Column summary"),
      table(["Column", "Type", "Missing", "Unique", "Range / top values"], (v.schema_info.columns || []).map((c) => [c.name, c.kind, (c.missing_pct ?? 0) + "%", c.unique ?? "—", summary(c)]))));
    return h("div", { class: "card" }, bits);
  }

  function summary(c) {
    if (c.kind === "numeric" || c.kind === "date") return c.min == null ? "—" : c.min + " to " + c.max + (c.mean != null ? " (mean " + c.mean + ")" : "");
    return (c.top || []).map((t) => t.value + " (" + t.count + ")").join(", ") || "—";
  }

  function reportBlock(r, k) {
    const li = (t) => h("li", null, t);
    const items = [];
    if ((r.dropped || []).length) items.push(li("Dropped: " + r.dropped.map((x) => x.column).join(", ")));
    if ((r.pseudonymised || []).length) items.push(li("Pseudonymised: " + r.pseudonymised.join(", ")));
    if ((r.generalised || []).length) items.push(li("Reduced to year: " + r.generalised.map((x) => x.column).join(", ")));
    if (r.duplicates_removed) items.push(li(r.duplicates_removed + " duplicate rows removed"));
    if (r.empty_rows_removed) items.push(li(r.empty_rows_removed + " empty rows removed"));
    const cm = Object.entries(r.coerced_to_missing || {});
    if (cm.length) items.push(li("Unreadable values set to missing: " + cm.map(([c, n]) => c + " (" + n + ")").join(", ")));
    if (k.k_min != null) items.push(li("Smallest group sharing " + k.columns.join(" + ") + ": " + k.k_min + " (" + k.rows_in_small_groups + " rows in groups under 5)"));
    return h("div", null, h("ul", { class: "plain" }, items), (r.warnings || []).map((w) => h("div", { class: "notice" }, w)));
  }

  function uploadWizard(d) {
    const root = h("details", { class: "card" }, h("summary", null, "Upload a new version"));
    const body = h("div"); root.append(body);
    const err = h("p", { class: "error", role: "alert" });
    const file = h("input", { id: "uf", type: "file", accept: ".csv,.xlsx" });
    const analyse = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      const f = file.files[0]; if (!f) { err.textContent = "Choose a .csv or .xlsx file first."; return; }
      const fd = new FormData(); fd.append("file", f);
      analyse.disabled = true; analyse.textContent = "Analysing…";
      try { review(f, await api("/v1/datasets/inspect", { method: "POST", body: fd })); }
      catch (e) { err.textContent = e.message; }
      finally { analyse.disabled = false; analyse.textContent = "Analyse file"; }
    } }, "Analyse file");
    body.append(h("p", null, "Step 1: choose a CSV or Excel file (first sheet is used). The file is analysed, nothing is stored yet."), field("File", file, "uf"), analyse, err);

    function review(f, a) {
      const sel = {}, why = {};
      const rowsEls = a.columns.map((c) => {
        const opts = [];
        if (c.can_keep) opts.push("keep");
        opts.push("drop");
        if (a.pseudonymisation_available && c.kind !== "date" && c.kind !== "text") opts.push("pseudonymise");
        if (c.kind === "date") opts.push("year_only");
        const s = h("select", { "aria-label": "Action for " + c.name }, opts.map((o) => h("option", { value: o, selected: o === c.suggested }, o.replace("_", " "))));
        const r = h("input", { placeholder: "Reason for keeping (10+ characters)", maxlength: 300, hidden: true, "aria-label": "Reason for keeping " + c.name });
        const sync = () => { r.hidden = !(s.value === "keep" && c.category !== "none"); };
        s.addEventListener("change", sync); sync();
        sel[c.name] = s; why[c.name] = r;
        return [c.name, c.kind, c.category === "none" ? (c.reason || "—") : h("span", { class: "badge bad" }, c.category.replace("_", " ")),
          h("div", null, s, r), h("span", { class: "muted" }, c.profile.hidden ? "hidden" : (c.profile.missing_pct ?? 0) + "% missing")];
      });
      const appr = h("input", { id: "ap", maxlength: 300, placeholder: "e.g. COMREC P.05/26/1234 or data-sharing agreement ref" });
      const dd = h("input", { id: "dd2", type: "checkbox", checked: true });
      const err2 = h("p", { class: "error", role: "alert" });
      const go = h("button", { class: "primary", onclick: async () => {
        err2.textContent = "";
        const plan = { drop_duplicates: dd.checked, columns: a.columns.map((c) => ({ name: c.name, action: sel[c.name].value, override_reason: sel[c.name].value === "keep" && c.category !== "none" ? why[c.name].value.trim() : null })) };
        const fd = new FormData(); fd.append("file", f); fd.append("plan", JSON.stringify(plan)); fd.append("approval_ref", appr.value.trim());
        go.disabled = true;
        try { await api("/v1/datasets/" + d.id + "/versions", { method: "POST", body: fd }); location.hash = "datasets/" + d.id; show("datasets/" + d.id); }
        catch (e) { err2.textContent = e.message; go.disabled = false; }
      } }, "Create version");
      const notes = [];
      if (a.warnings.length) notes.push(a.warnings.map((w) => h("div", { class: "notice" }, w)));
      const da = Object.entries(a.date_assumptions);
      if (da.length) notes.push(h("p", { class: "muted" }, "Date formats read as: " + da.map(([c, m]) => c + " (" + m + ")").join(", ") + "."));
      body.replaceChildren(
        h("p", null, a.rows.toLocaleString() + " rows, " + a.columns.length + " columns. Step 2: review each column. Flagged columns are dropped by default and their values are not shown."),
        ...notes.flat(), table(["Column", "Type", "Finding", "Action", "Quality"], rowsEls),
        h("div", { class: "form3" }, field("Approval reference (required)", appr, "ap"),
          h("div", null, h("label", { for: "dd2" }, "Remove duplicate rows"), dd)),
        h("p", { class: "muted" }, "Missing values are left as they are. Imputation and encoding are done inside training so they never leak across folds."),
        go, err2);
    }
    return root;
  }

  async function audit() {
    const rows = await api("/v1/audit?limit=100");
    return [h("h1", null, "Audit log"), h("p", null, "The 100 most recent events. Secrets and request bodies are never recorded."),
      rows.length ? table(["Time", "Actor", "Role", "Action", "Resource"], rows.map((r) => [fmt(r.ts), r.actor_name || "unauthenticated", r.role || "—", r.action, r.resource_type ? r.resource_type + " " + (r.resource_id || "") : "—"])) : empty("No events yet", "")];
  }

  async function access() {
    const out = h("div");
    const name = h("input", { id: "kn", maxlength: 120 });
    const role = h("select", { id: "kr" }, ["viewer", "data_scientist", "clinical_reviewer", "admin"].map((r) => h("option", { value: r }, r)));
    const err = h("p", { class: "error", role: "alert" });
    const create = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      try {
        const k = await api("/v1/admin/keys", { method: "POST", body: JSON.stringify({ name: name.value.trim(), role: role.value }) });
        out.replaceChildren(h("div", { class: "notice" }, h("b", null, "Copy this key now. It will not be shown again."), h("div", { class: "secret" }, k.key)));
        name.value = ""; refreshTable();
      } catch (e) { err.textContent = e.message; }
    } }, "Create key");
    const holder = h("div");
    const refreshTable = async () => {
      const ks = await api("/v1/admin/keys");
      holder.replaceChildren(table(["Name", "Role", "Created", "Last used", "Status", ""], ks.map((k) => [k.name, k.role, fmt(k.created_at), fmt(k.last_used_at), k.revoked_at ? "revoked" : "active",
        k.revoked_at ? "" : h("button", { class: "danger", onclick: async () => { if (confirm("Revoke key “" + k.name + "”?")) { try { await api("/v1/admin/keys/" + k.id, { method: "DELETE" }); refreshTable(); } catch (e) { alert(e.message); } } } }, "Revoke")])));
    };
    refreshTable().catch((e) => holder.replaceChildren(h("div", { class: "notice" }, e.message)));
    return [h("h1", null, "Access"), h("p", null, "Each person or service gets its own key and role. Revoke a key as soon as it is no longer needed."),
      h("div", { class: "inline" }, h("div", null, h("label", { for: "kn" }, "Key name"), name), h("div", null, h("label", { for: "kr" }, "Role"), role), create), err, out, holder];
  }

  function table(head, rows) {
    return h("div", { class: "tablewrap" }, h("table", null, h("thead", null, h("tr", null, head.map((x) => h("th", { scope: "col" }, x)))),
      h("tbody", null, rows.map((r) => h("tr", null, r.map((c) => h("td", null, c)))))));
  }

  async function show(id) {
    const [pid, arg] = String(id || "").split("/");
    const page = PAGES.find((p) => p.id === pid) || PAGES[0];
    document.querySelectorAll("#nav button").forEach((b) => { if (b.dataset.id === page.id) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
    const view = $("view");
    view.replaceChildren(h("p", null, "Loading…"));
    try { view.replaceChildren(...[].concat(await page.render(arg)).flat().filter(Boolean)); }
    catch (e) { view.replaceChildren(h("h1", null, page.label), h("div", { class: "notice" }, e.message)); }
    view.focus({ preventScroll: true });
  }

  async function start() {
    try { me = await api("/v1/me"); } catch (e) { return showLogin(e.message); }
    $("login").hidden = true; $("app").hidden = false;
    $("who").textContent = me.name + " (" + me.role.replace("_", " ") + ")";
    const allowed = PAGES.filter((p) => !p.perm || me.permissions.includes(p.perm));
    $("nav").replaceChildren(...allowed.map((p) => h("button", { "data-id": p.id, onclick: () => { location.hash = p.id; } }, p.label)));
    show(location.hash.slice(1));
  }
  function showLogin(msg) { $("app").hidden = true; $("login").hidden = false; $("login-error").textContent = msg || ""; $("key").focus(); }
  function signOut() { store.clear(); me = null; showLogin(); }

  $("signin").addEventListener("click", () => { const k = $("key").value.trim(); if (!k) return; store.set(k); $("key").value = ""; start(); });
  $("key").addEventListener("keydown", (e) => { if (e.key === "Enter") $("signin").click(); });
  $("signout").addEventListener("click", signOut);
  window.addEventListener("hashchange", () => me && show(location.hash.slice(1)));
  store.get() ? start() : showLogin();
})();
