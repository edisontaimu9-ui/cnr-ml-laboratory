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
      res = await fetch(API + path, { ...opts, headers: { Authorization: "Bearer " + store.get(), "Content-Type": "application/json" } });
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
      c.datasets === 0 ? empty("No datasets yet", "Dataset ingestion arrives in Phase 2. Nothing has been trained and no model is available for predictions.") : null,
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

  async function datasets() {
    const rows = await api("/v1/datasets");
    return [h("h1", null, "Datasets"), rows.length ? table(["Name", "Sensitivity", "Added"], rows.map((r) => [r.name, r.sensitivity, fmt(r.created_at)]))
      : empty("No datasets yet", "CSV and Excel upload, validation and de-identification arrive in Phase 2. Do not upload patient data until they are in place.")];
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
    const page = PAGES.find((p) => p.id === id) || PAGES[0];
    document.querySelectorAll("#nav button").forEach((b) => { if (b.dataset.id === page.id) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
    const view = $("view");
    view.replaceChildren(h("p", null, "Loading…"));
    try { view.replaceChildren(...[].concat(await page.render()).flat().filter(Boolean)); }
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
