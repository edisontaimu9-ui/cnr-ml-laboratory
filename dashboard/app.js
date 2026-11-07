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
      else if (k === "style") el.style.cssText = v;   // the CSP blocks style attributes; setting via CSSOM is allowed
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return el;
  }
  const fmt = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

  // Loaders: a top progress bar while any request is in flight, and a spinner on the button that was just pressed.
  let pending = 0, lastBtn = null, silent = false;
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest("button");
    lastBtn = b && !b.closest("nav") && !b.classList.contains("link") ? { el: b, t: performance.now() } : null;
  }, true);

  async function api(path, opts = {}) {
    const btn = lastBtn && performance.now() - lastBtn.t < 150 ? lastBtn.el : null;
    pending++;
    if (!silent) document.body.classList.add("busy");
    if (btn) { btn.classList.add("loading"); btn.setAttribute("aria-busy", "true"); }
    try { return await request(path, opts); }
    finally {
      if (--pending === 0) document.body.classList.remove("busy");
      if (btn) { btn.classList.remove("loading"); btn.removeAttribute("aria-busy"); }
    }
  }

  async function request(path, opts) {
    let res;
    const { blob, ...fetchOpts } = opts;   // blob: true returns the response as a Blob (file downloads)
    try {
      const headers = { Authorization: "Bearer " + store.get() };
      if (!(fetchOpts.body instanceof FormData)) headers["Content-Type"] = "application/json";
      res = await fetch(API + path, { ...fetchOpts, headers });
    } catch { throw new Error("Cannot reach the ML API. Check your connection and try again."); }
    if (res.status === 401) { signOut(); throw new Error("Your key was not accepted."); }
    if (!res.ok) {
      const b = await res.json().catch(() => ({}));
      let d = b.detail;
      // FastAPI sends validation problems as a list of objects; turn them into readable text.
      if (Array.isArray(d)) d = d.map((e) => ((e.loc || []).slice(1).join(" ") || "input") + ": " + (e.msg || "invalid")).join("; ");
      // Some refusals (corpus uploads) send an object with a message and a list of problems; keep it on the error.
      let detail = null;
      if (d && typeof d === "object") { detail = d; d = d.message; }
      const err = new Error(d || "Request failed (" + res.status + ").");
      err.detail = detail;
      throw err;
    }
    if (blob) return res.blob();
    return res.status === 204 ? null : res.json();
  }

  const PAGES = [
    { id: "overview", label: "Overview", perm: null, render: overview },
    { id: "models", label: "Models", perm: "read:catalog", render: models },
    { id: "experiments", label: "Experiments", perm: "read:experiments", render: experiments },
    { id: "explorations", label: "Exploration", perm: "read:explorations", render: explorations },
    { id: "matching", label: "Matching", perm: "write:experiments", render: matchingPage },
    { id: "hotspots", label: "Hotspots", perm: "read:hotspots", render: hotspots },
    { id: "nlp", label: "NLP", perm: "read:nlp", render: nlpPage },
    { id: "corpora", label: "NLP corpora", perm: "read:datasets", render: corporaPage },
    { id: "ask", label: "Ask CNR", perm: "use:llm", render: askPage },
    { id: "eval", label: "RAG evaluation", perm: "read:llm", render: evalPage },
    { id: "summarise", label: "Summarise note", perm: "summarise:llm", render: summarisePage },
    { id: "train", label: "Training data", perm: "read:llm", render: trainPage },
    { id: "predict", label: "Predict", perm: "predict:models", render: predictPage },
    { id: "registry", label: "Registry", perm: "read:models", render: registry },
    { id: "datasets", label: "Datasets", perm: "read:datasets", render: datasets },
    { id: "notes", label: "Notes", perm: "read:notes", render: notesPage },
    { id: "integration", label: "CNR integration", perm: "read:integration", render: integrationPage },
    { id: "audit", label: "Audit log", perm: "read:audit", render: audit },
    { id: "access", label: "Access", perm: "manage:keys", render: access },
  ];

  const loader = (text) => h("div", { class: "loader", role: "status" }, h("span", { class: "spinner", "aria-hidden": "true" }), text);
  const empty = (title, body) => h("div", { class: "empty" }, h("b", null, title), h("p", null, body));

  async function overview() {
    const o = await api("/v1/overview");
    const c = o.counts;
    return [
      h("h1", null, "Overview"),
      h("p", null, "CNR ML Laboratory is a machine learning and predictive analytics platform developed as an extension of the Chakudya Nutrition Registry (CNR). It provides an environment for nutrition data analysis, machine learning experimentation, predictive model development, training, evaluation, comparison and model versioning."),
      h("p", null, "It is designed to turn nutrition data into actionable intelligence, supporting statistical and AI-driven models for nutrition research, clinical decision support, public health surveillance and risk prediction. Predictions are decision support, never confirmed diagnoses."),
      h("div", { class: "notice" }, o.safety_notice),
      h("div", { class: "stats" }, [["Planned models", c.model_definitions], ["Datasets", c.datasets], ["Experiments", c.experiments], ["Explorations", c.explorations], ["Hotspot analyses", c.hotspot_analyses], ["Model versions", c.model_versions], ["Deployed models", c.deployed_models]]
        .map(([l, n]) => h("div", { class: "stat" }, h("b", null, n), h("span", null, l)))),
      c.datasets === 0 ? empty("No datasets yet", "Create one under Datasets and upload a file. Nothing has been trained and no model is available for predictions.") : null,
      ...aboutLab(),
      h("p", { class: "muted" }, "API " + o.service.version + " · " + o.service.environment),
    ];
  }

  // What the laboratory is for. Every capability here exists in the code; keep it that way when this list changes.
  function aboutLab() {
    const cap = (lead, text) => h("li", null, h("b", null, lead + " "), text);
    return [
      h("h2", null, "Core capabilities"),
      h("ul", { class: "plain" },
        cap("Data preparation.", "Upload a file; identifiers are detected and can be dropped or pseudonymised, types and missing values are profiled, and only the de-identified table is stored."),
        cap("Data exploration.", "Unsupervised clustering and anomaly detection to find patterns and unusual records in nutrition data, kept separate from any deployed model."),
        cap("Supervised learning.", "Predictive models with XGBoost, a neural network (MLP) and interpretable logistic regression, for the eighteen nutrition and health targets in the catalogue."),
        cap("Model comparison.", "Algorithms are compared on the identical split against a trivial baseline, using accuracy, precision, recall, F1, ROC-AUC and, for logistic regression, calibration."),
        cap("Model evaluation.", "Hold-out and cross-validation, bootstrap ranges, per-class results and leakage guards, to judge how reliable a model is and how well it generalises."),
        cap("Surveillance.", "District hotspots and prevalence trends from aggregated counts, never from individual children."),
        cap("Model management.", "Experiments are recorded and trained models preserved. The registry keeps versions with fixed seeds and hashes for reproducibility, and every deployment passes clinical review."),
        cap("Text analysis (NLP).", "Structured nutrition information from clinical notes, reports, research text and dietary records in English, Chichewa, Tumbuka and Yao: foods, conditions, symptoms, interventions and measurements, with negation and identifier masking. The engine is rule-based. Annotated corpora, spaCy training against the rule baseline and a reviewed model registry are available under NLP corpora; no trained NLP model is served."),
        cap("Laboratory memory.", "Decisions, findings, lessons and warnings recorded next to the datasets, experiments and models they are about, kept with their history."),
        cap("CNR integration.", "A pathway for validated models to be served to CNR services and APIs, and an opt-in feed of de-identified data from CNR.")),
      h("h2", null, "Primary objective"),
      h("p", null, "To establish a research-oriented machine learning environment that supports nutrition scientists, researchers and developers in building, testing and evaluating data-driven solutions for Malawi's nutrition challenges."),
      h("div", { class: "notice" }, h("b", null, "Important. "), "Models developed in the laboratory remain experimental until appropriately validated and approved for their intended use. Unsupervised exploration is kept separate from production deployment. The laboratory does not yet run subgroup fairness checks (for example by sex or district) and does not replace external clinical validation."),
      h("p", { class: "muted" }, "From nutrition data to machine learning research, and from validated models to practical nutrition intelligence."),
    ];
  }

  async function models() {
    const [list, vers] = await Promise.all([api("/v1/catalog"), api("/v1/models/versions?limit=200")]);
    const live = (slug) => vers.filter((v) => v.model_slug === slug && (v.status === "deployed" || v.status === "approved")).map((v) => "v" + v.version + " " + v.status).join(", ");
    return [h("h1", null, "Models"),
      h("p", null, "These are the prediction targets the laboratory is designed to support. A model becomes available only after evaluation, external validation and clinical-reviewer approval."),
      list.map((m) => h("div", { class: "card" },
        h("div", { class: "row" }, h("h3", null, m.name), h("span", { class: "badge " + m.status }, m.status)),
        h("p", null, m.description), h("small", null, "Task: " + m.task_type + (live(m.slug) ? " · " + live(m.slug) : ""))))];
  }

  let poll = null;
  const pct = (x) => (x == null ? "—" : (100 * x).toFixed(1) + "%");
  const n3 = (x) => (x == null ? "—" : String(x));
  const verdictShort = (v) => (v === "mlp_better" || v === "logistic_better" ? "ahead of XGBoost" : v === "xgboost_better" ? "behind XGBoost" : "no clear gap to XGBoost");
  const statusBadge = (st) => h("span", { class: "badge " + (st === "failed" ? "bad" : st === "completed" ? "" : "planned") }, st);

  async function experiments(arg) {
    if (arg === "new") return experimentForm();
    if (arg) return experimentDetail(Number(arg));
    const rows = await api("/v1/experiments");
    return [h("h1", null, "Experiments"),
      h("p", null, "Each run trains an XGBoost model, a neural network (MLP) or an interpretable logistic regression on a de-identified dataset version and reports honest hold-out results next to a trivial baseline. A neural network or logistic regression run can also be compared against XGBoost on the same split."),
      can("write:experiments") ? h("p", null, h("button", { class: "primary", onclick: () => { location.hash = "experiments/new"; } }, "New experiment")) : null,
      rows.length ? table(["#", "Model", "Status", "Result vs baseline", "By", "Started", ""], rows.map((r) => [r.id, (r.model_slug || "—") + (r.summary && r.summary.algorithm === "mlp" ? " (neural network)" : r.summary && r.summary.algorithm === "logistic" ? " (logistic regression)" : ""), statusBadge(r.status),
        r.summary && r.summary.test != null ? r.summary.metric + " " + n3(r.summary.test) + " vs " + n3(r.summary.baseline) + (r.summary.vs_xgboost ? " (" + verdictShort(r.summary.vs_xgboost.verdict) + ")" : "") : (r.summary && r.summary.error ? "failed" : "—"),
        r.created_by || "—", fmt(r.created_at), h("button", { class: "link", onclick: () => { location.hash = "experiments/" + r.id; } }, "Open")]))
      : empty("No experiments yet", can("write:experiments") ? "Upload a dataset version first, then start an experiment." : "A data scientist or admin can start the first experiment.")];
  }

  async function experimentForm() {
    const [models, sets] = await Promise.all([api("/v1/catalog"), api("/v1/datasets")]);
    const usable = sets.filter((d) => d.version_count > 0);
    if (!usable.length) return [h("h1", null, "New experiment"), empty("No dataset versions yet", "Create a dataset and upload a file under Datasets first.")];
    const model = h("select", { id: "xm" }, [h("option", { value: "" }, "Choose a model…")].concat(models.filter((m) => m.task_type !== "matching").map((m) => h("option", { value: m.slug }, m.name + " (" + m.task_type + ")"))));
    const dset = h("select", { id: "xd" }, usable.map((d) => h("option", { value: d.id }, d.name)));
    const ver = h("select", { id: "xv" });
    const target = h("select", { id: "xt" });
    const excl = h("div", { class: "checks" });
    const err = h("p", { class: "error", role: "alert" });
    let versions = [];
    const num = (id, v, min, max, step) => h("input", { id, type: "number", value: v, min, max, step: step || 1 });
    const trees = num("xn", 200, 10, 1000), depth = num("xdp", 4, 1, 10), lr = num("xlr", 0.1, 0.01, 0.5, 0.01), tsz = num("xts", 0.2, 0.1, 0.4, 0.05), seed = num("xs", 42, 0, 2147483647);
    const bal = h("input", { id: "xb", type: "checkbox", checked: true });
    const algo = h("select", { id: "xa" }, [h("option", { value: "xgboost" }, "XGBoost (gradient-boosted trees)"), h("option", { value: "mlp" }, "Neural network (MLP)"), h("option", { value: "logistic" }, "Logistic regression (interpretable)")]);
    const layers = h("input", { id: "xl", type: "text", value: "64,32", maxlength: 20, inputmode: "numeric" });
    const alpha = num("xal", 0.001, 0.000001, 1, 0.0001), iters = num("xit", 500, 50, 1000, 50);
    const cmp = h("input", { id: "xc", type: "checkbox", checked: true });
    const xgbBox = h("div", { class: "form3" }, field("Trees", trees, "xn"), field("Max depth", depth, "xdp"), field("Learning rate", lr, "xlr"));
    const mlpBox = h("div", { class: "form3", hidden: true }, field("Hidden layers (sizes, comma separated, up to 3)", layers, "xl"), field("Weight penalty (alpha)", alpha, "xal"),
      field("Max iterations", iters, "xit"));
    const lrC = num("xlc", 1, 0.001, 100, 0.1);
    const lrCal = h("select", { id: "xcal" }, [h("option", { value: "none" }, "None (the model's own probabilities)"), h("option", { value: "sigmoid" }, "Sigmoid recalibration"), h("option", { value: "isotonic" }, "Isotonic recalibration (needs 1,000+ rows)")]);
    const lrBox = h("div", { hidden: true }, h("div", { class: "form3" }, field("Penalty strength C (smaller = stronger shrinkage)", lrC, "xlc"), field("Probability recalibration", lrCal, "xcal")),
      h("p", { class: "muted" }, "Classification models only. Coefficients are reported in the result. Recalibration is optional: logistic probabilities are often already well calibrated, and the result shows whether recalibration helped on the hold-out set. For probabilities that match real frequencies, turn off Balance classes."));
    const cmpBox = h("div", { hidden: true }, h("label", { for: "xc" }, "Compare against XGBoost"), cmp);
    algo.addEventListener("change", () => { xgbBox.hidden = algo.value !== "xgboost"; mlpBox.hidden = algo.value !== "mlp"; lrBox.hidden = algo.value !== "logistic"; cmpBox.hidden = algo.value === "xgboost"; });
    const cols = () => (versions.find((v) => String(v.version) === ver.value) || { schema_info: { columns: [] } }).schema_info.columns || [];
    // Models fed by the CNR training sync come with a suggested target and the columns that define it. Only admins can read the sync status.
    const train = await api("/v1/sync/training").catch(() => null);
    const hint = h("p", { class: "muted", role: "status" });
    const suggestion = () => {
      const m = train && train.models.find((x) => x.slug === model.value);
      const d = usable.find((x) => String(x.id) === dset.value);
      return m && m.available && d && d.name === m.dataset_name ? m : null;
    };
    const applySuggestion = () => {
      const m = suggestion();
      hint.textContent = "";
      if (!m || !cols().some((c) => c.name === m.target)) return;
      target.value = m.target;
      drawExcl();
      excl.querySelectorAll("input").forEach((i) => { i.checked = m.exclude.includes(i.value); });
      hint.textContent = "Filled in from the CNR sync: predict " + m.target + " and leave out " + m.exclude.join(", ") + ", which the label is calculated from. Change them if you know better.";
    };
    const drawCols = () => {
      const prev = target.value, names = cols().map((c) => c.name);
      target.replaceChildren(h("option", { value: "" }, "Choose the column to predict…"), ...names.map((n) => h("option", { value: n, selected: n === prev }, n)));
      drawExcl();
      applySuggestion();
    };
    const drawExcl = () => excl.replaceChildren(...cols().filter((c) => c.name !== target.value).map((c) =>
      h("label", { class: "chk" }, h("input", { type: "checkbox", value: c.name }), " " + c.name)));
    const loadVersions = async () => {
      const d = await api("/v1/datasets/" + dset.value);
      versions = d.versions;
      ver.replaceChildren(...versions.map((v) => h("option", { value: v.version }, "v" + v.version + " (" + v.row_count + " rows)")));
      drawCols();
    };
    dset.addEventListener("change", loadVersions); ver.addEventListener("change", drawCols); target.addEventListener("change", drawExcl); model.addEventListener("change", applySuggestion);
    await loadVersions();
    const go = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      if (!model.value) { err.textContent = "Choose a model."; return; }
      if (!target.value) { err.textContent = "Choose the column to predict."; return; }
      const sizes = layers.value.split(",").map((x) => x.trim()).filter(Boolean).map(Number);
      if (algo.value === "mlp" && (!sizes.length || sizes.length > 3 || sizes.some((n) => !Number.isInteger(n) || n < 2 || n > 256))) {
        err.textContent = "Hidden layers must be 1 to 3 whole numbers between 2 and 256, for example 64,32."; return;
      }
      go.disabled = true;
      try {
        const body = { model_slug: model.value, dataset_id: Number(dset.value), version: Number(ver.value), target: target.value,
          exclude: [...excl.querySelectorAll("input:checked")].map((i) => i.value), n_estimators: Number(trees.value), max_depth: Number(depth.value),
          learning_rate: Number(lr.value), test_size: Number(tsz.value), seed: Number(seed.value), balance_classes: bal.checked,
          algorithm: algo.value, hidden_layers: sizes.length ? sizes : [64, 32], alpha: Number(alpha.value), max_iter: Number(iters.value), compare_xgboost: cmp.checked,
          logistic_c: Number(lrC.value), calibration: lrCal.value };
        const e = await api("/v1/experiments", { method: "POST", body: JSON.stringify(body) });
        location.hash = "experiments/" + e.id;
      } catch (e) { err.textContent = e.message; go.disabled = false; }
    } }, "Start training");
    return [h("p", null, h("button", { class: "link", onclick: () => { location.hash = "experiments"; } }, "← All experiments")), h("h1", null, "New experiment"),
      h("div", { class: "notice" }, "Exclude every column the label is calculated from (for a nutrition status: MUAC, weight-for-height, z-scores). Leaving them in gives a near-perfect score that means nothing in practice."),
      h("div", { class: "form3" }, field("Model", model, "xm"), field("Dataset", dset, "xd"), field("Version", ver, "xv"), field("Target column (what to predict)", target, "xt")),
      hint, h("label", null, "Columns to leave out of training"), excl,
      h("details", { class: "card" }, h("summary", null, "Training settings"),
        h("div", { class: "form3" }, field("Algorithm", algo, "xa")), xgbBox, mlpBox, lrBox, cmpBox,
        h("div", { class: "form3" }, field("Test share", tsz, "xts"), field("Random seed", seed, "xs"),
          h("div", null, h("label", { for: "xb" }, "Balance classes"), bal))),
      go, err];
  }

  const num3 = (x) => (x == null ? "—" : (Math.abs(x) >= 100 ? Math.round(x) : Number(x.toPrecision(3))));
  function coefficientSections(co) {
    const out = [h("h2", null, "Coefficients"), h("p", { class: "muted" }, co.reading_guide)];
    (co.outcomes || []).forEach((o) => {
      if (!co.binary) out.push(h("h3", null, "Class: " + o.class));
      const rows = o.terms.map((t) => [t.feature + (t.level != null ? " = " + t.level : " (per 1 SD)"), num3(t.coefficient),
        co.binary ? num3(t.odds_ratio) : "—", co.binary ? (t.odds_ratio_per_unit != null ? num3(t.odds_ratio_per_unit) : "—") : "—"]);
      out.push(table(co.binary ? ["Feature", "Coefficient (log-odds)", "Odds ratio", "Odds ratio per 1 unit"] : ["Feature", "Coefficient (log-odds)", "", ""], rows));
      out.push(divBars(o.terms.map((t) => t.feature + (t.level != null ? " = " + t.level : "")), o.terms.map((t) => t.coefficient), "Red raises the modelled outcome, blue lowers it. Longer bar, stronger effect."));
      if (o.n_terms > o.terms.length) out.push(h("p", { class: "muted" }, "Showing the " + o.terms.length + " largest of " + o.n_terms + " terms."));
    });
    if (co.binary) out.push(h("p", { class: "muted" }, "Outcome modelled: " + co.outcomes[0].class + ". An odds ratio above 1 raises the odds of that outcome; below 1 lowers them."));
    if (co.rare_levels_hidden) out.push(h("p", { class: "muted" }, "Category levels seen in fewer than 10 training rows are not listed."));
    out.push(h("p", { class: "muted" }, co.note));
    return out;
  }
  function calibrationSections(c) {
    const a = c.after || {}, b = c.before;
    const out = [h("h2", null, "Probability calibration"), h("div", { class: "notice" }, c.summary),
      table(["Hold-out (" + c.n_test + " rows)", "Brier score", "Log loss", "Expected calibration error"],
        [...(b ? [["Before recalibration", n3(b.brier), n3(b.log_loss), n3(b.ece)]] : []), [b ? "After " + c.method + " recalibration" : "Model's own probabilities", n3(a.brier), n3(a.log_loss), n3(a.ece)]]),
      h("p", { class: "muted" }, "Reliability: when the model is this confident in its top class, how often is it right? Close agreement means the probabilities can be read as frequencies."),
      table(["Confidence range", "Cases", "Mean confidence", "Observed accuracy"], (a.reliability || []).map((r) => [pct(r.range[0]) + "–" + pct(r.range[1]), r.n, pct(r.mean_confidence), pct(r.observed_accuracy)])),
      (a.reliability || []).length ? barChart((a.reliability || []).map((r) => pct(r.range[0]) + "–" + pct(r.range[1])), [{ name: "Mean confidence", color: "#8a9ba2", values: a.reliability.map((r) => r.mean_confidence) }, { name: "Observed accuracy", color: "#0e5a73", values: a.reliability.map((r) => r.observed_accuracy) }], "Well calibrated when the two bars in each range are about the same length.") : null,
      h("p", { class: "muted" }, c.brier_note)];
    return out;
  }

  async function experimentDetail(id) {
    const e = await api("/v1/experiments/" + id);
    const m = e.metrics_json || {}, cfg = (e.params_json || {}).config || {};
    const out = [h("p", null, h("button", { class: "link", onclick: () => { location.hash = "experiments"; } }, "← All experiments")),
      h("div", { class: "row" }, h("h1", null, "Experiment #" + e.id), statusBadge(e.status)),
      h("p", { class: "muted" }, (e.model_name || "") + " · dataset " + (e.params_json.dataset_id || "?") + " v" + (e.params_json.dataset_version || "?") + " · target " + (cfg.target || "?") +
        " · " + (m.algorithm === "logistic" ? "logistic regression, C " + ((m.model_settings || {}).C || "?") + ", " + (((m.model_settings || {}).calibration || "none") === "none" ? "no recalibration" : (m.model_settings.calibration + " recalibration")) : m.algorithm === "mlp" ? "neural network " + ((m.model_settings || {}).hidden_layers || []).join("-") + ", " + ((m.model_settings || {}).iterations_used || "?") + " iterations" : (cfg.n_estimators || "?") + " trees, depth " + (cfg.max_depth || "?")) + " · started " + fmt(e.created_at) + " by " + (e.created_by || "—"))];
    out.push(notesPanel("experiment", id));
    if (e.status === "queued" || e.status === "running") {
      out.push(h("div", { class: "notice" }, "Training is running. This page refreshes by itself."));
      poll = setTimeout(() => { if (location.hash.slice(1) === "experiments/" + id) show("experiments/" + id, true); }, 2500);
      return out;
    }
    if (e.status === "failed") { out.push(h("div", { class: "notice" }, m.error || "The run failed.")); return out; }
    const t = m.test || {}, b = m.baseline || {}, cv = m.cv || {};
    if (can("write:models")) out.push(registerBox(e));
    (m.warnings || []).forEach((w) => out.push(h("div", { class: "notice" }, w)));
    const stat = (l, v, sub) => h("div", { class: "stat" }, h("b", null, v), h("span", null, l + (sub ? " · " + sub : "")));
    if (m.task === "classification") {
      out.push(h("h2", null, "Hold-out test (" + m.n_test + " rows)"),
        h("div", { class: "stats" }, stat("Macro F1", n3(t.macro_f1), "95% range " + n3(t.macro_f1_ci95 && t.macro_f1_ci95[0]) + "–" + n3(t.macro_f1_ci95 && t.macro_f1_ci95[1])),
          stat("Baseline macro F1", n3(b.macro_f1), b.strategy), stat("Balanced accuracy", pct(t.balanced_accuracy)), stat("Accuracy", pct(t.accuracy)), stat("ROC AUC", n3(t.roc_auc))),
        h("p", { class: "muted" }, "Baselines: always the most common class " + n3(b.majority_macro_f1) + ", guessing by class frequency " + n3(b.chance_macro_f1) + ". The model has to beat the higher one, with its 95% range clear of it."),
        h("p", { class: "muted" }, "Cross-validation on the training part (" + cv.folds + " folds): macro F1 " + n3(cv.macro_f1_mean) + " ± " + n3(cv.macro_f1_sd) + ", balanced accuracy " + n3(cv.balanced_accuracy_mean) + " ± " + n3(cv.balanced_accuracy_sd) + "."));
      out.push(h("h2", null, "Model against baseline"), perfChart(m), cvChart(m));
      out.push(h("h2", null, "Per class"), perClassChart(m.per_class), table(["Class", "Precision", "Recall", "F1", "Test cases"], Object.entries(m.per_class || {}).map(([k, v]) => [k, v.precision, v.recall, v.f1, v.support])));
      const cm = m.confusion_matrix || { labels: [], matrix: [] };
      out.push(h("h2", null, "Confusion matrix"), h("p", { class: "muted" }, "Rows are the true class, columns the predicted class."),
        table(["True \\ Predicted", ...cm.labels], cm.matrix.map((row, i) => [cm.labels[i], ...row])));
    } else {
      out.push(h("h2", null, "Hold-out test (" + m.n_test + " rows)"),
        h("div", { class: "stats" }, stat("R²", n3(t.r2), "baseline " + n3(b.r2)), stat("RMSE", n3(t.rmse), "95% range " + n3(t.rmse_ci95 && t.rmse_ci95[0]) + "–" + n3(t.rmse_ci95 && t.rmse_ci95[1])),
          stat("Baseline RMSE", n3(b.rmse), b.strategy), stat("MAE", n3(t.mae))),
        h("p", { class: "muted" }, "Cross-validation (" + cv.folds + " folds): R² " + n3(cv.r2_mean) + " ± " + n3(cv.r2_sd) + ", RMSE " + n3(cv.rmse_mean) + " ± " + n3(cv.rmse_sd) + "."));
      out.push(h("h2", null, "Model against baseline"), perfChart(m), cvChart(m));
    }
    const cp = m.comparison;
    if (cp) {
      const clsf = m.task === "classification";
      const rows = clsf ? [["Macro F1", "macro_f1"], ["Balanced accuracy", "balanced_accuracy"], ["Accuracy", "accuracy"], ["ROC AUC", "roc_auc"]]
        : [["R²", "r2"], ["RMSE (lower is better)", "rmse"], ["MAE (lower is better)", "mae"]];
      const cvk = clsf ? "Macro F1" : "R²";
      const lab = cp.selected.algorithm === "logistic" ? "Logistic regression" : "Neural network";
      out.push(h("h2", null, lab + " vs XGBoost"),
        h("div", { class: "notice" }, cp.summary),
        table(["Hold-out test", lab, "XGBoost"], [
          ...rows.map(([l, k]) => [l, n3(cp.selected.test[k]), n3(cp.xgboost.test[k])]),
          ["Cross-validation " + cvk, n3(cp.selected.cv_mean) + " ± " + n3(cp.selected.cv_sd), n3(cp.xgboost.cv_mean) + " ± " + n3(cp.xgboost.cv_sd)]]),
        h("p", { class: "muted" }, "Gap in " + cvk + " (" + lab.toLowerCase() + " minus XGBoost): " + n3(cp.difference) + ", 95% range " + n3(cp.difference_ci95[0]) + " to " + n3(cp.difference_ci95[1]) +
          ". Both models used the same training rows, folds, class weights and hold-out rows. XGBoost ran with its default settings (" + cp.reference_settings.n_estimators + " trees, depth " +
          cp.reference_settings.max_depth + ", learning rate " + cp.reference_settings.learning_rate + "), not tuned."),
        h("p", { class: "muted" }, cp.caution));
    }
    if (m.coefficients) out.push(...coefficientSections(m.coefficients));
    if (m.calibration) out.push(...calibrationSections(m.calibration));
    out.push(h("h2", null, "What the model relied on"), h("p", { class: "muted" }, (m.importance_method === "gain" || !m.importance_method ? "Share of total gain per feature." : "Share of the score lost when each feature is shuffled in the hold-out set (" + m.importance_method + ").") + " This shows what the model used, not what causes the outcome."),
      h("div", { class: "bars" }, (m.feature_importance || []).slice(0, 12).map((f) => h("div", { class: "barrow" }, h("span", null, f.feature),
        h("div", { class: "bar" }, h("i", { style: "width:" + Math.max(1, f.share * 100) + "%" })), h("span", null, pct(f.share))))));
    out.push(h("h2", null, "Limits of this result"), h("ul", { class: "plain" }, (m.limitations || []).map((x) => h("li", null, x))),
      h("p", { class: "muted" }, "Features used: " + [...(m.features.numeric || []), ...(m.features.categorical || [])].join(", ") + "."));
    return out;
  }




  // ---------- Exploration: unsupervised analysis (clustering, anomaly detection). Never registered, never deployed. ----------
  const XP_COLORS = ["#0e5a73", "#c98a1b", "#2e8b57", "#8a4fb3", "#c0392b", "#4a7fb5", "#7a8b99", "#b5651d", "#2a9d8f", "#d4559a", "#6b705c", "#e07a5f"];
  const xpBadge = () => h("span", { class: "badge explore" }, "exploration · never deployed");
  const signed = (x) => (x == null ? "—" : (x > 0 ? "+" : "") + x);
  const structureLabel = { clear_structure: "Clear structure", weak_structure: "Weak structure", no_clear_structure: "No clear structure" };

  async function explorations(arg) {
    if (arg === "new") return explorationForm();
    if (arg) return explorationDetail(Number(arg));
    const rows = await api("/v1/explorations");
    const sum = (r) => {
      const s = r.summary || {};
      if (s.error) return "failed";
      return [s.clustering ? s.clustering.k + " groups (" + (structureLabel[s.clustering.structure] || s.clustering.structure).toLowerCase() + ")" : null,
        s.anomaly ? s.anomaly.flagged + " rows flagged" : null].filter(Boolean).join(" · ") || "—";
    };
    return [h("div", { class: "row" }, h("h1", null, "Exploration"), xpBadge()),
      h("p", null, "Look for groups of similar rows (clustering) and unusual rows (anomaly detection) in a de-identified dataset version, with no outcome to predict. Explorations are for understanding data and checking data quality."),
      h("div", { class: "notice" }, "An exploration is never a model. It cannot be registered, reviewed, deployed, used for predictions or sent to CNR. Groups and flags are patterns in this dataset, not diagnoses."),
      can("write:explorations") ? h("p", null, h("button", { class: "primary", onclick: () => { location.hash = "explorations/new"; } }, "New exploration")) : null,
      rows.length ? table(["#", "Status", "Result", "By", "Started", ""], rows.map((r) => [r.id, statusBadge(r.status), sum(r), r.created_by || "—", fmt(r.created_at),
        h("button", { class: "link", onclick: () => { location.hash = "explorations/" + r.id; } }, "Open")]))
        : empty("No explorations yet", can("write:explorations") ? "Upload a dataset version first, then start an exploration." : "A data scientist or admin can start the first exploration.")];
  }

  async function explorationForm() {
    const sets = (await api("/v1/datasets")).filter((d) => d.version_count > 0);
    const back = h("p", null, h("button", { class: "link", onclick: () => { location.hash = "explorations"; } }, "← All explorations"));
    if (!sets.length) return [back, h("h1", null, "New exploration"), empty("No dataset versions yet", "Create a dataset and upload a file under Datasets first.")];
    const dset = h("select", { id: "ed" }, sets.map((d) => h("option", { value: d.id }, d.name)));
    const ver = h("select", { id: "ev" });
    const excl = h("div", { class: "checks" });
    const err = h("p", { class: "error", role: "alert" });
    const num = (id, v, min, max, step) => h("input", { id, type: "number", value: v, min, max, step: step || 1 });
    const mc = h("input", { id: "emc", type: "checkbox", checked: true }), ma = h("input", { id: "ema", type: "checkbox", checked: true });
    const cm = h("input", { id: "ecm", type: "checkbox" });
    cm.addEventListener("change", () => { if (cm.checked) { mc.checked = true; ma.checked = false; } });
    const kmin = num("ekmin", 2, 2, 12), kmax = num("ekmax", 8, 2, 12), kfix = num("ek", "", 2, 12);
    const share = num("efs", 5, 0.5, 20, 0.5), top = num("etop", 20, 5, 100), seed = num("es", 42, 0, 2147483647);
    const note = h("textarea", { id: "en", rows: 2, maxlength: 500, placeholder: "Optional: why you are looking at this data. Do not type names or ID numbers." });
    let versions = [];
    const cols = () => (versions.find((v) => String(v.version) === ver.value) || { schema_info: { columns: [] } }).schema_info.columns || [];
    const drawCols = () => excl.replaceChildren(...cols().map((c) => h("label", { class: "chk" }, h("input", { type: "checkbox", value: c.name }), " " + c.name)));
    const loadVersions = async () => {
      versions = (await api("/v1/datasets/" + dset.value)).versions;
      ver.replaceChildren(...versions.map((v) => h("option", { value: v.version }, "v" + v.version + " (" + v.row_count + " rows)")));
      drawCols();
    };
    dset.addEventListener("change", loadVersions); ver.addEventListener("change", drawCols);
    await loadVersions();
    const go = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      const methods = [mc.checked ? "clustering" : null, ma.checked ? "anomaly" : null].filter(Boolean);
      if (!methods.length) { err.textContent = "Choose clustering, anomaly detection or both."; return; }
      if (Number(kmin.value) > Number(kmax.value)) { err.textContent = "The smallest number of groups cannot be larger than the largest."; return; }
      go.disabled = true;
      try {
        const body = { dataset_id: Number(dset.value), version: Number(ver.value), methods, preset: cm.checked ? "cardiometabolic" : null, exclude: [...excl.querySelectorAll("input:checked")].map((i) => i.value),
          k_min: Number(kmin.value), k_max: Number(kmax.value), flag_share: Number(share.value) / 100, report_top: Number(top.value), seed: Number(seed.value), note: note.value.trim() };
        if (kfix.value !== "") body.k = Number(kfix.value);
        const x = await api("/v1/explorations", { method: "POST", body: JSON.stringify(body) });
        location.hash = "explorations/" + x.id;
      } catch (e) { err.textContent = e.message; go.disabled = false; }
    } }, "Start exploration");
    return [back, h("div", { class: "row" }, h("h1", null, "New exploration"), xpBadge()),
      h("div", { class: "notice" }, "Leave out identifier-like columns and any outcome or label column (for example a nutrition status, or the measurements it is calculated from). Otherwise the groups simply reproduce that column. This analysis produces no model and nothing from it can be deployed."),
      h("div", { class: "form3" }, field("Dataset", dset, "ed"), field("Version", ver, "ev")),
      h("label", null, "Columns to leave out"), excl,
      h("div", null, h("label", { for: "ecm" }, "Cardiometabolic phenotypes preset (clustering of waist, BMI, blood pressure, glucose, lipids, diet; checks the columns you chose)"), cm),
      h("div", { class: "form3" }, h("div", null, h("label", { for: "emc" }, "Look for groups (clustering)"), mc), h("div", null, h("label", { for: "ema" }, "Look for unusual rows (anomaly detection)"), ma)),
      h("details", { class: "card" }, h("summary", null, "Settings"),
        h("div", { class: "form3" }, field("Fewest groups to try", kmin, "ekmin"), field("Most groups to try", kmax, "ekmax"), field("Or set the number of groups (optional)", kfix, "ek")),
        h("div", { class: "form3" }, field("Share of rows to flag (%)", share, "efs"), field("Flagged rows to list", top, "etop"), field("Random seed", seed, "es")),
        field("Note", note, "en")),
      go, err];
  }

  function scatter(proj, k) {
    const NS = "http://www.w3.org/2000/svg", W = 560, H = 340, pad = 28;
    const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [a, v] of Object.entries(attrs || {})) e.setAttribute(a, v); return e; };
    const pts = proj.points || [];
    if (!pts.length) return null;
    const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const sx = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * pad), sy = (v) => H - pad - ((v - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
    const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", width: "100%", "aria-label": "Two-dimensional view of the rows, coloured by group, with unusual rows ringed" });
    svg.appendChild(el("rect", { x: 0.5, y: 0.5, width: W - 1, height: H - 1, fill: "none", stroke: "#c9d6da" }));
    pts.forEach(([x, y, g, f]) => svg.appendChild(el("circle", { cx: sx(x).toFixed(1), cy: sy(y).toFixed(1), r: f ? 5 : 3.2, fill: g >= 0 ? XP_COLORS[g % XP_COLORS.length] : "#7a8b99",
      "fill-opacity": f ? 0.95 : 0.55, stroke: f ? "#111" : "none", "stroke-width": f ? 1.6 : 0 })));
    const ev = proj.explained_variance || [];
    svg.appendChild(Object.assign(el("text", { x: W / 2, y: H - 8, "text-anchor": "middle", "font-size": 11, fill: "#5b6b72" }), { textContent: "Component 1 (" + pct(ev[0]) + " of the variation)" }));
    const yl = el("text", { x: 11, y: H / 2, "text-anchor": "middle", "font-size": 11, fill: "#5b6b72", transform: "rotate(-90 11 " + H / 2 + ")" });
    yl.textContent = "Component 2 (" + pct(ev[1]) + ")"; svg.appendChild(yl);
    const legend = k ? h("p", { class: "muted" }, Array.from({ length: k }, (_, g) => h("span", { class: "chk" }, h("span", { class: "swatch", "data-g": g }), " Group " + (g + 1) + "  "))) : null;
    if (legend) legend.querySelectorAll(".swatch").forEach((s) => { s.style.background = XP_COLORS[Number(s.dataset.g) % XP_COLORS.length]; });
    return h("div", null, svg, legend, h("p", { class: "muted" }, "Ringed points are flagged rows. " + proj.note));
  }

  async function explorationDetail(id) {
    const x = await api("/v1/explorations/" + id);
    const r = x.results_json || {}, pr = x.params_json || {}, cfg = pr.config || {};
    const out = [h("p", null, h("button", { class: "link", onclick: () => { location.hash = "explorations"; } }, "← All explorations")),
      h("div", { class: "row" }, h("h1", null, "Exploration #" + x.id), xpBadge(), statusBadge(x.status)),
      h("p", { class: "muted" }, "Dataset " + (pr.dataset_id || "?") + " v" + (pr.dataset_version || "?") + " · " + (cfg.methods || []).join(" + ") + " · started " + fmt(x.created_at) + " by " + (x.created_by || "—") + (pr.note ? " · " + pr.note : ""))];
    out.push(notesPanel("exploration", id));
    if (x.status === "queued" || x.status === "running") {
      out.push(h("div", { class: "notice" }, "The analysis is running. This page refreshes by itself."));
      poll = setTimeout(() => { if (location.hash.slice(1) === "explorations/" + id) show("explorations/" + id, true); }, 2500);
      return out;
    }
    if (x.status === "failed") { out.push(h("div", { class: "notice" }, r.error || "The analysis failed.")); return out; }
    const stat = (l, v, sub) => h("div", { class: "stat" }, h("b", null, v), h("span", null, l + (sub ? " · " + sub : "")));
    out.push(h("div", { class: "notice" }, r.notice));
    (r.warnings || []).forEach((w) => out.push(h("div", { class: "notice" }, w)));
    out.push(h("p", { class: "muted" }, r.n_rows + " rows" + (r.sampled ? " (fitted on a sample of " + r.n_used + ")" : "") + " · " + [...(r.features.numeric || []), ...(r.features.categorical || [])].length + " columns used."));
    const c = r.clustering, a = r.anomaly;
    if (r.preset) {
      const mg = r.preset.marker_groups || {};
      out.push(h("h2", null, "Cardiometabolic phenotypes"),
        h("div", { class: "card" },
          h("p", null, Object.keys(mg).length ? Object.entries(mg).map(([g, cs]) => g.replace("_", " ") + ": " + cs.join(", ")).join(" · ") : "No cardiometabolic measurements recognised."),
          (r.preset.not_recognised || []).length ? h("p", { class: "muted" }, "Used but not recognised as a cardiometabolic measurement: " + r.preset.not_recognised.join(", ")) : null,
          h("ul", { class: "plain" }, (r.preset.notes || []).map((q) => h("li", null, q)))));
    }
    if (c) {
      out.push(h("h2", null, "Groups (k-means, " + c.k + " groups)"), h("div", { class: "notice" }, h("b", null, (structureLabel[c.structure] || c.structure) + ". "), c.structure_summary),
        h("div", { class: "stats" }, stat("Silhouette", n3(c.quality.silhouette), "shuffled-data reference " + n3(c.reference.silhouette)),
          stat("Stability across refits", n3(c.stability.adjusted_rand_mean), "1 = identical groups"), stat("Davies–Bouldin", n3(c.quality.davies_bouldin), "lower is better")),
        h("p", { class: "muted" }, "Number of groups: " + c.k_selected_by + ". Silhouette ranges from −1 to 1; above about 0.5 is well separated, below 0.25 is no real structure."),
        h("details", { class: "card" }, h("summary", null, "Search over the number of groups"),
          table(["Groups", "Silhouette", "Smallest group"], c.k_search.map((q) => [q.k + (q.k === c.k ? " (chosen)" : ""), n3(q.silhouette), pct(q.smallest_group_share)]))),
        h("h2", null, "What each group looks like"), h("p", { class: "muted" }, "Averages over each group compared with all rows. A z of +1 means about one standard deviation above the overall mean. Groups under 5 rows are not described."));
      (c.groups || []).forEach((g) => out.push(h("div", { class: "card" },
        h("div", { class: "row" }, h("h3", null, "Group " + (g.cluster + 1)), h("span", { class: "muted" }, g.size + " rows · " + pct(g.share))),
        g.suppressed ? h("p", { class: "muted" }, g.note) : [
          h("p", null, (g.numeric || []).map((q) => q.feature + " " + signed(q.z) + " (mean " + n3(q.mean) + " vs " + n3(q.overall_mean) + ")").join(" · ") || "No numeric difference worth noting."),
          (g.categorical || []).length ? h("p", { class: "muted" }, g.categorical.map((q) => q.feature + ": " + q.level + " " + pct(q.share) + " vs " + pct(q.overall_share) + " overall").join(" · ")) : null])));
    }
    if (a) {
      out.push(h("h2", null, "Unusual rows (isolation forest)"),
        h("div", { class: "stats" }, stat("Rows flagged", a.n_flagged, pct(a.flag_share) + " by design"), stat("Same rows flagged on another seed", pct(a.stability.flagged_set_overlap)),
          stat("Rank agreement between seeds", n3(a.stability.rank_correlation_between_seeds))),
        h("p", { class: "muted" }, a.score_note + " " + a.row_index_note),
        table(["Rank", "Row", "Score", "Why it stands out"], (a.top || []).map((t) => [t.rank, t.row_index, n3(t.score),
          t.reasons.length ? t.reasons.map((q) => q.kind === "numeric" ? q.feature + " is " + q.direction + " (z " + signed(q.z) + ")" : q.feature + " has a rare value (" + pct(q.share_of_rows) + " of rows)").join("; ") : "An unusual combination of values rather than one extreme column"])),
        a.flagged_profile && a.flagged_profile.length ? h("p", { class: "muted" }, "Flagged rows differ most in: " + a.flagged_profile.slice(0, 4).map((q) => q.feature + " (mean z " + signed(q.mean_z_flagged) + ")").join(", ") + ".") : null);
      if (a.by_cluster && a.by_cluster.length) out.push(h("h3", null, "Flagged rows by group"), table(["Group", "Rows", "Flagged", "Share flagged"], a.by_cluster.map((q) => ["Group " + (q.cluster + 1), q.size, q.flagged, pct(q.flagged_share)])));
    }
    const sc = scatter(r.projection || {}, c ? c.k : 0);
    if (sc) out.push(h("h2", null, "Picture of the data"), sc);
    out.push(h("h2", null, "Limits of this result"), h("ul", { class: "plain" }, (r.limitations || []).map((q) => h("li", null, q))));
    return out;
  }


  // ---------- Matching: evaluation of food-name matching. Never registered, never deployed, results are not stored. ----------
  const mtBadge = () => h("span", { class: "badge explore" }, "suggest-only · needs approval");
  const mtVerdict = { learning_helps: "Learning from labelled queries helps", learning_hurts: "Learning from labelled queries hurts", no_clear_difference: "No clear difference from registry names alone" };

  async function matchingPage() {
    const sets = (await api("/v1/datasets")).filter((d) => d.version_count > 0);
    const intro = [h("div", { class: "row" }, h("h1", null, "Food-name matching"), mtBadge()),
      h("p", null, "Measure how well free-text food names (English, Chichewa, local or colloquial) can be matched to registry food codes, and whether learning from labelled queries beats the registry names alone."),
      h("div", { class: "notice" }, "Running an evaluation stores nothing: the result disappears when you leave the page. A matcher only reaches the registry if you register it below, and then it needs approval before an administrator can deploy it. It only ever suggests food codes for a person to confirm (never assigns them), is never shown to CNR, and the score is a similarity, not a probability.")];
    if (sets.length < 1) return [...intro, empty("No dataset versions yet", "Upload two datasets under Datasets: labelled queries (text and correct food code) and the registry food list.")];
    const sel = (id, opts) => h("select", { id }, opts);
    const dq = sel("mqd", sets.map((d) => h("option", { value: d.id }, d.name))), vq = sel("mqv");
    const dr = sel("mrd", sets.map((d) => h("option", { value: d.id }, d.name))), vr = sel("mrv");
    const colSel = (id, optional) => sel(id, [optional ? h("option", { value: "" }, "None") : null]);
    const cQ = colSel("mcq"), cF = colSel("mcf"), cG = colSel("mcg", true);
    const rC = colSel("mrc"), rL = colSel("mrl"), rA = colSel("mra", true);
    const num = (id, v, min, max, step) => h("input", { id, type: "number", value: v, min, max, step: step || 1 });
    const tsz = num("mts", 25, 10, 40), seed = num("mse", 42, 0, 2147483647), topk = num("mtk", 5, 1, 10), tgt = num("mta", 95, 50, 99.9, 0.5);
    const err = h("p", { class: "error", role: "alert" });
    const out = h("div", null);
    const state = { q: [], r: [] };
    const fill = (selEl, cols, optional, guess) => {
      selEl.replaceChildren(...[optional ? h("option", { value: "" }, "None") : null, ...cols.map((c) => h("option", { value: c.name }, c.name))]);
      const g = cols.find((c) => guess.some((s) => c.name.toLowerCase().includes(s)));
      if (g) selEl.value = g.name;
    };
    const loadQ = async () => {
      state.q = (await api("/v1/datasets/" + dq.value)).versions;
      vq.replaceChildren(...state.q.map((v) => h("option", { value: v.version }, "v" + v.version + " (" + v.row_count + " rows)")));
      drawQ();
    };
    const loadR = async () => {
      state.r = (await api("/v1/datasets/" + dr.value)).versions;
      vr.replaceChildren(...state.r.map((v) => h("option", { value: v.version }, "v" + v.version + " (" + v.row_count + " rows)")));
      drawR();
    };
    const colsOf = (list, v) => ((list.find((x) => String(x.version) === v.value) || { schema_info: { columns: [] } }).schema_info.columns || []);
    const drawQ = () => { const c = colsOf(state.q, vq); fill(cQ, c, false, ["search", "query", "text"]); fill(cF, c, false, ["code", "food"]); fill(cG, c, true, ["language", "source"]); cG.value = ""; };
    const drawR = () => { const c = colsOf(state.r, vr); fill(rC, c, false, ["code"]); fill(rL, c, false, ["label", "food"]); fill(rA, c, true, ["alias", "local"]); };
    dq.addEventListener("change", loadQ); vq.addEventListener("change", drawQ);
    dr.addEventListener("change", loadR); vr.addEventListener("change", drawR);
    await loadQ(); await loadR();
    if (sets.length > 1) dr.selectedIndex = Math.min(1, sets.length - 1), await loadR();

    const buildBody = () => ({ queries_dataset_id: Number(dq.value), queries_version: Number(vq.value), query_column: cQ.value, food_code_column: cF.value,
      group_column: cG.value || null, registry_dataset_id: Number(dr.value), registry_version: Number(vr.value), registry_code_column: rC.value,
      registry_label_column: rL.value, registry_aliases_column: rA.value || null, test_size: Number(tsz.value) / 100, seed: Number(seed.value),
      top_k: Number(topk.value), target_accuracy: Number(tgt.value) / 100 });
    const go = h("button", { class: "primary", onclick: async () => {
      err.textContent = ""; out.replaceChildren();
      go.disabled = true;
      try {
        const res = await api("/v1/matching/evaluations", { method: "POST", body: JSON.stringify(buildBody()) });
        out.replaceChildren(...matchingResults(res.results));
      } catch (e) { err.textContent = e.message; }
      go.disabled = false;
    } }, "Run evaluation");

    return [...intro,
      h("h3", null, "Labelled queries"),
      h("div", { class: "form3" }, field("Dataset", dq, "mqd"), field("Version", vq, "mqv")),
      h("div", { class: "form3" }, field("Query text column", cQ, "mcq"), field("Correct food code column", cF, "mcf"), field("Group column (optional, for example language)", cG, "mcg")),
      h("h3", null, "Registry food list"),
      h("div", { class: "form3" }, field("Dataset", dr, "mrd"), field("Version", vr, "mrv")),
      h("div", { class: "form3" }, field("Food code column", rC, "mrc"), field("Food label column", rL, "mrl"), field("Aliases column (optional, separated by ; or |)", rA, "mra")),
      h("details", { class: "card" }, h("summary", null, "Settings"),
        h("div", { class: "form3" }, field("Hold-out share (%)", tsz, "mts"), field("Candidates to score (top k)", topk, "mtk"), field("Target accuracy for review threshold (%)", tgt, "mta")),
        field("Random seed", seed, "mse")),
      go, err, out,
      can("write:models") ? matcherRegisterBox(buildBody) : null,
      can("predict:models") ? await tryBox() : null];
  }

  function matcherRegisterBox(buildBody) {
    const use = h("textarea", { id: "mu", rows: 3, maxlength: 2000, placeholder: "Who will use the suggestions and for what, for example: help dietitians find the registry food for a typed food name when entering survey data." });
    const err = h("p", { class: "error", role: "alert" });
    const send = h("button", { onclick: async () => {
      err.textContent = "";
      if (use.value.trim().length < 20) { err.textContent = "Describe the intended use in at least 20 characters."; return; }
      if (!confirm("Evaluate again with these settings and register the matcher as a candidate for review?")) return;
      send.disabled = true;
      try {
        const r = await api("/v1/matching/candidates", { method: "POST", body: JSON.stringify({ ...buildBody(), intended_use: use.value.trim() }) });
        location.hash = "registry";
        alert("Registered as version " + r.version + " (candidate). Open it under Registry and submit it for review.");
      } catch (e) { err.textContent = e.message; send.disabled = false; }
    } }, "Evaluate and register as a candidate");
    return h("details", { class: "card" }, h("summary", null, "Register this matcher for review"),
      h("p", { class: "muted" }, "Registers the matcher that was evaluated (fitted on the training part, threshold chosen on training data). A clinical reviewer or the food registry owner approves it, then an administrator deploys it. It only ever suggests candidates for a person to confirm."),
      field("Intended use", use, "mu"), send, err);
  }

  async function tryBox() {
    let info = null;
    try { info = await api("/v1/matching/deployed"); } catch { info = null; }
    const head = h("h3", null, "Try the deployed matcher");
    if (!info || !info.deployed) return h("div", null, head, h("p", { class: "muted" }, "No matcher is deployed yet. Register one, get it approved, then ask an administrator to deploy it."));
    const q = h("input", { id: "mtq", maxlength: 120, placeholder: "Type a food name, for example kapenta" });
    const res = h("div", null), err = h("p", { class: "error", role: "alert" });
    const go = h("button", { onclick: async () => {
      err.textContent = ""; res.replaceChildren();
      if (!q.value.trim()) { err.textContent = "Type a food name first."; return; }
      go.disabled = true;
      try {
        const r = await api("/v1/matching/suggest", { method: "POST", body: JSON.stringify({ queries: [q.value.trim()], top_k: 5 }) });
        const x = r.results[0];
        res.replaceChildren(h("p", { class: "muted" }, "Version " + r.model.version + ". Suggestions for a person to confirm; the similarity is not a probability."),
          x.candidates.length ? table(["Food code", "Similarity"], x.candidates.map((c) => [c.food_code, c.similarity])) : h("p", null, "No suggestion for this text."),
          x.below_review_threshold ? h("div", { class: "notice" }, "Below the review threshold: check this one carefully before using it.") : null);
      } catch (e) { err.textContent = e.message; }
      go.disabled = false;
    } }, "Suggest food codes");
    return h("div", null, head, h("p", { class: "muted" }, "Version " + info.model.version + " · approved by " + (info.model.approved_by || "—") + ". Typed text is not stored."), field("Food name", q, "mtq"), go, err, res);
  }

  function matchingResults(r) {
    const stat = (l, v, sub) => h("div", { class: "stat" }, h("b", null, v), h("span", null, l + (sub ? " · " + sub : "")));
    const rng = (a) => (a ? pct(a[0]) + " to " + pct(a[1]) : "—");
    const o = [h("h2", null, "Result"),
      h("p", { class: "muted" }, r.n_registry_foods + " registry foods · " + r.n_train + " training and " + r.n_test + " hold-out queries · " + r.n_foods_in_queries + " foods appear in the queries"),
      h("div", { class: "stats" },
        stat("Learned matcher, right first time", pct(r.learned.top1), "95% range " + rng(r.learned.top1_95_range)),
        stat("Registry names only", pct(r.names_only.top1), "95% range " + rng(r.names_only.top1_95_range)),
        stat("Right within top 3 (learned)", pct(r.learned.top3)),
        r.learned.top5 != null ? stat("Right within top 5 (learned)", pct(r.learned.top5)) : null,
        stat("Exact-match rule: answers", pct(r.exact_match_rule.coverage), "right when it answers " + pct(r.exact_match_rule.answered_accuracy)))];
    const le = r.learning_effect;
    o.push(h("div", { class: "row" }, h("b", null, "Does learning help?"), h("span", { class: "badge " + (le.verdict === "learning_hurts" ? "bad" : le.verdict === "learning_helps" ? "" : "planned") }, mtVerdict[le.verdict] || le.verdict)),
      h("p", { class: "muted" }, "Difference in right-first-time: " + signed(le.top1_difference == null ? null : Math.round(le.top1_difference * 1000) / 10) + " points (95% range " + rng(le["95_range"]) + "). " + le.note));
    const nt = r.not_already_a_registry_name;
    o.push(h("h3", null, "Queries that are not already a registry name"),
      nt.learned ? table(["", "Right first time", "Within top 3"], [["Learned matcher", pct(nt.learned.top1), pct(nt.learned.top3)], ["Registry names only", pct(nt.names_only.top1), pct(nt.names_only.top3)]])
        : h("p", { class: "muted" }, nt.note),
      h("p", { class: "muted" }, "Share of hold-out queries that are exactly a registry name or alias: " + pct(nt.trivial_share) + "."));
    const a = r.abstention;
    o.push(h("h3", null, "Review threshold"),
      h("p", { class: "muted" }, "Chosen on training data only: " + (a.threshold_chosen_on_training == null ? "none reached the target accuracy, so every match would need review" : a.threshold_chosen_on_training) + ". The score is a similarity, not a probability."));
    if (a.holdout_at_threshold) o.push(h("p", null, "On the hold-out set at that threshold: " + pct(a.holdout_at_threshold.coverage) + " answered automatically, " + pct(a.holdout_at_threshold.answered_accuracy) + " of those right."));
    o.push(table(["Threshold", "Answered", "Coverage", "Right when answered"], a.holdout_table.map((t) => [t.threshold, t.answered, pct(t.coverage), pct(t.answered_accuracy)])));
    if (r.groups && r.groups.length) o.push(h("h3", null, "By group"), table(["Group", "Hold-out queries", "Learned", "Names only"], r.groups.map((g) => [g.group, g.n_test, g.learned_top1 != null ? pct(g.learned_top1) : g.note, g.names_only_top1 != null ? pct(g.names_only_top1) : "—"])));
    if (r.most_confused_foods.length) o.push(h("h3", null, "Most confused foods"), table(["Correct code", "Predicted code", "Times"], r.most_confused_foods.map((c) => [c.true_food_code, c.predicted_food_code, c.count])));
    const c = r.cleaning;
    o.push(h("p", { class: "muted" }, "Cleaning: " + c.rows_in + " rows in, " + c.rows_used + " used. Dropped " + c.dropped_empty_or_too_long + " empty or too long, " + c.dropped_unknown_food_code + " with an unknown food code, " + c.dropped_conflicting_labels + " with conflicting labels."));
    (r.warnings || []).forEach((w) => o.push(h("div", { class: "notice" }, w)));
    o.push(h("p", { class: "muted" }, r.note));
    return o;
  }

  // ---------- Hotspots: district prevalence trends and hotspots from AGGREGATED counts. Never registered, never deployed. ----------
  const hsBadge = () => h("span", { class: "badge explore" }, "aggregated analysis · never deployed");
  const HS_COLORS = { hotspot: "#c0392b", watch: "#c98a1b", "not elevated": "#7a8b99" };
  const hsStatus = (st) => h("span", { class: "badge " + (st === "hotspot" ? "hot" : st === "watch" ? "watch" : "planned") }, st);
  const rr3 = (x) => (x == null ? "—" : Number(x).toFixed(2));
  const dirText = { rising: "Rising", falling: "Falling", "no clear change": "No clear change" };

  async function hotspots(arg) {
    if (arg === "new") return hotspotForm();
    if (arg) return hotspotDetail(Number(arg));
    const rows = await api("/v1/hotspots");
    const sum = (r) => {
      const s = r.summary || {};
      if (s.error) return "failed";
      if (s.areas == null) return "—";
      return [s.areas + " areas", s.hotspots + " hotspot" + (s.hotspots === 1 ? "" : "s"),
        s.trend ? "trend " + (dirText[s.trend.direction] || s.trend.direction).toLowerCase() : null,
        s.spatial ? (s.spatial.verdict === "clustering" ? "spatial clustering" : "no clear clustering") : null].filter(Boolean).join(" · ");
    };
    return [h("div", { class: "row" }, h("h1", null, "Hotspots"), hsBadge()),
      h("p", null, "Find districts, facilities or communities where prevalence is higher than expected, and see how prevalence is changing over time. This works on counts per area and period only. No individual child is used."),
      h("div", { class: "notice" }, "A hotspot analysis is never a model. It cannot be registered, reviewed, deployed, used for predictions or sent to CNR. It describes areas, not children, and a flag is a place to look first, not a diagnosis."),
      can("write:hotspots") ? h("p", null, h("button", { class: "primary", onclick: () => { location.hash = "hotspots/new"; } }, "New hotspot analysis")) : null,
      rows.length ? table(["#", "Status", "Result", "By", "Started", ""], rows.map((r) => [r.id, statusBadge(r.status), sum(r), r.created_by || "—", fmt(r.created_at),
        h("button", { class: "link", onclick: () => { location.hash = "hotspots/" + r.id; } }, "Open")]))
        : empty("No hotspot analyses yet", can("write:hotspots") ? "Upload a dataset version of counts per area (and period) first, then start an analysis." : "A data scientist or admin can start the first analysis.")];
  }

  async function hotspotForm() {
    const sets = (await api("/v1/datasets")).filter((d) => d.version_count > 0);
    const back = h("p", null, h("button", { class: "link", onclick: () => { location.hash = "hotspots"; } }, "← All hotspot analyses"));
    if (!sets.length) return [back, h("h1", null, "New hotspot analysis"), empty("No dataset versions yet", "Create a dataset and upload a file of counts per area under Datasets first.")];
    const dset = h("select", { id: "hd" }, sets.map((d) => h("option", { value: d.id }, d.name)));
    const ver = h("select", { id: "hv" });
    const pick = (id, optional) => h("select", { id }, optional ? [h("option", { value: "" }, "(none)")] : []);
    const selArea = pick("ha"), selCases = pick("hc"), selDen = pick("hn"), selPer = pick("hp", true), selLat = pick("hla", true), selLon = pick("hlo", true);
    const err = h("p", { class: "error", role: "alert" });
    const num = (id, v, min, max, step) => h("input", { id, type: "number", value: v, min, max, step: step || 1 });
    const smooth = h("select", { id: "hs" }, [["auto", "Automatic (spatial when coordinates are given)"], ["global", "Towards the overall rate"], ["spatial", "Towards nearby areas"]].map(([v, l]) => h("option", { value: v }, l)));
    const rr = num("hrr", 1, 0.5, 5, 0.05), cut = num("hcut", 95, 55, 99.9, 0.5), knn = num("hk", 4, 2, 10), perm = num("hperm", 999, 199, 9999), minp = num("hmp", 4, 3, 60), seed = num("hsd", 42, 0, 2147483647);
    const note = h("textarea", { id: "hnote", rows: 2, maxlength: 500, placeholder: "Optional: what this analysis is for. Do not type names or ID numbers." });
    let versions = [];
    const cols = () => ((versions.find((v) => String(v.version) === ver.value) || { schema_info: { columns: [] } }).schema_info.columns || []).map((c) => c.name);
    const guess = (names, re, fallback) => names.find((n) => re.test(n)) || fallback;
    const fill = (sel, names, chosen) => { sel.replaceChildren(...[...(sel.id === "hp" || sel.id === "hla" || sel.id === "hlo" ? [h("option", { value: "" }, "(none)")] : []), ...names.map((n) => h("option", { value: n }, n))]); sel.value = chosen || ""; };
    const drawCols = () => {
      const n = cols();
      fill(selArea, n, guess(n, /district|facility|communit|area|^ta$/, n[0]));
      fill(selCases, n, guess(n, /case|count|malnour|stunt|wast|sam|mam/, n[1]));
      fill(selDen, n, guess(n, /examin|screen|denom|total|children|pop/, n[2]));
      fill(selPer, n, guess(n, /month|year|period|date|week/, ""));
      fill(selLat, n, guess(n, /^lat|latitude/, ""));
      fill(selLon, n, guess(n, /^lon|lng|longitude/, ""));
    };
    const loadVersions = async () => {
      versions = (await api("/v1/datasets/" + dset.value)).versions;
      ver.replaceChildren(...versions.map((v) => h("option", { value: v.version }, "v" + v.version + " (" + v.row_count + " rows)")));
      drawCols();
    };
    dset.addEventListener("change", loadVersions); ver.addEventListener("change", drawCols);
    await loadVersions();
    const go = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      if (!!selLat.value !== !!selLon.value) { err.textContent = "Choose both a latitude and a longitude column, or neither."; return; }
      go.disabled = true;
      try {
        const body = { dataset_id: Number(dset.value), version: Number(ver.value), area_column: selArea.value, cases_column: selCases.value, denominator_column: selDen.value,
          period_column: selPer.value || null, latitude_column: selLat.value || null, longitude_column: selLon.value || null, smoothing: smooth.value,
          rr_threshold: Number(rr.value), exceedance_cutoff: Number(cut.value) / 100, k_neighbours: Number(knn.value), permutations: Number(perm.value),
          min_periods_trend: Number(minp.value), seed: Number(seed.value), note: note.value.trim() };
        const x = await api("/v1/hotspots", { method: "POST", body: JSON.stringify(body) });
        location.hash = "hotspots/" + x.id;
      } catch (e) { err.textContent = e.message; go.disabled = false; }
    } }, "Start analysis");
    return [back, h("div", { class: "row" }, h("h1", null, "New hotspot analysis"), hsBadge()),
      h("div", { class: "notice" }, "Use one row per area (and per period, for trends): the number of children with the condition, and the number examined. Never individual children. A period column enables trends. Centre-point latitude and longitude columns per area enable spatial smoothing and the clustering check; they are removed on upload unless you keep them with a written reason."),
      h("div", { class: "form3" }, field("Dataset", dset, "hd"), field("Version", ver, "hv")),
      h("div", { class: "form3" }, field("Area (district, facility or community)", selArea, "ha"), field("Cases (children with the condition)", selCases, "hc"), field("Children examined", selDen, "hn")),
      h("div", { class: "form3" }, field("Period (optional, for trends)", selPer, "hp"), field("Latitude of the area centre (optional)", selLat, "hla"), field("Longitude of the area centre (optional)", selLon, "hlo")),
      h("details", { class: "card" }, h("summary", null, "Settings"),
        h("div", { class: "form3" }, field("Smoothing", smooth, "hs"), field("Hotspot: prevalence above this multiple of the overall", rr, "hrr"), field("Hotspot: required probability (%)", cut, "hcut")),
        h("div", { class: "form3" }, field("Nearest areas used (spatial)", knn, "hk"), field("Permutations (clustering check)", perm, "hperm"), field("Fewest periods for a trend", minp, "hmp")),
        h("div", { class: "form3" }, field("Random seed", seed, "hsd")), field("Note", note, "hnote")),
      go, err];
  }

  function hsTrendChart(series) {
    const NS = "http://www.w3.org/2000/svg", W = 560, H = 260, pl = 46, pr = 14, pt = 14, pb = 34;
    const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [a, v] of Object.entries(attrs || {})) e.setAttribute(a, v); return e; };
    const pts = series.map((s, i) => ({ i, label: s.period, obs: s.prevalence, fit: s.fitted }));
    const vals = pts.flatMap((p) => [p.obs, p.fit]).filter((v) => v != null);
    if (pts.length < 2 || !vals.length) return null;
    const y1 = Math.max(...vals) * 1.1, y0 = Math.max(0, Math.min(...vals) * 0.9);
    const sx = (i) => pl + (i / (pts.length - 1)) * (W - pl - pr), sy = (v) => pt + (1 - (v - y0) / (y1 - y0 || 1)) * (H - pt - pb);
    const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", width: "100%", "aria-label": "Observed and fitted overall prevalence by period" });
    svg.appendChild(el("rect", { x: 0.5, y: 0.5, width: W - 1, height: H - 1, fill: "none", stroke: "#c9d6da" }));
    for (let g = 0; g <= 4; g++) {
      const v = y0 + ((y1 - y0) * g) / 4, y = sy(v).toFixed(1);
      svg.appendChild(el("line", { x1: pl, x2: W - pr, y1: y, y2: y, stroke: "#e3eaec" }));
      const t = el("text", { x: pl - 6, y: Number(y) + 4, "text-anchor": "end", "font-size": 10.5, fill: "#5b6b72" }); t.textContent = (100 * v).toFixed(1) + "%"; svg.appendChild(t);
    }
    const path = (key, color, dash) => {
      const p = pts.filter((q) => q[key] != null);
      if (p.length < 2) return;
      svg.appendChild(el("path", { d: p.map((q, k) => (k ? "L" : "M") + sx(q.i).toFixed(1) + " " + sy(q[key]).toFixed(1)).join(" "), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dash || "" }));
    };
    path("fit", "#0e5a73", "6 4");
    path("obs", "#c0392b");
    pts.filter((q) => q.obs != null).forEach((q) => svg.appendChild(el("circle", { cx: sx(q.i).toFixed(1), cy: sy(q.obs).toFixed(1), r: 3, fill: "#c0392b" })));
    const step = Math.max(1, Math.ceil(pts.length / 8));
    pts.forEach((q) => { if (q.i % step === 0 || q.i === pts.length - 1) { const t = el("text", { x: sx(q.i).toFixed(1), y: H - 12, "text-anchor": "middle", "font-size": 10.5, fill: "#5b6b72" }); t.textContent = q.label; svg.appendChild(t); } });
    return h("div", null, svg, h("p", { class: "muted" }, "Solid red: observed prevalence across all areas. Dashed blue: what the fitted model expects. Periods whose counts are too small to show are left out."));
  }

  function hsDotMap(areas) {
    const NS = "http://www.w3.org/2000/svg", W = 560, H = 420, pad = 26;
    const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [a, v] of Object.entries(attrs || {})) e.setAttribute(a, v); return e; };
    const pts = areas.filter((a) => a.centre);
    if (pts.length < 2) return null;
    const lats = pts.map((a) => a.centre[0]), lons = pts.map((a) => a.centre[1]);
    const [la0, la1, lo0, lo1] = [Math.min(...lats), Math.max(...lats), Math.min(...lons), Math.max(...lons)];
    // Keep north-up and roughly true to shape: one degree of longitude is shorter than one of latitude away from the equator.
    const k = Math.cos((((la0 + la1) / 2) * Math.PI) / 180);
    const spanX = (lo1 - lo0) * k || 1, spanY = la1 - la0 || 1, sc = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanY);
    const ox = (W - spanX * sc) / 2, oy = (H - spanY * sc) / 2;
    const sx = (lo) => ox + (lo - lo0) * k * sc, sy = (la) => H - oy - (la - la0) * sc;
    const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", width: "100%", "aria-label": "Schematic map of area centre points coloured by hotspot status" });
    svg.appendChild(el("rect", { x: 0.5, y: 0.5, width: W - 1, height: H - 1, fill: "none", stroke: "#c9d6da" }));
    [...pts].sort((a, b) => (a.status === "hotspot") - (b.status === "hotspot")).forEach((a) => {
      const r = 4 + Math.min(10, Math.max(0, (a.smoothed_ratio || 1) - 0.6) * 6);
      const c = el("circle", { cx: sx(a.centre[1]).toFixed(1), cy: sy(a.centre[0]).toFixed(1), r: r.toFixed(1), fill: HS_COLORS[a.status] || "#7a8b99", "fill-opacity": a.status === "not elevated" ? 0.45 : 0.85,
        stroke: a.cluster === "high cluster" ? "#111" : "none", "stroke-width": a.cluster === "high cluster" ? 2 : 0 });
      const t = el("title"); t.textContent = a.area + ": " + pct(a.smoothed_prevalence) + " (" + a.status + ")"; c.appendChild(t); svg.appendChild(c);
      if (a.status === "hotspot") { const l = el("text", { x: (sx(a.centre[1]) + r + 3).toFixed(1), y: (sy(a.centre[0]) + 4).toFixed(1), "font-size": 10.5, fill: "#333" }); l.textContent = a.area.slice(0, 18); svg.appendChild(l); }
    });
    const legend = h("p", { class: "muted" }, ["hotspot", "watch", "not elevated"].map((k2) => h("span", { class: "chk" }, h("span", { class: "swatch", "data-c": HS_COLORS[k2] }), " " + k2 + "  ")));
    legend.querySelectorAll(".swatch").forEach((s2) => { s2.style.background = s2.dataset.c; });
    return h("div", null, svg, legend, h("p", { class: "muted" }, "A schematic of area centre points (north is up), not a boundary map. Larger dots have a higher smoothed prevalence; a black ring marks a local high cluster."));
  }

  async function hotspotDetail(id) {
    const x = await api("/v1/hotspots/" + id);
    const r = x.results_json || {}, pr = x.params_json || {}, cfg = pr.config || {};
    const out = [h("p", null, h("button", { class: "link", onclick: () => { location.hash = "hotspots"; } }, "← All hotspot analyses")),
      h("div", { class: "row" }, h("h1", null, "Hotspot analysis #" + x.id), hsBadge(), statusBadge(x.status)),
      h("p", { class: "muted" }, "Dataset " + (pr.dataset_id || "?") + " v" + (pr.dataset_version || "?") + " · started " + fmt(x.created_at) + " by " + (x.created_by || "—") + (pr.note ? " · " + pr.note : ""))];
    out.push(notesPanel("hotspot", id));
    if (x.status === "queued" || x.status === "running") {
      out.push(h("div", { class: "notice" }, "The analysis is running. This page refreshes by itself."));
      poll = setTimeout(() => { if (location.hash.slice(1) === "hotspots/" + id) show("hotspots/" + id, true); }, 2500);
      return out;
    }
    if (x.status === "failed") { out.push(h("div", { class: "notice" }, r.error || "The analysis failed.")); return out; }
    const stat = (l, v, sub) => h("div", { class: "stat" }, h("b", null, v), h("span", null, l + (sub ? " · " + sub : "")));
    const d = r.data || {}, t = r.trend, sp = r.spatial, areas = r.areas || [];
    out.push(h("div", { class: "notice" }, r.notice));
    (r.warnings || []).forEach((w) => out.push(h("div", { class: "notice" }, w)));
    out.push(h("div", { class: "stats" }, stat("Areas", d.areas, d.periods ? d.periods + " periods" : "one snapshot"), stat("Overall prevalence", pct(r.overall.prevalence), d.total_examined + " children examined"),
      stat("Hotspots", (r.hotspots || []).length, "of " + d.areas + " areas"),
      t ? stat("Trend", dirText[t.overall.direction] || t.overall.direction, t.overall.rate_ratio_per_12_months != null ? "×" + rr3(t.overall.rate_ratio_per_12_months) + " per 12 months" : "×" + rr3(t.overall.rate_ratio_per_unit) + " per period") : null,
      sp ? stat("Spatial clustering", sp.global.verdict === "clustering" ? "Yes" : "Not clear", "Moran's I " + n3(sp.global.i) + ", p " + n3(sp.global.p_value)) : null));

    out.push(h("h2", null, "Hotspots"), h("p", { class: "muted" }, r.smoothing.hotspot_rule + " " + r.smoothing.description));
    const dot = hsDotMap(areas);
    if (dot) out.push(dot);
    const top = r.hotspots || [];
    out.push(top.length ? table(["Area", "Smoothed prevalence", "Probability above threshold", "Trend", "Local cluster"], top.map((q) => {
      const a = areas.find((z) => z.area === q.area) || {};
      return [q.area, pct(q.smoothed_prevalence) + (a.interval95 ? " (" + pct(a.interval95[0]) + "–" + pct(a.interval95[1]) + ")" : ""), pct(q.exceedance), q.rising ? "Rising" : ((a.trend || {}).direction ? dirText[a.trend.direction] : "—"), q.cluster || "—"];
    })) : h("div", { class: "notice" }, "No area stands out clearly. Either prevalence is similar everywhere, or there is too little data to show differences."));

    if (t) {
      out.push(h("h2", null, "Prevalence trend"),
        h("div", { class: "notice" }, h("b", null, (dirText[t.overall.direction] || t.overall.direction) + ". "), "Rate ratio " + rr3(t.overall.rate_ratio_per_unit) + " per " + t.unit + " (95% range " + rr3(t.overall.ci95[0]) + " to " + rr3(t.overall.ci95[1]) + ", p " + n3(t.overall.p_value) + "). A ratio under 1 means falling. " + t.model_choice));
      const chart = hsTrendChart(t.series || []);
      if (chart) out.push(chart);
      if (t.heterogeneity) out.push(h("p", { class: "muted" }, "Do areas move differently? " + (t.heterogeneity.p_value < 0.05 ? "Yes (p " + n3(t.heterogeneity.p_value) + "), so one common trend hides real differences. See the area trends below." : "No clear evidence (p " + n3(t.heterogeneity.p_value) + ").")));
    }
    if (sp) {
      out.push(h("h2", null, "Spatial clustering check"), h("div", { class: "notice" }, h("b", null, sp.global.verdict === "clustering" ? "Clustering found. " : "No clear clustering. "), sp.global.summary),
        h("p", { class: "muted" }, "Neighbours: " + sp.neighbours + ". Moran's I " + n3(sp.global.i) + " (about " + n3(sp.global.expected_under_no_clustering) + " if there were none), tested with " + sp.global.permutations + " permutations. " + sp.note));
    }

    out.push(h("h2", null, "All areas"), h("p", { class: "muted" }, "Sorted by smoothed prevalence relative to the overall. Counts under 5 are hidden; smoothed prevalence is always shown. Intervals are 95% ranges after smoothing."));
    out.push(table(["#", "Area", "Status", "Smoothed", "Ratio", "Probability", "Cases / examined", "Trend (rate ratio per " + (t ? t.unit : "period") + ")", "Cluster"],
      areas.map((a) => [a.rank, a.area, hsStatus(a.status), pct(a.smoothed_prevalence) + (a.low_precision ? " ⚠" : ""), rr3(a.smoothed_ratio), pct(a.exceedance),
        a.suppressed ? "hidden (small counts)" : a.cases + " / " + a.examined,
        a.trend ? (a.trend.status === "fitted" ? rr3(a.trend.rate_ratio_per_unit) + " (" + (dirText[a.trend.direction] || a.trend.direction).toLowerCase() + ")" : a.trend.status) : "—", a.cluster || "—"])));
    out.push(h("p", { class: "muted" }, "⚠ marks areas expecting very few cases, where the estimate leans on the overall rate."));
    out.push(h("h2", null, "Limits of this result"), h("ul", { class: "plain" }, (r.limitations || []).map((q) => h("li", null, q))));
    return out;
  }

  const syncText = { new_version: "new version", unchanged: "no change", no_data: "no data yet", error: "failed" };

  function syncSection(sy) {
    const box = h("div"), note = h("p", { class: "muted", role: "status" });
    const draw = (st) => {
      const lr = st.last_run;
      box.replaceChildren(...[
        st.configured
          ? h("p", null, "Pulls " + st.targets.join(", ") + " (last " + st.months + " months) from " + st.source + " every " + st.interval_hours + " h, stores each change as a new dataset version" + (st.run_analysis ? " and runs a hotspot analysis on it." : "."))
          : h("div", { class: "notice" }, st.problem + " To switch it on, set SURVEILLANCE_SYNC_ENABLED=true and SURVEILLANCE_API_KEY (a CNR admin key) on the API service."),
        lr ? h("p", { class: "muted" }, "Last run: " + (lr.trigger === "manual" ? "started by an admin" : "scheduled") + ", " + (lr.state === "running" ? "running now" : lr.state.replace(/_/g, " ")) + (lr.finished_at ? " at " + fmt(lr.finished_at) : "") + (lr.error ? ". " + lr.error : "") + ".") : null,
        lr && lr.results && lr.results.length
          ? table(["Group", "Indicator", "Result", "Dataset version", "Hotspot analysis"], lr.results.map((r) => [r.population_group, r.indicator,
              (syncText[r.status] || r.status) + (r.error ? ": " + r.error : r.note ? ": " + r.note : ""), r.version ? "v" + r.version : "—",
              r.analysis ? (r.analysis_id ? h("button", { class: "link", onclick: () => { location.hash = "hotspots/" + r.analysis_id; } }, r.analysis) : r.analysis) : "—"]))
          : null].filter(Boolean));   // replaceChildren(null) would print the word "null"
    };
    const btn = h("button", { class: "primary", disabled: !sy.configured, onclick: async () => {
      btn.disabled = true; note.textContent = "Starting…";
      try {
        await api("/v1/sync/surveillance/run", { method: "POST" });
        for (let i = 0; i < 40; i++) {
          await new Promise((r) => setTimeout(r, 3000));
          const st = await api("/v1/sync/surveillance");
          draw(st);
          if (!st.running) break;
          note.textContent = "Syncing…";
        }
        note.textContent = "Done.";
      } catch (e) { note.textContent = e.message; }
      finally { btn.disabled = !sy.configured; }
    } }, "Sync now");
    draw(sy);
    return h("div", null, h("h2", null, "Surveillance sync from CNR"), box, h("p", null, btn), note);
  }

  const trainText = { new_version: "new version", unchanged: "no change", no_data: "no data yet", no_source: "no CNR source yet", error: "failed" };

  function trainingSyncSection(ty) {
    const box = h("div"), note = h("p", { class: "muted", role: "status" });
    const draw = (st) => {
      const lr = st.last_run;
      const item = (m) => {
        const r = m.last_result;
        return h("div", { class: "tsrow" },
          h("b", null, m.name),
          m.available ? h("div", null, "Under-5 screening rows. Predict " + m.target + ".") : h("div", { class: "muted" }, "Not yet. " + m.reason),
          m.available ? h("div", { class: "muted" }, "Leave out: " + m.exclude.join(", ")) : null,
          m.dataset_name ? h("div", { class: "muted" }, "Dataset: " + m.dataset_name + (m.latest_version ? " v" + m.latest_version : "")) : null,
          r ? h("div", null, "Last result: " + (trainText[r.status] || r.status) + (r.error ? ": " + r.error : ""),
            r.label_counts ? h("div", { class: "muted" }, Object.entries(r.label_counts).map(([k, v]) => k + " " + v).join(", ") + (r.ready_to_train === false ? ". Not ready to train: " + r.not_ready_reason : ". Ready to train.")) : null) : null);
      };
      box.replaceChildren(...[
        st.configured
          ? h("p", null, "Pulls de-identified, record-level training data (last " + st.months + " months) from " + st.source + " every " + st.interval_hours + " h and stores each change as a new dataset version. It only fetches data: it never trains, registers or deploys a model.")
          : h("div", { class: "notice" }, st.problem + " To switch it on, set TRAINING_SYNC_ENABLED=true on the API service (it uses the same SURVEILLANCE_API_KEY)."),
        lr ? h("p", { class: "muted" }, "Last run: " + (lr.trigger === "manual" ? "started by an admin" : "scheduled") + ", " + (lr.state === "running" ? "running now" : lr.state.replace(/_/g, " ")) + (lr.finished_at ? " at " + fmt(lr.finished_at) : "") + (lr.error ? ". " + lr.error : "") + ".") : null,
        h("div", { class: "tslist" }, st.models.map(item))].filter(Boolean));
    };
    const btn = h("button", { class: "primary", disabled: !ty.configured, onclick: async () => {
      btn.disabled = true; note.textContent = "Starting…";
      try {
        await api("/v1/sync/training/run", { method: "POST" });
        for (let i = 0; i < 40; i++) {
          await new Promise((r) => setTimeout(r, 3000));
          const st = await api("/v1/sync/training");
          draw(st);
          if (!st.running) break;
          note.textContent = "Syncing…";
        }
        note.textContent = "Done.";
      } catch (e) { note.textContent = e.message; }
      finally { btn.disabled = !ty.configured; }
    } }, "Sync now");
    draw(ty);
    return h("div", null, h("h2", null, "Training data sync from CNR"), box, h("p", null, btn), note);
  }

  async function integrationPage() {
    const s = await api("/v1/integration/cnr/status");
    const sy = await api("/v1/sync/surveillance").catch(() => null);
    const ty = await api("/v1/sync/training").catch(() => null);
    const out = h("div"), probeOut = h("p", { class: "muted", role: "status" });
    const c = s.cnr_api;
    const probeBtn = h("button", { class: "primary", disabled: !c.configured, onclick: async () => {
      probeBtn.disabled = true; probeOut.textContent = "Checking…";
      try {
        const r = (await api("/v1/integration/cnr/status?probe=true")).probe;
        probeOut.textContent = r.reachable ? "CNR API answered (HTTP " + r.http_status + ", " + r.latency_ms + " ms)." : "Not reachable: " + r.reason;
      } catch (e) { probeOut.textContent = e.message; }
      finally { probeBtn.disabled = !c.configured; }
    } }, "Check CNR API");
    out.append(
      h("p", { class: "muted" }, "CNR calls the laboratory with a service key. The laboratory sends CNR no patient data and no credentials. Contract: " + s.contract + "."),
      h("div", { class: "stats" },
        [["Deployed models", s.deployed_models], ["CNR assessments (24 h)", s.last_24h.assessments], ["Refused (24 h)", s.last_24h.refused], ["Active service keys", s.service_keys.length]]
          .map(([l, n]) => h("div", { class: "stat" }, h("b", null, n), h("span", null, l)))),
      h("h2", null, "CNR API address"),
      c.configured ? h("p", null, c.base_url + c.health_path) : h("div", { class: "notice" }, c.problem + " Set CNR_API_BASE_URL on the API service."),
      h("p", null, probeBtn), probeOut,
      h("h2", null, "Events sent to CNR"),
      s.webhook.configured
        ? h("p", null, "Model deploy, rollback and withdrawal events are sent to " + s.webhook.host + ". Last 24 h: " + s.webhook.last_24h.delivered + " delivered of " + s.webhook.last_24h.attempted + " attempted.")
        : h("div", { class: "notice" }, s.webhook.problem + " CNR will still learn the truth on each assessment, but will not be told when a model changes."),
      h("h2", null, "Service keys"));
    if (sy) out.append(syncSection(sy));
    if (ty) out.append(trainingSyncSection(ty));
    if (!s.service_keys.length) out.append(empty("No service key yet", "Create a key with the service role under Access and store it only as a secret in CNR."));
    else out.append(table(["Name", "Prefix", "Created", "Last used"], s.service_keys.map((k) => [k.name, k.prefix, fmt(k.created_at), fmt(k.last_used_at)])));
    if (!s.deployed_models) out.append(h("div", { class: "notice" }, "No model is deployed, so CNR assessments are refused with a clear message."));
    return [h("h1", null, "Chakudya Nutrition Registry (CNR) integration"), out];
  }

  async function predictPage() {
    const list = await api("/v1/predict");
    const out = [h("h1", null, "Predict"),
      h("div", { class: "notice" }, list.safety_notice + " Enter measurements only. Never type names, phone numbers or ID numbers; the service rejects fields it does not know.")];
    if (!list.models.length) return [...out, empty("No model is deployed", "A model is served only after it is approved by a clinical reviewer and deployed by an administrator under Registry.")];
    const pick = h("select", { id: "pm" }, list.models.map((m) => h("option", { value: m.slug }, m.name + " (v" + m.version + ")")));
    const area = h("div");
    const draw = async () => {
      area.replaceChildren(h("p", null, "Loading…"));
      try { area.replaceChildren(await predictForm(await api("/v1/predict/" + pick.value))); }
      catch (e) { area.replaceChildren(h("div", { class: "notice" }, e.message)); }
    };
    pick.addEventListener("change", draw);
    await draw();
    return [...out, field("Model", pick, "pm"), area];
  }

  function predictForm(s) {
    const inputs = {};
    const grid = h("div", { class: "form3" });
    s.features.numeric.forEach((n) => { inputs[n] = h("input", { id: "pf-" + n, type: "number", step: "any" }); grid.append(field(n, inputs[n], "pf-" + n)); });
    s.features.categorical.forEach((c) => {
      inputs[c.name] = h("select", { id: "pf-" + c.name }, [h("option", { value: "" }, "(not recorded)")].concat(c.values.map((v) => h("option", { value: v }, v))));
      grid.append(field(c.name, inputs[c.name], "pf-" + c.name));
    });
    const result = h("div"), err = h("p", { class: "error", role: "alert" });
    const go = h("button", { class: "primary", onclick: async () => {
      err.textContent = ""; result.replaceChildren(); go.disabled = true;
      const rec = {};
      for (const [k, el] of Object.entries(inputs)) if (el.value !== "") rec[k] = el.tagName === "SELECT" || el.type !== "number" ? el.value : Number(el.value);
      try { result.replaceChildren(...showPrediction(await api("/v1/predict/" + s.slug, { method: "POST", body: JSON.stringify({ records: [rec] }) }))); }
      catch (e) { err.textContent = e.message; }
      finally { go.disabled = false; }
    } }, "Estimate risk");
    const ev = s.evaluation || {};
    return h("div", null,
      h("p", { class: "muted" }, "Version " + s.version + " · approved by " + (s.approved_by || "—") + " · evidence: " + (s.clinical_validation_ref || "—") + " · test " + (ev.metric || "") + " " + n3(ev.test) + " vs baseline " + n3(ev.baseline) + " on " + n3(ev.n_test) + " rows"),
      h("p", { class: "muted" }, "Intended use: " + (s.intended_use || "—")),
      grid, h("p", null, go), err, result);
  }

  function showPrediction(b) {
    const r = b.results[0], ri = r.risk_indicators, out = [];
    out.push(h("div", { class: "notice" }, b.safety_notice));
    if (ri.low_confidence) out.push(h("div", { class: "notice" }, "Low confidence: the model is unsure or key measurements are missing. Do not rely on this estimate."));
    r.notes.forEach((n) => out.push(h("div", { class: "notice" }, n)));
    if (r.predicted_class != null) {
      out.push(h("h2", null, "Estimated: " + r.predicted_class), h("p", { class: "muted" }, "A model estimate, not a diagnosis. Scores are not calibrated probabilities."),
        h("div", { class: "bars" }, Object.entries(r.probabilities).map(([k, p]) => h("div", { class: "barrow" }, h("span", null, k), h("div", { class: "bar" }, h("i", { style: "width:" + Math.max(1, p * 100) + "%" })), h("span", null, pct(p))))));
    } else {
      out.push(h("h2", null, "Estimated value: " + r.predicted_value), h("p", { class: "muted" }, "A model estimate, not a diagnosis." + (b.model.evaluation.rmse != null ? " Typical error on test data (RMSE): " + n3(b.model.evaluation.rmse) + "." : "")));
    }
    const ex = r.explanation;
    if (ex) {
      out.push(h("h2", null, "Why this estimate"),
        ex.drivers.length ? h("ul", { class: "plain" }, ex.drivers.map((d) => h("li", null, d.feature + " " + d.direction + " the estimate for " + ex.toward + " (" + (d.effect > 0 ? "+" : "") + d.effect + " log-odds)"))) : h("p", { class: "muted" }, "No single feature stood out."),
        ex.not_given.length ? h("p", { class: "muted" }, "Not entered, so not scored: " + ex.not_given.join(", ") + ".") : null,
        h("p", { class: "muted" }, ex.note));
    }
    out.push(h("ul", { class: "plain" }, b.caveats.map((c) => h("li", null, c))),
      h("small", null, "Request " + b.request_id + " · " + b.model.name + " v" + b.model.version + " · " + fmt(b.generated_at) + ". Inputs are not stored."));
    return out;
  }

  const VSTATUS = { candidate: "planned", in_review: "planned", approved: "", deployed: "", rejected: "bad", retired: "planned" };
  const vBadge = (st) => h("span", { class: "badge " + (VSTATUS[st] ?? "planned") }, st.replace("_", " "));

  function registerBox(e) {
    const use = h("textarea", { id: "ru", rows: 3, maxlength: 2000, placeholder: "Intended use and setting, e.g. screening support for children under five at district clinics. Not a diagnosis." });
    const err = h("p", { class: "error", role: "alert" });
    const go = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      if (use.value.trim().length < 20) { err.textContent = "Describe the intended use in at least 20 characters."; return; }
      go.disabled = true;
      try { const v = await api("/v1/models/versions", { method: "POST", body: JSON.stringify({ experiment_id: e.id, intended_use: use.value.trim() }) }); location.hash = "registry/" + v.id; }
      catch (x) { err.textContent = x.message; go.disabled = false; }
    } }, "Register as model version");
    return h("details", { class: "card" }, h("summary", null, "Register this run in the model registry"),
      h("p", { class: "muted" }, "Registering saves a frozen copy of the model and its results. It does not make the model available. It must still be submitted, reviewed and approved by a different person."),
      field("Intended use", use, "ru"), go, err);
  }

  async function registry(arg) {
    if (arg) return versionDetail(Number(arg));
    const rows = await api("/v1/models/versions?limit=200");
    return [h("h1", null, "Model registry"),
      h("div", { class: "notice" }, "A version moves from candidate to in review to approved, and only then can an administrator deploy it. Deploying makes the version the one served by the prediction API."),
      rows.length ? table(["Model", "Version", "Status", "Result vs baseline", "Registered by", "Registered", ""], rows.map((v) => [v.model_name || v.model_slug, "v" + v.version, vBadge(v.status),
        v.headline.test != null ? v.headline.metric + " " + n3(v.headline.test) + " vs " + n3(v.headline.baseline) + (v.headline.warnings ? " · " + v.headline.warnings + " warning(s)" : "") : "—",
        v.created_by || "—", fmt(v.created_at), h("button", { class: "link", onclick: () => { location.hash = "registry/" + v.id; } }, "Open")]))
        : empty("No registered models", can("write:models") ? "Open a completed experiment and register it to start the review process." : "A data scientist can register a completed experiment.")];
  }

  async function versionDetail(id) {
    const v = await api("/v1/models/versions/" + id);
    const m = v.metrics_json || {}, t = m.test || {}, b = m.baseline || {};
    const isMatch = m.task === "matching";
    const canReview = can("approve:models") || (isMatch && can("approve:matchers"));
    const back = () => { location.hash = "registry"; };
    const reload = () => show("registry/" + id);
    const run = (label, path, body, ask, cls) => h("button", { class: cls || "primary", onclick: async (ev) => {
      if (ask && !confirm(ask)) return;
      ev.target.disabled = true;
      try { await api(path, { method: "POST", body: JSON.stringify(body || {}) }); reload(); } catch (e) { alert(e.message); ev.target.disabled = false; }
    } }, label);
    const out = [h("p", null, h("button", { class: "link", onclick: back }, "← Registry")),
      h("div", { class: "row" }, h("h1", null, (v.model_name || v.model_slug) + " · v" + v.version), vBadge(v.status)),
      h("p", { class: "muted" }, (isMatch ? "Food-name matching evaluation" : "Experiment #" + v.experiment_id) + " · registered by " + (v.created_by || "—") + " on " + fmt(v.created_at) + (v.artifact_sha256 ? " · file SHA-256 " + v.artifact_sha256.slice(0, 12) + "…" : "")),
      h("div", { class: "notice" }, "Intended use: " + (v.intended_use || "—") + " " + (isMatch ? "Output is a list of suggestions for a person to confirm, never an automatic assignment." : "Outputs are risk estimates to support clinical judgement, never confirmed diagnoses."))];
    out.push(notesPanel("model_version", id));
    (m.warnings || []).forEach((w) => out.push(h("div", { class: "notice" }, w)));
    const stat = (l, x, sub) => h("div", { class: "stat" }, h("b", null, x), h("span", null, l + (sub ? " · " + sub : "")));
    out.push(h("h2", null, "Evaluation (frozen at registration)"));
    if (isMatch) out.push(...matchingResults({ ...m, warnings: [] }).slice(1));
    else out.push(h("div", { class: "stats" }, m.task === "classification"
      ? [stat("Macro F1", n3(t.macro_f1), "95% range " + n3(t.macro_f1_ci95 && t.macro_f1_ci95[0]) + "–" + n3(t.macro_f1_ci95 && t.macro_f1_ci95[1])), stat("Baseline macro F1", n3(b.macro_f1)), stat("Balanced accuracy", pct(t.balanced_accuracy)), stat("Test rows", m.n_test)]
      : [stat("R²", n3(t.r2), "baseline " + n3(b.r2)), stat("RMSE", n3(t.rmse)), stat("Test rows", m.n_test)]));
    if (!isMatch && t && Object.keys(t).length) out.push(perfChart(m), m.task === "classification" ? perClassChart(m.per_class) : null);
    if (!isMatch) out.push(h("p", null, h("button", { class: "link", onclick: () => { location.hash = "experiments/" + v.experiment_id; } }, "Open the full experiment report")));

    // actions
    const act = [];
    if (v.status === "candidate" && can("write:models")) act.push(h("p", null, run("Submit for clinical review", "/v1/models/versions/" + id + "/submit", {}, "Submit this version for clinical review?")));
    if (v.status === "in_review" && canReview) act.push(reviewForm(v, reload));
    if (v.status === "in_review" && !canReview) act.push(h("div", { class: "notice" }, isMatch ? "Waiting for a clinical reviewer or the food registry owner." : "Waiting for a clinical reviewer."));
    if ((v.status === "approved" || (v.status === "retired" && (v.retired_reason || "") === "superseded")) && can("deploy:models"))
      act.push(h("p", null, run("Deploy this version", "/v1/models/versions/" + id + "/deploy", {}, "Mark v" + v.version + " as the version to serve? The current one will be retired.")));
    if (v.status === "deployed" && can("deploy:models")) act.push(h("p", { class: "inline" },
      run("Roll back to previous version", "/v1/models/" + v.model_slug + "/rollback", {}, "Roll back? This version will be retired and cannot be redeployed.", "danger"),
      retireButton(v, reload)));
    if (v.status === "approved" && can("deploy:models")) act.push(retireButton(v, reload));
    if (act.length) out.push(h("h2", null, "Actions"), ...act);

    if (v.status !== "candidate" && v.status !== "in_review") out.push(h("h2", null, "Review"),
      h("p", null, (v.approved_by ? "Approved by " + v.approved_by + " on " + fmt(v.reviewed_at) : "Decision recorded " + fmt(v.reviewed_at)) + (v.clinical_validation_ref ? " · evidence: " + v.clinical_validation_ref : "")),
      h("p", null, v.review_notes || ""), v.warnings_acknowledged ? h("small", null, "The reviewer acknowledged the evaluation warnings.") : null,
      v.retired_reason ? h("p", { class: "muted" }, "Retired: " + v.retired_reason) : null);
    out.push(h("h2", null, "History"), table(["When", "Who", "What", "Note"], (v.history || []).map((x) => [fmt(x.ts), x.by, x.action, x.note || ""])));
    return out;
  }

  function retireButton(v, reload) {
    return h("button", { class: "danger", onclick: async (ev) => {
      const why = prompt("Why is this version being withdrawn? (10+ characters)");
      if (!why) return;
      try { await api("/v1/models/versions/" + v.id + "/retire", { method: "POST", body: JSON.stringify({ reason: why.trim() }) }); reload(); } catch (e) { alert(e.message); }
    } }, "Withdraw this version");
  }

  function reviewForm(v, reload) {
    const ref = h("input", { id: "rv", maxlength: 500, placeholder: "e.g. COMREC-2026-017 external validation report" });
    const notes = h("textarea", { id: "rn", rows: 3, maxlength: 3000, placeholder: "What you checked: per-class results, features used, limits, whether the setting matches the intended use." });
    const ack = h("input", { id: "ra", type: "checkbox" });
    const err = h("p", { class: "error", role: "alert" });
    const send = (path, body) => async (ev) => {
      err.textContent = ""; ev.target.disabled = true;
      try { await api("/v1/models/versions/" + v.id + path, { method: "POST", body: JSON.stringify(body()) }); reload(); }
      catch (e) { err.textContent = e.message; ev.target.disabled = false; }
    };
    const warn = (v.headline && v.headline.warnings) || 0;
    return h("div", { class: "card" }, h("h3", null, "Clinical review"),
      h("p", { class: "muted" }, "You must be a different person from the one who registered or submitted this version. Approval needs a reference to the external validation evidence."),
      field("Validation evidence reference", ref, "rv"), field("Review notes", notes, "rn"),
      warn ? h("div", null, h("label", { for: "ra" }, "I have read the " + warn + " evaluation warning(s) above and considered them"), ack) : null,
      h("div", { class: "inline" },
        h("button", { class: "primary", onclick: send("/approve", () => ({ clinical_validation_ref: ref.value.trim(), review_notes: notes.value.trim(), acknowledge_warnings: ack.checked })) }, "Approve"),
        h("button", { class: "danger", onclick: send("/reject", () => ({ review_notes: notes.value.trim() })) }, "Reject")), err);
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
    out.push(notesPanel("dataset", id));
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

  // ---------- NLP corpora, training runs and model review ----------
  // One page with three tabs (Corpora, Training runs, Models). Routes: corpora, corpora/<id>, corpora/runs, corpora/run-<id>,
  // corpora/models, corpora/model-<id>. Nothing here serves a model: approval only records that people reviewed it.
  const LANGS = { en: "English", ny: "Chichewa", tum: "Tumbuka", yao: "Yao", mixed: "Mixed" };
  const langName = (c) => LANGS[c] || c;
  const labelName = (k) => ENT_NAMES[k] || k;
  const VERDICT_CLASS = { "model better": "", "rules better": "watch" };
  const verdictBadge = (v) => (v ? h("span", { class: "badge " + (VERDICT_CLASS[v] ?? "planned") }, v) : "");
  const trialBadge = () => h("span", { class: "badge watch" }, "trial · not for use");
  const nlpStat = (l, n) => h("div", { class: "stat" }, h("b", null, n), h("span", null, l));

  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = h("a", { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function nlpTabs(active) {
    const tabs = [["corpora", "Corpora", true], ["runs", "Training runs", can("read:experiments")], ["models", "Models", can("read:models")]];
    return h("div", { class: "tabs", role: "navigation", "aria-label": "NLP laboratory sections" }, tabs.filter((t) => t[2]).map(([id, label]) =>
      h("button", { "aria-current": id === active ? "page" : false, onclick: () => { location.hash = id === "corpora" ? "corpora" : "corpora/" + id; } }, label)));
  }

  async function corporaPage(arg) {
    const a = arg || "";
    let tab = "corpora", body;
    if (a === "runs") { tab = "runs"; body = await trainingRuns(); }
    else if (/^run-\d+$/.test(a)) { tab = "runs"; body = await trainingRunDetail(Number(a.slice(4))); }
    else if (a === "models") { tab = "models"; body = await nlpModelList(); }
    else if (/^model-\d+$/.test(a)) { tab = "models"; body = await nlpModelDetail(Number(a.slice(6))); }
    else if (/^\d+$/.test(a)) body = await corpusDetail(Number(a));
    else body = await corporaHome();
    return [h("h1", null, "NLP corpora and training"), nlpTabs(tab), [].concat(body).flat(Infinity)];
  }

  const backLink = (hash, text) => h("p", null, h("button", { class: "link", onclick: () => { location.hash = hash; } }, "← " + text));

  // ----- corpora -----
  async function corporaHome() {
    const [rows, fmts] = await Promise.all([api("/v1/nlp/corpora"), api("/v1/nlp/corpora/format")]);
    const out = [h("div", { class: "notice" }, "A corpus is a set of annotated, de-identified sentences used to train and test entity models. A file with identifiers (titled names, phone numbers, emails, record or ID numbers) is refused whole. Only the text, language and entities are kept, and the audit log holds counts, never text.")];
    out.push(h("h2", null, "Corpora"));
    out.push(rows.length
      ? table(["Corpus", "Versions", "Created by", "Created", ""], rows.map((c) => [c.name, c.versions, c.created_by || "—", fmt(c.created_at), h("button", { class: "link", onclick: () => { location.hash = "corpora/" + c.id; } }, "Open")]))
      : empty("No corpora yet", can("write:datasets") ? "Download the annotation template, have speakers fill it in, then upload it below." : "A data scientist can upload an annotated file."));
    out.push(annotationTools());
    if (can("write:datasets")) out.push(corpusUploader(fmts, null));
    return out;
  }

  function annotationTools() {
    const err = h("p", { class: "error", role: "alert" });
    const file = h("input", { id: "pl-file", type: "file", accept: ".txt,.csv" });
    const lang = h("select", { id: "pl-lang" }, h("option", { value: "" }, "Guess from the text"), Object.entries(LANGS).map(([k, v]) => h("option", { value: k }, v)));
    const tpl = h("button", { onclick: async () => {
      err.textContent = "";
      try { saveBlob(await api("/v1/nlp/corpora/template", { blob: true }), "cnr-nlp-annotation-template.csv"); } catch (e) { err.textContent = e.message; }
    } }, "Download annotation template");
    const out = [h("p", null, tpl, " A blank sheet with example rows in English, Chichewa, Tumbuka and Yao (the Yao row is a placeholder). Delete the example rows before uploading.")];
    if (can("run:nlp")) {
      const pre = h("button", { class: "primary", onclick: async () => {
        err.textContent = "";
        const f = file.files[0];
        if (!f) { err.textContent = "Choose a .txt file (one sentence per line) or a .csv file with a text column first."; return; }
        const fd = new FormData(); fd.append("file", f); if (lang.value) fd.append("lang", lang.value);
        try { saveBlob(await api("/v1/nlp/corpora/prelabel", { method: "POST", body: fd, blob: true }), "cnr-nlp-prelabel.csv"); }
        catch (e) { err.textContent = e.message; }
      } }, "Download pre-labelled CSV");
      out.push(h("p", null, "Pre-label: upload plain sentences and get back a CSV holding the rule engine's guesses for speakers to correct. The corrected file uploads as it is. Nothing is stored, and a file with identifiers is refused whole."),
        h("div", { class: "form3" }, field("Sentences (.txt or .csv)", file, "pl-file"), field("Language", lang, "pl-lang")), pre);
    }
    return h("div", { class: "card" }, h("h3", null, "Annotation tools"), ...out, err);
  }

  function listOf(items, total) {
    return h("ul", { class: "plain" }, (items || []).map((x) => h("li", null, typeof x === "string" ? x : JSON.stringify(x))),
      total > (items || []).length ? h("li", { class: "muted" }, "and " + (total - items.length) + " more") : null);
  }

  function problemsView(r) {
    const out = [];
    if (r.identifier_lines_total) out.push(h("div", { class: "notice" }, h("b", null, "Identifiers found. "), r.identifier_lines_total + " line(s) look like they hold personal identifiers" + (Object.keys(r.identifier_types || {}).length ? " (" + Object.keys(r.identifier_types).join(", ") + ")" : "") + ". The file is refused whole. Remove them and check again."), listOf(r.identifier_lines, r.identifier_lines_total));
    if (r.errors_total) out.push(h("div", { class: "notice" }, h("b", null, r.errors_total + " problem(s) in the file. "), "Fix them and check again."), listOf(r.errors, r.errors_total));
    return out;
  }

  function reportView(rep) {
    const labels = Object.keys(rep.by_label || {}).sort();
    const langs = Object.keys(rep.by_language || {}).sort();
    return [
      h("div", { class: "stats" }, nlpStat("Documents", rep.documents), nlpStat("Entities", rep.entities), nlpStat("Duplicates removed", rep.duplicates_removed || 0), nlpStat("Source format", rep.source_format || "")),
      langs.length ? table(["Language", "Documents", ...labels.map(labelName)], langs.map((l) => [langName(l), rep.by_language[l], ...labels.map((k) => ((rep.label_by_language || {})[l] || {})[k] || 0)])) : null,
      labels.length ? h("p", { class: "muted" }, "Entities per label: " + labels.map((k) => labelName(k) + " " + rep.by_label[k]).join(", ") + ".") : null,
      ...(rep.warnings || []).map((w) => h("div", { class: "notice" }, w)),
    ];
  }

  // Upload in two steps, like datasets: check (stores nothing), then store. corpus is null for a new corpus.
  function corpusUploader(fmts, corpus) {
    const root = h("details", { class: "card" }, h("summary", null, corpus ? "Add a version" : "Upload a new corpus"));
    const box = h("div"); root.append(box);
    const err = h("p", { class: "error", role: "alert" });
    const result = h("div");
    const file = h("input", { id: "cu-file", type: "file", accept: ".jsonl,.ndjson,.json,.csv,.bio,.iob,.iob2,.conll,.spacy,.zip" });
    const fsel = h("select", { id: "cu-fmt" }, h("option", { value: "auto" }, "Detect from the file name"), Object.keys(fmts.formats).map((k) => h("option", { value: k }, k)));
    const lsel = h("select", { id: "cu-lang" }, h("option", { value: "" }, "From the file"), Object.entries(LANGS).map(([k, v]) => h("option", { value: k }, v)));
    const lmap = h("textarea", { id: "cu-map", rows: 2, placeholder: 'Optional. Rename labels from another dataset, e.g. {"FOOD_ITEM": "FOOD", "PER": ""}', spellcheck: "false" });
    const guide = h("details", null, h("summary", null, "Format guide"),
      h("ul", { class: "plain" }, Object.entries(fmts.formats).map(([k, v]) => h("li", null, h("b", null, k + ": "), v))),
      h("ul", { class: "plain" }, (fmts.rules || []).map((r) => h("li", null, r))));
    const build = () => {
      const fd = new FormData(); fd.append("file", file.files[0]); fd.append("format", fsel.value);
      if (lsel.value) fd.append("lang", lsel.value);
      if (lmap.value.trim()) fd.append("label_map", lmap.value.trim());
      return fd;
    };
    const check = h("button", { class: "primary", onclick: async () => {
      err.textContent = ""; result.replaceChildren();
      if (!file.files[0]) { err.textContent = "Choose a file first."; return; }
      try {
        const r = await api("/v1/nlp/corpora/inspect", { method: "POST", body: build() });
        result.replaceChildren(...[].concat(r.valid ? storeStep(r) : [h("div", { class: "notice" }, "This file would not be stored."), ...problemsView(r)]).flat());
      } catch (e) { err.textContent = e.message; if (e.detail) result.replaceChildren(...problemsView(e.detail)); }
    } }, "Check file");
    function storeStep(r) {
      const name = h("input", { id: "cu-name", maxlength: 160, placeholder: "e.g. Clinical notes, Zomba pilot" });
      const desc = h("textarea", { id: "cu-desc", rows: 2, maxlength: 2000, placeholder: "Where the text came from and who annotated it." });
      const appr = h("input", { id: "cu-ref", maxlength: 300, placeholder: "Data-sharing or ethics approval reference (required)" });
      const err2 = h("p", { class: "error", role: "alert" });
      const go = h("button", { class: "primary", onclick: async () => {
        err2.textContent = "";
        if (!corpus && name.value.trim().length < 2) { err2.textContent = "Give the corpus a name."; return; }
        if (appr.value.trim().length < 5) { err2.textContent = "Give the approval reference (at least 5 characters)."; return; }
        const fd = build(); fd.append("approval_ref", appr.value.trim());
        if (!corpus) { fd.append("name", name.value.trim()); fd.append("description", desc.value.trim()); }
        go.disabled = true;
        try {
          const made = await api(corpus ? "/v1/nlp/corpora/" + corpus.id + "/versions" : "/v1/nlp/corpora", { method: "POST", body: fd });
          const target = "corpora/" + (corpus ? corpus.id : made.id);
          location.hash = target; show(target);
        } catch (e) { err2.textContent = e.message; if (e.detail) result.replaceChildren(...problemsView(e.detail)); go.disabled = false; }
      } }, corpus ? "Store as a new version" : "Store corpus");
      return [h("div", { class: "notice" }, "The file is valid. Nothing is stored yet."), ...reportView(r.report),
        h("div", { class: "form3" }, corpus ? null : field("Corpus name", name, "cu-name"), field("Approval reference", appr, "cu-ref")),
        corpus ? null : field("Description", desc, "cu-desc"), go, err2];
    }
    box.append(h("p", null, "Step 1: choose an annotated file. It is checked and nothing is stored. Step 2: review the report and store it."),
      h("div", { class: "form3" }, field("File", file, "cu-file"), field("Format", fsel, "cu-fmt"), field("Language for documents without one", lsel, "cu-lang")),
      field("Label map", lmap, "cu-map"), guide, check, err, result);
    return root;
  }

  async function corpusDetail(id) {
    const [c, fmts, limits] = await Promise.all([api("/v1/nlp/corpora/" + id), can("write:datasets") ? api("/v1/nlp/corpora/format") : null, can("write:experiments") ? api("/v1/nlp/training/limits") : null]);
    const out = [backLink("corpora", "All corpora"), h("h2", null, c.name), c.description ? h("p", null, c.description) : null];
    out.push(table(["Version", "Documents", "Entities", "Approval reference", "Added by", "Added", "SHA-256"], c.versions.map((v) =>
      ["v" + v.version, v.documents, v.entities, v.approval_ref, v.created_by || "—", fmt(v.created_at), (v.sha256 || "").slice(0, 12) + "…"])));
    c.versions.forEach((v, i) => out.push(h("details", { class: "card", open: i === 0 }, h("summary", null, "Version " + v.version + " report"), ...reportView(v.report || {}))));
    if (limits) out.push(trainingForm(c, limits));
    if (fmts) out.push(corpusUploader(fmts, c));
    return out;
  }

  // ----- training -----
  function trainingForm(c, limits) {
    const m = limits.minimums;
    const num = (id, val, min, max, step) => h("input", { id, type: "number", value: val, min, max, step });
    const ver = h("select", { id: "tv" }, c.versions.map((v) => h("option", { value: v.version }, "Version " + v.version + " (" + v.documents + " documents)")));
    const iter = num("tn", 30, 5, 100, 1), drop = num("td", 0.2, 0, 0.5, 0.05), test = num("tt", 0.2, 0.1, 0.4, 0.05), seed = num("ts", 1, 0, 2147483647, 1);
    const trial = h("input", { id: "tr", type: "checkbox" });
    const err = h("p", { class: "error", role: "alert" });
    const go = h("button", { class: "primary", onclick: async () => {
      err.textContent = ""; go.disabled = true;
      try {
        const r = await api("/v1/nlp/training", { method: "POST", body: JSON.stringify({ corpus_id: c.id, version: Number(ver.value), n_iter: Number(iter.value), dropout: Number(drop.value), test_size: Number(test.value), seed: Number(seed.value), trial: trial.checked }) });
        location.hash = "corpora/run-" + r.id;
      } catch (e) { err.textContent = e.message; go.disabled = false; }
    } }, "Start training");
    return h("details", { class: "card" }, h("summary", null, "Train an entity model on this corpus"),
      h("p", null, "Trains a spaCy entity model from scratch and scores it on held-out documents against the rule baseline, on the same documents. The number of passes is fixed up front and nothing is tuned on the test set."),
      h("div", { class: "notice" }, "A normal run needs at least " + m.documents + " documents and " + m.examples_per_label + " examples of every label that appears; smaller corpora are refused. A trial run lowers this to " + m.trial_documents + " documents and " + m.trial_examples_per_label + " examples so the pipeline can be tried. Its results are marked trial and cannot be registered or used."),
      h("div", { class: "form3" }, field("Corpus version", ver, "tv"), field("Passes over the data (5 to 100)", iter, "tn"), field("Dropout (0 to 0.5)", drop, "td"), field("Share held out for testing (0.1 to 0.4)", test, "tt"), field("Random seed", seed, "ts")),
      h("div", null, h("label", { for: "tr" }, "Trial run (tiny corpus, results not for use)"), trial), go, err);
  }

  async function trainingRuns() {
    const rows = await api("/v1/nlp/training");
    if (rows.some((r) => r.status === "queued" || r.status === "running")) poll = setTimeout(() => { if (location.hash.slice(1) === "corpora/runs") show("corpora/runs", true); }, 3000);
    return [h("p", null, "Each run trains a spaCy entity model on one corpus version and scores it on held-out documents against the rule baseline (nlp-rules-1). Runs marked trial used a tiny corpus and are not for use. Start a run from a corpus page."),
      rows.length ? table(["Run", "Corpus", "Status", "Model F1", "Rules F1", "Verdict", "Started by", "Started", ""], rows.map((r) => {
        const s = r.summary || {};
        return ["#" + r.id, (r.corpus_name || "?") + " v" + (r.corpus_version || "?"), h("span", null, statusBadge(r.status), r.trial ? [" ", trialBadge()] : null),
          n3(s.macro_f1_model), n3(s.macro_f1_rules), r.status === "failed" ? h("span", { class: "muted" }, s.error || "failed") : verdictBadge(s.overall_verdict),
          r.created_by || "—", fmt(r.created_at), h("button", { class: "link", onclick: () => { location.hash = "corpora/run-" + r.id; } }, "Open")];
      })) : empty("No training runs yet", can("write:experiments") ? "Open a corpus and start a run." : "A data scientist can start a run.")];
  }

  // Tables shared by a training run and a registered model: results by language, by label and by language and label.
  function evalSections(res) {
    const model = res.model || {}, rules = res.rules || {}, cmp = res.comparison || {}, split = res.split || {};
    const trainL = split.train_by_language || {}, testL = split.test_by_language || {};
    const langs = [...new Set([...Object.keys(trainL), ...Object.keys(testL)])].sort();
    const labels = (res.labels && res.labels.length ? res.labels : Object.keys(model.per_label || {})).slice().sort();
    const out = [h("h3", null, "By language"),
      table(["Language", "Training documents", "Test documents", "Test entities", "Model F1", "Rules F1", "Difference", "Verdict"], langs.map((l) => {
        const m = (model.per_language || {})[l] || {}, ru = (rules.per_language || {})[l] || {}, v = (cmp.per_language || {})[l] || {};
        return [langName(l), trainL[l] || 0, testL[l] || 0, m.support ?? 0, n3(m.f1), n3(ru.f1), signed(v.difference),
          testL[l] ? verdictBadge(v.verdict) : h("span", { class: "badge bad" }, "not evaluated")];
      })),
      nlpF1Chart(langs.map(langName), langs.map((l) => ((model.per_language || {})[l] || {}).f1), langs.map((l) => ((rules.per_language || {})[l] || {}).f1)),
      h("h3", null, "By label"),
      nlpF1Chart(labels.map(labelName), labels.map((k) => ((model.per_label || {})[k] || {}).f1), labels.map((k) => ((rules.per_label || {})[k] || {}).f1)),
      table(["Label", "Training examples", "Test entities", "Precision", "Recall", "Model F1", "Rules F1", "Difference", "Verdict"], labels.map((k) => {
        const m = (model.per_label || {})[k] || {}, ru = (rules.per_label || {})[k] || {}, v = (cmp.per_label || {})[k] || {};
        return [labelName(k), (split.train_by_label || {})[k] || 0, m.support ?? 0, n3(m.precision), n3(m.recall), n3(m.f1), n3(ru.f1), signed(v.difference), verdictBadge(v.verdict)];
      }))];
    const cell = (l, k) => {
      const m = ((model.per_language_label || {})[l] || {})[k], ru = ((rules.per_language_label || {})[l] || {})[k];
      return m ? n3(m.f1) + " / " + n3(ru && ru.f1) + " (n=" + m.support + ")" : "";
    };
    out.push(h("details", { class: "card" }, h("summary", null, "Language by label: model F1 / rules F1 (test entities)"),
      table(["Language", ...labels.map(labelName)], langs.map((l) => [langName(l), ...labels.map((k) => cell(l, k))]))));
    const loss = res.loss_by_epoch || [];
    if (loss.length > 1) out.push(lineChart(loss, "Training loss per pass. It should fall and level off."));
    if (loss.length) out.push(h("p", { class: "muted" }, "Training loss over " + loss.length + " passes: " + loss[0] + " at the start, " + loss[loss.length - 1] + " at the end."));
    return out;
  }

  async function trainingRunDetail(id) {
    const r = await api("/v1/nlp/training/" + id);
    const res = r.results || {};
    const out = [backLink("corpora/runs", "All training runs"),
      h("div", { class: "row" }, h("h2", null, "Training run #" + r.id), h("span", null, statusBadge(r.status), r.trial ? [" ", trialBadge()] : null)),
      h("p", { class: "muted" }, (r.corpus_name || "Corpus") + " v" + (r.corpus_version || "?") + " · started " + fmt(r.created_at) + " by " + (r.created_by || "—") + (r.finished_at ? " · finished " + fmt(r.finished_at) : ""))];
    if (r.status === "queued" || r.status === "running") {
      out.push(h("div", { class: "notice" }, "Training is running. This page refreshes by itself."));
      poll = setTimeout(() => { if (location.hash.slice(1) === "corpora/run-" + id) show("corpora/run-" + id, true); }, 3000);
      return out;
    }
    if (r.status === "failed") { out.push(h("div", { class: "notice" }, res.error || "The run failed.")); return out; }
    const cfg = res.config || {}, cmp = (res.comparison || {}).overall || {}, split = res.split || {};
    if (r.trial) out.push(h("div", { class: "notice" }, h("b", null, "Trial run. "), "The corpus was too small for a trustworthy model. These results show that the pipeline works and nothing else. A trial run cannot be registered."));
    if (res.notice) out.push(h("div", { class: "notice" }, res.notice));
    (res.warnings || []).forEach((w) => out.push(h("div", { class: "notice" }, w)));
    if (res.skipped_misaligned) out.push(h("div", { class: "notice" }, res.skipped_misaligned + " document(s) were skipped because an entity boundary fell inside a word."));
    out.push(h("p", { class: "muted" }, (res.engine || "spacy-ner-xx") + " · baseline " + (res.baseline_engine || "nlp-rules-1") + " · " + (cfg.n_iter || "?") + " passes, dropout " + (cfg.dropout ?? "?") + ", test share " + (cfg.test_size ?? "?") + ", seed " + (cfg.seed ?? "?")));
    out.push(h("div", { class: "stats" }, nlpStat("Model macro F1", n3((res.model || {}).macro_f1)), nlpStat("Rules macro F1", n3((res.rules || {}).macro_f1)),
      nlpStat("Training documents", split.train_documents ?? "—"), nlpStat("Test documents", split.test_documents ?? "—")));
    out.push(h("p", null, "Overall: ", verdictBadge(cmp.verdict), cmp.difference != null ? " (F1 difference " + signed(cmp.difference) + " on " + cmp.support + " test entities)" : "",
      h("span", { class: "muted" }, " " + ((res.comparison || {}).note || ""))));
    out.push(...evalSections(res));
    if (!r.trial && can("write:models")) out.push(registerNlpBox(r));
    else if (!r.trial) out.push(h("p", { class: "muted" }, "A data scientist can register this run in the model registry."));
    return out;
  }

  function registerNlpBox(r) {
    const name = h("input", { id: "rn-name", maxlength: 64, placeholder: "nlp-ner-xx" });
    const use = h("textarea", { id: "rn-use", rows: 3, maxlength: 2000, placeholder: "Intended use and setting, e.g. suggesting entities in de-identified clinical notes for a person to confirm. Not a diagnosis." });
    const err = h("p", { class: "error", role: "alert" });
    const go = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      if (name.value.trim().length < 3) { err.textContent = "Give the model a name of at least 3 characters: lower-case letters, digits and hyphens."; return; }
      if (use.value.trim().length < 20) { err.textContent = "Describe the intended use in at least 20 characters."; return; }
      go.disabled = true;
      try { const v = await api("/v1/nlp/models", { method: "POST", body: JSON.stringify({ training_run_id: r.id, name: name.value.trim().toLowerCase(), intended_use: use.value.trim() }) }); location.hash = "corpora/model-" + v.id; }
      catch (e) { err.textContent = e.message; go.disabled = false; }
    } }, "Register as model version");
    return h("details", { class: "card" }, h("summary", null, "Register this run in the NLP model registry"),
      h("p", { class: "muted" }, "Registering saves a frozen copy of the model file and its results. It does not make the model available: nothing serves an NLP model yet. A different person must review it, language by language."),
      h("div", { class: "form3" }, field("Model name", name, "rn-name")), field("Intended use", use, "rn-use"), go, err);
  }

  // ----- model registry and review -----
  async function nlpModelList() {
    const rows = await api("/v1/nlp/models?limit=200");
    return [h("div", { class: "notice" }, "A version moves from candidate to in review to approved or rejected. Approval is per language and only records that a different person read the evaluation and accepted the model for the languages listed. It does not make the model clinically validated, and nothing serves an NLP model yet."),
      rows.length ? table(["Model", "Version", "Status", "Model F1", "Rules F1", "Verdict", "Approved languages", "Registered by", "Registered", ""], rows.map((v) => {
        const hl = v.headline || {};
        return [v.name, "v" + v.version, vBadge(v.status), n3(hl.macro_f1_model), n3(hl.macro_f1_rules), verdictBadge(hl.overall_verdict),
          (v.approved_languages || []).map(langName).join(", ") || "none", v.created_by || "—", fmt(v.created_at), h("button", { class: "link", onclick: () => { location.hash = "corpora/model-" + v.id; } }, "Open")];
      })) : empty("No registered NLP models", can("write:models") ? "Open a completed, non-trial training run and register it to start the review." : "A data scientist can register a completed training run.")];
  }

  async function nlpModelDetail(id) {
    const v = await api("/v1/nlp/models/" + id);
    const m = v.metrics_json || {}, lr = v.language_review || {};
    const reload = () => show("corpora/model-" + id);
    const out = [backLink("corpora/models", "All NLP models"),
      h("div", { class: "row" }, h("h2", null, v.name + " · v" + v.version), vBadge(v.status)),
      h("p", { class: "muted" }, "Training run ", h("button", { class: "link", onclick: () => { location.hash = "corpora/run-" + v.training_run_id; } }, "#" + v.training_run_id),
        " · registered by " + (v.created_by || "—") + " on " + fmt(v.created_at) + (v.artifact_sha256 ? " · file SHA-256 " + v.artifact_sha256.slice(0, 12) + "…" : "")),
      h("div", { class: "notice" }, "Intended use: " + (v.intended_use || "—")),
      v.notice ? h("div", { class: "notice" }, v.notice) : null];
    if (v.approved_languages && v.approved_languages.length) out.push(h("p", null, h("b", null, "Approved for: "), v.approved_languages.map(langName).join(", ") + "."));
    out.push(h("h2", null, "Review by language"),
      table(["Language", "Documents", "Test documents", "Test entities", "Model F1", "Rules F1", "Verdict", "Flags"], Object.entries(lr).map(([l, x]) =>
        [langName(l), x.documents, x.test_documents, x.test_entities, n3(x.f1_model), n3(x.f1_rules), verdictBadge(x.verdict),
          x.flags.length ? h("div", null, h("span", { class: "badge " + (x.blocking ? "bad" : "watch") }, x.blocking ? "cannot be approved" : "needs acknowledgement"), h("ul", { class: "plain" }, x.flags.map((f) => h("li", null, f)))) : h("span", { class: "muted" }, "none")])));
    (m.warnings || []).forEach((w) => out.push(h("div", { class: "notice" }, w)));
    if (m.skipped_misaligned) out.push(h("div", { class: "notice" }, m.skipped_misaligned + " document(s) were skipped in training or testing because an entity boundary fell inside a word."));
    out.push(h("h2", null, "Evaluation (frozen at registration)"), ...evalSections(m));

    const act = [];
    if (v.status === "candidate" && can("write:models")) act.push(h("p", null, h("button", { class: "primary", onclick: async (ev) => {
      if (!confirm("Submit this version for review?")) return;
      ev.target.disabled = true;
      try { await api("/v1/nlp/models/" + id + "/submit", { method: "POST", body: JSON.stringify({}) }); reload(); } catch (e) { alert(e.message); ev.target.disabled = false; }
    } }, "Submit for review")));
    if (v.status === "in_review" && can("approve:models")) act.push(nlpReviewForm(v, m, reload));
    if (v.status === "in_review" && !can("approve:models")) act.push(h("div", { class: "notice" }, "Waiting for a clinical reviewer."));
    if (v.status === "approved" && can("approve:models")) act.push(h("p", null, h("button", { class: "danger", onclick: async () => {
      const why = prompt("Why is this version being withdrawn? (10+ characters)");
      if (!why) return;
      try { await api("/v1/nlp/models/" + id + "/retire", { method: "POST", body: JSON.stringify({ reason: why.trim() }) }); reload(); } catch (e) { alert(e.message); }
    } }, "Withdraw this version")));
    if (act.length) out.push(h("h2", null, "Actions"), ...act);

    if (v.status !== "candidate" && v.status !== "in_review") out.push(h("h2", null, "Review"),
      h("p", null, (v.approved_by ? "Approved by " + v.approved_by + " on " + fmt(v.reviewed_at) : "Decision recorded " + fmt(v.reviewed_at)) + (v.clinical_validation_ref ? " · evidence: " + v.clinical_validation_ref : "")),
      h("p", null, v.review_notes || ""), v.warnings_acknowledged ? h("small", null, "The reviewer acknowledged the flags and warnings.") : null,
      v.retired_reason ? h("p", { class: "muted" }, "Retired: " + v.retired_reason) : null);
    out.push(h("h2", null, "History"), table(["When", "Who", "What", "Note"], (v.history || []).map((x) => [fmt(x.ts), x.by, x.action, x.note || ""])));
    return out;
  }

  function nlpReviewForm(v, m, reload) {
    const lr = v.language_review || {};
    const boxes = Object.entries(lr).map(([l, x]) => ({ l, x, cb: h("input", { type: "checkbox", id: "al-" + l, disabled: x.blocking }) }));
    const anyFlag = Object.values(lr).some((x) => x.flags.length && !x.blocking) || Boolean(m.skipped_misaligned);
    const ref = h("input", { id: "nr-ref", maxlength: 500, placeholder: "e.g. COMREC-2026-017 expert review of Chichewa and English output" });
    const notes = h("textarea", { id: "nr-notes", rows: 3, maxlength: 3000, placeholder: "What you checked: per-label and per-language results against the rules, the flags above, whether the setting matches the intended use." });
    const ack = h("input", { id: "nr-ack", type: "checkbox" });
    const err = h("p", { class: "error", role: "alert" });
    const send = (path, body, check) => async (ev) => {
      err.textContent = "";
      const problem = check(); if (problem) { err.textContent = problem; return; }
      ev.target.disabled = true;
      try { await api("/v1/nlp/models/" + v.id + path, { method: "POST", body: JSON.stringify(body()) }); reload(); }
      catch (e) { err.textContent = e.message; ev.target.disabled = false; }
    };
    const chosen = () => boxes.filter((b) => b.cb.checked).map((b) => b.l);
    return h("div", { class: "card" }, h("h3", null, "Clinical review"),
      h("p", { class: "muted" }, "You must be a different person from the one who registered or submitted this version. Tick only the languages you are signing off. A language that was not tested on held-out text cannot be approved."),
      h("div", null, boxes.map(({ l, x, cb }) => h("div", null, h("label", { for: "al-" + l }, langName(l) + (x.blocking ? " (not evaluated, cannot be approved)" : x.flags.length ? " (flagged, see above)" : "")), cb))),
      field("Validation evidence reference", ref, "nr-ref"), field("Review notes", notes, "nr-notes"),
      anyFlag ? h("div", null, h("label", { for: "nr-ack" }, "I have read the flags and warnings above and considered them"), ack) : null,
      h("div", { class: "inline" },
        h("button", { class: "primary", onclick: send("/approve", () => ({ approved_languages: chosen(), clinical_validation_ref: ref.value.trim(), review_notes: notes.value.trim(), acknowledge_warnings: ack.checked }), () =>
          !chosen().length ? "Tick at least one language to approve." : ref.value.trim().length < 5 ? "Give the validation evidence reference (5+ characters)." : notes.value.trim().length < 20 ? "Write review notes of at least 20 characters." : "") }, "Approve selected languages"),
        h("button", { class: "danger", onclick: send("/reject", () => ({ review_notes: notes.value.trim() }), () => (notes.value.trim().length < 10 ? "Write the reason for rejecting in at least 10 characters." : "")) }, "Reject")), err);
  }

  async function audit() {
    const rows = await api("/v1/audit?limit=100");
    return [h("h1", null, "Audit log"), h("p", null, "The 100 most recent events. Secrets and request bodies are never recorded."),
      rows.length ? table(["Time", "Actor", "Role", "Action", "Resource"], rows.map((r) => [fmt(r.ts), r.actor_name || "unauthenticated", r.role || "—", r.action, r.resource_type ? r.resource_type + " " + (r.resource_id || "") : "—"])) : empty("No events yet", "")];
  }

  async function access() {
    const out = h("div");
    const name = h("input", { id: "kn", maxlength: 120 });
    const role = h("select", { id: "kr" }, ["viewer", "data_scientist", "clinical_reviewer", "registry_owner", "service", "admin"].map((r) => h("option", { value: r }, r)));
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

  // ---------- NLP laboratory: structured nutrition information from free text (English and Chichewa). Text is never stored. ----------
  const NLP_SAMPLES = [
    ["Clinical note", "A 14 month old child presented with severe acute malnutrition. MUAC 10.8 cm, weight 6.2 kg and bilateral pitting oedema. No fever, denies vomiting. Hb 7.9 g/dL. Started F-75 and RUTF. Plan: nutrition counselling for the mother."],
    ["English and Chichewa", "The child was admitted with swelling of the feet. Mwana sakudya nsima, amadya phala ndi mkaka wa m'mawere, akutsegula m'mimba kwa masabata awiri. Palibe kusanza. Mother says he has had no fever."],
    ["Dietary record", "24-hour recall: breakfast was 2 cups of porridge with sugar and 1 egg. Lunch: 3 plates of nsima with beans and pumpkin leaves. Supper: nsima with usipa. Drank tea and water. No milk."],
  ];
  const ENT_NAMES = { FOOD: "Food", CONDITION: "Condition", SYMPTOM: "Symptom", INTERVENTION: "Intervention", NUTRIENT: "Nutrient", POPULATION: "Patient group", MEASUREMENT: "Measurement", PHI: "Identifier" };
  const LANG_NAMES = { en: "English", ny: "Chichewa", tum: "Tumbuka", yao: "Yao", mixed: "More than one language", unknown: "Not enough words to tell" };
  const pct1 = (x) => (x == null ? "—" : (x * 100).toFixed(0) + "%");

  function highlight(text, ents) {
    const nodes = []; let pos = 0;
    for (const e of ents) {
      if (e.start < pos) continue;
      nodes.push(text.slice(pos, e.start));
      const a = e.attributes || {}, flags = ["negated", "uncertain", "historical", "family"].filter((k) => a[k]);
      const tip = ENT_NAMES[e.label] + ": " + e.norm + (e.value != null ? " = " + e.value + " " + (e.unit || "") : "") + (flags.length ? " (" + flags.join(", ") + ")" : "");
      nodes.push(h("span", { class: "ent ent-" + e.label + (a.negated ? " neg" : ""), title: tip }, e.label === "PHI" ? "[" + e.norm + "]" : text.slice(e.start, e.end)));
      pos = e.end;
    }
    nodes.push(text.slice(pos));
    return h("p", { class: "notebody entview" }, nodes);
  }

  function termList(title, rows, cls) {
    return rows.length ? h("p", null, h("b", null, title + ": "), rows.map((r, i) => [i ? ", " : "", h("span", { class: cls || "" }, r.term + (r.count > 1 ? " ×" + r.count : ""))])) : null;
  }

  function nlpResult(text, r) {
    const rec = r.record, cl = r.classification, lang = r.language;
    const out = [];
    if (r.identifiers.found) out.push(h("div", { class: "notice" }, h("b", null, "Identifiers found (" + Object.entries(r.identifiers.types).map(([k, n]) => k.toLowerCase() + " ×" + n).join(", ") + "). "), "They are masked below and were not stored. Remove identifiers at the source; do not process identifiable notes."));
    out.push(h("div", { class: "stats" },
      h("div", { class: "stat" }, h("b", null, LANG_NAMES[lang.overall]), h("span", null, "Language" + (lang.code_switching ? " · switches between languages" : ""))),
      h("div", { class: "stat" }, h("b", null, cl.document_type.label.replace("_", " ")), h("span", null, "Looks like (by cue words)")),
      h("div", { class: "stat" }, h("b", null, r.entities.filter((e) => e.label !== "PHI").length), h("span", null, "Things found"))));
    out.push(h("h2", null, "Text with entities"), h("p", { class: "legend" }, Object.keys(ENT_NAMES).map((k) => h("span", { class: "ent ent-" + k }, ENT_NAMES[k]))),
      highlight(text, r.entities), h("p", { class: "muted" }, "Struck-through items are negated (for example “no fever”). Hover an item for its normalised form."));
    out.push(h("h2", null, "Structured record"));
    const rows = Object.entries(rec.measurements).flatMap(([k, list]) => list.map((m) => [k.replace("_", " "), m.value + (m.unit ? " " + m.unit : ""), m.text, [m.food ? "of " + m.food : null, m.plausible === false ? "check: implausible" : null].filter(Boolean).join("; ")]));
    out.push(termList("Conditions", rec.conditions.present), termList("Conditions, negated", rec.conditions.negated, "neg"), termList("Conditions, uncertain", rec.conditions.uncertain),
      termList("Symptoms and signs", rec.symptoms.present), termList("Symptoms, negated", rec.symptoms.negated, "neg"), termList("Symptoms, uncertain", rec.symptoms.uncertain),
      termList("Foods", rec.foods), termList("Food groups", rec.food_groups), termList("Interventions", rec.interventions), termList("Nutrients", rec.nutrients), termList("Patient groups", rec.populations),
      rows.length ? table(["Measurement", "Value", "Text", "Note"], rows) : null);
    if (cl.topics.length) out.push(h("h2", null, "Topics"), h("ul", { class: "plain" }, cl.topics.map((t) => h("li", null, h("b", null, t.topic.replace(/_/g, " ") + " "), h("span", { class: "badge" }, t.strength), " " + t.evidence.join(", ")))));
    out.push(h("p", { class: "muted" }, "Topics and document type come from cue words and found entities. They are not probabilities and not a diagnosis."));
    r.warnings.forEach((w) => out.push(h("div", { class: "notice" }, w)));
    return out;
  }

  // ---- Ask CNR (LLM phase 2): a chat over Chakudya's RAG pipeline. The conversation lives in this page's memory only: it is not saved in
  // the browser, and the server stores nothing. Text is always inserted as text, never as HTML.
  const ASK_INTENTS = { food_search: "Food search", barcode_search: "Barcode search", nutrition_question: "Nutrition question", exchange_list: "Exchange list", enteral_formula: "Enteral formula", general_chat: "General chat", unknown: "Unknown" };
  const ASK_SCOPES = [["clinical", "Clinical"], ["general", "General"], ["both", "Both"]];

  function askAnswer(text, valid, onCite) {
    const out = [];
    text.split(/(\[\d{1,3}\])/).forEach((part) => {
      const m = /^\[(\d{1,3})\]$/.exec(part);
      if (m && valid.has(Number(m[1]))) out.push(h("sup", null, h("button", { class: "cite", type: "button", "aria-label": "Show source " + m[1], onclick: () => onCite(Number(m[1])) }, m[1])));
      else if (part) out.push(part);
    });
    return h("div", { class: "answer" }, out);
  }

  function askBotMessage(d) {
    const items = new Map();
    const cites = (d.citations || []).map((c) => {
      const li = h("li", { class: c.cited ? "cited" : "" }, h("b", null, "[" + c.id + "] "), c.title || c.source || "Source", c.title && c.source ? h("span", { class: "muted" }, " · " + c.source) : null, " ", h("span", { class: "badge " + (c.cited ? "" : "planned") }, c.cited ? "cited" : "not cited"));
      items.set(c.id, li);
      return li;
    });
    const focus = (n) => { const li = items.get(n); if (!li) return; li.scrollIntoView({ block: "nearest" }); li.classList.add("hl"); setTimeout(() => li.classList.remove("hl"), 1500); };
    const head = h("div", { class: "msg-head" },
      h("span", { class: "badge" }, "Intent: " + (ASK_INTENTS[d.intent] || ASK_INTENTS.unknown)),
      h("span", { class: "badge explore" }, "Scope: " + (ASK_SCOPES.find((s) => s[0] === d.context) || ["", d.context])[1]),
      h("span", { class: "badge " + (d.grounded ? "" : "watch") }, d.grounded ? "Cited" : "Not grounded"),
      d.latency_ms != null ? h("span", { class: "muted" }, d.latency_ms + " ms") : null);
    return h("div", { class: "msg bot" }, head, askAnswer(d.answer || "", new Set(items.keys()), focus),
      (d.warnings || []).length ? h("ul", { class: "plain warn" }, d.warnings.map((w) => h("li", null, w))) : null,
      d.masked_question ? h("p", { class: "muted" }, "Identifiers were masked before sending. What was sent: " + d.masked_question) : null,
      cites.length ? h("details", { open: true }, h("summary", null, "Sources (" + cites.length + ")"), h("ol", { class: "sources" }, cites)) : null);
  }

  function askChunks(d) {
    return h("div", { class: "msg bot" },
      h("div", { class: "msg-head" }, h("span", { class: "badge explore" }, "Sources only"), h("span", { class: "muted" }, d.count + " found · " + d.latency_ms + " ms")),
      d.count === 0 ? h("p", null, "No matching source text was found.") : null,
      d.chunks.map((c) => h("div", { class: "chunk" },
        h("div", { class: "msg-head" }, h("b", null, "#" + c.rank + " " + (c.title || c.source || "Source")), c.title && c.source ? h("span", { class: "muted" }, c.source) : null,
          c.context ? h("span", { class: "badge" }, c.context) : null, c.score != null ? h("span", { class: "muted" }, "score " + c.score) : null),
        h("p", { class: "snippet" }, c.snippet + (c.truncated ? " …" : "")))),
      (d.warnings || []).length ? h("ul", { class: "plain warn" }, d.warnings.map((w) => h("li", null, w))) : null,
      d.masked_question ? h("p", { class: "muted" }, "Identifiers were masked before sending. What was sent: " + d.masked_question) : null);
  }

  async function askPage() {
    const c = await api("/v1/llm/contract");
    let scope = "both", history = [], busy = false;
    const log = h("div", { class: "chat-log", role: "log", "aria-live": "polite" });
    const scopeBtns = ASK_SCOPES.map(([v, l]) => h("button", { type: "button", class: "seg", "aria-pressed": String(v === scope), onclick: () => { scope = v; scopeBtns.forEach((b, i) => b.setAttribute("aria-pressed", String(ASK_SCOPES[i][0] === v))); } }, l));
    const mode = h("select", { id: "ask-mode", "aria-label": "What to return" }, h("option", { value: "ask" }, "Answer with citations"), h("option", { value: "retrieve" }, "Source passages only"));
    const ids = h("select", { id: "ask-ids", "aria-label": "Identifiable text" },
      h("option", { value: "reject" }, "Refuse text that has identifiers"), h("option", { value: "mask" }, "Mask identifiers and continue"));
    ids.value = c.identifiers.default === "mask" ? "mask" : "reject";
    const box = h("textarea", { id: "ask-q", rows: "3", maxlength: String(c.identifiers.max_chars), spellcheck: "true", placeholder: "Ask about nutrition criteria, guidelines, foods or formulas. Do not include names or contact details." });
    const err = h("p", { class: "error", role: "alert" });
    const add = (node) => { log.append(node); node.scrollIntoView({ block: "nearest" }); };
    const send = h("button", { class: "primary", type: "button", onclick: async () => {
      const q = box.value.trim();
      if (!q || busy) { if (!q) err.textContent = "Type a question first."; return; }
      err.textContent = ""; busy = true; send.disabled = true;
      add(h("div", { class: "msg user" }, q));
      box.value = "";
      const pending = h("div", { class: "msg bot muted", role: "status" }, "Searching the knowledge base…");
      add(pending);
      try {
        if (mode.value === "retrieve") {
          const d = await api("/v1/llm/retrieve", { method: "POST", body: JSON.stringify({ query: q, context: scope, top_k: 5, identifiers: ids.value }) });
          pending.replaceWith(askChunks(d));
        } else {
          const d = await api("/v1/llm/ask", { method: "POST", body: JSON.stringify({ question: q, context: scope, top_k: 6, history: history.slice(-6), identifiers: ids.value }) });
          pending.replaceWith(askBotMessage(d));
          history.push({ role: "user", text: q });
          if (!d.answer_withheld) history.push({ role: "assistant", text: d.answer });
          history = history.slice(-6);
        }
      } catch (e) {
        pending.replaceWith(h("div", { class: "msg bot refused", role: "alert" }, h("b", null, "Not sent or no answer. "), e.message));
      } finally { busy = false; send.disabled = false; box.focus(); }
    } }, "Ask");
    box.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); send.click(); } });
    const clear = h("button", { type: "button", onclick: () => { history = []; log.replaceChildren(); err.textContent = ""; } }, "Clear chat");
    return [
      h("h1", null, "Ask CNR"),
      h("p", null, "Ask a question in plain language. Answers come from the Chakudya knowledge base and foods registry, each with its sources. No new language model is used in this laboratory; this page only relays your question to Chakudya."),
      h("div", { class: "notice" }, c.notice),
      h("div", { class: "ask-controls" },
        h("div", null, h("label", null, "Knowledge scope"), h("div", { class: "segs", role: "group", "aria-label": "Knowledge scope" }, scopeBtns)),
        h("div", null, h("label", { for: "ask-mode" }, "Return"), mode),
        h("div", null, h("label", { for: "ask-ids" }, "Identifiable text"), ids)),
      log,
      h("label", { for: "ask-q" }, "Your question"), box, err,
      h("div", { class: "ask-actions" }, send, clear, h("span", { class: "muted" }, "Ctrl+Enter to send · " + c.limits.per_minute_per_key + " per minute")),
      h("p", { class: "muted" }, "This conversation is kept in this page only. It is not saved in your browser, and the laboratory stores no questions or answers. Follow-up questions send your last few turns to Chakudya so it can resolve what \"it\" refers to."),
    ];
  }

  // ---- Summarise note (LLM phase 4): de-identified clinical note -> facts (rule engine) -> optional model summary restricted to those facts ->
  // clinician review. The note, the facts and the summary live in this page's memory only: not in browser storage, not on the server. All text is
  // inserted as text, never as HTML. The summary can be copied only after the signed-in clinician has recorded a review.
  const SUM_STATUS = { present: ["present", ""], negated: ["negated", "planned"], uncertain: ["uncertain", "watch"], historical: ["historical", "planned"], family: ["family history", "planned"] };
  const SUM_REASON = { NUMBER: "a number not in the note", MEASUREMENT: "a measurement that does not match", ENTITY: "a term that is not a cited fact", NEGATION: "a status that does not match the note",
    SCREEN: "a diagnosis or prescription about a person", IDENTIFIER: "an identifier", NO_FACT: "no valid fact cited", FORMAT: "wrong format" };

  function sumFactsTable(facts) {
    if (!facts.length) return h("p", null, "No facts were found in the note.");
    return h("div", { class: "tablewrap" }, h("table", null,
      h("thead", null, h("tr", null, ["ID", "Type", "Term", "Status", "Value", "From the note"].map((t) => h("th", { scope: "col" }, t)))),
      h("tbody", null, facts.map((f) => {
        const [lab, cls] = SUM_STATUS[f.status] || [f.status, "planned"];
        return h("tr", { id: "fact-" + f.id }, h("td", null, f.id), h("td", null, f.kind), h("td", null, f.term),
          h("td", null, h("span", { class: "badge " + cls }, lab)),
          h("td", null, f.value != null ? f.value + " " + (f.unit || "") + (f.plausible === false ? " (outside plausible range)" : "") + (f.unit_guessed ? " (unit guessed)" : "") : "—"),
          h("td", null, h("code", null, f.evidence)));
      }))));
  }

  function sumSummary(sm) {
    if (!sm) return h("p", { class: "muted" }, "No model summary was requested: only the extracted facts are shown.");
    const rej = sm.rejected || { count: 0, reasons: {} };
    const why = Object.entries(rej.reasons || {}).map(([k, v]) => v + " × " + (SUM_REASON[k] || k)).join("; ");
    return h("div", null,
      h("p", { class: "muted" }, "Written by " + sm.generated_by.model + " (Groq, open weights) from the facts only · prompt " + sm.generated_by.prompt + (sm.latency_ms ? " · " + sm.latency_ms + " ms" : "")),
      sm.sentences.length ? h("ol", { class: "sum-sentences" }, sm.sentences.map((x) => h("li", null, x.text, " ",
        x.facts.map((id) => h("a", { class: "badge", href: "#summarise", onclick: (ev) => { ev.preventDefault(); const t = document.getElementById("fact-" + id); if (t) { t.scrollIntoView({ block: "nearest" }); t.classList.add("hl"); setTimeout(() => t.classList.remove("hl"), 1500); } } }, id)),
        (x.flags || []).map((fl) => h("span", { class: "badge watch", title: fl.detail }, "check: " + (fl.words || []).join(", ")))))) : h("p", { class: "error" }, "No sentence passed validation, so there is no summary."),
      rej.count ? h("p", { class: "muted" }, rej.count + " sentence(s) written by the model were dropped: " + why + ".") : null,
      sm.coverage && sm.coverage.not_mentioned.length ? h("p", { class: "muted" }, "Facts not mentioned in the summary: " + sm.coverage.not_mentioned.join(", ") + ".") : null);
  }

  function sumReview(d, state) {
    const f = h("input", { type: "checkbox", id: "sum-f" }), n = h("input", { type: "checkbox", id: "sum-n" }), msg = h("p", { role: "status" });
    const box = h("div", { class: "notice" });
    const copyArea = h("textarea", { rows: "5", id: "sum-copy", "aria-label": "Reviewed summary" }); copyArea.hidden = true;
    const copyBtn = h("button", { type: "button", disabled: true, onclick: async () => { try { await navigator.clipboard.writeText(copyArea.value); msg.textContent = "Copied."; } catch { msg.textContent = "Copy failed: select the text and copy it."; } } }, "Copy summary");
    const send = (decision) => async () => {
      msg.textContent = "";
      try {
        const r = await api("/v1/llm/summarise/review", { method: "POST", body: JSON.stringify({ request_id: d.request_id, decision, facts_checked: f.checked, numbers_checked: n.checked }) });
        state.reviewed = decision; box.querySelectorAll("button,input").forEach((x) => { x.disabled = true; });
        msg.textContent = r.detail;
        if (decision !== "rejected" && d.summary && d.summary.text) { copyArea.value = d.summary.text; copyArea.readOnly = decision === "accepted"; copyArea.hidden = false; copyBtn.disabled = false; }
      } catch (e) { msg.textContent = e.message; }
    };
    box.append(h("b", null, "Clinician review required"), h("p", null, "This is decision support, not a record. Check it against the note before any use. The summary can be copied only after you record a review."),
      h("label", { class: "check" }, f, " " + d.review.checklist[0]), h("label", { class: "check" }, n, " " + d.review.checklist[1]), h("p", { class: "muted" }, d.review.checklist[2]),
      h("div", { class: "ask-actions" }, h("button", { class: "primary", type: "button", onclick: send("accepted") }, "Accept as checked"),
        h("button", { type: "button", onclick: send("accepted_with_edits") }, "Accept with my edits"), h("button", { type: "button", onclick: send("rejected") }, "Reject"),
        h("span", { class: "muted" }, "Expires in " + Math.round(d.review.expires_in_s / 60) + " min")), msg, copyArea, copyBtn);
    return box;
  }

  async function summarisePage() {
    const st = await api("/v1/llm/summarise/status"), sm = st.summarisation;
    const note = h("textarea", { id: "sum-note", rows: "9", maxlength: String(st.limits.max_note_chars), spellcheck: "false", placeholder: "Paste a de-identified clinical note. Remove names, phone numbers, record numbers and addresses first." });
    const ids = h("select", { id: "sum-ids", "aria-label": "Identifiable text" }, h("option", { value: "reject" }, "Refuse notes that have identifiers"), h("option", { value: "mask" }, "Mask identifiers and continue"));
    const useModel = h("input", { type: "checkbox", id: "sum-model", checked: sm.ready || (st.openweight || {}).ready, disabled: !(sm.ready || (st.openweight || {}).ready) });
    const ow = st.openweight || { ready: false, models: [] };
    const anyReady = sm.ready || ow.ready;
    const provider = h("select", { id: "sum-provider", "aria-label": "Model provider", onchange: () => fillModels() },
      h("option", { value: "groq", disabled: !sm.ready }, "Groq" + (sm.ready ? "" : " (not configured)")), h("option", { value: "openweight", disabled: !ow.ready, selected: !sm.ready && ow.ready }, "Open-weight (Qwen / Gemma)" + (ow.ready ? "" : " (not configured)")));
    const model = h("select", { id: "sum-model-name", "aria-label": "Model", disabled: !anyReady });
    const fillModels = () => { const src = provider.value === "openweight" ? ow : sm; model.replaceChildren(...(src.models || []).map((m) => h("option", { value: m, selected: m === src.default_model }, m))); };
    fillModels();
    const err = h("p", { class: "error", role: "alert" }), out = h("div", { "aria-live": "polite" });
    const run = h("button", { class: "primary", type: "button", onclick: async () => {
      const text = note.value.trim();
      if (!text) { err.textContent = "Paste a note first."; return; }
      err.textContent = ""; run.disabled = true; out.replaceChildren(loader("Extracting facts" + (useModel.checked ? " and writing a summary…" : "…")));
      try {
        const d = await api("/v1/llm/summarise", { method: "POST", body: JSON.stringify({ note: text, identifiers: ids.value, use_model: useModel.checked && anyReady, model: model.value || null, provider: provider.value }) });
        const state = { reviewed: null };
        out.replaceChildren(
          h("h2", null, "Result"), (d.warnings || []).length ? h("ul", { class: "plain warn" }, d.warnings.map((w) => h("li", null, w))) : null,
          d.masked_note ? h("details", null, h("summary", null, "Identifiers were masked: " + d.identifiers.found + " found"), h("p", { class: "snippet" }, d.masked_note)) : null,
          h("h3", null, "Summary"), sumSummary(d.summary), h("h3", null, "Facts the summary is restricted to (" + d.extraction.facts.length + ")"), sumFactsTable(d.extraction.facts),
          h("ul", { class: "plain muted" }, d.extraction.limits.map((l) => h("li", null, l))), sumReview(d, state));
      } catch (e) { out.replaceChildren(h("div", { class: "msg bot refused", role: "alert" }, h("b", null, "Not processed. "), e.message)); }
      finally { run.disabled = false; }
    } }, "Extract and summarise");
    const clear = h("button", { type: "button", onclick: () => { note.value = ""; out.replaceChildren(); err.textContent = ""; } }, "Clear note and result");
    return [h("h1", null, "Summarise note"),
      h("p", null, "Turn a de-identified clinical note into structured facts and, optionally, a short summary. The language model sees only the extracted facts, never the note, and every number and term in its sentences is checked against the note."),
      h("div", { class: "notice" }, st.notice),
      anyReady ? null : h("div", { class: "notice" }, "The model summary is unavailable: " + (sm.problem || "not configured") + " Fact extraction still works."),
      h("div", { class: "ask-controls" }, h("div", null, h("label", { for: "sum-model" }, h("span", null, "Write a summary with the model "), useModel), provider, model), h("div", null, h("label", { for: "sum-ids" }, "Identifiable text"), ids)),
      h("label", { for: "sum-note" }, "De-identified note"), note, err,
      h("div", { class: "ask-actions" }, run, clear, h("span", { class: "muted" }, "Up to " + st.limits.max_note_chars + " characters")),
      h("p", { class: "muted" }, "Nothing is stored: the note, the facts and the summary stay in this page and disappear when you leave it or clear it."), out];
  }

  // ---- Training data (fine-tuning): reviewed knowledge examples -> approved by a different person -> JSONL for LoRA training on an external GPU.
  // Never patient text. All text is inserted as text. Nothing is kept in the browser.
  async function trainPage() {
    const d = await api("/v1/llm/train/examples"), c = d.counts || {}, ap = d.approved_by_split || {};
    const q = h("input", { id: "tr-q", type: "text", maxlength: "1000", placeholder: "A knowledge question (no patient information)" });
    const pa = h("textarea", { id: "tr-p", rows: "5", placeholder: "Source passages the answer is written from. Separate passages with a line containing only ---. Leave empty for none." });
    const an = h("textarea", { id: "tr-a", rows: "4", placeholder: "The answer. Cite the passages as [1], [2]. Knowledge only: no diagnosis or prescription about a person." });
    const sp = h("select", { id: "tr-s", "aria-label": "Split" }, h("option", { value: "train" }, "Training"), h("option", { value: "val" }, "Validation"));
    const err = h("p", { class: "error", role: "alert" });
    const form = can("write:llm_gold") ? [h("h2", null, "Add an example"), h("label", { for: "tr-q" }, "Question"), q, h("label", { for: "tr-p" }, "Source passages"), pa, h("label", { for: "tr-a" }, "Answer"), an,
      h("div", { class: "ask-controls" }, h("div", null, h("label", { for: "tr-s" }, "Use for"), sp)), err,
      h("button", { class: "primary", onclick: async () => {
        err.textContent = "";
        try {
          await api("/v1/llm/train/examples", { method: "POST", body: JSON.stringify({ examples: [{ question: q.value, passages: pa.value.split(/\n-{3,}\n/).map((x) => x.trim()).filter(Boolean), answer: an.value, split: sp.value }] }) });
          show("train");
        } catch (e) { err.textContent = e.message; }
      } }, "Add example")] : [];
    const shaOut = h("p", { class: "muted", "aria-live": "polite" });
    const exp = (split) => h("button", { onclick: async () => {
      try {
        const blob = await api("/v1/llm/train/export?split=" + split, { blob: true });
        const sha = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", await blob.arrayBuffer()))).map((x) => x.toString(16).padStart(2, "0")).join("");
        saveBlob(blob, "cnr-" + split + ".jsonl"); shaOut.textContent = "Saved cnr-" + split + ".jsonl · SHA-256 " + sha;
      } catch (e) { shaOut.textContent = e.message; }
    } }, "Download " + (split === "train" ? "training" : "validation") + " JSONL (" + (ap[split] || 0) + ")");
    const review = (x, decision) => h("button", { class: "link", onclick: async () => { try { await api("/v1/llm/train/examples/" + x.id + "/review", { method: "POST", body: JSON.stringify({ decision }) }); show("train"); } catch (e) { alert(e.message); } } }, decision === "approved" ? "Approve" : "Reject");
    const del = (x) => h("button", { class: "link", onclick: async () => { if (!confirm("Delete this draft?")) return; try { await api("/v1/llm/train/examples/" + x.id, { method: "DELETE" }); show("train"); } catch (e) { alert(e.message); } } }, "Delete");
    const rows = (d.examples || []).map((x) => [x.question, x.answer.slice(0, 120) + (x.answer.length > 120 ? "…" : ""), x.split, h("span", { class: "badge " + (x.status === "approved" ? "" : x.status === "rejected" ? "bad" : "watch") }, x.status), x.created_by || "—",
      h("span", null, x.status === "draft" && can("rate:llm") ? [review(x, "approved"), " ", review(x, "rejected"), " "] : null, x.status !== "approved" && can("write:llm_gold") ? del(x) : null)]);
    return [h("h1", null, "Training data"),
      h("p", null, "Reviewed knowledge examples for fine-tuning an open-weight model with LoRA on an external GPU. An example is exported only after a different person approves it. Questions that are in a held-out evaluation set are refused, so test questions cannot leak into training."),
      h("div", { class: "stats" }, nlpStat("Drafts", String(c.draft || 0)), nlpStat("Approved", String(c.approved || 0)), nlpStat("Rejected", String(c.rejected || 0))),
      can("export:llm_train") ? h("div", { class: "ask-actions" }, exp("train"), exp("val")) : null, shaOut,
      h("p", { class: "muted" }, "Train with training/train_lora.py (see training/README.md). The export uses the same prompt format the lab uses when it asks a model a question."),
      rows.length ? table(["Question", "Answer", "Use", "Status", "Author", ""], rows) : empty("No examples yet", can("write:llm_gold") ? "Add the first example below." : "A data scientist can add examples."), form];
  }

  // ---- RAG evaluation laboratory (LLM phase 3). Routes: eval (question sets), eval/set-<id>, eval/runs, eval/run-<id>, eval/result-<id>.
  // Gold questions are knowledge questions written by dietitians, never patient text. Nothing is kept in the browser: every value comes from
  // the server, and all text is inserted as text, never as HTML.
  const EVAL_OUTCOME = { answered: ["Answered", ""], withheld: ["Withheld", "watch"], refused_policy: ["Refused", "watch"], refused_identifiers: ["Refused (identifiers)", "watch"], error: ["Error", "bad"] };
  const evalRate = (r) => (!r || r.rate == null ? "—" : pct1(r.rate) + " (" + r.k + "/" + r.n + (r.ci95 ? ", 95% CI " + pct1(r.ci95[0]) + "–" + pct1(r.ci95[1]) : "") + ")");
  const evalTick = (v) => (v == null ? "—" : v ? "yes" : "no");
  const evalMs = (l) => (l && l.median_ms != null ? "median " + l.median_ms + " ms, p95 " + l.p95_ms + " ms (" + l.n + " calls)" : "—");
  const evalLines = (t) => t.split("\n").map((x) => x.trim()).filter(Boolean);
  const evalOutcomeBadge = (o) => { const [l, c] = EVAL_OUTCOME[o] || [o || "—", "planned"]; return h("span", { class: "badge " + c }, l); };

  function evalTabs(active) {
    return h("div", { class: "tabs", role: "navigation", "aria-label": "Evaluation sections" }, [["sets", "Question sets"], ["runs", "Runs"]].map(([id, label]) =>
      h("button", { "aria-current": id === active ? "page" : false, onclick: () => { location.hash = id === "sets" ? "eval" : "eval/" + id; } }, label)));
  }

  async function evalPage(arg) {
    const a = arg || "";
    let tab = "sets", body;
    if (a === "runs") { tab = "runs"; body = await evalRuns(); }
    else if (/^run-\d+$/.test(a)) { tab = "runs"; body = await evalRunDetail(Number(a.slice(4))); }
    else if (/^result-\d+$/.test(a)) { tab = "runs"; body = await evalResultDetail(Number(a.slice(7))); }
    else if (/^set-\d+$/.test(a)) body = await evalSetDetail(Number(a.slice(4)));
    else body = await evalSets();
    return [h("h1", null, "RAG evaluation laboratory"), evalTabs(tab), [].concat(body).flat(Infinity)];
  }

  async function evalSets() {
    const [rows, m] = await Promise.all([api("/v1/llm/eval/sets"), api("/v1/llm/eval/method")]);
    const out = [h("p", null, "Measure how well Ask CNR retrieves and cites its sources, and whether it refuses what it should. Build a gold-standard question set, run it, then have reviewers rate the answers."),
      h("div", { class: "notice" }, m.notice)];
    out.push(rows.length ? table(["Set", "Questions", "Answerable", "Should refuse", "Held-out", "Runs", "Created by", ""], rows.map((s) =>
      [s.name, s.questions, s.answerable, s.should_refuse, s.heldout, s.runs, s.created_by || "—", h("button", { class: "link", onclick: () => { location.hash = "eval/set-" + s.id; } }, "Open")]))
      : empty("No question sets yet", can("write:llm_gold") ? "Create a set below, then add questions with the sources and facts a correct answer must have." : "A dietitian or data scientist can create a set."));
    if (can("write:llm_gold")) {
      const name = h("input", { id: "ev-name", maxlength: "160" }), desc = h("input", { id: "ev-desc", maxlength: "1000" }), err = h("p", { class: "error", role: "alert" });
      out.push(h("h2", null, "New question set"), h("label", { for: "ev-name" }, "Name"), name, h("label", { for: "ev-desc" }, "Description (optional)"), desc, err,
        h("button", { class: "primary", onclick: async () => {
          err.textContent = "";
          if (!name.value.trim()) { err.textContent = "Give the set a name."; return; }
          try { const s = await api("/v1/llm/eval/sets", { method: "POST", body: JSON.stringify({ name: name.value.trim(), description: desc.value.trim() }) }); location.hash = "eval/set-" + s.id; }
          catch (e) { err.textContent = e.message; }
        } }, "Create set"));
    }
    out.push(h("details", null, h("summary", null, "How results are measured, and what they cannot show"),
      h("ul", null, Object.entries(m.metrics).map(([k, v]) => h("li", null, h("b", null, k.replace(/_/g, " ") + ": "), v))),
      h("ul", { class: "plain warn" }, m.limits.map((l) => h("li", null, l)))));
    return out;
  }

  function evalParseLines(text) {
    // one question per line: question | source one; source two | fact one; fact two      (or: question | refuse)
    return evalLines(text).map((line) => {
      const [q, second = "", third = ""] = line.split("|").map((x) => x.trim());
      const split = (x) => x.split(";").map((y) => y.trim()).filter(Boolean);
      return second.toLowerCase() === "refuse" ? { question: q, should_refuse: true } : { question: q, expected_sources: split(second), expected_facts: split(third) };
    });
  }

  async function evalSetDetail(id) {
    const [s, runs] = await Promise.all([api("/v1/llm/eval/sets/" + id), api("/v1/llm/eval/runs?set_id=" + id)]);
    const reload = () => show("eval/set-" + id);
    const active = s.questions.filter((q) => !q.archived);
    const out = [backLink("eval", "All question sets"), h("h2", null, s.name), s.description ? h("p", null, s.description) : null,
      h("p", { class: "muted" }, active.length + " active question" + (active.length === 1 ? "" : "s") + " · fingerprint " + s.sha256.slice(0, 12) + " · created by " + (s.created_by || "—") + " · up to " + s.limits.max_questions + " questions"),
      s.questions.length ? table(["Question", "Scope", "Split", "Expected sources", "Expected facts", ""], s.questions.map((q) => [
        h("span", { class: q.archived ? "neg" : "" }, q.question), q.context, q.split,
        q.should_refuse ? h("span", { class: "badge watch" }, "should refuse") : q.expected_sources.join("; ") || "—", q.should_refuse ? "—" : q.expected_facts.join("; ") || "—",
        q.archived ? "archived" : can("write:llm_gold") ? h("button", { class: "link", onclick: async () => { try { await api("/v1/llm/eval/questions/" + q.id + "/archive", { method: "POST" }); reload(); } catch (e) { alert(e.message); } } }, "Archive") : ""]))
        : empty("No questions yet", "Add questions below.")];
    if (can("write:llm_gold")) {
      const q = h("textarea", { id: "ev-q", rows: "2", maxlength: "1000", placeholder: "A knowledge question, for example: What is the MUAC cut-off for severe acute malnutrition? No names or contact details." });
      const ctx = h("select", { id: "ev-ctx" }, [["both", "Both"], ["clinical", "Clinical"], ["general", "General"]].map(([v, l]) => h("option", { value: v }, l)));
      const split = h("select", { id: "ev-split" }, h("option", { value: "dev" }, "Development"), h("option", { value: "heldout" }, "Held-out (kept for the final check)"));
      const src = h("textarea", { id: "ev-src", rows: "2", placeholder: "One per line: part of the source or title that should be retrieved and cited, for example: Malawi Guidelines" });
      const facts = h("textarea", { id: "ev-facts", rows: "2", placeholder: "One per line: a short fact a correct answer states, for example: 115 mm" });
      const refuse = h("input", { id: "ev-refuse", type: "checkbox" });
      const err = h("p", { class: "error", role: "alert" });
      out.push(h("h2", null, "Add a question"), h("label", { for: "ev-q" }, "Question"), q,
        h("div", { class: "ask-controls" }, h("div", null, h("label", { for: "ev-ctx" }, "Knowledge scope"), ctx), h("div", null, h("label", { for: "ev-split" }, "Split"), split)),
        h("label", { class: "chk" }, refuse, " The laboratory should refuse this (for example, it asks for a diagnosis or a dose for a patient)"),
        h("label", { for: "ev-src" }, "Expected sources"), src, h("label", { for: "ev-facts" }, "Expected facts"), facts, err,
        h("button", { class: "primary", onclick: async () => {
          err.textContent = "";
          const body = { question: q.value.trim(), context: ctx.value, split: split.value, should_refuse: refuse.checked, expected_sources: refuse.checked ? [] : evalLines(src.value), expected_facts: refuse.checked ? [] : evalLines(facts.value) };
          if (!body.question) { err.textContent = "Type the question first."; return; }
          try { await api("/v1/llm/eval/sets/" + id + "/questions", { method: "POST", body: JSON.stringify({ questions: [body] }) }); reload(); }
          catch (e) { err.textContent = e.message; }
        } }, "Add question"));
      const bulk = h("textarea", { id: "ev-bulk", rows: "5", placeholder: "question | source one; source two | fact one; fact two\nWhat is the MUAC cut-off for SAM? | Malawi Guidelines | 115 mm\nDoes this child have kwashiorkor? | refuse" });
      const berr = h("p", { class: "error", role: "alert" });
      out.push(h("details", null, h("summary", null, "Add several questions at once"),
        h("p", { class: "muted" }, "One question per line: question | expected sources | expected facts, separated by semicolons. A line ending in | refuse is a question the laboratory should refuse. Up to 50 lines; if any line is not accepted, none are added."), bulk, berr,
        h("button", { onclick: async () => {
          berr.textContent = "";
          const qs = evalParseLines(bulk.value);
          if (!qs.length) { berr.textContent = "Paste at least one line."; return; }
          try { await api("/v1/llm/eval/sets/" + id + "/questions", { method: "POST", body: JSON.stringify({ questions: qs }) }); reload(); }
          catch (e) { berr.textContent = e.message; }
        } }, "Add these questions")));
    }
    if (can("use:llm") && active.length) {
      const split = h("select", { id: "ev-run-split" }, h("option", { value: "dev" }, "Development questions"), h("option", { value: "heldout" }, "Held-out questions"), h("option", { value: "all" }, "All questions"));
      const topk = h("select", { id: "ev-topk" }, [3, 5, 8, 10].map((n) => h("option", { value: String(n), selected: n === 5 }, "Top " + n + " sources")));
      const err = h("p", { class: "error", role: "alert" });
      const owst = await api("/v1/llm/summarise/status").then((x) => x.openweight || {}).catch(() => ({}));
      const gen = h("select", { id: "ev-gen" }, h("option", { value: "" }, "Chakudya's own pipeline"),
        owst.server_ready ? (owst.models || []).map((m) => h("option", { value: m }, "Open-weight: " + m)) : null);
      out.push(h("h2", null, "Run an evaluation"),
        h("div", { class: "notice" }, "Each question is checked like a live question, then sent to Chakudya twice (sources, then answer). Every call counts toward your rate limit, so a large set takes several minutes. Held-out questions are for the final check only; do not tune against them."),
        h("div", { class: "ask-controls" }, h("div", null, h("label", { for: "ev-run-split" }, "Questions"), split), h("div", null, h("label", { for: "ev-topk" }, "Retrieval depth"), topk), h("div", null, h("label", { for: "ev-gen" }, "Answers written by"), gen)), err,
        h("button", { class: "primary", onclick: async () => {
          err.textContent = "";
          try { const r = await api("/v1/llm/eval/sets/" + id + "/runs", { method: "POST", body: JSON.stringify({ split: split.value, top_k: Number(topk.value), model: gen.value || null }) }); location.hash = "eval/run-" + r.id; }
          catch (e) { err.textContent = e.message; }
        } }, "Start run"));
    }
    out.push(h("h2", null, "Runs of this set"), evalRunTable(runs));
    return out;
  }

  function evalRunTable(rows) {
    if (!rows.length) return empty("No runs yet", "Start a run from a question set.");
    return table(["Run", "Set", "Status", "Questions", "Config", "Started by", "Started", ""], rows.map((r) => ["#" + r.id, r.set_name || "—",
      h("span", { class: "badge " + (r.status === "completed" ? "" : r.status === "failed" ? "bad" : "watch") }, r.status + (r.stop_reason ? " · " + r.stop_reason.replace(/_/g, " ") : "")),
      r.completed + " / " + r.question_count, (r.config.split || "—") + ", top " + (r.config.top_k || "—") + " · " + (r.config.generator || "chakudya"), r.created_by || "—", fmt(r.created_at),
      h("button", { class: "link", onclick: () => { location.hash = "eval/run-" + r.id; } }, "Open")]));
  }

  async function evalRuns() {
    const rows = await api("/v1/llm/eval/runs"), done = rows.filter((r) => r.status === "completed");
    const opt = (r) => h("option", { value: String(r.id) }, "#" + r.id + " · " + (r.config.generator || "chakudya") + " · " + r.set_name + " · " + (r.config.split || ""));
    const a = h("select", { id: "cmp-a", "aria-label": "Run A (baseline)" }, done.map(opt)), b = h("select", { id: "cmp-b", "aria-label": "Run B (candidate)" }, done.map(opt));
    if (done.length > 1) b.selectedIndex = 1;
    const out = h("div", { "aria-live": "polite" }), err = h("p", { class: "error", role: "alert" });
    const cmp = done.length < 2 ? h("p", { class: "muted" }, "Complete two runs on the same questions (for example the base model, then a fine-tuned one) to compare them.") : [
      h("h2", null, "Compare two runs"),
      h("p", { class: "muted" }, "A is the baseline (for example the base model), B the candidate (for example a fine-tuned adapter). Both runs must have used the same questions."),
      h("div", { class: "ask-controls" }, h("div", null, h("label", { for: "cmp-a" }, "A · baseline"), a), h("div", null, h("label", { for: "cmp-b" }, "B · candidate"), b)), err,
      h("button", { class: "primary", onclick: async () => {
        err.textContent = ""; out.replaceChildren();
        try {
          const c = await api("/v1/llm/eval/compare?a=" + a.value + "&b=" + b.value), sa = c.a.summary, sb = c.b.summary;
          const verdict = { a_better: "A is better", b_better: "B is better", no_clear_difference: "No clear difference" }[c.verdict];
          out.replaceChildren(
            h("div", { class: "notice " + (c.b_may_replace_a ? "ok" : "") }, h("b", null, verdict + ". "), "Mean difference (B minus A) " + (c.mean_difference_b_minus_a ?? "—") + (c.ci95 ? ", 95% interval " + c.ci95[0] + " to " + c.ci95[1] : "") + " over " + c.paired_questions + " paired questions. " +
              (c.b_may_replace_a ? "B passes every check here." : "Do not replace A with B on this result.")),
            c.guards.length ? h("ul", { class: "plain warn" }, c.guards.map((x) => h("li", null, x))) : null,
            c.readiness_both_sufficient ? null : h("p", { class: "muted" }, "Not enough evidence in at least one run (see each run's readiness), so no quality claim is possible."),
            table(["Measure", "A · " + c.a.label, "B · " + c.b.label], [
              ["Fact coverage", pct1(sa.groundedness.fact_coverage.mean), pct1(sb.groundedness.fact_coverage.mean)],
              ["Answers grounded", evalRate(sa.groundedness.grounded_rate), evalRate(sb.groundedness.grounded_rate)],
              ["Numbers supported by sources", evalRate(sa.groundedness.numeric_support_rate), evalRate(sb.groundedness.numeric_support_rate)],
              ["Refusals as expected", evalRate(sa.refusals.recall), evalRate(sb.refusals.recall)],
              ["Over-refusal", evalRate(sa.refusals.over_refusal_rate), evalRate(sb.refusals.over_refusal_rate)],
              ["Errors", String(sa.errors), String(sb.errors)]]),
            barChart(["Fact coverage", "Answers grounded", "Numbers supported", "Refusals as expected"],
              [{ name: "A · " + c.a.label, color: "#8a9ba2", values: [sa.groundedness.fact_coverage.mean, sa.groundedness.grounded_rate.rate, sa.groundedness.numeric_support_rate.rate, sa.refusals.recall.rate] },
               { name: "B · " + c.b.label, color: "#0e5a73", values: [sb.groundedness.fact_coverage.mean, sb.groundedness.grounded_rate.rate, sb.groundedness.numeric_support_rate.rate, sb.refusals.recall.rate] }],
              "Higher is better for every bar."),
            h("p", { class: "muted" }, c.note));
        } catch (e) { err.textContent = e.message; }
      } }, "Compare")];
    return [h("p", null, "Every run is kept with its configuration and the fingerprint of the questions it used, so two runs are comparable only when the fingerprints match."), cmp, out, h("h2", null, "All runs"), evalRunTable(rows)];
  }

  async function evalRunDetail(id) {
    const r = await api("/v1/llm/eval/runs/" + id), s = r.summary, g = s.groundedness, rv = s.reviewer, ready = s.readiness;
    if (r.status === "queued" || r.status === "running") poll = setTimeout(() => { if (location.hash.slice(1) === "eval/run-" + id) show("eval/run-" + id, true); }, 3000);
    const out = [backLink("eval/runs", "All runs"), h("h2", null, "Run #" + r.id + " · " + (r.set_name || "")),
      h("p", null, h("span", { class: "badge " + (r.status === "completed" ? "" : r.status === "failed" ? "bad" : "watch") }, r.status + (r.stop_reason ? " · " + r.stop_reason.replace(/_/g, " ") : "")), " ",
        r.completed + " of " + r.question_count + " questions · " + r.config.split + " · top " + r.config.top_k + " · fingerprint " + r.set_sha256.slice(0, 12) + " · started by " + (r.created_by || "—") + " · " + fmt(r.created_at))];
    if ((r.status === "queued" || r.status === "running") && can("use:llm"))
      out.push(h("button", { onclick: async () => { try { await api("/v1/llm/eval/runs/" + id + "/stop", { method: "POST" }); show("eval/run-" + id); } catch (e) { alert(e.message); } } }, "Stop run"));
    out.push(ready.sufficient_for_quality_claims
      ? h("div", { class: "notice ok" }, "These results meet the minimum evidence thresholds. " + ready.note)
      : h("div", { class: "notice" }, h("b", null, "Not enough evidence for any quality claim. "), h("ul", { class: "plain" }, ready.reasons.map((x) => h("li", null, x)))));
    out.push(h("div", { class: "stats" }, nlpStat("Retrieval hit rate", pct1(s.retrieval.hit_rate.rate)), nlpStat("Cited an expected source", pct1(s.citations.cited_source_hit_rate.rate)),
      nlpStat("Answers grounded", pct1(g.grounded_rate.rate)), nlpStat("Refusals as expected", pct1(s.refusals.recall.rate))));
    out.push(h("h2", null, "At a glance"), barChart(["Retrieval hit rate", "Cited expected source", "Answers grounded", "Numbers supported", "Fact coverage", "Refusals as expected"],
      [{ name: "This run", color: "#0e5a73", values: [s.retrieval.hit_rate.rate, s.citations.cited_source_hit_rate.rate, g.grounded_rate.rate, g.numeric_support_rate.rate, g.fact_coverage.mean, s.refusals.recall.rate] }],
      "Higher is better for every bar. A missing bar means no question could measure it."));
    out.push(table(["Measure", "Result"], [
      ["Retrieval hit rate", evalRate(s.retrieval.hit_rate)], ["Mean reciprocal rank", s.retrieval.mean_reciprocal_rank == null ? "—" : s.retrieval.mean_reciprocal_rank],
      ["Answer cites an expected source", evalRate(s.citations.cited_source_hit_rate)],
      ["Citation validity", s.citations.validity.mean == null ? "—" : pct1(s.citations.validity.mean) + " of markers valid; " + s.citations.validity.answers_with_invalid_marker + " of " + s.citations.validity.answers_with_markers + " answers had an invalid marker"],
      ["Answers that cite a listed source", evalRate(g.grounded_rate)], ["Numbers supported by retrieved text", evalRate(g.numeric_support_rate)],
      ["Expected facts found", g.fact_coverage.mean == null ? "—" : pct1(g.fact_coverage.mean) + " over " + g.fact_coverage.questions + " questions"],
      ["Should-refuse questions refused or withheld", evalRate(s.refusals.recall)], ["Answerable questions refused (over-refusal)", evalRate(s.refusals.over_refusal_rate)],
      ["Errors", s.errors], ["Latency: answer", evalMs(s.latency.ask)], ["Latency: sources", evalMs(s.latency.retrieve)]]));
    out.push(h("h2", null, "Reviewer ratings"), h("p", null, rv.results_rated + " of " + rv.results_total + " results rated (" + pct1(rv.coverage) + ") by " + rv.reviewers + " reviewer" + (rv.reviewers === 1 ? "" : "s") +
      (rv.mean_accuracy != null ? " · mean accuracy " + rv.mean_accuracy + " of 5 · supported " + evalRate(rv.supported) + " · safe " + evalRate(rv.safe) : "")));
    out.push(h("h2", null, "Results"), r.results.length ? table(["Question", "Outcome", "Source retrieved", "Source cited", "Cites a source", "Unsupported numbers", "Facts", "Answer ms", "Ratings", ""], r.results.map((x) => [
      (x.should_refuse ? "[should refuse] " : "") + (x.question.length > 90 ? x.question.slice(0, 90) + "…" : x.question), evalOutcomeBadge(x.outcome), evalTick(x.retrieval_hit), evalTick(x.cited_source_hit),
      evalTick(x.grounded), x.numbers_unsupported == null ? "—" : x.numbers_unsupported, x.facts_total ? x.facts_found + "/" + x.facts_total : "—", x.latency_ask_ms == null ? "—" : x.latency_ask_ms, x.ratings,
      h("button", { class: "link", onclick: () => { location.hash = "eval/result-" + x.result_id; } }, "Open")])) : empty("No results yet", "Results appear here as each question finishes."));
    return out;
  }

  async function evalResultDetail(id) {
    const x = await api("/v1/llm/eval/results/" + id), q = x.question, m = x.metrics;
    const out = [backLink("eval/run-" + x.run_id, "Back to run #" + x.run_id), h("h2", null, "Result"), h("p", null, h("b", null, q.question)),
      h("p", { class: "muted" }, "Scope " + q.context + " · " + q.split + (q.should_refuse ? " · the laboratory should refuse this" : "")),
      q.expected_sources.length ? h("p", null, h("b", null, "Expected sources: "), q.expected_sources.join("; ")) : null, q.expected_facts.length ? h("p", null, h("b", null, "Expected facts: "), q.expected_facts.join("; ")) : null,
      h("p", null, evalOutcomeBadge(m.outcome), " ", m.refusal_category ? h("span", { class: "muted" }, "reason: " + m.refusal_category.replace(/_/g, " ")) : null),
      x.answer ? h("div", { class: "msg bot" }, h("div", { class: "answer" }, x.answer)) : h("p", { class: "muted" }, "No answer was returned for this question.")];
    if (m.retrieved) out.push(h("h3", null, "Sources retrieved"), table(["Rank", "Source", "Title", "Score"], m.retrieved.map((c) => [c.rank, c.source || "—", c.title || "—", c.score == null ? "—" : c.score])));
    const checks = [["Expected source retrieved", evalTick(m.retrieval_hit) + (m.first_hit_rank ? " (rank " + m.first_hit_rank + ")" : "")], ["Expected source cited", evalTick(m.cited_source_hit)],
      ["Citation markers", m.markers_valid == null ? "—" : m.markers_valid + " valid, " + m.markers_invalid + " invalid"], ["Cites a listed source", evalTick(m.grounded)],
      ["Numbers not in retrieved text", m.numbers_total == null ? "—" : m.numbers_unsupported + " of " + m.numbers_total + ((m.unsupported_numbers || []).length ? " (" + m.unsupported_numbers.join(", ") + ")" : "")],
      ["Expected facts found", m.facts_total ? m.facts_found + " of " + m.facts_total + ((m.missing_facts || []).length ? " · missing: " + m.missing_facts.join("; ") : "") : "—"],
      ["Latency", (m.latency_retrieve_ms != null ? "sources " + m.latency_retrieve_ms + " ms" : "") + (m.latency_ask_ms != null ? " · answer " + m.latency_ask_ms + " ms" : "") || "—"]];
    out.push(h("h3", null, "Automatic checks"), table(["Check", "Result"], checks), h("p", { class: "muted" }, x.answer_note));
    out.push(h("h3", null, "Reviewer ratings"));
    out.push(x.ratings.length ? table(["Reviewer", "Accuracy", "Supported", "Safe", "Comment", "Updated"], x.ratings.map((g) => [g.reviewer + (g.mine ? " (you)" : ""), g.accuracy + " / 5", evalTick(g.supported), evalTick(g.safe), g.comment || "—", fmt(g.updated_at)])) : empty("Not rated yet", ""));
    if (can("rate:llm") && m.outcome !== "error") {
      const mine = x.ratings.find((g) => g.mine) || { accuracy: 3, supported: false, safe: true, comment: "" };
      const acc = h("select", { id: "rt-acc" }, [1, 2, 3, 4, 5].map((n) => h("option", { value: String(n), selected: n === mine.accuracy }, n + " of 5")));
      const sup = h("input", { id: "rt-sup", type: "checkbox", checked: mine.supported }), safe = h("input", { id: "rt-safe", type: "checkbox", checked: mine.safe });
      const com = h("textarea", { id: "rt-com", rows: "2", maxlength: "600", placeholder: "Optional. No names or contact details." });
      com.value = mine.comment || "";
      const err = h("p", { class: "error", role: "alert" });
      out.push(h("h3", null, x.ratings.some((g) => g.mine) ? "Change your rating" : "Rate this answer"), h("p", { class: "muted" }, x.rating_scale.accuracy),
        h("label", { for: "rt-acc" }, "Accuracy"), acc,
        h("label", { class: "chk" }, sup, " " + x.rating_scale.supported), h("label", { class: "chk" }, safe, " " + x.rating_scale.safe),
        h("label", { for: "rt-com" }, "Comment"), com, err,
        h("button", { class: "primary", onclick: async () => {
          err.textContent = "";
          try { await api("/v1/llm/eval/results/" + id + "/rating", { method: "POST", body: JSON.stringify({ accuracy: Number(acc.value), supported: sup.checked, safe: safe.checked, comment: com.value.trim() }) }); show("eval/result-" + id); }
          catch (e) { err.textContent = e.message; }
        } }, "Save rating"));
    }
    return out;
  }

  async function nlpPage() {
    const info = await api("/v1/nlp/info");
    const result = h("div"), bench = h("div"), lexBox = h("div");
    const area = h("textarea", { id: "nlp-text", rows: "7", maxlength: String(info.limits.chars_per_text), spellcheck: "false", placeholder: "Paste de-identified text: a clinical note, report, abstract or dietary record, in English, Chichewa or both." });
    const mode = h("select", { id: "nlp-mode" }, h("option", { value: "mask" }, "Mask identifiers and continue"), h("option", { value: "reject" }, "Refuse text that has identifiers"));
    const err = h("p", { class: "error", role: "alert" });
    const run = h("button", { class: "primary", onclick: async () => {
      err.textContent = ""; const text = area.value;
      if (!text.trim()) { err.textContent = "Paste some text first."; return; }
      try { const d = await api("/v1/nlp/analyze", { method: "POST", body: JSON.stringify({ texts: [text], identifiers: mode.value }) }); result.replaceChildren(...nlpResult(text, d.results[0]).flat().filter(Boolean)); }
      catch (e) { err.textContent = e.message; result.replaceChildren(); }
    } }, "Analyse text");
    const samples = h("p", null, "Try a sample: ", NLP_SAMPLES.map(([n, t]) => [h("button", { class: "link", onclick: () => { area.value = t; } }, n), " "]));
    const bBtn = h("button", { onclick: async () => {
      try { const b = await api("/v1/nlp/benchmark");
        bench.replaceChildren(h("div", { class: "notice" }, b.notice), h("div", { class: "stats" }, h("div", { class: "stat" }, h("b", null, pct1(b.overall.f1)), h("span", null, "F1 on " + b.sentences + " synthetic sentences")), h("div", { class: "stat" }, h("b", null, pct1(b.context_flag_accuracy)), h("span", null, "Negation, uncertainty, family flags"))),
          table(["Label", "Expected", "Found", "Precision", "Recall", "F1"], Object.entries(b.per_label).map(([k, v]) => [ENT_NAMES[k] || k, v.expected, v.found, pct1(v.precision), pct1(v.recall), pct1(v.f1)])),
          b.missed.length || b.extra.length ? table(["Sentence", "Problem", "Label", "Text"], [...b.missed.map((m) => [m.sentence, "missed", m.label, m.text]), ...b.extra.map((m) => [m.sentence, "extra", m.label, m.text])]) : h("p", { class: "muted" }, "No misses and no extras on this set."));
      } catch (e) { bench.replaceChildren(h("div", { class: "notice" }, e.message)); } } }, "Run regression benchmark");
    const lq = h("input", { id: "lex-q", type: "search", placeholder: "Search terms", maxlength: "60", autocomplete: "off" });
    const ll = h("select", { id: "lex-l" }, h("option", { value: "" }, "All types"), Object.keys(ENT_NAMES).filter((k) => k !== "MEASUREMENT" && k !== "PHI").map((k) => h("option", { value: k }, ENT_NAMES[k])));
    const lg = h("select", { id: "lex-g" }, h("option", { value: "" }, "All languages"), h("option", { value: "en" }, "English"), h("option", { value: "ny" }, "Chichewa"), h("option", { value: "tum" }, "Tumbuka"), h("option", { value: "yao" }, "Yao"));
    const loadLex = async () => {
      const p = new URLSearchParams({ limit: "500" }); if (lq.value.trim()) p.set("q", lq.value.trim()); if (ll.value) p.set("label", ll.value); if (lg.value) p.set("lang", lg.value);
      try { const d = await api("/v1/nlp/lexicon?" + p); lexBox.replaceChildren(h("p", { class: "muted" }, d.total + " terms" + (d.total > d.terms.length ? " (first " + d.terms.length + " shown)" : "") + ". None has been clinically reviewed yet."), table(["Term", "Means", "Type", "Group", "Language"], d.terms.map((t) => [t.term, t.norm, ENT_NAMES[t.label], t.group, t.lang === "both" ? "English and Chichewa" : LANG_NAMES[t.lang] || t.lang]))); }
      catch (e) { lexBox.replaceChildren(h("div", { class: "notice" }, e.message)); }
    };
    [ll, lg].forEach((el) => el.addEventListener("change", loadLex)); lq.addEventListener("keydown", (e) => { if (e.key === "Enter") loadLex(); });
    return [h("h1", null, "NLP laboratory"),
      h("p", null, "Extracts structured nutrition information from clinical notes, reports, research text and dietary records, in English, Chichewa, Tumbuka and Yao: named entities (foods, conditions, symptoms, interventions, nutrients, patient groups), measurements with units, negation and uncertainty, language, and a document type."),
      h("div", { class: "notice" }, info.notice),
      table(["Language", "Support"], info.languages.map((l) => [l.name, l.support])),
      info.lexicon_file_errors.length ? h("div", { class: "notice" }, "Some rows in the term files were skipped: " + info.lexicon_file_errors.slice(0, 5).join("; ")) : null,
      h("div", { class: "stats" }, h("div", { class: "stat" }, h("b", null, info.lexicon.terms), h("span", null, "terms in the lists")), h("div", { class: "stat" }, h("b", null, info.lexicon.clinically_reviewed), h("span", null, "clinically reviewed")), h("div", { class: "stat" }, h("b", null, info.kind), h("span", null, "engine " + info.engine))),
      can("run:nlp") ? [h("h2", null, "Analyse text"), samples, h("label", { for: "nlp-text" }, "Text"), area, h("label", { for: "nlp-mode" }, "If identifiers are found"), mode, h("p", null, run), err, result]
        : h("div", { class: "notice" }, "Your role can read the terminology and benchmark but cannot analyse text."),
      h("h2", null, "Regression benchmark"), h("p", { class: "muted" }, "Checks the engine against a few synthetic sentences. It protects against regressions; it says nothing about accuracy on real notes."), h("p", null, bBtn), bench,
      h("details", { class: "card", ontoggle: (ev) => { if (ev.target.open && !lexBox.childNodes.length) loadLex(); } }, h("summary", null, "Terminology lists"),
        h("div", { class: "inline" }, h("div", null, h("label", { for: "lex-q" }, "Search"), lq), h("div", null, h("label", { for: "lex-l" }, "Type"), ll), h("div", null, h("label", { for: "lex-g" }, "Language"), lg)), lexBox)];
  }

  // ---------- Notes: laboratory memory. People record decisions, findings, lessons and warnings next to the thing they are about. ----------
  // Notes hold words only (the API refuses emails, phone and ID numbers). Archived notes are kept but hidden.
  const NOTE_KINDS = ["note", "decision", "finding", "lesson", "warning", "question"];
  const NOTE_SUBJECTS = { general: "Whole lab", dataset: "Dataset", dataset_version: "Dataset version", experiment: "Experiment", exploration: "Exploration", hotspot: "Hotspot analysis", model: "Model", model_version: "Model version" };
  const NOTE_LINKS = { dataset: "datasets/", experiment: "experiments/", exploration: "explorations/", hotspot: "hotspots/", model_version: "registry/" };
  let noteSeq = 0;

  function noteForm(init, onSave, onCancel) {
    const n = ++noteSeq, v = init || {};
    const kind = h("select", { id: "nk" + n }, NOTE_KINDS.map((k) => h("option", { value: k, selected: k === (v.kind || "note") }, k)));
    const title = h("input", { id: "nt" + n, type: "text", maxlength: "160", value: v.title || "", autocomplete: "off" });
    const body = h("textarea", { id: "nb" + n, rows: "4", maxlength: "4000" }, v.body || "");
    const tags = h("input", { id: "ng" + n, type: "text", maxlength: "200", value: (v.tags || []).join(", "), autocomplete: "off", placeholder: "e.g. leakage, data quality" });
    const pin = h("input", { id: "np" + n, type: "checkbox", checked: !!v.pinned });
    const err = h("p", { class: "error", role: "alert" });
    const save = h("button", { class: "primary", onclick: async () => {
      err.textContent = "";
      try { await onSave({ kind: kind.value, title: title.value, body: body.value, tags: tags.value.split(",").map((x) => x.trim()).filter(Boolean), pinned: pin.checked }); }
      catch (e) { err.textContent = e.message; }
    } }, v.id ? "Save changes" : "Save note");
    return h("div", { class: "card noteform" },
      h("div", { class: "inline" }, h("div", null, h("label", { for: "nt" + n }, "Title"), title), h("div", null, h("label", { for: "nk" + n }, "Type"), kind), h("span")),
      h("label", { for: "nb" + n }, "What should the lab remember?"), body,
      h("label", { for: "ng" + n }, "Tags (comma separated, optional)"), tags,
      h("p", null, h("label", null, pin, " Pin: show this note on every page and to the assistant first")),
      h("p", { class: "muted" }, "Words only. Do not paste patient data, names, phone or ID numbers."),
      h("p", null, save, " ", onCancel ? h("button", { class: "link", onclick: onCancel }, "Cancel") : null), err);
  }

  function noteCard(n, reload, showSubject) {
    const mine = me && (me.role === "admin" || me.name === n.created_by) && can("write:notes") && !n.archived_at;
    const card = h("div", { class: "card note" + (n.pinned ? " pinned" : "") });
    const view = () => {
      const subj = showSubject && n.subject_type !== "general"
        ? (NOTE_LINKS[n.subject_type] ? h("button", { class: "link", onclick: () => { location.hash = NOTE_LINKS[n.subject_type] + n.subject_id; } }, NOTE_SUBJECTS[n.subject_type] + " " + n.subject_id)
          : NOTE_SUBJECTS[n.subject_type] + " " + n.subject_id) : (showSubject ? "Whole lab" : null);
      const hist = h("div");
      const histBox = n.edits ? h("details", { ontoggle: async (ev) => {
        if (!ev.target.open || hist.childNodes.length) return;
        try { const d = await api("/v1/notes/" + n.id); hist.replaceChildren(...d.history.slice().reverse().map((x) => h("div", { class: "notice" }, h("b", null, x.title), h("p", { class: "notebody" }, x.body), h("small", null, "Earlier wording, replaced " + fmt(x.at) + " by " + x.by)))); }
        catch (e) { hist.replaceChildren(h("div", { class: "notice" }, e.message)); }
      } }, h("summary", null, "Earlier wording (" + n.edits + ")"), hist) : null;
      card.replaceChildren(
        h("div", { class: "row" }, h("h3", null, n.title), h("span", null, h("span", { class: "badge " + (n.kind === "warning" ? "watch" : n.kind === "decision" ? "explore" : "") }, n.kind), n.pinned ? " " : null, n.pinned ? h("span", { class: "badge" }, "pinned") : null, n.archived_at ? " " : null, n.archived_at ? h("span", { class: "badge bad" }, "archived") : null)),
        h("p", { class: "notebody" }, n.body),
        h("p", { class: "muted" }, [n.created_by || "—", fmt(n.created_at), n.edits ? "edited " + fmt(n.updated_at) : null, subj, n.tags.length ? n.tags.map((t) => "#" + t).join(" ") : null].filter((x) => x != null).flatMap((x, i) => (i ? [" · ", x] : [x]))),
        histBox,
        mine ? h("p", null,
          h("button", { class: "link", onclick: async () => { try { await api("/v1/notes/" + n.id, { method: "PATCH", body: JSON.stringify({ pinned: !n.pinned }) }); reload(); } catch (e) { alert(e.message); } } }, n.pinned ? "Unpin" : "Pin"), " ",
          h("button", { class: "link", onclick: () => card.replaceChildren(noteForm(n, async (b) => { await api("/v1/notes/" + n.id, { method: "PATCH", body: JSON.stringify(b) }); reload(); }, view)) }, "Edit"), " ",
          h("button", { class: "link", onclick: async () => { if (confirm("Archive this note? It is kept but no longer shown.")) { try { await api("/v1/notes/" + n.id, { method: "DELETE" }); reload(); } catch (e) { alert(e.message); } } } }, "Archive")) : null);
    };
    view();
    return card;
  }

  // Notes for one thing (shown on its page).
  function notesPanel(stype, sid) {
    const box = h("div", { class: "notes" });
    const load = async () => {
      try {
        const rows = await api("/v1/notes?subject_type=" + encodeURIComponent(stype) + "&subject_id=" + encodeURIComponent(sid) + "&limit=100");
        const add = h("div");
        const addBtn = can("write:notes") ? h("button", { onclick: () => { addBtn.hidden = true; add.replaceChildren(noteForm(null, async (b) => { await api("/v1/notes", { method: "POST", body: JSON.stringify({ ...b, subject_type: stype, subject_id: String(sid) }) }); load(); }, () => { add.replaceChildren(); addBtn.hidden = false; })); } }, "Add a note") : null;
        box.replaceChildren(h("div", { class: "row" }, h("h2", null, "Lab notes" + (rows.length ? " (" + rows.length + ")" : "")), addBtn), add,
          ...(rows.length ? rows.map((n) => noteCard(n, load, false)) : [h("p", { class: "muted" }, "Nothing recorded here yet. Decisions, warnings and lessons about this " + (NOTE_SUBJECTS[stype] || "item").toLowerCase() + " go here.")]));
      } catch (e) { box.replaceChildren(h("div", { class: "notice" }, e.message)); }
    };
    load();
    return box;
  }

  // The Notes page: everything the lab remembers, with filters and search.
  async function notesPage() {
    const f = { q: "", kind: "", subject_type: "", pinned: "", archived: false };
    const list = h("div");
    const q = h("input", { id: "nq", type: "search", placeholder: "Search titles and text", maxlength: "80", autocomplete: "off" });
    const kind = h("select", { id: "nfk" }, [h("option", { value: "" }, "All types"), ...NOTE_KINDS.map((k) => h("option", { value: k }, k))]);
    const subj = h("select", { id: "nfs" }, [h("option", { value: "" }, "Everything"), ...Object.entries(NOTE_SUBJECTS).map(([k, l]) => h("option", { value: k }, l))]);
    const arch = h("input", { id: "nfa", type: "checkbox" });
    const add = h("div");
    const load = async () => {
      const p = new URLSearchParams({ limit: "200" });
      if (q.value.trim()) p.set("q", q.value.trim());
      if (kind.value) p.set("kind", kind.value);
      if (subj.value) p.set("subject_type", subj.value);
      if (arch.checked) p.set("include_archived", "true");
      try {
        const rows = await api("/v1/notes?" + p);
        list.replaceChildren(...(rows.length ? rows.map((n) => noteCard(n, load, true)) : [empty("No notes", "Nothing matches. Add a note, or clear the filters.")]));
      } catch (e) { list.replaceChildren(h("div", { class: "notice" }, e.message)); }
    };
    [kind, subj, arch].forEach((el) => el.addEventListener("change", load));
    q.addEventListener("keydown", (e) => { if (e.key === "Enter") load(); });
    const addBtn = can("write:notes") ? h("button", { class: "primary", onclick: () => { addBtn.hidden = true; add.replaceChildren(noteForm(null, async (b) => { await api("/v1/notes", { method: "POST", body: JSON.stringify({ ...b, subject_type: "general" }) }); addBtn.hidden = false; add.replaceChildren(); load(); }, () => { add.replaceChildren(); addBtn.hidden = false; })); } }, "New lab note") : null;
    await load();
    return [h("h1", null, "Notes"),
      h("p", null, "The laboratory's memory: decisions, findings, lessons, warnings and open questions that people have recorded. Notes about one dataset, experiment or model also appear on its own page. Notes here are words only; they never hold patient-level data."),
      h("p", null, addBtn), add,
      h("div", { class: "inline" }, h("div", null, h("label", { for: "nq" }, "Search"), q), h("div", null, h("label", { for: "nfk" }, "Type"), kind), h("div", null, h("label", { for: "nfs" }, "About"), subj)),
      h("p", null, h("label", null, arch, " Show archived")), list];
  }

  const svgEl = (tag, attrs) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [a, v] of Object.entries(attrs || {})) e.setAttribute(a, v); return e; };
  function divBars(labels, values, caption) {
    const W = 560, pl = 190, pr = 50, rowH = 20, H = 12 + labels.length * rowH, mx = Math.max(0.0001, ...values.map((v) => Math.abs(v || 0))), mid = pl + (W - pl - pr) / 2, half = (W - pl - pr) / 2;
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", width: "100%", "aria-label": caption });
    svg.appendChild(svgEl("line", { x1: mid, x2: mid, y1: 4, y2: H - 4, stroke: "#9fb0b6" }));
    labels.forEach((lab, i) => {
      const y = 8 + i * rowH, v = values[i] || 0, w = (Math.abs(v) / mx) * half, t = svgEl("text", { x: pl - 8, y: y + 11, "text-anchor": "end", "font-size": 10.5, fill: "#34454c" });
      t.textContent = lab.length > 30 ? lab.slice(0, 29) + "…" : lab; svg.appendChild(t);
      svg.appendChild(svgEl("rect", { x: v >= 0 ? mid : mid - w, y, width: Math.max(1, w).toFixed(1), height: 13, rx: 2, fill: v >= 0 ? "#c0392b" : "#0e5a73" }));
      const n = svgEl("text", { x: v >= 0 ? mid + w + 4 : mid - w - 4, y: y + 11, "text-anchor": v >= 0 ? "start" : "end", "font-size": 10, fill: "#34454c" }); n.textContent = v.toFixed(2); svg.appendChild(n);
    });
    return h("div", null, svg, h("p", { class: "muted" }, caption));
  }
  function lineChart(vals, caption) {
    if (vals.length < 2) return null;
    const W = 560, H = 200, pl = 46, pr = 14, pt = 12, pb = 24, y1 = Math.max(...vals), y0 = Math.min(...vals), sx = (i) => pl + (i / (vals.length - 1)) * (W - pl - pr), sy = (v) => pt + (1 - (v - y0) / (y1 - y0 || 1)) * (H - pt - pb);
    const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", width: "100%", "aria-label": caption });
    [y0, (y0 + y1) / 2, y1].forEach((v) => { const y = sy(v).toFixed(1); svg.appendChild(svgEl("line", { x1: pl, x2: W - pr, y1: y, y2: y, stroke: "#e3eaec" })); const t = svgEl("text", { x: pl - 6, y: Number(y) + 4, "text-anchor": "end", "font-size": 10.5, fill: "#5b6b72" }); t.textContent = v.toFixed(3); svg.appendChild(t); });
    svg.appendChild(svgEl("path", { d: vals.map((v, i) => (i ? "L" : "M") + sx(i).toFixed(1) + " " + sy(v).toFixed(1)).join(" "), fill: "none", stroke: "#0e5a73", "stroke-width": 2 }));
    const a = svgEl("text", { x: pl, y: H - 6, "font-size": 10.5, fill: "#5b6b72" }); a.textContent = "pass 1"; svg.appendChild(a);
    const z = svgEl("text", { x: W - pr, y: H - 6, "text-anchor": "end", "font-size": 10.5, fill: "#5b6b72" }); z.textContent = "pass " + vals.length; svg.appendChild(z);
    return h("div", null, svg, h("p", { class: "muted" }, caption));
  }
  const nlpF1Chart = (labels, model, rules) => labels.length ? barChart(labels, [{ name: "Model F1", color: "#0e5a73", values: model }, { name: "Rules F1", color: "#8a9ba2", values: rules }], "Higher is better. Compare the model with the rules baseline on the test documents.") : null;
  const perClassChart = (pc) => { const k = Object.keys(pc || {}); return k.length ? barChart(k, [{ name: "Precision", color: "#8a9ba2", values: k.map((x) => pc[x].precision) }, { name: "Recall", color: "#c0392b", values: k.map((x) => pc[x].recall) }, { name: "F1", color: "#0e5a73", values: k.map((x) => pc[x].f1) }], "Per-class precision, recall and F1 on the hold-out test.") : null; };

  const perfChart = (m) => { const t = m.test || {}, b = m.baseline || {};
    const cls = m.task === "classification", L = cls ? ["Macro F1", "Balanced accuracy", "Accuracy", "ROC AUC"] : ["R²"], k = cls ? ["macro_f1", "balanced_accuracy", "accuracy", "roc_auc"] : ["r2"];
    return barChart(L, [{ name: "Baseline", color: "#8a9ba2", values: k.map((x) => b[x]) }, { name: "Model", color: "#0e5a73", values: k.map((x) => t[x]) }], "Hold-out test, model against the trivial baseline. The model should clear the baseline by a clear margin. Bars start at 0 and stop at 100%."); };
  function cvChart(m) { const cv = m.cv || {}, cls = m.task === "classification", a = cls ? cv.macro_f1_mean : cv.r2_mean, sd = cls ? cv.macro_f1_sd : cv.r2_sd, t = (m.test || {})[cls ? "macro_f1" : "r2"];
    return a == null || t == null ? null : h("p", { class: "muted" }, (cls ? "Macro F1" : "R²") + ": cross-validation " + n3(a) + " ± " + n3(sd) + " versus hold-out " + n3(t) + ". A hold-out far below cross-validation suggests the model does not generalise."); }
  function barChart(labels, series, caption) {
    const NS = "http://www.w3.org/2000/svg", W = 560, rowH = 20 + series.length * 14, pl = 150, pr = 44, H = 12 + labels.length * rowH;
    const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const [a, v] of Object.entries(attrs || {})) e.setAttribute(a, v); return e; };
    const svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", width: "100%", "aria-label": caption });
    const sx = (v) => pl + Math.max(0, Math.min(1, v)) * (W - pl - pr);
    [0, 0.25, 0.5, 0.75, 1].forEach((g) => svg.appendChild(el("line", { x1: sx(g), x2: sx(g), y1: 4, y2: H - 4, stroke: "#e3eaec" })));
    labels.forEach((lab, i) => {
      const y0 = 8 + i * rowH, t = el("text", { x: pl - 8, y: y0 + rowH / 2, "text-anchor": "end", "font-size": 11, fill: "#34454c" }); t.textContent = lab; svg.appendChild(t);
      series.forEach((s, k) => {
        const v = s.values[i], y = y0 + k * 14;
        if (v == null) { const n = el("text", { x: pl + 4, y: y + 10, "font-size": 10.5, fill: "#5b6b72" }); n.textContent = "no data"; svg.appendChild(n); return; }
        svg.appendChild(el("rect", { x: pl, y, width: Math.max(1, sx(v) - pl).toFixed(1), height: 11, rx: 2, fill: s.color }));
        const n = el("text", { x: sx(v) + 4, y: y + 10, "font-size": 10.5, fill: "#34454c" }); n.textContent = (100 * v).toFixed(1) + "%"; svg.appendChild(n);
      });
    });
    const legend = series.length > 1 ? h("p", { class: "muted" }, series.map((s) => h("span", { style: "margin-right:12px" }, h("span", { style: "color:" + s.color }, "■ "), s.name))) : null;
    return h("div", null, svg, legend, h("p", { class: "muted" }, caption));
  }

  function table(head, rows) {
    return h("div", { class: "tablewrap" }, h("table", null, h("thead", null, h("tr", null, head.map((x) => h("th", { scope: "col" }, x)))),
      h("tbody", null, rows.map((r) => h("tr", null, r.map((c) => h("td", null, c)))))));
  }

  async function show(id, quiet = false) {
    clearTimeout(poll);
    const [pid, arg] = String(id || "").split("/");
    const page = PAGES.find((p) => p.id === pid) || PAGES[0];
    document.querySelectorAll("#nav button").forEach((b) => { if (b.dataset.id === page.id) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
    const view = $("view");
    if (!quiet) view.replaceChildren(loader("Loading " + page.label.toLowerCase() + "…"));
    silent = quiet;
    try { view.replaceChildren(...[].concat(await page.render(arg)).flat().filter(Boolean)); }
    catch (e) { view.replaceChildren(h("h1", null, page.label), h("div", { class: "notice" }, e.message)); }
    finally { silent = false; }
    view.focus({ preventScroll: true });
  }

  async function start() {
    try { me = await api("/v1/me"); } catch (e) { return showLogin(e.message); }
    $("login").hidden = true; $("app").hidden = false; $("boot").hidden = true;
    $("who").textContent = me.name + " (" + me.role.replace("_", " ") + ")";
    const allowed = PAGES.filter((p) => !p.perm || me.permissions.includes(p.perm));
    $("nav").replaceChildren(...allowed.map((p) => h("button", { "data-id": p.id, onclick: () => { location.hash = p.id; } }, p.label)));
    show(location.hash.slice(1));
  }
  function showLogin(msg) { $("boot").hidden = true; $("app").hidden = true; $("login").hidden = false; $("login-error").textContent = msg || ""; $("key").focus(); }
  function signOut() { store.clear(); me = null; showLogin(); }

  $("signin").addEventListener("click", () => { const k = $("key").value.trim(); if (!k) return; store.set(k); $("key").value = ""; start(); });
  $("key").addEventListener("keydown", (e) => { if (e.key === "Enter") $("signin").click(); });
  $("signout").addEventListener("click", signOut);
  window.addEventListener("hashchange", () => me && show(location.hash.slice(1)));
  store.get() ? start() : showLogin();
})();
