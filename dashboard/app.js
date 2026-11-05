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
    try {
      const headers = { Authorization: "Bearer " + store.get() };
      if (!(opts.body instanceof FormData)) headers["Content-Type"] = "application/json";
      res = await fetch(API + path, { ...opts, headers });
    } catch { throw new Error("Cannot reach the ML API. Check your connection and try again."); }
    if (res.status === 401) { signOut(); throw new Error("Your key was not accepted."); }
    if (!res.ok) {
      const b = await res.json().catch(() => ({}));
      let d = b.detail;
      // FastAPI sends validation problems as a list of objects; turn them into readable text.
      if (Array.isArray(d)) d = d.map((e) => ((e.loc || []).slice(1).join(" ") || "input") + ": " + (e.msg || "invalid")).join("; ");
      throw new Error(d || "Request failed (" + res.status + ").");
    }
    return res.status === 204 ? null : res.json();
  }

  const PAGES = [
    { id: "overview", label: "Overview", perm: null, render: overview },
    { id: "models", label: "Models", perm: "read:catalog", render: models },
    { id: "experiments", label: "Experiments", perm: "read:experiments", render: experiments },
    { id: "explorations", label: "Exploration", perm: "read:explorations", render: explorations },
    { id: "hotspots", label: "Hotspots", perm: "read:hotspots", render: hotspots },
    { id: "predict", label: "Predict", perm: "predict:models", render: predictPage },
    { id: "registry", label: "Registry", perm: "read:models", render: registry },
    { id: "datasets", label: "Datasets", perm: "read:datasets", render: datasets },
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
      h("p", null, "The Machine Learning Laboratory of the Chakudya Nutrition Registry (CNR). Upload and de-identify datasets, train and evaluate models, review and approve them in the registry, and serve the deployed version to CNR. Predictions are decision support, never confirmed diagnoses."),
      h("div", { class: "notice" }, o.safety_notice),
      h("div", { class: "stats" }, [["Planned models", c.model_definitions], ["Datasets", c.datasets], ["Experiments", c.experiments], ["Explorations", c.explorations], ["Hotspot analyses", c.hotspot_analyses], ["Model versions", c.model_versions], ["Deployed models", c.deployed_models]]
        .map(([l, n]) => h("div", { class: "stat" }, h("b", null, n), h("span", null, l)))),
      c.datasets === 0 ? empty("No datasets yet", "Create one under Datasets and upload a file. Nothing has been trained and no model is available for predictions.") : null,
      h("p", { class: "muted" }, "API " + o.service.version + " · " + o.service.environment),
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
    const model = h("select", { id: "xm" }, [h("option", { value: "" }, "Choose a model…")].concat(models.map((m) => h("option", { value: m.slug }, m.name + " (" + m.task_type + ")"))));
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
    const drawCols = () => {
      const prev = target.value, names = cols().map((c) => c.name);
      target.replaceChildren(h("option", { value: "" }, "Choose the column to predict…"), ...names.map((n) => h("option", { value: n, selected: n === prev }, n)));
      drawExcl();
    };
    const drawExcl = () => excl.replaceChildren(...cols().filter((c) => c.name !== target.value).map((c) =>
      h("label", { class: "chk" }, h("input", { type: "checkbox", value: c.name }), " " + c.name)));
    const loadVersions = async () => {
      const d = await api("/v1/datasets/" + dset.value);
      versions = d.versions;
      ver.replaceChildren(...versions.map((v) => h("option", { value: v.version }, "v" + v.version + " (" + v.row_count + " rows)")));
      drawCols();
    };
    dset.addEventListener("change", loadVersions); ver.addEventListener("change", drawCols); target.addEventListener("change", drawExcl);
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
      h("label", null, "Columns to leave out of training"), excl,
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
      out.push(h("h2", null, "Per class"), table(["Class", "Precision", "Recall", "F1", "Test cases"], Object.entries(m.per_class || {}).map(([k, v]) => [k, v.precision, v.recall, v.f1, v.support])));
      const cm = m.confusion_matrix || { labels: [], matrix: [] };
      out.push(h("h2", null, "Confusion matrix"), h("p", { class: "muted" }, "Rows are the true class, columns the predicted class."),
        table(["True \\ Predicted", ...cm.labels], cm.matrix.map((row, i) => [cm.labels[i], ...row])));
    } else {
      out.push(h("h2", null, "Hold-out test (" + m.n_test + " rows)"),
        h("div", { class: "stats" }, stat("R²", n3(t.r2), "baseline " + n3(b.r2)), stat("RMSE", n3(t.rmse), "95% range " + n3(t.rmse_ci95 && t.rmse_ci95[0]) + "–" + n3(t.rmse_ci95 && t.rmse_ci95[1])),
          stat("Baseline RMSE", n3(b.rmse), b.strategy), stat("MAE", n3(t.mae))),
        h("p", { class: "muted" }, "Cross-validation (" + cv.folds + " folds): R² " + n3(cv.r2_mean) + " ± " + n3(cv.r2_sd) + ", RMSE " + n3(cv.rmse_mean) + " ± " + n3(cv.rmse_sd) + "."));
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
        const body = { dataset_id: Number(dset.value), version: Number(ver.value), methods, exclude: [...excl.querySelectorAll("input:checked")].map((i) => i.value),
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
      box.replaceChildren(
        st.configured
          ? h("p", null, "Pulls " + st.targets.join(", ") + " (last " + st.months + " months) from " + st.source + " every " + st.interval_hours + " h, stores each change as a new dataset version" + (st.run_analysis ? " and runs a hotspot analysis on it." : "."))
          : h("div", { class: "notice" }, st.problem + " To switch it on, set SURVEILLANCE_SYNC_ENABLED=true and SURVEILLANCE_API_KEY (a CNR admin key) on the API service."),
        lr ? h("p", { class: "muted" }, "Last run: " + (lr.trigger === "manual" ? "started by an admin" : "scheduled") + ", " + (lr.state === "running" ? "running now" : lr.state.replace(/_/g, " ")) + (lr.finished_at ? " at " + fmt(lr.finished_at) : "") + (lr.error ? ". " + lr.error : "") + ".") : null,
        lr && lr.results && lr.results.length
          ? table(["Group", "Indicator", "Result", "Dataset version", "Hotspot analysis"], lr.results.map((r) => [r.population_group, r.indicator,
              (syncText[r.status] || r.status) + (r.error ? ": " + r.error : r.note ? ": " + r.note : ""), r.version ? "v" + r.version : "—",
              r.analysis ? (r.analysis_id ? h("button", { class: "link", onclick: () => { location.hash = "hotspots/" + r.analysis_id; } }, r.analysis) : r.analysis) : "—"]))
          : null);
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

  async function integrationPage() {
    const s = await api("/v1/integration/cnr/status");
    const sy = await api("/v1/sync/surveillance").catch(() => null);
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
    const back = () => { location.hash = "registry"; };
    const reload = () => show("registry/" + id);
    const run = (label, path, body, ask, cls) => h("button", { class: cls || "primary", onclick: async (ev) => {
      if (ask && !confirm(ask)) return;
      ev.target.disabled = true;
      try { await api(path, { method: "POST", body: JSON.stringify(body || {}) }); reload(); } catch (e) { alert(e.message); ev.target.disabled = false; }
    } }, label);
    const out = [h("p", null, h("button", { class: "link", onclick: back }, "← Registry")),
      h("div", { class: "row" }, h("h1", null, (v.model_name || v.model_slug) + " · v" + v.version), vBadge(v.status)),
      h("p", { class: "muted" }, "Experiment #" + v.experiment_id + " · registered by " + (v.created_by || "—") + " on " + fmt(v.created_at) + (v.artifact_sha256 ? " · file SHA-256 " + v.artifact_sha256.slice(0, 12) + "…" : "")),
      h("div", { class: "notice" }, "Intended use: " + (v.intended_use || "—") + " Outputs are risk estimates to support clinical judgement, never confirmed diagnoses.")];
    (m.warnings || []).forEach((w) => out.push(h("div", { class: "notice" }, w)));
    const stat = (l, x, sub) => h("div", { class: "stat" }, h("b", null, x), h("span", null, l + (sub ? " · " + sub : "")));
    out.push(h("h2", null, "Evaluation (frozen at registration)"));
    out.push(h("div", { class: "stats" }, m.task === "classification"
      ? [stat("Macro F1", n3(t.macro_f1), "95% range " + n3(t.macro_f1_ci95 && t.macro_f1_ci95[0]) + "–" + n3(t.macro_f1_ci95 && t.macro_f1_ci95[1])), stat("Baseline macro F1", n3(b.macro_f1)), stat("Balanced accuracy", pct(t.balanced_accuracy)), stat("Test rows", m.n_test)]
      : [stat("R²", n3(t.r2), "baseline " + n3(b.r2)), stat("RMSE", n3(t.rmse)), stat("Test rows", m.n_test)]));
    out.push(h("p", null, h("button", { class: "link", onclick: () => { location.hash = "experiments/" + v.experiment_id; } }, "Open the full experiment report")));

    // actions
    const act = [];
    if (v.status === "candidate" && can("write:models")) act.push(h("p", null, run("Submit for clinical review", "/v1/models/versions/" + id + "/submit", {}, "Submit this version for clinical review?")));
    if (v.status === "in_review" && can("approve:models")) act.push(reviewForm(v, reload));
    if (v.status === "in_review" && !can("approve:models")) act.push(h("div", { class: "notice" }, "Waiting for a clinical reviewer."));
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
    const role = h("select", { id: "kr" }, ["viewer", "data_scientist", "clinical_reviewer", "service", "admin"].map((r) => h("option", { value: r }, r)));
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
