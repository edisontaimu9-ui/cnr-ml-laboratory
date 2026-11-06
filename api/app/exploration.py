"""Unsupervised exploration: clustering (k-means) and anomaly detection (isolation forest).

An exploration is a different kind of thing from an experiment, and the code keeps them apart on purpose.
- It has no label, no catalogue model and no train/test split, so nothing here can be called a validated model.
- It returns written results only. No fitted object is returned or saved (no joblib, no artifact), so there is
  nothing the registry could register and nothing serving or the CNR integration could load.
- This module must not import the registry, serving, predict, webhook or integration code. A test enforces it.

Honesty rules, in the same spirit as `training.py`
- Clustering always returns groups, even for random data. So every result carries a structure verdict built from
  the silhouette score, how stable the groups are across refits, and a column-shuffled reference with no real
  structure.
- Anomaly detection always flags a fixed share of rows by design. The result says so, and reports how much the
  flagged set changes with the random seed.
- Group profiles are aggregates over at least MIN_GROUP_REPORT rows; smaller groups are suppressed. Flagged rows are
  identified by position only, with reasons given as standardised deviations, never as raw values or category names.
"""
import re
import warnings as pywarnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MIN_ROWS = 50
MAX_WORK_ROWS = 20000        # fitting, k search, stability and the reference run on at most this many rows
MAX_PROJECTION_POINTS = 600
MIN_GROUP_REPORT = 5         # groups smaller than this are never profiled (same threshold as dataset profiles)
SIL_SAMPLE = 3000
STABILITY_REFITS = 5
REFERENCE_SHUFFLES = 3

NOTICE = ("Exploratory analysis only. Groups and flags are patterns found in this one dataset. They are not diagnoses, "
          "risk classes or validated findings, and an exploration can never be registered, approved, deployed or sent to CNR.")
LIMITATIONS = [
    "Groups and flags describe patterns in this dataset only. They have not been checked against outcomes or validated externally.",
    "K-means returns compact, similar-sized groups even when the data has none. Read the structure verdict before interpreting any group.",
    "An unusual row is not necessarily an error or an unwell person. It can be a data-entry problem, a rare but genuine case, or a small subgroup.",
    "Nothing from an exploration is saved as a model. It cannot be used to predict anything for a new record.",
]


PRESETS = ("cardiometabolic",)

# Cardiometabolic marker groups, recognised from column names (whole words, or substrings of the run-together name).
MARKER_GROUPS = {
    "adiposity": ({"bmi", "whr", "hip", "waist", "weight", "bodyfat"}, ("waist", "bodyfat", "adipos")),
    "blood_pressure": ({"sbp", "dbp", "bp", "map", "systolic", "diastolic"}, ("bloodpressure", "systolic", "diastolic")),
    "glucose": ({"hba1c", "a1c", "fpg", "fbg", "fbs", "rbs", "ogtt", "glucose", "insulin", "homa"}, ("hba1c", "glucose", "homair")),
    "lipids": ({"tg", "hdl", "ldl", "chol", "cholesterol", "triglycerides", "lipid"}, ("triglycer", "cholesterol", "lipid")),
    "lifestyle": ({"salt", "sugar", "alcohol", "smoking", "tobacco", "activity", "sedentary", "sleep", "fruit", "vegetable", "diet"},
                  ("physicalactivity", "sugary", "fruitveg")),
}
# Columns that carry an already-assigned label, diagnosis, outcome or treatment: groups built on them just reproduce it.
LABEL_LIKE = ("syndrome", "diagnos", "outcome", "label", "prediab", "hypertens", "diabetes_status", "diabetic", "metformin",
              "statin", "antihypert", "treatment", "medication", "riskclass", "risk_class")
HIGH_CORR = 0.9
DOMINANCE = 0.6


class ExplorationError(ValueError):
    """Message is safe to show to the user."""


@dataclass
class Config:
    features: list[str] | None = None
    exclude: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=lambda: ["clustering", "anomaly"])
    k: int | None = None
    k_min: int = 2
    k_max: int = 8
    flag_share: float = 0.05
    report_top: int = 20
    seed: int = 42
    preset: str | None = None                  # None | "cardiometabolic" (see _preset)


# ---------- preparation ----------

def _columns(df: pd.DataFrame, cfg: Config, warnings: list[str]) -> tuple[list[str], list[str]]:
    excl = set(cfg.exclude)
    cand = cfg.features or [c for c in df.columns if c not in excl]
    cand = [c for c in cand if c not in excl]
    missing = [c for c in cand if c not in df.columns]
    if missing:
        raise ExplorationError("Unknown column: " + missing[0][:80])
    n = len(df)
    num, cat = [], []
    for c in cand:
        s = df[c]
        nun = s.nunique(dropna=True)
        miss = float(s.isna().mean())
        if nun <= 1:
            warnings.append(f"'{c}' was skipped because it has a single value.")
            continue
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if is_num and pd.api.types.is_integer_dtype(s) and nun > 0.95 * n:
            warnings.append(f"'{c}' was skipped: nearly every value is different (looks like a row number or identifier).")
            continue
        if not is_num and nun > max(50, 0.5 * n):
            warnings.append(f"'{c}' was skipped: too many distinct values (looks like an identifier or free text).")
            continue
        if miss > 0.4:
            warnings.append(f"'{c}' is {miss:.0%} missing. Missing values were filled in (median, or 'missing' for categories), "
                            "which can create artificial groups. Consider leaving this column out.")
        (num if is_num else cat).append(c)
    if len(num) + len(cat) < 2:
        raise ExplorationError("At least two usable columns are needed. Check the columns you left out.")
    return num, cat


def _encoder(num: list[str], cat: list[str]) -> ColumnTransformer:
    # Numeric columns are standardised; categories are one-hot encoded (0/1). The order is numeric first, then categories.
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


def _clean(df: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    out = df[num + cat].copy()
    for c in cat:
        out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
        out[c] = out[c].map(lambda v: v if v is np.nan or pd.isna(v) else str(v))
    return out


def _r(x, nd: int = 4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


# ---------- cardiometabolic preset ----------

def _marker_group(col: str) -> str | None:
    low = col.lower()
    words = {w for w in re.split(r"[^a-z0-9]+", low) if w}
    flat = low.replace("_", "").replace(" ", "")
    for name, (tokens, subs) in MARKER_GROUPS.items():
        if words & tokens or any(sub in flat for sub in subs):
            return name
    return None


def _preset(data: pd.DataFrame, num: list[str], cat: list[str], warnings: list[str]) -> dict:
    """Cardiometabolic phenotyping: checks that the columns suit it and says what the groups can and cannot mean.
    Everything here is a written note; it changes no fitted result."""
    cols = num + cat
    groups: dict[str, list[str]] = {}
    other: list[str] = []
    for c in cols:
        g = _marker_group(c)
        (groups.setdefault(g, []) if g else other).append(c)
    notes: list[str] = []
    label_like = [c for c in cols if any(t in c.lower().replace(" ", "_") for t in LABEL_LIKE)]
    if label_like:
        warnings.append(f"'{label_like[0][:60]}' looks like an already-assigned diagnosis, label, outcome or treatment"
                        + (f" (and {len(label_like) - 1} more)" if len(label_like) > 1 else "")
                        + ". Groups that include it mostly reproduce that column instead of finding patterns in the measurements. Leave it out and "
                        "compare the groups with it afterwards.")
    if not groups:
        warnings.append("No cardiometabolic measurements were recognised from the column names (waist, BMI, blood pressure, glucose or HbA1c, "
                        "lipids, diet or activity). The groups may describe something else, such as age or place.")
    elif len(groups) == 1:
        warnings.append(f"Only one kind of cardiometabolic measurement was recognised ({next(iter(groups)).replace('_', ' ')}). Phenotypes "
                        "built on one axis are just bands of that measurement.")
    sizes = {g: len(v) for g, v in groups.items()}
    total = sum(sizes.values())
    if len(groups) > 1 and total >= 4:
        big = max(sizes, key=sizes.get)
        if sizes[big] / total > DOMINANCE:
            warnings.append(f"{sizes[big]} of the {total} recognised measurements are {big.replace('_', ' ')} columns. K-means weights every column equally, "
                            "so these will dominate the groups. Consider keeping one or two of them.")
    nums = [c for c in num if data[c].notna().sum() >= 20 and data[c].nunique() > 1]
    pairs = []
    if len(nums) >= 2:
        corr = data[nums].corr(method="spearman").abs()
        for i, a in enumerate(nums):
            for b in nums[i + 1:]:
                v = corr.loc[a, b]
                if np.isfinite(v) and v >= HIGH_CORR:
                    pairs.append((a, b, float(v)))
    pairs.sort(key=lambda t: -t[2])
    if pairs:
        shown = "; ".join(f"{a} and {b} (rho {v:.2f})" for a, b, v in pairs[:3])
        warnings.append(f"Some columns move almost together: {shown}. Measuring the same thing twice doubles its weight in the groups. "
                        "Keep one of each pair.")
    notes.append("Phenotypes are descriptive patterns of measurements in this dataset. They are not disease classes, risk tiers or a "
                 "metabolic syndrome definition, and a group should not be named or treated as one.")
    notes.append("Age, sex and body size often drive the groups. Check each group's profile for that before reading it as a metabolic pattern.")
    return {"name": "cardiometabolic", "marker_groups": {g: groups[g] for g in sorted(groups)}, "not_recognised": other,
            "label_like_columns": label_like, "correlated_pairs": [{"a": a, "b": b, "rho": _r(v, 2)} for a, b, v in pairs[:10]],
            "notes": notes}


# ---------- clustering ----------

def _silhouette(X: np.ndarray, labels: np.ndarray, seed: int) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(silhouette_score(X, labels, sample_size=min(len(X), SIL_SAMPLE), random_state=seed))


def _structure_verdict(sil: float, stability: float, ref: float) -> tuple[str, str]:
    if sil < 0.25 or stability < 0.6 or sil - ref < 0.05:
        return "no_clear_structure", ("No clear group structure. The groups are not well separated, are not stable when the data is "
                                      "resampled, or look no better than shuffled data. Treat them as an arbitrary way of slicing the data.")
    if sil >= 0.5 and stability >= 0.8:
        return "clear_structure", "The groups are well separated and stable when the data is resampled, and clearly beat shuffled data."
    return "weak_structure", ("Some structure, but the groups overlap or move a little between refits. Use them as a lead to look at, "
                              "not as established categories.")


def _profile(df: pd.DataFrame, labels: np.ndarray, num: list[str], cat: list[str]) -> list[dict]:
    n = len(df)
    ov_mean = {c: float(df[c].mean()) for c in num}
    ov_sd = {c: float(df[c].std(ddof=0)) for c in num}
    ov_share = {c: df[c].astype(str).where(df[c].notna(), "missing").value_counts(normalize=True) for c in cat}
    out = []
    for g in sorted(set(labels)):
        mask = labels == g
        size = int(mask.sum())
        item = {"cluster": int(g), "size": size, "share": _r(size / n)}
        if size < MIN_GROUP_REPORT:
            item.update(suppressed=True, note=f"Fewer than {MIN_GROUP_REPORT} rows, so no profile is shown.")
            out.append(item)
            continue
        sub = df[mask]
        nums = []
        for c in num:
            if ov_sd[c] <= 0 or sub[c].notna().sum() == 0:
                continue
            m = float(sub[c].mean())
            nums.append({"feature": c, "mean": _r(m), "overall_mean": _r(ov_mean[c]), "z": _r((m - ov_mean[c]) / ov_sd[c], 2)})
        nums.sort(key=lambda d: -abs(d["z"] or 0))
        cats = []
        for c in cat:
            vc = sub[c].astype(str).where(sub[c].notna(), "missing").value_counts()
            lvl, cnt = vc.index[0], int(vc.iloc[0])
            if cnt < MIN_GROUP_REPORT:
                continue
            share, base = cnt / size, float(ov_share[c].get(lvl, 0.0))
            cats.append({"feature": c, "level": str(lvl)[:60], "share": _r(share), "overall_share": _r(base), "gap": _r(share - base)})
        cats.sort(key=lambda d: -abs(d["gap"] or 0))
        item.update(numeric=nums[:6], categorical=cats[:3])
        out.append(item)
    return out


def _cluster(df: pd.DataFrame, pre: ColumnTransformer, X: np.ndarray, Xw: np.ndarray, widx: np.ndarray,
             num: list[str], cat: list[str], cfg: Config, warnings: list[str]) -> tuple[dict, np.ndarray, KMeans]:
    ks = [cfg.k] if cfg.k else list(range(cfg.k_min, cfg.k_max + 1))
    ks = [k for k in ks if 2 <= k < len(Xw)]
    sweep = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(Xw)
        sil = _silhouette(Xw, km.labels_, cfg.seed)
        if sil is None:       # k-means collapsed to one group (many duplicate rows)
            continue
        smallest = float(np.bincount(km.labels_, minlength=k).min()) / len(Xw)
        sweep.append({"k": k, "silhouette": _r(sil), "inertia": _r(km.inertia_, 2), "smallest_group_share": _r(smallest)})
    if not sweep:
        raise ExplorationError("Could not find separate groups in these columns. Check for many identical rows or choose other columns.")
    best = max(sweep, key=lambda d: (d["silhouette"], -d["k"]))
    k = best["k"]
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(Xw)
    labels_w = km.labels_
    labels_full = km.predict(X)

    # Stability: refit on random 80% subsamples and compare with the reference grouping (adjusted Rand index, 1 = identical).
    rng = np.random.default_rng(cfg.seed)
    aris = []
    for i in range(STABILITY_REFITS):
        sub = rng.choice(len(Xw), size=max(k + 1, int(0.8 * len(Xw))), replace=False)
        k2 = KMeans(n_clusters=k, n_init=5, random_state=cfg.seed + 1 + i).fit(Xw[sub])
        aris.append(adjusted_rand_score(labels_w, k2.predict(Xw)))
    stab_mean, stab_sd = float(np.mean(aris)), float(np.std(aris))

    # Reference: the same analysis on data whose columns were shuffled independently, which removes the relationships
    # between columns. Real structure should score clearly higher than this.
    dfw = df.iloc[widx]
    refs = []
    for i in range(REFERENCE_SHUFFLES):
        r = np.random.default_rng(cfg.seed + 100 + i)
        shuf = pd.DataFrame({c: r.permutation(dfw[c].to_numpy()) for c in dfw.columns})
        Xr = pre.transform(shuf)
        kr = KMeans(n_clusters=k, n_init=5, random_state=cfg.seed).fit(Xr)
        s = _silhouette(Xr, kr.labels_, cfg.seed)
        if s is not None:
            refs.append(s)
    ref_sil = float(np.mean(refs)) if refs else 0.0

    sil = best["silhouette"]
    verdict, text = _structure_verdict(sil, stab_mean, ref_sil)
    sizes = np.bincount(labels_full, minlength=k)
    if verdict == "no_clear_structure":
        warnings.append("Clustering found no clear structure (see the verdict). The groups below are not evidence of real subtypes.")
    if sizes.min() / len(labels_full) < 0.02:
        warnings.append("At least one group holds under 2% of the rows. A very small group is often a handful of outliers rather than a real group.")
    if not cfg.k and len(sweep) > 2 and (k == cfg.k_max or (k == cfg.k_min and cfg.k_min > 2)):
        warnings.append(f"The best score was at the edge of the range searched (k = {k}). A different range might score better.")

    res = {"algorithm": "kmeans", "k": k, "k_selected_by": "highest silhouette in the range searched" if not cfg.k else "set by the requester",
           "quality": {"silhouette": sil, "davies_bouldin": _r(davies_bouldin_score(Xw, labels_w)),
                       "calinski_harabasz": _r(calinski_harabasz_score(Xw, labels_w), 1), "scored_on_rows": int(len(Xw))},
           "stability": {"refits": STABILITY_REFITS, "adjusted_rand_mean": _r(stab_mean), "adjusted_rand_sd": _r(stab_sd)},
           "reference": {"strategy": "same analysis on data with every column shuffled independently", "silhouette": _r(ref_sil),
                         "shuffles": len(refs)},
           "structure": verdict, "structure_summary": text, "k_search": sweep,
           "groups": _profile(df, labels_full, num, cat)}
    return res, labels_full, km


# ---------- anomaly detection ----------

def _anomaly(df: pd.DataFrame, X: np.ndarray, Xw: np.ndarray, num: list[str], cat: list[str], cfg: Config,
             labels_full: np.ndarray | None, warnings: list[str]) -> tuple[dict, IsolationForest, np.ndarray]:
    n = len(X)

    def fit(seed: int) -> IsolationForest:
        return IsolationForest(n_estimators=200, max_samples=min(256, len(Xw)), random_state=seed, n_jobs=1).fit(Xw)

    iso = fit(cfg.seed)
    raw = -iso.score_samples(X)                       # higher = easier to isolate = more unusual
    n_flag = max(1, int(round(cfg.flag_share * n)))
    order = np.argsort(-raw, kind="stable")
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:n_flag]] = True

    raw2 = -fit(cfg.seed + 1).score_samples(X)
    flagged2 = np.zeros(n, dtype=bool)
    flagged2[np.argsort(-raw2, kind="stable")[:n_flag]] = True
    rho = float(spearmanr(raw, raw2)[0])
    jac = float((flagged & flagged2).sum() / max(1, (flagged | flagged2).sum()))
    if jac < 0.6:
        warnings.append("The flagged rows change a lot when the random seed changes. Treat the flags as a soft lead, not a firm list.")

    # Why a row looks unusual: standardised deviation per numeric column, and rarity of a category. Never raw values or level names.
    zn = X[:, :len(num)]
    cat_share = {}
    for c in cat:
        lv = df[c].astype(str).where(df[c].notna(), "missing")
        cat_share[c] = lv.map(lv.value_counts(normalize=True)).to_numpy()
    top = []
    for rank, i in enumerate(order[:cfg.report_top], start=1):
        reasons = []
        for j in np.argsort(-np.abs(zn[i]))[:3]:
            if abs(zn[i, j]) >= 2.0:
                reasons.append({"feature": num[j], "kind": "numeric", "z": _r(zn[i, j], 2), "direction": "high" if zn[i, j] > 0 else "low"})
        for c in cat:
            if cat_share[c][i] < 0.05:
                reasons.append({"feature": c, "kind": "rare_category", "share_of_rows": _r(cat_share[c][i])})
        top.append({"rank": rank, "row_index": int(i), "score": _r(raw[i]), "reasons": reasons})

    prof = []
    for j, c in enumerate(num):
        m = float(zn[flagged, j].mean()) if flagged.any() else 0.0
        prof.append({"feature": c, "mean_z_flagged": _r(m, 2)})
    prof.sort(key=lambda d: -abs(d["mean_z_flagged"] or 0))

    by_cluster = None
    if labels_full is not None:
        by_cluster = []
        for g in sorted(set(labels_full)):
            m = labels_full == g
            if int(m.sum()) >= MIN_GROUP_REPORT:
                by_cluster.append({"cluster": int(g), "size": int(m.sum()), "flagged": int((flagged & m).sum()),
                                   "flagged_share": _r((flagged & m).sum() / m.sum())})
    warnings.append(f"The {cfg.flag_share:.1%} flagged is a setting, not a finding: exactly that share of rows is always flagged, "
                    "whether or not any row is truly unusual. Look at the scores and reasons, not just the flag.")
    res = {"algorithm": "isolation_forest", "flag_share": cfg.flag_share, "n_flagged": int(n_flag),
           "score_note": "Higher means easier to isolate from the rest. Only the ranking is meaningful; scores are not probabilities.",
           "score_quantiles": {f"p{q}": _r(np.percentile(raw, q)) for q in (50, 90, 95, 99)},
           "stability": {"rank_correlation_between_seeds": _r(rho, 3), "flagged_set_overlap": _r(jac, 3)},
           "row_index_note": "row_index is the position (from 0) in the stored, de-identified dataset version.",
           "top": top, "flagged_profile": prof[:8], "by_cluster": by_cluster}
    return res, iso, raw


# ---------- entry point ----------

def run(df: pd.DataFrame, cfg: Config) -> dict:
    """Returns the written result. No fitted object leaves this function."""
    for m in cfg.methods:
        if m not in ("clustering", "anomaly"):
            raise ExplorationError("Unknown method. Choose clustering, anomaly or both.")
    if not cfg.methods:
        raise ExplorationError("Choose at least one method.")
    warnings: list[str] = []
    if len(df) < MIN_ROWS:
        raise ExplorationError(f"Too few rows ({len(df)}). At least {MIN_ROWS} are needed.")
    if cfg.preset is not None and cfg.preset not in PRESETS:
        raise ExplorationError("Unknown preset. Choose cardiometabolic or leave it empty.")
    num, cat = _columns(df, cfg, warnings)
    data = _clean(df, num, cat)
    preset = _preset(data, num, cat, warnings) if cfg.preset == "cardiometabolic" else None
    pre = _encoder(num, cat)
    n = len(data)
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        X = pre.fit_transform(data)
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(cfg.seed)
        sampled = n > MAX_WORK_ROWS
        widx = np.sort(rng.choice(n, size=MAX_WORK_ROWS, replace=False)) if sampled else np.arange(n)
        Xw = X[widx]
        if sampled:
            warnings.append(f"Fitting, the search for the number of groups and the checks used a random sample of {MAX_WORK_ROWS} of {n} rows. "
                            "Every row was still assigned to a group and scored.")
        if n < 500:
            warnings.append(f"Small dataset ({n} rows). Groups and flags can shift a lot with a few rows. Treat this as a first look.")

        clustering = labels = km = None
        if "clustering" in cfg.methods:
            clustering, labels, km = _cluster(data, pre, X, Xw, widx, num, cat, cfg, warnings)
        anomaly = iso = raw = None
        if "anomaly" in cfg.methods:
            anomaly, iso, raw = _anomaly(data, X, Xw, num, cat, cfg, labels, warnings)

        # Two-dimensional view (PCA) of a sample of rows, for plotting. Coordinates are on the standardised scale.
        pts = rng.choice(len(Xw), size=min(MAX_PROJECTION_POINTS, len(Xw)), replace=False)
        pca = PCA(n_components=2, random_state=cfg.seed).fit(Xw)
        xy = pca.transform(Xw[pts])
        if km is not None:
            gl = km.labels_[pts]
        else:
            gl = np.full(len(pts), -1)
        if anomaly is not None:
            thr = np.sort(raw)[::-1][anomaly["n_flagged"] - 1]
            fl = (-iso.score_samples(Xw[pts]) >= thr).astype(int)
        else:
            fl = np.zeros(len(pts), dtype=int)
    projection = {"method": "pca", "explained_variance": [_r(v) for v in pca.explained_variance_ratio_],
                  "points": [[_r(a, 3), _r(b, 3), int(g), int(f)] for (a, b), g, f in zip(xy, gl, fl)],
                  "note": "A flat view of many columns. Overlap in this picture does not mean overlap in the full data, and the reverse."}
    return {"kind": "exploration", "deployable": False, "notice": NOTICE, "methods": list(cfg.methods),
            "n_rows": int(n), "n_used": int(len(Xw)), "sampled": bool(sampled),
            "features": {"numeric": num, "categorical": cat},
            "clustering": clustering, "anomaly": anomaly, "projection": projection, "preset": preset,
            "warnings": warnings, "limitations": LIMITATIONS}
