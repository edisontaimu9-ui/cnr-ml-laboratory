"""District hotspot and prevalence-trend analysis on AGGREGATED counts.

Input is one row per area (and, optionally, per period): a count of cases, the number of children examined
(the denominator) and, optionally, the area's centre point. No individual child is ever needed or accepted here.

What it does
- Prevalence trend: Poisson regression (area fixed effects + a common time trend, log(denominator) offset). If the
  counts vary more than Poisson allows, a negative binomial model is fitted too and the better one is used.
  Area-specific trends are fitted separately and tested for differences between areas.
- Smoothing: empirical-Bayes (Poisson-gamma) shrinkage pulls small, noisy areas towards the overall rate. With
  coordinates, a spatial version shrinks each area towards its nearest neighbours instead.
- Hotspots: the probability that an area's true prevalence exceeds the overall prevalence (times a chosen ratio).
- Spatial clustering check: global Moran's I (permutation test) and local Getis-Ord Gi* on standardised residuals,
  with false-discovery-rate control. It needs coordinates and at least MIN_AREAS_SPATIAL areas.

Same rules as `exploration.py`
- Written results only. Nothing fitted is returned or saved, so there is nothing to register, deploy, serve or send to CNR.
- This module must not import the registry, serving, predict, webhook or integration code. A test enforces it.
- Counts that could point to a few children are suppressed (see `_small`); smoothed prevalence is still shown.
- Honest about what it cannot show: area-level patterns are not individual risk, and a flag is a statistical lead.
"""
import warnings as pywarnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

MIN_AREAS = 3
MIN_AREAS_SPATIAL = 8
MIN_CELL = 5                 # counts below this are not shown (same threshold as dataset profiles)
LOW_EXPECTED = 5.0           # an area expecting fewer cases than this is flagged as low precision
OVERDISPERSION_LIMIT = 1.5   # Pearson dispersion above this triggers the negative binomial comparison
CV_FLOOR = 0.05              # smallest between-area variation the smoothing prior will assume once differences are detected
POOLED_CV = 0.01             # with no evidence of real differences every area is pooled to the overall rate
MAX_AREAS = 1000

NOTICE = ("Aggregated area-level analysis only. Hotspots and trends describe patterns in the counts supplied. They are not "
          "diagnoses and say nothing about any individual child. An analysis can never be registered, approved, deployed or sent to CNR.")
LIMITATIONS = [
    "Results describe areas, not children. A high-prevalence area does not mean every child in it is at risk, and a low one does not mean none is.",
    "A hotspot flag means the data are unlikely under 'this area is at the overall rate'. Many areas are tested at once, so some flags will be chance. Treat them as places to look first.",
    "Trends reflect who was examined. Changes in screening coverage, outreach campaigns, reporting completeness or case definitions can look like changes in prevalence.",
    "The denominator must be the children examined (or the population at risk) for the same area and period as the cases. Mixed denominators give misleading rates.",
    "Areas are compared through their centre points. Real boundaries, travel routes and health-facility catchments are not modelled.",
]


class HotspotError(ValueError):
    """Message is safe to show to the user."""


@dataclass
class Config:
    area_col: str = "district"
    cases_col: str = "cases"
    denominator_col: str = "examined"
    period_col: str | None = None
    lat_col: str | None = None
    lon_col: str | None = None
    smoothing: str = "auto"            # auto | global | spatial
    rr_threshold: float = 1.0          # hotspot = prevalence above (overall prevalence x this ratio)
    exceedance_cutoff: float = 0.95
    k_neighbours: int = 4
    permutations: int = 999
    min_periods_trend: int = 4
    report_top: int = 20
    seed: int = 42


def _r(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _small(cases: float, denom: float) -> bool:
    """True when a count could point to a handful of children (cases, or the children without the condition)."""
    other = denom - cases
    return denom < MIN_CELL or 0 < cases < MIN_CELL or 0 < other < MIN_CELL


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------- preparation ----------

def _period_index(s: pd.Series, warnings: list[str]) -> tuple[np.ndarray, list[str], str]:
    """Returns (time in units, label per row, unit name). Missing periods are fine; time follows the real spacing."""
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s
    elif pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy(dtype=float)
        return v, [str(int(x)) if float(x).is_integer() else str(x) for x in v], "period"
    else:
        d = pd.to_datetime(s, errors="coerce", format="mixed")
        if d.isna().any():
            cats = sorted(s.astype(str).unique())
            warnings.append("Periods are text that could not be read as dates, so they were ordered alphabetically "
                            "and treated as equally spaced. Use dates, or year / month numbers, if the order matters.")
            rank = {c: i for i, c in enumerate(cats)}
            return s.astype(str).map(rank).to_numpy(dtype=float), s.astype(str).tolist(), "period"
    months = (d.dt.year * 12 + d.dt.month).to_numpy(dtype=float)
    return months, d.dt.strftime("%Y-%m").tolist(), "month"


def _prepare(df: pd.DataFrame, cfg: Config, warnings: list[str]) -> tuple[pd.DataFrame, dict]:
    need = [cfg.area_col, cfg.cases_col, cfg.denominator_col]
    opt = [c for c in (cfg.period_col, cfg.lat_col, cfg.lon_col) if c]
    if bool(cfg.lat_col) != bool(cfg.lon_col):
        raise HotspotError("Give both a latitude and a longitude column, or neither.")
    for c in need + opt:
        if c not in df.columns:
            raise HotspotError("Column not in this dataset version: " + str(c)[:80])
    if len({*need, *opt}) != len(need + opt):
        raise HotspotError("Each role (area, cases, denominator, period, latitude, longitude) needs a different column.")
    d = pd.DataFrame({"area": df[cfg.area_col].astype(object).where(df[cfg.area_col].notna(), None)})
    d["cases"] = pd.to_numeric(df[cfg.cases_col], errors="coerce")
    d["n"] = pd.to_numeric(df[cfg.denominator_col], errors="coerce")
    meta: dict = {"unit": None, "periods": None}
    if cfg.period_col:
        t, labels, unit = _period_index(df[cfg.period_col], warnings)
        d["t"], d["plabel"], meta["unit"] = t, labels, unit
    if cfg.lat_col:
        d["lat"] = pd.to_numeric(df[cfg.lat_col], errors="coerce")
        d["lon"] = pd.to_numeric(df[cfg.lon_col], errors="coerce")
    n0 = len(d)
    bad = d["area"].isna() | d["cases"].isna() | d["n"].isna()
    if cfg.period_col:
        bad |= d["t"].isna()
    if bad.sum():
        warnings.append(f"{int(bad.sum())} of {n0} rows were dropped because the area, cases, denominator or period was missing or not a number.")
    d = d[~bad].copy()
    d["area"] = d["area"].astype(str).str.strip()
    neg = (d["cases"] < 0) | (d["n"] <= 0)
    over = d["cases"] > d["n"]
    inv = neg | over
    if inv.sum():
        if inv.sum() > 0.1 * max(1, len(d)):
            raise HotspotError("Many rows are invalid: negative counts, a denominator of zero, or more cases than children examined. "
                               "Check which columns hold the cases and the denominator.")
        warnings.append(f"{int(inv.sum())} rows were dropped: negative counts, a denominator of zero, or more cases than children examined.")
        d = d[~inv].copy()
    if (d["cases"] % 1 != 0).any():
        warnings.append("Some case counts are not whole numbers and were rounded. Check that this column holds counts, not percentages.")
    d["cases"] = d["cases"].round().astype(int)
    if len(d) == 0:
        raise HotspotError("No usable rows after cleaning.")
    keys = ["area", "t"] if cfg.period_col else ["area"]
    agg: dict = {"cases": "sum", "n": "sum"}
    if cfg.period_col:
        agg["plabel"] = "first"
    if cfg.lat_col:
        agg.update(lat="mean", lon="mean")
    g = d.groupby(keys, sort=True, as_index=False).agg(agg)
    if len(g) < len(d):
        warnings.append(f"{len(d) - len(g)} rows shared the same area and period and were added together.")
    if g["area"].nunique() < MIN_AREAS:
        raise HotspotError(f"At least {MIN_AREAS} areas are needed (found {g['area'].nunique()}).")
    if g["area"].nunique() > MAX_AREAS:
        raise HotspotError(f"Too many areas ({g['area'].nunique()}). The limit is {MAX_AREAS}.")
    if cfg.period_col:
        meta["periods"] = int(g["t"].nunique())
    return g, meta


def _area_table(g: pd.DataFrame, cfg: Config, warnings: list[str]) -> pd.DataFrame:
    a = g.groupby("area", sort=True).agg(y=("cases", "sum"), n=("n", "sum"), periods=("cases", "size")).reset_index()
    if cfg.lat_col:
        by = g.groupby("area")
        lat, lon = by["lat"].mean(), by["lon"].mean()
        spread = max(float((by["lat"].max() - by["lat"].min()).max(skipna=True) or 0),
                     float((by["lon"].max() - by["lon"].min()).max(skipna=True) or 0))
        if spread > 0.5:
            warnings.append("Coordinates differ within some areas. The average per area was used as its centre point.")
        a["lat"], a["lon"] = a["area"].map(lat).to_numpy(), a["area"].map(lon).to_numpy()
        ok = a["lat"].between(-90, 90) & a["lon"].between(-180, 180)
        if not ok.all():
            warnings.append("Some areas have missing or impossible coordinates, so the spatial parts were skipped.")
            a["lat"], a["lon"] = np.nan, np.nan
    return a


# ---------- empirical-Bayes smoothing ----------

def _eb_prior(y: np.ndarray, n: np.ndarray) -> tuple[float, float, bool, float | None]:
    """Marshall (1991) method-of-moments Poisson-gamma prior.

    Returns (overall rate, relative variance (cv^2), whether areas differ more than chance, p-value of that test).
    The test is the Poisson chi-square for homogeneity across areas. Without evidence of real differences the caller
    pools every area to the overall rate instead of trusting a noisy variance estimate."""
    m = y.sum() / n.sum()
    if m <= 0 or len(y) < 2:
        return 0.0, CV_FLOOR ** 2, False, None
    exp = n * m
    chi2 = float(np.sum((y - exp) ** 2 / exp))
    p = float(stats.chi2.sf(chi2, len(y) - 1))
    r = y / n
    s2 = float(np.sum(n * (r - m) ** 2) / n.sum() - m / n.mean())
    cv2 = s2 / m ** 2
    extra = bool(p < 0.05 and cv2 > 0)
    return float(m), float(max(cv2, CV_FLOOR ** 2)), extra, p


def _posterior(y: np.ndarray, n: np.ndarray, prior_mean: np.ndarray, cv2: float, ref: float) -> dict:
    """Gamma posterior with prior mean per area and a shared coefficient of variation."""
    shape0 = 1.0 / cv2
    pm = np.maximum(prior_mean, 1e-12)
    rate0 = shape0 / pm
    shape, rate = shape0 + y, rate0 + n
    mean = shape / rate
    lo, hi = stats.gamma.ppf(0.025, shape, scale=1 / rate), stats.gamma.ppf(0.975, shape, scale=1 / rate)
    exceed = stats.gamma.sf(ref, shape, scale=1 / rate)
    return {"mean": mean, "lo": lo, "hi": hi, "exceed": exceed}


def _haversine(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = la[:, None] - la[None, :], lo[:, None] - lo[None, :]
    h = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _neighbours(lat: np.ndarray, lon: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    dist = _haversine(lat, lon)
    np.fill_diagonal(dist, np.inf)
    nb = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return nb, dist


# ---------- trend ----------

def _fit_glm(y, X, off, family, **kw):
    import statsmodels.api as sm
    return sm.GLM(y, X, family=family, offset=off).fit(**kw)


def _nb_alpha(y, X, off, mu_pois) -> tuple[float, str]:
    """Negative binomial dispersion: maximum likelihood, falling back to a moment estimate."""
    import statsmodels.api as sm
    try:
        res = sm.NegativeBinomial(y, X, offset=off).fit(disp=0, maxiter=300, method="bfgs")
        a = float(res.params[-1])
        if np.isfinite(a) and a > 1e-8:
            return a, "maximum likelihood"
    except Exception:
        pass
    a = float(max(0.0, np.sum((y - mu_pois) ** 2 - mu_pois) / np.sum(mu_pois ** 2)))
    return max(a, 1e-6), "moment estimate"


def _trend(g: pd.DataFrame, cfg: Config, unit: str, warnings: list[str]) -> dict | None:
    import statsmodels.api as sm
    periods = g["t"].nunique()
    if periods < cfg.min_periods_trend:
        warnings.append(f"Only {periods} period(s) in the data. At least {cfg.min_periods_trend} are needed for a trend, so the trend part was skipped.")
        return None
    y = g["cases"].to_numpy(dtype=float)
    n = g["n"].to_numpy(dtype=float)
    off = np.log(n)
    tc = g["t"].to_numpy(dtype=float)
    t0 = float(tc.mean())
    t = tc - t0
    codes = pd.get_dummies(g["area"], dtype=float)
    if codes.shape[1] + 1 >= len(g):
        warnings.append("Too few area-period rows for the number of areas, so the trend part was skipped.")
        return None
    X = np.column_stack([codes.to_numpy(), t])
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        try:
            pois = _fit_glm(y, X, off, sm.families.Poisson())
        except Exception:
            warnings.append("The trend model did not converge, so the trend part was skipped.")
            return None
        dof = max(1, len(y) - X.shape[1])
        disp = float(pois.pearson_chi2 / dof)
        fam_name, fam, alpha, alpha_how, aic_nb = "poisson", sm.families.Poisson(), None, None, None
        cand = {"poisson": {"aic": _r(pois.aic, 1), "dispersion": _r(disp, 3)}}
        res, scale = pois, "X2" if disp > OVERDISPERSION_LIMIT else None
        if disp > OVERDISPERSION_LIMIT:
            alpha, alpha_how = _nb_alpha(y, X, off, pois.mu)
            try:
                nb_res = _fit_glm(y, X, off, sm.families.NegativeBinomial(alpha=alpha))
                # AIC of the negative binomial from its own log-likelihood with the estimated alpha.
                ll = float(np.sum(stats.nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * nb_res.mu))))
                aic_nb = 2 * (X.shape[1] + 1) - 2 * ll
                cand["negative_binomial"] = {"aic": _r(aic_nb, 1), "alpha": _r(alpha, 4),
                                             "dispersion": _r(nb_res.pearson_chi2 / dof, 3)}
                if aic_nb < pois.aic - 2:
                    fam_name, fam, res, scale = "negative_binomial", sm.families.NegativeBinomial(alpha=alpha), nb_res, None
            except Exception:
                pass
        beta, se = float(res.params[-1]), float(res.bse[-1])
        if scale == "X2":
            se = float(_fit_glm(y, X, off, sm.families.Poisson(), scale="X2").bse[-1])
        z = beta / se if se > 0 else 0.0
        p = float(2 * stats.norm.sf(abs(z)))
        mu = res.mu

    def rr(b):
        return float(np.exp(b))

    block = {
        "model": fam_name,
        "model_choice": ("Counts vary about as much as a Poisson model expects, so Poisson regression was used." if fam_name == "poisson" and disp <= OVERDISPERSION_LIMIT
                         else "Counts vary more than Poisson allows. The negative binomial model fits clearly better and was used." if fam_name == "negative_binomial"
                         else "Counts vary more than Poisson allows, but the negative binomial model was not clearly better. "
                              "Poisson was kept and its standard errors were widened to match the observed variation."),
        "dispersion": _r(disp, 3), "candidates": cand, "alpha_method": alpha_how,
        "unit": unit, "periods": int(periods),
        "overall": {"rate_ratio_per_unit": _r(rr(beta)), "ci95": [_r(rr(beta - 1.96 * se)), _r(rr(beta + 1.96 * se))],
                    "p_value": _r(p, 4),
                    "direction": ("rising" if p < 0.05 and beta > 0 else "falling" if p < 0.05 else "no clear change")},
    }
    if unit == "month":
        block["overall"]["rate_ratio_per_12_months"] = _r(rr(12 * beta))
        block["overall"]["ci95_12_months"] = [_r(rr(12 * (beta - 1.96 * se))), _r(rr(12 * (beta + 1.96 * se)))]

    # Overall observed vs fitted prevalence by period.
    gp = g.assign(mu=mu).groupby("t", sort=True).agg(y=("cases", "sum"), n=("n", "sum"), mu=("mu", "sum"), label=("plabel", "first"))
    series = []
    for _, row in gp.iterrows():
        item = {"period": str(row["label"]), "examined": int(row["n"]), "prevalence": _r(row["y"] / row["n"]),
                "fitted": _r(row["mu"] / row["n"])}
        if _small(row["y"], row["n"]):
            item.update(examined=None, prevalence=None, suppressed=True)
        else:
            item["cases"] = int(row["y"])
        series.append(item)
    block["series"] = series

    # Area-specific trends, fitted one area at a time with the chosen family and dispersion.
    rows = []
    famobj = sm.families.NegativeBinomial(alpha=alpha) if fam_name == "negative_binomial" else sm.families.Poisson()
    for area, sub in g.groupby("area", sort=True):
        ys, ns, ts = sub["cases"].to_numpy(float), sub["n"].to_numpy(float), sub["t"].to_numpy(float) - t0
        item = {"area": area, "periods": int(len(sub))}
        if len(sub) < cfg.min_periods_trend or ys.sum() < 10 or np.ptp(ts) == 0:
            item["status"] = "insufficient data"
            rows.append(item)
            continue
        try:
            with pywarnings.catch_warnings():
                pywarnings.simplefilter("ignore")
                Xa = np.column_stack([np.ones(len(sub)), ts])
                fit = _fit_glm(ys, Xa, np.log(ns), famobj)
            b, s = float(fit.params[1]), float(fit.bse[1]) * (np.sqrt(max(1.0, disp)) if fam_name == "poisson" else 1.0)
            if not np.isfinite(b) or not np.isfinite(s) or s <= 0:
                raise ValueError
            item.update(status="fitted", beta=b, se=s, p=float(2 * stats.norm.sf(abs(b / s))))
        except Exception:
            item["status"] = "did not converge"
        rows.append(item)
    ok = [r for r in rows if r["status"] == "fitted"]
    if ok:
        q = _bh(np.array([r["p"] for r in ok]))
        for r, qq in zip(ok, q):
            r["q"] = float(qq)
        w = np.array([1 / r["se"] ** 2 for r in ok])
        b = np.array([r["beta"] for r in ok])
        bbar = float(np.sum(w * b) / w.sum())
        Q = float(np.sum(w * (b - bbar) ** 2))
        block["heterogeneity"] = {"test": "Cochran Q on the area-specific trends", "areas_tested": len(ok), "q": _r(Q, 2),
                                  "df": len(ok) - 1, "p_value": _r(float(stats.chi2.sf(Q, max(1, len(ok) - 1))), 4),
                                  "meaning": "A small p-value means areas are moving at different speeds, so one common trend hides real differences."
                                  if len(ok) > 1 else None}
    block["_area_trends"] = rows
    return block


# ---------- spatial clustering check ----------

def _spatial(a: pd.DataFrame, m: float, cv2: float, cfg: Config, warnings: list[str]) -> dict | None:
    lat, lon = a["lat"].to_numpy(float), a["lon"].to_numpy(float)
    n_areas = len(a)
    if not np.isfinite(lat).all():
        return None
    if n_areas < MIN_AREAS_SPATIAL:
        warnings.append(f"Only {n_areas} areas have coordinates. The spatial clustering check needs at least {MIN_AREAS_SPATIAL}, so it was skipped.")
        return None
    k = int(min(max(cfg.k_neighbours, 2), n_areas - 3))
    nb, _ = _neighbours(lat, lon, k)
    y, n = a["y"].to_numpy(float), a["n"].to_numpy(float)
    mu = n * m
    z = (y - mu) / np.sqrt(np.maximum(mu + cv2 * mu ** 2, 1e-12))     # standardised excess, allowing extra-Poisson variation
    zc = z - z.mean()
    denom = float(np.sum(zc ** 2))
    if denom <= 0:
        return None

    def moran(v):
        lag = v[nb].mean(axis=1)
        return float(np.sum(v * lag) / np.sum(v ** 2))

    I = moran(zc)
    rng = np.random.default_rng(cfg.seed)
    sims = np.array([moran(rng.permutation(zc)) for _ in range(cfg.permutations)])
    p = float((1 + np.sum(sims >= I)) / (cfg.permutations + 1))
    ex = -1.0 / (n_areas - 1)
    sd_sim = float(sims.std()) or 1.0
    clustered = p < 0.05 and I > 0

    # Local Gi*: mean of standardised excess over the area and its neighbours, against the mean of the same number of
    # values drawn at random from the other areas (exact sampling moments, no resampling).
    loc = []
    others_n = n_areas - 1
    obs, ez, vz = np.zeros(n_areas), np.zeros(n_areas), np.zeros(n_areas)
    for i in range(n_areas):
        rest = np.delete(zc, i)
        mu_o, var_o = float(rest.mean()), float(rest.var())
        obs[i] = (zc[i] + zc[nb[i]].sum()) / (k + 1)
        ez[i] = (zc[i] + k * mu_o) / (k + 1)
        vz[i] = k * var_o * (others_n - k) / max(others_n - 1, 1) / (k + 1) ** 2
    gz = (obs - ez) / np.sqrt(np.maximum(vz, 1e-12))
    gp = 2 * stats.norm.sf(np.abs(gz))
    gq = _bh(gp)
    for i, row in enumerate(a.itertuples()):
        lab = "high cluster" if gq[i] < 0.05 and gz[i] > 0 else "low cluster" if gq[i] < 0.05 else None
        loc.append({"area": row.area, "gi_z": _r(gz[i], 2), "p_value": _r(gp[i], 4), "q_value": _r(gq[i], 4), "cluster": lab})
    if n_areas < 15:
        warnings.append("With fewer than 15 areas the clustering check has little power. 'No clear clustering' does not rule clustering out.")
    return {"neighbours": f"{k} nearest areas by distance between centre points", "k": k,
            "global": {"statistic": "Moran's I", "i": _r(I), "expected_under_no_clustering": _r(ex),
                       "p_value": _r(p, 4), "permutations": cfg.permutations, "simulated_sd": _r(sd_sim),
                       "verdict": "clustering" if clustered else "no_clear_clustering",
                       "summary": ("Areas with higher-than-expected prevalence sit near each other more often than chance alone would explain. "
                                   "Look at the local results to see where." if clustered else
                                   "Nothing here shows high or low areas grouping together more than chance would. Individual hotspots may still matter.")},
            "local": loc,
            "note": ("Computed on standardised differences from the overall rate, using the data's own extra variation. "
                     "Local results are corrected for testing many areas (false discovery rate).")}


# ---------- entry point ----------

def run(df: pd.DataFrame, cfg: Config) -> dict:
    """Returns the written result. No fitted object leaves this function."""
    if cfg.smoothing not in ("auto", "global", "spatial"):
        raise HotspotError("Smoothing must be auto, global or spatial.")
    if not (0.5 <= cfg.rr_threshold <= 5):
        raise HotspotError("The rate ratio threshold must be between 0.5 and 5.")
    if not (0.5 < cfg.exceedance_cutoff < 1):
        raise HotspotError("The exceedance cutoff must be between 0.5 and 1.")
    warnings: list[str] = []
    g, meta = _prepare(df, cfg, warnings)
    a = _area_table(g, cfg, warnings)
    has_xy = bool(cfg.lat_col) and a["lat"].notna().all()
    if cfg.smoothing == "spatial" and not has_xy:
        raise HotspotError("Spatial smoothing needs valid latitude and longitude columns.")
    use_spatial = has_xy and cfg.smoothing in ("auto", "spatial") and len(a) >= MIN_AREAS_SPATIAL
    if cfg.smoothing == "spatial" and not use_spatial:
        raise HotspotError(f"Spatial smoothing needs at least {MIN_AREAS_SPATIAL} areas.")

    y, n = a["y"].to_numpy(float), a["n"].to_numpy(float)
    total_y, total_n = float(y.sum()), float(n.sum())
    if total_y <= 0:
        raise HotspotError("There are no cases in this data, so there is nothing to map.")
    m, cv2, extra, p_het = _eb_prior(y, n)
    ref = m * cfg.rr_threshold
    prior_cv2 = cv2 if extra else POOLED_CV ** 2
    glob = _posterior(y, n, np.full(len(a), m), prior_cv2, ref)
    if not extra:
        warnings.append("Area prevalences vary no more than chance would produce" + (f" (homogeneity test p = {p_het:.2f})" if p_het is not None else "") +
                        ", so every area was pooled to the overall prevalence and none is flagged. "
                        "This does not prove that areas are alike, only that this data cannot show differences.")
        if use_spatial:
            warnings.append("Spatial smoothing was not applied because there is no evidence of real differences between areas to smooth.")
        use_spatial = False
    spat = None
    if use_spatial:
        k = int(min(max(cfg.k_neighbours, 2), len(a) - 3))
        nb, _ = _neighbours(a["lat"].to_numpy(float), a["lon"].to_numpy(float), k)
        local_mean = np.array([(y[i] + y[nb[i]].sum()) / (n[i] + n[nb[i]].sum()) for i in range(len(a))])
        spat = _posterior(y, n, local_mean, prior_cv2, ref)
    chosen = spat if (spat is not None and cfg.smoothing in ("auto", "spatial")) else glob
    method = "spatial_eb" if chosen is spat else "global_eb"

    exp_cases = n * m
    areas = []
    for i, row in enumerate(a.itertuples()):
        small = _small(row.y, row.n)
        it = {"area": row.area, "smoothed_prevalence": _r(chosen["mean"][i]),
              "interval95": [_r(chosen["lo"][i]), _r(chosen["hi"][i])],
              "smoothed_ratio": _r(chosen["mean"][i] / m, 3), "exceedance": _r(chosen["exceed"][i], 4),
              "global_smoothed_prevalence": _r(glob["mean"][i]), "low_precision": bool(exp_cases[i] < LOW_EXPECTED),
              "periods": int(row.periods), "suppressed": bool(small)}
        if small:
            it.update(cases=None, examined=None, raw_prevalence=None)
        else:
            it.update(cases=int(row.y), examined=int(row.n), raw_prevalence=_r(row.y / row.n))
        if has_xy:
            it["centre"] = [_r(row.lat, 2), _r(row.lon, 2)]           # about 1 km: an area centre, never a household
        it["status"] = ("hotspot" if chosen["exceed"][i] >= cfg.exceedance_cutoff
                        else "watch" if chosen["exceed"][i] >= 0.8 else "not elevated")
        areas.append(it)

    trend = None
    if cfg.period_col:
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            trend = _trend(g, cfg, meta["unit"], warnings)
    if trend is not None:
        by = {r["area"]: r for r in trend.pop("_area_trends")}
        for it in areas:
            r = by.get(it["area"], {"status": "insufficient data"})
            if r["status"] != "fitted":
                it["trend"] = {"status": r["status"]}
                continue
            b, s = r["beta"], r["se"]
            sig = r["q"] < 0.05
            it["trend"] = {"status": "fitted", "rate_ratio_per_unit": _r(np.exp(b)),
                           "ci95": [_r(np.exp(b - 1.96 * s)), _r(np.exp(b + 1.96 * s))],
                           "q_value": _r(r["q"], 4), "direction": ("rising" if sig and b > 0 else "falling" if sig else "no clear change")}

    spatial = None
    if has_xy and len(a) >= 1:
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            spatial = _spatial(a, m, cv2, cfg, warnings)
    if spatial is not None:
        loc = {r["area"]: r for r in spatial["local"]}
        for it in areas:
            r = loc.get(it["area"])
            it["cluster"] = r["cluster"] if r else None
    elif cfg.lat_col is None:
        warnings.append("No coordinates were given, so the spatial smoothing and the spatial clustering check were skipped. "
                        "Add a latitude and longitude column (a centre point per area) to enable them.")

    areas.sort(key=lambda d: (-(d["smoothed_ratio"] or 0), d["area"]))
    for rank, it in enumerate(areas, start=1):
        it["rank"] = rank
    hot = [it for it in areas if it["status"] == "hotspot"]
    shortlist = [{"area": it["area"], "smoothed_prevalence": it["smoothed_prevalence"], "exceedance": it["exceedance"],
                  "rising": (it.get("trend") or {}).get("direction") == "rising",
                  "cluster": it.get("cluster")} for it in hot[:cfg.report_top]]

    if len(a) < 10:
        warnings.append(f"Only {len(a)} areas. Smoothing and hotspot calls are unreliable with so few areas.")
    if any(it["low_precision"] for it in areas):
        cnt = sum(it["low_precision"] for it in areas)
        warnings.append(f"{cnt} area(s) expect fewer than {int(LOW_EXPECTED)} cases at the overall rate. Their estimates lean heavily on the overall rate and their intervals are wide.")
    if any(it["suppressed"] for it in areas):
        warnings.append(f"Counts under {MIN_CELL} (or leaving fewer than {MIN_CELL} children in a group) are hidden for those areas. Smoothed prevalence is still shown.")
    if hot:
        warnings.append("Every area is checked against the overall rate, so a few flagged areas can be chance. Use the exceedance probability and the trend together, not the flag alone.")

    return {"kind": "hotspot_analysis", "deployable": False, "notice": NOTICE,
            "data": {"areas": int(len(a)), "rows": int(len(g)), "periods": meta["periods"], "total_examined": int(total_n),
                     "total_cases": int(total_y) if not _small(total_y, total_n) else None,
                     "has_coordinates": bool(has_xy)},
            "settings": {"area": cfg.area_col, "cases": cfg.cases_col, "denominator": cfg.denominator_col, "period": cfg.period_col,
                         "latitude": cfg.lat_col, "longitude": cfg.lon_col, "rr_threshold": cfg.rr_threshold,
                         "exceedance_cutoff": cfg.exceedance_cutoff, "smoothing_requested": cfg.smoothing},
            "overall": {"prevalence": _r(m), "extra_variation_between_areas": bool(extra),
                        "between_area_cv": _r(np.sqrt(cv2), 3) if extra else None,
                        "homogeneity_p_value": _r(p_het, 4)},
            "smoothing": {"method": method,
                          "description": ("Each area is pulled towards its nearest neighbours' rate, more strongly when it has few children examined."
                                          if method == "spatial_eb" else
                                          "Each area is pulled towards the overall rate, more strongly when it has few children examined."
                                          if extra else "No evidence that areas differ, so every area is pooled to the overall prevalence."),
                          "hotspot_rule": f"Hotspot: at least {cfg.exceedance_cutoff:.0%} probability that prevalence exceeds {cfg.rr_threshold:g} x the overall prevalence. "
                                          "Watch: at least 80%."},
            "hotspots": shortlist, "areas": areas, "trend": trend, "spatial": spatial,
            "warnings": warnings, "limitations": LIMITATIONS}
