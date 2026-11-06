"""XGBoost, neural network (MLP) and logistic regression training and evaluation.

Rules
- Every preprocessing step lives inside the sklearn Pipeline, so it is fitted on training data only.
- The hold-out test set is split off first and touched once, after cross-validation on the training part.
- Results are always reported next to a trivial baseline, with an uncertainty range.
- Loud warnings for the usual ways a medical model looks better than it is (leakage, tiny test sets).
- A neural network or logistic regression run can be compared against XGBoost on the identical split, folds and sample weights.
- Logistic regression is the interpretable option: its coefficients are reported, and its probabilities are calibrated
  inside the pipeline (cross-fitted on the training part only) and checked on the hold-out set.
"""
import re
import warnings as pywarnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

MIN_ROWS, MIN_PER_CLASS, MAX_CLASSES = 50, 10, 12
ALGORITHMS = ("xgboost", "mlp", "logistic")
CALIBRATIONS = ("sigmoid", "isotonic", "none")
MAX_COEF_TERMS = 40            # coefficient rows returned per outcome (largest first)
ISOTONIC_MIN_TRAIN = 1000      # isotonic calibration overfits below roughly this many training rows
MLP_SMALL_TRAIN = 500          # below this many training rows a neural network is unstable

LIMITATIONS = [
    "Scores come from an internal hold-out split of one dataset. They are not external validation.",
    "A model trained on data from one site or period may perform worse elsewhere.",
    "Outputs are risk estimates to support, not replace, clinical judgement.",
]
MLP_LIMITATIONS = [
    "Neural networks usually need more data than gradient-boosted trees. On small tables the score can move a lot with the random seed.",
]
LR_LIMITATIONS = [
    "Logistic regression assumes each feature moves the log-odds in a straight line and finds no interactions on its own. "
    "A tree model can score higher when relationships bend or combine.",
    "Coefficients describe association in this dataset, not cause. When features are correlated, the penalty shares credit between them "
    "in a way that can change with the data, so read single coefficients with care.",
    "Calibrated probabilities were checked on one hold-out set. Calibration does not carry over to a different site or period.",
]
COMPARISON_LIMITATION = (
    "The comparison uses one hold-out split. Choosing the higher-scoring model on that same split makes its score look a little better "
    "than it will be on new data, so treat the winner as a lead to confirm, not a result."
)


class TrainingError(ValueError):
    """Message is safe to show to the user."""


@dataclass
class Config:
    target: str
    task: str                                  # classification | regression
    features: list[str] | None = None
    exclude: list[str] = field(default_factory=list)
    test_size: float = 0.2
    seed: int = 42
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    balance_classes: bool = True
    algorithm: str = "xgboost"                 # xgboost | mlp | logistic
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32])   # MLP only
    alpha: float = 1e-3                        # MLP only: L2 penalty
    max_iter: int = 500                        # MLP and logistic: upper limit on training iterations
    compare_xgboost: bool = True               # MLP and logistic: also fit XGBoost on the same split
    logistic_c: float = 1.0                    # logistic only: inverse L2 strength (smaller = stronger penalty)
    leakage_tokens: list[str] = field(default_factory=list)      # column-name words that may not be features (see catalog.py)
    leakage_substrings: list[str] = field(default_factory=list)  # same, for run-together names such as "bloodpressure"
    warn_substrings: list[str] = field(default_factory=list)     # label-definition features that are allowed but warned about
    calibration: str = "none"                  # logistic only: none | sigmoid | isotonic (recalibration is opt-in, see _calibration_verdict)


def _is_leaky(col: str, cfg: Config) -> bool:
    low = col.lower()
    words = {w for w in re.split(r"[^a-z0-9]+", low) if w}
    return bool(words & set(cfg.leakage_tokens)) or any(sub in low.replace("_", "").replace(" ", "") for sub in cfg.leakage_substrings)


def _prepare(df: pd.DataFrame, cfg: Config, warnings: list[str]):
    if cfg.target not in df.columns:
        raise TrainingError("The target column is not in this dataset version.")
    n0 = len(df)
    df = df[df[cfg.target].notna()].copy()
    if len(df) < n0:
        warnings.append(f"{n0 - len(df)} rows without a target value were excluded.")
    if len(df) < MIN_ROWS:
        raise TrainingError(f"Too few labelled rows ({len(df)}). At least {MIN_ROWS} are needed.")
    excl = set(cfg.exclude)
    cand = cfg.features or [c for c in df.columns if c != cfg.target and c not in excl]
    missing = [c for c in cand if c not in df.columns]
    if missing:
        raise TrainingError("Unknown feature column: " + missing[0][:80])
    if cfg.target in cand:
        raise TrainingError("The target cannot also be a feature.")
    banned = [c for c in cand if _is_leaky(c, cfg)]
    if banned:
        raise TrainingError(f"'{banned[0][:60]}' defines or reveals this model's label and cannot be a feature. "
                            "Exclude it (and any other blood pressure or diagnosis columns) and run again.")
    overlap = [c for c in cand if any(sub in c.lower().replace("_", "").replace(" ", "") for sub in cfg.warn_substrings)]
    if overlap:
        warnings.append(f"'{overlap[0][:60]}' is part of this model's label definition. It is allowed because it is the cheap measurement the "
                        "screen relies on, but it gives the model a head start: scores overstate how well the other features predict the label. "
                        "Train the same model with it excluded and compare before trusting the result.")
    num, cat, n = [], [], len(df)
    for c in cand:
        s = df[c]
        nun = s.nunique(dropna=True)
        if nun <= 1:
            warnings.append(f"'{c}' was skipped because it has a single value.")
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num.append(c)
        elif nun > max(50, 0.5 * n):
            warnings.append(f"'{c}' was skipped: too many distinct values (looks like an identifier or free text).")
        else:
            cat.append(c)
    if not num and not cat:
        raise TrainingError("No usable feature columns are left.")
    for c in cat:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        df[c] = df[c].map(lambda v: v if v is np.nan or pd.isna(v) else str(v))
    X = df[num + cat]
    if cfg.task == "classification":
        yr = df[cfg.target]
        if pd.api.types.is_numeric_dtype(yr) and yr.nunique() > 20:
            raise TrainingError("The target looks continuous. Choose a regression model or a categorical target.")
        ys = yr.astype(str)
        counts = ys.value_counts()
        if len(counts) < 2:
            raise TrainingError("The target has only one class.")
        if len(counts) > MAX_CLASSES:
            raise TrainingError(f"The target has {len(counts)} classes (limit {MAX_CLASSES}).")
        small = counts[counts < MIN_PER_CLASS]
        if not small.empty:
            raise TrainingError(f"Each class needs at least {MIN_PER_CLASS} examples. Too small: "
                                + ", ".join(f"{k} ({v})" for k, v in small.items())
                                + ". Merge classes or collect more data.")
        le = LabelEncoder()
        y = le.fit_transform(ys)
        return X, y, num, cat, [str(c) for c in le.classes_], {str(k): int(v) for k, v in counts.items()}
    y = pd.to_numeric(df[cfg.target], errors="coerce")
    if y.isna().any():
        raise TrainingError("The target has non-numeric values; a regression model needs a numeric target.")
    return X, y.to_numpy(dtype=float), num, cat, None, None


def _cat_steps() -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])


def _pipeline(num, cat, cfg: Config, cal_folds: int | None = None) -> Pipeline:
    if cfg.algorithm == "mlp":
        return _mlp_pipeline(num, cat, cfg)
    if cfg.algorithm == "logistic":
        return _lr_pipeline(num, cat, cfg, cal_folds)
    return _xgb_pipeline(num, cat, cfg)


def _lr_pipeline(num, cat, cfg: Config, cal_folds: int | None, calibrate: bool = True) -> Pipeline:
    # Median imputation and scaling are fitted inside the pipeline (training folds only). Scaling puts every numeric
    # coefficient on the same "per standard deviation" footing. Step names match the other pipelines so serving works unchanged.
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", _cat_steps(), cat),
    ])
    base = LogisticRegression(C=cfg.logistic_c, max_iter=cfg.max_iter, random_state=cfg.seed)
    if calibrate and cfg.calibration != "none":
        # ensemble=False: the calibrator is fitted on cross-fitted scores and the logistic model itself is refitted on all
        # training rows, so its coefficients stay a single readable set.
        base = CalibratedClassifierCV(base, method=cfg.calibration, ensemble=False,
                                      cv=StratifiedKFold(cal_folds or 3, shuffle=True, random_state=cfg.seed))
    return Pipeline([("pre", pre), ("clf", base)])


def _mlp_pipeline(num, cat, cfg: Config) -> Pipeline:
    # A neural network cannot take missing values or unscaled inputs, so the imputer and scaler are fitted
    # inside the pipeline (training folds only). Step names match the XGBoost pipeline so serving works unchanged.
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", _cat_steps(), cat),
    ])
    kw = dict(hidden_layer_sizes=tuple(int(n) for n in cfg.hidden_layers), alpha=cfg.alpha, max_iter=cfg.max_iter,
              early_stopping=False, random_state=cfg.seed)
    if cfg.task == "classification":
        model = MLPClassifier(**kw)
    else:
        model = TransformedTargetRegressor(regressor=MLPRegressor(**kw), transformer=StandardScaler())
    return Pipeline([("pre", pre), ("clf", model)])


def _xgb_pipeline(num, cat, cfg: Config) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num),  # XGBoost handles missing numbers natively
        ("cat", _cat_steps(), cat),
    ])
    kw = dict(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, learning_rate=cfg.learning_rate,
              subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2,
              random_state=cfg.seed, verbosity=0, importance_type="gain")
    model = XGBClassifier(**kw) if cfg.task == "classification" else XGBRegressor(**kw)
    return Pipeline([("pre", pre), ("clf", model)])


def _split_name(name: str, cat: list[str]) -> tuple[str, str | None]:
    """'num__age' -> ('age', None); 'cat__district_Zomba' -> ('district', 'Zomba')."""
    if name.startswith("num__"):
        return name[5:], None
    body = name[5:]
    f = next((c for c in sorted(cat, key=len, reverse=True) if body.startswith(c + "_")), body)
    return f, body[len(f) + 1:] if body.startswith(f + "_") else body


def _agg_importance(names, values, cat: list[str]) -> list[dict]:
    agg: dict[str, float] = {}
    for name, v in zip(names, values):
        f = _split_name(name, cat)[0]
        agg[f] = agg.get(f, 0.0) + float(v)
    tot = sum(agg.values()) or 1.0
    return [{"feature": f, "share": round(v / tot, 4)} for f, v in sorted(agg.items(), key=lambda kv: -kv[1])[:20]]


def _importance(pipe: Pipeline, cat: list[str]) -> list[dict]:
    return _agg_importance(pipe.named_steps["pre"].get_feature_names_out(), pipe.named_steps["clf"].feature_importances_, cat)


def _lr_base(pipe: Pipeline) -> LogisticRegression:
    """The fitted logistic model itself, whether or not it sits inside a calibrator."""
    m = pipe.named_steps["clf"]
    return m.calibrated_classifiers_[0].estimator if hasattr(m, "calibrated_classifiers_") else m


def _lr_importance(pipe: Pipeline, cat: list[str]) -> list[dict]:
    """Size of the standardised coefficients (averaged over classes), summed per original column."""
    coef = np.abs(_lr_base(pipe).coef_).mean(axis=0)
    return _agg_importance(pipe.named_steps["pre"].get_feature_names_out(), coef, cat)


def _coefficients(pipe: Pipeline, num: list[str], cat: list[str], classes: list[str], Xtr: pd.DataFrame) -> dict:
    """Readable coefficients. Numeric columns are standardised, so a coefficient is the change in log-odds for one
    standard deviation. Category levels seen in fewer than MIN_PER_CLASS training rows are left out, so a rare level
    (a small village, say) is never named in a result."""
    pre, base = pipe.named_steps["pre"], _lr_base(pipe)
    names = pre.get_feature_names_out()
    try:
        sc = pre.named_transformers_["num"].named_steps["sc"]
        sds = dict(zip(num, (float(x) for x in sc.scale_)))
    except Exception:
        sds = {}
    counts = {c: Xtr[c].astype(object).where(Xtr[c].notna(), "missing").astype(str).value_counts().to_dict() for c in cat}
    binary = len(classes) == 2
    sets = [(classes[1], base.coef_[0], float(base.intercept_[0]))] if binary else \
           [(classes[i], base.coef_[i], float(base.intercept_[i])) for i in range(len(classes))]
    outcomes, hidden = [], 0
    for cls, vec, icpt in sets:
        terms = []
        for j, name in enumerate(names):
            f, level = _split_name(name, cat)
            c = float(vec[j])
            t = {"feature": f, "kind": "number" if level is None else "category", "coefficient": round(c, 4)}
            if level is None:
                sd = sds.get(f)
                t["sd"] = round(sd, 4) if sd else None
                if binary and sd:
                    t["odds_ratio_per_unit"] = round(float(np.exp(np.clip(c / sd, -30, 30))), 4)
            else:
                if counts.get(f, {}).get(level, 0) < MIN_PER_CLASS:
                    hidden += 1
                    continue
                t["level"] = "(missing)" if level == "missing" else level
            if binary:
                t["odds_ratio"] = round(float(np.exp(np.clip(c, -30, 30))), 4)
            terms.append(t)
        terms.sort(key=lambda t: -abs(t["coefficient"]))
        outcomes.append({"class": cls, "intercept": round(icpt, 4), "n_terms": len(terms), "terms": terms[:MAX_COEF_TERMS]})
    guide = ("Numeric features are standardised: a coefficient is the change in log-odds of the outcome for a one standard-deviation rise "
             "(odds_ratio is exp of it; odds_ratio_per_unit is per one unit of the original measurement). Category coefficients are "
             "relative to the other levels of the same column. Positive pushes towards the outcome, negative away from it.")
    if not binary:
        guide = ("Each class has its own set of coefficients, on the log-odds scale: positive pushes a case towards that class, negative "
                 "away from it, relative to the other classes. Numeric features are standardised (per one standard deviation).")
    return {"outcome_scope": "positive class: " + classes[1] if binary else "one set per class", "binary": binary,
            "outcomes": outcomes, "rare_levels_hidden": hidden, "reading_guide": guide,
            "note": "Coefficients describe the underlying logistic model. Calibration then rescales its probabilities slightly, so a "
                    "calibrated probability can differ a little from what the coefficients alone imply."}


def _calibration_verdict(before: dict, after: dict) -> str:
    """Plain wording for what recalibration did on the hold-out set. A logistic model is fitted by minimising log loss, so its
    probabilities are often already well calibrated and recalibration can add noise instead of removing bias."""
    d_brier, d_ece = after["brier"] - before["brier"], after["ece"] - before["ece"]
    if d_brier < -0.005 and d_ece <= 0.005:
        return "Recalibration lowered the Brier score on the hold-out set, so the probabilities are closer to the observed frequencies."
    if d_brier > 0.005:
        return ("Recalibration made the Brier score worse on the hold-out set. The plain logistic probabilities are the better ones here; "
                "run again with calibration off.")
    return ("Recalibration did not clearly change the Brier score on the hold-out set. With no clear gain, the plain logistic "
            "probabilities are the simpler choice.")


def _calibration_report(yte, proba, k: int, bins: int = 5) -> dict:
    """Brier score, log loss and a top-label reliability table (expected calibration error). Lower is better."""
    onehot = np.eye(k)[yte]
    brier = float(np.mean((proba[:, 1] - yte) ** 2)) if k == 2 else float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    conf, hit = proba.max(axis=1), (proba.argmax(axis=1) == yte)
    edges, rows, ece = np.linspace(0, 1, bins + 1), [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            gap = abs(float(hit[m].mean()) - float(conf[m].mean()))
            ece += m.mean() * gap
            rows.append({"range": [round(float(lo), 2), round(float(hi), 2)], "n": int(m.sum()),
                         "mean_confidence": round(float(conf[m].mean()), 4), "observed_accuracy": round(float(hit[m].mean()), 4)})
    return {"brier": round(brier, 4), "log_loss": round(float(log_loss(yte, proba, labels=list(range(k)))), 4),
            "ece": round(float(ece), 4), "reliability": rows}


def _perm_importance(pipe: Pipeline, Xte: pd.DataFrame, yte, clf: bool, seed: int) -> list[dict]:
    """Neural networks have no built-in importance, so each original column is shuffled in the hold-out set and the
    drop in score is measured. Columns are shuffled before preprocessing, so one-hot groups need no aggregation."""
    r = permutation_importance(pipe, Xte, yte, scoring="f1_macro" if clf else "r2", n_repeats=5, random_state=seed, n_jobs=1)
    vals = np.clip(r.importances_mean, 0, None)
    tot = float(vals.sum())
    rows = [{"feature": str(c), "share": round(float(v) / tot, 4) if tot > 0 else 0.0} for c, v in zip(Xte.columns, vals)]
    return sorted(rows, key=lambda d: -d["share"])[:20]


def _n_iter(pipe: Pipeline) -> int:
    m = pipe.named_steps["clf"]
    return int(getattr(getattr(m, "regressor_", m), "n_iter_", 0))


def _ms(vals) -> tuple[float, float]:
    return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)


CLF_KEYS = ("macro_f1", "balanced_accuracy", "accuracy", "roc_auc")
REG_KEYS = ("r2", "rmse", "mae")


def _compare_with_xgboost(cfg: Config, num, cat, Xtr, ytr, Xte, yte, w, splitter, pred, test: dict, cvr: dict, k: int | None) -> dict:
    """Fit XGBoost (default settings, not tuned) on the same training rows, folds, weights and hold-out rows as the
    selected model, then compare the two with a paired bootstrap so the gap comes with an uncertainty range."""
    clf = cfg.task == "classification"
    ref_cfg = Config(target=cfg.target, task=cfg.task, seed=cfg.seed, balance_classes=cfg.balance_classes)
    ref = _xgb_pipeline(num, cat, ref_cfg)
    fp = {"clf__sample_weight": w} if w is not None else {}
    boot_rng = np.random.default_rng(cfg.seed + 1)
    diffs = []
    if clf:
        cv = cross_validate(ref, Xtr, ytr, cv=splitter, scoring={"macro_f1": "f1_macro"}, params=fp, error_score="raise")
        ref.fit(Xtr, ytr, **fp)
        rpred, rproba = ref.predict(Xte), ref.predict_proba(Xte)
        labels = list(range(k))
        try:
            auc = float(roc_auc_score(yte, rproba[:, 1]) if k == 2 else
                        roc_auc_score(yte, rproba, multi_class="ovr", average="macro", labels=labels))
        except ValueError:
            auc = None
        f1 = float(precision_recall_fscore_support(yte, rpred, labels=labels, zero_division=0)[2].mean())
        rtest = {"macro_f1": round(f1, 4), "balanced_accuracy": round(float(balanced_accuracy_score(yte, rpred)), 4),
                 "accuracy": round(float((rpred == yte).mean()), 4), "roc_auc": None if auc is None else round(auc, 4)}
        metric, keys = "macro_f1", CLF_KEYS
        a, b = test["macro_f1"], rtest["macro_f1"]
        for _ in range(300):
            i = boot_rng.integers(0, len(yte), len(yte))
            labs = np.unique(yte[i])
            diffs.append(f1_score(yte[i], pred[i], average="macro", labels=labs, zero_division=0)
                         - f1_score(yte[i], rpred[i], average="macro", labels=labs, zero_division=0))
        cv_ref = _ms(cv["test_macro_f1"])
        cv_sel = (cvr["macro_f1_mean"], cvr["macro_f1_sd"])
    else:
        cv = cross_validate(ref, Xtr, ytr, cv=splitter, scoring={"r2": "r2"}, error_score="raise")
        ref.fit(Xtr, ytr)
        rpred = ref.predict(Xte)
        rtest = {"r2": round(float(r2_score(yte, rpred)), 4), "rmse": round(float(np.sqrt(mean_squared_error(yte, rpred))), 4),
                 "mae": round(float(mean_absolute_error(yte, rpred)), 4)}
        metric, keys = "r2", REG_KEYS
        a, b = test["r2"], rtest["r2"]
        for _ in range(300):
            i = boot_rng.integers(0, len(yte), len(yte))
            diffs.append(float(r2_score(yte[i], pred[i]) - r2_score(yte[i], rpred[i])))
        cv_ref = _ms(cv["test_r2"])
        cv_sel = (cvr["r2_mean"], cvr["r2_sd"])
    lo, hi = round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)
    name = {"macro_f1": "macro F1", "r2": "R2"}[metric]
    algo = cfg.algorithm
    label = {"mlp": "neural network", "logistic": "logistic regression"}[algo]
    if lo > 0:
        verdict, text = f"{algo}_better", f"The {label} scored higher than XGBoost on {name}, and the 95% range of the gap stays above zero."
    elif hi < 0:
        verdict, text = "xgboost_better", f"XGBoost scored higher than the {label} on {name}, and the 95% range of the gap stays below zero."
    else:
        verdict, text = "no_clear_difference", (f"No clear difference in {name}: the 95% range of the gap includes zero, "
                                                 "so this test set cannot separate the two models.")
    if algo == "logistic" and verdict != "xgboost_better":
        text += " The simpler model is not clearly worse here, and its readable coefficients make it the easier one to explain to health workers."
    elif algo == "logistic":
        text += " The heavier model may be worth its lost explainability, but confirm on new data before relying on the gap."
    return {"reference": "xgboost", "metric": metric, "same_split": True,
            "reference_settings": {"n_estimators": ref_cfg.n_estimators, "max_depth": ref_cfg.max_depth,
                                   "learning_rate": ref_cfg.learning_rate, "tuned": False},
            "selected": {"algorithm": algo, "test": {x: test.get(x) for x in keys}, "cv_mean": cv_sel[0], "cv_sd": cv_sel[1]},
            "xgboost": {"algorithm": "xgboost", "test": rtest, "cv_mean": cv_ref[0], "cv_sd": cv_ref[1]},
            "difference": round(float(a - b), 4), "difference_ci95": [lo, hi], "verdict": verdict, "summary": text,
            "caution": COMPARISON_LIMITATION}


def run(df: pd.DataFrame, cfg: Config) -> tuple[dict, dict]:
    """Returns (result for the experiment record, bundle to persist)."""
    warnings: list[str] = []
    if cfg.algorithm not in ALGORITHMS:
        raise TrainingError("Unknown algorithm. Choose xgboost, mlp or logistic.")
    if cfg.algorithm == "logistic" and cfg.task != "classification":
        raise TrainingError("Logistic regression is a classification method. Choose XGBoost or a neural network for a numeric target.")
    if cfg.calibration not in CALIBRATIONS:
        raise TrainingError("Unknown calibration method. Choose sigmoid, isotonic or none.")
    X, y, num, cat, classes, class_counts = _prepare(df, cfg, warnings)
    clf = cfg.task == "classification"
    mlp, lr = cfg.algorithm == "mlp", cfg.algorithm == "logistic"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed,
                                          stratify=y if clf else None)
    w = compute_sample_weight("balanced", ytr) if (clf and cfg.balance_classes) else None
    fit_params = {"clf__sample_weight": w} if w is not None else {}
    pipe = _pipeline(num, cat, cfg, max(2, min(5, int(np.bincount(ytr).min()))) if clf else None)
    rng = np.random.default_rng(cfg.seed)

    if clf:
        k = len(classes)
        folds = max(2, min(5, int(np.bincount(ytr).min())))
        splitter = StratifiedKFold(folds, shuffle=True, random_state=cfg.seed)
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore", ConvergenceWarning)
            cv = cross_validate(pipe, Xtr, ytr, cv=splitter,
                                scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"},
                                params=fit_params, error_score="raise")
            pipe.fit(Xtr, ytr, **fit_params)
        pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)
        labels = list(range(k))
        p, r, f, s = precision_recall_fscore_support(yte, pred, labels=labels, zero_division=0)
        try:
            auc = float(roc_auc_score(yte, proba[:, 1]) if k == 2 else
                        roc_auc_score(yte, proba, multi_class="ovr", average="macro", labels=labels))
        except ValueError:
            auc = None
        main = float(f.mean())
        base_pred = np.full(len(yte), np.bincount(ytr).argmax())
        bf = precision_recall_fscore_support(yte, base_pred, labels=labels, zero_division=0)[2]
        # A second, fairer baseline: guessing in proportion to class frequency. For balanced classes the
        # "most common class" baseline is unrealistically weak (macro F1 about 0.33 for two classes).
        prior = np.bincount(ytr, minlength=k) / len(ytr)
        chance = float(np.mean([precision_recall_fscore_support(yte, rng.choice(k, size=len(yte), p=prior),
                                                                labels=labels, zero_division=0)[2].mean() for _ in range(200)]))
        boots = []
        for _ in range(300):
            i = rng.integers(0, len(yte), len(yte))
            boots.append(f1_score(yte[i], pred[i], average="macro", labels=np.unique(yte[i]), zero_division=0))
        test = {"accuracy": round(float((pred == yte).mean()), 4), "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
                "macro_f1": round(main, 4), "macro_precision": round(float(p.mean()), 4), "macro_recall": round(float(r.mean()), 4),
                "roc_auc": None if auc is None else round(auc, 4),
                "macro_f1_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
        base_f1 = max(float(bf.mean()), chance)
        baseline = {"strategy": "the better of: always the most common class, or guessing by class frequency",
                    "accuracy": round(float((base_pred == yte).mean()), 4), "macro_f1": round(base_f1, 4),
                    "majority_macro_f1": round(float(bf.mean()), 4), "chance_macro_f1": round(chance, 4),
                    "balanced_accuracy": round(1 / k, 4)}
        cvr = {"folds": folds}
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            cvr[key + "_mean"], cvr[key + "_sd"] = _ms(cv["test_" + key])
        detail = {"classes": classes, "class_counts": class_counts,
                  "per_class": {classes[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                                             "f1": round(float(f[i]), 4), "support": int(s[i])} for i in labels},
                  "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(yte, pred, labels=labels).tolist()}}
        headline, base_headline, cv_headline = main, baseline["macro_f1"], cvr["macro_f1_mean"]
    else:
        folds = 5
        splitter = KFold(folds, shuffle=True, random_state=cfg.seed)
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore", ConvergenceWarning)
            cv = cross_validate(pipe, Xtr, ytr, cv=splitter,
                                scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
                                error_score="raise")
            pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        boots = []
        for _ in range(300):
            i = rng.integers(0, len(yte), len(yte))
            boots.append(float(np.sqrt(mean_squared_error(yte[i], pred[i]))))
        bpred = np.full(len(yte), ytr.mean())
        test = {"r2": round(float(r2_score(yte, pred)), 4), "rmse": round(rmse, 4),
                "mae": round(float(mean_absolute_error(yte, pred)), 4),
                "rmse_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
        baseline = {"strategy": "always predict the training mean", "r2": round(float(r2_score(yte, bpred)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(yte, bpred))), 4),
                    "mae": round(float(mean_absolute_error(yte, bpred)), 4)}
        cvr = {"folds": folds, "r2_mean": _ms(cv["test_r2"])[0], "r2_sd": _ms(cv["test_r2"])[1],
               "rmse_mean": _ms(-cv["test_rmse"])[0], "rmse_sd": _ms(-cv["test_rmse"])[1],
               "mae_mean": _ms(-cv["test_mae"])[0], "mae_sd": _ms(-cv["test_mae"])[1]}
        detail = {}
        headline, base_headline, cv_headline = test["r2"], baseline["r2"], cvr["r2_mean"]

    importance = _perm_importance(pipe, Xte, yte, clf, cfg.seed) if mlp else _lr_importance(pipe, cat) if lr else _importance(pipe, cat)
    comparison = (_compare_with_xgboost(cfg, num, cat, Xtr, ytr, Xte, yte, w, splitter, pred, test, cvr, k if clf else None)
                  if (mlp or lr) and cfg.compare_xgboost else None)
    calibration = coefficients = None
    if lr:
        iters = int(np.max(_lr_base(pipe).n_iter_))
        coefficients = _coefficients(pipe, num, cat, classes, Xtr)
        calibration = {"method": cfg.calibration, "n_test": int(len(yte)), "after": _calibration_report(yte, proba, k),
                       "brier_note": "For two classes the Brier score is the usual one (squared error of the probability of the second "
                                     "class); for more classes it is summed over classes. Lower is better; 0 is perfect."}
        if cfg.calibration != "none":
            with pywarnings.catch_warnings():
                pywarnings.simplefilter("ignore", ConvergenceWarning)
                raw = _lr_pipeline(num, cat, cfg, None, calibrate=False).fit(Xtr, ytr, **fit_params)
            calibration["before"] = _calibration_report(yte, raw.predict_proba(Xte), k)
            calibration["summary"] = _calibration_verdict(calibration["before"], calibration["after"])
        else:
            calibration["summary"] = ("No recalibration step was applied: these are the logistic model's own probabilities. "
                                      "The reliability table shows how closely they match the observed frequencies on the hold-out set.")
        if len(yte) < 200:
            warnings.append(f"The hold-out set has only {len(yte)} rows, so the calibration check is rough. Treat probabilities as approximate.")
    n_all, n_te = len(X), len(Xte)
    if lr:
        if iters >= cfg.max_iter:
            warnings.append(f"Logistic regression used all {cfg.max_iter} iterations without settling, so the coefficients may not be final. "
                            "Raise the iteration limit or increase the penalty (lower C).")
        if cfg.balance_classes:
            warnings.append("Classes were balanced during training, so the probabilities describe a population with equal-sized classes, "
                            "not the real mix of cases. For probabilities that match real frequencies, turn off class balancing.")
        if cfg.calibration == "isotonic" and len(Xtr) < ISOTONIC_MIN_TRAIN:
            warnings.append(f"Isotonic calibration with only {len(Xtr)} training rows can overfit. Sigmoid calibration is steadier on small tables.")
    if mlp:
        iters = _n_iter(pipe)
        if iters >= cfg.max_iter:
            warnings.append(f"The neural network used all {cfg.max_iter} training iterations without settling, so it may be under-trained. "
                            "Raise the iteration limit or simplify the network.")
        if len(Xtr) < MLP_SMALL_TRAIN:
            warnings.append(f"Only {len(Xtr)} rows were used for training. Neural networks are unstable at this size; "
                            "scores can change a lot with the random seed.")
    if n_all < 500:
        warnings.append(f"Small dataset ({n_all} rows). Treat these results as a check that the pipeline works, not as evidence of performance.")
    if n_te < 50:
        warnings.append(f"The test set has only {n_te} rows, so scores are unstable. See the confidence range.")
    if headline >= 0.97 or (importance and importance[0]["share"] >= 0.7):
        warnings.append("Suspiciously strong result" + (f" (one feature, '{importance[0]['feature']}', carries {importance[0]['share']:.0%} of the model)" if importance else "")
                        + ". Check for features that are derived from the label or measured after it, such as the measurements a nutrition status is calculated from. Exclude them and retrain.")
    ci_low = test["macro_f1_ci95"][0] if clf else None
    if headline <= base_headline + 0.02 or (ci_low is not None and ci_low <= base_headline):
        warnings.append("The model does not clearly beat the baseline: the baseline score sits inside the 95% range or above the result. "
                        "It has not shown that it learned a useful pattern from these features.")
    if abs(headline - cv_headline) > 0.15:
        warnings.append("Cross-validation and test scores differ a lot, a sign of instability on this amount of data.")

    limits = LIMITATIONS + (MLP_LIMITATIONS if mlp else []) + (LR_LIMITATIONS if lr else []) + ([COMPARISON_LIMITATION] if comparison else [])
    result = {"task": cfg.task, "algorithm": cfg.algorithm, "n_rows": int(n_all), "n_train": int(len(Xtr)), "n_test": int(n_te),
              "features": {"numeric": num, "categorical": cat}, "baseline": baseline, "test": test, "cv": cvr,
              **detail, "feature_importance": importance,
              "importance_method": "permutation on the hold-out set" if mlp else "absolute standardised coefficient" if lr else "gain",
              "warnings": warnings, "limitations": limits}
    if mlp:
        result["model_settings"] = {"hidden_layers": [int(n) for n in cfg.hidden_layers], "alpha": cfg.alpha,
                                    "max_iter": cfg.max_iter, "iterations_used": iters}
    if lr:
        result["model_settings"] = {"C": cfg.logistic_c, "penalty": "l2", "calibration": cfg.calibration,
                                    "max_iter": cfg.max_iter, "iterations_used": iters}
        result["coefficients"], result["calibration"] = coefficients, calibration
    if comparison:
        result["comparison"] = comparison
    bundle = {"pipeline": pipe, "task": cfg.task, "target": cfg.target, "classes": classes,
              "numeric": num, "categorical": cat}
    return result, bundle
