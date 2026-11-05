"""XGBoost and neural network (MLP) training and evaluation.

Rules
- Every preprocessing step lives inside the sklearn Pipeline, so it is fitted on training data only.
- The hold-out test set is split off first and touched once, after cross-validation on the training part.
- Results are always reported next to a trivial baseline, with an uncertainty range.
- Loud warnings for the usual ways a medical model looks better than it is (leakage, tiny test sets).
- A neural network run can be compared against XGBoost on the identical split, folds and sample weights.
"""
import warnings as pywarnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

MIN_ROWS, MIN_PER_CLASS, MAX_CLASSES = 50, 10, 12
ALGORITHMS = ("xgboost", "mlp")
MLP_SMALL_TRAIN = 500          # below this many training rows a neural network is unstable

LIMITATIONS = [
    "Scores come from an internal hold-out split of one dataset. They are not external validation.",
    "A model trained on data from one site or period may perform worse elsewhere.",
    "Outputs are risk estimates to support, not replace, clinical judgement.",
]
MLP_LIMITATIONS = [
    "Neural networks usually need more data than gradient-boosted trees. On small tables the score can move a lot with the random seed.",
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
    algorithm: str = "xgboost"                 # xgboost | mlp
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32])   # MLP only
    alpha: float = 1e-3                        # MLP only: L2 penalty
    max_iter: int = 500                        # MLP only: upper limit on training epochs
    compare_xgboost: bool = True               # MLP only: also fit XGBoost on the same split


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


def _pipeline(num, cat, cfg: Config) -> Pipeline:
    return _mlp_pipeline(num, cat, cfg) if cfg.algorithm == "mlp" else _xgb_pipeline(num, cat, cfg)


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


def _importance(pipe: Pipeline, cat: list[str]) -> list[dict]:
    names = pipe.named_steps["pre"].get_feature_names_out()
    agg: dict[str, float] = {}
    for name, v in zip(names, pipe.named_steps["clf"].feature_importances_):
        if name.startswith("num__"):
            f = name[5:]
        else:
            body = name[5:]
            f = next((c for c in sorted(cat, key=len, reverse=True) if body.startswith(c + "_")), body)
        agg[f] = agg.get(f, 0.0) + float(v)
    tot = sum(agg.values()) or 1.0
    return [{"feature": f, "share": round(v / tot, 4)} for f, v in sorted(agg.items(), key=lambda kv: -kv[1])[:20]]


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
    if lo > 0:
        verdict, text = "mlp_better", f"The neural network scored higher than XGBoost on {name}, and the 95% range of the gap stays above zero."
    elif hi < 0:
        verdict, text = "xgboost_better", f"XGBoost scored higher than the neural network on {name}, and the 95% range of the gap stays below zero."
    else:
        verdict, text = "no_clear_difference", (f"No clear difference in {name}: the 95% range of the gap includes zero, "
                                                 "so this test set cannot separate the two models.")
    return {"reference": "xgboost", "metric": metric, "same_split": True,
            "reference_settings": {"n_estimators": ref_cfg.n_estimators, "max_depth": ref_cfg.max_depth,
                                   "learning_rate": ref_cfg.learning_rate, "tuned": False},
            "selected": {"algorithm": "mlp", "test": {x: test.get(x) for x in keys}, "cv_mean": cv_sel[0], "cv_sd": cv_sel[1]},
            "xgboost": {"algorithm": "xgboost", "test": rtest, "cv_mean": cv_ref[0], "cv_sd": cv_ref[1]},
            "difference": round(float(a - b), 4), "difference_ci95": [lo, hi], "verdict": verdict, "summary": text,
            "caution": COMPARISON_LIMITATION}


def run(df: pd.DataFrame, cfg: Config) -> tuple[dict, dict]:
    """Returns (result for the experiment record, bundle to persist)."""
    warnings: list[str] = []
    if cfg.algorithm not in ALGORITHMS:
        raise TrainingError("Unknown algorithm. Choose xgboost or mlp.")
    X, y, num, cat, classes, class_counts = _prepare(df, cfg, warnings)
    clf = cfg.task == "classification"
    mlp = cfg.algorithm == "mlp"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed,
                                          stratify=y if clf else None)
    w = compute_sample_weight("balanced", ytr) if (clf and cfg.balance_classes) else None
    fit_params = {"clf__sample_weight": w} if w is not None else {}
    pipe = _pipeline(num, cat, cfg)
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

    importance = _perm_importance(pipe, Xte, yte, clf, cfg.seed) if mlp else _importance(pipe, cat)
    comparison = (_compare_with_xgboost(cfg, num, cat, Xtr, ytr, Xte, yte, w, splitter, pred, test, cvr, k if clf else None)
                  if mlp and cfg.compare_xgboost else None)
    n_all, n_te = len(X), len(Xte)
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

    limits = LIMITATIONS + (MLP_LIMITATIONS if mlp else []) + ([COMPARISON_LIMITATION] if comparison else [])
    result = {"task": cfg.task, "algorithm": cfg.algorithm, "n_rows": int(n_all), "n_train": int(len(Xtr)), "n_test": int(n_te),
              "features": {"numeric": num, "categorical": cat}, "baseline": baseline, "test": test, "cv": cvr,
              **detail, "feature_importance": importance,
              "importance_method": "permutation on the hold-out set" if mlp else "gain",
              "warnings": warnings, "limitations": limits}
    if mlp:
        result["model_settings"] = {"hidden_layers": [int(n) for n in cfg.hidden_layers], "alpha": cfg.alpha,
                                    "max_iter": cfg.max_iter, "iterations_used": iters}
    if comparison:
        result["comparison"] = comparison
    bundle = {"pipeline": pipe, "task": cfg.task, "target": cfg.target, "classes": classes,
              "numeric": num, "categorical": cat}
    return result, bundle
