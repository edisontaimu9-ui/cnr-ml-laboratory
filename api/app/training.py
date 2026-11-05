"""XGBoost training and evaluation (Phase 3).

Rules
- Every preprocessing step lives inside the sklearn Pipeline, so it is fitted on training data only.
- The hold-out test set is split off first and touched once, after cross-validation on the training part.
- Results are always reported next to a trivial baseline, with an uncertainty range.
- Loud warnings for the usual ways a medical model looks better than it is (leakage, tiny test sets).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

MIN_ROWS, MIN_PER_CLASS, MAX_CLASSES = 50, 10, 12

LIMITATIONS = [
    "Scores come from an internal hold-out split of one dataset. They are not external validation.",
    "A model trained on data from one site or period may perform worse elsewhere.",
    "Outputs are risk estimates to support, not replace, clinical judgement.",
]


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


def _pipeline(num, cat, cfg: Config) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num),  # XGBoost handles missing numbers natively
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
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


def _ms(vals) -> tuple[float, float]:
    return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)


def run(df: pd.DataFrame, cfg: Config) -> tuple[dict, dict]:
    """Returns (result for the experiment record, bundle to persist)."""
    warnings: list[str] = []
    X, y, num, cat, classes, class_counts = _prepare(df, cfg, warnings)
    clf = cfg.task == "classification"
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed,
                                          stratify=y if clf else None)
    w = compute_sample_weight("balanced", ytr) if (clf and cfg.balance_classes) else None
    fit_params = {"clf__sample_weight": w} if w is not None else {}
    pipe = _pipeline(num, cat, cfg)
    rng = np.random.default_rng(cfg.seed)

    if clf:
        k = len(classes)
        folds = max(2, min(5, int(np.bincount(ytr).min())))
        cv = cross_validate(pipe, Xtr, ytr, cv=StratifiedKFold(folds, shuffle=True, random_state=cfg.seed),
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
        boots = []
        for _ in range(300):
            i = rng.integers(0, len(yte), len(yte))
            boots.append(f1_score(yte[i], pred[i], average="macro", labels=np.unique(yte[i]), zero_division=0))
        test = {"accuracy": round(float((pred == yte).mean()), 4), "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
                "macro_f1": round(main, 4), "macro_precision": round(float(p.mean()), 4), "macro_recall": round(float(r.mean()), 4),
                "roc_auc": None if auc is None else round(auc, 4),
                "macro_f1_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
        baseline = {"strategy": "always predict the most common class", "accuracy": round(float((base_pred == yte).mean()), 4),
                    "macro_f1": round(float(bf.mean()), 4), "balanced_accuracy": round(1 / k, 4)}
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
        cv = cross_validate(pipe, Xtr, ytr, cv=KFold(folds, shuffle=True, random_state=cfg.seed),
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

    importance = _importance(pipe, cat)
    n_all, n_te = len(X), len(Xte)
    if n_all < 500:
        warnings.append(f"Small dataset ({n_all} rows). Treat these results as a check that the pipeline works, not as evidence of performance.")
    if n_te < 50:
        warnings.append(f"The test set has only {n_te} rows, so scores are unstable. See the confidence range.")
    if headline >= 0.97 or (importance and importance[0]["share"] >= 0.7):
        warnings.append("Suspiciously strong result" + (f" (one feature, '{importance[0]['feature']}', carries {importance[0]['share']:.0%} of the model)" if importance else "")
                        + ". Check for features that are derived from the label or measured after it, such as the measurements a nutrition status is calculated from. Exclude them and retrain.")
    if headline <= base_headline + 0.02:
        warnings.append("The model does not clearly beat the trivial baseline. It has not learned a useful pattern from these features.")
    if abs(headline - cv_headline) > 0.15:
        warnings.append("Cross-validation and test scores differ a lot, a sign of instability on this amount of data.")

    result = {"task": cfg.task, "n_rows": int(n_all), "n_train": int(len(Xtr)), "n_test": int(n_te),
              "features": {"numeric": num, "categorical": cat}, "baseline": baseline, "test": test, "cv": cvr,
              **detail, "feature_importance": importance, "warnings": warnings, "limitations": LIMITATIONS}
    bundle = {"pipeline": pipe, "task": cfg.task, "target": cfg.target, "classes": classes,
              "numeric": num, "categorical": cat}
    return result, bundle
