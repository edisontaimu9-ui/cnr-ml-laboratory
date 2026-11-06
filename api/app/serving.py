"""Prediction serving.

Only the single *deployed* version of a model is ever served. Its file is read as bytes, hashed against the
SHA-256 recorded at registration and only then loaded, so a file that changed on disk is never executed.
Inputs are validated against the features the model was trained on and are never stored or logged."""
import io
import math
import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

import hashlib

from . import storage
from .models import ModelDefinition, ModelVersion
from .training import _lr_base, _split_name


class ServingError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


_cache: dict[str, tuple[int, str, dict]] = {}
_lock = threading.Lock()
MAX_TEXT = 100
MISSING_WARN = 0.5
MAX_DRIVERS = 5
EXPLAIN_NOTE = ("These are the features that pushed this estimate up or down, from the model's coefficients. They describe the model, "
                "not the cause of the condition, and the calibrated probability can differ slightly from what they imply alone.")


def deployed(db: Session, slug: str) -> tuple[ModelDefinition, ModelVersion | None]:
    d = db.scalar(select(ModelDefinition).where(ModelDefinition.slug == slug))
    if d is None:
        raise ServingError(404, "Unknown model.")
    v = db.scalar(select(ModelVersion).where(ModelVersion.model_definition_id == d.id, ModelVersion.status == "deployed"))
    return d, v


def load_bundle(v: ModelVersion, slug: str) -> dict:
    with _lock:
        hit = _cache.get(slug)
        if hit and hit[0] == v.id and hit[1] == v.artifact_sha256:
            return hit[2]
    try:
        data = storage.load_bytes(v.artifact_uri)
    except Exception:
        raise ServingError(503, "The deployed model file is unavailable. Predictions are paused.")
    if hashlib.sha256(data).hexdigest() != v.artifact_sha256:
        raise ServingError(503, "The deployed model file does not match its recorded hash. Predictions are paused.")
    bundle = joblib.load(io.BytesIO(data))      # our own file, verified byte-for-byte above
    with _lock:
        _cache[slug] = (v.id, v.artifact_sha256, bundle)
    return bundle


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def categories(bundle: dict) -> dict[str, list[str]]:
    """Values seen in training for each categorical feature (the imputer's 'missing' filler is hidden)."""
    try:
        enc = bundle["pipeline"].named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
        return {c: sorted(str(x) for x in cats if str(x) != "missing") for c, cats in zip(bundle["categorical"], enc.categories_)}
    except Exception:
        return {c: [] for c in bundle["categorical"]}


def headline(m: dict) -> dict:
    if m.get("task") == "matching":
        return {"metric": "top1", "test": (m.get("learned") or {}).get("top1"), "baseline": (m.get("names_only") or {}).get("top1"),
                "n_test": m.get("n_test"), "warnings": len(m.get("warnings", [])), "rmse": None}
    t, b = m.get("test", {}), m.get("baseline", {})
    key = "macro_f1" if m.get("task") == "classification" else "r2"
    return {"metric": key, "test": t.get(key), "baseline": b.get(key), "n_test": m.get("n_test"),
            "warnings": len(m.get("warnings", [])), "rmse": t.get("rmse")}


def _frame(bundle: dict, records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[list[str]], list[float]]:
    num, cat, known = bundle["numeric"], bundle["categorical"], categories(bundle)
    cols = set(num) | set(cat)
    rows, notes, miss = [], [], []
    for i, rec in enumerate(records, 1):
        if not isinstance(rec, dict) or not rec:
            raise ServingError(422, f"Record {i} must be an object of feature values.")
        unknown = [k for k in rec if k not in cols]
        if unknown:
            raise ServingError(422, f"Record {i}: '{str(unknown[0])[:60]}' is not a feature of this model. "
                                    "Only the listed measurements are accepted.")
        row, rn = {}, []
        for c in num:
            v = rec.get(c)
            if v is None or v == "":
                row[c] = np.nan
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ServingError(422, f"Record {i}: '{c}' must be a number.")
            else:
                row[c] = float(v)
        for c in cat:
            v = rec.get(c)
            if v is None or v == "":
                row[c] = np.nan
                continue
            if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                raise ServingError(422, f"Record {i}: '{c}' must be text.")
            s = str(v).strip()
            if len(s) > MAX_TEXT:
                raise ServingError(422, f"Record {i}: '{c}' is too long.")
            row[c] = s
            if known.get(c) and s not in known[c]:
                rn.append(f"'{c}' has a value the model did not see in training; it was treated as unknown.")
        given = sum(1 for k in (*num, *cat) if not (isinstance(row[k], float) and math.isnan(row[k])))
        if given == 0:
            raise ServingError(422, f"Record {i} has no usable values.")
        frac = 1 - given / (len(num) + len(cat))
        if frac > MISSING_WARN:
            rn.append(f"{round(frac * 100)}% of the features are missing, so this estimate is unreliable.")
        rows.append(row); notes.append(rn); miss.append(round(frac, 3))
    df = pd.DataFrame(rows, columns=[*num, *cat])
    for c in num:
        df[c] = df[c].astype(float)
    for c in cat:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df, notes, miss


def _explain(bundle: dict, X: pd.DataFrame, proba: np.ndarray) -> list[dict] | None:
    """Per-case drivers for a logistic regression model: each feature's share of the log-odds towards the predicted class
    (value x coefficient on the standardised inputs, summed over a column's category levels). Returns None for other models.
    Features the caller left out are listed as not given, not scored, because their value was filled in. Raw input values
    are never echoed back."""
    try:
        base = _lr_base(bundle["pipeline"])
        if not hasattr(base, "coef_"):
            return None
    except Exception:
        return None
    pre, classes, cat = bundle["pipeline"].named_steps["pre"], bundle["classes"], bundle["categorical"]
    Z = np.asarray(pre.transform(X), dtype=float)
    names = pre.get_feature_names_out()
    coef = base.coef_
    out = []
    for i in range(len(X)):
        c = int(np.argmax(proba[i]))
        if len(classes) == 2:
            w = coef[0] if c == 1 else -coef[0]
        else:
            w = coef[c] - coef.mean(axis=0)         # softmax ignores a shift shared by all classes
        agg: dict[str, float] = {}
        for name, z, wj in zip(names, Z[i], w):
            f = _split_name(name, cat)[0]
            agg[f] = agg.get(f, 0.0) + float(z * wj)
        given = {f: v for f, v in agg.items() if not pd.isna(X.iloc[i][f])}
        drivers = sorted(given.items(), key=lambda kv: -abs(kv[1]))[:MAX_DRIVERS]
        out.append({"method": "logistic regression coefficients", "toward": classes[c],
                    "drivers": [{"feature": f, "effect": round(v, 3), "direction": "raises" if v > 0 else "lowers"}
                                for f, v in drivers if abs(v) >= 0.005],
                    "not_given": [f for f in agg if f not in given], "note": EXPLAIN_NOTE})
    return out


def predict(bundle: dict, records: list[dict[str, Any]]) -> list[dict]:
    X, notes, miss = _frame(bundle, records)
    pipe = bundle["pipeline"]
    out = []
    if bundle["task"] == "classification":
        proba, classes = pipe.predict_proba(X), bundle["classes"]
        for p, rn, mf in zip(proba, notes, miss):
            order = np.argsort(-p)
            top, second = float(p[order[0]]), float(p[order[1]]) if len(p) > 1 else 0.0
            low = top < 0.5 or (top - second) < 0.15 or mf > MISSING_WARN
            out.append({"kind": "risk_estimate", "predicted_class": classes[int(order[0])],
                        "probabilities": {classes[j]: round(float(p[j]), 4) for j in order},
                        "risk_indicators": {"top_probability": round(top, 4), "margin_to_next": round(top - second, 4),
                                            "missing_fraction": mf, "low_confidence": bool(low)},
                        "notes": rn})
        for o, ex in zip(out, _explain(bundle, X, proba) or [None] * len(out)):
            if ex:
                o["explanation"] = ex
    else:
        for v, rn, mf in zip(pipe.predict(X), notes, miss):
            out.append({"kind": "risk_estimate", "predicted_value": round(float(v), 4),
                        "risk_indicators": {"missing_fraction": mf, "low_confidence": mf > MISSING_WARN},
                        "notes": rn})
    return out
