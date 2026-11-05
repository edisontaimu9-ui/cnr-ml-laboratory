import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/ex.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import storage  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Experiment, utcnow  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}


def make_csv(n=240, seed=1):
    rng = np.random.default_rng(seed)
    rows = ["age,weight,district,muac,status,score,rare"]
    for i in range(n):
        age = int(rng.integers(6, 59))
        weight = round(float(rng.normal(8 + age * 0.12, 1.5)), 1)
        z = (weight - (8 + age * 0.12)) + rng.normal(0, 0.6)
        status = "severe" if z < -1.6 else "moderate" if z < -0.5 else "normal"
        muac = {"normal": 14, "moderate": 12.3, "severe": 10.9}[status] + rng.normal(0, 0.05)   # leaks the label
        district = rng.choice(["Zomba", "Blantyre", "Mangochi"])
        score = round(float(weight * 2 + rng.normal(0, 1)), 2)
        rare = "yes" if i < 3 else "no"
        rows.append(f"{age},{weight},{district},{muac:.2f},{status},{score},{rare}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ds(client):
    d = client.post("/v1/datasets", json={"name": "train-ds"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-001"},
                    files={"file": ("t.csv", make_csv(), "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


def train(client, ds, **over):
    body = {"model_slug": "acute-malnutrition-classification", "dataset_id": ds, "target": "status",
            "exclude": ["muac", "score", "rare"], "n_estimators": 40, **over}
    r = client.post("/v1/experiments", json=body, headers=ADMIN)
    assert r.status_code == 201, r.text
    return client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()  # background task already ran


def test_classification_run_is_complete_and_honest(client, ds):
    e = train(client, ds)
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert m["n_train"] + m["n_test"] == 240 and m["n_test"] == 48
    assert {"accuracy", "macro_f1", "balanced_accuracy", "macro_f1_ci95"} <= set(m["test"])
    assert m["test"]["macro_f1_ci95"][0] <= m["test"]["macro_f1"] <= m["test"]["macro_f1_ci95"][1]
    assert "macro_f1" in m["baseline"] and m["cv"]["folds"] >= 2
    cm = m["confusion_matrix"]["matrix"]
    assert len(cm) == 3 and sum(map(sum, cm)) == m["n_test"]
    assert m["feature_importance"][0]["feature"] in ("age", "weight", "district")
    assert any("Small dataset" in w for w in m["warnings"]) and m["limitations"]
    assert storage.exists(m["artifact_uri"])
    cat = {c["slug"]: c["status"] for c in client.get("/v1/catalog", headers=ADMIN).json()}
    assert cat["acute-malnutrition-classification"] == "in_development"   # never "available" from here
    assert client.get("/v1/models/versions", headers=ADMIN).json() == []


def test_leaky_feature_is_flagged(client, ds):
    e = train(client, ds, exclude=["score", "rare"])
    assert e["status"] == "completed"
    assert any("Suspiciously strong" in w for w in e["metrics_json"]["warnings"])


def test_tiny_class_is_rejected_with_reason(client, ds):
    e = train(client, ds, target="rare", exclude=["muac"])
    assert e["status"] == "failed" and "at least 10 examples" in e["metrics_json"]["error"]


def test_regression_run_and_task_mismatch(client, ds):
    e = train(client, ds, model_slug="nutritional-outcome-prediction", target="score", exclude=["muac", "rare", "status"])
    assert e["status"] == "completed" and {"r2", "rmse", "mae"} <= set(e["metrics_json"]["test"])
    assert e["metrics_json"]["test"]["r2"] > e["metrics_json"]["baseline"]["r2"]
    bad = train(client, ds, model_slug="nutritional-outcome-prediction", target="status", exclude=[])
    assert bad["status"] == "failed" and "numeric" in bad["metrics_json"]["error"]


def test_validation_and_permissions(client, ds):
    body = {"model_slug": "diabetes-risk", "dataset_id": ds, "target": "nope"}
    assert client.post("/v1/experiments", json=body, headers=ADMIN).status_code == 422
    assert client.post("/v1/experiments", json={**body, "model_slug": "x"}, headers=ADMIN).status_code == 404
    viewer = client.post("/v1/admin/keys", json={"name": "viewer-ex", "role": "viewer"}, headers=ADMIN).json()["key"]
    assert client.post("/v1/experiments", json={**body, "target": "status"}, headers={"X-API-Key": viewer}).status_code == 403
    assert client.get("/v1/experiments", headers={"X-API-Key": viewer}).status_code == 200


def test_one_at_a_time(client, ds):
    with SessionLocal() as db:
        db.add(Experiment(model_definition_id=1, framework="xgboost", status="running", created_at=utcnow()))
        db.commit()
    r = client.post("/v1/experiments", headers=ADMIN, json={"model_slug": "diabetes-risk", "dataset_id": ds, "target": "status"})
    assert r.status_code == 409
    with SessionLocal() as db:
        for e in db.query(Experiment).filter(Experiment.status == "running"):
            e.status = "failed"
        db.commit()


def test_audit_has_no_data(client):
    log = client.get("/v1/audit?limit=200", headers=ADMIN).text
    assert "experiment.created" in log and "experiment.completed" in log and "Zomba" not in log


def test_no_signal_is_called_out(client):
    rng = np.random.default_rng(5)
    rows = ["x1,x2,label"] + [f"{rng.normal():.3f},{rng.normal():.3f},{rng.choice(['a', 'b'])}" for _ in range(300)]
    d = client.post("/v1/datasets", json={"name": "noise-ds"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-009"},
                    files={"file": ("n.csv", "\n".join(rows) + "\n", "text/csv")})
    assert r.status_code == 201, r.text
    e = train(client, d["id"], target="label", exclude=[])
    assert e["status"] == "completed"
    b = e["metrics_json"]["baseline"]
    assert b["macro_f1"] >= max(b["majority_macro_f1"], b["chance_macro_f1"]) - 1e-9 and b["chance_macro_f1"] > 0.4
    assert any("does not clearly beat the baseline" in w for w in e["metrics_json"]["warnings"])


# ---- neural network (MLP) and the comparison against XGBoost ----

MLP = {"algorithm": "mlp", "hidden_layers": [16, 8], "max_iter": 150}


def test_default_run_is_xgboost_without_comparison(client, ds):
    e = train(client, ds)
    assert e["framework"] == "xgboost" and e["metrics_json"]["algorithm"] == "xgboost"
    assert "comparison" not in e["metrics_json"] and e["metrics_json"]["importance_method"] == "gain"
    assert e["summary"]["algorithm"] == "xgboost" and "vs_xgboost" not in e["summary"]


def test_mlp_classification_compares_against_xgboost(client, ds):
    e = train(client, ds, **MLP)
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert e["framework"] == "mlp" and m["algorithm"] == "mlp" and m["model_settings"]["hidden_layers"] == [16, 8]
    assert m["importance_method"].startswith("permutation") and m["feature_importance"][0]["feature"] in ("age", "weight", "district")
    c = m["comparison"]
    assert c["reference"] == "xgboost" and c["same_split"] and c["metric"] == "macro_f1" and c["reference_settings"]["tuned"] is False
    assert c["selected"]["test"]["macro_f1"] == m["test"]["macro_f1"]          # the headline is the selected model's own score
    assert {"macro_f1", "balanced_accuracy", "accuracy", "roc_auc"} <= set(c["xgboost"]["test"])
    lo, hi = c["difference_ci95"]
    assert lo <= hi and abs(c["difference"] - (c["selected"]["test"]["macro_f1"] - c["xgboost"]["test"]["macro_f1"])) < 1e-3
    assert c["verdict"] in ("mlp_better", "xgboost_better", "no_clear_difference") and c["summary"] and c["caution"]
    assert c["caution"] in m["limitations"] and any("Neural networks" in x for x in m["limitations"])
    assert any("Only" in w and "Neural networks are unstable" in w for w in m["warnings"])          # 192 training rows
    assert e["summary"]["algorithm"] == "mlp" and e["summary"]["vs_xgboost"]["verdict"] == c["verdict"]
    assert storage.exists(m["artifact_uri"])


def test_mlp_comparison_is_reproducible_and_optional(client, ds):
    a, b = train(client, ds, **MLP)["metrics_json"], train(client, ds, **MLP)["metrics_json"]
    assert a["test"] == b["test"] and a["comparison"] == b["comparison"]
    off = train(client, ds, compare_xgboost=False, **MLP)["metrics_json"]
    assert "comparison" not in off and off["test"] == a["test"]
    assert not any("Choosing the higher-scoring" in x for x in off["limitations"])


def test_mlp_regression_run(client, ds):
    e = train(client, ds, model_slug="nutritional-outcome-prediction", target="score", exclude=["muac", "rare", "status"], **MLP)
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert m["test"]["r2"] > m["baseline"]["r2"] and m["comparison"]["metric"] == "r2"
    assert {"r2", "rmse", "mae"} <= set(m["comparison"]["xgboost"]["test"])


def test_mlp_input_is_validated(client, ds):
    body = {"model_slug": "diabetes-risk", "dataset_id": ds, "target": "status", "algorithm": "mlp"}
    for bad in ({"algorithm": "svm"}, {"hidden_layers": []}, {"hidden_layers": [8, 8, 8, 8]}, {"hidden_layers": [1]},
                {"hidden_layers": [1000]}, {"max_iter": 5}, {"alpha": -1}):
        assert client.post("/v1/experiments", json={**body, **bad}, headers=ADMIN).status_code == 422, bad


def test_mlp_bundle_serves_with_missing_and_unseen_values(client, ds):
    import io

    import joblib

    from app import serving
    m = train(client, ds, **MLP)["metrics_json"]
    bundle = joblib.load(io.BytesIO(storage.load_bytes(m["artifact_uri"])))
    out = serving.predict(bundle, [{"age": 20, "weight": 9.1, "district": "Zomba"}, {"age": 30, "district": "Unknownville"}, {"weight": 8.0}])
    assert len(out) == 3 and all(abs(sum(o["probabilities"].values()) - 1) < 1e-3 for o in out)
    assert set(serving.categories(bundle)["district"]) == {"Blantyre", "Mangochi", "Zomba"}


def test_mlp_experiment_can_be_registered_with_its_comparison_frozen(client, ds):
    # its own slug, so the version numbers other test modules rely on are not disturbed
    e = train(client, ds, model_slug="ckd-nutrition-risk", **MLP)
    assert e["status"] == "completed", e["metrics_json"]
    r = client.post("/v1/models/versions", headers=ADMIN, json={
        "experiment_id": e["id"], "intended_use": "Research use only, to test the registry with a neural network run."})
    assert r.status_code == 201, r.text
    v = client.get(f"/v1/models/versions/{r.json()['id']}", headers=ADMIN).json()
    assert v["metrics_json"]["algorithm"] == "mlp" and v["metrics_json"]["comparison"]["reference"] == "xgboost"
    assert v["headline"]["metric"] == "macro_f1" and v["artifact_sha256"]


# ---- interpretable logistic regression (calibrated probabilities) ----

LR = {"algorithm": "logistic"}


def test_logistic_classification_reports_coefficients_and_calibration(client, ds):
    e = train(client, ds, calibration="sigmoid", **LR)
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert e["framework"] == "logistic" and m["algorithm"] == "logistic" and m["importance_method"].startswith("absolute")
    assert m["model_settings"]["C"] == 1.0 and m["model_settings"]["calibration"] == "sigmoid"
    assert m["calibration"]["summary"].startswith("Recalibration")
    co = m["coefficients"]
    assert co["binary"] is False and {o["class"] for o in co["outcomes"]} == {"moderate", "normal", "severe"}
    assert {t["feature"] for o in co["outcomes"] for t in o["terms"]} <= {"age", "weight", "district"}
    assert any(t["kind"] == "category" and t["level"] in ("Zomba", "Blantyre", "Mangochi") for t in co["outcomes"][0]["terms"])
    for o in co["outcomes"]:                                                           # largest effects first
        sizes = [abs(t["coefficient"]) for t in o["terms"]]
        assert sizes == sorted(sizes, reverse=True)
    cal = m["calibration"]
    assert cal["method"] == "sigmoid" and {"brier", "log_loss", "ece", "reliability"} <= set(cal["after"]) and "before" in cal and cal["summary"]
    assert sum(b["n"] for b in cal["after"]["reliability"]) == m["n_test"]
    assert 0 <= cal["after"]["ece"] <= 1
    assert any("balanced during training" in w for w in m["warnings"])                 # balanced classes by default
    assert any("straight line" in x for x in m["limitations"])
    c = m["comparison"]
    assert c["reference"] == "xgboost" and c["selected"]["algorithm"] == "logistic" and c["same_split"]
    assert c["verdict"] in ("logistic_better", "xgboost_better", "no_clear_difference")
    assert e["summary"]["algorithm"] == "logistic" and "brier" in e["summary"]["calibration"]
    assert storage.exists(m["artifact_uri"])


def test_logistic_binary_gives_odds_ratios_and_hides_rare_levels(client, ds):
    # binary target built from the training table: severe vs not (one outcome, odds ratios present)
    rng = np.random.default_rng(11)
    rows = ["age,weight,district,flag"]
    for i in range(300):
        age, w = int(rng.integers(6, 59)), float(rng.normal(10, 2))
        d = "Zomba" if i % 2 else "Blantyre"
        if i < 4:
            d = "Tiny"                                                  # fewer than 10 rows: must not be named
        flag = "yes" if (w + rng.normal(0, 1.2)) < 9.3 else "no"
        rows.append(f"{age},{w:.2f},{d},{flag}")
    d = client.post("/v1/datasets", json={"name": "lr-binary"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-LR1"},
                    files={"file": ("b.csv", "\n".join(rows) + "\n", "text/csv")})
    assert r.status_code == 201, r.text
    e = train(client, d["id"], target="flag", exclude=[], balance_classes=False, **LR)         # default: no recalibration
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    co = m["coefficients"]
    assert co["binary"] and len(co["outcomes"]) == 1 and co["outcomes"][0]["class"] == "yes"
    terms = co["outcomes"][0]["terms"]
    assert all("odds_ratio" in t for t in terms) and "Tiny" not in str(m) and co["rare_levels_hidden"] >= 1
    wt = next(t for t in terms if t["feature"] == "weight")
    assert wt["coefficient"] < 0 and wt["odds_ratio"] < 1 and wt["sd"] and wt["odds_ratio_per_unit"] < 1     # heavier -> less likely
    assert m["feature_importance"][0]["feature"] == "weight"
    assert "before" not in m["calibration"] and m["calibration"]["method"] == "none"
    assert not any("balanced during training" in w for w in m["warnings"])


def test_logistic_calibration_runs_with_isotonic_and_warns_when_small(client, ds):
    e = train(client, ds, calibration="isotonic", compare_xgboost=False, **LR)
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert m["calibration"]["method"] == "isotonic" and "comparison" not in m
    assert any("Isotonic calibration" in w for w in m["warnings"])


def test_calibration_verdict_wording():
    from app.training import _calibration_verdict as v
    base = {"brier": 0.30, "ece": 0.10}
    assert "lowered" in v(base, {"brier": 0.25, "ece": 0.06})
    assert "worse" in v(base, {"brier": 0.34, "ece": 0.12}) and "off" in v(base, {"brier": 0.34, "ece": 0.12})
    assert "did not clearly change" in v(base, {"brier": 0.301, "ece": 0.10})


def test_logistic_is_reproducible(client, ds):
    a, b = train(client, ds, **LR)["metrics_json"], train(client, ds, **LR)["metrics_json"]
    assert a["test"] == b["test"] and a["coefficients"] == b["coefficients"] and a["calibration"] == b["calibration"]


def test_logistic_refused_for_regression(client, ds):
    body = {"model_slug": "nutritional-outcome-prediction", "dataset_id": ds, "target": "score", "algorithm": "logistic"}
    r = client.post("/v1/experiments", json=body, headers=ADMIN)
    assert r.status_code == 422 and "classification" in r.json()["detail"]


def test_logistic_input_is_validated(client, ds):
    body = {"model_slug": "diabetes-risk", "dataset_id": ds, "target": "status", "algorithm": "logistic"}
    for bad in ({"logistic_c": 0}, {"logistic_c": 1000}, {"calibration": "platt"}, {"max_iter": 5}):
        assert client.post("/v1/experiments", json={**body, **bad}, headers=ADMIN).status_code == 422, bad


def test_logistic_bundle_serves_calibrated_probabilities(client, ds):
    import io

    import joblib

    from app import serving
    m = train(client, ds, **LR)["metrics_json"]
    bundle = joblib.load(io.BytesIO(storage.load_bytes(m["artifact_uri"])))
    out = serving.predict(bundle, [{"age": 20, "weight": 9.1, "district": "Zomba"}, {"age": 30, "district": "Unknownville"}, {"weight": 8.0}])
    assert len(out) == 3 and all(abs(sum(o["probabilities"].values()) - 1) < 1e-3 for o in out)
    assert set(serving.categories(bundle)["district"]) == {"Blantyre", "Mangochi", "Zomba"}


def test_logistic_experiment_can_be_registered(client, ds):
    e = train(client, ds, model_slug="ckd-nutrition-risk", **LR)
    assert e["status"] == "completed", e["metrics_json"]
    r = client.post("/v1/models/versions", headers=ADMIN, json={
        "experiment_id": e["id"], "intended_use": "Research use only, to test the registry with a logistic regression run."})
    assert r.status_code == 201, r.text
    v = client.get(f"/v1/models/versions/{r.json()['id']}", headers=ADMIN).json()
    assert v["metrics_json"]["algorithm"] == "logistic" and v["metrics_json"]["coefficients"]["outcomes"] and v["artifact_sha256"]


def test_logistic_prediction_explains_itself_without_echoing_values(client, ds):
    import io

    import joblib

    from app import serving
    m = train(client, ds, **LR)["metrics_json"]
    bundle = joblib.load(io.BytesIO(storage.load_bytes(m["artifact_uri"])))
    out = serving.predict(bundle, [{"age": 20, "weight": 6.0, "district": "Zomba"}, {"age": 30}, {"age": 20, "weight": 9.1, "district": "Unknownville"}])
    ex = out[0]["explanation"]
    assert ex["toward"] == out[0]["predicted_class"] and 1 <= len(ex["drivers"]) <= 5 and ex["not_given"] == []
    sizes = [abs(d["effect"]) for d in ex["drivers"]]
    assert sizes == sorted(sizes, reverse=True) and all(d["direction"] in ("raises", "lowers") for d in ex["drivers"])
    assert "6.0" not in str(ex) and "Zomba" not in str(ex)
    assert set(out[1]["explanation"]["not_given"]) == {"weight", "district"}
    assert all(d["feature"] == "age" for d in out[1]["explanation"]["drivers"])
    assert out[2]["explanation"]["note"]


def test_binary_explanation_direction_matches_coefficient_sign(client):
    import io

    import joblib
    import pandas as pd

    from app import serving, training
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"weight": rng.normal(10, 2, 300)})
    df["flag"] = np.where(df.weight + rng.normal(0, 1, 300) < 9.5, "yes", "no")
    _, bundle = training.run(df, training.Config(target="flag", task="classification", algorithm="logistic", balance_classes=False, compare_xgboost=False))
    low, high = serving.predict(bundle, [{"weight": 6.0}, {"weight": 14.0}])
    assert low["predicted_class"] == "yes" and low["explanation"]["drivers"][0]["direction"] == "raises"
    assert high["predicted_class"] == "no" and high["explanation"]["drivers"][0]["direction"] == "raises"   # towards "no"


def test_other_models_have_no_explanation(client, ds):
    import io

    import joblib

    from app import serving
    m = train(client, ds)["metrics_json"]
    bundle = joblib.load(io.BytesIO(storage.load_bytes(m["artifact_uri"])))
    assert "explanation" not in serving.predict(bundle, [{"age": 20, "weight": 9.1, "district": "Zomba"}])[0]
