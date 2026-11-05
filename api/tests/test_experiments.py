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
