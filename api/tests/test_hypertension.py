import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/ht.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
SLUG = "hypertension-risk"


def make_csv(n=600, seed=3):
    """Synthetic screening survey; the label comes from measured BP, which is also kept as columns to test the guard."""
    rng = np.random.default_rng(seed)
    rows = ["age,bmi,waist_cm,salt_intake,alcohol_use,activity,systolic_bp,diastolic_bp,hypertension"]
    for _ in range(n):
        age = int(rng.integers(20, 80))
        bmi = round(float(rng.normal(25 + (age - 45) * 0.03, 4)), 1)
        waist = round(float(bmi * 3.4 + rng.normal(0, 6)), 1)
        salt = rng.choice(["low", "medium", "high"], p=[0.3, 0.45, 0.25])
        alcohol = rng.choice(["none", "occasional", "frequent"], p=[0.5, 0.35, 0.15])
        activity = rng.choice(["low", "moderate", "high"], p=[0.35, 0.4, 0.25])
        risk = (0.05 * (age - 45) + 0.12 * (bmi - 25) + 0.5 * (salt == "high") + 0.6 * (alcohol == "frequent")
                - 0.5 * (activity == "high") + rng.normal(0, 1))
        sbp = int(round(118 + 6 * risk + rng.normal(0, 6)))
        dbp = int(round(76 + 3.5 * risk + rng.normal(0, 4)))
        rows.append(f"{age},{bmi},{waist},{salt},{alcohol},{activity},{sbp},{dbp},{'yes' if (sbp >= 140 or dbp >= 90) else 'no'}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ds(client):
    d = client.post("/v1/datasets", json={"name": "ht-ds"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-HT"},
                    files={"file": ("h.csv", make_csv(), "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


def start(client, ds, **over):
    body = {"model_slug": SLUG, "dataset_id": ds, "target": "hypertension", "exclude": ["systolic_bp", "diastolic_bp"],
            "n_estimators": 40, **over}
    r = client.post("/v1/experiments", json=body, headers=ADMIN)
    assert r.status_code == 201, r.text
    return client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()


def test_catalogue_lists_hypertension_as_planned_with_its_own_label_note(client):
    cat = {m["slug"]: m for m in client.get("/v1/catalog", headers=ADMIN).json()}
    m = cat[SLUG]
    assert m["status"] == "planned" and m["task_type"] == "classification"
    assert "blood pressure" in m["label_source"].lower() and "never be used as features" in m["label_source"]
    assert "To be defined" in cat["diabetes-risk"]["label_source"]  # other models keep the generic note


def test_trains_without_bp_features(client, ds):
    e = start(client, ds)
    assert e["status"] == "completed", e
    feats = e["metrics_json"]["features"]
    assert set(feats["numeric"]) == {"age", "bmi", "waist_cm"}
    assert set(feats["categorical"]) == {"salt_intake", "alcohol_use", "activity"}
    assert e["metrics_json"]["test"]["balanced_accuracy"] > e["metrics_json"]["baseline"]["balanced_accuracy"]


def test_logistic_works_and_gives_odds_ratios(client, ds):
    e = start(client, ds, algorithm="logistic")
    assert e["status"] == "completed" and "coefficients" in e["metrics_json"]


@pytest.mark.parametrize("bad", [["diastolic_bp"], ["systolic_bp"], ["systolic_bp", "diastolic_bp"]])
def test_blood_pressure_columns_are_refused_as_features(client, ds, bad):
    keep = [c for c in ("systolic_bp", "diastolic_bp") if c not in bad]
    e = start(client, ds, exclude=keep)
    assert e["status"] == "failed"
    assert "cannot be a feature" in e["metrics_json"]["error"]


def test_guard_applies_only_to_hypertension_model(client, ds):
    r = client.post("/v1/experiments", headers=ADMIN, json={
        "model_slug": "diabetes-risk", "dataset_id": ds, "target": "hypertension", "exclude": ["systolic_bp", "diastolic_bp"],
        "n_estimators": 40})
    assert r.status_code == 201
    e = client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()
    assert e["status"] == "completed"
