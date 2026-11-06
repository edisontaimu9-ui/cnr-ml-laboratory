import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/pd.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
SLUG = "prediabetes-screen"
LAB_COLS = ["hba1c_pct", "fasting_glucose", "on_metformin", "diabetes_diagnosis"]


def make_csv(n=700, seed=9):
    rng = np.random.default_rng(seed)
    rows = ["age,bmi,waist_cm,family_history,sugary_drinks_per_week,activity,hba1c_pct,fasting_glucose,on_metformin,diabetes_diagnosis,raised_hba1c"]
    for _ in range(n):
        age = int(rng.integers(20, 80))
        bmi = round(float(rng.normal(25 + (age - 45) * 0.03, 4.5)), 1)
        waist = round(float(bmi * 3.3 + rng.normal(0, 7)), 1)
        fam = rng.choice(["yes", "no"], p=[0.25, 0.75])
        sugar = int(rng.integers(0, 15))
        act = rng.choice(["low", "moderate", "high"], p=[0.35, 0.4, 0.25])
        load = 0.03 * (age - 45) + 0.1 * (bmi - 25) + 0.5 * (fam == "yes") + 0.05 * sugar - 0.4 * (act == "high") + rng.normal(0, 0.8)
        a1c = round(float(5.3 + 0.25 * load + rng.normal(0, 0.25)), 1)
        glu = round(float(5.0 + 0.4 * load + rng.normal(0, 0.5)), 1)
        rows.append(f"{age},{bmi},{waist},{fam},{sugar},{act},{a1c},{glu},no,no,{'yes' if a1c >= 5.7 else 'no'}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ds(client):
    d = client.post("/v1/datasets", json={"name": "pd-ds"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-PD"},
                    files={"file": ("p.csv", make_csv(), "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


def start(client, ds, **over):
    body = {"model_slug": SLUG, "dataset_id": ds, "target": "raised_hba1c", "exclude": LAB_COLS, "n_estimators": 40, **over}
    r = client.post("/v1/experiments", json=body, headers=ADMIN)
    assert r.status_code == 201, r.text
    return client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()


def test_catalogue_entry_and_label_note(client):
    m = {c["slug"]: c for c in client.get("/v1/catalog", headers=ADMIN).json()}[SLUG]
    assert m["status"] == "planned" and m["task_type"] == "classification"
    assert "5.7%" in m["label_source"] and "never be used as features" in m["label_source"]


def test_trains_without_lab_features(client, ds):
    e = start(client, ds)
    assert e["status"] == "completed", e["metrics_json"]
    f = e["metrics_json"]["features"]
    assert set(f["numeric"]) == {"age", "bmi", "waist_cm", "sugary_drinks_per_week"}
    assert set(f["categorical"]) == {"family_history", "activity"}


def test_logistic_works(client, ds):
    e = start(client, ds, algorithm="logistic")
    assert e["status"] == "completed" and "coefficients" in e["metrics_json"]


@pytest.mark.parametrize("col", LAB_COLS)
def test_each_lab_or_treatment_column_is_refused(client, ds, col):
    e = start(client, ds, exclude=[c for c in LAB_COLS if c != col])
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


def test_guard_does_not_apply_to_other_models(client, ds):
    r = client.post("/v1/experiments", headers=ADMIN, json={"model_slug": "hospital-malnutrition-risk", "dataset_id": ds,
                    "target": "raised_hba1c", "exclude": LAB_COLS, "n_estimators": 40})
    e = client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()
    assert e["status"] == "completed"
