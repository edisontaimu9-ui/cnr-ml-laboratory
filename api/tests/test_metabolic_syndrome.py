import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/ms.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
SLUG = "metabolic-syndrome-screen"
LABEL_COLS = ["triglycerides_mmol", "hdl_mmol", "fasting_glucose", "systolic_bp", "diastolic_bp", "on_statin"]


def make_csv(n=700, seed=5):
    """Synthetic survey: label = 3 of 5 criteria, built from lab columns that must never be features."""
    rng = np.random.default_rng(seed)
    rows = ["age,bmi,waist_cm,sugar_servings,fried_food_days,triglycerides_mmol,hdl_mmol,fasting_glucose,systolic_bp,diastolic_bp,on_statin,mets_syndrome"]
    for _ in range(n):
        age = int(rng.integers(20, 80))
        bmi = round(float(rng.normal(25 + (age - 45) * 0.03, 4.5)), 1)
        waist = round(float(bmi * 3.3 + rng.normal(0, 7)), 1)
        sugar, fried = int(rng.integers(0, 6)), int(rng.integers(0, 8))
        load = 0.04 * (age - 45) + 0.14 * (bmi - 25) + 0.15 * sugar + 0.08 * fried + rng.normal(0, 0.8)
        tg = round(float(1.4 + 0.35 * load + rng.normal(0, 0.4)), 2)
        hdl = round(float(1.2 - 0.12 * load + rng.normal(0, 0.2)), 2)
        glu = round(float(5.2 + 0.3 * load + rng.normal(0, 0.5)), 1)
        sbp = int(round(120 + 6 * load + rng.normal(0, 8)))
        dbp = int(round(78 + 3 * load + rng.normal(0, 5)))
        statin = "yes" if rng.random() < 0.03 else "no"
        crit = [waist >= 90, tg >= 1.7, hdl < 1.0, sbp >= 130 or dbp >= 85, glu >= 5.6]
        rows.append(f"{age},{bmi},{waist},{sugar},{fried},{tg},{hdl},{glu},{sbp},{dbp},{statin},{'yes' if sum(crit) >= 3 else 'no'}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ds(client):
    d = client.post("/v1/datasets", json={"name": "ms-ds"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-MS"},
                    files={"file": ("m.csv", make_csv(), "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


def start(client, ds, **over):
    body = {"model_slug": SLUG, "dataset_id": ds, "target": "mets_syndrome", "exclude": LABEL_COLS, "n_estimators": 40, **over}
    r = client.post("/v1/experiments", json=body, headers=ADMIN)
    assert r.status_code == 201, r.text
    return client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()


def test_catalogue_entry_and_label_note(client):
    m = {c["slug"]: c for c in client.get("/v1/catalog", headers=ADMIN).json()}[SLUG]
    assert m["status"] == "planned" and m["task_type"] == "classification"
    assert "three or more" in m["label_source"] and "never be used as features" in m["label_source"]


def test_trains_on_cheap_measurements_and_warns_about_waist(client, ds):
    e = start(client, ds)
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert set(m["features"]["numeric"]) == {"age", "bmi", "waist_cm", "sugar_servings", "fried_food_days"}
    assert any("part of this model's label definition" in w and "waist_cm" in w for w in m["warnings"])


def test_no_waist_run_has_no_overlap_warning(client, ds):
    e = start(client, ds, exclude=[*LABEL_COLS, "waist_cm"])
    assert e["status"] == "completed", e["metrics_json"]
    assert not any("label definition" in w for w in e["metrics_json"]["warnings"])


@pytest.mark.parametrize("col", LABEL_COLS)
def test_each_label_component_is_refused_as_a_feature(client, ds, col):
    e = start(client, ds, exclude=[c for c in LABEL_COLS if c != col])
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


def test_guards_do_not_apply_to_other_models(client, ds):
    r = client.post("/v1/experiments", headers=ADMIN, json={"model_slug": "diabetes-risk", "dataset_id": ds,
                    "target": "mets_syndrome", "exclude": ["on_statin"], "n_estimators": 40})
    e = client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()
    assert e["status"] == "completed"
    assert not any("label definition" in w for w in e["metrics_json"]["warnings"])
