"""Cardiometabolic phenotypes: the lab's existing k-means exploration with a preset. Not a model, never deployed."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/cp.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}


def phenotype_csv(per=130, seed=11, messy=False):
    """Three planted phenotypes: central adiposity + raised BP and glucose; dyslipidaemia at normal weight; healthy."""
    rng = np.random.default_rng(seed)
    head = "age,waist_cm,bmi,systolic_bp,fasting_glucose,triglycerides,hdl,sugary_drinks_per_week"
    rows = [head + (",weight_kg,mets_syndrome" if messy else "")]
    specs = [(104, 33, 148, 7.2, 1.6, 1.0, 9), (82, 24, 122, 5.0, 3.0, 0.8, 3), (78, 22, 114, 4.8, 1.0, 1.5, 2)]
    for waist, bmi, sbp, glu, tg, hdl, sugar in specs:
        for _ in range(per):
            b = bmi + rng.normal(0, 1.2)
            r = [int(rng.integers(25, 70)), waist + rng.normal(0, 3), b, sbp + rng.normal(0, 5), glu + rng.normal(0, 0.4),
                 tg + rng.normal(0, 0.25), hdl + rng.normal(0, 0.12), max(0, sugar + rng.normal(0, 1.5))]
            line = ",".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r)
            if messy:
                line += f",{b * 2.6 + rng.normal(0, 0.3):.2f},{'yes' if (waist > 100) else 'no'}"
            rows.append(line)
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def upload(client, name, csv):
    d = client.post("/v1/datasets", json={"name": name}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-CP"},
                    files={"file": ("c.csv", csv, "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


@pytest.fixture(scope="module")
def clean(client):
    return upload(client, "cp-clean", phenotype_csv())


@pytest.fixture(scope="module")
def messy(client):
    return upload(client, "cp-messy", phenotype_csv(messy=True))


def explore(client, ds, **over):
    r = client.post("/v1/explorations", headers=ADMIN, json={"dataset_id": ds, **over})
    assert r.status_code == 201, r.text
    return client.get(f"/v1/explorations/{r.json()['id']}", headers=ADMIN).json()


def test_preset_finds_planted_phenotypes_and_stays_an_exploration(client, clean):
    x = explore(client, clean, preset="cardiometabolic")
    assert x["status"] == "completed", x["results_json"]
    assert x["kind"] == "exploration" and x["deployable"] is False
    r = x["results_json"]
    assert r["methods"] == ["clustering"] and r["anomaly"] is None      # clustering only by default
    assert r["clustering"]["k"] == 3 and r["clustering"]["structure"] == "clear_structure"
    p = r["preset"]
    assert p["name"] == "cardiometabolic"
    assert set(p["marker_groups"]) == {"adiposity", "blood_pressure", "glucose", "lipids", "lifestyle"}
    assert "age" in p["not_recognised"] and any("not disease classes" in n for n in p["notes"])
    assert not any("already-assigned" in w or "almost together" in w for w in r["warnings"])


def test_preset_warns_about_label_columns_and_duplicated_measurements(client, messy):
    r = explore(client, messy, preset="cardiometabolic")["results_json"]
    assert any("mets_syndrome" in w and "already-assigned" in w for w in r["warnings"])
    assert any("move almost together" in w and "bmi" in w and "weight_kg" in w for w in r["warnings"])
    assert r["preset"]["label_like_columns"] == ["mets_syndrome"]
    assert {(q["a"], q["b"]) for q in r["preset"]["correlated_pairs"]} >= {("bmi", "weight_kg")}


def test_preset_warns_when_measurements_are_not_recognised(client):
    rng = np.random.default_rng(2)
    csv = "a,b,c\n" + "\n".join(f"{rng.normal():.3f},{rng.normal():.3f},{rng.normal():.3f}" for _ in range(120)) + "\n"
    r = explore(client, upload(client, "cp-unknown", csv), preset="cardiometabolic")["results_json"]
    assert any("No cardiometabolic measurements were recognised" in w for w in r["warnings"])


def test_without_preset_nothing_changes(client, clean):
    x = explore(client, clean)
    r = x["results_json"]
    assert r["preset"] is None and r["methods"] == ["clustering", "anomaly"]
    assert not any("cardiometabolic" in w.lower() for w in r["warnings"])


def test_explicit_methods_are_respected_with_the_preset(client, clean):
    r = explore(client, clean, preset="cardiometabolic", methods=["clustering", "anomaly"])["results_json"]
    assert r["methods"] == ["clustering", "anomaly"]


def test_unknown_preset_is_refused(client, clean):
    assert client.post("/v1/explorations", headers=ADMIN, json={"dataset_id": clean, "preset": "nope"}).status_code == 422


def test_phenotypes_are_invisible_to_the_catalogue_registry_and_service_role(client, clean):
    explore(client, clean, preset="cardiometabolic")
    assert client.get("/v1/overview", headers=ADMIN).json()["counts"]["model_definitions"] == 14
    slugs = {m["slug"] for m in client.get("/v1/catalog", headers=ADMIN).json()}
    assert not any("phenotype" in s or "cardiometabolic" in s for s in slugs)
    assert client.get("/v1/models/versions", headers=ADMIN).json() == []
    key = client.post("/v1/admin/keys", json={"name": "svc-cp", "role": "service"}, headers=ADMIN).json()["key"]
    assert client.get("/v1/explorations", headers={"X-API-Key": key}).status_code == 403
