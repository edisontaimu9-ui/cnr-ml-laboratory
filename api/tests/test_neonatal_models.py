import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/nn.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
LBW, PGF = "low-birth-weight-risk", "preterm-growth-faltering"
LBW_LEAK = ["birth_weight_g", "sga_flag", "apgar_5min", "birth_length_cm"]
PGF_LEAK = ["discharge_weight_g", "discharge_z", "z_change", "weight_gain_velocity"]


def lbw_csv(n=800, seed=21):
    rng = np.random.default_rng(seed)
    rows = ["maternal_age,maternal_bmi,muac_cm,maternal_height_cm,parity,anc_visits,haemoglobin,gestational_age_weeks,"
            "birth_weight_g,sga_flag,apgar_5min,birth_length_cm,lbw"]
    for _ in range(n):
        age = int(rng.integers(15, 42)); bmi = rng.normal(22.5, 3.2); muac = bmi * 1.1 + rng.normal(0, 1.5)
        height = rng.normal(157, 6); parity = int(rng.integers(0, 6)); anc = int(rng.integers(0, 9)); hb = rng.normal(11, 1.4)
        risk = -0.25 * (bmi - 22.5) - 0.3 * (anc - 4) * 0.3 - 0.35 * (hb - 11) + 0.04 * abs(age - 26) + rng.normal(0, 0.9)
        ga = float(np.clip(39 - 0.7 * max(risk, 0) + rng.normal(0, 1.5), 28, 42))
        bw = float(2300 + 500 * (ga - 36) * 0.4 + 120 * (-risk) + rng.normal(0, 260) + 600)
        sga = "yes" if bw < 2400 and rng.random() < 0.5 else "no"
        rows.append(f"{age},{bmi:.1f},{muac:.1f},{height:.1f},{parity},{anc},{hb:.1f},{ga:.1f},{bw:.0f},{sga},"
                    f"{int(rng.integers(6, 10))},{bw / 70:.1f},{'yes' if bw < 2500 else 'no'}")
    return "\n".join(rows) + "\n"


def pgf_csv(n=500, seed=31):
    rng = np.random.default_rng(seed)
    rows = ["ga_weeks,birth_weight_g,birth_z,sex,multiple_birth,antenatal_steroids,sepsis_first_week,ventilation_days,feed_start_day,"
            "protein_g_kg_day_w1,discharge_weight_g,discharge_z,z_change,weight_gain_velocity,faltering"]
    for _ in range(n):
        ga = float(rng.uniform(26, 34)); z0 = rng.normal(0, 1)
        bw = 1000 + 150 * (ga - 26) + 260 * z0
        sepsis = rng.random() < 0.2; vent = int(max(0, rng.normal(8 - (ga - 26), 4))); feed = int(rng.integers(1, 7))
        prot = rng.normal(2.5, 0.6)
        drop = 0.35 + 0.18 * z0 + 0.5 * sepsis + 0.04 * vent + 0.08 * feed - 0.3 * (prot - 2.5) + rng.normal(0, 0.5)
        z1 = z0 - drop
        rows.append(f"{ga:.1f},{bw:.0f},{z0:.2f},{rng.choice(['m', 'f'])},{rng.choice(['yes', 'no'], p=[.2, .8])},{rng.choice(['yes', 'no'], p=[.7, .3])},"
                    f"{'yes' if sepsis else 'no'},{vent},{feed},{prot:.2f},{bw * 2:.0f},{z1:.2f},{-drop:.2f},{drop * 4:.1f},{'yes' if drop >= 1.0 else 'no'}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def upload(client, name, csv):
    d = client.post("/v1/datasets", json={"name": name}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-NN"},
                    files={"file": ("n.csv", csv, "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


@pytest.fixture(scope="module")
def lbw(client):
    return upload(client, "lbw-ds", lbw_csv())


@pytest.fixture(scope="module")
def pgf(client):
    return upload(client, "pgf-ds", pgf_csv())


def start(client, slug, ds, target, exclude, **over):
    r = client.post("/v1/experiments", headers=ADMIN, json={"model_slug": slug, "dataset_id": ds, "target": target,
                                                            "exclude": exclude, "n_estimators": 40, **over})
    assert r.status_code == 201, r.text
    return client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()


def test_catalogue_entries(client):
    cat = {m["slug"]: m for m in client.get("/v1/catalog", headers=ADMIN).json()}
    for slug, text in ((LBW, "2,500 g"), (PGF, "Fenton")):
        assert cat[slug]["status"] == "planned" and cat[slug]["task_type"] == "classification"
        assert text in cat[slug]["label_source"] and "never be used as features" in cat[slug]["label_source"]
    assert "not re-implemented here" in cat[PGF]["label_source"]


# ---- low birth weight ----

def test_lbw_trains_antenatally_and_warns_about_gestational_age(client, lbw):
    e = start(client, LBW, lbw, "lbw", LBW_LEAK)
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert "maternal_bmi" in m["features"]["numeric"] and "gestational_age_weeks" in m["features"]["numeric"]
    assert any("only known at or after delivery" in w and "gestational_age_weeks" in w for w in m["warnings"])


def test_lbw_without_gestational_age_has_no_delivery_warning(client, lbw):
    e = start(client, LBW, lbw, "lbw", [*LBW_LEAK, "gestational_age_weeks"])
    assert e["status"] == "completed", e["metrics_json"]
    assert not any("only known at or after delivery" in w for w in e["metrics_json"]["warnings"])


@pytest.mark.parametrize("col", LBW_LEAK)
def test_lbw_refuses_birth_measurements(client, lbw, col):
    e = start(client, LBW, lbw, "lbw", [c for c in LBW_LEAK if c != col])
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


# ---- preterm growth faltering ----

def test_pgf_trains_on_early_features_and_birth_weight_is_allowed(client, pgf):
    e = start(client, PGF, pgf, "faltering", PGF_LEAK)
    assert e["status"] == "completed", e["metrics_json"]
    f = e["metrics_json"]["features"]
    assert {"ga_weeks", "birth_weight_g", "birth_z", "ventilation_days"} <= set(f["numeric"])
    assert not any("only known at or after delivery" in w for w in e["metrics_json"]["warnings"])


@pytest.mark.parametrize("col", PGF_LEAK)
def test_pgf_refuses_later_growth_columns(client, pgf, col):
    e = start(client, PGF, pgf, "faltering", [c for c in PGF_LEAK if c != col])
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


def test_guards_are_per_model(client, lbw, pgf):
    # birth weight is a legitimate feature for faltering but not for LBW; discharge columns do not trip the LBW guard
    e = start(client, "diabetes-risk", lbw, "lbw", [])
    assert e["status"] == "completed"
    assert not any("only known at or after delivery" in w for w in e["metrics_json"]["warnings"])
