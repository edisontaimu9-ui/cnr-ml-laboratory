import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/sa.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import training  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
OW, ST = "school-age-overweight-risk", "school-age-stunting-risk"
# The child's own measurements and anything derived from them.
OW_LEAK = ["weight_kg", "height_cm", "bmi", "bmi_z", "haz", "stunted"]
ST_LEAK = ["height_cm", "haz", "knee_height_cm", "overweight"]


def school_csv(n=900, seed=41):
    rng = np.random.default_rng(seed)
    rows = ["age_months,sex,diet_diversity,sugary_drinks_week,screen_hours,activity_days,food_secure,maternal_bmi,maternal_height_cm,"
            "birth_weight_g,submission_date,weight_kg,height_cm,bmi,bmi_z,haz,knee_height_cm,waist_cm,overweight,stunted"]
    for i in range(n):
        age = int(rng.integers(61, 228)); sex = rng.choice(["m", "f"]); div = int(rng.integers(1, 10)); sugar = int(rng.integers(0, 15))
        screen = rng.uniform(0, 6); act = int(rng.integers(0, 8)); secure = rng.choice(["yes", "no"], p=[0.7, 0.3])
        mbmi = rng.normal(24, 4); mh = rng.normal(157, 6); bw = rng.normal(3000, 450)
        adip = 0.07 * (sugar - 7) + 0.25 * (screen - 3) - 0.15 * (act - 4) + 0.08 * (mbmi - 24) + rng.normal(0, 0.8)
        grow = -0.5 * (secure == "no") + 0.04 * (mh - 157) + 0.0003 * (bw - 3000) + 0.06 * (div - 5) + rng.normal(0, 0.8)
        height = 105 + 0.45 * (age - 61) + 6 * grow; weight = 18 + 0.15 * (age - 61) + 3 * adip + 0.3 * grow
        bmi = weight / (height / 100) ** 2
        rows.append(f"{age},{sex},{div},{sugar},{screen:.1f},{act},{secure},{mbmi:.1f},{mh:.1f},{bw:.0f},2026-{1 + i % 12:02d}-01,"
                    f"{weight:.1f},{height:.1f},{bmi:.1f},{adip:.2f},{grow:.2f},{height * 0.27:.1f},{50 + 3 * adip + rng.normal(0, 3):.1f},"
                    f"{'yes' if adip > 0.7 else 'no'},{'yes' if grow < -1.0 else 'no'}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ds(client):
    d = client.post("/v1/datasets", json={"name": "school-ds"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-SA"},
                    files={"file": ("s.csv", school_csv(), "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


def start(client, slug, ds, target, exclude, **over):
    r = client.post("/v1/experiments", headers=ADMIN, json={"model_slug": slug, "dataset_id": ds, "target": target,
                                                            "exclude": exclude, "n_estimators": 40, **over})
    assert r.status_code == 201, r.text
    return client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()


def test_catalogue_entries(client):
    cat = {m["slug"]: m for m in client.get("/v1/catalog", headers=ADMIN).json()}
    for slug, text in ((OW, "BMI-for-age"), (ST, "height-for-age")):
        assert cat[slug]["status"] == "planned" and cat[slug]["task_type"] == "classification"
        assert text in cat[slug]["label_source"] and "never be used as features" in cat[slug]["label_source"]
        assert "not re-implemented here" in cat[slug]["label_source"]


def test_overweight_trains_with_parent_and_birth_measurements(client, ds):
    # waist is allowed (with a warning); the others are the child's own measurements and must be excluded.
    e = start(client, OW, ds, "overweight", [*OW_LEAK, "knee_height_cm", "stunted"])
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert {"maternal_bmi", "maternal_height_cm", "birth_weight_g", "age_months", "submission_date"} & set(m["features"]["numeric"] + m["features"]["categorical"]) >= {
        "maternal_bmi", "maternal_height_cm", "birth_weight_g"}
    assert any("waist_cm" in w and "tracks the label closely" in w for w in m["warnings"])


def test_overweight_without_waist_has_no_body_size_warning(client, ds):
    e = start(client, OW, ds, "overweight", [*OW_LEAK, "knee_height_cm", "stunted", "waist_cm"])
    assert e["status"] == "completed", e["metrics_json"]
    assert not any("tracks the label closely" in w for w in e["metrics_json"]["warnings"])


@pytest.mark.parametrize("col", OW_LEAK)
def test_overweight_refuses_the_childs_own_measurements(client, ds, col):
    rest = [c for c in [*OW_LEAK, "knee_height_cm", "waist_cm"] if c != col]
    e = start(client, OW, ds, "overweight", rest)
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


def test_stunting_allows_weight_with_warning_and_parent_height(client, ds):
    e = start(client, ST, ds, "stunted", ["height_cm", "haz", "knee_height_cm", "overweight", "bmi", "bmi_z", "waist_cm"])
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert "maternal_height_cm" in m["features"]["numeric"] and "weight_kg" in m["features"]["numeric"]
    assert any("weight_kg" in w and "tracks height closely" in w for w in m["warnings"])


@pytest.mark.parametrize("col", ST_LEAK)
def test_stunting_refuses_height_and_stand_ins(client, ds, col):
    rest = [c for c in [*ST_LEAK, "weight_kg", "bmi", "bmi_z", "waist_cm"] if c != col]
    e = start(client, ST, ds, "stunted", rest)
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


def test_guards_are_per_model(client, ds):
    # weight is refused for overweight, only warned about for stunting
    e = start(client, OW, ds, "overweight", [c for c in OW_LEAK if c != "weight_kg"] + ["knee_height_cm", "waist_cm"])
    assert e["status"] == "failed" and "weight_kg" in e["metrics_json"]["error"]


def test_name_matching_rules():
    cfg = training.Config(target="t", task="classification", leakage_tokens=["bmi"], leakage_substrings=["bmi", "height", "weight"],
                          leakage_exempt=["mother", "maternal", "birth"])
    leaky = lambda c: training._is_leaky(c, cfg)  # noqa: E731
    assert leaky("bmi") and leaky("child_bmi") and leaky("Height (cm)") and leaky("weight_kg")
    assert not leaky("submission_date")                       # 'bmi' inside another word
    assert not leaky("maternal_height_cm") and not leaky("birth_weight_g") and not leaky("maternal_bmi")
