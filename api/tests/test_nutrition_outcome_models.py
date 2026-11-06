import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/no.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import training  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
RF, LS, RA, WL = ("refeeding-syndrome-risk", "malnutrition-prolonged-stay-risk", "malnutrition-readmission-risk", "weight-loss-trajectory")

# Columns each model must refuse, and columns that are refused only because they do not say when they were measured.
RF_LEAK = ["phosphate_day3", "potassium_nadir", "electrolyte_replacement", "refeeding_hypophosphataemia", "d2_magnesium"]
RF_UNTIMED = ["phosphate", "serum_potassium", "hypokalaemia"]
LS_LEAK = ["length_of_stay_days", "los", "discharge_weight_kg", "readmitted_30d", "died_in_hospital", "hospital_complication"]
RA_LEAK = ["readmitted_days", "followup_weight_kg", "attended_followup_visit", "died_after_discharge", "emergency_visit_after"]
WL_LEAK = ["followup_weight_kg", "weight_change_pct", "weight_velocity", "final_weight_kg", "loss_flag"]
WL_UNTIMED = ["weight_3m_kg", "muac_cm", "bmi"]


def _yn(rng, p):
    return "yes" if rng.random() < p else "no"


def refeeding_csv(n=700, seed=51):
    rng = np.random.default_rng(seed)
    rows = ["age,sex,baseline_bmi,weight_loss_pct_prior_3m,days_poor_intake,alcohol_use,chemotherapy,diuretic_use,baseline_phosphate,"
            "baseline_potassium,baseline_magnesium,feeding_route,starting_kcal_kg,thiamine_prophylaxis,"
            "phosphate,serum_potassium,hypokalaemia,phosphate_day3,potassium_nadir,electrolyte_replacement,refeeding_hypophosphataemia,d2_magnesium,rfs"]
    for _ in range(n):
        age = int(rng.integers(18, 85)); bmi = rng.normal(19, 4); loss = max(0, rng.normal(8, 7)); intake = int(max(0, rng.normal(6, 5)))
        alc = _yn(rng, .15); chemo = _yn(rng, .2); diur = _yn(rng, .25)
        phos = rng.normal(1.0, 0.25); k = rng.normal(4.0, 0.5); mg = rng.normal(0.8, 0.1)
        route = rng.choice(["oral", "enteral", "parenteral"], p=[.5, .35, .15]); start = rng.normal(20, 6)
        risk = (-0.18 * (bmi - 19) + 0.08 * (loss - 8) + 0.12 * (intake - 6) + 0.6 * (alc == "yes") - 1.8 * (phos - 1.0)
                - 0.6 * (k - 4.0) + 0.5 * (route == "parenteral") + rng.normal(0, 0.8) - 1.6)
        rfs = risk > 0
        rows.append(f"{age},{rng.choice(['m', 'f'])},{bmi:.1f},{loss:.1f},{intake},{alc},{chemo},{diur},{phos:.2f},{k:.1f},{mg:.2f},{route},{start:.0f},"
                    f"{_yn(rng, .4)},{phos:.2f},{k:.1f},{'yes' if k < 3.5 else 'no'},{phos - 0.4 * rfs:.2f},{k - 0.7 * rfs:.1f},"
                    f"{'yes' if rfs else 'no'},{'yes' if rfs and rng.random() < .8 else 'no'},{mg - 0.1 * rfs:.2f},{'yes' if rfs else 'no'}")
    return "\n".join(rows) + "\n"


def stay_csv(n=800, seed=52):
    rng = np.random.default_rng(seed)
    rows = ["age,sex,admission_dx_group,hiv_status,admission_bmi,muac_cm,must_score,albumin,recent_weight_loss_pct,ward,dietitian_review,"
            "length_of_stay_days,los,discharge_weight_kg,readmitted_30d,died_in_hospital,hospital_complication,prolonged"]
    for _ in range(n):
        age = int(rng.integers(18, 90)); dx = rng.choice(["infection", "cancer", "surgical", "cardiac"]); hiv = rng.choice(["pos", "neg"], p=[.2, .8])
        bmi = rng.normal(21, 4.5); muac = bmi * 1.2 + rng.normal(0, 1.5); must = int(np.clip(rng.normal(1, 1.2), 0, 4))
        alb = rng.normal(34, 6); loss = max(0, rng.normal(5, 6)); ward = rng.choice(["a", "b", "c"])
        z = 0.03 * (age - 50) + 0.5 * must - 0.06 * (alb - 34) + 0.04 * loss + 0.5 * (hiv == "pos") + 0.6 * (dx == "cancer") + rng.normal(0, 0.8)
        los = float(np.clip(6 + 3.5 * z + rng.normal(0, 1.5), 1, 60)); long_ = los > 8
        rows.append(f"{age},{rng.choice(['m', 'f'])},{dx},{hiv},{bmi:.1f},{muac:.1f},{must},{alb:.1f},{loss:.1f},{ward},{_yn(rng, .5)},"
                    f"{los:.0f},{los:.0f},{55 - loss * 0.1:.1f},{_yn(rng, .1)},{_yn(rng, .05)},{_yn(rng, .2 if long_ else .05)},{'yes' if long_ else 'no'}")
    return "\n".join(rows) + "\n"


def readmit_csv(n=800, seed=53):
    rng = np.random.default_rng(seed)
    rows = ["age,sex,diagnosis_group,hiv_status,previous_admissions,length_of_stay_days,discharge_bmi,discharge_weight_kg,discharge_muac,food_secure,"
            "distance_km,discharge_supplement,readmitted_days,followup_weight_kg,attended_followup_visit,died_after_discharge,emergency_visit_after,readmitted"]
    for _ in range(n):
        age = int(rng.integers(18, 90)); dx = rng.choice(["infection", "cancer", "surgical", "cardiac"]); hiv = rng.choice(["pos", "neg"], p=[.2, .8])
        prev = int(rng.poisson(1.0)); los = int(max(1, rng.normal(8, 4))); bmi = rng.normal(20, 4); secure = rng.choice(["yes", "no"], p=[.65, .35])
        dist = rng.uniform(1, 60)
        z = (0.35 * prev - 0.12 * (bmi - 20) + 0.04 * (los - 8) + 0.5 * (secure == "no") + 0.012 * dist + 0.4 * (hiv == "pos") + rng.normal(0, 0.9) - 1.4)
        re = z > 0
        rows.append(f"{age},{rng.choice(['m', 'f'])},{dx},{hiv},{prev},{los},{bmi:.1f},{bmi * 2.8:.1f},{bmi * 1.2:.1f},{secure},{dist:.0f},"
                    f"{_yn(rng, .3)},{int(rng.integers(1, 30)) if re else 0},{bmi * 2.8:.1f},{_yn(rng, .5)},{_yn(rng, .03)},{'yes' if re else 'no'},"
                    f"{'yes' if re else 'no'}")
    return "\n".join(rows) + "\n"


def weightloss_csv(n=800, seed=54):
    rng = np.random.default_rng(seed)
    rows = ["age,sex,baseline_weight_kg,baseline_bmi,weight_loss_prior_6m_pct,cancer_stage,hiv_art,appetite_score,nausea,food_secure,nutrition_counselling,"
            "weight_3m_kg,muac_cm,bmi,followup_weight_kg,final_weight_kg,weight_change_pct,weight_velocity,loss_flag,significant_loss"]
    for _ in range(n):
        age = int(rng.integers(18, 85)); w0 = rng.normal(62, 11); bmi = w0 / 1.65 ** 2; prior = max(0, rng.normal(4, 5))
        stage = int(rng.integers(1, 5)); art = rng.choice(["yes", "no"], p=[.3, .7]); app = int(rng.integers(1, 6)); nausea = _yn(rng, .3)
        secure = rng.choice(["yes", "no"], p=[.7, .3])
        z = 0.2 * prior + 0.45 * (stage - 2.5) - 0.4 * (app - 3) + 0.5 * (nausea == "yes") + 0.4 * (secure == "no") + rng.normal(0, 0.9) - 0.6
        pct = float(np.clip(1 + 2.4 * z + rng.normal(0, 1.2), -3, 25)); sig = pct >= 5
        w3 = w0 * (1 - pct / 100)
        rows.append(f"{age},{rng.choice(['m', 'f'])},{w0:.1f},{bmi:.1f},{prior:.1f},{stage},{art},{app},{nausea},{secure},{_yn(rng, .4)},"
                    f"{w3:.1f},{bmi * 1.2:.1f},{w3 / 1.65 ** 2:.1f},{w3:.1f},{w3:.1f},{-pct:.1f},{-pct / 3:.2f},{'yes' if sig else 'no'},{'yes' if sig else 'no'}")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _upload(client, name, csv):
    d = client.post("/v1/datasets", json={"name": name}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-NO"},
                    files={"file": ("d.csv", csv, "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


@pytest.fixture(scope="module")
def rf_ds(client):
    return _upload(client, "refeeding-ds", refeeding_csv())


@pytest.fixture(scope="module")
def ls_ds(client):
    return _upload(client, "stay-ds", stay_csv())


@pytest.fixture(scope="module")
def ra_ds(client):
    return _upload(client, "readmit-ds", readmit_csv())


@pytest.fixture(scope="module")
def wl_ds(client):
    return _upload(client, "weightloss-ds", weightloss_csv())


def start(client, slug, ds, target, exclude, **over):
    r = client.post("/v1/experiments", headers=ADMIN, json={"model_slug": slug, "dataset_id": ds, "target": target,
                                                            "exclude": exclude, "n_estimators": 40, **over})
    assert r.status_code == 201, r.text
    return client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()


def feats(e):
    f = e["metrics_json"]["features"]
    return set(f["numeric"] + f["categorical"])


def test_catalogue_entries(client):
    cat = {m["slug"]: m for m in client.get("/v1/catalog", headers=ADMIN).json()}
    expect = {RF: "refeeding syndrome", LS: "length of stay", RA: "readmission", WL: "weight change"}
    for slug, text in expect.items():
        assert cat[slug]["status"] == "planned" and cat[slug]["task_type"] == "classification"
        assert text in cat[slug]["label_source"]
        assert "never be used as features" in cat[slug]["label_source"]
        assert "never" in cat[slug]["label_source"].split("A positive result means")[1]


# ---------- refeeding syndrome ----------

def test_refeeding_trains_on_baseline_electrolytes_and_warns_about_feeding_decisions(client, rf_ds):
    bad = [*RF_LEAK, *RF_UNTIMED]
    e = start(client, RF, rf_ds, "rfs", bad)
    assert e["status"] == "completed", e["metrics_json"]
    assert {"baseline_phosphate", "baseline_potassium", "baseline_magnesium"} <= feats(e)
    assert any("starting_kcal_kg" in w and "fed more slowly" in w for w in e["metrics_json"]["warnings"])


def test_refeeding_without_feeding_decisions_has_no_confounding_warning(client, rf_ds):
    e = start(client, RF, rf_ds, "rfs", [*RF_LEAK, *RF_UNTIMED, "starting_kcal_kg", "thiamine_prophylaxis"])
    assert e["status"] == "completed", e["metrics_json"]
    assert not any("fed more slowly" in w for w in e["metrics_json"]["warnings"])


@pytest.mark.parametrize("col", RF_LEAK)
def test_refeeding_refuses_values_after_feeding_and_treatment(client, rf_ds, col):
    rest = [c for c in [*RF_LEAK, *RF_UNTIMED] if c != col]
    e = start(client, RF, rf_ds, "rfs", rest)
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


@pytest.mark.parametrize("col", RF_UNTIMED)
def test_refeeding_refuses_electrolytes_that_do_not_say_when_they_were_measured(client, rf_ds, col):
    rest = [c for c in [*RF_LEAK, *RF_UNTIMED] if c != col]
    e = start(client, RF, rf_ds, "rfs", rest)
    err = e["metrics_json"]["error"]
    assert e["status"] == "failed" and col in err and "does not say when it was measured" in err and "baseline_phosphate" in err


# ---------- prolonged stay ----------

def test_prolonged_stay_trains_at_admission_and_warns_about_in_stay_nutrition_care(client, ls_ds):
    e = start(client, LS, ls_ds, "prolonged", LS_LEAK)
    assert e["status"] == "completed", e["metrics_json"]
    assert {"admission_bmi", "must_score", "albumin"} <= feats(e)
    assert any("dietitian_review" in w and "takes time" in w for w in e["metrics_json"]["warnings"])


@pytest.mark.parametrize("col", LS_LEAK)
def test_prolonged_stay_refuses_stay_discharge_and_outcome_columns(client, ls_ds, col):
    rest = [c for c in LS_LEAK if c != col]
    e = start(client, LS, ls_ds, "prolonged", rest)
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


# ---------- readmission ----------

def test_readmission_allows_stay_and_discharge_measurements(client, ra_ds):
    # known at discharge, so allowed here even though the stay model refuses them
    e = start(client, RA, ra_ds, "readmitted", RA_LEAK)
    assert e["status"] == "completed", e["metrics_json"]
    assert {"length_of_stay_days", "discharge_bmi", "discharge_weight_kg", "previous_admissions"} <= feats(e)
    assert any("discharge_supplement" in w and "look low-risk" in w for w in e["metrics_json"]["warnings"])


@pytest.mark.parametrize("col", RA_LEAK)
def test_readmission_refuses_anything_after_discharge(client, ra_ds, col):
    rest = [c for c in RA_LEAK if c != col]
    e = start(client, RA, ra_ds, "readmitted", rest)
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


def test_guards_are_per_model_stay_versus_readmission(client, ls_ds, ra_ds):
    # length of stay is the label of one model and a legitimate feature of the other
    e = start(client, LS, ls_ds, "prolonged", [c for c in LS_LEAK if c != "length_of_stay_days"])
    assert e["status"] == "failed" and "length_of_stay_days" in e["metrics_json"]["error"]
    e = start(client, RA, ra_ds, "readmitted", RA_LEAK)
    assert e["status"] == "completed", e["metrics_json"]


# ---------- weight-loss trajectory ----------

def test_weight_loss_trains_on_baseline_and_prior_loss(client, wl_ds):
    e = start(client, WL, wl_ds, "significant_loss", [*WL_LEAK, *WL_UNTIMED])
    assert e["status"] == "completed", e["metrics_json"]
    assert {"baseline_weight_kg", "baseline_bmi", "weight_loss_prior_6m_pct"} <= feats(e)
    assert any("nutrition_counselling" in w and "after baseline" in w for w in e["metrics_json"]["warnings"])


@pytest.mark.parametrize("col", WL_LEAK)
def test_weight_loss_refuses_follow_up_and_change_columns(client, wl_ds, col):
    rest = [c for c in [*WL_LEAK, *WL_UNTIMED] if c != col]
    e = start(client, WL, wl_ds, "significant_loss", rest)
    assert e["status"] == "failed" and "cannot be a feature" in e["metrics_json"]["error"]


@pytest.mark.parametrize("col", WL_UNTIMED)
def test_weight_loss_refuses_measures_that_do_not_say_when_they_were_taken(client, wl_ds, col):
    rest = [c for c in [*WL_LEAK, *WL_UNTIMED] if c != col]
    e = start(client, WL, wl_ds, "significant_loss", rest)
    err = e["metrics_json"]["error"]
    assert e["status"] == "failed" and col in err and "does not say when it was taken" in err


def test_models_without_timing_rules_are_unaffected(client):
    cfg = training.Config(target="t", task="classification")
    assert not training._untimed("weight_kg", cfg) and not training._untimed("phosphate", cfg)


def test_timing_rule_matching():
    cfg = training.Config(target="t", task="classification", timing_required=["weight", "po4"], timing_markers=["baseline", "prior", "pre"])
    untimed = lambda c: training._untimed(c, cfg)  # noqa: E731
    assert untimed("weight_kg") and untimed("weight_3m") and untimed("PO4")
    assert not untimed("baseline_weight_kg") and not untimed("weight_loss_prior_6m_pct") and not untimed("pre_feeding_po4")
    assert not untimed("age") and not untimed("appetite_score")
    assert untimed("weight_preoperative_kg") is True        # 'pre' is a short marker: whole words only, so this stays refused
