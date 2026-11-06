"""Automatic CNR training-data sync: the per-model registry, fetching, storing record-level data, guards, endpoints, and a real
training run on the synced data."""
import ast
import http.server
import json
import os
import tempfile
import threading
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/trainsync.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import training_sync as ts  # noqa: E402
from app.catalog import CATALOG  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Dataset, DatasetVersion  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
APP_DIR = Path(__file__).resolve().parent.parent / "app"
SECRET = "chakudya-admin-key-0123456789"

# The columns CNR's GET /surveillance/training-export returns for under5 (src/surveillance.js, trainingColumns("under5")).
CNR_COLUMNS = ["month", "district", "region", "sex", "age_band", "setting", "stunting", "wasting", "underweight", "muac_gam",
               "acute_malnutrition", "overweight", "anaemia_suspected", "low_dietary_diversity", "food_insecure_household",
               "recent_diarrhoea", "bilateral_oedema", "not_exclusively_breastfed", "any_undernutrition"]
DISTRICTS = [("Zomba", "Southern"), ("Lilongwe", "Central"), ("Mzimba", "Northern"), ("Mangochi", "Southern")]


def training_csv(seed=1, per_group=14, months=6, signal=True) -> bytes:
    """What CNR's training export returns: one row per screened child, coarse columns, WHO statuses, derived label.
    Groups are large (as CNR guarantees) and many rows are identical, as real coarse records are."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(months):
        for district, region in DISTRICTS:
            for sex in ("female", "male"):
                for band in ("0-5m", "6-23m", "24-59m"):
                    for _ in range(per_group):
                        fi, ld, dr = rng.random() < 0.4, rng.random() < 0.4, rng.random() < 0.25
                        p = 0.08 + (0.3 * fi + 0.2 * ld + 0.12 * dr + 0.1 * (band == "6-23m") if signal else 0.25)
                        stunted = rng.random() < p
                        wasted = rng.random() < p / 3
                        underw = rng.random() < p / 2
                        any_u = stunted or wasted or underw
                        rows.append({
                            "month": f"2025-{m + 1:02d}", "district": district, "region": region, "sex": sex, "age_band": band, "setting": "community",
                            "stunting": "moderate" if stunted else "normal", "wasting": "moderate" if wasted else "normal",
                            "underweight": "moderate" if underw else "normal", "muac_gam": "moderate" if wasted else "normal",
                            "acute_malnutrition": "moderate" if wasted else "normal", "overweight": "normal", "anaemia_suspected": "no",
                            "low_dietary_diversity": "yes" if ld else "no", "food_insecure_household": "yes" if fi else "no",
                            "recent_diarrhoea": "yes" if dr else "no", "bilateral_oedema": "no", "not_exclusively_breastfed": "no",
                            "any_undernutrition": "yes" if any_u else "no"})
    return pd.DataFrame(rows, columns=CNR_COLUMNS).to_csv(index=False).encode()


def settings(**o) -> Settings:
    base = dict(training_sync_enabled=True, surveillance_api_key=SECRET, cnr_api_base_url="http://127.0.0.1:9",
                training_sync_models="child-malnutrition-risk", environment="test")
    return Settings(_env_file=None, **{**base, **o})


@pytest.fixture(autouse=True)
def _reset():
    ts.LAST_RUN.clear()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def _started(client):
    """Starting the app creates the tables the sync writes to."""


# ───────────── the registry ─────────────

def test_every_catalogue_model_has_an_entry_and_the_gaps_are_explained():
    slugs = [slug for slug, *_ in CATALOG]
    assert set(ts.SOURCES) == set(slugs), "a catalogue model without a registry entry would silently never sync"
    fed = [s for s in slugs if ts.SOURCES[s].population_group]
    assert fed == ["child-malnutrition-risk"]
    for slug in slugs:
        src = ts.SOURCES[slug]
        if src.population_group is None:
            assert len(src.reason) > 40 and not src.target, slug     # says why, and names no pretend label
        else:
            assert src.target and src.population_group in {"under5", "school_age", "adolescent", "adult", "pregnant_lactating"}


def test_child_model_holds_out_the_columns_that_define_its_label():
    src = ts.SOURCES["child-malnutrition-risk"]
    assert src.target == "any_undernutrition" and src.target in CNR_COLUMNS
    # the label is built from these three; the rest are statuses taken from the same measurements
    assert {"stunting", "wasting", "underweight", "muac_gam", "acute_malnutrition", "bilateral_oedema", "overweight"} == set(src.exclude)
    assert set(src.exclude) <= set(CNR_COLUMNS)
    features = [c for c in CNR_COLUMNS if c != src.target and c not in src.exclude]
    assert {"food_insecure_household", "low_dietary_diversity", "recent_diarrhoea", "not_exclusively_breastfed", "age_band"} <= set(features)


def test_models_setting_is_parsed_and_unknown_slugs_refused():
    assert [m for m in ts.parse_models("all")] == [slug for slug, *_ in CATALOG]
    assert ts.parse_models("") == ts.parse_models("all")
    assert ts.parse_models("child-malnutrition-risk, child-malnutrition-risk") == ["child-malnutrition-risk"]
    with pytest.raises(ts.SyncError):
        ts.parse_models("child-malnutrition-risk,not-a-model")


# ───────────── configuration ─────────────

def test_config_problems_are_explained_without_the_key():
    assert "off" in ts.config_problem(settings(training_sync_enabled=False))
    assert "SURVEILLANCE_API_KEY" in ts.config_problem(settings(surveillance_api_key=""))
    assert "CNR_API_BASE_URL" in ts.config_problem(settings(cnr_api_base_url=""))
    assert "https" in ts.config_problem(settings(environment="production", cnr_api_base_url="http://x.example"))
    assert "Unknown model" in ts.config_problem(settings(training_sync_models="nope"))
    assert "has a CNR training source" in ts.config_problem(settings(training_sync_models="diabetes-risk"))
    assert "MONTHS" in ts.config_problem(settings(training_sync_months=99))
    assert ts.config_problem(settings()) is None
    assert ts.config_problem(settings(training_sync_models="all")) is None
    for s in (settings(), settings(surveillance_api_key="")):
        assert SECRET not in json.dumps(ts.status(s), default=str)


# ───────────── fetching (a real local HTTP server) ─────────────

class _CNR(http.server.BaseHTTPRequestHandler):
    seen: list = []
    mode = "ok"

    def do_GET(self):
        _CNR.seen.append((self.path, self.headers.get("Authorization")))
        if _CNR.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://evil.example/")
            self.end_headers()
        elif _CNR.mode in ("400", "403", "503"):
            self.send_response(int(_CNR.mode))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"error","message":"nope"}')
        else:
            body = training_csv(per_group=1, months=1) if _CNR.mode == "ok" else b"x" * (2 * 1024 * 1024)
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def cnr():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _CNR)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _CNR.seen, _CNR.mode = [], "ok"
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_fetch_sends_the_key_as_bearer_to_the_fixed_address_only(cnr):
    s = settings(cnr_api_base_url=cnr, training_sync_months=6)
    raw = ts.fetch_training_csv(s, ts.SOURCES["child-malnutrition-risk"])
    assert raw.startswith(b"month,district,region")
    path, auth = _CNR.seen[0]
    assert path == "/surveillance/training-export?population_group=under5&months=6&format=csv"
    assert auth == "Bearer " + SECRET


def test_fetch_refuses_redirects_errors_and_oversize_without_leaking_the_key(cnr):
    s = settings(cnr_api_base_url=cnr, training_fetch_max_mb=1)
    src = ts.SOURCES["child-malnutrition-risk"]
    for mode, text in (("redirect", "redirect"), ("400", "too large"), ("403", "not an admin key"), ("503", "sql/050"), ("big", "larger than 1 MB")):
        _CNR.mode = mode
        with pytest.raises(ts.SyncError) as e:
            ts.fetch_training_csv(s, src)
        assert text in str(e.value) and SECRET not in str(e.value), mode


# ───────────── storing ─────────────

def test_sync_stores_record_level_rows_and_keeps_repeated_rows():
    raw = training_csv(seed=1)
    n = len(pd.read_csv(pd.io.common.BytesIO(raw)))
    repeats = int(pd.read_csv(pd.io.common.BytesIO(raw)).duplicated().sum())
    assert repeats > 400, "the fixture must contain many identical rows, or this test proves nothing"
    out = ts.run_once(settings(), "manual", fetch=lambda *_: raw)
    r = out["results"][0]
    assert out["state"] == "completed" and r["status"] == "new_version" and r["version"] == 1, r
    assert r["rows"] == n, f"all {repeats} repeated coarse records must be kept; dropping them would erase the frequencies a model learns from"
    assert set(r["label_counts"]) == {"yes", "no"} and sum(r["label_counts"].values()) == n
    assert r["ready_to_train"] is True and "not_ready_reason" not in r
    with SessionLocal() as db:
        d = db.query(Dataset).filter(Dataset.name == "cnr-training-child-malnutrition-risk").one()
        assert d.sensitivity == "confidential" and d.created_by == "training-sync"
        assert "any_undernutrition" in d.description and "muac_gam" in d.description
        v = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == d.id).one()
        assert v.deidentified and v.row_count == n and v.approval_ref == "CNR de-identified training export (automatic)"
        assert v.report_json["duplicates_removed"] == 0
        assert v.report_json["k_anonymity"]["k_min"] >= 14       # group sizes survive intact


def test_unchanged_export_makes_no_new_version_and_a_changed_one_makes_version_two():
    raw = training_csv(seed=1)
    r = ts.run_once(settings(), "schedule", fetch=lambda *_: raw)["results"][0]
    assert r["status"] == "unchanged" and r["version"] == 1, r
    r = ts.run_once(settings(), "schedule", fetch=lambda *_: training_csv(seed=2))["results"][0]
    assert r["status"] == "new_version" and r["version"] == 2, r
    with SessionLocal() as db:
        d = db.query(Dataset).filter(Dataset.name == "cnr-training-child-malnutrition-risk").one()
        assert db.query(DatasetVersion).filter(DatasetVersion.dataset_id == d.id).count() == 2


def test_models_without_a_source_are_reported_and_cnr_is_not_called_for_them():
    calls = []

    def fetch(s, src):
        calls.append(src.population_group)
        return training_csv(seed=2)
    out = ts.run_once(settings(training_sync_models="all"), "manual", fetch=fetch)
    by = {r["model_slug"]: r for r in out["results"]}
    assert len(by) == len(CATALOG) and calls == ["under5"], "only the model with a source reaches CNR"
    assert by["child-malnutrition-risk"]["status"] == "unchanged"
    for slug, r in by.items():
        if slug != "child-malnutrition-risk":
            assert r["status"] == "no_source" and r["note"] == ts.SOURCES[slug].reason
    assert out["state"] == "completed" and out["summary"]["failed"] == 0 and out["summary"]["no_source"] == len(CATALOG) - 1


def test_too_little_data_is_stored_but_says_it_is_not_ready_to_train():
    tiny = pd.read_csv(pd.io.common.BytesIO(training_csv(seed=9))).head(30).to_csv(index=False).encode()
    with SessionLocal() as db:
        r = ts.sync_model(db, settings(), "child-malnutrition-risk", fetch=lambda *_: tiny)
    assert r["status"] == "new_version" and r["ready_to_train"] is False and "Fewer than 50" in r["not_ready_reason"], r
    one_class = pd.read_csv(pd.io.common.BytesIO(training_csv(seed=10)))
    one_class["any_undernutrition"] = "no"
    with SessionLocal() as db:
        r = ts.sync_model(db, settings(), "child-malnutrition-risk", fetch=lambda *_: one_class.to_csv(index=False).encode())
    assert r["ready_to_train"] is False and "one class" in r["not_ready_reason"], r


def test_an_export_from_an_outdated_cnr_is_refused_clearly():
    old = b"month,district,region,population_group,indicator,examined,cases,severe_cases\n2025-01,Zomba,Southern,under5,stunting,100,20,5\n"
    with SessionLocal() as db:
        r = ts.sync_model(db, settings(), "child-malnutrition-risk", fetch=lambda *_: old)
    assert r["status"] == "error" and "any_undernutrition" in r["error"] and "training-export" in r["error"], r
    only_label = pd.read_csv(pd.io.common.BytesIO(training_csv(seed=3)))[["any_undernutrition", "stunting", "wasting"]].to_csv(index=False).encode()
    with SessionLocal() as db:
        r = ts.sync_model(db, settings(), "child-malnutrition-risk", fetch=lambda *_: only_label)
    assert r["status"] == "error" and "no feature columns" in r["error"], r


def test_empty_export_and_fetch_errors_are_reported_per_model():
    header_only = (",".join(CNR_COLUMNS) + "\n").encode()
    with SessionLocal() as db:
        assert ts.sync_model(db, settings(), "child-malnutrition-risk", fetch=lambda *_: header_only)["status"] == "no_data"

    def boom(s, src):
        raise ts.SyncError("CNR answered HTTP 503: surveillance storage is not set up on CNR.")
    out = ts.run_once(settings(), "schedule", fetch=boom)
    assert out["state"] == "completed_with_errors" and out["results"][0]["status"] == "error" and "503" in out["results"][0]["error"]


def test_one_sync_at_a_time_and_unconfigured_runs_say_so():
    assert ts._lock.acquire()
    try:
        assert ts.run_once(settings(), "manual", fetch=lambda *_: training_csv()) is None
    finally:
        ts._lock.release()
    out = ts.run_once(settings(surveillance_api_key=""), "schedule")
    assert out["state"] == "not_configured" and "SURVEILLANCE_API_KEY" in out["error"]


# ───────────── the point of it: the synced data trains a model ─────────────

def _experiment(client, dataset_id, **over):
    src = ts.SOURCES["child-malnutrition-risk"]
    body = {"model_slug": "child-malnutrition-risk", "dataset_id": dataset_id, "target": src.target, "exclude": list(src.exclude),
            "n_estimators": 60, **over}
    r = client.post("/v1/experiments", json=body, headers=ADMIN)
    assert r.status_code == 201, r.text
    return client.get(f"/v1/experiments/{r.json()['id']}", headers=ADMIN).json()


def test_synced_data_trains_with_the_suggested_settings_and_finds_the_planted_signal(client):
    versions_before = len(client.get("/v1/models/versions", headers=ADMIN).json())    # other test modules share this database
    ts.run_once(settings(), "manual", fetch=lambda *_: training_csv(seed=5))
    with SessionLocal() as db:
        d = db.query(Dataset).filter(Dataset.name == "cnr-training-child-malnutrition-risk").one()
        dataset_id = d.id
    e = _experiment(client, dataset_id)
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert m["test"]["macro_f1"] > m["baseline"]["macro_f1"] + 0.03, "household and feeding context carries the planted signal"
    top = [f["feature"] for f in m["feature_importance"][:4]]
    assert any(x in top for x in ("food_insecure_household", "low_dietary_diversity", "recent_diarrhoea")), top
    for held_out in ts.SOURCES["child-malnutrition-risk"].exclude:
        assert held_out not in " ".join(f["feature"] for f in m["feature_importance"])
    assert not any("leak" in w.lower() for w in m["warnings"]), m["warnings"]
    cat = {c["slug"]: c["status"] for c in client.get("/v1/catalog", headers=ADMIN).json()}
    assert cat["child-malnutrition-risk"] == "in_development"      # a run never makes a model available
    assert len(client.get("/v1/models/versions", headers=ADMIN).json()) == versions_before, "a synced dataset and a run register nothing"


def test_without_the_default_exclusions_the_guard_flags_leakage(client):
    with SessionLocal() as db:
        dataset_id = db.query(Dataset).filter(Dataset.name == "cnr-training-child-malnutrition-risk").one().id
    e = _experiment(client, dataset_id, exclude=[])
    assert e["status"] == "completed", e["metrics_json"]
    m = e["metrics_json"]
    assert m["test"]["macro_f1"] > 0.97, "training on the components of the label simply reads the answer"
    assert m["feature_importance"][0]["feature"] in ("stunting", "underweight", "wasting")
    assert any("Suspiciously strong" in w for w in m["warnings"]), m["warnings"]


# ───────────── endpoints ─────────────

def key_for(client, role):
    r = client.post("/v1/admin/keys", json={"name": f"train-sync-{role}", "role": role}, headers=ADMIN)
    assert r.status_code == 201
    return {"Authorization": "Bearer " + r.json()["key"]}


def test_endpoints_are_admin_only_list_every_model_and_never_show_the_key(client, monkeypatch):
    import app.routers.sync as router
    monkeypatch.setattr(router, "get_settings", lambda: settings(training_sync_models="all"))
    monkeypatch.setattr(ts, "fetch_training_csv", lambda *_: training_csv(seed=5))
    assert client.get("/v1/sync/training").status_code == 401
    for role in ("viewer", "data_scientist", "service"):
        h = key_for(client, role)
        assert client.get("/v1/sync/training", headers=h).status_code == 403
        assert client.post("/v1/sync/training/run", headers=h).status_code == 403
    st = client.get("/v1/sync/training", headers=ADMIN)
    assert st.status_code == 200 and st.json()["configured"] is True and SECRET not in st.text
    models = {m["slug"]: m for m in st.json()["models"]}
    assert set(models) == {slug for slug, *_ in CATALOG}
    child = models["child-malnutrition-risk"]
    assert child["available"] and child["target"] == "any_undernutrition" and "stunting" in child["exclude"]
    assert child["dataset_name"] == "cnr-training-child-malnutrition-risk" and child["latest_version"] >= 1
    assert not models["diabetes-risk"]["available"] and "individual-level" in models["diabetes-risk"]["reason"]
    assert client.post("/v1/sync/training/run", headers=ADMIN).status_code == 202
    after = client.get("/v1/sync/training", headers=ADMIN).json()
    assert after["last_run"]["state"] == "completed" and after["last_run"]["trigger"] == "manual"
    assert {m["slug"]: m for m in after["models"]}["child-malnutrition-risk"]["last_result"]["status"] in ("unchanged", "new_version")
    audit = client.get("/v1/audit?limit=300", headers=ADMIN).text
    assert "training_sync.started" in audit and "training_sync.completed" in audit
    assert SECRET not in audit and "Zomba" not in audit


def test_manual_run_refuses_when_not_configured_or_already_running(client, monkeypatch):
    import app.routers.sync as router
    monkeypatch.setattr(router, "get_settings", lambda: settings(training_sync_enabled=False))
    r = client.post("/v1/sync/training/run", headers=ADMIN)
    assert r.status_code == 409 and "off" in r.json()["detail"]
    monkeypatch.setattr(router, "get_settings", lambda: settings())
    assert ts._lock.acquire()
    try:
        assert client.post("/v1/sync/training/run", headers=ADMIN).status_code == 409
    finally:
        ts._lock.release()


def test_catalogue_contract_with_cnr_is_unchanged(client):
    item = client.get("/v1/catalog/child-malnutrition-risk", headers=ADMIN).json()
    assert set(item) == {"slug", "name", "task_type", "description", "label_source", "status"}


# ───────────── isolation ─────────────

def test_the_sync_only_reads_and_the_analysis_code_never_imports_it():
    src = (APP_DIR / "training_sync.py").read_text() + (APP_DIR / "surveillance_sync.py").read_text()
    assert 'method="GET"' in src and "data=" not in src      # no body is ever sent to CNR
    for rel in ("exploration.py", "hotspot.py", "routers/hotspots.py", "routers/explorations.py", "serving.py", "routers/predict.py"):
        names = set()
        for node in ast.walk(ast.parse((APP_DIR / rel).read_text())):
            if isinstance(node, ast.ImportFrom):
                names |= {p for p in (node.module or "").split(".") if p} | {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                names |= {p for a in node.names for p in a.name.split(".")}
        assert "training_sync" not in names, rel


def test_the_sync_never_trains_registers_or_deploys():
    names = set()
    for node in ast.walk(ast.parse((APP_DIR / "training_sync.py").read_text())):
        if isinstance(node, ast.ImportFrom):
            names |= {p for p in (node.module or "").split(".") if p} | {a.name for a in node.names}
    assert not names & {"registry", "serving", "predict", "webhook", "experiments", "train", "run_experiment"}, names
