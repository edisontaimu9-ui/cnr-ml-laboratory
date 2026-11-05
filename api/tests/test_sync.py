"""Automatic CNR surveillance sync: fetching, storing, analysing, scheduling guards and the admin endpoints."""
import ast
import http.server
import json
import os
import tempfile
import threading
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/sync.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import surveillance_sync as sync  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Dataset, DatasetVersion, HotspotAnalysis  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
APP_DIR = Path(__file__).resolve().parent.parent / "app"
SECRET = "chakudya-admin-key-0123456789"


def export_csv(seed=1, months=12, areas=28) -> bytes:
    """What CNR's /surveillance/export returns: aggregated district-by-month counts."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(areas):
        base = 0.40 if i < 3 else 0.15
        for t in range(months):
            n = int(rng.integers(120, 300))
            rows.append((f"2025-{t + 1:02d}", f"District{i:02d}", "Southern", "under5", "stunting", n, int(rng.binomial(n, base)), 0))
    df = pd.DataFrame(rows, columns=["month", "district", "region", "population_group", "indicator", "examined", "cases", "severe_cases"])
    return df.to_csv(index=False).encode()


def settings(**o) -> Settings:
    base = dict(surveillance_sync_enabled=True, surveillance_api_key=SECRET, cnr_api_base_url="http://127.0.0.1:9",
                surveillance_sync_targets="under5:stunting", environment="test")
    return Settings(_env_file=None, **{**base, **o})


@pytest.fixture(autouse=True)
def _no_other_jobs(monkeypatch):
    monkeypatch.setattr(sync, "another_job_running", lambda db: False)
    sync.LAST_RUN.clear()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def _started(client):
    """Starting the app creates the tables the sync writes to."""


# ───────────── configuration ─────────────

def test_config_problems_are_explained_without_the_key():
    assert "off" in sync.config_problem(settings(surveillance_sync_enabled=False))
    assert "SURVEILLANCE_API_KEY" in sync.config_problem(settings(surveillance_api_key=""))
    assert "CNR_API_BASE_URL" in sync.config_problem(settings(cnr_api_base_url=""))
    assert "https" in sync.config_problem(settings(environment="production", cnr_api_base_url="http://x.example"))
    assert "Bad target" in sync.config_problem(settings(surveillance_sync_targets="elderly:stunting"))
    assert "MONTHS" in sync.config_problem(settings(surveillance_sync_months=99))
    assert sync.config_problem(settings()) is None
    for s in (settings(), settings(surveillance_api_key="")):
        assert SECRET not in json.dumps(sync.status(s), default=str)


def test_targets_are_parsed_and_deduplicated():
    assert sync.parse_targets("under5:stunting, under5:wasting,under5:stunting") == [("under5", "stunting"), ("under5", "wasting")]
    for bad in ("stunting", "under5:", "under5:a b", "adult:../x"):
        with pytest.raises(sync.SyncError):
            sync.parse_targets(bad)


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
        elif _CNR.mode in ("403", "503"):
            self.send_response(int(_CNR.mode))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"error","message":"nope"}')
        elif _CNR.mode == "json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
        else:
            body = export_csv(months=2, areas=3) if _CNR.mode == "ok" else b"x" * (2 * 1024 * 1024)
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
    s = settings(cnr_api_base_url=cnr, surveillance_sync_months=6)
    raw = sync.fetch_csv(s, "under5", "stunting")
    assert raw.startswith(b"month,district")
    path, auth = _CNR.seen[0]
    assert path == "/surveillance/export?population_group=under5&indicator=stunting&months=6&format=csv"
    assert auth == "Bearer " + SECRET


def test_fetch_refuses_redirects_errors_non_csv_and_oversize(cnr):
    s = settings(cnr_api_base_url=cnr, surveillance_fetch_max_mb=1)
    for mode, text in (("redirect", "redirect"), ("403", "not an admin key"), ("503", "not set up"),
                       ("json", "did not return a CSV"), ("big", "larger than 1 MB")):
        _CNR.mode = mode
        with pytest.raises(sync.SyncError) as e:
            sync.fetch_csv(s, "under5", "stunting")
        assert text in str(e.value) and SECRET not in str(e.value)


# ───────────── storing and analysing ─────────────

def test_sync_stores_a_deidentified_version_and_runs_the_hotspot_analysis():
    s = settings()
    out = sync.run_once(s, "manual", fetch=lambda *_: export_csv(seed=1))
    r = out["results"][0]
    assert out["state"] == "completed" and r["status"] == "new_version" and r["version"] == 1, r
    assert r["analysis"] == "completed", r
    with SessionLocal() as db:
        d = db.query(Dataset).filter(Dataset.name == "cnr-surveillance-under5-stunting").one()
        assert d.sensitivity == "internal" and d.created_by == "surveillance-sync"
        v = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == d.id).one()
        assert v.deidentified and v.row_count == 28 * 12
        h = db.get(HotspotAnalysis, r["analysis_id"])
        assert h.results_json["data"]["areas"] == 28 and h.results_json["data"]["periods"] == 12
        assert h.params_json["config"]["area_col"] == "district" and h.params_json["config"]["period_col"] == "month"


def test_unchanged_export_makes_no_new_version_and_no_new_analysis():
    s = settings()
    csv = export_csv(seed=1)
    r = sync.run_once(s, "schedule", fetch=lambda *_: csv)["results"][0]
    assert r["status"] == "unchanged" and r["version"] == 1 and "analysis" not in r, r
    with SessionLocal() as db:
        d = db.query(Dataset).filter(Dataset.name == "cnr-surveillance-under5-stunting").one()
        assert db.query(DatasetVersion).filter(DatasetVersion.dataset_id == d.id).count() == 1


def test_changed_export_makes_version_two():
    r = sync.run_once(settings(), "schedule", fetch=lambda *_: export_csv(seed=2))["results"][0]
    assert r["status"] == "new_version" and r["version"] == 2 and r["analysis"] == "completed", r


def test_analysis_can_be_switched_off_and_a_skipped_one_is_retried(monkeypatch):
    sync.run_once(settings(), "manual", fetch=lambda *_: export_csv(seed=3))   # version 3 gets its analysis
    monkeypatch.setattr(sync, "another_job_running", lambda db: True)
    r = sync.run_once(settings(), "manual", fetch=lambda *_: export_csv(seed=4))["results"][0]
    assert r["status"] == "new_version" and r["analysis"] == "skipped", r
    monkeypatch.setattr(sync, "another_job_running", lambda db: False)
    r = sync.run_once(settings(), "manual", fetch=lambda *_: export_csv(seed=4))["results"][0]
    assert r["status"] == "unchanged" and r["analysis"] == "completed", r     # same file, analysis now runs
    r = sync.run_once(settings(surveillance_sync_run_analysis=False), "manual", fetch=lambda *_: export_csv(seed=5))["results"][0]
    assert r["status"] == "new_version" and "analysis" not in r


def test_empty_export_and_fetch_errors_are_reported_per_target():
    header_only = b"month,district,region,population_group,indicator,examined,cases,severe_cases\n"

    def fetch(s, group, indicator):
        if indicator == "wasting":
            raise sync.SyncError("CNR answered HTTP 503: surveillance storage is not set up on CNR.")
        return header_only
    out = sync.run_once(settings(surveillance_sync_targets="under5:stunting,under5:wasting"), "schedule", fetch=fetch)
    assert [r["status"] for r in out["results"]] == ["no_data", "error"]
    assert out["state"] == "completed_with_errors" and out["summary"]["failed"] == 1
    assert "503" in out["results"][1]["error"]


def test_one_sync_at_a_time_and_unconfigured_runs_say_so():
    assert sync._lock.acquire()
    try:
        assert sync.run_once(settings(), "manual", fetch=lambda *_: export_csv()) is None
    finally:
        sync._lock.release()
    out = sync.run_once(settings(surveillance_api_key=""), "schedule")
    assert out["state"] == "not_configured" and "SURVEILLANCE_API_KEY" in out["error"]


# ───────────── endpoints ─────────────

def key_for(client, role):
    r = client.post("/v1/admin/keys", json={"name": f"sync-{role}", "role": role}, headers=ADMIN)
    assert r.status_code == 201
    return {"Authorization": "Bearer " + r.json()["key"]}


def test_endpoints_are_admin_only_and_never_show_the_key(client, monkeypatch):
    import app.routers.sync as router
    monkeypatch.setattr(router, "get_settings", lambda: settings())
    monkeypatch.setattr(sync, "fetch_csv", lambda *_: export_csv(seed=7))
    assert client.get("/v1/sync/surveillance").status_code == 401
    for role in ("viewer", "data_scientist", "service"):
        h = key_for(client, role)
        assert client.get("/v1/sync/surveillance", headers=h).status_code == 403
        assert client.post("/v1/sync/surveillance/run", headers=h).status_code == 403
    st = client.get("/v1/sync/surveillance", headers=ADMIN)
    assert st.status_code == 200 and st.json()["configured"] is True and st.json()["targets"] == ["under5:stunting"]
    assert SECRET not in st.text
    r = client.post("/v1/sync/surveillance/run", headers=ADMIN)
    assert r.status_code == 202
    after = client.get("/v1/sync/surveillance", headers=ADMIN).json()
    assert after["last_run"]["state"] == "completed" and after["last_run"]["trigger"] == "manual"
    audit = client.get("/v1/audit?limit=300", headers=ADMIN).text
    assert "surveillance_sync.started" in audit and "surveillance_sync.completed" in audit
    assert SECRET not in audit and "District00" not in audit


def test_manual_run_refuses_when_not_configured_or_already_running(client, monkeypatch):
    import app.routers.sync as router
    monkeypatch.setattr(router, "get_settings", lambda: settings(surveillance_sync_enabled=False))
    r = client.post("/v1/sync/surveillance/run", headers=ADMIN)
    assert r.status_code == 409 and "off" in r.json()["detail"]
    monkeypatch.setattr(router, "get_settings", lambda: settings())
    assert sync._lock.acquire()
    try:
        assert client.post("/v1/sync/surveillance/run", headers=ADMIN).status_code == 409
    finally:
        sync._lock.release()


# ───────────── isolation ─────────────

def test_hotspot_code_does_not_import_the_sync_and_the_sync_sends_nothing_individual():
    for rel in ("hotspot.py", "routers/hotspots.py"):
        names = set()
        for node in ast.walk(ast.parse((APP_DIR / rel).read_text())):
            if isinstance(node, ast.ImportFrom):
                names |= {p for p in (node.module or "").split(".") if p} | {a.name for a in node.names}
        assert "surveillance_sync" not in names, rel
    src = (APP_DIR / "surveillance_sync.py").read_text()
    assert "method=\"GET\"" in src and "data=" not in src      # it only reads; no body is ever sent to CNR
