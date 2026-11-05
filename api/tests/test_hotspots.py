import ast
import json
import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/hs.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import hotspot  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Experiment, HotspotAnalysis, utcnow  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
APP_DIR = Path(__file__).resolve().parent.parent / "app"
KEEP_XY = {"columns": [{"name": "lat", "action": "keep", "override_reason": "centre point of a district, aggregated data"},
                       {"name": "lon", "action": "keep", "override_reason": "centre point of a district, aggregated data"}]}


def synth(seed=1, n_areas=28, months=12, cluster=True, trend=0.0, rising=True):
    """Aggregated monthly counts. The four northernmost areas have double prevalence; area D00 rises 10% a month."""
    rng = np.random.default_rng(seed)
    lat, lon = -9.5 - rng.random(n_areas) * 7.5, 33 + rng.random(n_areas) * 2.5
    hot = list(np.argsort(-lat)[:4]) if cluster else []
    rows = []
    for i in range(n_areas):
        base = 0.20 * (2.0 if i in hot else 1.0)
        for t in range(months):
            n = int(rng.integers(150, 400))
            p = min(0.9, base * np.exp(trend * t + (0.10 * t if (i == 0 and rising) else 0)))
            rows.append((f"D{i:02d}", f"2025-{t + 1:02d}", int(rng.binomial(n, p)), n, lat[i], lon[i]))
    df = pd.DataFrame(rows, columns=["district", "month", "cases", "examined", "lat", "lon"])
    return df, {f"D{i:02d}" for i in hot}


def cfg(**o):
    return hotspot.Config(area_col="district", period_col="month", lat_col="lat", lon_col="lon", **o)


def upload(client, name, df, plan=None, ref="TEST-HS"):
    d = client.post("/v1/datasets", json={"name": name}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": ref, "plan": json.dumps(plan or {})},
                    files={"file": ("x.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def planted(client):
    return upload(client, "hs-planted", synth(seed=1)[0], KEEP_XY)


BODY = {"area_column": "district", "cases_column": "cases", "denominator_column": "examined", "period_column": "month",
        "latitude_column": "lat", "longitude_column": "lon"}


def analyse(client, ds, **over):
    r = client.post("/v1/hotspots", json={"dataset_id": ds, **BODY, **over}, headers=ADMIN)
    assert r.status_code == 201, r.text
    return client.get(f"/v1/hotspots/{r.json()['id']}", headers=ADMIN).json()   # background task already ran


# ---------- the statistics ----------

def test_planted_hotspots_are_found_and_clustering_is_detected():
    df, hot = synth(seed=1, rising=False)
    r = hotspot.run(df, cfg())
    flagged = {h["area"] for h in r["hotspots"]}
    assert hot <= flagged and not (flagged - hot)
    sp = r["spatial"]
    assert sp["global"]["verdict"] == "clustering" and sp["global"]["i"] > sp["global"]["expected_under_no_clustering"]
    labelled = {a["area"] for a in r["areas"] if a.get("cluster") == "high cluster"}
    assert labelled and labelled <= hot                                    # local clusters sit on real hotspots
    assert not any(a.get("cluster") == "low cluster" and a["area"] in hot for a in r["areas"])
    assert r["smoothing"]["method"] == "spatial_eb" and r["deployable"] is False and r["kind"] == "hotspot_analysis"
    assert all(len(a["centre"]) == 2 and round(a["centre"][0], 2) == a["centre"][0] for a in r["areas"])
    ranks = [a["rank"] for a in r["areas"]]
    assert ranks == list(range(1, len(ranks) + 1))
    assert all(a["status"] == "hotspot" for a in r["areas"] if a["area"] in hot)


def test_a_rising_area_is_detected_and_areas_that_move_differently_are_reported():
    df, _ = synth(seed=1)
    r = hotspot.run(df, cfg())
    rising = [a["area"] for a in r["areas"] if (a["trend"].get("direction") == "rising")]
    assert rising == ["D00"]
    t = r["trend"]
    assert t["unit"] == "month" and t["overall"]["rate_ratio_per_12_months"] is not None
    assert t["heterogeneity"]["p_value"] < 0.01
    assert [s["period"] for s in t["series"]][:2] == ["2025-01", "2025-02"] and len(t["series"]) == 12
    d0 = next(a for a in r["areas"] if a["area"] == "D00")["trend"]
    assert 1.07 < d0["rate_ratio_per_unit"] < 1.14                           # planted 10% a month
    assert any(h["rising"] for h in r["hotspots"] if h["area"] == "D00")


def test_a_real_overall_trend_is_recovered():
    df, _ = synth(seed=4, cluster=False, trend=-0.03, rising=False)
    t = hotspot.run(df, cfg())["trend"]
    assert t["overall"]["direction"] == "falling" and 0.95 < t["overall"]["rate_ratio_per_unit"] < 0.99
    lo, hi = t["overall"]["ci95"]
    assert lo < t["overall"]["rate_ratio_per_unit"] < hi


def test_no_structure_means_no_hotspots_no_trend_no_clustering():
    """Nothing planted. Across seeds the analysis must stay quiet, which is the honest-evaluation check."""
    hot, mor, trend = 0, 0, 0
    for s in range(25):
        r = hotspot.run(synth(seed=500 + s, cluster=False, trend=0.0, rising=False)[0], cfg(seed=s))
        hot += len(r["hotspots"])
        mor += r["spatial"]["global"]["verdict"] == "clustering"
        trend += r["trend"]["overall"]["direction"] != "no clear change"
    assert hot == 0 and mor <= 4 and trend <= 3
    assert r["overall"]["extra_variation_between_areas"] is False
    assert any("none is flagged" in w for w in r["warnings"]) and "pooled" in r["smoothing"]["description"]
    assert all(a["status"] == "not elevated" for a in r["areas"])


def test_extra_variation_switches_to_negative_binomial():
    df, _ = synth(seed=5, cluster=False, rising=False)
    rng = np.random.default_rng(1)
    df["cases"] = np.minimum(df["examined"], (df["cases"] * rng.gamma(4, 0.25, len(df))).round().astype(int))
    t = hotspot.run(df, cfg())["trend"]
    assert t["model"] == "negative_binomial" and t["dispersion"] > 5
    assert t["candidates"]["negative_binomial"]["aic"] < t["candidates"]["poisson"]["aic"]
    assert "negative binomial" in t["model_choice"]


def test_poisson_is_kept_when_counts_are_not_overdispersed():
    t = hotspot.run(synth(seed=6, cluster=False, rising=False)[0], cfg())["trend"]
    assert t["model"] == "poisson" and t["dispersion"] <= hotspot.OVERDISPERSION_LIMIT


def test_smoothing_pulls_small_areas_towards_the_rest():
    rows = [(f"A{i}", 200, 1000) for i in range(10)] + [("Small", 3, 10), ("Big", 400, 1000)]
    df = pd.DataFrame(rows, columns=["district", "cases", "examined"])
    r = hotspot.run(df, hotspot.Config(area_col="district"))
    by = {a["area"]: a for a in r["areas"]}
    overall = r["overall"]["prevalence"]
    assert r["smoothing"]["method"] == "global_eb"
    # Small raw rate is 30% on 10 children. After smoothing it stays near the overall rate with a wide interval.
    assert abs(by["Small"]["smoothed_prevalence"] - overall) < abs(0.3 - overall) / 2
    width = lambda a: a["interval95"][1] - a["interval95"][0]  # noqa: E731
    assert width(by["Small"]) > 2 * width(by["Big"]) and by["Small"]["low_precision"] and by["Small"]["status"] == "not elevated"
    assert by["Big"]["status"] == "hotspot"


def test_snapshot_without_periods_or_coordinates():
    df = synth(seed=1, rising=False)[0].groupby("district", as_index=False)[["cases", "examined"]].sum()
    r = hotspot.run(df, hotspot.Config(area_col="district"))
    assert r["trend"] is None and r["spatial"] is None and r["smoothing"]["method"] == "global_eb"
    assert any("No coordinates" in w for w in r["warnings"])
    assert all("cluster" not in a and "centre" not in a for a in r["areas"])


def test_few_periods_skip_the_trend_with_a_reason():
    df, _ = synth(seed=1, months=3)
    r = hotspot.run(df, cfg())
    assert r["trend"] is None and any("At least 4 are needed" in w for w in r["warnings"])


def test_small_counts_are_hidden_but_smoothed_prevalence_is_shown():
    df = synth(seed=1, rising=False)[0].groupby("district", as_index=False)[["cases", "examined"]].sum()
    df.loc[0, ["cases", "examined"]] = [3, 40]
    df.loc[1, ["cases", "examined"]] = [38, 40]            # fewer than 5 children WITHOUT the condition also identifies
    r = hotspot.run(df, hotspot.Config(area_col="district"))
    sup = [a for a in r["areas"] if a["suppressed"]]
    assert {a["area"] for a in sup} == {"D00", "D01"}
    for a in sup:
        assert a["cases"] is None and a["examined"] is None and a["raw_prevalence"] is None and a["smoothed_prevalence"] is not None
    assert any("hidden" in w for w in r["warnings"])


def test_period_totals_with_small_cells_are_hidden():
    rows = [(f"A{i}", f"2025-{m:02d}", 1 if i == 0 else 30, 2 if i == 0 else 100) for i in range(4) for m in range(1, 7)]
    df = pd.DataFrame(rows, columns=["district", "month", "cases", "examined"])
    r = hotspot.run(df, hotspot.Config(area_col="district", period_col="month"))
    assert r["trend"] is not None and not any(s.get("suppressed") for s in r["trend"]["series"])   # totals are large
    tiny = pd.DataFrame([(a, f"2025-{m:02d}", 1, 3) for a in ("A", "B", "C") for m in range(1, 6)], columns=df.columns)
    r2 = hotspot.run(tiny, hotspot.Config(area_col="district", period_col="month"))
    assert r2["trend"] is not None
    assert all(s.get("suppressed") and "cases" not in s for s in r2["trend"]["series"])
    assert all(s["prevalence"] is None and s["examined"] is None for s in r2["trend"]["series"])


def test_period_formats():
    base, _ = synth(seed=1, months=6, rising=False)
    as_year = base.assign(month=base["month"].str[-2:].astype(int) + 2019)
    assert hotspot.run(as_year, cfg())["trend"]["unit"] == "period"
    words = base.assign(month=base["month"].map(lambda s: "wk-" + s[-2:]))
    r = hotspot.run(words, cfg())
    assert any("alphabetically" in w for w in r["warnings"])
    dup = pd.concat([base, base.iloc[:5]])
    assert any("added together" in w for w in hotspot.run(dup, cfg())["warnings"])


def test_invalid_input_is_explained():
    ok = synth(seed=1, rising=False)[0]
    with pytest.raises(hotspot.HotspotError, match="Many rows are invalid"):
        hotspot.run(ok.assign(cases=ok["examined"] + 5), cfg())
    with pytest.raises(hotspot.HotspotError, match="At least 3 areas"):
        hotspot.run(ok[ok["district"].isin(["D00", "D01"])], cfg())
    with pytest.raises(hotspot.HotspotError, match="no cases"):
        hotspot.run(ok.assign(cases=0), cfg())
    with pytest.raises(hotspot.HotspotError, match="both a latitude and a longitude"):
        hotspot.run(ok, hotspot.Config(area_col="district", lat_col="lat"))
    with pytest.raises(hotspot.HotspotError, match="different column"):
        hotspot.run(ok, hotspot.Config(area_col="district", cases_col="cases", denominator_col="cases"))
    with pytest.raises(hotspot.HotspotError, match="Spatial smoothing needs"):
        hotspot.run(ok.drop(columns=["lat", "lon"]), hotspot.Config(area_col="district", period_col="month", smoothing="spatial"))
    one_bad = ok.copy()
    one_bad.loc[0, "cases"] = -1
    assert any("dropped" in w for w in hotspot.run(one_bad, cfg())["warnings"])
    few = ok[ok["district"].isin([f"D{i:02d}" for i in range(6)])]
    r = hotspot.run(few, cfg())
    assert r["spatial"] is None and any("at least 8" in w for w in r["warnings"]) and any("Only 6 areas" in w for w in r["warnings"])


def test_results_are_deterministic_and_json_safe():
    df, _ = synth(seed=1)
    a, b = hotspot.run(df, cfg(seed=7)), hotspot.run(df, cfg(seed=7))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    json.dumps(a, allow_nan=False)                                        # no NaN or infinity anywhere


# ---------- the API ----------

def test_api_runs_an_analysis_and_summarises_it(client, planted):
    x = analyse(client, planted)
    assert x["status"] == "completed", x["results_json"]
    r, s = x["results_json"], x["summary"]
    assert x["kind"] == "hotspot_analysis" and x["deployable"] is False and r["deployable"] is False
    assert "can never be registered" in r["notice"]
    assert s["areas"] == 28 and s["periods"] == 12 and s["hotspots"] >= 4
    assert s["trend"]["direction"] in ("rising", "falling", "no clear change") and s["spatial"]["verdict"] == "clustering"
    assert client.get("/v1/hotspots", headers=ADMIN).json()[0]["id"] == x["id"]
    assert client.get("/v1/overview", headers=ADMIN).json()["counts"]["hotspot_analyses"] >= 1
    assert r["limitations"] and r["settings"]["latitude"] == "lat"


def test_api_without_coordinates_or_periods(client, planted):
    x = analyse(client, planted, latitude_column=None, longitude_column=None, period_column=None)
    assert x["status"] == "completed" and x["results_json"]["trend"] is None and x["results_json"]["spatial"] is None


def test_api_failures_are_explained(client):
    text = pd.DataFrame({"district": ["A", "B", "C", "D"], "cases": ["x", "y", "z", "w"], "examined": [10, 20, 30, 40]})
    ds = upload(client, "hs-text", text)
    x = analyse(client, ds, period_column=None, latitude_column=None, longitude_column=None)
    assert x["status"] == "failed" and x["results_json"]["error"] == "No usable rows after cleaning."


def test_validation(client, planted):
    post = lambda **o: client.post("/v1/hotspots", json={"dataset_id": planted, **BODY, **o}, headers=ADMIN)  # noqa: E731
    assert post(area_column="nope").status_code == 422
    assert post(cases_column="nope").status_code == 422
    assert post(period_column="nope").status_code == 422
    assert post(longitude_column=None).status_code == 422                       # latitude without longitude
    assert post(cases_column="examined").status_code == 422                     # same column twice
    assert post(latitude_column=None, longitude_column=None, smoothing="spatial").status_code == 422
    assert post(rr_threshold=9).status_code == 422 and post(exceedance_cutoff=1.0).status_code == 422
    assert post(k_neighbours=1).status_code == 422 and post(permutations=10).status_code == 422
    assert post(smoothing="magic").status_code == 422
    assert client.post("/v1/hotspots", json={"dataset_id": 99999, **BODY}, headers=ADMIN).status_code == 404
    assert client.get("/v1/hotspots/99999", headers=ADMIN).status_code == 404


def test_coordinates_need_a_recorded_reason_to_be_kept(client):
    d = client.post("/v1/datasets", json={"name": "hs-xy-default"}, headers=ADMIN).json()
    df = synth(seed=1)[0]
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-HS"},
                    files={"file": ("x.csv", df.to_csv(index=False), "text/csv")})
    assert r.status_code == 201
    cols = {c["name"] for c in client.get(f"/v1/datasets/{d['id']}", headers=ADMIN).json()["versions"][0]["schema_info"]["columns"]}
    assert "lat" not in cols and "lon" not in cols                               # dropped by default, as for any address-like column
    bad = client.post("/v1/hotspots", json={"dataset_id": d["id"], **BODY}, headers=ADMIN)
    assert bad.status_code == 422 and "lat" in bad.json()["detail"]


def test_permissions(client, planted):
    viewer = client.post("/v1/admin/keys", json={"name": "viewer-hs", "role": "viewer"}, headers=ADMIN).json()["key"]
    service = client.post("/v1/admin/keys", json={"name": "service-hs", "role": "service"}, headers=ADMIN).json()["key"]
    reviewer = client.post("/v1/admin/keys", json={"name": "rev-hs", "role": "clinical_reviewer"}, headers=ADMIN).json()["key"]
    scientist = client.post("/v1/admin/keys", json={"name": "ds-hs", "role": "data_scientist"}, headers=ADMIN).json()["key"]
    body = {"dataset_id": planted, **BODY}
    assert client.get("/v1/hotspots", headers={"X-API-Key": viewer}).status_code == 200
    assert client.get("/v1/hotspots", headers={"X-API-Key": reviewer}).status_code == 200
    assert client.post("/v1/hotspots", json=body, headers={"X-API-Key": viewer}).status_code == 403
    assert client.post("/v1/hotspots", json=body, headers={"X-API-Key": reviewer}).status_code == 403
    assert client.get("/v1/hotspots", headers={"X-API-Key": service}).status_code == 403     # CNR's machine key sees none of it
    assert client.post("/v1/hotspots", json=body, headers={"X-API-Key": service}).status_code == 403
    assert client.get("/v1/hotspots").status_code == 401
    assert client.post("/v1/hotspots", json=body, headers={"X-API-Key": scientist}).status_code == 201


def test_an_analysis_can_never_become_a_model(client, planted):
    root = Path(get_settings().dataset_storage_dir)

    def snapshot():
        ov = client.get("/v1/overview", headers=ADMIN).json()
        return {"models": ov["counts"]["model_versions"], "deployed": ov["counts"]["deployed_models"], "experiments": ov["counts"]["experiments"],
                "catalog": {c["slug"]: c["status"] for c in client.get("/v1/catalog", headers=ADMIN).json()},
                "versions": client.get("/v1/models/versions", headers=ADMIN).json(),
                "cnr": client.get("/v1/integration/cnr/models", headers=ADMIN).json(),
                "model_files": sorted(str(p) for p in root.rglob("*.joblib")) if root.exists() else []}

    before = snapshot()
    x = analyse(client, planted)
    assert x["deployable"] is False and snapshot() == before
    text = json.dumps(x)
    assert "artifact" not in text and "local://" not in text and ".joblib" not in text
    with SessionLocal() as db:
        assert db.get(HotspotAnalysis, x["id"]) is not None
        assert not hasattr(HotspotAnalysis, "model_definition_id") and not hasattr(HotspotAnalysis, "artifact_uri")
    audit = client.get("/v1/audit?limit=300", headers=ADMIN).text
    assert "hotspot.created" in audit and "hotspot.completed" in audit
    assert "D05" not in audit and "D00" not in audit                              # area names never go to the audit log


def test_hotspot_code_is_isolated_from_registry_serving_and_cnr():
    forbidden = {"registry", "serving", "predict", "webhook", "integration", "joblib", "pickle", "catalog"}
    for rel in ("hotspot.py", "routers/hotspots.py"):
        src = (APP_DIR / rel).read_text()
        names = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names |= {part for a in node.names for part in a.name.split(".")}
            elif isinstance(node, ast.ImportFrom):
                names |= {part for part in (node.module or "").split(".") if part} | {a.name for a in node.names}
        assert not (names & forbidden), f"{rel} imports {sorted(names & forbidden)}"
        assert "save_bytes" not in src and "ModelVersion" not in src.replace("DatasetVersion", "")
    reg = (APP_DIR / "routers/registry.py").read_text()
    assert "Hotspot" not in reg and "hotspots" not in reg


def test_one_heavy_job_at_a_time_across_all_kinds(client, planted):
    with SessionLocal() as db:
        db.add(HotspotAnalysis(status="running", created_at=utcnow()))
        db.commit()
    body = {"dataset_id": planted, **BODY}
    assert client.post("/v1/hotspots", json=body, headers=ADMIN).status_code == 409
    assert client.post("/v1/explorations", json={"dataset_id": planted}, headers=ADMIN).status_code == 409
    with SessionLocal() as db:
        for h in db.query(HotspotAnalysis).filter(HotspotAnalysis.status == "running"):
            h.status = "failed"
        db.commit()
        db.add(Experiment(model_definition_id=1, framework="xgboost", status="running", created_at=utcnow()))
        db.commit()
    assert client.post("/v1/hotspots", json=body, headers=ADMIN).status_code == 409
    with SessionLocal() as db:
        for e in db.query(Experiment).filter(Experiment.status == "running"):
            e.status = "failed"
        db.commit()
