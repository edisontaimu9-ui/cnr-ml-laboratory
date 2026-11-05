import ast
import json
import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/xp.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Experiment, Exploration, utcnow  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
APP_DIR = Path(__file__).resolve().parent.parent / "app"


def upload(client, name, csv, ref="TEST-XP"):
    d = client.post("/v1/datasets", json={"name": name}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": ref},
                    files={"file": ("x.csv", csv, "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


def blobs_csv(per=150, seed=3):
    """Two clearly separated groups in three numeric columns, plus a category."""
    rng = np.random.default_rng(seed)
    rows = ["a,b,c,district"]
    for centre, lab in ((0.0, "Zomba"), (7.0, "Blantyre")):
        for _ in range(per):
            v = rng.normal(centre, 1.0, 3)
            rows.append(f"{v[0]:.3f},{v[1]:.3f},{v[2]:.3f},{lab}")
    return "\n".join(rows) + "\n"


def noise_csv(n=300, seed=5):
    rng = np.random.default_rng(seed)
    rows = ["a,b,c"] + [",".join(f"{rng.normal():.3f}" for _ in range(3)) for _ in range(n)]
    return "\n".join(rows) + "\n"


def outlier_csv(n=295, seed=7):
    """Ordinary rows, then five rows that are extreme on two columns. A rare category appears twice."""
    rng = np.random.default_rng(seed)
    rows = ["a,b,c,area"]
    for i in range(n):
        area = "Rarelandia" if i in (10, 11) else rng.choice(["north", "south", "east"])
        rows.append(f"{rng.normal():.3f},{rng.normal():.3f},{rng.normal():.3f},{area}")
    for a, b in ((9.3, -9.1), (-8.7, 9.6), (9.9, 8.8), (-9.4, -8.9), (13.2, 0.4)):   # spread out so they cannot hide each other
        rows.append(f"{a:.3f},{b:.3f},{rng.normal():.3f},north")
    return "\n".join(rows) + "\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def blobs(client):
    return upload(client, "xp-blobs", blobs_csv())


@pytest.fixture(scope="module")
def noise(client):
    return upload(client, "xp-noise", noise_csv())


@pytest.fixture(scope="module")
def outliers(client):
    return upload(client, "xp-outliers", outlier_csv())


def explore(client, ds, **over):
    r = client.post("/v1/explorations", json={"dataset_id": ds, **over}, headers=ADMIN)
    assert r.status_code == 201, r.text
    return client.get(f"/v1/explorations/{r.json()['id']}", headers=ADMIN).json()   # background task already ran


def test_clear_groups_are_found_and_labelled_as_such(client, blobs):
    x = explore(client, blobs, methods=["clustering"])
    assert x["status"] == "completed", x["results_json"]
    c = x["results_json"]["clustering"]
    assert c["k"] == 2 and c["structure"] == "clear_structure"
    assert c["quality"]["silhouette"] > 0.5 and c["stability"]["adjusted_rand_mean"] > 0.8
    assert c["reference"]["silhouette"] < c["quality"]["silhouette"]          # beats the shuffled-data reference
    assert sorted(g["size"] for g in c["groups"]) == [150, 150]
    assert x["results_json"]["anomaly"] is None and [s["k"] for s in c["k_search"]] == list(range(2, 9))
    assert x["summary"]["clustering"]["structure"] == "clear_structure"


def test_random_data_is_not_presented_as_groups(client, noise):
    x = explore(client, noise, methods=["clustering"])
    assert x["status"] == "completed"
    c = x["results_json"]["clustering"]
    assert c["structure"] == "no_clear_structure"
    assert any("no clear structure" in w for w in x["results_json"]["warnings"])


def test_anomalies_are_ranked_with_safe_reasons(client, outliers):
    x = explore(client, outliers, methods=["anomaly"], flag_share=0.02, report_top=10)
    assert x["status"] == "completed", x["results_json"]
    a = x["results_json"]["anomaly"]
    assert x["results_json"]["clustering"] is None and a["n_flagged"] == 6
    assert set(range(295, 300)) <= {t["row_index"] for t in a["top"][:8]}      # the five planted rows are all near the top
    first = a["top"][0]
    assert first["reasons"] and all(r["kind"] in ("numeric", "rare_category") for r in first["reasons"])
    planted = [t for t in a["top"] if t["row_index"] >= 295]
    assert all({r["feature"] for r in t["reasons"]} & {"a", "b"} for t in planted)
    assert [t["rank"] for t in a["top"]] == list(range(1, 11)) and a["top"][0]["score"] >= a["top"][-1]["score"]
    assert {"flagged_set_overlap", "rank_correlation_between_seeds"} <= set(a["stability"])
    assert any("is a setting, not a finding" in w for w in x["results_json"]["warnings"])
    # Reasons carry standardised deviations and shares only: no raw values, and category names are never returned.
    allowed = {"feature", "kind", "z", "direction", "share_of_rows"}
    assert all(set(r) <= allowed for t in a["top"] for r in t["reasons"])
    assert "Rarelandia" not in json.dumps(x["results_json"])


def test_both_methods_cross_reference_and_project(client, outliers):
    x = explore(client, outliers, k=2)
    r = x["results_json"]
    assert x["status"] == "completed" and r["clustering"]["k"] == 2 and r["anomaly"]["by_cluster"]
    p = r["projection"]
    assert len(p["points"]) <= 600 and all(len(pt) == 4 for pt in p["points"]) and len(p["explained_variance"]) == 2


def test_an_exploration_can_never_become_a_model(client, blobs, outliers):
    """Behaviour: running explorations changes nothing that registry, serving or CNR can see."""
    settings = get_settings()
    root = Path(settings.dataset_storage_dir)

    def snapshot():
        ov = client.get("/v1/overview", headers=ADMIN).json()
        return {"models": ov["counts"]["model_versions"], "deployed": ov["counts"]["deployed_models"],
                "experiments": ov["counts"]["experiments"],
                "catalog": {c["slug"]: c["status"] for c in client.get("/v1/catalog", headers=ADMIN).json()},
                "versions": client.get("/v1/models/versions", headers=ADMIN).json(),
                "cnr": client.get("/v1/integration/cnr/models", headers=ADMIN).json(),
                "model_files": sorted(str(p) for p in root.rglob("*.joblib")) if root.exists() else []}

    before = snapshot()
    x = explore(client, blobs)
    explore(client, outliers)
    assert x["kind"] == "exploration" and x["deployable"] is False and x["results_json"]["deployable"] is False
    assert "can never be registered" in x["results_json"]["notice"]
    assert snapshot() == before
    assert all(s == "planned" for s in before["catalog"].values()) or before["catalog"]   # catalogue untouched
    # No path, artifact or storage URI leaks into the result.
    text = json.dumps(x)
    assert "artifact" not in text and "local://" not in text and ".joblib" not in text
    # An exploration is not an experiment, so the registry has no way to pick it up.
    with SessionLocal() as db:
        assert db.get(Exploration, x["id"]) is not None
        assert not hasattr(Exploration, "model_definition_id") and not hasattr(Exploration, "artifact_uri")
    audit = client.get("/v1/audit?limit=200", headers=ADMIN).text
    assert "exploration.created" in audit and "exploration.completed" in audit and "Zomba" not in audit


def test_exploration_code_is_isolated_from_registry_serving_and_cnr():
    """Static guard: these modules may not import anything that can register, serve or announce a model."""
    forbidden = {"registry", "serving", "predict", "webhook", "integration", "joblib", "pickle", "catalog"}
    for rel in ("exploration.py", "jobs.py", "routers/explorations.py"):
        src = (APP_DIR / rel).read_text()
        names = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names |= {part for a in node.names for part in a.name.split(".")}
            elif isinstance(node, ast.ImportFrom):
                names |= {part for part in (node.module or "").split(".") if part} | {a.name for a in node.names}
        assert not (names & forbidden), f"{rel} imports {sorted(names & forbidden)}"
        assert "save_bytes" not in src and "ModelVersion" not in src.replace("DatasetVersion", "")
    # The registry only ever reads Experiment rows, so it cannot see the Exploration table.
    reg = (APP_DIR / "routers/registry.py").read_text()
    assert "Exploration" not in reg and "explorations" not in reg


def test_permissions(client, blobs):
    viewer = client.post("/v1/admin/keys", json={"name": "viewer-xp", "role": "viewer"}, headers=ADMIN).json()["key"]
    service = client.post("/v1/admin/keys", json={"name": "service-xp", "role": "service"}, headers=ADMIN).json()["key"]
    scientist = client.post("/v1/admin/keys", json={"name": "ds-xp", "role": "data_scientist"}, headers=ADMIN).json()["key"]
    body = {"dataset_id": blobs, "methods": ["clustering"]}
    assert client.get("/v1/explorations", headers={"X-API-Key": viewer}).status_code == 200
    assert client.post("/v1/explorations", json=body, headers={"X-API-Key": viewer}).status_code == 403
    assert client.get("/v1/explorations", headers={"X-API-Key": service}).status_code == 403    # the CNR machine key sees none of it
    assert client.post("/v1/explorations", json=body, headers={"X-API-Key": service}).status_code == 403
    assert client.post("/v1/explorations", json=body, headers={"X-API-Key": scientist}).status_code == 201


def test_validation(client, blobs):
    ok = {"dataset_id": blobs}
    post = lambda **o: client.post("/v1/explorations", json={**ok, **o}, headers=ADMIN)  # noqa: E731
    assert post(exclude=["nope"]).status_code == 422
    assert post(features=["a", "nope"]).status_code == 422
    assert post(methods=[]).status_code == 422
    assert post(methods=["clustering", "clustering"]).status_code == 422
    assert post(methods=["classification"]).status_code == 422
    assert post(k_min=6, k_max=3).status_code == 422
    assert post(flag_share=0.9).status_code == 422
    assert client.post("/v1/explorations", json={"dataset_id": 99999}, headers=ADMIN).status_code == 404
    assert client.get("/v1/explorations/99999", headers=ADMIN).status_code == 404


def test_failures_are_explained(client):
    few = upload(client, "xp-few", "a,b\n" + "\n".join(f"{i * 0.5},{i * 0.3 + 1}" for i in range(30)) + "\n")
    x = explore(client, few)
    assert x["status"] == "failed" and "Too few rows" in x["results_json"]["error"]
    one = upload(client, "xp-one", "a,b\n" + "\n".join(f"{(i * 37) % 101 / 10},same" for i in range(120)) + "\n")
    y = explore(client, one)
    assert y["status"] == "failed" and "two usable columns" in y["results_json"]["error"]


def test_identifier_like_columns_are_skipped(client):
    rng = np.random.default_rng(11)
    rows = ["rownum,a,b"] + [f"{i},{rng.normal():.3f},{rng.normal():.3f}" for i in range(200)]
    ds = upload(client, "xp-rownum", "\n".join(rows) + "\n")
    x = explore(client, ds, methods=["clustering"])
    assert x["status"] == "completed"
    assert "rownum" not in x["results_json"]["features"]["numeric"]
    assert any("rownum" in w and "identifier" in w for w in x["results_json"]["warnings"])


def test_one_heavy_job_at_a_time_across_both_kinds(client, blobs):
    with SessionLocal() as db:
        db.add(Exploration(status="running", created_at=utcnow()))
        db.commit()
    assert client.post("/v1/explorations", json={"dataset_id": blobs}, headers=ADMIN).status_code == 409
    assert client.post("/v1/experiments", headers=ADMIN, json={"model_slug": "diabetes-risk", "dataset_id": blobs,
                                                               "target": "district"}).status_code == 409
    with SessionLocal() as db:
        for x in db.query(Exploration).filter(Exploration.status == "running"):
            x.status = "failed"
        db.commit()
    with SessionLocal() as db:
        db.add(Experiment(model_definition_id=1, framework="xgboost", status="running", created_at=utcnow()))
        db.commit()
    assert client.post("/v1/explorations", json={"dataset_id": blobs}, headers=ADMIN).status_code == 409
    with SessionLocal() as db:
        for e in db.query(Experiment).filter(Experiment.status == "running"):
            e.status = "failed"
        db.commit()
