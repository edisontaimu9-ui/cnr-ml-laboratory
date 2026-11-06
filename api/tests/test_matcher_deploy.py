"""Food-name matcher through the review gate and into service: suggest-only, registry-owner scope, hash check, isolation from CNR."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/md.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import serving, storage, webhook  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import ModelDefinition, ModelVersion  # noqa: E402
from test_matching import BODY, queries_csv, registry_csv  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
USE = "Help dietitians find the registry food for a typed food name when entering survey data."
REVIEW = {"clinical_validation_ref": "FOODREG-REVIEW-001", "review_notes": "Checked the hold-out results with the registry team.",
          "acknowledge_warnings": True}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def upload(client, name, csv):
    d = client.post("/v1/datasets", json={"name": name}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-DEPLOY"},
                    files={"file": ("m.csv", csv, "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


def key(client, role):
    r = client.post("/v1/admin/keys", headers=ADMIN, json={"name": f"{role} test key", "role": role})
    assert r.status_code == 201, r.text
    return {"Authorization": "Bearer " + r.json()["key"]}


@pytest.fixture(scope="module")
def world(client):
    qid, rid = upload(client, "deploy-queries", queries_csv()), upload(client, "deploy-registry", registry_csv())
    w = {"qid": qid, "rid": rid, "ds": key(client, "data_scientist"), "owner": key(client, "registry_owner"),
         "service": key(client, "service"), "viewer": key(client, "viewer"), "reviewer": key(client, "clinical_reviewer")}
    yield w
    # The test database is shared with other test files: leave no versions or deployed models behind.
    with SessionLocal() as db:
        d = db.query(ModelDefinition).filter(ModelDefinition.slug == "food-name-matching").one()
        for v in db.query(ModelVersion).filter(ModelVersion.model_definition_id == d.id).all():
            db.delete(v)
        for v in db.query(ModelVersion).filter(ModelVersion.version == 99, ModelVersion.created_by_key_id == 0).all():
            db.delete(v)
        d.status = "planned"
        db.commit()
    serving.clear_cache()


def candidate(client, w, **extra):
    body = {**BODY, "queries_dataset_id": w["qid"], "registry_dataset_id": w["rid"], "intended_use": USE, **extra}
    return client.post("/v1/matching/candidates", headers=w["ds"], json=body)


def test_candidate_goes_through_review_and_the_registry_owner_can_approve(client, world, monkeypatch):
    sent = []
    monkeypatch.setattr(webhook, "emit", lambda *a, **k: sent.append(a))
    r = candidate(client, world)
    assert r.status_code == 201, r.text
    vid = r.json()["id"]
    d = client.get(f"/v1/models/versions/{vid}", headers=ADMIN).json()
    assert d["status"] == "candidate" and d["headline"]["metric"] == "top1" and "Suggest-only" in d["intended_use"]
    assert d["headline"]["test"] > d["headline"]["baseline"]
    assert client.post(f"/v1/models/versions/{vid}/submit", headers=world["ds"], json={}).status_code == 200
    # nobody is served until approved and deployed
    assert client.post("/v1/matching/suggest", headers=world["service"], json={"queries": ["kapenta"]}).status_code == 409
    assert client.post(f"/v1/models/versions/{vid}/approve", headers=world["viewer"], json=REVIEW).status_code == 403
    ok = client.post(f"/v1/models/versions/{vid}/approve", headers=world["owner"], json=REVIEW)
    assert ok.status_code == 200, ok.text
    assert client.post(f"/v1/models/versions/{vid}/deploy", headers=world["owner"], json={}).status_code == 403
    dep = client.post(f"/v1/models/versions/{vid}/deploy", headers=ADMIN, json={})
    assert dep.status_code == 200, dep.text
    assert sent == []                                   # CNR is not told about a food matcher
    world["vid"] = vid


def test_a_clinical_reviewer_can_also_approve_a_matcher(client, world):
    vid = candidate(client, world, seed=7).json()["id"]
    client.post(f"/v1/models/versions/{vid}/submit", headers=world["ds"], json={})
    r = client.post(f"/v1/models/versions/{vid}/approve", headers=world["reviewer"], json=REVIEW)
    assert r.status_code == 200, r.text


def test_registry_owner_cannot_review_or_register_anything_else(client, world):
    with SessionLocal() as db:
        other = db.query(ModelDefinition).filter(ModelDefinition.task_type == "classification").first()
        v = ModelVersion(model_definition_id=other.id, version=99, status="in_review", metrics_json={}, history_json=[],
                         created_by_key_id=0, submitted_by_key_id=0)
        db.add(v); db.commit(); vid = v.id
    assert client.post(f"/v1/models/versions/{vid}/approve", headers=world["owner"], json=REVIEW).status_code == 403
    assert client.post(f"/v1/models/versions/{vid}/reject", headers=world["owner"], json={"review_notes": "Not suitable here."}).status_code == 403
    assert client.post(f"/v1/models/versions/{vid}/approve", headers=world["reviewer"], json=REVIEW).status_code != 403
    assert candidate(client, {**world, "ds": world["owner"]}).status_code == 403


def test_suggest_only_response_and_privacy(client, world):
    r = client.post("/v1/matching/suggest", headers=world["service"], json={"queries": ["kapenta", "tilapya", "  ", "???"], "top_k": 3})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["suggestion_only"] is True and "not a probability" in j["score_is"]
    assert len(j["results"]) == 4
    assert j["results"][0]["candidates"][0]["food_code"] == "F09"            # kapenta is a colloquial name for small dried fish
    assert j["results"][1]["candidates"][0]["food_code"] == "F08"
    assert j["results"][2]["candidates"] == [] and j["results"][2]["below_review_threshold"] is True
    assert all(len(x["candidates"]) <= 3 for x in j["results"])
    assert "assigned" not in " ".join(j["results"][0].keys())
    blob = r.text
    assert "kapenta" not in blob and "tilapya" not in blob                    # query text is never echoed
    log = client.get("/v1/audit?action=matching.suggested", headers=ADMIN).json()
    assert log and all("kapenta" not in str(e["detail"]) for e in log)
    assert log[0]["detail"]["queries"] == 4
    assert client.get("/v1/matching/deployed", headers=world["service"]).json()["deployed"] is True


def test_suggest_input_limits_and_permissions(client, world):
    assert client.post("/v1/matching/suggest", headers=world["service"], json={"queries": []}).status_code == 422
    assert client.post("/v1/matching/suggest", headers=world["service"], json={"queries": ["x" * 121]}).status_code == 422
    assert client.post("/v1/matching/suggest", headers=world["service"], json={"queries": ["a"] * 51}).status_code == 422
    assert client.post("/v1/matching/suggest", headers=world["viewer"], json={"queries": ["a"]}).status_code == 403
    assert client.post("/v1/matching/suggest", json={"queries": ["a"]}).status_code in (401, 403)


def test_matcher_is_invisible_to_the_prediction_and_cnr_endpoints_and_experiments(client, world):
    assert all(m["slug"] != "food-name-matching" for m in client.get("/v1/predict", headers=world["service"]).json()["models"])
    assert client.post("/v1/predict/food-name-matching", headers=world["service"], json={"records": [{"a": 1}]}).status_code == 422
    assert client.post("/v1/integration/cnr/assess/food-name-matching", headers=world["service"], json={"cases": [{"features": {"a": 1}}]}).status_code in (409, 422)
    assert all(m["slug"] != "food-name-matching" for m in client.get("/v1/integration/cnr/models", headers=world["service"]).json()["models"])
    r = client.post("/v1/experiments", headers=ADMIN, json={"model_slug": "food-name-matching", "dataset_id": world["qid"], "target": "food_code"})
    assert r.status_code == 422 and "/v1/matching" in r.text


def test_a_changed_file_pauses_suggestions(client, world):
    with SessionLocal() as db:
        v = db.get(ModelVersion, world["vid"])
        path = storage._path(v.artifact_uri)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tamper")
        serving.clear_cache()
        r = client.post("/v1/matching/suggest", headers=world["service"], json={"queries": ["kapenta"]})
        assert r.status_code == 503
    finally:
        path.write_bytes(original)
        serving.clear_cache()
    assert client.post("/v1/matching/suggest", headers=world["service"], json={"queries": ["kapenta"]}).status_code == 200
