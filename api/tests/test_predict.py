"""Phase 5: serving, input validation, refusal when nothing is deployed, rate limits."""
import json

import pytest
from fastapi.testclient import TestClient

from app import serving, storage
from app.main import app
from app.ratelimit import SlidingWindow
from app.routers import predict as predict_router
from test_experiments import ADMIN, make_csv, train

SLUG = "child-malnutrition-risk"      # different slug from the registry tests so the two modules never collide
APPROVE = {"clinical_validation_ref": "COMREC-2026-020 external validation", "review_notes": "Reviewed for the serving tests.",
           "acknowledge_warnings": True}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, role, name):
    k = client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()
    return {"X-API-Key": k["key"]}


@pytest.fixture(scope="module")
def people(client):
    return {"svc": key(client, "service", "p-svc"), "rev": key(client, "clinical_reviewer", "p-rev"),
            "ds": key(client, "data_scientist", "p-ds"), "view": key(client, "viewer", "p-view")}


def test_nothing_deployed_means_no_prediction(client, people):
    rec = {"records": [{"age": 20, "weight": 10}]}
    r = client.post(f"/v1/predict/{SLUG}", json=rec, headers=people["svc"])
    assert r.status_code == 409 and "No approved model" in r.json()["detail"]
    assert client.post("/v1/predict/not-a-model", json=rec, headers=people["svc"]).status_code == 404
    assert client.get("/v1/predict", headers=people["svc"]).json()["models"] == []
    assert client.post(f"/v1/predict/{SLUG}", json=rec).status_code == 401
    assert client.post(f"/v1/predict/{SLUG}", json=rec, headers=people["view"]).status_code == 403
    audit = client.get("/v1/audit?limit=50", headers=ADMIN).json()
    assert any(a["action"] == "prediction.refused" for a in audit)


@pytest.fixture(scope="module")
def deployed(client, people):
    d = client.post("/v1/datasets", json={"name": "serve-ds"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-SRV"},
                    files={"file": ("t.csv", make_csv(seed=5), "text/csv")})
    assert r.status_code == 201, r.text
    e = train(client, d["id"], model_slug=SLUG)
    assert e["status"] == "completed", e["metrics_json"]
    v = client.post("/v1/models/versions", json={"experiment_id": e["id"], "intended_use": "Serving test only; not for clinical use."},
                    headers=people["ds"]).json()
    client.post(f"/v1/models/versions/{v['id']}/submit", json={}, headers=people["ds"])
    assert client.post(f"/v1/models/versions/{v['id']}/approve", json=APPROVE, headers=people["rev"]).status_code == 200
    assert client.post(f"/v1/models/versions/{v['id']}/deploy", json={}, headers=ADMIN).status_code == 200
    yield v["id"]
    # leave the shared test database as we found it for later modules
    client.post(f"/v1/models/versions/{v['id']}/retire", json={"reason": "Serving tests finished."}, headers=ADMIN)


def test_catalogue_and_schema(client, people, deployed):
    lst = client.get("/v1/predict", headers=people["svc"]).json()
    assert [m["slug"] for m in lst["models"]] == [SLUG] and "never confirmed diagnoses" in lst["safety_notice"]
    s = client.get(f"/v1/predict/{SLUG}", headers=people["svc"]).json()
    assert set(s["features"]["numeric"]) == {"age", "weight"}
    assert s["features"]["categorical"][0]["name"] == "district"
    assert set(s["features"]["categorical"][0]["values"]) == {"Zomba", "Blantyre", "Mangochi"}
    assert s["classes"] == ["moderate", "normal", "severe"] and s["clinical_validation_ref"].startswith("COMREC")


def test_prediction_is_labelled_and_audited_without_inputs(client, people, deployed):
    r = client.post(f"/v1/predict/{SLUG}", json={"records": [{"age": 20, "weight": 6.0, "district": "Zomba"}]}, headers=people["svc"])
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["decision_support_only"] is True and "never confirmed diagnoses" in b["safety_notice"]
    assert b["model"]["version_id"] == deployed and b["model"]["version"] == 1 and b["request_id"] and b["generated_at"].endswith("Z")
    res = b["results"][0]
    assert res["kind"] == "risk_estimate" and res["predicted_class"] in {"moderate", "normal", "severe"}
    assert abs(sum(res["probabilities"].values()) - 1) < 1e-3
    assert set(res["risk_indicators"]) == {"top_probability", "margin_to_next", "missing_fraction", "low_confidence"}
    assert b["caveats"]
    audit = [a for a in client.get("/v1/audit?limit=100", headers=ADMIN).json() if a["action"] == "prediction.made"]
    assert audit and audit[0]["detail"]["records"] == 1
    blob = json.dumps(audit[0])
    assert "Zomba" not in blob and "weight" not in blob          # inputs never reach the audit log


def test_batch_and_validation(client, people, deployed):
    url, h = f"/v1/predict/{SLUG}", people["svc"]
    ok = client.post(url, json={"records": [{"age": 10, "weight": 7}, {"age": 30, "weight": 11.5, "district": "Blantyre"}, {"age": 50}]}, headers=h)
    assert ok.status_code == 200 and len(ok.json()["results"]) == 3
    assert client.post(url, json={"records": [{"age": 10, "patient_name": "X"}]}, headers=h).status_code == 422   # no free fields
    assert client.post(url, json={"records": [{"age": "ten"}]}, headers=h).status_code == 422
    assert client.post(url, json={"records": [{"age": True}]}, headers=h).status_code == 422
    assert client.post(url, json={"records": [{}]}, headers=h).status_code == 422
    assert client.post(url, json={"records": [{"age": None, "weight": None}]}, headers=h).status_code == 422
    assert client.post(url, json={"records": []}, headers=h).status_code == 422
    assert client.post(url, json={"records": [{"age": 1}] * 51}, headers=h).status_code == 422
    odd = client.post(url, json={"records": [{"age": 20, "weight": 9, "district": "Atlantis"}]}, headers=h).json()["results"][0]
    assert any("did not see" in n for n in odd["notes"])
    sparse = client.post(url, json={"records": [{"age": 20}]}, headers=h).json()["results"][0]
    assert sparse["risk_indicators"]["low_confidence"] is True and sparse["risk_indicators"]["missing_fraction"] > 0.5
    assert not any("Atlantis" in n for n in odd["notes"])           # values are not echoed back


def test_service_role_is_narrow(client, people):
    h = people["svc"]
    assert client.get("/v1/experiments", headers=h).status_code == 403
    assert client.get("/v1/audit", headers=h).status_code == 403
    assert client.get("/v1/datasets", headers=h).status_code == 403
    assert client.post("/v1/admin/keys", json={"name": "zz", "role": "admin"}, headers=h).status_code == 403


def test_rate_limit_per_key(client, people, deployed, monkeypatch):
    monkeypatch.setattr(predict_router, "_limiter", SlidingWindow(3, 60))
    body = {"records": [{"age": 20, "weight": 9}]}
    codes = [client.post(f"/v1/predict/{SLUG}", json=body, headers=people["svc"]).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = client.post(f"/v1/predict/{SLUG}", json=body, headers=people["svc"])
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert client.post(f"/v1/predict/{SLUG}", json=body, headers=people["rev"]).status_code == 200   # other keys unaffected


def test_wrong_key_flood_is_throttled_but_valid_keys_still_work(client, people):
    ip = {"X-Forwarded-For": "203.0.113.77"}
    codes = [client.get("/v1/me", headers={**ip, "X-API-Key": f"bad{i}"}).status_code for i in range(22)]
    assert codes[0] == 401 and codes[-1] == 429
    assert client.get("/v1/me", headers={**ip, **people["svc"]}).status_code == 200


def test_changed_model_file_is_never_served(client, people, deployed):
    v = client.get(f"/v1/models/versions/{deployed}", headers=ADMIN).json()
    path = storage._path(f"local://models/{SLUG}/v{v['version']}.joblib")
    original = path.read_bytes()
    serving.clear_cache()                                            # as after a restart
    path.write_bytes(original + b"x")
    try:
        r = client.post(f"/v1/predict/{SLUG}", json={"records": [{"age": 20, "weight": 9}]}, headers=people["svc"])
        assert r.status_code == 503 and "hash" in r.json()["detail"]
    finally:
        path.write_bytes(original)
        serving.clear_cache()
    assert client.post(f"/v1/predict/{SLUG}", json={"records": [{"age": 20, "weight": 9}]}, headers=people["rev"]).status_code == 200


def test_retiring_stops_predictions_immediately(client, people, deployed):
    assert client.post(f"/v1/models/versions/{deployed}/retire", json={"reason": "Stop serving now."}, headers=ADMIN).status_code == 200
    r = client.post(f"/v1/predict/{SLUG}", json={"records": [{"age": 20, "weight": 9}]}, headers=people["svc"])
    assert r.status_code == 409
