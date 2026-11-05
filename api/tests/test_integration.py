"""Phase 6: CNR integration contract, status and reachability check."""
import json

import pytest
from fastapi.testclient import TestClient

from app import integration
from app.config import Settings, get_settings
from app.main import app
from app.routers import integration as integ_router
from app.ratelimit import SlidingWindow
from test_experiments import ADMIN, make_csv, train
from test_predict import APPROVE, key

SLUG = "hospital-malnutrition-risk"      # its own slug so this module never collides with the others


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def people(client):
    return {"svc": key(client, "service", "i-svc"), "rev": key(client, "clinical_reviewer", "i-rev"),
            "ds": key(client, "data_scientist", "i-ds"), "view": key(client, "viewer", "i-view")}


def test_nothing_deployed_is_refused_not_guessed(client, people):
    body = {"cases": [{"case_ref": "c1", "features": {"age": 20, "weight": 9}}]}
    r = client.post(f"/v1/integration/cnr/assess/{SLUG}", json=body, headers=people["svc"])
    assert r.status_code == 409 and "No approved model" in r.json()["detail"]
    assert client.get("/v1/integration/cnr/models", headers=people["svc"]).json()["models"] == []


@pytest.fixture(scope="module")
def deployed(client, people):
    d = client.post("/v1/datasets", json={"name": "integ-ds"}, headers=ADMIN).json()
    assert client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-INT"},
                       files={"file": ("t.csv", make_csv(seed=9), "text/csv")}).status_code == 201
    e = train(client, d["id"], model_slug=SLUG)
    assert e["status"] == "completed", e["metrics_json"]
    v = client.post("/v1/models/versions", json={"experiment_id": e["id"], "intended_use": "Integration test only; not for clinical use."},
                    headers=people["ds"]).json()
    client.post(f"/v1/models/versions/{v['id']}/submit", json={}, headers=people["ds"])
    assert client.post(f"/v1/models/versions/{v['id']}/approve", json=APPROVE, headers=people["rev"]).status_code == 200
    assert client.post(f"/v1/models/versions/{v['id']}/deploy", json={}, headers=ADMIN).status_code == 200
    yield v["id"]
    client.post(f"/v1/models/versions/{v['id']}/retire", json={"reason": "Integration tests finished."}, headers=ADMIN)


def test_models_contract(client, people, deployed):
    m = client.get("/v1/integration/cnr/models", headers=people["svc"]).json()
    assert m["contract"] == integration.CONTRACT and [x["slug"] for x in m["models"]] == [SLUG]
    assert m["models"][0]["available"] and set(m["models"][0]["features"]["numeric"]) == {"age", "weight"}


def test_assess_echoes_case_refs_and_is_labelled(client, people, deployed):
    body = {"cases": [{"case_ref": "CNR-A1", "features": {"age": 20, "weight": 6.0, "district": "Zomba"}},
                      {"case_ref": "CNR-A2", "features": {"age": 40, "weight": 12}}]}
    r = client.post(f"/v1/integration/cnr/assess/{SLUG}", json=body, headers={**people["svc"], "X-Request-Id": "cnr-req-77"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert [x["case_ref"] for x in b["results"]] == ["CNR-A1", "CNR-A2"]
    assert b["contract"] == integration.CONTRACT and b["correlation_id"] == "cnr-req-77"
    assert b["decision_support_only"] is True and "never confirmed diagnoses" in b["safety_notice"]
    assert all(x["kind"] == "risk_estimate" for x in b["results"])
    audit = [a for a in client.get("/v1/audit?limit=100", headers=ADMIN).json() if a["action"] == "integration.assess"]
    assert audit and audit[0]["detail"]["records"] == 2 and audit[0]["detail"]["correlation_id"] == "cnr-req-77"
    blob = json.dumps(audit[0])
    assert "CNR-A1" not in blob and "Zomba" not in blob and "weight" not in blob   # refs and inputs are not logged


def test_validation(client, people, deployed):
    url, h = f"/v1/integration/cnr/assess/{SLUG}", people["svc"]
    post = lambda cases: client.post(url, json={"cases": cases}, headers=h).status_code  # noqa: E731
    f = {"age": 20, "weight": 9}
    assert post([{"case_ref": "a@b.com", "features": f}]) == 422            # looks like an email
    assert post([{"case_ref": "0999123456", "features": f}]) == 422         # looks like a phone/ID number
    assert post([{"case_ref": "has space", "features": f}]) == 422
    assert post([{"case_ref": "x", "features": f}, {"case_ref": "x", "features": f}]) == 422   # duplicate
    assert post([{"case_ref": "x", "features": {"age": 20, "patient_name": "N"}}]) == 422     # unknown field
    assert post([]) == 422
    assert post([{"case_ref": f"c{i}", "features": f} for i in range(51)]) == 422
    bad = client.post(url, json={"cases": [{"case_ref": "x", "features": f}]}, headers={**h, "X-Request-Id": "bad id!"})
    assert bad.status_code == 200 and "correlation_id" not in bad.json()     # invalid header is ignored, not echoed


def test_roles(client, people, deployed):
    url = f"/v1/integration/cnr/assess/{SLUG}"
    body = {"cases": [{"case_ref": "r1", "features": {"age": 20, "weight": 9}}]}
    assert client.post(url, json=body).status_code == 401
    assert client.post(url, json=body, headers=people["view"]).status_code == 403
    assert client.get("/v1/integration/cnr/status", headers=people["svc"]).status_code == 403   # service key cannot read admin status
    assert client.get("/v1/integration/cnr/status", headers=people["rev"]).status_code == 403


def test_assess_shares_the_prediction_rate_limit(client, people, deployed, monkeypatch):
    from app.routers import predict as pr
    monkeypatch.setattr(pr, "_limiter", SlidingWindow(2, 60))
    body = {"cases": [{"case_ref": "z", "features": {"age": 20, "weight": 9}}]}
    codes = [client.post(f"/v1/integration/cnr/assess/{SLUG}", json=body, headers=people["svc"]).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_status(client, people, deployed):
    s = client.get("/v1/integration/cnr/status", headers=ADMIN).json()
    assert s["contract"] == integration.CONTRACT and s["probe"] is None
    assert any(k["name"] == "i-svc" for k in s["service_keys"]) and s["deployed_models"] >= 1
    assert s["last_24h"]["assessments"] >= 1
    assert "key_hash" not in json.dumps(s)                       # no secrets or hashes in the status view


def test_config_checks_and_probe(monkeypatch):
    mk = lambda **k: Settings(_env_file=None, **k)  # noqa: E731
    assert "not set" in integration.config_problem(mk(cnr_api_base_url=""))
    assert "credentials" in integration.config_problem(mk(cnr_api_base_url="https://u:p@x.example"))
    assert "https" in integration.config_problem(mk(cnr_api_base_url="http://x.example", environment="production"))
    assert integration.config_problem(mk(cnr_api_base_url="https://x.example")) is None
    assert integration.config_problem(mk(cnr_api_base_url="ftp://x.example")) is not None
    assert integration.probe(mk(cnr_api_base_url=""))["reachable"] is False
    # an unreachable address is reported, not raised
    r = integration.probe(mk(cnr_api_base_url="http://127.0.0.1:9", cnr_api_timeout_s=1))
    assert r["reachable"] is False and "No answer" in r["reason"]


def test_probe_endpoint_sends_nothing_secret(client, people, monkeypatch):
    seen = {}
    monkeypatch.setattr(integ_router.integration, "probe", lambda s: seen.setdefault("called", True) and {"reachable": True, "http_status": 200, "latency_ms": 5})
    s = client.get("/v1/integration/cnr/status?probe=true", headers=ADMIN).json()
    assert s["probe"] == {"reachable": True, "http_status": 200, "latency_ms": 5} and seen["called"]
    assert any(a["action"] == "integration.probe" for a in client.get("/v1/audit?limit=20", headers=ADMIN).json())
    monkeypatch.setattr(integ_router, "_probe_limiter", SlidingWindow(1, 60))
    assert client.get("/v1/integration/cnr/status?probe=true", headers=ADMIN).status_code == 200
    assert client.get("/v1/integration/cnr/status?probe=true", headers=ADMIN).status_code == 429


# ---------- lab -> CNR events ----------
from app import webhook                                                    # noqa: E402

SECRET = "w" * 40


def _cfg(monkeypatch, url="https://cnr.example/hooks/ml-lab", secret=SECRET, env="development"):
    s = get_settings()
    monkeypatch.setattr(s, "cnr_webhook_url", url)
    monkeypatch.setattr(s, "cnr_webhook_secret", secret)
    monkeypatch.setattr(s, "environment", env)
    return s


def test_webhook_config_checks(monkeypatch):
    s = _cfg(monkeypatch, url="")
    assert "not set" in webhook.config_problem(s)
    assert "credentials" in webhook.config_problem(_cfg(monkeypatch, url="https://u:p@x.example/h"))
    assert "https" in webhook.config_problem(_cfg(monkeypatch, url="http://x.example/h", env="production"))
    assert "32" in webhook.config_problem(_cfg(monkeypatch, secret="short"))
    assert webhook.config_problem(_cfg(monkeypatch)) is None


def test_signature_matches_what_cnr_verifies():
    import hashlib, hmac
    body = b'{"a":1}'
    want = "sha256=" + hmac.new(SECRET.encode(), b"1700000000." + body, hashlib.sha256).hexdigest()
    assert webhook.sign(SECRET, "1700000000", body) == want
    assert webhook.sign(SECRET, "1700000001", body) != want          # timestamp is covered, so replays with a new time fail


def test_event_holds_no_patient_or_evaluation_data():
    e = webhook.build_event("withdrawn", "child-malnutrition-risk", 3, 12, reason="withdrawn")
    assert e["event"] == "model.retired" and e["model"] == {"slug": "child-malnutrition-risk", "version": 3, "version_id": 12}
    assert set(e) == {"event_id", "event", "occurred_at", "contract", "model", "detail"} and e["detail"] == {"reason": "withdrawn"}


def test_emit_is_off_without_config_and_never_blocks(monkeypatch):
    _cfg(monkeypatch, url="")
    sent = []
    monkeypatch.setattr(webhook, "_spawn", lambda fn, *a: sent.append(a))
    assert webhook.emit("deployed", "m", 1, 1) is False and sent == []
    _cfg(monkeypatch)
    assert webhook.emit("deployed", "m", 1, 1) is True and sent[0][0]["event"] == "model.deployed"


def test_delivery_retries_then_audits_without_secrets(client, monkeypatch):
    _cfg(monkeypatch)
    monkeypatch.setattr(webhook, "_BACKOFF_S", (0, 0))
    calls = []
    def fake_post(s, ev):
        calls.append(ev["event_id"])
        if len(calls) < 3:
            raise ConnectionError("down")
        return 200
    monkeypatch.setattr(webhook, "_post", fake_post)
    ev = webhook.build_event("deployed", "child-malnutrition-risk", 1, 1, superseded_version=None)
    webhook._deliver(ev)
    assert len(calls) == 3
    row = next(a for a in client.get("/v1/audit?limit=20", headers=ADMIN).json() if a["action"] == "integration.notify")
    assert row["detail"]["delivered"] is True and row["detail"]["event"] == "model.deployed"
    assert SECRET not in json.dumps(row)


def test_a_refusal_is_not_retried(client, monkeypatch):
    _cfg(monkeypatch)
    monkeypatch.setattr(webhook, "_BACKOFF_S", (0, 0))
    calls = []
    monkeypatch.setattr(webhook, "_post", lambda s, ev: calls.append(1) or 401)
    webhook._deliver(webhook.build_event("withdrawn", "m", 1, 1))
    assert len(calls) == 1
    row = next(a for a in client.get("/v1/audit?limit=20", headers=ADMIN).json() if a["action"] == "integration.notify")
    assert row["detail"]["delivered"] is False and row["detail"]["http_status"] == 401


def test_deploy_and_withdraw_queue_events(client, people, monkeypatch):
    _cfg(monkeypatch)
    events = []
    monkeypatch.setattr(webhook, "_spawn", lambda fn, ev: events.append(ev))
    slug = "food-insecurity-prediction"
    d = client.post("/v1/datasets", json={"name": "wh-ds"}, headers=ADMIN).json()
    assert client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-WH"},
                       files={"file": ("t.csv", make_csv(seed=11), "text/csv")}).status_code == 201
    e = train(client, d["id"], model_slug=slug)
    assert e["status"] == "completed", e["metrics_json"]
    v = client.post("/v1/models/versions", json={"experiment_id": e["id"], "intended_use": "Webhook test only; not for clinical use."},
                    headers=people["ds"]).json()
    client.post(f"/v1/models/versions/{v['id']}/submit", json={}, headers=people["ds"])
    assert client.post(f"/v1/models/versions/{v['id']}/approve", json=APPROVE, headers=people["rev"]).status_code == 200
    assert events == []                                                    # approval alone tells CNR nothing
    assert client.post(f"/v1/models/versions/{v['id']}/deploy", json={}, headers=ADMIN).status_code == 200
    assert [x["event"] for x in events] == ["model.deployed"] and events[0]["model"]["slug"] == slug
    assert client.post(f"/v1/models/versions/{v['id']}/retire", json={"reason": "Webhook tests finished."}, headers=ADMIN).status_code == 200
    assert [x["event"] for x in events] == ["model.deployed", "model.retired"]
    assert "Webhook tests" not in json.dumps(events)                       # free-text reasons never leave the lab


def test_status_reports_webhook(client, monkeypatch):
    _cfg(monkeypatch)
    w = client.get("/v1/integration/cnr/status", headers=ADMIN).json()["webhook"]
    assert w["configured"] is True and w["host"] == "cnr.example" and SECRET not in json.dumps(w)
    _cfg(monkeypatch, url="")
    assert client.get("/v1/integration/cnr/status", headers=ADMIN).json()["webhook"]["configured"] is False
