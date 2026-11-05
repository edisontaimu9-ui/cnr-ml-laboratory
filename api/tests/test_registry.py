"""Registry: review gate, deployment designation and rollback."""
import hashlib

import pytest
from fastapi.testclient import TestClient

from app import storage
from app.main import app
from test_experiments import ADMIN, make_csv, train

SLUG = "acute-malnutrition-classification"
USE = "Support screening of children under five at Zomba clinics; never a diagnosis."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, role, name):
    k = client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()
    return {"X-API-Key": k["key"]}


@pytest.fixture(scope="module")
def people(client):
    return {"ds": key(client, "data_scientist", "reg-ds"), "rev": key(client, "clinical_reviewer", "reg-rev"),
            "view": key(client, "viewer", "reg-view")}


@pytest.fixture(scope="module")
def dsid(client):
    d = client.post("/v1/datasets", json={"name": "registry-ds"}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-REG"},
                    files={"file": ("t.csv", make_csv(seed=7), "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


def new_exp(client, dsid, **kw):
    e = train(client, dsid, **kw)
    assert e["status"] == "completed", e["metrics_json"]
    return e["id"]


def register(client, exp, headers):
    return client.post("/v1/models/versions", json={"experiment_id": exp, "intended_use": USE}, headers=headers)


APPROVE = {"clinical_validation_ref": "COMREC-2026-017 external validation", "review_notes": "Reviewed per-class results and limits.",
           "acknowledge_warnings": True}


def test_register_freezes_record_and_hides_paths(client, dsid, people):
    exp = new_exp(client, dsid)
    r = register(client, exp, people["ds"])
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["status"] == "candidate" and v["version"] == 1 and v["experiment_id"] == exp
    assert "artifact_uri" not in v and "artifact_uri" not in v["metrics_json"]
    assert len(v["artifact_sha256"]) == 64 and v["history"][0]["action"] == "registered"
    assert register(client, exp, people["ds"]).status_code == 409           # one experiment, one version
    cat = {c["slug"]: c["status"] for c in client.get("/v1/catalog", headers=ADMIN).json()}
    assert cat[SLUG] == "in_development"
    assert register(client, exp, people["view"]).status_code == 403          # viewers cannot register


def test_cannot_register_unfinished_or_unknown(client, people):
    assert register(client, 999999, people["ds"]).status_code == 404
    assert client.post("/v1/models/versions", json={"experiment_id": 1, "intended_use": "short"},
                       headers=people["ds"]).status_code == 422


def test_full_review_and_deploy_flow_with_four_eyes(client, people):
    v = client.get("/v1/models/versions?status=candidate", headers=ADMIN).json()[0]
    vid = v["id"]
    # approve straight from candidate is refused
    assert client.post(f"/v1/models/versions/{vid}/approve", json=APPROVE, headers=people["rev"]).status_code == 409
    assert client.post(f"/v1/models/versions/{vid}/submit", json={}, headers=people["ds"]).json()["status"] == "in_review"
    # the submitter (and registrant) cannot approve their own work, even as admin... here: same data_scientist key lacks the permission
    assert client.post(f"/v1/models/versions/{vid}/approve", json=APPROVE, headers=people["ds"]).status_code == 403
    # reviewer must acknowledge warnings (small dataset => at least one warning)
    r = client.post(f"/v1/models/versions/{vid}/approve", json={**APPROVE, "acknowledge_warnings": False}, headers=people["rev"])
    assert r.status_code == 422 and "warning" in r.json()["detail"]
    # weak evidence reference is refused by validation
    assert client.post(f"/v1/models/versions/{vid}/approve", json={**APPROVE, "clinical_validation_ref": "ok"},
                       headers=people["rev"]).status_code == 422
    ok = client.post(f"/v1/models/versions/{vid}/approve", json=APPROVE, headers=people["rev"])
    assert ok.status_code == 200 and ok.json()["status"] == "approved" and ok.json()["warnings_acknowledged"] is True
    assert ok.json()["approved_by"] == "reg-rev"
    # reviewers cannot deploy; admin can
    assert client.post(f"/v1/models/versions/{vid}/deploy", json={}, headers=people["rev"]).status_code == 403
    d = client.post(f"/v1/models/versions/{vid}/deploy", json={"note": "first release"}, headers=ADMIN)
    assert d.status_code == 200 and d.json()["status"] == "deployed"
    cat = {c["slug"]: c["status"] for c in client.get("/v1/catalog", headers=ADMIN).json()}
    assert cat[SLUG] == "deployed"
    assert client.get("/v1/overview", headers=ADMIN).json()["counts"]["deployed_models"] == 1
    actions = [h["action"] for h in client.get(f"/v1/models/versions/{vid}", headers=people["view"]).json()["history"]]
    assert actions == ["registered", "submitted for clinical review", "approved", "deployed"]
    audit = {a["action"] for a in client.get("/v1/audit?limit=200", headers=ADMIN).json()}
    assert {"model.registered", "model.submitted", "model.approved", "model.deployed"} <= audit


def test_same_key_cannot_review_its_own_submission(client, dsid):
    exp = new_exp(client, dsid, seed=11)
    vid = register(client, exp, ADMIN).json()["id"]
    client.post(f"/v1/models/versions/{vid}/submit", json={}, headers=ADMIN)
    r = client.post(f"/v1/models/versions/{vid}/approve", json=APPROVE, headers=ADMIN)
    assert r.status_code == 403 and "different person" in r.json()["detail"]
    r = client.post(f"/v1/models/versions/{vid}/reject", json={"review_notes": "Not validated."}, headers=ADMIN)
    assert r.status_code == 403


def test_reject_is_terminal_and_never_deployable(client, dsid, people):
    exp = new_exp(client, dsid, seed=12)
    vid = register(client, exp, people["ds"]).json()["id"]
    client.post(f"/v1/models/versions/{vid}/submit", json={}, headers=people["ds"])
    r = client.post(f"/v1/models/versions/{vid}/reject", json={"review_notes": "Baseline not beaten."}, headers=people["rev"])
    assert r.json()["status"] == "rejected" and r.json()["review_notes"] == "Baseline not beaten."
    assert client.post(f"/v1/models/versions/{vid}/deploy", json={}, headers=ADMIN).status_code == 409
    assert client.post(f"/v1/models/versions/{vid}/approve", json=APPROVE, headers=people["rev"]).status_code == 409


def approved(client, dsid, people, seed):
    vid = register(client, new_exp(client, dsid, seed=seed), people["ds"]).json()["id"]
    client.post(f"/v1/models/versions/{vid}/submit", json={}, headers=people["ds"])
    assert client.post(f"/v1/models/versions/{vid}/approve", json=APPROVE, headers=people["rev"]).status_code == 200
    return vid


def test_supersede_then_rollback_then_no_redeploy_of_rolled_back(client, dsid, people):
    v1 = client.get(f"/v1/models/versions?model={SLUG}&status=deployed", headers=ADMIN).json()[0]["id"]
    v2 = approved(client, dsid, people, seed=21)
    assert client.post(f"/v1/models/versions/{v2}/deploy", json={}, headers=ADMIN).status_code == 200
    states = {x["id"]: x["status"] for x in client.get(f"/v1/models/versions?model={SLUG}", headers=ADMIN).json()}
    assert states[v1] == "retired" and states[v2] == "deployed"       # only one deployed at a time
    rb = client.post(f"/v1/models/{SLUG}/rollback", json={}, headers=ADMIN)
    assert rb.status_code == 200 and rb.json()["id"] == v1 and rb.json()["status"] == "deployed"
    bad = client.get(f"/v1/models/versions/{v2}", headers=ADMIN).json()
    assert bad["status"] == "retired" and bad["retired_reason"] == "rolled back"
    assert client.post(f"/v1/models/versions/{v2}/deploy", json={}, headers=ADMIN).status_code == 409
    # nothing earlier to roll back to now
    assert client.post(f"/v1/models/{SLUG}/rollback", json={}, headers=ADMIN).status_code == 409


def test_tampered_artifact_is_refused(client, dsid, people):
    v3 = approved(client, dsid, people, seed=31)
    path = storage._path(f"local://models/{SLUG}/v{client.get(f'/v1/models/versions/{v3}', headers=ADMIN).json()['version']}.joblib")
    path.write_bytes(path.read_bytes() + b"x")
    r = client.post(f"/v1/models/versions/{v3}/deploy", json={}, headers=ADMIN)
    assert r.status_code == 409 and "hash" in r.json()["detail"]


def test_retire_requires_reason_and_admin(client, people):
    dep = client.get(f"/v1/models/versions?model={SLUG}&status=deployed", headers=ADMIN).json()[0]["id"]
    assert client.post(f"/v1/models/versions/{dep}/retire", json={"reason": "too short"}, headers=people["rev"]).status_code == 403
    assert client.post(f"/v1/models/versions/{dep}/retire", json={"reason": "x"}, headers=ADMIN).status_code == 422
    r = client.post(f"/v1/models/versions/{dep}/retire", json={"reason": "Data drift found in review."}, headers=ADMIN)
    assert r.json()["status"] == "retired" and r.json()["retired_reason"].startswith("withdrawn")
    assert client.post(f"/v1/models/versions/{dep}/deploy", json={}, headers=ADMIN).status_code == 409
