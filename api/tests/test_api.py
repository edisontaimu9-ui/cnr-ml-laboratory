import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["BOOTSTRAP_ADMIN_KEY"] = "x" * 40
os.environ["ENVIRONMENT"] = "test"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + "x" * 40}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def make_key(client, role):
    r = client.post("/v1/admin/keys", json={"name": f"t-{role}", "role": role}, headers=ADMIN)
    assert r.status_code == 201
    return r.json()


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["database"] == "ok"


def test_auth_required(client):
    assert client.get("/v1/overview").status_code == 401
    assert client.get("/v1/overview", headers={"X-API-Key": "nope"}).status_code == 401


def test_overview_and_catalog(client):
    ov = client.get("/v1/overview", headers=ADMIN).json()
    assert ov["counts"]["model_definitions"] == 18
    assert ov["counts"]["deployed_models"] == 0
    assert "never confirmed diagnoses" in ov["safety_notice"]
    cat = client.get("/v1/catalog", headers=ADMIN).json()
    assert {m["status"] for m in cat} == {"planned"}


def test_empty_registry_is_honestly_empty(client):
    for path in ("/v1/datasets", "/v1/experiments", "/v1/models/versions"):
        assert client.get(path, headers=ADMIN).json() == []


def test_viewer_cannot_read_audit_or_manage_keys(client):
    created = make_key(client, "viewer")
    h = {"X-API-Key": created["key"]}
    assert client.get("/v1/catalog", headers=h).status_code == 200
    assert client.get("/v1/audit", headers=h).status_code == 403
    assert client.post("/v1/admin/keys", json={"name": "zz", "role": "admin"}, headers=h).status_code == 403


def test_key_secret_is_never_listed_or_audited(client):
    created = make_key(client, "data_scientist")
    listing = client.get("/v1/admin/keys", headers=ADMIN).text
    audit = client.get("/v1/audit", headers=ADMIN).text
    assert created["key"] not in listing and created["key"] not in audit
    assert "key.created" in audit


def test_revoke_key(client):
    created = make_key(client, "viewer")
    h = {"X-API-Key": created["key"]}
    assert client.delete(f"/v1/admin/keys/{created['id']}", headers=ADMIN).status_code == 204
    assert client.get("/v1/catalog", headers=h).status_code == 401


def test_cannot_revoke_own_key(client):
    me_id = next(k["id"] for k in client.get("/v1/admin/keys", headers=ADMIN).json()
                 if k["name"] == "bootstrap-admin")
    assert client.delete(f"/v1/admin/keys/{me_id}", headers=ADMIN).status_code == 400
