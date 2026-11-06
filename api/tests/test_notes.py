"""Laboratory memory: notes attached to lab objects. Roles, subject checks, privacy guard, history, archive, recall."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/notes.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


@pytest.fixture(scope="module")
def world(client):
    d = client.post("/v1/datasets", json={"name": "notes-ds"}, headers=ADMIN).json()
    return {"dataset": d["id"], "ds": key(client, "ds-notes", "data_scientist"), "ds2": key(client, "ds2-notes", "data_scientist"),
            "viewer": key(client, "viewer-notes", "viewer"), "service": key(client, "service-notes", "service"),
            "owner": key(client, "owner-notes", "registry_owner"), "reviewer": key(client, "rev-notes", "clinical_reviewer")}


def new(client, headers, **o):
    body = {"subject_type": "general", "title": "Check district names", "body": "District spellings differ between sources.", **o}
    return client.post("/v1/notes", json=body, headers=headers)


def test_roles(client, world):
    assert new(client, world["ds"]).status_code == 201
    assert new(client, world["reviewer"], title="Reviewer note").status_code == 201
    assert new(client, ADMIN, title="Admin note").status_code == 201
    for who in ("viewer", "owner", "service"):
        assert new(client, world[who]).status_code == 403
    assert client.get("/v1/notes", headers=world["viewer"]).status_code == 200
    assert client.get("/v1/notes", headers=world["owner"]).status_code == 200
    assert client.get("/v1/notes", headers=world["service"]).status_code == 403     # the CNR machine key sees none of it
    assert client.get("/v1/notes/recall?subject_type=general", headers=world["service"]).status_code == 403


def test_subject_checks(client, world):
    ds = str(world["dataset"])
    ok = new(client, world["ds"], subject_type="dataset", subject_id=ds, kind="warning", title="Small sample")
    assert ok.status_code == 201 and ok.json()["subject_id"] == ds
    assert new(client, world["ds"], subject_type="dataset", subject_id="99999").status_code == 404
    assert new(client, world["ds"], subject_type="dataset", subject_id="abc").status_code == 404
    assert new(client, world["ds"], subject_type="dataset").status_code == 422
    assert new(client, world["ds"], subject_type="general", subject_id="1").status_code == 422
    assert new(client, world["ds"], subject_type="model", subject_id="hypertension-risk").status_code == 201
    assert new(client, world["ds"], subject_type="model", subject_id="no-such-model").status_code == 404
    assert new(client, world["ds"], subject_type="nonsense").status_code == 422
    assert new(client, world["ds"], kind="gossip").status_code == 422


def test_privacy_guard_and_validation(client, world):
    for text in ("Call 0999123456 about it", "Email jane@example.org the file", "Patient 123456789012 was dropped"):
        assert new(client, world["ds"], body=text).status_code == 422
    assert new(client, world["ds"], title="Contact jane@example.org").status_code == 422
    assert new(client, world["ds"], tags=["jane@example.org"]).status_code == 422
    assert new(client, world["ds"], body="Dropped 12 rows with missing age.").status_code == 201
    assert new(client, world["ds"], title=" ").status_code == 422
    assert new(client, world["ds"], body="x" * 4001).status_code == 422
    assert new(client, world["ds"], tags=["a"] * 9).status_code == 422
    t = new(client, world["ds"], tags=["Leakage", "leakage", "data quality"]).json()["tags"]
    assert t == ["leakage", "data quality"]


def test_edit_pin_history_archive(client, world):
    n = new(client, world["ds"], title="Threshold choice", body="Used 0.4.", kind="decision").json()
    nid = n["id"]
    assert client.patch(f"/v1/notes/{nid}", json={"body": "Hacked"}, headers=world["ds2"]).status_code == 403   # not the author
    assert client.patch(f"/v1/notes/{nid}", json={"body": "Used 0.35 after review."}, headers=world["ds"]).status_code == 200
    assert client.patch(f"/v1/notes/{nid}", json={"pinned": True}, headers=ADMIN).json()["pinned"] is True     # admin may change any
    got = client.get(f"/v1/notes/{nid}", headers=world["viewer"]).json()
    assert got["body"] == "Used 0.35 after review." and got["edits"] == 1 and got["history"][0]["body"] == "Used 0.4."
    assert client.patch(f"/v1/notes/{nid}", json={"body": "jane@example.org"}, headers=world["ds"]).status_code == 422
    assert client.delete(f"/v1/notes/{nid}", headers=world["ds2"]).status_code == 403
    assert client.delete(f"/v1/notes/{nid}", headers=world["ds"]).status_code == 204
    assert nid not in [x["id"] for x in client.get("/v1/notes?limit=200", headers=ADMIN).json()]
    assert nid in [x["id"] for x in client.get("/v1/notes?include_archived=true&limit=200", headers=ADMIN).json()]
    assert client.get(f"/v1/notes/{nid}", headers=ADMIN).json()["archived_at"] is not None      # kept, not deleted
    assert client.patch(f"/v1/notes/{nid}", json={"title": "Again"}, headers=world["ds"]).status_code == 409
    assert client.get("/v1/notes/999999", headers=ADMIN).status_code == 404


def test_filters_search_and_order(client, world):
    ds = str(world["dataset"])
    new(client, world["ds"], subject_type="dataset", subject_id=ds, kind="lesson", title="Zomba clinic lesson",
        body="Under-5 counts doubled after the registry change.", tags=["surveillance"])
    p = new(client, world["ds"], title="Pinned rule", body="Never use post-outcome columns.", kind="warning", pinned=True).json()
    lst = client.get("/v1/notes?limit=200", headers=ADMIN).json()
    assert lst[0]["pinned"] is True                                                         # pinned first
    assert [x["id"] for x in client.get("/v1/notes?kind=lesson", headers=ADMIN).json()] != []
    assert all(x["kind"] == "lesson" for x in client.get("/v1/notes?kind=lesson", headers=ADMIN).json())
    assert all("surveillance" in x["tags"] for x in client.get("/v1/notes?tag=Surveillance", headers=ADMIN).json())
    hits = client.get("/v1/notes?q=doubled", headers=ADMIN).json()
    assert len(hits) == 1 and hits[0]["title"] == "Zomba clinic lesson"
    assert client.get("/v1/notes?q=%25", headers=ADMIN).json() == []                        # % is literal, not a wildcard
    assert all(x["subject_id"] == ds for x in client.get(f"/v1/notes?subject_type=dataset&subject_id={ds}", headers=ADMIN).json())
    r = client.get(f"/v1/notes/recall?subject_type=dataset&subject_id={ds}", headers=world["viewer"]).json()
    assert any(x["id"] == p["id"] for x in r["pinned"]) and len(r["about_subject"]) >= 2
    assert all(x["archived_at"] is None for x in r["pinned"] + r["about_subject"])


def test_audit_never_holds_note_text(client, world):
    new(client, world["ds"], title="Secret-ish title", body="Distinctive body text for audit check.")
    rows = client.get("/v1/audit?limit=200", headers=ADMIN).json()
    mine = [r for r in rows if r["action"].startswith("note.")]
    assert mine and "Distinctive body text" not in str(mine) and "Secret-ish title" not in str(mine)
