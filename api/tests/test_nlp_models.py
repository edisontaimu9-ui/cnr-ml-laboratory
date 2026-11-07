"""NLP model registry and review gate (stage 3): trial runs refused, per-language approval, separation of duties, frozen evaluation,
file integrity, and the lifecycle."""
import json
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/nlpmodels.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATASET_STORAGE_DIR", f"{_tmp}/store")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import nlp_registry as reg  # noqa: E402
from app.main import app  # noqa: E402
from test_nlp_training import REF, synthetic  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
USE = "Pre-labelling Chichewa and English nutrition sentences for annotators to correct."
NOTES = "Read the per-language results and the held-out errors; fine for pre-labelling only."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


def results(train, test, verdicts=None, skipped=0):
    return {"split": {"train_by_language": train, "test_by_language": test}, "skipped_misaligned": skipped,
            "model": {"per_language": {k: {"f1": 0.8, "support": 30} for k in test}}, "rules": {"per_language": {k: {"f1": 0.7} for k in test}},
            "comparison": {"per_language": {k: {"verdict": (verdicts or {}).get(k, "no clear difference")} for k in test}}}


def test_language_flags_block_unevaluated_and_flag_weak_languages():
    r = results({"en": 150, "ny": 40, "tum": 1}, {"en": 30, "ny": 8}, {"ny": "rules better"})
    f = reg.language_flags(r)
    assert f["en"]["flags"] == [] and not f["en"]["blocking"]
    assert f["tum"]["blocking"] and "not evaluated" in f["tum"]["flags"][0]
    assert len(f["ny"]["flags"]) == 2 and not f["ny"]["blocking"]            # few documents, and rules better


def test_approval_rules():
    r = results({"en": 150, "ny": 40, "tum": 1}, {"en": 30, "ny": 8})
    assert reg.approval_problems(r, ["en"], False) == []
    assert "cannot be approved" in reg.approval_problems(r, ["tum"], True)[0]
    assert "not trained on" in reg.approval_problems(r, ["yao"], True)[0]
    assert "acknowledge_warnings" in reg.approval_problems(r, ["en", "ny"], False)[0]
    assert reg.approval_problems(r, ["en", "ny"], True) == []
    assert "acknowledge_warnings" in reg.approval_problems(results({"en": 150}, {"en": 30}, skipped=2), ["en"], False)[0]


def upload_corpus(client, docs, name):
    raw = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs).encode()
    r = client.post("/v1/nlp/corpora", files={"file": ("c.jsonl", raw, "application/json")}, data={"name": name, "approval_ref": REF}, headers=ADMIN)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def train(client, cid, seed=1, trial=False):
    r = client.post("/v1/nlp/training", json={"corpus_id": cid, "n_iter": 5, "seed": seed, "trial": trial}, headers=ADMIN)
    assert r.status_code == 201, r.text
    got = client.get(f"/v1/nlp/training/{r.json()['id']}", headers=ADMIN).json()
    assert got["status"] == "completed", got
    return got["id"]


def test_lifecycle_per_language_approval_and_separation(client):
    cid = upload_corpus(client, synthetic(120, 30), "stage3-corpus")
    run1, run2 = train(client, cid, 1), train(client, cid, 2)
    ds, rev = key(client, "ds-nlp-model", "data_scientist"), key(client, "rev-nlp-model", "clinical_reviewer")
    viewer = key(client, "viewer-nlp-model", "viewer")

    # registering
    body = {"training_run_id": run1, "name": "nlp-ner-xx", "intended_use": USE}
    assert client.post("/v1/nlp/models", json=body, headers=viewer).status_code == 403
    assert client.post("/v1/nlp/models", json={**body, "name": "Bad Name!"}, headers=ds).status_code == 422
    assert client.post("/v1/nlp/models", json={**body, "training_run_id": 9999}, headers=ds).status_code == 404
    r = client.post("/v1/nlp/models", json=body, headers=ds)
    assert r.status_code == 201 and r.json()["version"] == 1 and r.json()["status"] == "candidate"
    vid = r.json()["id"]
    assert client.post("/v1/nlp/models", json=body, headers=ds).status_code == 409                     # same run twice
    flags = r.json()["language_review"]
    assert flags["en"]["flags"] == [] and any("few documents" in f for f in flags["ny"]["flags"])
    assert "artifact_uri" not in json.dumps(r.json())

    # the gate
    appr = {"approved_languages": ["en"], "clinical_validation_ref": "Reviewed with KUHeS dietetics staff, 2026-11", "review_notes": NOTES}
    assert client.post(f"/v1/nlp/models/{vid}/approve", json=appr, headers=rev).status_code == 409    # not submitted yet
    assert client.post(f"/v1/nlp/models/{vid}/submit", json={"note": "ready"}, headers=ds).status_code == 200
    assert client.post(f"/v1/nlp/models/{vid}/submit", json={}, headers=ds).status_code == 409
    assert client.post(f"/v1/nlp/models/{vid}/approve", json=appr, headers=ds).status_code == 403     # no approve permission
    bad = client.post(f"/v1/nlp/models/{vid}/approve", json={**appr, "approved_languages": ["yao"]}, headers=rev)
    assert bad.status_code == 422 and "not trained on" in bad.text
    weak = client.post(f"/v1/nlp/models/{vid}/approve", json={**appr, "approved_languages": ["en", "ny"]}, headers=rev)
    assert weak.status_code == 422 and "acknowledge_warnings" in weak.text
    assert client.post(f"/v1/nlp/models/{vid}/approve", json={**appr, "review_notes": "too short"}, headers=rev).status_code == 422

    ok = client.post(f"/v1/nlp/models/{vid}/approve", json={**appr, "approved_languages": ["en", "ny"], "acknowledge_warnings": True}, headers=rev)
    assert ok.status_code == 200 and ok.json()["status"] == "approved" and ok.json()["approved_languages"] == ["en", "ny"]
    assert ok.json()["warnings_acknowledged"] is True and ok.json()["clinical_validation_ref"].startswith("Reviewed")
    assert [h["action"] for h in ok.json()["history"]][:3] == ["registered", "submitted for review", "approved for en, ny"]

    # a second version, rejected; version numbers continue
    v2 = client.post("/v1/nlp/models", json={**body, "training_run_id": run2}, headers=ADMIN).json()
    assert v2["version"] == 2
    client.post(f"/v1/nlp/models/{v2['id']}/submit", json={}, headers=ADMIN)
    own = client.post(f"/v1/nlp/models/{v2['id']}/approve", json=appr, headers=ADMIN)               # admin registered and submitted it
    assert own.status_code == 403 and "different person" in own.text
    rj = client.post(f"/v1/nlp/models/{v2['id']}/reject", json={"review_notes": "Rules do better on measurements."}, headers=rev)
    assert rj.status_code == 200 and rj.json()["status"] == "rejected" and rj.json()["approved_languages"] == []
    assert client.post(f"/v1/nlp/models/{v2['id']}/retire", json={"reason": "not approved, cannot retire"}, headers=rev).status_code == 409

    # retiring an approved version
    rt = client.post(f"/v1/nlp/models/{vid}/retire", json={"reason": "Replaced by better annotation round"}, headers=rev)
    assert rt.status_code == 200 and rt.json()["status"] == "retired" and rt.json()["retired_reason"].startswith("withdrawn")

    # reads
    assert [m["status"] for m in client.get("/v1/nlp/models", headers=viewer).json()] == ["rejected", "retired"]
    assert client.get("/v1/nlp/models?status=retired", headers=viewer).json()[0]["id"] == vid
    assert client.get(f"/v1/nlp/models/{vid}", headers=viewer).json()["metrics_json"]["split"]["test_documents"] > 0
    assert client.get("/v1/nlp/models/9999", headers=viewer).status_code == 404
    assert client.get("/v1/nlp/models").status_code in (401, 403)


def test_trial_runs_cannot_be_registered(client):
    cid = upload_corpus(client, synthetic(12, 8), "stage3-trial")
    run = train(client, cid, trial=True)
    ds = key(client, "ds-nlp-trial", "data_scientist")
    r = client.post("/v1/nlp/models", json={"training_run_id": run, "name": "nlp-ner-trial", "intended_use": USE}, headers=ds)
    assert r.status_code == 409 and "trial" in r.text


def test_tampered_model_file_is_not_approved(client):
    from app import storage
    from app.db import SessionLocal
    from app.models import NlpModelVersion
    cid = upload_corpus(client, synthetic(120, 30), "stage3-tamper")
    run = train(client, cid)
    ds, rev = key(client, "ds-nlp-tamper", "data_scientist"), key(client, "rev-nlp-tamper", "clinical_reviewer")
    v = client.post("/v1/nlp/models", json={"training_run_id": run, "name": "nlp-ner-tamper", "intended_use": USE}, headers=ds).json()
    client.post(f"/v1/nlp/models/{v['id']}/submit", json={}, headers=ds)
    with SessionLocal() as db:
        uri = db.get(NlpModelVersion, v["id"]).artifact_uri
    storage.save_bytes(uri[len("local://"):], b"not the model")
    r = client.post(f"/v1/nlp/models/{v['id']}/approve", headers=rev, json={
        "approved_languages": ["en"], "clinical_validation_ref": "Reviewed with staff 2026-11", "review_notes": NOTES, "acknowledge_warnings": True})
    assert r.status_code == 409 and "hash" in r.text
