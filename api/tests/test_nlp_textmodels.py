"""Validation split, language identification and document-type classifier, corpus categories, and deployment and serving of NLP models."""
import json
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/nlptext.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATASET_STORAGE_DIR", f"{_tmp}/store")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import nlp_corpus, nlp_textmodels as tm, nlp_training as nt  # noqa: E402
from app.main import app  # noqa: E402
from test_nlp_training import REF, synthetic  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
USE = "Finding entities and languages in Chichewa and English nutrition sentences."
NOTES = "Read the per-language results and the held-out errors; fine for pre-labelling only."
APPROVE = {"clinical_validation_ref": "Reviewed with KUHeS dietetics staff, 2026-11", "review_notes": NOTES, "acknowledge_warnings": True}


def with_categories(docs):
    cats = ["dietary_record", "clinical_note"]
    for i, d in enumerate(docs):
        d["category"] = cats[i % 2]
    return docs


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


# ---------------------------------------------------------------- split
def test_split3_is_disjoint_stratified_repeatable_and_keeps_the_test_set():
    docs = synthetic(60, 40)
    tr, va, te = nt.split3(docs, 0.1, 0.2, seed=3)
    texts = lambda ds: {d["text"] for d in ds}  # noqa: E731
    assert not (texts(tr) & texts(va)) and not (texts(tr) & texts(te)) and not (texts(va) & texts(te))
    assert len(tr) + len(va) + len(te) == len(docs)
    assert {d["lang"] for d in va} == {"en", "ny"} and {d["lang"] for d in te} == {"en", "ny"}
    assert nt.split3(docs, 0.1, 0.2, seed=3) == (tr, va, te)
    _, old_test = nt.split(docs, 0.2, seed=3)
    assert texts(old_test) == texts(te)                               # a validation share only takes documents out of training
    tr0, va0, te0 = nt.split3(docs, 0.0, 0.2, seed=3)
    assert va0 == [] and texts(te0) == texts(te)


def test_split3_keeps_two_training_documents_in_tiny_groups():
    tiny = [{"text": f"t{i}", "lang": "en", "entities": []} for i in range(4)]
    tr, va, te = nt.split3(tiny, 0.3, 0.2, seed=1)
    assert len(tr) >= 2 and len(te) >= 1


def test_entity_run_uses_validation_to_pick_the_epoch():
    res, blob = nt.run(synthetic(80, 60), n_iter=6, val_size=0.1)
    v = res["validation"]
    assert v["documents"] > 0 and len(v["f1_by_epoch"]) == 6 and 1 <= v["best_epoch"] <= 6
    assert res["split"]["validation_documents"] == v["documents"] and res["config"]["val_size"] == 0.1
    off, _ = nt.run(synthetic(80, 60), n_iter=5, val_size=0)
    assert off["validation"]["documents"] == 0 and off["validation"]["best_epoch"] is None


# ---------------------------------------------------------------- text models
def test_language_identifier_and_classifier_report_like_the_entity_model():
    docs = with_categories(synthetic(80, 60))
    res, blob = tm.run(docs, "langid", seed=1)
    assert res["task"] == "langid" and res["labels"] == ["en", "ny"]
    assert res["model"]["overall"]["accuracy"] >= 0.9 and res["majority_class"]["test_accuracy"] < 0.9
    assert set(res["model"]["per_label"]) == {"en", "ny"} and set(res["model"]["per_language"]) == {"en", "ny"}
    assert res["validation"]["documents"] > 0 and res["validation"]["chosen_C"] in tm.C_GRID
    assert res["comparison"]["overall"]["verdict"] and sum(map(sum, res["model"]["confusion"]["matrix"])) == res["split"]["test_documents"]
    model, meta = tm.load_model(blob)
    assert tm.predict(model, meta, "Mwana amadya nsima ndipo ali ndi kutupa.")["label"] == "ny"
    assert tm.predict(model, meta, "The child ate rice and had fever.")["label"] == "en"
    res2, _ = tm.run(docs, "classify", seed=1)
    assert res2["task"] == "classify" and set(res2["labels"]) == {"dietary_record", "clinical_note"} and res2["rules"]["overall"]["support"] > 0


def test_text_models_refuse_too_little_or_unlabelled_data():
    with pytest.raises(nt.TrainingError, match="category"):
        tm.check_enough(synthetic(80, 60), "classify", trial=False)
    with pytest.raises(nt.TrainingError, match="only one"):
        tm.check_enough(synthetic(80, 0), "langid", trial=False)
    with pytest.raises(nt.TrainingError, match="too few"):
        tm.check_enough(synthetic(80, 6), "langid", trial=False)
    assert any("TRIAL" in w for w in tm.check_enough(synthetic(12, 8), "langid", trial=True))


def test_corpus_categories_are_validated_and_counted():
    ok = nlp_corpus.parse(b'{"text":"The child ate rice.","lang":"en","category":"Dietary_Record","entities":[]}\n', fmt="jsonl")
    assert ok["valid"] and ok["docs"][0]["category"] == "dietary_record" and ok["report"]["by_category"] == {"dietary_record": 1}
    bad = nlp_corpus.parse(b'{"text":"The child ate rice.","lang":"en","category":"poem","entities":[]}\n', fmt="jsonl")
    assert not bad["valid"] and "category must be one of" in bad["errors"][0]
    plain = nlp_corpus.parse(b'{"text":"The child ate rice.","lang":"en","entities":[]}\n', fmt="jsonl")
    assert "category" not in plain["docs"][0]
    csv_ok = nlp_corpus.parse(b"text,lang,category,entities\nThe child ate rice.,en,dietary_record,child|POPULATION\n", fmt="csv")
    assert csv_ok["valid"] and csv_ok["docs"][0]["category"] == "dietary_record"


# ---------------------------------------------------------------- deployment and serving
def upload(client, docs, name):
    raw = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs).encode()
    r = client.post("/v1/nlp/corpora", files={"file": ("c.jsonl", raw, "application/json")}, data={"name": name, "approval_ref": REF}, headers=ADMIN)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def train(client, cid, task, seed=1):
    r = client.post("/v1/nlp/training", json={"corpus_id": cid, "task": task, "n_iter": 5, "seed": seed}, headers=ADMIN)
    assert r.status_code == 201, r.text
    got = client.get(f"/v1/nlp/training/{r.json()['id']}", headers=ADMIN).json()
    assert got["status"] == "completed", got
    return got["id"]


def approved(client, run, name, langs, ds, rev):
    v = client.post("/v1/nlp/models", json={"training_run_id": run, "name": name, "intended_use": USE}, headers=ds).json()
    assert client.post(f"/v1/nlp/models/{v['id']}/submit", json={}, headers=ds).status_code == 200
    r = client.post(f"/v1/nlp/models/{v['id']}/approve", json={**APPROVE, "approved_languages": langs}, headers=rev)
    assert r.status_code == 200, r.text
    return v["id"]


def test_training_endpoint_accepts_tasks_and_refuses_classifier_without_categories(client):
    cid = upload(client, synthetic(80, 60), "plain-corpus")
    r = client.post("/v1/nlp/training", json={"corpus_id": cid, "task": "classify"}, headers=ADMIN)
    assert r.status_code == 422 and "category" in r.text
    assert client.post("/v1/nlp/training", json={"corpus_id": cid, "task": "nonsense"}, headers=ADMIN).status_code == 422
    run = train(client, cid, "langid")
    got = client.get(f"/v1/nlp/training/{run}", headers=ADMIN).json()
    assert got["task"] == "langid" and got["results"]["engine"] == "sklearn-langid"
    assert client.get("/v1/nlp/training/limits", headers=ADMIN).json()["minimums"]["documents_per_class_text_models"] == 20


def test_deploy_serve_rollback_and_retire(client):
    cid = upload(client, with_categories(synthetic(80, 60)), "serve-corpus")
    ds, rev = key(client, "ds-serve", "data_scientist"), key(client, "rev-serve", "clinical_reviewer")
    viewer, service = key(client, "viewer-serve", "viewer"), key(client, "svc-serve", "service")

    # language identification: approved for English and Chichewa
    lid = approved(client, train(client, cid, "langid"), "nlp-langid-xx", ["en", "ny"], ds, rev)
    assert client.post(f"/v1/nlp/models/{lid}/deploy", json={}, headers=ds).status_code == 403            # only an administrator deploys
    assert client.get("/v1/nlp/serving", headers=viewer).status_code == 403
    assert client.post("/v1/nlp/serving/nlp-langid-xx/predict", json={"text": "Mwana ali ndi kutupa."}, headers=service).status_code == 404   # not deployed yet
    dep = client.post(f"/v1/nlp/models/{lid}/deploy", json={"note": "first release"}, headers=ADMIN)
    assert dep.status_code == 200 and dep.json()["status"] == "deployed" and dep.json()["deployed_by"]
    assert client.post(f"/v1/nlp/models/{lid}/deploy", json={}, headers=ADMIN).status_code == 409          # already deployed
    listed = client.get("/v1/nlp/serving", headers=service).json()["models"]
    assert [m["name"] for m in listed] == ["nlp-langid-xx"] and listed[0]["kind"] == "langid" and listed[0]["approved_languages"] == ["en", "ny"]
    p = client.post("/v1/nlp/serving/nlp-langid-xx/predict", json={"text": "Mwana amadya nsima ndipo ali ndi kutupa."}, headers=service)
    assert p.status_code == 200 and p.json()["result"]["label"] == "ny" and p.json()["result"]["reviewed"] is True and "notice" in p.json()
    # identifiers are refused and never echoed
    bad = client.post("/v1/nlp/serving/nlp-langid-xx/predict", json={"text": "Call 0999123456 about the child."}, headers=service)
    assert bad.status_code == 422 and "0999123456" not in bad.text
    assert client.post("/v1/nlp/serving/nlp-langid-xx/predict", json={"text": "x", "lang": "xx"}, headers=service).status_code == 422

    # entity model approved for English only: other languages fall back to the rule engine and say so
    ner = approved(client, train(client, cid, "ner"), "nlp-ner-xx", ["en"], ds, rev)
    assert client.post(f"/v1/nlp/models/{ner}/deploy", json={}, headers=ADMIN).status_code == 200
    en = client.post("/v1/nlp/serving/nlp-ner-xx/predict", json={"text": "The child ate rice and had fever.", "lang": "en"}, headers=service).json()["result"]
    assert en["model_used"] is True and en["kind"] == "ner" and isinstance(en["entities"], list)
    ny = client.post("/v1/nlp/serving/nlp-ner-xx/predict", json={"text": "Mwana ali ndi kutupa.", "lang": "ny"}, headers=service).json()["result"]
    assert ny["model_used"] is False and "not approved for ny" in ny["reason"] and ny["answered_by"].startswith("nlp-rules")

    # classifier on a second version: deploying it retires the first as superseded, and rollback brings the first back
    c1 = approved(client, train(client, cid, "classify", 1), "nlp-classify-xx", ["en", "ny"], ds, rev)
    c2 = approved(client, train(client, cid, "classify", 2), "nlp-classify-xx", ["en", "ny"], ds, rev)
    assert client.post(f"/v1/nlp/models/{c1}/deploy", json={}, headers=ADMIN).status_code == 200
    assert client.post(f"/v1/nlp/models/{c2}/deploy", json={}, headers=ADMIN).status_code == 200
    assert client.get(f"/v1/nlp/models/{c1}", headers=viewer).json()["status"] == "retired"
    assert client.get(f"/v1/nlp/models/{c1}", headers=viewer).json()["retired_reason"] == "superseded"
    assert client.post(f"/v1/nlp/models/{c1}/deploy", json={}, headers=ADMIN).status_code == 409            # only rollback restores it
    cl = client.post("/v1/nlp/serving/nlp-classify-xx/predict", json={"text": "The child ate rice and beans.", "lang": "en"}, headers=service).json()
    assert cl["model"]["version"] == 2 and cl["result"]["model_used"] is True and cl["result"]["label"] in ("dietary_record", "clinical_note")
    rb = client.post("/v1/nlp/models/by-name/nlp-classify-xx/rollback", json={}, headers=ADMIN)
    assert rb.status_code == 200 and rb.json()["version"] == 1 and rb.json()["status"] == "deployed"
    assert client.get(f"/v1/nlp/models/{c2}", headers=viewer).json()["status"] == "retired"
    assert client.post("/v1/nlp/models/by-name/nlp-classify-xx/rollback", json={}, headers=ADMIN).status_code == 409   # nothing earlier left

    # withdrawing a deployed version stops serving it
    rt = client.post(f"/v1/nlp/models/{lid}/retire", json={"reason": "Replaced by a better annotation round"}, headers=rev)
    assert rt.status_code == 200 and rt.json()["status"] == "retired"
    assert client.post("/v1/nlp/serving/nlp-langid-xx/predict", json={"text": "Mwana ali ndi kutupa."}, headers=service).status_code == 404
    assert client.post(f"/v1/nlp/models/{lid}/deploy", json={}, headers=ADMIN).status_code == 409           # withdrawn stays withdrawn


def test_a_tampered_model_file_is_not_deployed_or_served(client):
    from app import storage
    from app.db import SessionLocal
    from app.models import NlpModelVersion
    cid = upload(client, synthetic(80, 60), "tamper-serve-corpus")
    ds, rev = key(client, "ds-tam", "data_scientist"), key(client, "rev-tam", "clinical_reviewer")
    service = key(client, "svc-tam", "service")
    vid = approved(client, train(client, cid, "langid"), "nlp-tamper-xx", ["en"], ds, rev)
    with SessionLocal() as db:
        uri = db.get(NlpModelVersion, vid).artifact_uri
    good = storage.load_bytes(uri)
    storage.save_bytes(uri[len("local://"):], b"not the model")
    r = client.post(f"/v1/nlp/models/{vid}/deploy", json={}, headers=ADMIN)
    assert r.status_code == 409 and "hash" in r.text
    assert client.get(f"/v1/nlp/models/{vid}", headers=ADMIN).json()["status"] == "approved"
    # restored, it deploys; tampered again after deployment, serving refuses to load it
    storage.save_bytes(uri[len("local://"):], good)
    assert client.post(f"/v1/nlp/models/{vid}/deploy", json={}, headers=ADMIN).status_code == 200
    from app import nlp_serving
    nlp_serving.clear_cache()
    storage.save_bytes(uri[len("local://"):], b"not the model")
    p = client.post("/v1/nlp/serving/nlp-tamper-xx/predict", json={"text": "The child ate rice."}, headers=service)
    assert p.status_code == 409 and "hash" in p.text
