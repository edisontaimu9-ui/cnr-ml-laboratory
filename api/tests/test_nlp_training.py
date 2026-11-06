"""NLP training (stage 2): refusal of small corpora, stratified split, exact scoring, verdict rules, held-out evaluation against the
rule baseline, model file round trip, and the API flow."""
import json
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/nlptrain.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATASET_STORAGE_DIR", f"{_tmp}/store")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import nlp_training as nt  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
REF = "KUHeS ethics approval 2026/041"
FOODS_EN, SYMS_EN = ["rice", "beans", "maize", "fish", "milk"], ["oedema", "fever", "cough", "vomiting"]
FOODS_NY, SYMS_NY = ["nsima", "nyemba", "nsomba", "mkaka"], ["kutupa", "malungo", "kusanza"]


def mk(text, lang, *ents):
    return {"text": text, "lang": lang, "entities": [{"start": text.index(s), "end": text.index(s) + len(s), "label": lab} for s, lab in ents]}


def synthetic(n_en=70, n_ny=50):
    docs = []
    for i in range(n_en):
        f, s = FOODS_EN[i % len(FOODS_EN)], SYMS_EN[(i // len(FOODS_EN)) % len(SYMS_EN)]
        docs.append(mk(f"The child ate {f} and had {s}. Visit {i}.", "en", ("child", "POPULATION"), (f, "FOOD"), (s, "SYMPTOM")))
    for i in range(n_ny):
        f, s = FOODS_NY[i % len(FOODS_NY)], SYMS_NY[(i // len(FOODS_NY)) % len(SYMS_NY)]
        docs.append(mk(f"Mwana amadya {f} ndipo ali ndi {s}. Ulendo {i}.", "ny", ("Mwana", "POPULATION"), (f, "FOOD"), (s, "SYMPTOM")))
    return docs


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_small_corpus_is_refused_with_reasons():
    with pytest.raises(nt.TrainingError) as e:
        nt.check_enough(synthetic(5, 5), trial=False)
    assert "documents" in str(e.value) and "FOOD" in str(e.value)


def test_corpus_without_entities_is_refused_even_in_trial():
    with pytest.raises(nt.TrainingError):
        nt.check_enough([{"text": "x", "lang": "en", "entities": []}] * 200, trial=True)


def test_trial_lowers_the_bar_and_says_so():
    warnings = nt.check_enough(synthetic(8, 6), trial=True)
    assert any("TRIAL" in w for w in warnings)


def test_enough_data_passes_but_small_language_is_flagged():
    warnings = nt.check_enough(synthetic(80, 60), trial=False)
    assert any("Only 60 ny" in w for w in warnings) and not any("TRIAL" in w for w in warnings)


def test_split_is_stratified_disjoint_and_repeatable():
    docs = synthetic(40, 20) + [mk("Ŵana wakulya somba.", "tum", ("Ŵana", "POPULATION"), ("somba", "FOOD"))]
    tr, te = nt.split(docs, 0.2, seed=3)
    assert {d["lang"] for d in te} == {"en", "ny"}                       # a language with one document stays in training
    assert not {d["text"] for d in tr} & {d["text"] for d in te} and len(tr) + len(te) == len(docs)
    assert (tr, te) == nt.split(docs, 0.2, seed=3) and te != nt.split(docs, 0.2, seed=4)[1]


def test_scoring_is_exact_and_split_by_language_and_label():
    gold = [[(0, 5, "FOOD"), (10, 15, "SYMPTOM")], [(0, 3, "FOOD")]]
    pred = [[(0, 5, "FOOD"), (10, 14, "SYMPTOM")], [(0, 3, "FOOD"), (5, 8, "FOOD")]]       # one wrong boundary, one extra
    s = nt.score(gold, pred, ["en", "ny"], ["FOOD", "SYMPTOM"])
    assert s["per_label"]["FOOD"] == {"precision": 0.667, "recall": 1.0, "f1": 0.8, "support": 2, "predicted": 3}
    assert s["per_label"]["SYMPTOM"]["f1"] == 0.0 and s["per_label"]["SYMPTOM"]["support"] == 1
    assert s["per_language"]["ny"]["precision"] == 0.5 and s["per_language_label"]["en"]["FOOD"]["f1"] == 1.0


def test_verdicts_need_enough_examples_and_a_real_difference():
    cell = lambda f1, n: {"f1": f1, "support": n}  # noqa: E731
    assert nt._verdict(cell(0.9, 5), cell(0.2, 5))["verdict"] == "too few examples"
    assert nt._verdict(cell(0.9, 40), cell(0.7, 40))["verdict"] == "model better"
    assert nt._verdict(cell(0.6, 40), cell(0.8, 40))["verdict"] == "rules better"
    assert nt._verdict(cell(0.82, 40), cell(0.80, 40))["verdict"] == "no clear difference"


def test_run_trains_evaluates_on_held_out_documents_and_saves_a_loadable_model():
    docs = synthetic(80, 60)
    res, blob = nt.run(docs, n_iter=8, seed=2)
    sp = res["split"]
    assert sp["train_documents"] + sp["test_documents"] == len(docs) and set(sp["test_by_language"]) == {"en", "ny"}
    assert set(res["model"]["per_label"]) == {"FOOD", "POPULATION", "SYMPTOM"} and set(res["rules"]["per_label"]) == set(res["model"]["per_label"])
    assert len(res["loss_by_epoch"]) == 8 and res["trial"] is False and "notice" in res
    assert res["comparison"]["overall"]["verdict"] in {"model better", "rules better", "no clear difference", "too few examples", "not comparable"}
    nlp = nt.load_model(blob)
    assert nlp.pipe_names == ["ner"] and nlp("The child ate rice and had fever.") is not None
    assert not json.dumps(res).count("Visit")                              # no training text leaks into the results


def test_boundary_mismatch_skips_documents_and_refuses_when_widespread():
    ok = synthetic(80, 60)
    bad = mk("The child ate rice.", "en", ("child", "POPULATION"))
    bad["entities"][0].update(start=4, end=7)                              # "hil": starts inside a word
    res, _ = nt.run(ok + [dict(bad, text=bad["text"] + f" {i}") for i in range(3)], n_iter=5)
    assert res["skipped_misaligned"] == 3 and any("skipped" in w for w in res["warnings"])
    with pytest.raises(nt.TrainingError):
        nt.run(ok + [dict(bad, text=bad["text"] + f" {i}") for i in range(40)], n_iter=5)


def up(client, docs, name):
    raw = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs).encode()
    r = client.post("/v1/nlp/corpora", files={"file": ("c.jsonl", raw, "application/json")}, data={"name": name, "approval_ref": REF}, headers=ADMIN)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_api_flow_refuses_small_then_trial_runs_and_reads_back(client):
    cid = up(client, synthetic(12, 8), "stage2-small")
    r = client.post("/v1/nlp/training", json={"corpus_id": cid}, headers=ADMIN)
    assert r.status_code == 422 and "Not trained" in r.text
    r = client.post("/v1/nlp/training", json={"corpus_id": cid, "trial": True, "n_iter": 5}, headers=ADMIN)
    assert r.status_code == 201 and r.json()["trial"] is True
    got = client.get(f"/v1/nlp/training/{r.json()['id']}", headers=ADMIN).json()
    assert got["status"] == "completed", got
    assert got["summary"]["trial"] is True and got["results"]["warnings"][0].startswith("TRIAL") and got["corpus_version"] == 1
    assert "artifact_uri" not in json.dumps(got)
    assert client.get("/v1/nlp/training", headers=ADMIN).json()[0]["id"] == got["id"]
    assert client.get("/v1/nlp/training/limits", headers=ADMIN).json()["minimums"]["examples_per_label"] == 50
    assert client.get("/v1/nlp/training/9999", headers=ADMIN).status_code == 404
    assert client.post("/v1/nlp/training", json={"corpus_id": 9999}, headers=ADMIN).status_code == 404


def test_api_permissions_and_audit_keep_no_text(client):
    viewer = {"X-API-Key": client.post("/v1/admin/keys", json={"name": "v-nlp-train", "role": "viewer"}, headers=ADMIN).json()["key"]}
    assert client.get("/v1/nlp/training", headers=viewer).status_code == 200
    assert client.post("/v1/nlp/training", json={"corpus_id": 1, "trial": True}, headers=viewer).status_code == 403
    assert client.get("/v1/nlp/training").status_code in (401, 403)
    log = json.dumps(client.get("/v1/admin/audit", headers=ADMIN).json()) if client.get("/v1/admin/audit", headers=ADMIN).status_code == 200 else ""
    assert "Mwana amadya" not in log and "The child ate" not in log
