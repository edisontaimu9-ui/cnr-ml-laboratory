"""NLP corpora: JSONL validation, identifier refusal, versioning, storage and access rules."""
import gzip
import json
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/corpus.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATASET_STORAGE_DIR", f"{_tmp}/store")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import nlp_corpus, storage  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
REF = "KUHeS ethics approval 2026/041"


def doc(text, lang, *ents):
    return {"text": text, "lang": lang, "entities": [{"start": text.index(s), "end": text.index(s) + len(s), "label": lab} for s, lab in ents]}


def jsonl(*docs):
    return "\n".join(d if isinstance(d, str) else json.dumps(d, ensure_ascii=False) for d in docs).encode("utf-8")


GOOD = [doc("Mwana ali ndi kutupa.", "ny", ("Mwana", "POPULATION"), ("kutupa", "SYMPTOM")),
        doc("Ŵana wakulya somba.", "tum", ("Ŵana", "POPULATION"), ("somba", "FOOD")),
        doc("The child ate rice.", "en", ("child", "POPULATION"), ("rice", "FOOD"))]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


def upload(client, raw, headers=ADMIN, path="/v1/nlp/corpora", **form):
    return client.post(path, files={"file": ("c.jsonl", raw, "application/json")}, data=form, headers=headers)


def test_good_file_is_valid_and_reported():
    r = nlp_corpus.parse(jsonl(*GOOD))
    assert r["valid"] and r["errors"] == [] and len(r["docs"]) == 3
    assert r["report"]["by_language"] == {"ny": 1, "tum": 1, "en": 1} and r["report"]["entities"] == 6
    assert any("FOOD has" in w for w in r["report"]["warnings"])          # tiny corpus is flagged, not refused


@pytest.mark.parametrize("bad, expect", [
    ('{"text": "x"', "not valid JSON"),
    (json.dumps({"lang": "en"}), "needs a text string"),
    (json.dumps({"text": "Rice", "lang": "klingon", "entities": []}), "lang must be one of"),
    (json.dumps({"text": "Rice", "lang": "en", "entities": [{"start": 0, "end": 4, "label": "PHI"}]}), "label must be one of"),
    (json.dumps({"text": "Rice", "lang": "en", "entities": [{"start": 0, "end": 40, "label": "FOOD"}]}), "outside the text"),
    (json.dumps({"text": "Rice beans", "lang": "en", "entities": [{"start": 0, "end": 4, "label": "FOOD"}, {"start": 3, "end": 10, "label": "FOOD"}]}), "overlap"),
    (json.dumps({"text": "Rice beans", "lang": "en", "entities": [{"start": 0, "end": 4, "label": "FOOD"}, {"start": 2, "end": 6, "label": "FOOD"}]}), "overlap"),
    (json.dumps({"text": "Rice beans", "lang": "en", "entities": [{"start": 0, "end": 4.5, "label": "FOOD"}]}), "integer start and end"),
    (json.dumps({"text": "Rice beans", "lang": "en", "entities": "nope"}), "entities must be a list"),
    (json.dumps({"text": "Rice  beans", "lang": "en", "entities": [{"start": 4, "end": 6, "label": "FOOD"}]}), "spaces at its edges"),
])
def test_bad_lines_are_reported_with_line_numbers(bad, expect):
    r = nlp_corpus.parse(jsonl(GOOD[0], bad))
    assert not r["valid"] and r["docs"] == [] and any(e.startswith("line 2:") and expect in e for e in r["errors"])


def test_identifiers_refuse_the_whole_file_without_echoing_them():
    r = nlp_corpus.parse(jsonl(GOOD[0], doc("Call Mr Banda on 0999123456 about the child.", "en", ("child", "POPULATION"))))
    assert not r["valid"] and r["identifier_lines"] == [2] and set(r["identifier_types"]) >= {"PHONE", "NAME"}
    assert "Banda" not in json.dumps(r) and "0999123456" not in json.dumps(r)


def test_duplicates_removed_and_extra_fields_dropped():
    d = dict(GOOD[2], patient_phone="0999123456", note="extra")
    r = nlp_corpus.parse(jsonl(d, GOOD[2]))
    assert r["valid"] and len(r["docs"]) == 1 and set(r["docs"][0]) == {"text", "lang", "entities"} and r["report"]["duplicates_removed"] == 1
    assert b"0999123456" not in nlp_corpus.serialise(r["docs"])


def test_not_utf8_and_empty_files():
    with pytest.raises(nlp_corpus.CorpusError):
        nlp_corpus.parse(b"\xff\xfe\x00bad")
    with pytest.raises(nlp_corpus.CorpusError):
        nlp_corpus.parse(b"\n  \n")


def test_inspect_stores_nothing_and_reports(client):
    r = upload(client, jsonl(*GOOD), path="/v1/nlp/corpora/inspect")
    assert r.status_code == 200 and r.json()["valid"] is True and r.json()["report"]["documents"] == 3
    assert client.get("/v1/nlp/corpora", headers=ADMIN).json() == []
    bad = upload(client, jsonl(GOOD[0], "oops"), path="/v1/nlp/corpora/inspect")
    assert bad.status_code == 200 and bad.json()["valid"] is False and bad.json()["errors"][0].startswith("line 2:")


def test_create_version_and_read_back(client):
    r = upload(client, jsonl(*GOOD), name="Pilot notes", description="synthetic", approval_ref=REF)
    assert r.status_code == 201, r.text
    body = r.json()
    cid = body["id"]
    assert body["version"]["version"] == 1 and body["version"]["documents"] == 3
    assert upload(client, jsonl(*GOOD), name="Pilot notes", approval_ref=REF).status_code == 409           # name taken
    assert upload(client, jsonl(*GOOD), path=f"/v1/nlp/corpora/{cid}/versions", approval_ref=REF).status_code == 409   # same file
    more = GOOD + [doc("Amadya nsima ndi nyemba.", "ny", ("nsima", "FOOD"), ("nyemba", "FOOD"))]
    v2 = upload(client, jsonl(*more), path=f"/v1/nlp/corpora/{cid}/versions", approval_ref=REF)
    assert v2.status_code == 201 and v2.json()["version"]["version"] == 2 and v2.json()["version"]["documents"] == 4
    detail = client.get(f"/v1/nlp/corpora/{cid}", headers=ADMIN).json()
    assert [v["version"] for v in detail["versions"]] == [2, 1]
    assert client.get("/v1/nlp/corpora", headers=ADMIN).json()[0]["versions"] == 2
    stored = [json.loads(ln) for ln in gzip.decompress(storage.load_bytes(f"local://nlp{cid}/v1.jsonl.gz")).decode().splitlines()]
    assert len(stored) == 3 and stored[0]["entities"][0]["label"] == "POPULATION"
    assert client.get("/v1/nlp/corpora/9999", headers=ADMIN).status_code == 404


def test_invalid_file_is_refused_and_nothing_is_created(client):
    before = len(client.get("/v1/nlp/corpora", headers=ADMIN).json())
    r = upload(client, jsonl(GOOD[0], doc("Dr Phiri 0888123456 saw the child.", "en", ("child", "POPULATION"))), name="Leaky", approval_ref=REF)
    assert r.status_code == 422 and r.json()["detail"]["identifier_lines"] == [2]
    assert "Phiri" not in r.text and "0888123456" not in r.text
    assert upload(client, jsonl("nope"), name="Broken", approval_ref=REF).status_code == 422
    assert len(client.get("/v1/nlp/corpora", headers=ADMIN).json()) == before


def test_approval_reference_is_required(client):
    assert upload(client, jsonl(*GOOD), name="No ref", approval_ref="x").status_code == 422
    assert upload(client, jsonl(*GOOD), name="No ref").status_code == 422


def test_roles_and_audit_keep_no_text(client):
    viewer, scientist = key(client, "v-corpus", "viewer"), key(client, "ds-corpus", "data_scientist")
    assert upload(client, jsonl(*GOOD), headers=viewer, name="Viewer try", approval_ref=REF).status_code == 403
    assert client.get("/v1/nlp/corpora", headers=viewer).status_code == 200
    assert upload(client, jsonl(*GOOD), headers=scientist, name="Scientist corpus", approval_ref=REF).status_code == 201
    assert client.get("/v1/nlp/corpora/format", headers=viewer).json()["labels"]
    assert client.get("/v1/nlp/corpora").status_code == 401
    log = client.get("/v1/audit?limit=200", headers=ADMIN).text
    assert "nlp.corpus_created" in log and "Mwana ali ndi kutupa" not in log and "0999123456" not in log
