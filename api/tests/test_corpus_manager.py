"""Corpus Manager & Dataset Builder: parsing, quality checks, approval, merging, exports and task datasets."""
import gzip
import io
import json
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/cm.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATASET_STORAGE_DIR", f"{_tmp}/store")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import corpus_manager as cm  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}


def jl(*objs):
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in objs).encode()


def ents(text, *pairs):
    return [{"start": text.index(s), "end": text.index(s) + len(s), "label": lab} for s, lab in pairs]


# ------------------------------------------------------------------ pure logic
def test_normalisation_folds_case_punctuation_and_spacing():
    assert cm.text_hash("Mwana  ali ndi kutupa.") == cm.text_hash("mwana ali ndi KUTUPA")
    assert cm.text_hash("rice") != cm.text_hash("rice and beans")


def test_txt_upload_is_unannotated_and_uses_default_language():
    r = cm.parse_upload(b"The child ate rice.\n\nMaize flour is a staple.\n", filename="a.txt", lang="en")
    assert r["valid"] and len(r["rows"]) == 2
    assert {x["lang"] for x in r["rows"]} == {"en"} and {x["status"] for x in r["rows"]} == {"unannotated"}


def test_csv_with_entities_cell_and_category():
    raw = b"text,lang,category,entities,extra\nThe child ate rice.,en,dietary_record,child|POPULATION; rice|FOOD,secret\n"
    r = cm.parse_upload(raw, filename="a.csv")
    row = r["rows"][0]
    assert row["status"] == "annotated" and row["category"] == "dietary_record"
    assert [(e["label"]) for e in row["entities"]] == ["POPULATION", "FOOD"]
    assert "extra" not in row and "secret" not in json.dumps(row)                     # unused columns never reach storage


def test_jsonl_keeps_bad_annotations_and_flags_them():
    t = "The child ate rice."
    raw = jl({"text": t, "lang": "en", "entities": [{"start": 0, "end": 999, "label": "FOOD"}]},
             {"text": "Another one.", "lang": "klingon"}, {"text": "Third text here.", "lang": "en", "entities": "nonsense"})
    r = cm.parse_upload(raw, filename="a.jsonl")
    assert r["valid"] and len(r["rows"]) == 3
    recs = [{**x, "id": i, "norm_hash": cm.text_hash(x["text"])} for i, x in enumerate(r["rows"], 1)]
    q = cm.quality(recs, ["ner", "classification", "langid"])
    assert q["invalid"]["invalid_entities"]["count"] == 2 and q["invalid"]["invalid_language"]["count"] == 1
    assert q["missing"]["missing_category"]["count"] == 3
    assert not q["ready_for_approval"]


def test_identifiers_refuse_the_whole_upload_and_report_locations_only():
    r = cm.parse_upload(b"Fine text here.\nCall Mr Banda on 0999123456 about the child.\n", filename="a.txt", lang="en")
    assert not r["valid"] and r["rows"] == [] and r["identifier_lines"] == ["line 2"]
    assert "Banda" not in json.dumps(r)


def test_unreadable_rows_are_skipped_not_fatal_and_only_csv_jsonl_txt_allowed():
    r = cm.parse_upload(b'{"text": "Good one here."}\nnot json\n{"nope": 1}\n', filename="a.jsonl", lang="en")
    assert r["valid"] and len(r["rows"]) == 1 and r["skipped_total"] == 2
    with pytest.raises(cm.CorpusManagerError):
        cm.parse_upload(b"x", filename="a.xlsx")
    with pytest.raises(cm.CorpusManagerError):
        cm.parse_upload(b"a,b\n1,2\n", filename="a.csv")                                # no text column


def test_duplicate_groups_and_resolution_keep_best_annotated_copy():
    recs = [{"id": 1, "text": "Rice.", "norm_hash": cm.text_hash("Rice."), "status": "unannotated", "lang": "en", "category": None, "entities": None, "parse_note": None},
            {"id": 2, "text": "rice", "norm_hash": cm.text_hash("rice"), "status": "annotated", "lang": "en", "category": None, "entities": [], "parse_note": None}]
    q = cm.quality(recs, ["langid"])
    assert q["duplicates"]["groups"] == 1 and q["duplicates"]["conflicting_groups"] == 1 and not q["ready_for_approval"]
    assert cm.duplicate_extras(recs) == [1]


def _src(cid, name, domain, docs):
    return {"corpus_id": cid, "name": name, "domain": domain, "snapshot_id": cid * 10, "docs": docs}


def _doc(i, text, lang="en", category=None, entities=None, status="annotated"):
    return {"id": i, "text": text, "lang": lang, "category": category, "entities": entities, "status": status, "source": None}


def test_merge_combines_complementary_annotations_and_lists_every_source():
    a = _src(1, "A", "english", [_doc(1, "The child ate rice.", category="dietary_record"), _doc(2, "Only in A.")])
    b = _src(2, "B", "clinical_nutrition", [_doc(1, "the child ate rice", entities=[])])
    res = cm.merge([a, b])
    assert res["report"]["records_out"] == 2 and res["report"]["duplicates_merged"] == 1 and res["report"]["conflicts"] == 0
    row = next(r for r in res["rows"] if "rice" in r["text"])
    assert row["category"] == "dietary_record" and row["entities"] == [] and row["corpora"] == ["A", "B"]
    assert [r["uid"] for r in res["rows"]] == ["m0000001", "m0000002"]


def test_merge_conflict_policies():
    a = _src(1, "A", "english", [_doc(1, "The child ate rice.", lang="en"), _doc(2, "Unique A text.")])
    b = _src(2, "B", "chichewa", [_doc(1, "The child ate rice.", lang="ny")])
    drop = cm.merge([a, b], "drop_conflicts")
    assert drop["report"]["conflicts"] == 1 and drop["report"]["records_out"] == 1 and drop["report"]["records_left_out_for_conflict"] == 2
    first = cm.merge([a, b], "keep_first")
    assert first["report"]["records_out"] == 2 and next(r for r in first["rows"] if "rice" in r["text"])["lang"] == "en"
    assert cm.merge([b, a], "keep_first")["rows"][0]["lang"] == "ny"                    # priority is the order given
    with pytest.raises(cm.MergeConflict) as e:
        cm.merge([a, b], "refuse")
    assert e.value.conflicts[0]["fields"] == ["lang"] and "rice" not in json.dumps(e.value.conflicts)


def test_task_datasets_select_the_right_records_and_split_repeatably():
    rows = [{"uid": f"m{i:07d}", "text": f"Text number {i} about food", "lang": "en" if i % 2 else "ny", "category": "other" if i % 3 else "lab_report",
             "entities": [] if i % 5 else None, "corpora": ["A"], "domains": ["english"]} for i in range(1, 61)]
    rows.append({"uid": "m9999999", "text": "Mixed language text", "lang": "mixed", "category": None, "entities": None, "corpora": ["A"], "domains": []})
    ner = cm.build_task(rows, "ner")
    assert ner["report"]["records"] == 48 and ner["report"]["excluded"]["no entity annotation"] == 13
    lid = cm.build_task(rows, "langid")
    assert lid["report"]["records"] == 60 and "mixed-language" in next(iter(lid["report"]["excluded"]))
    assert cm.build_task(rows, "langid", include_mixed=True)["report"]["records"] == 61
    c1, c2 = cm.build_task(rows, "classification", seed=7), cm.build_task(rows, "classification", seed=7)
    assert c1["rows"] == c2["rows"] and set(c1["report"]["by_split"]) == {"train", "validation", "test"}
    assert cm.build_task(rows, "classification", seed=8)["rows"] != c1["rows"]
    for lab in ("other", "lab_report"):                                                  # stratified: every class reaches the test set
        assert any(r["split"] == "test" and r["label"] == lab for r in c1["rows"])
    with pytest.raises(cm.CorpusManagerError):
        cm.build_task(rows, "ner", test_size=0.6)


def test_exports_round_trip_in_all_three_formats():
    rows = [{"uid": "m1", "text": "Mwana ali ndi kutupa, ŵana.", "lang": "ny", "category": "other", "entities": [{"start": 0, "end": 5, "label": "POPULATION"}],
             "annotation_status": "annotated", "source": None, "corpora": ["A", "B"], "domains": ["chichewa"]},
            {"uid": "m2", "text": "=SUM(A1)", "lang": "en", "category": None, "entities": None, "annotation_status": "annotated", "source": "x", "corpora": ["A"], "domains": []}]
    c = pd.read_csv(io.BytesIO(cm.export_rows(rows, cm.MASTER_COLUMNS, "csv")), keep_default_na=False)
    assert c.loc[0, "text"] == "Mwana ali ndi kutupa, ŵana." and json.loads(c.loc[0, "entities"])[0]["label"] == "POPULATION" and c.loc[0, "corpora"] == "A;B"
    assert c.loc[1, "text"] == "=SUM(A1)"
    safe = pd.read_csv(io.BytesIO(cm.export_rows(rows, cm.MASTER_COLUMNS, "csv", spreadsheet_safe=True)), keep_default_na=False)
    assert safe.loc[1, "text"] == "'=SUM(A1)"
    j = [json.loads(x) for x in cm.export_rows(rows, cm.MASTER_COLUMNS, "jsonl").decode().splitlines()]
    assert j[0]["entities"][0]["end"] == 5 and j[1]["entities"] is None
    p = pd.read_parquet(io.BytesIO(cm.export_rows(rows, cm.MASTER_COLUMNS, "parquet")))
    assert list(p.columns) == cm.MASTER_COLUMNS and p.loc[0, "text"] == rows[0]["text"] and len(p) == 2
    with pytest.raises(cm.CorpusManagerError):
        cm.export_rows(rows, cm.MASTER_COLUMNS, "xlsx")


# ------------------------------------------------------------------ API workflow
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


@pytest.fixture(scope="module")
def people(client):
    return {"ds": key(client, "annotator", "data_scientist"), "rev": key(client, "reviewer", "clinical_reviewer"), "view": key(client, "viewer", "viewer")}


def make(client, h, name, domain, **kw):
    r = client.post("/v1/cm/corpora", json={"name": name, "domain": domain, **kw}, headers=h)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def upload(client, h, cid, raw, name="c.jsonl", **form):
    return client.post(f"/v1/cm/corpora/{cid}/uploads", files={"file": (name, raw, "application/octet-stream")}, data=form, headers=h)


def good_docs(prefix, lang, n=3):
    out = []
    for i in range(n):
        t = f"{prefix} child number {i} ate rice"
        out.append({"text": t, "lang": lang, "category": "dietary_record", "entities": ents(t, ("child", "POPULATION"), ("rice", "FOOD"))})
    return out


def approve_path(client, people, cid):
    assert client.post(f"/v1/cm/corpora/{cid}/submit", headers=people["ds"]).status_code == 200
    return client.post(f"/v1/cm/corpora/{cid}/approve", json={"note": "Checked by clinical reviewer"}, headers=people["rev"])


def test_roles_and_validation_of_new_corpus(client, people):
    body = {"name": "Perm check", "domain": "english"}
    assert client.post("/v1/cm/corpora", json=body, headers=people["view"]).status_code == 403
    assert client.post("/v1/cm/corpora", json={"name": "Bad domain", "domain": "cooking"}, headers=people["ds"]).status_code == 422
    assert client.post("/v1/cm/corpora", json={**body, "tasks": ["ner", "poetry"]}, headers=people["ds"]).status_code == 422
    cid = make(client, people["ds"], "Perm check", "english")
    assert client.post("/v1/cm/corpora", json=body, headers=people["ds"]).status_code == 409
    assert client.get("/v1/cm/corpora", headers=people["view"]).status_code == 200               # viewers see counts
    assert client.get(f"/v1/cm/corpora/{cid}/records", headers=people["view"]).status_code == 403  # but not text
    info = client.get("/v1/cm/info", headers=people["view"]).json()
    assert set(info["domains"]) >= {"english", "chichewa", "clinical_nutrition", "food_nutrients", "community_nutrition", "research_documents"}


def test_full_workflow_from_upload_to_master_and_task_datasets(client, people):
    ds, rev = people["ds"], people["rev"]
    en = make(client, ds, "English nutrition", "english", default_lang="en")
    ny = make(client, ds, "Chichewa nutrition", "chichewa", default_lang="ny")

    # English corpus arrives as TXT: nothing is annotated yet.
    txt = "\n".join(["Breastfeed the baby for six months.", "Give the child nsima and beans.", "Wash hands before feeding.", "Give the child nsima and beans!"]).encode()
    r = upload(client, ds, en, txt, name="en.txt")
    assert r.status_code == 201 and r.json()["report"]["unannotated"] == 4
    assert upload(client, ds, en, txt, name="en.txt").status_code == 409                          # same file twice
    q = client.get(f"/v1/cm/corpora/{en}/quality", headers=ds).json()
    assert q["missing"]["records"] == 4 and q["duplicates"]["groups"] == 1 and not q["ready_for_approval"]
    assert client.post(f"/v1/cm/corpora/{en}/submit", headers=ds).status_code == 422             # blockers listed
    assert client.post(f"/v1/cm/corpora/{en}/approve", json={"note": "no way"}, headers=rev).status_code == 409

    # Annotate: duplicates resolved, category and entities added, statuses set.
    assert client.post(f"/v1/cm/corpora/{en}/resolve-duplicates", headers=ds).json()["rejected"] == 1
    recs = client.get(f"/v1/cm/corpora/{en}/records", headers=ds).json()["records"]
    live = [x for x in recs if x["status"] != "rejected"]
    assert len(live) == 3 and any(x["status"] == "rejected" for x in recs)                       # rejected, never deleted
    for x in live:
        words = [("baby", "POPULATION")] if "baby" in x["text"] else [("child", "POPULATION"), ("nsima", "FOOD")] if "nsima" in x["text"] else []
        bad = client.patch(f"/v1/cm/corpora/{en}/records/{x['id']}", json={"category": "poetry"}, headers=ds)
        assert bad.status_code == 422
        r = client.patch(f"/v1/cm/corpora/{en}/records/{x['id']}", json={"category": "dietary_record", "entities": ents(x["text"], *words), "status": "annotated"}, headers=ds)
        assert r.status_code == 200, r.text
    assert client.patch(f"/v1/cm/corpora/{en}/records/{live[0]['id']}", json={"entities": [{"start": 0, "end": 999, "label": "FOOD"}]}, headers=ds).status_code == 422
    # reviewing your own annotation is refused
    assert client.patch(f"/v1/cm/corpora/{en}/records/{live[0]['id']}", json={"status": "reviewed"}, headers=ds).status_code == 403
    assert client.patch(f"/v1/cm/corpora/{en}/records/{live[0]['id']}", json={"status": "reviewed"}, headers=ADMIN).status_code == 200

    # Chichewa corpus arrives annotated as JSONL.
    r = upload(client, ds, ny, jl(*good_docs("Mwana", "ny")))
    assert r.status_code == 201 and r.json()["report"]["annotated"] == 3

    # Approval needs a different person than the last editor, and freezes a snapshot.
    assert client.post(f"/v1/cm/corpora/{en}/submit", headers=ds).status_code == 200
    assert client.post(f"/v1/cm/corpora/{en}/approve", json={"note": "Reviewed all records"}, headers=ds).status_code == 403
    r = client.post(f"/v1/cm/corpora/{en}/approve", json={"note": "Reviewed all records"}, headers=rev)
    assert r.status_code == 200 and r.json()["state"] == "approved" and r.json()["snapshots"][0]["documents"] == 3
    assert approve_path(client, people, ny).status_code == 200

    # Only approved corpora merge; a draft is refused.
    draft = make(client, ds, "Still draft", "other")
    assert client.post("/v1/cm/masters", json={"corpus_ids": [en, draft]}, headers=ds).status_code == 422
    assert client.post("/v1/cm/masters/preview", json={"corpus_ids": [en, ny], "on_conflict": "bad"}, headers=ds).status_code == 422

    runs_before = client.get("/v1/nlp/training", headers=ds).json()
    prev = client.post("/v1/cm/masters/preview", json={"corpus_ids": [en, ny]}, headers=ds).json()["report"]
    assert prev["records_out"] == 6 and client.get("/v1/cm/masters", headers=ds).json() == []   # preview stores nothing
    m1 = client.post("/v1/cm/masters", json={"corpus_ids": [en, ny], "note": "first"}, headers=ds)
    assert m1.status_code == 201, m1.text
    m1 = m1.json()
    assert m1["version"] == 1 and m1["records"] == 6 and [s["name"] for s in m1["sources"]] == ["English nutrition", "Chichewa nutrition"]
    assert m1["report"]["by_language"] == {"en": 3, "ny": 3}

    # Corpora stay the original sources: editing one reopens it but does not touch the master.
    x = live[1]
    assert client.patch(f"/v1/cm/corpora/{en}/records/{x['id']}", json={"category": "other"}, headers=ds).status_code == 200
    assert client.get(f"/v1/cm/corpora/{en}", headers=ds).json()["state"] == "draft"
    assert client.post("/v1/cm/masters", json={"corpus_ids": [en, ny]}, headers=ds).status_code == 422
    m1_again = client.get(f"/v1/cm/masters/{m1['id']}", headers=ds).json()
    assert m1_again["sha256"] == m1["sha256"] and m1_again["report"]["by_category"] == {"dietary_record": 6}
    m2 = client.post("/v1/cm/masters", json={"corpus_ids": [ny]}, headers=ds).json()
    assert m2["name"] == "cnr-nutrition-master" and m2["version"] == 2                                  # the same name continues its version series
    assert client.post("/v1/cm/masters", json={"name": "Other series!", "corpus_ids": [ny]}, headers=ds).json()["name"] == "Other-series"

    # The preserved original upload still reads as it was ingested.
    up = client.get(f"/v1/cm/corpora/{en}", headers=ds).json()["uploads"][0]
    orig = client.get(f"/v1/cm/corpora/{en}/uploads/{up['id']}/original", headers=ds)
    rows = [json.loads(x) for x in gzip.decompress(orig.content).decode().splitlines()]
    assert len(rows) == 4 and all(r["status"] == "unannotated" and r["category"] is None for r in rows)

    # Exports.
    mid = m1["id"]
    csv_r = client.get(f"/v1/cm/masters/{mid}/export?format=csv", headers=ds)
    assert csv_r.status_code == 200 and len(pd.read_csv(io.BytesIO(csv_r.content))) == 6 and csv_r.headers["X-Master-Version"] == "1"
    assert len(client.get(f"/v1/cm/masters/{mid}/export?format=jsonl", headers=ds).content.splitlines()) == 6
    pq = client.get(f"/v1/cm/masters/{mid}/export?format=parquet", headers=ds)
    assert len(pd.read_parquet(io.BytesIO(pq.content))) == 6
    assert client.get(f"/v1/cm/masters/{mid}/export?format=xlsx", headers=ds).status_code == 422
    assert client.get(f"/v1/cm/masters/{mid}/export?format=csv", headers=people["view"]).status_code == 403

    # Task datasets.
    s = client.get(f"/v1/cm/masters/{mid}/tasks/ner/summary", headers=people["view"])
    assert s.status_code == 200 and s.json()["records"] == 6 and "text" not in json.dumps(s.json())
    ner = client.get(f"/v1/cm/masters/{mid}/tasks/ner/export?format=jsonl&test_size=0&val_size=0", headers=ds)
    docs = [json.loads(x) for x in ner.content.decode().splitlines()]
    assert len(docs) == 6 and set(docs[0]) == {"uid", "text", "lang", "entities", "split"}
    cls = pd.read_csv(io.BytesIO(client.get(f"/v1/cm/masters/{mid}/tasks/classification/export?format=csv", headers=ds).content))
    assert list(cls.columns) == ["uid", "text", "label", "lang", "split"] and set(cls["label"]) == {"dietary_record"}
    lid = pd.read_parquet(io.BytesIO(client.get(f"/v1/cm/masters/{mid}/tasks/langid/export?format=parquet&seed=3", headers=ds).content))
    assert set(lid["lang"]) == {"en", "ny"} and list(lid.columns) == ["uid", "text", "lang", "split"]
    assert client.get(f"/v1/cm/masters/{mid}/tasks/poetry/summary", headers=ds).status_code == 422
    assert client.get(f"/v1/cm/masters/{mid}/tasks/ner/export?test_size=0.9", headers=ds).status_code == 422

    # Building datasets never trains anything.
    assert client.get("/v1/nlp/training", headers=ds).json() == runs_before


def test_cross_corpus_conflict_surfaces_in_quality_and_refuse_policy(client, people):
    ds = people["ds"]
    a = make(client, ds, "Conflict A", "english", tasks=["langid"])
    b = make(client, ds, "Conflict B", "chichewa", tasks=["langid"])
    assert upload(client, ds, a, jl({"text": "Nsima with relish is eaten daily.", "lang": "en"}, {"text": "Unique to A here.", "lang": "en"})).status_code == 201
    assert upload(client, ds, b, jl({"text": "Nsima with relish is eaten daily!", "lang": "ny"})).status_code == 201
    q = client.get(f"/v1/cm/corpora/{b}/quality", headers=ds).json()
    assert q["also_in_other_corpora"] == {"Conflict A": 1} and q["ready_for_approval"]
    assert approve_path(client, people, a).status_code == 200 and approve_path(client, people, b).status_code == 200
    r = client.post("/v1/cm/masters", json={"corpus_ids": [a, b], "on_conflict": "refuse"}, headers=ds)
    assert r.status_code == 422 and r.json()["detail"]["conflicts"][0]["fields"] == ["lang"]
    keep = client.post("/v1/cm/masters", json={"name": "conflicts", "corpus_ids": [b, a], "on_conflict": "keep_first"}, headers=ds).json()
    assert keep["records"] == 2 and keep["report"]["conflicts"] == 1
    drop = client.post("/v1/cm/masters", json={"name": "conflicts", "corpus_ids": [a, b]}, headers=ds).json()
    assert drop["version"] == 2 and drop["records"] == 1


def test_identifiers_are_refused_whole_and_audited_without_text(client, people):
    ds = people["ds"]
    cid = make(client, ds, "Identifier check", "clinical_nutrition", default_lang="en")
    r = upload(client, ds, cid, b"A fine sentence.\nPatient Mr Phiri phone 0888123456 was seen.\n", name="n.txt")
    assert r.status_code == 422 and r.json()["detail"]["identifier_lines"] == ["line 2"] and "Phiri" not in r.text
    assert client.get(f"/v1/cm/corpora/{cid}", headers=ds).json()["records"] == 0
    log = client.get("/v1/audit?limit=500", headers=ADMIN)
    assert log.status_code == 200
    actions = {e["action"] for e in log.json()}
    assert "cm.upload_rejected" in actions
    assert "Phiri" not in log.text and "0888123456" not in log.text                              # counts and locations only


def test_inspect_stores_nothing_and_bulk_updates_are_all_or_none(client, people):
    ds = people["ds"]
    cid = make(client, ds, "Bulk check", "food_nutrients", tasks=["langid"])
    raw = b"text,lang\nBeans are rich in protein.,\nGroundnuts give energy and fat.,\n"
    chk = client.post(f"/v1/cm/corpora/{cid}/inspect", files={"file": ("a.csv", raw)}, headers=ds).json()
    assert chk["valid"] and chk["report"]["by_language"] == {"(missing)": 2}
    assert client.get(f"/v1/cm/corpora/{cid}", headers=ds).json()["records"] == 0
    assert upload(client, ds, cid, raw, name="a.csv").status_code == 201
    ids = [x["id"] for x in client.get(f"/v1/cm/corpora/{cid}/records?problem=missing", headers=ds).json()["records"]]
    assert len(ids) == 2
    bad = client.post(f"/v1/cm/corpora/{cid}/records/bulk", json={"ids": ids, "set": {"lang": "klingon"}}, headers=ds)
    assert bad.status_code == 422 and client.get(f"/v1/cm/corpora/{cid}/records?problem=missing", headers=ds).json()["total"] == 2
    ok = client.post(f"/v1/cm/corpora/{cid}/records/bulk", json={"ids": ids, "set": {"lang": "english", "status": "annotated"}}, headers=ds)
    assert ok.status_code == 200 and ok.json()["corpus"]["by_status"] == {"annotated": 2}          # "english" is read as en
    assert client.get(f"/v1/cm/corpora/{cid}/quality", headers=ds).json()["ready_for_approval"]


def test_ner_export_is_accepted_by_the_existing_nlp_corpus_validator():
    from app import nlp_corpus
    t = "The child ate rice."
    rows = [{"uid": "m1", "text": t, "lang": "en", "category": None, "entities": ents(t, ("child", "POPULATION"), ("rice", "FOOD")), "corpora": ["A"], "domains": []},
            {"uid": "m2", "text": "Nothing to mark here", "lang": "en", "category": None, "entities": [], "corpora": ["A"], "domains": []}]
    task = cm.build_task(rows, "ner", test_size=0, val_size=0)
    res = nlp_corpus.parse(cm.export_rows(task["rows"], task["columns"], "jsonl"), filename="ner.jsonl")
    assert res["valid"] and res["report"]["documents"] == 2 and res["report"]["entities"] == 2


# ------------------------------------------------------------------ more upload formats
def _docx(*paras):
    import zipfile
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paras)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f"<w:document {ns}><w:body>{body}</w:body></w:document>")
    return buf.getvalue()


def test_tsv_json_and_xlsx_read_like_csv_and_jsonl():
    t = cm.parse_upload(b"text\tlang\tcategory\nThe child ate rice.\ten\tdietary_record\n", filename="a.tsv")
    assert t["rows"][0]["category"] == "dietary_record" and t["report"]["source_format"] == "tsv"
    j = cm.parse_upload(json.dumps({"documents": [{"text": "Beans are food.", "lang": "en", "entities": []}, {"nope": 1}]}).encode(), filename="a.json")
    assert len(j["rows"]) == 1 and j["rows"][0]["status"] == "annotated" and j["skipped_total"] == 1
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Text", "Lang", "Entities"])
    ws.append(["The child ate rice.", "en", "child|POPULATION"])
    ws.append([None, None, None])
    ws.append(["Maize flour is a staple.", "en", None])
    buf = io.BytesIO()
    wb.save(buf)
    x = cm.parse_upload(buf.getvalue(), filename="a.xlsx")
    assert len(x["rows"]) == 2 and x["rows"][0]["entities"][0]["label"] == "POPULATION" and x["rows"][1]["status"] == "unannotated"
    with pytest.raises(cm.CorpusManagerError):
        cm.parse_upload(b"not a workbook", filename="a.xlsx")


def test_documents_are_split_into_unannotated_sentences():
    para = "Breastfeed the baby for six months. Then add nsima and beans to the diet."
    for name, raw in (("a.docx", _docx(para)), ("a.md", ("# Feeding\n\n" + para).encode()), ("a.html", f"<html><body><p>{para}</p></body></html>".encode())):
        r = cm.parse_upload(raw, filename=name, lang="en")
        assert r["valid"] and len(r["rows"]) >= 2 and {x["status"] for x in r["rows"]} == {"unannotated"} and r["report"]["source_format"] == "document", name
    assert cm.parse_upload(_docx("Fine text here for the test, with enough words."), filename="noext", fmt="docx", lang="en")["valid"]
    with pytest.raises(cm.CorpusManagerError):
        cm.parse_upload(b"not a word file", filename="a.docx", lang="en")


def test_annotation_tool_formats_bio_and_conll_keep_entities():
    bio = b"# lang = en\nThe O\nchild B-POPULATION\nate O\nrice B-FOOD\n.  O\n"
    r = cm.parse_upload(bio, filename="a.bio")
    assert r["valid"] and [e["label"] for e in r["rows"][0]["entities"]] == ["POPULATION", "FOOD"] and r["rows"][0]["lang"] == "en"
    with pytest.raises(cm.CorpusManagerError):
        cm.parse_upload(b"x", filename="a.exe")


def test_identifiers_are_refused_in_every_new_format_too():
    raw = _docx("Call Mr Banda on 0999123456 about the child today please.")
    r = cm.parse_upload(raw, filename="a.docx", lang="en")
    assert not r["valid"] and r["identifier_lines_total"] == 1


def test_api_accepts_a_document_upload(client, people):
    ds = people["ds"]
    cid = make(client, ds, "Docs upload", "research_documents", default_lang="en", tasks=["langid"])
    r = upload(client, ds, cid, _docx("Exclusive breastfeeding protects infants. Complementary foods start at six months."), name="paper.docx")
    assert r.status_code == 201 and r.json()["report"]["source_format"] == "document" and r.json()["upload_id"]
    assert "docx" in client.get("/v1/cm/info", headers=ds).json()["upload_accept"]
