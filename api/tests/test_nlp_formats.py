"""NLP corpora: every supported input format ends up as the same validated canonical documents."""
import gzip
import io
import json
import os
import tempfile
import zipfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/formats.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATASET_STORAGE_DIR", f"{_tmp}/store")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import nlp_corpus, storage  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
REF = "KUHeS ethics approval 2026/041"


def spans(doc):
    return [(doc["text"][e["start"]:e["end"]], e["label"]) for e in doc["entities"]]


def parse(raw, **kw):
    r = nlp_corpus.parse(raw.encode("utf-8") if isinstance(raw, str) else raw, **kw)
    assert r["valid"], r["errors"] + r["identifier_lines"]
    return r


# ---------------------------------------------------------------- JSON and JSONL
def test_jsonl_accepts_list_entities_and_doccano_style():
    raw = "\n".join([json.dumps({"text": "Mwana ali ndi kutupa.", "lang": "ny", "entities": [[0, 5, "POPULATION"]]}),
                     json.dumps({"text": "The child ate rice.", "lang": "en", "label": [[4, 9, "population"], [14, 18, "FOOD"]]})])
    r = parse(raw, filename="a.jsonl")
    assert [spans(d) for d in r["docs"]] == [[("Mwana", "POPULATION")], [("child", "POPULATION"), ("rice", "FOOD")]]
    assert r["report"]["source_format"] == "jsonl"


def test_json_array_object_and_spacy_pairs_with_default_lang():
    arr = json.dumps([{"text": "The child ate rice.", "entities": [{"start": 14, "end": 18, "label": "FOOD"}]}])
    assert parse(arr, filename="a.json", lang="en")["docs"][0]["lang"] == "en"
    obj = json.dumps({"documents": [{"text": "Rice.", "lang": "en", "entities": []}]})
    assert len(parse(obj, filename="a.json")["docs"]) == 1
    pairs = json.dumps([["Mwana ali bwino.", {"entities": [[0, 5, "POPULATION"]]}]])
    assert spans(parse(pairs, filename="a.json", lang="ny")["docs"][0]) == [("Mwana", "POPULATION")]
    bad = nlp_corpus.parse(arr.encode(), filename="a.json")                      # no language anywhere
    assert not bad["valid"] and "it is missing" in bad["errors"][0]
    with pytest.raises(nlp_corpus.CorpusError):
        nlp_corpus.parse(b'{"text": "x"', filename="a.json")


def test_label_map_renames_and_drops_labels():
    t = "Dr visited the child who ate rice."
    ent = lambda w, label: {"start": t.index(w), "end": t.index(w) + len(w), "label": label}   # noqa: E731
    raw = json.dumps({"text": t, "lang": "en", "entities": [ent("child", "pop"), ent("rice", "FOOD_ITEM"), ent("Dr", "PER")]})
    assert not nlp_corpus.parse(raw.encode(), filename="a.jsonl")["valid"]
    r = parse(raw, filename="a.jsonl", label_map={"pop": "POPULATION", "FOOD_ITEM": "FOOD", "PER": ""})
    assert spans(r["docs"][0]) == [("child", "POPULATION"), ("rice", "FOOD")]


# ---------------------------------------------------------------- CSV
def test_csv_one_row_per_entity():
    raw = ("doc_id,text,lang,start,end,label\n1,Mwana ali ndi kutupa.,ny,0,5,POPULATION\n1,Mwana ali ndi kutupa.,ny,14,20,SYMPTOM\n"
           "2,The child ate rice.,en,,,\n")
    r = parse(raw, filename="a.csv")
    assert [spans(d) for d in r["docs"]] == [[("Mwana", "POPULATION"), ("kutupa", "SYMPTOM")], []]
    bad = nlp_corpus.parse("text,lang,start,end,label\nRice,en,x,4,FOOD\n".encode(), filename="a.csv")
    assert not bad["valid"] and "start and end must be whole numbers" in bad["errors"][0]


def test_csv_entities_cell_matches_whole_words_every_time():
    raw = 'text,lang,entities\nRice and rice with beans.,en,"rice|FOOD; beans|FOOD"\nMwana ali ndi kutupa.,ny,Mwana|POPULATION;kutupa|SYMPTOM\n'
    r = parse(raw, filename="a.csv")
    assert spans(r["docs"][0]) == [("Rice", "FOOD"), ("rice", "FOOD"), ("beans", "FOOD")] and spans(r["docs"][1])[1] == ("kutupa", "SYMPTOM")
    missing = nlp_corpus.parse(b"text,lang,entities\nRice.,en,beans|FOOD\n", filename="a.csv")
    assert not missing["valid"] and "was not found in the text" in missing["errors"][0] and missing["errors"][0].startswith("row 2:")
    plain = parse("text,lang\nThe child ate rice.,en\n", filename="a.csv")
    assert plain["docs"][0]["entities"] == [] and any("no entity columns" in w for w in plain["report"]["warnings"])
    with pytest.raises(nlp_corpus.CorpusError):
        nlp_corpus.parse(b"sentence,lang\nx,en\n", filename="a.csv")


# ---------------------------------------------------------------- BIO / IOB2 and CoNLL
BIO = """# lang = ny
Mwana B-POPULATION
ali O
ndi O
kutupa B-SYMPTOM
. O

# lang = en
The O
child B-POPULATION
ate O
brown B-FOOD
rice I-FOOD
. O
"""


def test_bio_iob2_builds_text_and_entities():
    r = parse(BIO, filename="a.bio")
    assert [d["lang"] for d in r["docs"]] == ["ny", "en"]
    assert r["docs"][0]["text"] == "Mwana ali ndi kutupa ."
    assert spans(r["docs"][0]) == [("Mwana", "POPULATION"), ("kutupa", "SYMPTOM")]
    assert spans(r["docs"][1]) == [("child", "POPULATION"), ("brown rice", "FOOD")]


def test_bio_is_strict_about_iob2_and_reports_the_line():
    bad = nlp_corpus.parse(b"# lang = en\nThe O\nrice I-FOOD\n", filename="a.iob2")
    assert not bad["valid"] and bad["errors"][0].startswith("line 3:") and "B-" in bad["errors"][0]
    odd = nlp_corpus.parse(b"# lang = en\nThe O\nrice FOOD\n", filename="a.bio")
    assert not odd["valid"] and "not O, B-LABEL or I-LABEL" in odd["errors"][0]
    nolang = nlp_corpus.parse(b"The O\nrice B-FOOD\n", filename="a.bio")
    assert not nolang["valid"] and "it is missing" in nolang["errors"][0]
    assert parse("The O\nrice B-FOOD\n", filename="a.bio", lang="en")["docs"][0]["lang"] == "en"


def test_conll_uses_last_column_skips_docstart_and_accepts_iob1():
    raw = "-DOCSTART- -X- -X- O\n\nAmadya NNP B-NP O\nnsima NN I-NP I-FOOD\nndi CC O O\nnyemba NN I-NP I-FOOD\n. . O O\n\nMwana NN B-NP B-POPULATION\nakudya VB O O\n"
    r = parse(raw, filename="train.conll", lang="ny")
    assert len(r["docs"]) == 2 and spans(r["docs"][0]) == [("nsima", "FOOD"), ("nyemba", "FOOD")]
    assert spans(r["docs"][1]) == [("Mwana", "POPULATION")]
    iob1_adjacent = "Rice I-FOOD\nbeans B-FOOD\n"                              # IOB1 marks a new adjacent entity of the same type with B-
    assert [e for e in parse(iob1_adjacent, filename="a.conll", lang="en")["docs"][0]["entities"]] == [
        {"start": 0, "end": 4, "label": "FOOD"}, {"start": 5, "end": 10, "label": "FOOD"}]


def test_identifiers_in_tagged_text_are_refused():
    r = nlp_corpus.parse(b"# lang = en\nSeen O\nby O\nDr O\nBanda O\ntoday O\n", filename="a.bio")
    assert not r["valid"] and r["identifier_lines"] == ["line 2"] and "NAME" in r["identifier_types"]
    assert "Banda" not in json.dumps(r)


# ---------------------------------------------------------------- spaCy DocBin
def docbin(lang_in_user_data=True):
    spacy = pytest.importorskip("spacy")
    from spacy.tokens import DocBin, Span
    nlp = spacy.blank("xx")
    db = DocBin(store_user_data=True)
    d = nlp.make_doc("Mwana ali ndi kutupa.")
    d.ents = [Span(d, 0, 1, label="POPULATION"), Span(d, 3, 4, label="SYMPTOM")]
    if lang_in_user_data:
        d.user_data["lang"] = "ny"
    db.add(d)
    db.add(nlp.make_doc("Rice."))
    return db.to_bytes()


def test_docbin_reads_entities_and_language():
    r = parse(docbin(), filename="train.spacy", lang="en")
    assert r["docs"][0]["lang"] == "ny" and spans(r["docs"][0]) == [("Mwana", "POPULATION"), ("kutupa", "SYMPTOM")]
    assert r["docs"][1]["lang"] == "en" and r["docs"][1]["entities"] == []
    bad = nlp_corpus.parse(docbin(False), filename="train.spacy")
    assert not bad["valid"] and "it is missing" in bad["errors"][0]


def test_docbin_garbage_is_refused():
    pytest.importorskip("spacy")
    with pytest.raises(nlp_corpus.CorpusError):
        nlp_corpus.parse(b"not a docbin at all", filename="x.spacy")


# ---------------------------------------------------------------- BRAT standoff
def brat_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


def test_brat_pairs_language_folders_and_ignored_relations():
    z = brat_zip({"ny/note1.txt": "Mwana ali ndi kutupa.", "ny/note1.ann": "T1\tPOPULATION 0 5\tMwana\nT2\tSYMPTOM 14 20\tkutupa\nR1\tCauses Arg1:T1 Arg2:T2\n",
                  "note2.txt": "The child ate rice.", "note2.ann": "T1\tFood 14 18\trice\n", "annotation.conf": "[entities]\n", "__MACOSX/x.txt": "junk"})
    r = parse(z, filename="brat.zip", lang="en")
    by = {d["text"]: d for d in r["docs"]}
    assert by["Mwana ali ndi kutupa."]["lang"] == "ny" and spans(by["Mwana ali ndi kutupa."]) == [("Mwana", "POPULATION"), ("kutupa", "SYMPTOM")]
    assert by["The child ate rice."]["lang"] == "en" and spans(by["The child ate rice."]) == [("rice", "FOOD")]
    assert any("1 BRAT annotations that are not entities" in w for w in r["report"]["warnings"])


@pytest.mark.parametrize("files, expect", [
    ({"a.txt": "Rice beans.", "a.ann": "T1\tFOOD 0 4;5 10\tRice beans\n"}, "discontinuous"),
    ({"a.txt": "Rice beans.", "a.ann": "T1\tFOOD 0 4\tWrong\n"}, "does not match its offsets"),
    ({"a.txt": "Rice beans."}, "needs both a .txt and a .ann"),
    ({"a.txt": "Rice beans.", "a.ann": "T1\tFOOD zero 4\tRice\n"}, "could not be read"),
])
def test_brat_problems_are_reported_by_file(files, expect):
    r = nlp_corpus.parse(brat_zip(files), filename="b.zip", lang="en")
    assert not r["valid"] and expect in r["errors"][0] and r["errors"][0].startswith("a")


def test_brat_long_text_is_split_at_line_breaks_and_bad_zips_refused():
    text = "Rice is good.\n" * 700                                               # about 9,800 characters
    last = text.rindex("Rice")
    ann = f"T1\tFOOD 0 4\tRice\nT2\tFOOD {last} {last + 4}\tRice\n"
    r = nlp_corpus.parse(brat_zip({"long.txt": text, "long.ann": ann}), filename="b.zip", lang="en")
    assert r["valid"] and len(r["docs"]) == 2 and all(len(d["text"]) <= 8000 for d in r["docs"])
    assert [len(d["entities"]) for d in r["docs"]] == [1, 1] and spans(r["docs"][1]) == [("Rice", "FOOD")]
    assert r["docs"][1]["entities"][0]["start"] == len(r["docs"][1]["text"]) - len("Rice is good.\n")
    with pytest.raises(nlp_corpus.CorpusError):
        nlp_corpus.parse(b"PK not really a zip", filename="b.zip")


def test_brat_identifiers_are_refused():
    r = nlp_corpus.parse(brat_zip({"a.txt": "Call 0999123456 now.", "a.ann": ""}), filename="b.zip", lang="en")
    assert not r["valid"] and r["identifier_lines"] == ["a"] and "0999123456" not in json.dumps(r)


# ---------------------------------------------------------------- detection and API
def test_format_detection_and_errors():
    for bad_kwargs in ({"filename": "notes.txt"}, {"filename": "a.jsonl", "fmt": "xml"}):
        with pytest.raises(nlp_corpus.CorpusError):
            nlp_corpus.parse(b'{"text":"x"}', **bad_kwargs)
    assert nlp_corpus.parse(BIO.encode(), filename="notes.txt", fmt="IOB2")["report"]["source_format"] == "bio"
    assert nlp_corpus.parse(b'{"text":"Rice","lang":"en"}')["report"]["source_format"] == "jsonl"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def upload(client, name, content, filename, **form):
    return client.post("/v1/nlp/corpora", files={"file": (filename, content, "application/octet-stream")},
                       data={"name": name, "approval_ref": REF, **form}, headers=ADMIN)


def test_api_accepts_bio_and_brat_and_stores_canonical_jsonl(client):
    fmt = client.get("/v1/nlp/corpora/format", headers=ADMIN).json()
    assert set(fmt["formats"]) == {"jsonl", "json", "csv", "bio", "conll", "docbin", "brat"}
    r = upload(client, "BIO set", BIO.encode(), "notes.bio")
    assert r.status_code == 201, r.text
    assert r.json()["version"]["report"]["source_format"] == "bio"
    cid = r.json()["id"]
    stored = [json.loads(ln) for ln in gzip.decompress(storage.load_bytes(f"local://nlp{cid}/v1.jsonl.gz")).decode().splitlines()]
    assert set(stored[0]) == {"text", "lang", "entities"} and stored[0]["lang"] == "ny"
    z = brat_zip({"a.txt": "The child ate rice.", "a.ann": "T1\tFOOD 14 18\trice\n"})
    assert upload(client, "BRAT set", z, "brat.zip", lang="en").status_code == 201
    assert upload(client, "BRAT nolang", z, "brat.zip").status_code == 422
    mapped = upload(client, "Mapped", json.dumps({"text": "The child ate rice.", "lang": "en", "entities": [[14, 18, "FOOD_ITEM"]]}).encode(),
                    "m.jsonl", label_map='{"FOOD_ITEM": "FOOD"}')
    assert mapped.status_code == 201
    assert upload(client, "Bad map", b"{}", "m.jsonl", label_map="[1]").status_code == 422
    assert upload(client, "Bad lang", b"{}", "m.jsonl", lang="klingon").status_code == 422
    assert upload(client, "Unknown ext", b"x", "notes.txt").status_code == 422
    inspect = client.post("/v1/nlp/corpora/inspect", files={"file": ("a.conll", b"Rice I-FOOD\n", "text/plain")}, data={"lang": "en"}, headers=ADMIN)
    assert inspect.status_code == 200 and inspect.json()["valid"] and inspect.json()["report"]["source_format"] == "conll"
