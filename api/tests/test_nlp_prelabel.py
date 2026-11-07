"""NLP pre-label export and annotation template: round trip into corpus upload, identifier refusal, overlaps, access."""
import csv
import io
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/prelabel.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATASET_STORAGE_DIR", f"{_tmp}/store")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import nlp_corpus, nlp_prelabel  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
SENTENCES = "The child ate rice and beans.\nMwana ali ndi kutupa.\nPatient has no oedema, MUAC 11.5 cm.\n\nUnknown words only here.\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_template_is_a_valid_corpus_file_with_four_languages():
    r = nlp_corpus.parse(nlp_prelabel.template_csv().encode(), filename="t.csv")
    assert r["valid"], r["errors"]
    assert {d["lang"] for d in r["docs"]} == {"en", "ny", "tum", "yao"}
    assert all(row["notes"].startswith("EXAMPLE") for row in rows(nlp_prelabel.template_csv()))


def test_prelabel_guesses_entities_and_skips_blank_lines():
    out = rows(nlp_prelabel.prelabel(SENTENCES.encode(), "s.txt")["csv"])
    assert len(out) == 4 and out[0]["doc_id"] == "line-1"
    assert "rice|FOOD" in out[0]["entities"] and "child|POPULATION" in out[0]["entities"]
    assert out[1]["lang"] == "ny" and "kutupa|SYMPTOM" in out[1]["entities"]
    assert "MUAC 11.5 cm|MEASUREMENT" in out[2]["entities"]


def test_unknown_language_is_left_blank_and_blocks_upload_until_filled():
    res = nlp_prelabel.prelabel(SENTENCES.encode(), "s.txt")
    last = rows(res["csv"])[3]
    assert last["lang"] == "" and "fill in lang" in last["notes"]
    assert not nlp_corpus.parse(res["csv"].encode(), filename="p.csv")["valid"]


def test_corrected_file_round_trips_into_corpus_upload():
    res = nlp_prelabel.prelabel(SENTENCES.encode(), "s.txt", lang="en")
    parsed = nlp_corpus.parse(res["csv"].encode(), filename="p.csv")
    assert parsed["valid"], parsed["errors"]
    assert len(parsed["docs"]) == 4 and parsed["report"]["by_language"] == {"en": 4}


def test_overlapping_terms_are_left_out_not_written_wrongly():
    res = nlp_prelabel.prelabel(b"She ate maize flour and maize.", "s.txt", lang="en")
    row = rows(res["csv"])[0]
    assert "maize flour|FOOD" in row["entities"] and "left out" in row["notes"]
    assert nlp_corpus.parse(res["csv"].encode(), filename="p.csv")["valid"]


def test_csv_input_keeps_doc_ids():
    out = rows(nlp_prelabel.prelabel(b"doc_id,text\nA1,The child ate rice.\n", "in.csv", lang="en")["csv"])
    assert out[0]["doc_id"] == "A1"


def test_identifiers_refuse_the_whole_file_without_echoing_text():
    with pytest.raises(nlp_prelabel.PrelabelError) as e:
        nlp_prelabel.prelabel(b"The child ate rice.\nSeen by Dr Banda today.\n", "s.txt")
    assert "line-2" in str(e.value) and "Banda" not in str(e.value)


def test_bad_inputs():
    with pytest.raises(nlp_prelabel.PrelabelError):
        nlp_prelabel.prelabel(b"\n\n", "s.txt")
    with pytest.raises(nlp_prelabel.PrelabelError):
        nlp_prelabel.prelabel(b"a,b\n1,2\n", "s.csv")


def test_endpoints(client):
    t = client.get("/v1/nlp/corpora/template", headers=ADMIN)
    assert t.status_code == 200 and t.headers["content-type"].startswith("text/csv") and "example-yao" in t.text
    r = client.post("/v1/nlp/corpora/prelabel", files={"file": ("s.txt", SENTENCES.encode(), "text/plain")}, data={"lang": "en"}, headers=ADMIN)
    assert r.status_code == 200 and r.headers["x-prelabel-documents"] == "4" and "rice|FOOD" in r.text
    assert client.post("/v1/nlp/corpora/prelabel", files={"file": ("s.txt", b"x", "text/plain")}, data={"lang": "xx"}, headers=ADMIN).status_code == 422
    bad = client.post("/v1/nlp/corpora/prelabel", files={"file": ("s.txt", b"Call 0999123456 now.\n", "text/plain")}, headers=ADMIN)
    assert bad.status_code == 422 and "0999123456" not in bad.text


def test_prelabel_needs_run_nlp(client):
    assert client.post("/v1/nlp/corpora/prelabel", files={"file": ("s.txt", b"x", "text/plain")}).status_code in (401, 403)


def test_prelabel_endpoint_accepts_a_word_document(client):
    import zipfile
    b = io.BytesIO()
    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("word/document.xml", f'<w:document {w}><w:body><w:p><w:r><w:t>The child ate rice and beans.</w:t></w:r></w:p></w:body></w:document>')
    r = client.post("/v1/nlp/corpora/prelabel", files={"file": ("notes.docx", b.getvalue(), "application/octet-stream")},
                    data={"lang": "en"}, headers=ADMIN)
    assert r.status_code == 200 and "rice|FOOD" in r.text and r.headers["X-Prelabel-Documents"] == "1"
    bad = client.post("/v1/nlp/corpora/prelabel", files={"file": ("scan.docx", b"not a zip", "application/octet-stream")}, headers=ADMIN)
    assert bad.status_code == 422 and "not a valid .docx" in bad.text
