"""NLP laboratory: entity recognition, measurements, context, language, identifiers, classification, API rules."""
import ast
import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/nlp.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import nlp_lab  # noqa: E402
from app.main import app  # noqa: E402
from app.nlp_gold import run_benchmark  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
APP_DIR = Path(__file__).resolve().parent.parent / "app"


def ents(text, label=None):
    return [e for e in nlp_lab.analyze(text)["entities"] if label is None or e["label"] == label]


def meas(text, kind):
    return [e for e in ents(text, "MEASUREMENT") if e["mtype"] == kind]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


# ---------------------------------------------------------------- engine
def test_measurements_are_normalised():
    assert (meas("MUAC 11.5 cm", "muac")[0]["value"], meas("MUAC 11.5 cm", "muac")[0]["unit"]) == (115.0, "mm")
    assert meas("MUAC 115", "muac")[0]["unit_guessed"] is True
    assert meas("Hb 78 g/L", "haemoglobin")[0]["value"] == 7.8
    assert meas("height 0.85 m", "height")[0]["value"] == 85.0
    assert meas("weight 6200 g", "weight")[0]["value"] == 6.2
    assert meas("glucose 126 mg/dl", "glucose")[0]["value"] == 7.0
    assert meas("WHZ -3.2", "z_score")[0]["value"] == -3.2          # the sign is kept
    assert meas("WHZ −3,2", "z_score")[0]["value"] == -3.2          # unicode minus and decimal comma
    assert meas("a 14 month old", "age")[0]["unit"] == "months"
    assert meas("aged 5 years", "age")[0]["value"] == 5
    assert meas("diarrhoea for 3 days", "duration")
    assert not meas("diarrhoea for 3 days", "age")                   # a duration is not an age


def test_implausible_values_are_flagged_not_hidden():
    w = meas("weight 650 kg", "weight")[0]
    assert w["value"] == 650 and w["plausible"] is False
    assert nlp_lab.analyze("weight 650 kg")["warnings"][0].startswith("Some measurements")
    assert meas("MUAC 11.5 cm", "muac")[0]["plausible"] is True


def test_lexicon_english_and_chichewa():
    got = {(e["label"], e["norm"]) for e in ents("Mwana akutsegula m'mimba, amadya nsima ndi nyemba; mkaka wa m'mawere. Kutupa.")}
    assert {("POPULATION", "child"), ("SYMPTOM", "diarrhoea"), ("FOOD", "nsima"), ("FOOD", "beans"), ("FOOD", "breast milk"),
            ("SYMPTOM", "swelling")} <= got
    assert [e["norm"] for e in ents("kutsegula mmimba", "SYMPTOM")] == ["diarrhoea"]        # apostrophe is optional
    assert [e["norm"] for e in ents("two Eggs and TEA", "FOOD")] == ["eggs", "tea"]          # case-insensitive, plural handled
    assert [e["norm"] for e in ents("Started RUTF. SAM.")] == ["RUTF", "severe acute malnutrition"]
    assert ents("Sam went home and sam is here") == []                                      # acronyms are case-sensitive
    assert [e["text"] for e in ents("vitamin A deficiency", "CONDITION")] == ["vitamin A deficiency"]   # longest match wins
    assert ents("vitamin A deficiency", "NUTRIENT") == []


def test_context_flags():
    a = lambda t, term: next(e for e in ents(t) if e["text"].lower() == term)["attributes"]  # noqa: E731
    assert a("No fever and cough, but vomiting present.", "cough")["negated"] is True
    assert a("No fever and cough, but vomiting present.", "vomiting")["negated"] is False   # scope stops at 'but'
    assert a("HIV negative.", "hiv")["negated"] is True
    assert a("Suspected anaemia.", "anaemia")["uncertain"] is True
    assert a("Family history of diabetes.", "diabetes")["family"] is True
    assert a("History of malaria.", "malaria")["historical"] is True
    assert a("Mwana ali ndi kutupa ndipo palibe kusanza.", "kusanza")["negated"] is True
    assert a("Mwana ali ndi kutupa ndipo palibe kusanza.", "kutupa")["negated"] is False
    assert a("Severe acute malnutrition confirmed.", "severe acute malnutrition")["negated"] is False


def test_negation_does_not_cross_sentences():
    r = ents("No fever. Vomiting since morning.")
    assert [(e["norm"], e["attributes"]["negated"]) for e in r] == [("fever", True), ("vomiting", False)]


def test_portions_link_to_the_food():
    p = meas("Ate 2 cups of rice with beans", "portion")[0]
    assert p["value"] == 2 and p["food"] == "rice"


def test_language_identification():
    assert nlp_lab.analyze("The child has fever and was admitted to the ward.")["language"]["overall"] == "en"
    assert nlp_lab.analyze("Mwana ali ndi kutupa ndipo amadya phala.")["language"]["overall"] == "ny"
    mixed = nlp_lab.analyze("The child was admitted with fever. Mwana akudya nsima koma sakudya nyama.")["language"]
    assert mixed["code_switching"] is True and {s["language"] for s in mixed["sentences"]} == {"en", "ny"}
    assert nlp_lab.analyze("12345")["language"]["overall"] == "unknown"


def test_identifiers_are_masked_and_redacted():
    r = nlp_lab.analyze("Call Mr Phiri on 0999 123 456 or jane@example.org. MRN: AB12345. ID 123456789012. Child has fever.")
    assert r["identifiers"]["found"] >= 4
    assert {"NAME", "PHONE", "EMAIL"} <= set(r["identifiers"]["types"])
    assert all(e["text"] == "[redacted]" for e in r["entities"] if e["label"] == "PHI")
    masked = r["identifiers"]["masked_text"]
    for leak in ("Phiri", "0999", "jane@", "AB12345", "123456789012"):
        assert leak not in masked
    assert "fever" in masked
    assert "Personal names cannot be detected reliably" in " ".join(r["warnings"])
    assert nlp_lab.analyze("Child has fever.")["identifiers"] == {"found": 0, "types": {}, "masked_text": None}


def test_offsets_point_at_the_original_text():
    t = "Patient’s weight −3 kg lost; MUAC 11.5 cm — RUTF given."      # curly apostrophe, unicode minus and dash
    for e in nlp_lab.analyze(t)["entities"]:
        assert t[e["start"]:e["end"]] == e["text"]


def test_record_and_classification():
    t = "Patient admitted with severe acute malnutrition. MUAC 10.8 cm, bilateral pitting oedema. Plan: F-75 and RUTF. No diarrhoea."
    r = nlp_lab.analyze(t)
    assert r["classification"]["document_type"]["label"] == "clinical_note"
    top = r["classification"]["topics"][0]
    assert top["topic"] == "acute_malnutrition" and top["strength"] == "strong"
    assert {x["term"] for x in r["record"]["conditions"]["present"]} == {"severe acute malnutrition"}
    assert {x["term"] for x in r["record"]["symptoms"]["negated"]} == {"diarrhoea"}
    assert "infection_and_nutrition" not in {x["topic"] for x in r["classification"]["topics"]}   # a negated finding is not evidence
    d = nlp_lab.analyze("For breakfast she ate 2 cups of porridge and 1 egg; lunch was nsima with beans and fish.")
    assert d["classification"]["document_type"]["label"] == "dietary_record"
    res = nlp_lab.analyze("Methods: a cross-sectional study of 300 participants (n = 300). Prevalence was 12% (95% CI 9-15, p < 0.05).")
    assert res["classification"]["document_type"]["label"] == "research_text"
    assert nlp_lab.analyze("Hello there.")["classification"]["document_type"]["label"] == "unclear"


def test_benchmark_regression_guard():
    b = run_benchmark()
    assert b["overall"]["f1"] >= 0.9 and b["context_flag_accuracy"] >= 0.95
    assert "does not measure accuracy on real" in b["notice"]


def test_engine_is_isolated_from_models_storage_and_serving():
    banned = {"models", "db", "storage", "serving", "registry", "webhook", "training"}
    for name in ("nlp_lab.py", "nlp_lexicon.py", "nlp_gold.py"):
        for node in ast.walk(ast.parse((APP_DIR / name).read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                assert not ({node.module} | {a.name for a in node.names}) & banned, name


# ---------------------------------------------------------------- API
def test_roles_and_limits(client):
    ds, rev, viewer, service, owner = (key(client, "ds-nlp", "data_scientist"), key(client, "rev-nlp", "clinical_reviewer"),
                                       key(client, "viewer-nlp", "viewer"), key(client, "service-nlp", "service"),
                                       key(client, "owner-nlp", "registry_owner"))
    body = {"texts": ["Child has fever."]}
    for who in (ds, rev, ADMIN):
        assert client.post("/v1/nlp/analyze", json=body, headers=who).status_code == 200
    for who in (viewer, owner, service):
        assert client.post("/v1/nlp/analyze", json=body, headers=who).status_code == 403
    for who in (viewer, owner, ds):
        assert client.get("/v1/nlp/info", headers=who).status_code == 200
        assert client.get("/v1/nlp/benchmark", headers=who).status_code == 200
    assert client.get("/v1/nlp/info", headers=service).status_code == 403
    assert client.post("/v1/nlp/analyze", json={"texts": []}, headers=ADMIN).status_code == 422
    assert client.post("/v1/nlp/analyze", json={"texts": ["a"] * 21}, headers=ADMIN).status_code == 422
    assert client.post("/v1/nlp/analyze", json={"texts": ["  "]}, headers=ADMIN).status_code == 422
    assert client.post("/v1/nlp/analyze", json={"texts": ["x" * 8001]}, headers=ADMIN).status_code == 422
    assert client.post("/v1/nlp/analyze", json={"texts": ["ok"], "identifiers": "keep"}, headers=ADMIN).status_code == 422


def test_response_shape_and_masking(client):
    r = client.post("/v1/nlp/analyze", headers=ADMIN, json={"texts": ["Mwana ali ndi kutupa. Call 0999123456.", "Ate 2 cups of rice."]}).json()
    assert r["stored"] is False and r["engine"] == nlp_lab.ENGINE_VERSION and "no diagnosis" in r["notice"]
    assert len(r["results"]) == 2 and r["results"][0]["identifiers"]["found"] == 1
    assert "0999123456" not in str(r)                                                    # never echoed back
    assert r["results"][1]["record"]["measurements"]["portion"][0]["food"] == "rice"


def test_reject_mode_refuses_identifiers(client):
    r = client.post("/v1/nlp/analyze", headers=ADMIN, json={"texts": ["fine text", "Phone 0999123456"], "identifiers": "reject"})
    assert r.status_code == 422 and "text 2" in r.json()["detail"] and "0999123456" not in r.text
    assert client.post("/v1/nlp/analyze", headers=ADMIN, json={"texts": ["fine text"], "identifiers": "reject"}).status_code == 200


def test_audit_holds_counts_never_text(client):
    client.post("/v1/nlp/analyze", headers=ADMIN, json={"texts": ["Distinctive kwashiorkor sentence. Phone 0999123456."]})
    rows = [x for x in client.get("/v1/audit?limit=200", headers=ADMIN).json() if x["action"] == "nlp.analyze"]
    assert rows and rows[0]["detail"]["texts"] == 1 and rows[0]["detail"]["identifiers_found"] == 1
    assert "Distinctive" not in str(rows) and "0999123456" not in str(rows) and "kwashiorkor" not in str(rows)


def test_lexicon_listing(client):
    all_ = client.get("/v1/nlp/lexicon?limit=500", headers=ADMIN).json()
    assert all_["total"] > 200 and all(t["reviewed"] is False for t in all_["terms"])
    ny = client.get("/v1/nlp/lexicon?lang=ny&label=food", headers=ADMIN).json()["terms"]
    assert ny and {t["label"] for t in ny} == {"FOOD"} and "nsima" in {t["term"] for t in ny}
    assert any(t["norm"] == "diarrhoea" for t in client.get("/v1/nlp/lexicon?q=tsegula", headers=ADMIN).json()["terms"])
    info = client.get("/v1/nlp/info", headers=ADMIN).json()
    assert info["lexicon"]["clinically_reviewed"] == 0 and info["kind"] == "rule-based baseline"


# ---------------------------------------------------------------- Tumbuka and Yao (speaker-supplied CSV lists)
def test_tumbuka_seed_terms_and_language():
    r = nlp_lab.analyze("Ŵana wakulya somba na ufwa.")
    assert r["language"]["overall"] == "tum"
    assert {(e["norm"], e["lang"]) for e in r["entities"]} == {("child", "tum"), ("fish", "tum"), ("flour", "tum")}
    assert nlp_lab.analyze("Mwana akudya nsima koma sakudya nyama.")["language"]["overall"] == "ny"   # Chichewa is not mistaken for Tumbuka


def test_csv_lexicon_loader_accepts_good_rows_and_reports_bad_ones(tmp_path):
    from app import nlp_lexicon as lx
    (tmp_path / "yao.csv").write_text("# comment\nterm,norm,label,group,lang,case_sensitive\n"
                                      "ugali,porridge,FOOD,cereal,yao,\n"
                                      ",nothing,FOOD,,yao,\n"
                                      "x,y,MEASUREMENT,,yao,\n"
                                      "x,y,FOOD,,klingon,\n"
                                      "ABC,thing,CONDITION,,yao,yes\n", encoding="utf-8")
    (tmp_path / "markers.csv").write_text("lang,word\nyao,ni\nen,the\nyao,two words\n", encoding="utf-8")
    (tmp_path / "_template.csv").write_text("term,norm,label,lang\nignored,ignored,FOOD,yao\n", encoding="utf-8")
    entries, markers, errors = lx.load_csv_lexicons(tmp_path)
    assert [(e["term"], e["lang"], e["case_sensitive"]) for e in entries] == [("ugali", "yao", False), ("ABC", "yao", True)]
    assert markers == {"yao": {"ni"}}
    assert len(errors) == 5 and any("label must be" in e for e in errors) and any("lang must be" in e for e in errors)
    assert lx.load_csv_lexicons(tmp_path / "missing") == ([], {}, [])


def test_shipped_lexicon_files_have_no_errors_and_yao_is_empty(client):
    info = client.get("/v1/nlp/info", headers=ADMIN).json()
    assert info["lexicon_file_errors"] == []
    assert {x["code"] for x in info["languages"]} == {"en", "ny", "tum", "yao"}
    assert client.get("/v1/nlp/lexicon?lang=yao", headers=ADMIN).json()["total"] == 0
    assert client.get("/v1/nlp/lexicon?lang=tum", headers=ADMIN).json()["total"] == 3
    assert client.get("/v1/nlp/lexicon?lang=ny&limit=500", headers=ADMIN).json()["total"] > 100
