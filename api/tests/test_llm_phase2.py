"""LLM intelligence, phase 2: Ask CNR (RAG and natural-language interaction).

Covers: response shaping (citations, intent, answer screen), /v1/llm/ask and /v1/llm/retrieve against a local fake Chakudya server
(no real network), the phase 1 gate in front of both, failure handling, roles, per-key limits, counts-only audit, key secrecy and the
dashboard chat page."""
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/llm2.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app import llm_audit, llm_limits, llm_rag  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import AuditLog  # noqa: E402
from app.ratelimit import SlidingWindow  # noqa: E402
from app.routers import llm as llm_router  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
ROOT = Path(__file__).resolve().parent.parent.parent
FAKE_KEY = "ck_live_" + "Z9q8W7e6R5t4Y3u2" * 2

ASK_OK = {"status": "success", "query": "q", "context": "both", "cache": "MISS", "data": {
    "answer": "Severe acute malnutrition is defined by a MUAC below 115 mm [1]. WHO also uses weight-for-height [2].",
    "intent": "nutrition_question", "barcode_detected": None,
    "sources": [{"id": 1, "source": "MW Guidelines 2023", "title": "SAM criteria"}, {"id": 2, "source": "WHO 2013", "title": "Updates on SAM"}]}}
RETRIEVE_OK = {"status": "success", "count": 2, "cache": "HIT", "data": [
    {"content": "MUAC below 115 mm identifies SAM in children 6 to 59 months.", "source": "MW Guidelines 2023", "context": "clinical",
     "similarity": 0.8123456, "metadata": {"page": 14, "tags": ["a", "b"], "title": "SAM criteria"}},
    {"content": "x" * 3000, "source": "Long doc", "context": "general", "score": 0.5, "metadata": {}}]}


class _Fake(BaseHTTPRequestHandler):
    seen: list = []
    reply: tuple = (200, ASK_OK)

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Fake.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": json.loads(body or b"{}")})
        status, payload = _Fake.reply
        if self.path == "/rag/retrieve" and payload is ASK_OK:
            payload = RETRIEVE_OK
        data = json.dumps(payload).encode()
        self.send_response(status), self.send_header("Content-Type", "application/json"), self.send_header("Content-Length", str(len(data))), self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def chakudya(monkeypatch):
    _Fake.seen, _Fake.reply = [], (200, ASK_OK)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = Settings(llm_enabled=True, chakudya_api_url=f"http://127.0.0.1:{srv.server_address[1]}", chakudya_consumer_key=FAKE_KEY, environment="test")
    monkeypatch.setattr(llm_router, "get_settings", lambda: s)
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(1000, 60))
    monkeypatch.setattr(llm_limits, "per_hour", SlidingWindow(10000, 3600))
    monkeypatch.setattr(llm_router, "_audit_throttle", SlidingWindow(1, 60))
    yield _Fake
    srv.shutdown()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


def audit_rows(*actions):
    with SessionLocal() as db:
        return [{"action": r.action, "detail": r.detail} for r in db.scalars(select(AuditLog)) if not actions or r.action in actions]


def ask(client, question="What is the MUAC cut-off for severe acute malnutrition?", headers=ADMIN, **kw):
    return client.post("/v1/llm/ask", json={"question": question, **kw}, headers=headers)


# ================================================================= 1. shaping (no network)
def shaped(answer="Fine [1].", sources=None, intent="nutrition_question", **extra):
    raw = {"status": "success", "cache": "HIT", "data": {"answer": answer, "intent": intent, "sources": sources if sources is not None else [{"id": 1, "source": "S", "title": "T"}]}}
    raw.update(extra)
    return llm_rag.shape_ask(raw, context="both", top_k=6, history_turns=0, latency_ms=12)


def test_citations_are_matched_to_listed_sources():
    s = shaped("MUAC below 115 mm [1]. Weight-for-height also counts [2].", [{"id": 1, "source": "A", "title": "a"}, {"id": 2, "source": "B", "title": "b"}, {"id": 3, "source": "C", "title": "c"}])
    assert s.body["grounded"] is True and [c["cited"] for c in s.body["citations"]] == [True, True, False]
    assert s.audit["sources"] == 3 and s.audit["cited"] == 2 and s.audit["invalid_citations"] == 0 and s.audit["cache"] == "hit"


def test_a_citation_without_a_source_is_removed_and_counted():
    s = shaped("Claim [1]. Invented claim [7].")
    assert "[7]" not in s.body["answer"] and "[1]" in s.body["answer"]
    assert s.audit["invalid_citations"] == 1 and any("did not match" in w for w in s.body["warnings"])


def test_an_answer_that_cites_nothing_is_not_called_grounded():
    s = shaped("A confident answer with no citation.")
    assert s.body["grounded"] is False and any("does not cite" in w for w in s.body["warnings"])
    s = shaped("An answer with no sources at all.", sources=[])
    assert s.body["grounded"] is False and any("No source" in w for w in s.body["warnings"])


def test_an_answer_that_diagnoses_is_withheld_with_no_citations():
    s = shaped("The child has kwashiorkor [1].")
    assert s.body["answer_withheld"] is True and "kwashiorkor" not in s.body["answer"] and s.body["grounded"] is False
    assert s.body["withheld_categories"] == ["output_diagnosis"] and all(c["cited"] is False for c in s.body["citations"])
    assert s.audit["output_blocked"] is True and s.audit["cited"] == 0
    p = shaped("I would prescribe 500 mg of amoxicillin for the patient [1].")
    assert p.body["answer_withheld"] is True and "output_prescription" in p.body["withheld_categories"]


def test_guideline_text_about_criteria_is_not_withheld():
    s = shaped("A clinician diagnoses SAM when MUAC is below 115 mm [1]. Guideline text lists 100 mg doses for severe cases [1].")
    assert s.body["answer_withheld"] is False and s.body["grounded"] is True


def test_intent_is_limited_to_the_published_list_and_shape_is_checked():
    assert shaped(intent="<script>alert(1)</script>").body["intent"] == "unknown"
    assert shaped(intent="exchange_list").body["intent"] == "exchange_list"
    for raw in ({"status": "error", "data": {}}, {"status": "success", "data": []}, {"status": "success", "data": {"answer": 5}}, {"status": "success"}):
        with pytest.raises(llm_rag.ShapeError):
            llm_rag.shape_ask(raw, context="both", top_k=6, history_turns=0, latency_ms=1)


def test_control_characters_and_length_are_limited():
    s = shaped("Text\x00with\x07controls [1]." + "y" * 20000)
    assert "\x00" not in s.body["answer"] and "\x07" not in s.body["answer"] and len(s.body["answer"]) <= llm_rag.MAX_ANSWER_CHARS


def test_retrieved_chunks_are_shortened_and_metadata_is_scalar_only():
    s = llm_rag.shape_retrieve(RETRIEVE_OK, context="both", top_k=5, latency_ms=3)
    a, b = s.body["chunks"]
    assert a["score"] == 0.8123 and a["metadata"] == {"page": 14, "title": "SAM criteria"} and a["context"] == "clinical" and a["truncated"] is False
    assert len(b["snippet"]) == llm_rag.MAX_SNIPPET_CHARS and b["truncated"] is True and b["score"] == 0.5
    assert llm_rag.shape_retrieve(RETRIEVE_OK, context="both", top_k=1, latency_ms=3).body["count"] == 1
    with pytest.raises(llm_rag.ShapeError):
        llm_rag.shape_retrieve({"status": "success", "data": {"x": 1}}, context="both", top_k=5, latency_ms=1)


# ================================================================= 2. /v1/llm/ask
def test_ask_returns_a_cited_answer_and_sends_only_the_contract_fields(client, chakudya):
    r = ask(client, context="clinical", top_k=4)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["intent"] == "nutrition_question" and d["context"] == "clinical" and d["grounded"] is True and d["answer_withheld"] is False
    assert [c["cited"] for c in d["citations"]] == [True, True] and d["decision_support_only"] is True and d["stored"] is False
    assert d["contract"] == "cnr-llm-contract/1" and d["decision"] == "allowed" and "latency_ms" in d and "masked_question" not in d
    hit = chakudya.seen[0]
    assert hit["path"] == "/rag/ask" and hit["auth"] == "Bearer " + FAKE_KEY
    assert hit["body"] == {"query": "What is the MUAC cut-off for severe acute malnutrition?", "context": "clinical", "top_k": 4}   # no session_id, nothing else
    assert r.headers["X-RateLimit-Remaining-Minute"]


def test_ask_defaults_to_both_and_validates_input(client, chakudya):
    assert ask(client).json()["context"] == "both" and chakudya.seen[0]["body"]["context"] == "both" and chakudya.seen[0]["body"]["top_k"] == 6
    for bad in ({"context": "everything"}, {"top_k": 0}, {"top_k": 11}, {"question": ""}, {"history": [{"role": "system", "text": "x"}]}):
        assert ask(client, **bad).status_code == 422, bad
    n = len(chakudya.seen)
    assert client.post("/v1/llm/ask", json={"question": "x", "session_id": "abc"}, headers=ADMIN).status_code == 200
    assert "session_id" not in chakudya.seen[n]["body"]       # an unknown field is ignored, never forwarded


def test_ask_refuses_diagnosis_and_prescribing_before_anything_is_sent(client, chakudya):
    for q, cat in (("Does the child have kwashiorkor? SENTINEL-NOSEND", "diagnosis_request"), ("Please prescribe something for her SENTINEL-NOSEND", "prescribing_request")):
        r = ask(client, q)
        assert r.status_code == 422 and r.json()["category"] == cat and r.json()["decision"] == "refused" and "SENTINEL-NOSEND" not in r.text
    assert chakudya.seen == []


def test_ask_refuses_identifiers_by_default_and_masks_on_request(client, chakudya):
    q = "Mr Sentinelname phone 0999000111 mail banda.sentinel@example.com asks about the MUAC cut-off"
    r = ask(client, q)
    assert r.status_code == 422 and r.json()["category"] == "identifiers_found" and chakudya.seen == []
    assert "banda.sentinel" not in r.text and "0999000111" not in r.text and "Sentinelname" not in r.text
    r = ask(client, q, identifiers="mask")
    assert r.status_code == 200 and r.json()["decision"] == "masked" and r.json()["identifiers"]["found"] >= 3
    sent = json.dumps(chakudya.seen[0]["body"])
    assert "banda.sentinel" not in sent and "0999000111" not in sent and "Sentinelname" not in sent
    assert r.json()["masked_question"] == chakudya.seen[0]["body"]["query"]


def test_history_passes_the_same_gate_and_is_clipped(client, chakudya):
    hist = [{"role": "user", "text": f"turn {i}"} for i in range(8)] + [{"role": "assistant", "text": "a" * 900}]
    r = ask(client, "And for children under five?", history=hist)
    assert r.status_code == 200 and r.json()["decision"] == "allowed"
    sent = chakudya.seen[0]["body"]["history"]
    assert len(sent) == 6 and all(len(t["text"]) <= 400 for t in sent) and sent[-1]["role"] == "assistant"
    # identifiers inside an earlier turn
    chakudya.seen.clear()
    dirty = [{"role": "user", "text": "My mail is banda.sentinel@example.com"}]
    r = ask(client, "And for children?", history=dirty)
    assert r.status_code == 422 and r.json()["category"] == "identifiers_found" and chakudya.seen == []
    r = ask(client, "And for children?", history=dirty, identifiers="mask")
    assert r.status_code == 200 and "banda.sentinel" not in json.dumps(chakudya.seen[0]["body"])
    # a diagnosis request earlier in the chat cannot be smuggled in through history
    chakudya.seen.clear()
    r = ask(client, "Yes, go on", history=[{"role": "user", "text": "Does the child have kwashiorkor?"}])
    assert r.status_code == 422 and r.json()["category"] == "diagnosis_request" and chakudya.seen == []
    assert ask(client, "ok", history=[{"role": "user", "text": "   "}]).status_code == 200


def test_an_answer_that_diagnoses_is_withheld_over_http(client, chakudya):
    chakudya.reply = (200, {"status": "success", "data": {"answer": "The child has kwashiorkor [1].", "intent": "nutrition_question",
                                                          "sources": [{"id": 1, "source": "S", "title": "T"}]}})
    r = ask(client)
    d = r.json()
    assert r.status_code == 200 and d["answer_withheld"] is True and "kwashiorkor" not in r.text and "withheld" in d["answer"].lower()
    assert [x for x in audit_rows("llm.ask") if x["detail"].get("output_blocked") is True]


# ================================================================= 3. failures never leak
def test_upstream_failures_are_reported_without_the_key_or_response_text(client, chakudya):
    chakudya.reply = (401, {"error": "bad key " + FAKE_KEY})
    r = ask(client)
    assert r.status_code == 502 and r.json()["decision"] == "error" and FAKE_KEY not in r.text and "bad key" not in r.text
    chakudya.reply = (200, {"status": "success", "data": {"unexpected": "shape SENTINEL-BODY"}})
    r = ask(client)
    assert r.status_code == 502 and "SENTINEL-BODY" not in r.text
    reasons = [x["detail"].get("reason") for x in audit_rows("llm.ask") if x["detail"].get("decision") == "error"]
    assert "upstream_error" in reasons and "bad_shape" in reasons


def test_not_configured_is_a_503_and_calls_nothing(client, monkeypatch):
    s = Settings(llm_enabled=False, environment="test")
    monkeypatch.setattr(llm_router, "get_settings", lambda: s)
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(1000, 60))
    r = ask(client)
    assert r.status_code == 503 and "LLM_ENABLED" in r.json()["detail"]
    assert client.post("/v1/llm/retrieve", json={"query": "iron"}, headers=ADMIN).status_code == 503


# ================================================================= 4. /v1/llm/retrieve
def test_retrieve_returns_shortened_chunks(client, chakudya):
    r = client.post("/v1/llm/retrieve", json={"query": "MUAC cut-off", "context": "clinical", "top_k": 2}, headers=ADMIN)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["count"] == 2 and d["chunks"][0]["rank"] == 1 and len(d["chunks"][1]["snippet"]) <= llm_rag.MAX_SNIPPET_CHARS and d["stored"] is False
    hit = chakudya.seen[0]
    assert hit["path"] == "/rag/retrieve" and hit["body"] == {"query": "MUAC cut-off", "context": "clinical", "top_k": 2}


def test_retrieve_uses_the_same_gate(client, chakudya):
    assert client.post("/v1/llm/retrieve", json={"query": "Does the child have kwashiorkor?"}, headers=ADMIN).status_code == 422
    assert client.post("/v1/llm/retrieve", json={"query": "mail banda.sentinel@example.com iron"}, headers=ADMIN).status_code == 422
    assert chakudya.seen == []
    assert client.post("/v1/llm/retrieve", json={"query": "iron", "top_k": 21}, headers=ADMIN).status_code == 422


# ================================================================= 5. roles and limits
def test_roles(client, chakudya):
    assert client.post("/v1/llm/ask", json={"question": "iron"}).status_code == 401
    for role, expected in (("viewer", 403), ("service", 403), ("registry_owner", 403), ("data_scientist", 200), ("clinical_reviewer", 200)):
        h = key(client, "p2-" + role, role)
        assert client.post("/v1/llm/ask", json={"question": "What is iron?"}, headers=h).status_code == expected, role
        assert client.post("/v1/llm/retrieve", json={"query": "iron"}, headers=h).status_code == expected, role


def test_ask_and_retrieve_share_the_per_key_limit(client, chakudya, monkeypatch):
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(2, 60))
    h = key(client, "p2-limit", "data_scientist")
    assert client.post("/v1/llm/ask", json={"question": "iron"}, headers=h).status_code == 200
    assert client.post("/v1/llm/retrieve", json={"query": "iron"}, headers=h).status_code == 200
    r = client.post("/v1/llm/ask", json={"question": "iron"}, headers=h)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert len(chakudya.seen) == 2                      # the third request never reached Chakudya
    assert client.post("/v1/llm/ask", json={"question": "iron"}, headers=key(client, "p2-limit-2", "data_scientist")).status_code == 200


# ================================================================= 6. audit and secrecy
def test_audit_holds_counts_and_codes_only(client, chakudya):
    secrets = ["SENTINEL-QUESTION-ZEBRA", "SENTINEL-ANSWER-GIRAFFE", "SENTINEL-SOURCE-OTTER", "SENTINEL-HISTORY-HERON", "banda.audit@example.com"]
    chakudya.reply = (200, {"status": "success", "cache": "MISS", "data": {"answer": "SENTINEL-ANSWER-GIRAFFE [1]", "intent": "nutrition_question",
                                                                           "sources": [{"id": 1, "source": "SENTINEL-SOURCE-OTTER", "title": "SENTINEL-SOURCE-OTTER"}]}})
    n0 = len(audit_rows())
    ask(client, "SENTINEL-QUESTION-ZEBRA what is the MUAC cut-off?", history=[{"role": "user", "text": "SENTINEL-HISTORY-HERON hello"}])
    ask(client, "SENTINEL-QUESTION-ZEBRA mail banda.audit@example.com")                      # refused
    ask(client, "Does the child have kwashiorkor? SENTINEL-QUESTION-ZEBRA")                   # refused
    client.post("/v1/llm/retrieve", json={"query": "SENTINEL-QUESTION-ZEBRA iron"}, headers=ADMIN)
    rows = audit_rows("llm.ask", "llm.retrieve")
    assert len(audit_rows()) > n0 and rows
    blob = json.dumps(rows)
    for s in secrets:
        assert s not in blob, s
    ok = [r for r in rows if r["detail"].get("decision") == "allowed" and r["action"] == "llm.ask"][-1]["detail"]
    assert ok["sources"] == 1 and ok["cited"] == 1 and ok["intent"] == "nutrition_question" and ok["context"] == "both" and ok["history_turns"] == 1
    assert ok["cache"] == "miss" and isinstance(ok["latency_ms"], int) and ok["chars"] > 0
    allowed = set(llm_audit.ALLOWED)
    for r in rows:
        assert set(r["detail"]) <= allowed, set(r["detail"]) - allowed


def test_the_key_never_appears_in_any_response_or_audit_row(client, chakudya):
    texts = [ask(client).text, client.post("/v1/llm/retrieve", json={"query": "iron"}, headers=ADMIN).text,
             client.get("/v1/llm/contract", headers=ADMIN).text, client.get("/v1/llm/status", headers=ADMIN).text]
    chakudya.reply = (503, {"echo": FAKE_KEY})
    texts.append(ask(client).text)
    for t in texts:
        assert FAKE_KEY not in t
    assert FAKE_KEY not in json.dumps(audit_rows())


def test_the_contract_lists_the_new_endpoints(client):
    eps = client.get("/v1/llm/contract", headers=ADMIN).json()["endpoints"]
    assert "POST /v1/llm/ask" in eps and "POST /v1/llm/retrieve" in eps


def test_nothing_in_the_laboratory_stores_questions_or_answers():
    src = "\n".join(p.read_text() for p in (ROOT / "api" / "app").glob("llm_*.py")) + (ROOT / "api" / "app" / "routers" / "llm.py").read_text()
    assert "db.add(" not in src and "session_id" not in src      # no rows are written and no session is forwarded
    assert "from . import models" not in src


# ================================================================= 7. dashboard
def test_dashboard_has_the_ask_page_without_browser_storage_of_chats():
    js = (ROOT / "dashboard" / "app.js").read_text()
    assert '"/v1/llm/ask"' in js and '"/v1/llm/retrieve"' in js and 'id: "ask"' in js and '"use:llm"' in js
    assert "localStorage" not in js and "indexedDB" not in js.lower()
    for needle in ("Clinical", "General", "Both"):
        assert needle in js
    for f in (ROOT / "dashboard").glob("*"):
        if f.suffix in (".js", ".html"):
            low = f.read_text().lower()
            assert "chakudya_consumer_key" not in low and "ck_live_" not in low, f.name
    sw = (ROOT / "dashboard" / "sw.js").read_text()
    assert "shell-v" in sw


def test_documentation_describes_phase_2():
    doc = (ROOT / "docs" / "LLM_INTELLIGENCE.md").read_text()
    assert "## Phase 2" in doc and "/v1/llm/ask" in doc and "/v1/llm/retrieve" in doc
