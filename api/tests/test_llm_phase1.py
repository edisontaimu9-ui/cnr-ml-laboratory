"""LLM intelligence, phase 1: contract and safety rules.

Covers: decision-support boundary (requests and answers), identifier gate (mask / refuse), counts-only audit, per-key rate limits,
server-side-only Chakudya address and key, and the gateway transport (against a local HTTP server, no real network)."""
import json
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/llm1.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app import llm_audit, llm_gateway, llm_limits, llm_policy, nlp_lab  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import AuditLog  # noqa: E402
from app.ratelimit import SlidingWindow  # noqa: E402
from app.routers import llm as llm_router  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = ROOT / "api" / "app"
FAKE_KEY = "ck_live_" + "Z9q8W7e6R5t4Y3u2" * 2


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


@pytest.fixture()
def fresh_limits(monkeypatch):
    """Generous, empty limiters so tests do not influence each other."""
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(1000, 60))
    monkeypatch.setattr(llm_limits, "per_hour", SlidingWindow(10000, 3600))
    monkeypatch.setattr(llm_router, "_audit_throttle", SlidingWindow(1, 60))


def audit_rows(action=None):
    with SessionLocal() as db:
        q = select(AuditLog)
        if action:
            q = q.where(AuditLog.action == action)
        return [{"action": r.action, "detail": r.detail, "resource_id": r.resource_id, "actor": r.actor_name, "ip": r.ip} for r in db.scalars(q)]


# ================================================================= 1. decision-support boundary: requests
DIAGNOSIS_QUESTIONS = [
    "Diagnose this child", "Can you diagnose this patient?", "Does the child have kwashiorkor?", "Does my baby have severe acute malnutrition",
    "What is wrong with my child?", "What disease does she have", "Is he malnourished?", "Tell me if the patient has diabetes",
    "Is this child suffering from marasmus?", "What is the diagnosis for the patient with MUAC 110 mm?",
    "D\u200biagnose him", "Ｄｉａｇｎｏｓｅ ｔｈｉｓ ｃｈｉｌｄ",       # zero-width and full-width evasion
]
PRESCRIBING_QUESTIONS = [
    "What dose of amoxicillin should I give the child", "Please prescribe something for the patient", "What medicine should I give?",
    "What dosage for my patient with anaemia", "How many mg should I give to the baby",
]
GENERAL_QUESTIONS = [
    "What is the WHO MUAC cut-off for severe acute malnutrition in children 6-59 months?", "How is kwashiorkor different from marasmus?",
    "What does the guideline say about vitamin A supplementation dose for infants 6-11 months?", "Which foods in Malawi are rich in iron?",
    "What are the diagnostic criteria for gestational diabetes?", "How do I calculate energy requirements for a burn patient?",
    "Explain the ASPEN refeeding risk criteria.", "What is the Fenton chart used for?", "Summarise the documented findings in this note.",
]


@pytest.mark.parametrize("q", DIAGNOSIS_QUESTIONS)
def test_diagnosis_requests_are_refused(q):
    assert llm_policy.screen_request(q) == "diagnosis_request"


@pytest.mark.parametrize("q", PRESCRIBING_QUESTIONS)
def test_prescribing_requests_are_refused(q):
    assert llm_policy.screen_request(q) == "prescribing_request"


@pytest.mark.parametrize("q", GENERAL_QUESTIONS)
def test_general_and_criteria_questions_are_allowed(q):
    assert llm_policy.screen_request(q) is None


# ================================================================= 1b. decision-support boundary: answers
@pytest.mark.parametrize("answer, category", [
    ("The child has severe acute malnutrition.", "output_diagnosis"),
    ("This patient is suffering from kwashiorkor.", "output_diagnosis"),
    ("You have diabetes.", "output_diagnosis"),
    ("She is severely malnourished.", "output_diagnosis"),
    ("My diagnosis is anaemia.", "output_diagnosis"),
    ("I diagnose iron deficiency.", "output_diagnosis"),
    ("Give the child 100 mg of zinc.", "output_prescription"),
    ("You should take 500 mg of iron twice a day.", "output_prescription"),
    ("I prescribe 2 tablets daily.", "output_prescription"),
])
def test_answers_that_diagnose_or_prescribe_are_blocked(answer, category):
    v = llm_policy.screen_output(answer)
    assert not v.allowed and category in v.categories
    text, _ = llm_policy.enforce_output(answer)
    assert text == llm_policy.REFUSED_OUTPUT and answer not in text


@pytest.mark.parametrize("answer", [
    "A MUAC below 115 mm meets the WHO criterion for severe acute malnutrition; a clinician confirms the diagnosis.",
    "Kwashiorkor is characterised by bilateral pitting oedema.",
    "The WHO guideline dose is 20 mg of zinc per day for 10 to 14 days in children with diarrhoea (source: WHO 2013).",
    "These findings may be consistent with anaemia and need clinical assessment.",
    "The note states that the patient has a history of hypertension.",
])
def test_criteria_guideline_and_hedged_text_is_not_blocked(answer):
    # The last example quotes a document ("the note states"), which the subject patterns do not match.
    assert llm_policy.screen_output(answer).allowed


def test_the_refusal_text_itself_passes_the_output_screen():
    assert llm_policy.screen_output(llm_policy.REFUSED_OUTPUT).allowed


def test_system_rules_keep_the_boundary_lines():
    r = llm_policy.SYSTEM_RULES.lower()
    assert "never diagnose" in r and "never prescribe" in r and "only the sources" in r and "identifiers" in r and "urgent" in r
    assert any("never diagnoses" in x for x in llm_policy.RULES)


# ================================================================= 2. identifier gate
IDENT_SAMPLES = [
    "Mr Banda, phone +265 999 123 456, email banda@example.com, MRN: AB12345, id 1234567890123.",
    "Dr Mwale reviewed the child. Weight 6.2 kg, MUAC 11.5 cm.",
    "No identifiers here: Hb 7.8 g/dL.",
    "Mayi Phiri attended OPD no 7788/22.",
]


@pytest.mark.parametrize("t", IDENT_SAMPLES)
def test_screen_identifiers_matches_the_nlp_analysis(t):
    full = nlp_lab.analyze(t)
    phi = nlp_lab.screen_identifiers(t)
    assert len(phi) == full["identifiers"]["found"]
    assert {e["norm"] for e in phi} == set(full["identifiers"]["types"])
    assert nlp_lab.mask_identifiers(t, phi) == (full["identifiers"]["masked_text"] or t)


def test_reject_mode_refuses_identifiable_text():
    with pytest.raises(llm_policy.PolicyRefusal) as e:
        llm_policy.prepare_for_external("Contact Mr Banda on banda@example.com", "reject", max_chars=4000)
    assert e.value.category == "identifiers_found" and e.value.details["found"] == 2
    assert "banda" not in json.dumps(e.value.details).lower()


def test_mask_mode_replaces_identifiers_and_the_masked_text_is_clean():
    p = llm_policy.prepare_for_external("Contact Mr Banda on banda@example.com about MUAC 11.5 cm", "mask", max_chars=4000)
    assert p.masked and p.identifiers_found == 2
    assert "banda" not in p.text.lower() and "[EMAIL]" in p.text and "[NAME]" in p.text and "MUAC 11.5 cm" in p.text
    assert nlp_lab.screen_identifiers(p.text) == []


def test_clean_text_passes_unchanged_with_an_honest_warning():
    p = llm_policy.prepare_for_external("What is the MUAC cut-off for SAM?", "reject", max_chars=4000)
    assert not p.masked and p.text == "What is the MUAC cut-off for SAM?"
    assert any("not detected" in w for w in p.warnings)


def test_look_alike_and_invisible_characters_do_not_bypass_the_detector():
    fullwidth = "call ０９９９１２３４５６７８"                       # full-width digits
    zero_width = "id number 12345\u200b67890\u200c123"               # zero-width characters inside a long number
    for t in (fullwidth, zero_width):
        with pytest.raises(llm_policy.PolicyRefusal):
            llm_policy.prepare_for_external(t, "reject", max_chars=4000)


def test_empty_and_overlong_text_is_refused_and_mode_off_does_not_exist():
    for t, cat in (("   ", "empty"), ("a" * 50, "too_long")):
        with pytest.raises(llm_policy.PolicyRefusal) as e:
            llm_policy.prepare_for_external(t, "reject", max_chars=40)
        assert e.value.category == cat
    with pytest.raises(ValueError):
        llm_policy.prepare_for_external("hello", "off", max_chars=40)


# ================================================================= 3. audit allow-list
def test_audit_allow_list_keeps_counts_and_drops_text():
    kept = llm_audit.clean({"chars": 120, "identifiers_found": 2, "identifier_types": {"EMAIL": 1, "NAME": 1}, "decision": "refused",
                            "category": "identifiers_found", "masked": False})
    assert kept == {"chars": 120, "identifiers_found": 2, "identifier_types": {"EMAIL": 1, "NAME": 1}, "decision": "refused",
                    "category": "identifiers_found", "masked": False}
    dropped = llm_audit.clean({"text": "the patient Mr Banda", "decision": "The patient has kwashiorkor", "category": "x y",
                               "chars": "120", "identifier_types": {"banda@example.com": 1}, "question": "?", "kind": "a" * 80,
                               "identifiers_found": True, "reason": "Mr Banda"})
    assert dropped == {}


# ================================================================= 4. rate limits
def test_limiter_is_per_key_with_minute_and_hour_windows(monkeypatch):
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(3, 60))
    monkeypatch.setattr(llm_limits, "per_hour", SlidingWindow(5, 3600))
    for _ in range(3):
        llm_limits.acquire(1)
    with pytest.raises(llm_limits.RateLimited) as e:
        llm_limits.acquire(1)
    assert e.value.scope == "minute" and e.value.retry_after >= 1
    assert llm_limits.acquire(2)["minute"] == 2           # another key is unaffected
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(100, 60))
    llm_limits.acquire(1), llm_limits.acquire(1)          # the hour window already holds 3 of 5 from above
    with pytest.raises(llm_limits.RateLimited) as e:
        llm_limits.acquire(1)
    assert e.value.scope == "hour"


# ================================================================= 5. API
def test_roles_and_authentication(client, fresh_limits):
    body = {"text": "What is the MUAC cut-off for SAM?"}
    assert client.post("/v1/llm/precheck", json=body).status_code == 401
    assert client.post("/v1/llm/precheck", json=body, headers={"X-API-Key": "nope"}).status_code == 401
    for role in ("viewer", "service", "registry_owner"):
        assert client.post("/v1/llm/precheck", json=body, headers=key(client, f"p1-{role}", role)).status_code == 403
    for role in ("data_scientist", "clinical_reviewer", "admin"):
        h = key(client, f"p1-ok-{role}", role)
        assert client.post("/v1/llm/precheck", json=body, headers=h).status_code == 200
    # the contract is readable by viewers, never by the machine key
    assert client.get("/v1/llm/contract", headers=key(client, "p1-v", "viewer")).status_code == 200
    assert client.get("/v1/llm/contract", headers=key(client, "p1-svc", "service")).status_code == 403
    assert client.get("/v1/llm/contract").status_code == 401


def test_allowed_question(client, fresh_limits):
    h = key(client, "p1-allowed", "data_scientist")
    r = client.post("/v1/llm/precheck", json={"text": "What is the MUAC cut-off for SAM?"}, headers=h)
    d = r.json()
    assert r.status_code == 200 and d["decision"] == "allowed" and "masked_text" not in d
    assert d["decision_support_only"] is True and d["stored"] is False and d["contract"] == "cnr-llm-contract/1"
    assert d["identifiers"] == {"found": 0, "types": {}} and "X-RateLimit-Remaining-Minute" in r.headers


def test_boundary_refusals_over_http_do_not_echo_the_text(client, fresh_limits):
    h = key(client, "p1-refuse", "clinical_reviewer")
    for q, cat in (("Does the child have kwashiorkor? SENTINEL-ECHO-1", "diagnosis_request"), ("Please prescribe something SENTINEL-ECHO-2", "prescribing_request")):
        r = client.post("/v1/llm/precheck", json={"text": q}, headers=h)
        assert r.status_code == 422 and r.json()["decision"] == "refused" and r.json()["category"] == cat
        assert "SENTINEL-ECHO" not in r.text and r.json()["decision_support_only"] is True


def test_documents_skip_the_question_screen_but_not_the_identifier_gate(client, fresh_limits):
    h = key(client, "p1-doc", "data_scientist")
    note = "Child diagnosed with kwashiorkor last month. MUAC 11.2 cm."
    assert client.post("/v1/llm/precheck", json={"text": note, "kind": "document"}, headers=h).json()["decision"] == "allowed"
    r = client.post("/v1/llm/precheck", json={"text": note + " Mr Banda, 0999123456", "kind": "document"}, headers=h)
    assert r.status_code == 422 and r.json()["category"] == "identifiers_found" and r.json()["found"] == 2
    assert "Banda" not in r.text and "0999" not in r.text


def test_identifier_policy_default_reject_and_request_may_choose_mask(client, fresh_limits):
    h = key(client, "p1-ident", "data_scientist")
    text = "Ask Dr Mwale (mwale@example.com) about the MUAC 11.5 cm reading"
    r = client.post("/v1/llm/precheck", json={"text": text}, headers=h)
    assert r.status_code == 422 and r.json()["category"] == "identifiers_found"
    assert r.json()["types"] == {"EMAIL": 1, "NAME": 1}
    r = client.post("/v1/llm/precheck", json={"text": text, "identifiers": "mask"}, headers=h)
    d = r.json()
    assert r.status_code == 200 and d["decision"] == "masked" and "mwale" not in d["masked_text"].lower() and "MUAC 11.5 cm" in d["masked_text"]
    assert client.post("/v1/llm/precheck", json={"text": text, "identifiers": "off"}, headers=h).status_code == 422   # no way to switch it off


def test_text_longer_than_the_limit_is_refused(client, fresh_limits):
    h = key(client, "p1-long", "data_scientist")
    r = client.post("/v1/llm/precheck", json={"text": "a " * 3000}, headers=h)
    assert r.status_code == 422 and r.json()["category"] == "too_long"


def test_audit_log_never_holds_clinical_text_or_identifiers(client, fresh_limits):
    h = key(client, "p1-audit", "data_scientist")
    sentinels = ["SENTINEL-ZEBRA", "banda.sentinel@example.com", "0999000111", "Mr Sentinelname", "kwashiorkor-sentinel"]
    texts = [
        "SENTINEL-ZEBRA: what is the cut-off for kwashiorkor-sentinel?",
        "Does the child have kwashiorkor? SENTINEL-ZEBRA",
        "SENTINEL-ZEBRA Mr Sentinelname phone 0999000111 mail banda.sentinel@example.com",
    ]
    for t in texts:
        client.post("/v1/llm/precheck", json={"text": t}, headers=h)
        client.post("/v1/llm/precheck", json={"text": t, "identifiers": "mask"}, headers=h)
        client.post("/v1/llm/precheck", json={"text": t, "kind": "document"}, headers=h)
    rows = audit_rows("llm.precheck")
    assert len(rows) >= 9
    dump = json.dumps(rows).lower()
    for s in sentinels:
        assert s.lower() not in dump
    allowed_keys = set(llm_audit.ALLOWED)
    for r in rows:
        assert set(r["detail"]) <= allowed_keys
        assert isinstance(r["detail"]["chars"], int)
    assert any(r["detail"].get("identifiers_found") == 3 and r["detail"].get("identifier_types") == {"EMAIL": 1, "PHONE": 1, "NAME": 1}
               or r["detail"].get("identifiers_found", 0) >= 2 for r in rows)


def test_rate_limit_per_key_over_http(client, monkeypatch):
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(3, 60))
    monkeypatch.setattr(llm_limits, "per_hour", SlidingWindow(100, 3600))
    monkeypatch.setattr(llm_router, "_audit_throttle", SlidingWindow(1, 60))
    a, b = key(client, "p1-rl-a", "data_scientist"), key(client, "p1-rl-b", "data_scientist")
    ok = {"text": "What is MUAC?"}
    assert client.post("/v1/llm/precheck", json=ok, headers={"X-API-Key": "wrong"}).status_code == 401   # failed auth does not use anyone's allowance
    codes = [client.post("/v1/llm/precheck", json=body, headers=a).status_code for body in (ok, {"text": "Diagnose this child"}, {"text": "Mr Banda"}, ok, ok, ok)]
    assert codes == [200, 422, 422, 429, 429, 429]          # refused requests use the allowance too
    r = client.post("/v1/llm/precheck", json=ok, headers=a)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert client.post("/v1/llm/precheck", json=ok, headers=b).status_code == 200      # another key is unaffected
    limited = [x for x in audit_rows("llm.rate_limited") if x["actor"] == "p1-rl-a"]
    assert len(limited) == 1 and limited[0]["detail"]["scope"] == "minute"          # one audit row per key per minute, not one per rejection


def test_contract_and_status(client):
    c = client.get("/v1/llm/contract", headers=key(client, "p1-c", "viewer")).json()
    assert c["contract"] == "cnr-llm-contract/1" and c["stored"] is False and c["identifiers"]["default"] == "reject"
    assert any("never diagnoses" in x for x in c["rules"]) and any("server only" in x for x in c["rules"])
    assert c["limitations"] and "diagnosis_request" in c["refusal_categories"]
    assert client.get("/v1/llm/status", headers=key(client, "p1-s", "viewer")).status_code == 403
    assert client.get("/v1/llm/status", headers=key(client, "p1-s2", "clinical_reviewer")).status_code == 403
    st = client.get("/v1/llm/status", headers=ADMIN).json()
    assert st["enabled"] is False and st["ready"] is False and "consumer_key_configured" in st


# ================================================================= 6. Chakudya address and key stay on the server
def make_settings(**kw) -> Settings:
    base = dict(llm_enabled=True, chakudya_api_url="https://api.chakudya.example", chakudya_consumer_key=FAKE_KEY, environment="test")
    base.update(kw)
    return Settings(**base)


def test_key_is_never_visible_in_settings_status_or_responses(client, monkeypatch):
    s = make_settings()
    assert FAKE_KEY not in repr(s) and FAKE_KEY not in str(s) and FAKE_KEY not in str(s.model_dump())
    monkeypatch.setattr(llm_router, "get_settings", lambda: s)
    st = client.get("/v1/llm/status", headers=ADMIN)
    d = st.json()
    assert d["ready"] is True and d["chakudya_host"] == "api.chakudya.example" and d["consumer_key_configured"] is True and d["problem"] is None
    assert FAKE_KEY not in st.text and "api.chakudya.example/" not in st.text
    assert FAKE_KEY not in client.get("/v1/llm/contract", headers=ADMIN).text
    assert FAKE_KEY not in client.get("/v1/overview", headers=ADMIN).text and FAKE_KEY not in client.get("/v1/me", headers=ADMIN).text
    assert FAKE_KEY not in json.dumps(audit_rows())


def test_only_the_gateway_reads_the_consumer_key_and_the_dashboard_never_sees_it():
    readers = [p.name for p in APP_DIR.rglob("*.py") if "chakudya_consumer_key" in p.read_text() and p.name not in ("config.py", "llm_gateway.py")]
    assert readers == []
    # each secret has exactly one reader: the Chakudya key in llm_gateway.py, the Groq key (phase 4) in llm_groq.py
    assert "get_secret_value" not in "".join(p.read_text() for p in APP_DIR.rglob("*.py") if p.name not in ("llm_gateway.py", "llm_groq.py"))
    assert [p.name for p in APP_DIR.rglob("*.py") if "groq_api_key" in p.read_text() and p.name not in ("config.py", "llm_groq.py")] == []
    assert "chakudya_consumer_key" not in (APP_DIR / "llm_groq.py").read_text() and "groq_api_key" not in (APP_DIR / "llm_gateway.py").read_text()
    for f in (ROOT / "dashboard").glob("*"):
        if f.suffix in (".js", ".html", ".webmanifest", ".css"):
            low = f.read_text().lower()
            assert "consumer_key" not in low and "chakudya_api_url" not in low, f.name
    env_example = (ROOT / ".env.example").read_text()
    assert re.search(r"^CHAKUDYA_CONSUMER_KEY=\s*$", env_example, re.M) and "LLM_ENABLED=false" in env_example
    render = (ROOT / "render.yaml").read_text()
    assert re.search(r"- key: CHAKUDYA_CONSUMER_KEY\n\s+sync: false", render)


@pytest.mark.parametrize("kw, fragment", [
    ({"llm_enabled": False}, "off"),
    ({"chakudya_api_url": "", "cnr_api_base_url": ""}, "not set"),
    ({"chakudya_consumer_key": ""}, "CHAKUDYA_CONSUMER_KEY"),
    ({"chakudya_consumer_key": "short"}, "CHAKUDYA_CONSUMER_KEY"),
    ({"chakudya_api_url": "https://user:pw@api.example"}, "credentials"),
    ({"chakudya_api_url": "https://api.example/?k=1"}, "query"),
    ({"chakudya_api_url": "ftp://api.example"}, "http"),
    ({"chakudya_api_url": "http://api.example", "environment": "production"}, "https"),
])
def test_gateway_config_problems(kw, fragment):
    problem = llm_gateway.config_problem(make_settings(**kw))
    assert problem and fragment in problem and FAKE_KEY not in problem


def test_gateway_falls_back_to_the_cnr_address_and_is_ready():
    s = make_settings(chakudya_api_url="", cnr_api_base_url="https://api.chakudya.xyz/")
    assert llm_gateway.config_problem(s) is None and llm_gateway.base_url(s) == "https://api.chakudya.xyz"


# ---- transport, against a local server
class _Handler(BaseHTTPRequestHandler):
    seen: list = []
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        _Handler.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "rid": self.headers.get("X-Request-Id"), "body": body})
        m = _Handler.mode
        if m == "redirect":
            self.send_response(302), self.send_header("Location", "http://127.0.0.1:1/steal"), self.end_headers()
            return
        if m in ("401", "503"):
            self.send_response(int(m)), self.send_header("Content-Type", "application/json"), self.end_headers()
            self.wfile.write(b'{"error":"echo of the key: ' + FAKE_KEY.encode() + b'"}')
            return
        payload = {"ok": True} if m == "ok" else b"x" * 3000 if m == "big" else None
        ctype = "text/html" if m == "html" else "application/json"
        data = b"<html>hi</html>" if m == "html" else json.dumps(payload).encode() if m == "ok" else payload if m == "big" else b"[1,2]" if m == "list" else b"{not json"
        self.send_response(200), self.send_header("Content-Type", ctype), self.send_header("Content-Length", str(len(data))), self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    _Handler.seen, _Handler.mode = [], "ok"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_gateway_sends_the_key_only_as_a_bearer_header_to_the_configured_host(server):
    s = make_settings(chakudya_api_url=server)
    out = llm_gateway.post_json(s, "/rag/ask", {"question": "What is MUAC?"}, correlation_id="req-1")
    assert out == {"ok": True}
    hit = _Handler.seen[0]
    assert hit["path"] == "/rag/ask" and hit["auth"] == "Bearer " + FAKE_KEY and hit["rid"] == "req-1"
    assert FAKE_KEY.encode() not in hit["body"] and json.loads(hit["body"]) == {"question": "What is MUAC?"}


def test_gateway_allows_only_the_contract_routes(server):
    s = make_settings(chakudya_api_url=server)
    for path in ("/rag/ingest", "/admin/keys", "/rag/ask/../x", "//evil.example/rag/ask", "rag/ask", "/foods/lookup"):
        with pytest.raises(llm_gateway.GatewayError):
            llm_gateway.post_json(s, path, {})
    assert _Handler.seen == []
    llm_gateway.post_json(s, "/rag/retrieve", {"query": "iron"})
    assert [h["path"] for h in _Handler.seen] == ["/rag/retrieve"]


@pytest.mark.parametrize("mode, fragment", [("redirect", "redirect"), ("401", "refused"), ("503", "unavailable"), ("html", "JSON"),
                                           ("bad", "invalid JSON"), ("list", "shape"), ("big", "larger")])
def test_gateway_failures_are_short_and_never_leak_the_key(server, mode, fragment):
    _Handler.mode = mode
    s = make_settings(chakudya_api_url=server, llm_max_response_kb=1)
    with pytest.raises(llm_gateway.GatewayError) as e:
        llm_gateway.post_json(s, "/rag/ask", {"question": "q"})
    assert fragment.lower() in str(e.value).lower() and FAKE_KEY not in str(e.value)
    assert all(h["path"] == "/rag/ask" for h in _Handler.seen) and len(_Handler.seen) == 1     # the redirect target was never contacted


def test_gateway_refuses_when_not_configured_without_any_network_call(server):
    with pytest.raises(llm_gateway.GatewayError):
        llm_gateway.post_json(make_settings(chakudya_api_url=server, llm_enabled=False), "/rag/ask", {})
    assert _Handler.seen == []


def test_gateway_unreachable_host_gives_a_key_free_message():
    s = make_settings(chakudya_api_url="http://127.0.0.1:9", llm_timeout_s=2)
    with pytest.raises(llm_gateway.GatewayError) as e:
        llm_gateway.post_json(s, "/rag/ask", {})
    assert "No usable answer" in str(e.value) and FAKE_KEY not in str(e.value)


# ================================================================= 7. opt-in reuse of the CNR admin key (SURVEILLANCE_API_KEY)
ADMIN_CNR_KEY = "cnr_admin_" + "Q1w2E3r4T5y6U7i8" * 2


def test_admin_key_is_not_used_unless_opted_in():
    s = make_settings(chakudya_consumer_key="", surveillance_api_key=ADMIN_CNR_KEY, cnr_api_base_url="https://api.chakudya.example")
    assert "CHAKUDYA_CONSUMER_KEY" in llm_gateway.config_problem(s)
    assert llm_gateway.public_status(s)["key_source"] is None


def test_opt_in_uses_the_admin_key_for_the_cnr_host_only():
    ok = make_settings(chakudya_consumer_key="", surveillance_api_key=ADMIN_CNR_KEY, llm_use_surveillance_key=True,
                       chakudya_api_url="", cnr_api_base_url="https://api.chakudya.example")
    assert llm_gateway.config_problem(ok) is None
    st = llm_gateway.public_status(ok)
    assert st["key_source"] == "surveillance_admin" and st["uses_admin_key"] is True and st["note"] and ADMIN_CNR_KEY not in json.dumps(st)
    same_host = make_settings(chakudya_consumer_key="", surveillance_api_key=ADMIN_CNR_KEY, llm_use_surveillance_key=True,
                              chakudya_api_url="https://api.chakudya.example/v2", cnr_api_base_url="https://api.chakudya.example")
    assert llm_gateway.config_problem(same_host) is None
    other = make_settings(chakudya_consumer_key="", surveillance_api_key=ADMIN_CNR_KEY, llm_use_surveillance_key=True,
                          chakudya_api_url="https://elsewhere.example", cnr_api_base_url="https://api.chakudya.example")
    problem = llm_gateway.config_problem(other)
    assert problem and "CNR address" in problem and ADMIN_CNR_KEY not in problem
    with pytest.raises(llm_gateway.GatewayError):
        llm_gateway.post_json(other, "/rag/ask", {})


def test_opt_in_without_an_admin_key_names_the_missing_setting():
    s = make_settings(chakudya_consumer_key="", surveillance_api_key="", llm_use_surveillance_key=True)
    assert "SURVEILLANCE_API_KEY" in llm_gateway.config_problem(s)


def test_consumer_key_wins_when_both_are_set(server):
    s = make_settings(chakudya_api_url=server, cnr_api_base_url=server, surveillance_api_key=ADMIN_CNR_KEY, llm_use_surveillance_key=True)
    assert llm_gateway.public_status(s)["key_source"] == "consumer"
    llm_gateway.post_json(s, "/rag/retrieve", {"query": "iron"})
    assert _Handler.seen[0]["auth"] == "Bearer " + FAKE_KEY


def test_admin_key_goes_only_as_bearer_to_the_cnr_host_and_never_leaks(server):
    s = make_settings(chakudya_consumer_key="", chakudya_api_url="", cnr_api_base_url=server, surveillance_api_key=ADMIN_CNR_KEY,
                      llm_use_surveillance_key=True)
    assert llm_gateway.post_json(s, "/rag/ask", {"question": "q"}) == {"ok": True}
    hit = _Handler.seen[0]
    assert hit["auth"] == "Bearer " + ADMIN_CNR_KEY and ADMIN_CNR_KEY.encode() not in hit["body"]
    for mode in ("redirect", "401", "503"):
        _Handler.mode = mode
        with pytest.raises(llm_gateway.GatewayError) as e:
            llm_gateway.post_json(s, "/rag/ask", {"question": "q"})
        assert ADMIN_CNR_KEY not in str(e.value)


def test_admin_key_never_appears_in_responses_or_audit(client, monkeypatch):
    s = make_settings(chakudya_consumer_key="", chakudya_api_url="", cnr_api_base_url="https://api.chakudya.example",
                      surveillance_api_key=ADMIN_CNR_KEY, llm_use_surveillance_key=True)
    monkeypatch.setattr(llm_router, "get_settings", lambda: s)
    r = client.get("/v1/llm/status", headers=ADMIN)
    assert r.json()["key_source"] == "surveillance_admin" and ADMIN_CNR_KEY not in r.text
    assert ADMIN_CNR_KEY not in client.get("/v1/llm/contract", headers=ADMIN).text and ADMIN_CNR_KEY not in json.dumps(audit_rows())


def test_only_the_sync_modules_and_the_gateway_read_the_admin_key():
    readers = {p.name for p in APP_DIR.rglob("*.py") if "surveillance_api_key" in p.read_text()}
    assert readers <= {"config.py", "llm_gateway.py", "surveillance_sync.py", "training_sync.py"}
    # The dashboard's sync help text already names the variable; it must hold no key value and no LLM key setting.
    for f in (ROOT / "dashboard").glob("*"):
        if f.suffix in (".js", ".html"):
            low = f.read_text().lower()
            assert "llm_use_surveillance_key" not in low and "cnr_admin_" not in low and "bearer " + ADMIN_CNR_KEY.lower() not in low, f.name
