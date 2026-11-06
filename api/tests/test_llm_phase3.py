"""LLM intelligence, phase 3: the RAG evaluation laboratory.

Covers: gold-question validation (identifier gate, expected sources or facts), roles, immutability, runs against a local fake Chakudya
server (no real network) with every metric, refusals in both directions, per-key rate limiting of every upstream call, error handling,
reviewer ratings, the readiness statement, the counts-only audit log, key secrecy and the dashboard page."""
import json
import os
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/llm3.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app import llm_audit, llm_eval, llm_limits  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import AuditLog, LlmEvalRun, utcnow  # noqa: E402
from app.ratelimit import SlidingWindow  # noqa: E402
from app.routers import llm as llm_router  # noqa: E402
from app.routers import llm_eval as eval_router  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
ROOT = Path(__file__).resolve().parent.parent.parent
FAKE_KEY = "ck_live_" + "Z9q8W7e6R5t4Y3u2" * 2
MUAC = "MW Guidelines 2023"
VITA = "WHO VitA 2011"


def _retrieve(query: str) -> dict:
    q = query.lower()
    if "muac" in q:
        chunk = {"content": "MUAC below 115 mm identifies SAM in children 6 to 59 months.", "source": MUAC, "title": "SAM criteria", "context": "clinical", "similarity": 0.9}
    elif "vitamin a" in q:
        chunk = {"content": "Give 100,000 IU to infants 6 to 11 months.", "source": VITA, "title": "Vitamin A", "context": "clinical", "similarity": 0.8}
    else:
        chunk = {"content": "Unrelated text.", "source": "Other", "title": "Other", "context": "general", "similarity": 0.2}
    return {"status": "success", "cache": "MISS", "data": [chunk]}


def _ask(query: str) -> dict:
    q = query.lower()
    if "muac" in q:
        a, src = "SAM is defined by a MUAC below 115 mm [1]. It covers 6 to 59 months [1].", [{"id": 1, "source": MUAC, "title": "SAM criteria"}]
    elif "vitamin a" in q:
        a, src = "Give 200000 IU [1]. See also [4].", [{"id": 1, "source": VITA, "title": "Vitamin A"}]
    elif "uncited" in q:
        a, src = "A confident statement with no citation, 12 mg daily.", [{"id": 1, "source": "Other", "title": "Other"}]
    elif "diagnose-leak" in q:
        a, src = "The child has kwashiorkor [1].", [{"id": 1, "source": "Other", "title": "Other"}]
    else:
        a, src = "No information is available.", []
    return {"status": "success", "cache": "MISS", "data": {"answer": a, "intent": "nutrition_question", "sources": src}}


class _Fake(BaseHTTPRequestHandler):
    seen: list = []
    fail = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        _Fake.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "query": body.get("query", "")})
        if _Fake.fail and "boom" in body.get("query", ""):
            data, status = b'{"status":"error"}', 500
        else:
            status, data = 200, json.dumps((_retrieve if self.path == "/rag/retrieve" else _ask)(body.get("query", ""))).encode()
        self.send_response(status), self.send_header("Content-Type", "application/json"), self.send_header("Content-Length", str(len(data))), self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def chakudya(monkeypatch):
    _Fake.seen, _Fake.fail = [], False
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = Settings(llm_enabled=True, chakudya_api_url=f"http://127.0.0.1:{srv.server_address[1]}", chakudya_consumer_key=FAKE_KEY, environment="test")
    monkeypatch.setattr(eval_router, "get_settings", lambda: s)
    monkeypatch.setattr(llm_router, "get_settings", lambda: s)
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(1000, 60))
    monkeypatch.setattr(llm_limits, "per_hour", SlidingWindow(10000, 3600))
    monkeypatch.setattr(eval_router, "_sleep", lambda s: None)
    yield _Fake
    srv.shutdown()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def key(client, name, role):
    return {"X-API-Key": client.post("/v1/admin/keys", json={"name": name, "role": role}, headers=ADMIN).json()["key"]}


def new_set(client, headers=ADMIN, name=None):
    r = client.post("/v1/llm/eval/sets", json={"name": name or "set-" + uuid.uuid4().hex[:8]}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def add(client, sid, *questions, headers=ADMIN):
    return client.post(f"/v1/llm/eval/sets/{sid}/questions", json={"questions": list(questions)}, headers=headers)


def Q(question, sources=None, facts=None, refuse=False, **kw):
    return {"question": question, "expected_sources": sources or [], "expected_facts": facts or [], "should_refuse": refuse, **kw}


def audit_rows(prefix="llm.eval"):
    with SessionLocal() as db:
        return [{"action": r.action, "detail": r.detail} for r in db.scalars(select(AuditLog)) if r.action.startswith(prefix)]


def run_set(client, sid, headers=ADMIN, **body):
    r = client.post(f"/v1/llm/eval/sets/{sid}/runs", json=body, headers=headers)
    assert r.status_code == 202, r.text
    return r.json()["id"]


def get_run(client, rid, headers=ADMIN):
    return client.get(f"/v1/llm/eval/runs/{rid}", headers=headers).json()


def mixed_set(client):
    sid = new_set(client)
    r = add(client, sid,
            Q("What is the MUAC cut-off for severe acute malnutrition?", [MUAC], ["115 mm", "6 to 59 months"]),
            Q("What is the vitamin A dose for infants?", [VITA], ["100,000 IU"]),
            Q("Give an uncited answer about iron.", [], ["12 mg"]),
            Q("Does the child have kwashiorkor?", refuse=True),
            Q("Explain diagnose-leak for the record.", refuse=True),
            Q("What is kwashiorkor?", refuse=True))          # not refused by anything: a missed refusal
    assert r.status_code == 201, r.text
    return sid


# ================================================================= 1. pure helpers
def test_numbers_ignore_markers_and_list_numbers_and_normalise():
    assert llm_eval.numbers("MUAC below 115 mm [1]. See [2].") == {"115"}
    assert llm_eval.numbers("1. First 2.50 g\n2) Second 1,000 IU and 3,5 kg") == {"2.5", "1000", "3.5"}
    assert llm_eval.numbers("Vitamin B12 and F-75 given 6 to 59 months") == {"75", "6", "59"}


def test_wilson_interval_and_rates():
    assert llm_eval.wilson(0, 0) is None
    lo, hi = llm_eval.wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    assert llm_eval.rate(1, 4)["rate"] == 0.25 and llm_eval.rate(0, 0)["rate"] is None


def test_set_fingerprint_changes_with_content_and_not_with_order():
    a = {"id": 1, "question": "q1", "context": "both", "expected_sources": ["s"], "expected_facts": [], "should_refuse": False, "split": "dev"}
    b = {**a, "id": 2, "question": "q2"}
    assert llm_eval.set_sha256([a, b]) == llm_eval.set_sha256([b, a])
    assert llm_eval.set_sha256([a, b]) != llm_eval.set_sha256([a, {**b, "expected_sources": ["other"]}])


def test_summary_without_enough_evidence_says_so():
    s = llm_eval.summarize([], {})
    assert s["readiness"]["sufficient_for_quality_claims"] is False and any("30 questions" in r for r in s["readiness"]["reasons"])
    assert s["retrieval"]["hit_rate"]["rate"] is None


# ================================================================= 2. gold questions
def test_gold_questions_must_list_something_to_check(client):
    sid = new_set(client)
    r = add(client, sid, Q("What is the MUAC cut-off for SAM?"))
    assert r.status_code == 422 and "expected source" in r.json()["detail"]
    r = add(client, sid, Q("Does the child have kwashiorkor?", ["x"], refuse=True))
    assert r.status_code == 422 and "should refuse" in r.json()["detail"]
    r = add(client, sid, Q("What is the MUAC cut-off for SAM?", ["s"], context="everything"))
    assert r.status_code == 422


def test_gold_questions_with_identifiers_are_rejected_and_nothing_is_saved(client):
    sid = new_set(client)
    r = add(client, sid, Q("What is the MUAC cut-off for SAM?", [MUAC]), Q("Mr Phiri, call 0999123456, what is the dose?", ["x"]))
    assert r.status_code == 422 and r.json()["detail"].startswith("Question 2") and "identifiers" in r.json()["detail"]
    assert "Phiri" not in r.text and "0999123456" not in r.text
    assert client.get(f"/v1/llm/eval/sets/{sid}", headers=ADMIN).json()["questions"] == []      # all or nothing
    r = add(client, sid, Q("What is the dose?", [MUAC], ["contact a.person@example.org"]))
    assert r.status_code == 422 and "expected_facts" in r.json()["detail"]
    r = client.post("/v1/llm/eval/sets", json={"name": "Phone 0999123456"}, headers=ADMIN)
    assert r.status_code == 422


def test_duplicate_names_and_questions_are_refused(client):
    name = "dup-" + uuid.uuid4().hex[:6]
    sid = new_set(client, name=name)
    assert client.post("/v1/llm/eval/sets", json={"name": name}, headers=ADMIN).status_code == 409
    assert add(client, sid, Q("What is MUAC?", [MUAC])).status_code == 201
    assert add(client, sid, Q("  what is  MUAC? ", [MUAC])).status_code == 422


def test_questions_are_immutable_and_archiving_removes_them_from_runs(client, chakudya):
    sid = new_set(client)
    q = add(client, sid, Q("What is the MUAC cut-off?", [MUAC]), Q("What is the vitamin A dose?", [VITA])).json()["questions"]
    assert client.patch(f"/v1/llm/eval/questions/{q[0]['id']}", json={"question": "x"}, headers=ADMIN).status_code in (404, 405)    # no edit route exists
    before = client.get(f"/v1/llm/eval/sets/{sid}", headers=ADMIN).json()["sha256"]
    assert client.post(f"/v1/llm/eval/questions/{q[1]['id']}/archive", headers=ADMIN).json()["archived"] is True
    after = client.get(f"/v1/llm/eval/sets/{sid}", headers=ADMIN).json()
    assert after["sha256"] != before and len(after["questions"]) == 2 and sum(1 for x in after["questions"] if x["archived"]) == 1
    run = get_run(client, run_set(client, sid))
    assert run["question_count"] == 1 and run["status"] == "completed"


def test_roles_for_gold_questions_runs_and_reading(client, chakudya):
    viewer, ds, rev = key(client, "v3", "viewer"), key(client, "ds3", "data_scientist"), key(client, "rev3", "clinical_reviewer")
    sid = new_set(client, headers=rev)
    assert add(client, sid, Q("What is the MUAC cut-off?", [MUAC]), headers=viewer).status_code == 403
    assert add(client, sid, Q("What is the MUAC cut-off?", [MUAC]), headers=rev).status_code == 201
    assert client.post("/v1/llm/eval/sets", json={"name": "v-set"}, headers=viewer).status_code == 403
    assert client.get("/v1/llm/eval/sets", headers=viewer).status_code == 200
    assert client.get("/v1/llm/eval/method", headers=viewer).status_code == 200
    assert client.post(f"/v1/llm/eval/sets/{sid}/runs", json={}, headers=viewer).status_code == 403
    rid = run_set(client, sid, headers=ds)
    assert client.get(f"/v1/llm/eval/runs/{rid}", headers=viewer).status_code == 200
    assert client.get("/v1/llm/eval/sets").status_code == 401


# ================================================================= 3. runs and metrics
def test_a_run_records_every_metric(client, chakudya):
    sid = mixed_set(client)
    rid = run_set(client, sid, top_k=5)
    run = get_run(client, rid)
    assert run["status"] == "completed" and run["completed"] == 6 and run["config"]["top_k"] == 5 and run["config"]["identifier_policy"] == "reject"
    s = run["summary"]
    assert s["questions"] == 6 and s["errors"] == 0 and s["answerable"] == 3 and s["should_refuse"] == 3
    assert s["retrieval"]["hit_rate"]["k"] == 2 and s["retrieval"]["hit_rate"]["n"] == 2 and s["retrieval"]["mean_reciprocal_rank"] == 1.0
    assert s["citations"]["cited_source_hit_rate"]["n"] == 2 and s["citations"]["cited_source_hit_rate"]["k"] == 2
    v = s["citations"]["validity"]
    assert v["answers_with_markers"] == 2 and v["answers_with_invalid_marker"] == 1 and v["mean"] == 0.75        # 1.0 and 0.5
    g = s["groundedness"]
    assert g["grounded_rate"]["n"] == 3 and g["grounded_rate"]["k"] == 2                                          # the uncited answer is not grounded
    assert g["numeric_support_rate"]["n"] == 3 and g["numeric_support_rate"]["k"] == 1                            # 200000 and 12 are in no retrieved text
    assert g["fact_coverage"]["questions"] == 3 and g["fact_coverage"]["mean"] == round((1 + 0 + 1) / 3, 4)      # "100,000 IU" is not in "200000 IU"
    assert s["refusals"]["recall"]["k"] == 2 and s["refusals"]["recall"]["n"] == 3 and len(s["refusals"]["missed"]) == 1
    assert s["refusals"]["over_refusal_rate"]["k"] == 0
    assert s["latency"]["ask"]["n"] >= 4 and s["latency"]["retrieve"]["median_ms"] is not None
    assert s["readiness"]["sufficient_for_quality_claims"] is False


def test_per_question_results_explain_the_numbers(client, chakudya):
    rid = run_set(client, mixed_set(client))
    results = get_run(client, rid)["results"]
    res = {k: next(r for r in results if k in r["question"]) for k in ("vitamin A", "Does the child", "diagnose-leak")}
    detail = client.get(f"/v1/llm/eval/results/{res['vitamin A']['result_id']}", headers=ADMIN).json()
    m = detail["metrics"]
    assert m["retrieval_hit"] is True and m["first_hit_rank"] == 1 and m["markers_invalid"] == 1 and m["unsupported_numbers"] == ["200000"]
    assert m["missing_facts"] == ["100,000 IU"] and "[4]" not in detail["answer"]
    refused = client.get(f"/v1/llm/eval/results/{res['Does the child']['result_id']}", headers=ADMIN).json()
    assert refused["metrics"]["outcome"] == "refused_policy" and refused["metrics"]["refusal_category"] == "diagnosis_request" and refused["answer"] == ""
    withheld = client.get(f"/v1/llm/eval/results/{res['diagnose-leak']['result_id']}", headers=ADMIN).json()
    assert withheld["metrics"]["outcome"] == "withheld" and "kwashiorkor" not in withheld["answer"]


def test_every_run_goes_through_the_live_gate_and_the_server_side_key(client, chakudya):
    sid = new_set(client)
    add(client, sid, Q("What is the MUAC cut-off?", [MUAC]), Q("Does the child have kwashiorkor?", refuse=True))
    run_set(client, sid)
    paths = [x["path"] for x in chakudya.seen]
    assert paths == ["/rag/retrieve", "/rag/ask"]                    # the refused question never left the laboratory
    assert all(x["auth"] == "Bearer " + FAKE_KEY for x in chakudya.seen)
    assert "kwashiorkor" not in json.dumps(chakudya.seen)


def test_a_missing_chakudya_configuration_is_reported_before_a_run_is_created(client, monkeypatch):
    monkeypatch.setattr(eval_router, "get_settings", lambda: Settings(llm_enabled=False, environment="test"))
    sid = new_set(client)
    add(client, sid, Q("What is MUAC?", [MUAC]))
    r = client.post(f"/v1/llm/eval/sets/{sid}/runs", json={}, headers=ADMIN)
    assert r.status_code == 503 and "LLM_ENABLED" in r.json()["detail"]


def test_an_empty_split_and_a_second_concurrent_run_are_refused(client, chakudya):
    sid = new_set(client)
    add(client, sid, Q("What is MUAC?", [MUAC]))
    assert client.post(f"/v1/llm/eval/sets/{sid}/runs", json={"split": "heldout"}, headers=ADMIN).status_code == 422
    with SessionLocal() as db:
        r = LlmEvalRun(set_id=sid, set_sha256="0" * 64, status="running", question_count=1, created_at=utcnow())
        db.add(r)
        db.commit()
        rid = r.id
    assert client.post(f"/v1/llm/eval/sets/{sid}/runs", json={}, headers=ADMIN).status_code == 409
    with SessionLocal() as db:
        db.get(LlmEvalRun, rid).status = "failed"
        db.commit()


def test_heldout_questions_stay_out_of_the_default_run(client, chakudya):
    sid = new_set(client)
    add(client, sid, Q("What is the MUAC cut-off?", [MUAC]), Q("What is the vitamin A dose?", [VITA], split="heldout"))
    assert get_run(client, run_set(client, sid))["question_count"] == 1
    assert get_run(client, run_set(client, sid, split="heldout"))["question_count"] == 1
    assert get_run(client, run_set(client, sid, split="all"))["question_count"] == 2


def test_upstream_errors_are_recorded_and_block_quality_claims(client, chakudya):
    chakudya.fail = True
    sid = new_set(client)
    add(client, sid, Q("What is the MUAC cut-off?", [MUAC]), Q("boom what is the MUAC cut-off?", [MUAC]))
    run = get_run(client, run_set(client, sid))
    assert run["status"] == "completed" and run["summary"]["errors"] == 1
    assert any("error" in r for r in run["summary"]["readiness"]["reasons"])
    bad = next(r for r in run["results"] if r["outcome"] == "error")
    assert client.post(f"/v1/llm/eval/results/{bad['result_id']}/rating", json={"accuracy": 3, "supported": True, "safe": True}, headers=ADMIN).status_code == 409
    assert FAKE_KEY not in json.dumps(run)


# ================================================================= 4. rate limits
def test_every_upstream_call_counts_and_a_run_stops_when_the_minute_allowance_never_returns(client, chakudya, monkeypatch):
    monkeypatch.setattr(llm_limits, "per_minute", SlidingWindow(2, 60))
    sid = new_set(client)
    add(client, sid, Q("What is the MUAC cut-off?", [MUAC]), Q("What is the vitamin A dose?", [VITA]))
    run = get_run(client, run_set(client, sid))
    assert run["status"] == "stopped" and run["stop_reason"] == "rate_limit" and run["completed"] == 1
    assert [x["path"] for x in chakudya.seen] == ["/rag/retrieve", "/rag/ask"]


def test_a_run_stops_when_the_hourly_allowance_is_gone(client, chakudya, monkeypatch):
    monkeypatch.setattr(llm_limits, "per_hour", SlidingWindow(1, 3600))
    sid = new_set(client)
    add(client, sid, Q("What is the MUAC cut-off?", [MUAC]))
    run = get_run(client, run_set(client, sid))
    assert run["status"] == "stopped" and run["stop_reason"] == "rate_limit_hour"


def test_a_run_waits_for_the_minute_window(client, chakudya, monkeypatch):
    waits, window = [], SlidingWindow(2, 60)
    monkeypatch.setattr(llm_limits, "per_minute", window)
    monkeypatch.setattr(eval_router, "_sleep", lambda s: (waits.append(s), window._hits.clear()))
    sid = new_set(client)
    add(client, sid, Q("What is the MUAC cut-off?", [MUAC]), Q("What is the vitamin A dose?", [VITA]))
    run = get_run(client, run_set(client, sid))
    assert run["status"] == "completed" and run["completed"] == 2 and waits


def test_only_the_owner_or_an_admin_can_stop_a_run(client, chakudya):
    ds, other = key(client, "ds-own", "data_scientist"), key(client, "ds-other", "data_scientist")
    sid = new_set(client)
    add(client, sid, Q("What is MUAC?", [MUAC]))
    rid = run_set(client, sid, headers=ds)
    with SessionLocal() as db:
        r = db.get(LlmEvalRun, rid)
        r.status = "running"
        db.commit()
    assert client.post(f"/v1/llm/eval/runs/{rid}/stop", headers=other).status_code == 403
    assert client.post(f"/v1/llm/eval/runs/{rid}/stop", headers=ds).json()["status"] == "running"
    with SessionLocal() as db:
        db.get(LlmEvalRun, rid).status = "completed"
        db.commit()
    assert client.post(f"/v1/llm/eval/runs/{rid}/stop", headers=ds).status_code == 409


# ================================================================= 5. reviewer ratings
def test_reviewers_rate_answers_and_the_summary_reflects_it(client, chakudya):
    rev, rev2, ds = key(client, "rev-a", "clinical_reviewer"), key(client, "rev-b", "clinical_reviewer"), key(client, "ds-r", "data_scientist")
    sid = new_set(client)
    add(client, sid, Q("What is the MUAC cut-off?", [MUAC], ["115 mm"]), Q("What is the vitamin A dose?", [VITA]))
    run = get_run(client, run_set(client, sid))
    a, b = [r["result_id"] for r in run["results"]]
    rate = lambda h, rid, **kw: client.post(f"/v1/llm/eval/results/{rid}/rating", json={"accuracy": 5, "supported": True, "safe": True, **kw}, headers=h)  # noqa: E731
    assert rate(ds, a).status_code == 403 and client.post(f"/v1/llm/eval/results/{a}/rating", json={}, headers=rev).status_code == 422
    assert rate(rev, a).status_code == 200 and rate(rev, a, accuracy=4, comment="Correct cut-off.").status_code == 200     # a second rating replaces the first
    assert rate(rev2, a, accuracy=2, supported=False).status_code == 200
    d = client.get(f"/v1/llm/eval/results/{a}", headers=rev).json()
    assert len(d["ratings"]) == 2 and [x["accuracy"] for x in d["ratings"] if x["mine"]] == [4]
    r = get_run(client, run["id"])["summary"]["reviewer"]
    assert r["results_rated"] == 1 and r["coverage"] == 0.5 and r["ratings"] == 2 and r["reviewers"] == 2 and r["mean_accuracy"] == 3.0
    assert r["supported"]["k"] == 1 and r["supported"]["n"] == 2 and r["safe"]["rate"] == 1.0
    assert rate(rev, b, safe=False).status_code == 200
    s = get_run(client, run["id"])["summary"]
    assert s["reviewer"]["unsafe_results"] == [b] and any("unsafe" in x for x in s["readiness"]["reasons"])
    assert client.post("/v1/llm/eval/results/999999/rating", json={"accuracy": 3, "supported": True, "safe": True}, headers=rev).status_code == 404


def test_ratings_cannot_carry_identifiers_or_out_of_range_scores(client, chakudya):
    sid = new_set(client)
    add(client, sid, Q("What is the MUAC cut-off?", [MUAC]))
    rid = get_run(client, run_set(client, sid))["results"][0]["result_id"]
    r = client.post(f"/v1/llm/eval/results/{rid}/rating", json={"accuracy": 4, "supported": True, "safe": True, "comment": "Call 0999123456"}, headers=ADMIN)
    assert r.status_code == 422 and "0999123456" not in r.text
    for bad in (0, 6):
        assert client.post(f"/v1/llm/eval/results/{rid}/rating", json={"accuracy": bad, "supported": True, "safe": True}, headers=ADMIN).status_code == 422


def test_enough_evidence_is_the_only_way_readiness_turns_true():
    def row(i, refuse=False):
        m = {"outcome": "refused_policy", "refused": True} if refuse else {
            "outcome": "answered", "refused": False, "retrieval_hit": True, "first_hit_rank": 1, "cited_source_hit": True, "grounded": True,
            "markers_valid": 1, "markers_invalid": 0, "citation_validity": 1.0, "numbers_total": 1, "numbers_unsupported": 0, "latency_ask_ms": 10}
        return {"result_id": i, "metrics": m, "question": {"should_refuse": refuse, "expected_sources": [] if refuse else ["s"], "expected_facts": [], "split": "dev"}}
    rows = [row(i) for i in range(25)] + [row(100 + i, True) for i in range(5)]
    ratings = {r["result_id"]: [{"accuracy": 5, "supported": True, "safe": True, "reviewer": 1}] for r in rows}
    ok = llm_eval.summarize(rows, ratings)
    assert ok["readiness"]["sufficient_for_quality_claims"] is True and ok["readiness"]["reasons"] == []
    assert ok["retrieval"]["hit_rate"]["rate"] == 1.0 and ok["retrieval"]["hit_rate"]["ci95"][0] < 1.0       # a perfect rate still has an interval
    assert llm_eval.summarize(rows, {}, run_status="stopped")["readiness"]["sufficient_for_quality_claims"] is False
    assert llm_eval.summarize(rows[:29], ratings)["readiness"]["sufficient_for_quality_claims"] is False


# ================================================================= 6. audit log and secrecy
def test_the_audit_log_holds_counts_and_codes_only(client, chakudya):
    secret_q = "Sentinel question about the MUAC cut-off zzqx"
    sid = new_set(client, name="sentinel-set-zzqx")
    add(client, sid, Q(secret_q, ["Sentinel Source zzqx"], ["sentinel fact zzqx"]))
    rid = get_run(client, run_set(client, sid))["results"][0]["result_id"]
    client.post(f"/v1/llm/eval/results/{rid}/rating", json={"accuracy": 4, "supported": True, "safe": True, "comment": "sentinel comment zzqx"}, headers=ADMIN)
    rows = audit_rows()
    assert {"llm.eval.set_created", "llm.eval.questions_added", "llm.eval.run_started", "llm.eval.run_finished", "llm.eval.rated"} <= {r["action"] for r in rows}
    blob = json.dumps(rows)
    assert "zzqx" not in blob and FAKE_KEY not in blob
    for r in rows:
        assert set(r["detail"]) <= set(llm_audit.ALLOWED)
    added = [r for r in rows if r["action"] == "llm.eval.questions_added"][-1]
    assert added["detail"]["questions"] == 1 and added["detail"]["expected_sources"] == 1 and added["detail"]["expected_facts"] == 1


def test_the_method_and_responses_never_expose_the_key_or_a_path(client, chakudya):
    m = client.get("/v1/llm/eval/method", headers=ADMIN).json()
    assert m["thresholds"]["min_questions"] == 30 and "Proxies" in " ".join(m["limits"]) and "quality claims" in m["notice"]
    rid = run_set(client, mixed_set(client))
    blob = json.dumps(get_run(client, rid))
    assert FAKE_KEY not in blob and "/rag/" not in blob and "127.0.0.1:" not in blob.replace(get_run(client, rid)["config"]["chakudya_host"] or "", "")


# ================================================================= 7. dashboard
def test_dashboard_has_the_evaluation_page_and_keeps_text_out_of_browser_storage():
    js = (ROOT / "dashboard" / "app.js").read_text()
    for marker in ('id: "eval"', "/v1/llm/eval/sets", "/v1/llm/eval/runs", "/rating", "readiness", "evalPage"):
        assert marker in js, marker
    assert "localStorage" not in js[js.index("function evalPage"):js.index("function evalPage") + 20000]
