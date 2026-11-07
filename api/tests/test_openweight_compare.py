"""Model comparison runs, the compare endpoint, and the fine-tuning data store / JSONL export (local fake servers, no real network)."""
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import SecretStr

from test_llm_phase3 import ADMIN, MUAC, Q, add, chakudya, client, eval_router, get_run, key, new_set, run_set  # noqa: F401  (fixtures)
from app import llm_openweight as ow
from app.routers import llm_train as train_router  # noqa: F401

OW_KEY = "ow-secret-" + "z" * 12


class _Model(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        _Model.seen.append({"auth": self.headers.get("Authorization"), "model": body["model"], "body": body})
        user = body["messages"][1]["content"]
        if body["model"] == "Qwen/Qwen3.5-9B":     # careful model: restates the first source and cites it
            src = re.search(r"\[1\] (.*)", user).group(1)
            text = "<think>x</think>" + src + " [1]"
        else:                                      # careless model: invents a figure, cites nothing
            text = "The cut-off is 999 mg."
        out = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200), self.send_header("Content-Type", "application/json"), self.send_header("Content-Length", str(len(out))), self.end_headers(), self.wfile.write(out)


@pytest.fixture()
def models(chakudya, monkeypatch):
    _Model.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = eval_router.get_settings().model_copy(update={"openweight_enabled": True, "openweight_api_url": f"http://127.0.0.1:{srv.server_address[1]}/v1",
                                                      "openweight_api_key": SecretStr(OW_KEY)})
    monkeypatch.setattr(eval_router, "get_settings", lambda: s)
    yield _Model
    srv.shutdown()


def muac_set(client):
    sid = new_set(client)
    qs = [Q(f"What is the MUAC cut-off for severe acute malnutrition? (variant {i})", [MUAC], ["115 mm"]) for i in range(8)] + [Q("Does the child have kwashiorkor?", refuse=True)]
    assert add(client, sid, *qs).status_code == 201
    return sid


def test_a_model_run_answers_from_the_same_retrieved_sources_and_never_calls_rag_ask(client, models, chakudya):
    sid = muac_set(client)
    rid = run_set(client, sid, model="qwen3.5-9b")
    r = get_run(client, rid)
    assert r["status"] == "completed" and r["config"]["generator"] == "qwen3.5-9b"
    assert r["summary"]["groundedness"]["fact_coverage"]["mean"] == 1.0
    assert r["summary"]["groundedness"]["grounded_rate"]["rate"] == 1.0
    assert not any(x["path"] == "/rag/ask" for x in chakudya.seen) and any(x["path"] == "/rag/retrieve" for x in chakudya.seen)
    assert all(x["auth"] == "Bearer " + OW_KEY and x["model"] == "Qwen/Qwen3.5-9B" for x in models.seen)
    assert OW_KEY not in json.dumps(r)


def test_model_must_be_allowed_and_configured(client, chakudya, models):
    sid = muac_set(client)
    assert client.post(f"/v1/llm/eval/sets/{sid}/runs", json={"model": "llama-9000"}, headers=ADMIN).status_code == 422


def test_compare_prefers_the_careful_model_and_flags_the_careless_one(client, models, chakudya):
    sid = muac_set(client)
    good, bad = run_set(client, sid, model="qwen3.5-9b"), run_set(client, sid, model="gemma-4-e4b")
    c = client.get(f"/v1/llm/eval/compare?a={good}&b={bad}", headers=ADMIN).json()
    assert c["verdict"] == "a_better" and c["paired_questions"] == 8 and c["mean_difference_b_minus_a"] == -1.0 and not c["b_may_replace_a"]
    rev = client.get(f"/v1/llm/eval/compare?a={bad}&b={good}", headers=ADMIN).json()
    assert rev["verdict"] == "b_better" and rev["readiness_both_sufficient"] is False and rev["b_may_replace_a"] is False   # too few questions to claim anything
    assert any("numeric support" in g for g in c["guards"])                                    # B invented a figure no source supports
    assert OW_KEY not in json.dumps(c)


def test_runs_on_different_questions_cannot_be_compared(client, models, chakudya):
    a, b = run_set(client, muac_set(client), model="qwen3.5-9b"), run_set(client, muac_set(client), model="qwen3.5-9b")
    assert client.get(f"/v1/llm/eval/compare?a={a}&b={b}", headers=ADMIN).status_code == 409
    assert client.get(f"/v1/llm/eval/compare?a={a}&b={a}", headers=ADMIN).status_code == 422


# ---------------------------------------------------------------- fine-tuning data
def ex(q="What defines severe acute malnutrition by MUAC?", passages=None, answer="A MUAC below 115 mm defines it [1].", **kw):
    return {"question": q, "passages": ["MUAC below 115 mm defines SAM in children 6 to 59 months."] if passages is None else passages, "answer": answer, **kw}


def test_examples_are_screened(client):
    ds = key(client, "ds-train", "data_scientist")
    post = lambda *e: client.post("/v1/llm/train/examples", json={"examples": list(e)}, headers=ds)  # noqa: E731
    assert post(ex("Q one for screening?")).status_code == 201
    assert post(ex("Q one for screening?")).status_code == 422                                        # duplicate
    assert post(ex("Q two?", answer="No citation here.")).status_code == 422                            # must cite
    assert post(ex("Q three?", answer="See [3].")).status_code == 422                                   # marker without passage
    assert post(ex("Q four?", answer="The child has kwashiorkor [1].")).status_code == 422              # states a diagnosis
    assert post(ex("Q five? call 0999123456")).status_code == 422                                       # identifier
    assert post(ex("Q six?", passages=["mail me at a.b@example.org"])).status_code == 422               # identifier in a passage


def test_heldout_gold_questions_can_never_be_training_examples(client):
    sid = new_set(client)
    assert add(client, sid, Q("Heldout only question about zinc?", facts=["10 mg"], split="heldout")).status_code == 201
    r = client.post("/v1/llm/train/examples", json={"examples": [ex("Heldout only   question about ZINC?")]}, headers=ADMIN)
    assert r.status_code == 422 and "held-out" in r.text


def test_review_needs_another_person_and_export_has_only_approved_in_the_run_time_format(client):
    ds = key(client, "ds-export", "data_scientist")
    viewer = key(client, "viewer-export", "viewer")
    r = client.post("/v1/llm/train/examples", json={"examples": [ex("Export question one?"), ex("Export question two?", split="val")]}, headers=ds)
    ids = [e["id"] for e in r.json()["examples"]]
    assert client.post(f"/v1/llm/train/examples/{ids[0]}/review", json={"decision": "approved"}, headers=ds).status_code == 403   # author, and not a reviewer role
    assert client.post(f"/v1/llm/train/examples/{ids[0]}/review", json={"decision": "approved"}, headers=ADMIN).status_code == 200
    assert client.post(f"/v1/llm/train/examples/{ids[1]}/review", json={"decision": "rejected"}, headers=ADMIN).status_code == 200
    assert client.get("/v1/llm/train/export?split=val", headers=ADMIN).status_code == 404                                            # rejected is never exported
    assert client.get("/v1/llm/train/export", headers=viewer).status_code == 403
    out = client.get("/v1/llm/train/export?split=train", headers=ds)
    assert out.status_code == 200 and out.headers["content-type"].startswith("application/x-ndjson")
    assert out.headers["x-content-sha256"] == hashlib.sha256(out.content).hexdigest()
    rows = [json.loads(x) for x in out.text.splitlines()]
    mine = next(x for x in rows if "Export question one?" in x["messages"][1]["content"])
    assert mine["messages"][:2] == ow.build_rag_messages("Export question one?", [{"text": "MUAC below 115 mm defines SAM in children 6 to 59 months."}])
    assert mine["messages"][2] == {"role": "assistant", "content": "A MUAC below 115 mm defines it [1]."}
    assert not any("Export question two?" in x["messages"][1]["content"] for x in rows)
    assert client.delete(f"/v1/llm/train/examples/{ids[0]}", headers=ADMIN).status_code == 409                                      # approved cannot be deleted


def test_training_audit_rows_hold_counts_only(client):
    from test_llm_phase3 import audit_rows
    rows = audit_rows("llm.train")
    assert rows and all(set(r["detail"]) <= {"examples", "split", "review"} for r in rows)
