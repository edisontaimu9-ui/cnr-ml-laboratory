"""Open-weight provider: allow-list, served names, thinking-tag stripping, server-side key, summarise hook (local fake server, no real network)."""
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/ow.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
from app import llm_openweight as ow  # noqa: E402
from app.config import Settings  # noqa: E402

KEY = "ow-secret-" + "k" * 12


class _H(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        _H.seen.append({"auth": self.headers.get("Authorization"), "body": body, "path": self.path})
        out = json.dumps({"choices": [{"message": {"content": "<think>hmm</think>{\"sentences\": []}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()
        self.send_response(200), self.send_header("Content-Type", "application/json"), self.send_header("Content-Length", str(len(out))), self.end_headers(), self.wfile.write(out)


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _H.seen.clear()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def cfg(url, **kw):
    return Settings(summary_enabled=True, openweight_enabled=True, openweight_api_url=url, openweight_api_key=KEY, **kw)


def test_catalogue_order_and_ids():
    assert [k for k, v in ow.CATALOG.items() if v["choice"] == 1] == ["qwen3.5-9b", "qwen3.5-27b"]
    assert [k for k, v in ow.CATALOG.items() if v["choice"] == 2] == ["gemma-4-e4b", "gemma-4-12b"]
    s = Settings()
    assert ow.default_model(s) == "qwen3.5-9b" and ow.models(s) == ["qwen3.5-9b", "qwen3.5-27b", "gemma-4-e4b", "gemma-4-12b"]


def test_off_by_default_and_problems():
    assert "off" in ow.key_problem(Settings(summary_enabled=True))
    assert ow.key_problem(cfg("ftp://x")) is not None
    assert ow.key_problem(Settings(summary_enabled=True, openweight_enabled=True, openweight_api_url="http://h/v1", environment="production"))


def test_call_uses_served_name_key_and_strips_thinking(server):
    s = cfg(server, openweight_served="gemma-4-e4b=my-gemma")
    r = ow.chat_json(s, "gemma-4-e4b", [{"role": "user", "content": "x"}], max_tokens=50)
    assert json.loads(r["content"]) == {"sentences": []} and r["tokens_in"] == 5
    seen = _H.seen[0]
    assert seen["auth"] == "Bearer " + KEY and seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "my-gemma" and seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    ow.chat_json(s, "qwen3.5-9b", [{"role": "user", "content": "x"}], max_tokens=50)
    assert _H.seen[1]["body"]["model"] == "Qwen/Qwen3.5-9B"


def test_allow_list_and_secret_never_exposed(server):
    s = cfg(server)
    with pytest.raises(ow.OpenWeightError):
        ow.chat_json(s, "llama-9000", [], max_tokens=5)
    assert KEY not in json.dumps(ow.public_status(s)) and KEY not in repr(s)


def test_unreachable_error_has_no_key():
    with pytest.raises(ow.OpenWeightError) as e:
        ow.chat_json(cfg("http://127.0.0.1:9/v1"), "qwen3.5-9b", [], max_tokens=5)
    assert KEY not in str(e.value)


def test_summarise_endpoint_with_openweight_provider(server, monkeypatch):
    from fastapi.testclient import TestClient
    from app import config
    from app.main import app
    s = cfg(server)
    monkeypatch.setattr(config, "get_settings", lambda: s)
    import app.routers.llm_summary as r
    import app.routers.llm as base
    monkeypatch.setattr(r, "get_settings", lambda: s)
    monkeypatch.setattr(base, "get_settings", lambda: s)
    adm = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
    cm = TestClient(app)
    c = cm.__enter__()
    note = "Child, weight 7.2 kg, MUAC 112 mm. No oedema."
    ok = c.post("/v1/llm/summarise", json={"note": note, "provider": "openweight", "model": "gemma-4-e4b"}, headers=adm)
    assert ok.status_code == 200, ok.text
    assert ok.json()["summary"]["generated_by"]["provider"] == "openweight" and _H.seen[-1]["body"]["model"] == "google/gemma-4-E4B-it"
    assert KEY not in ok.text
    bad = c.post("/v1/llm/summarise", json={"note": note, "provider": "openweight", "model": "llama-9000"}, headers=adm)
    assert bad.status_code == 422
    st = c.get("/v1/llm/summarise/status", headers=adm).json()
    assert st["openweight"]["ready"] is True and KEY not in json.dumps(st)
    cm.__exit__(None, None, None)
