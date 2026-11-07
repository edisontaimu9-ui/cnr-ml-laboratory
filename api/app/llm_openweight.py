"""Open-weight models served by an OpenAI-compatible endpoint you control (vLLM, SGLang, Ollama, RunPod, Together...).

Same rules as `llm_groq`: address and key come from the server environment only, no request can change them, redirects are refused,
the key is never logged, returned or put in an error. A request may only choose a model from the allow-list.

Catalogue (first choice Qwen3.5, second choice Gemma 4). Hugging Face ids are configuration: confirm them before relying on them.
A LoRA adapter served by vLLM (`--lora-modules name=path`) is added through OPENWEIGHT_SERVED, e.g. `cnr-qwen=cnr-qwen-lora`.
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from .config import Settings
from .integration import _NoRedirect

PATH = "/chat/completions"

CATALOG = {
    "qwen3.5-9b":   {"family": "qwen",  "choice": 1, "hf": "Qwen/Qwen3.5-9B",        "min_vram_gb_lora": 24,  "note": "QLoRA fits a 24 GB GPU (Colab L4, Kaggle)."},
    "qwen3.5-27b":  {"family": "qwen",  "choice": 1, "hf": "Qwen/Qwen3.5-27B",       "min_vram_gb_lora": 48,  "note": "QLoRA needs about 48 GB (A100 / L40S / H100)."},
    "gemma-4-e4b":  {"family": "gemma", "choice": 2, "hf": "google/gemma-4-E4B-it",  "min_vram_gb_lora": 16,  "note": "QLoRA fits 16 GB (Colab T4 tight, L4 fine)."},
    "gemma-4-12b":  {"family": "gemma", "choice": 2, "hf": "google/gemma-4-12B",     "min_vram_gb_lora": 24,  "note": "QLoRA fits 24 GB. Check the repo for an -it variant."},
}


class OpenWeightError(RuntimeError):
    """A short message that never contains the key, a URL path or any request or response text."""


def _split(v: str) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


def models(s: Settings) -> list[str]:
    allowed = _split(s.openweight_models)
    d = s.openweight_model.strip()
    if d and d not in allowed:
        allowed.insert(0, d)
    return allowed


def default_model(s: Settings) -> str:
    return s.openweight_model.strip()


def served_name(s: Settings, model: str) -> str:
    """The name the server knows the model by: OPENWEIGHT_SERVED override ("id=name,id=name"), else the Hugging Face id, else the id."""
    for pair in _split(s.openweight_served):
        k, _, v = pair.partition("=")
        if k.strip() == model and v.strip():
            return v.strip()
    return CATALOG.get(model, {}).get("hf", model)


def _key(s: Settings) -> str:
    return s.openweight_api_key.get_secret_value().strip()


def base_url(s: Settings) -> str:
    return (s.openweight_api_url or "").strip().rstrip("/")


def key_problem(s: Settings) -> str | None:
    """For summarisation: needs SUMMARY_ENABLED as well."""
    if not s.summary_enabled:
        return "Summarisation is off (SUMMARY_ENABLED=false)."
    return problem(s)


def problem(s: Settings) -> str | None:
    """None when the model server can be called (used by comparison runs, which do not need SUMMARY_ENABLED)."""
    if not s.openweight_enabled:
        return "Open-weight models are off (OPENWEIGHT_ENABLED=false)."
    u = urllib.parse.urlsplit(base_url(s))
    if u.scheme not in ("https", "http") or not u.hostname:
        return "OPENWEIGHT_API_URL must be an http(s) address."
    if u.username or u.password or u.query or u.fragment:
        return "OPENWEIGHT_API_URL must not contain credentials, a query or a fragment."
    if s.is_production and u.scheme != "https":
        return "In production OPENWEIGHT_API_URL must use https."
    k = _key(s)
    if k and (len(k) < 8 or any(c.isspace() for c in k)):
        return "OPENWEIGHT_API_KEY is too short or contains spaces."
    if not default_model(s):
        return "OPENWEIGHT_MODEL is not set."
    return None


def public_status(s: Settings) -> dict:
    u = urllib.parse.urlsplit(base_url(s))
    bad = key_problem(s)
    return {"enabled": s.openweight_enabled, "ready": bad is None, "problem": bad, "provider": "openweight", "host": u.hostname,
            "server_ready": problem(s) is None, "key_configured": bool(_key(s)), "default_model": default_model(s), "models": models(s),
            "catalog": [{"id": k, **v, "served_as": served_name(s, k), "allowed": k in models(s)} for k, v in CATALOG.items()]}


_THINK = re.compile(r"<think>.*?</think>|<\|channel>thought.*?<channel\|>", re.S)


def chat_json(s: Settings, model: str, messages: list[dict], *, max_tokens: int, correlation_id: str | None = None) -> dict:
    """One chat completion that must answer with a JSON object. Returns {"content", "tokens_in", "tokens_out"}."""
    bad = key_problem(s)
    if bad:
        raise OpenWeightError(bad)
    return _chat(s, model, messages, max_tokens=max_tokens, correlation_id=correlation_id, json_mode=True)


def chat_text(s: Settings, model: str, messages: list[dict], *, max_tokens: int, correlation_id: str | None = None) -> dict:
    """One plain-text chat completion (used by comparison runs). Same return shape as chat_json."""
    return _chat(s, model, messages, max_tokens=max_tokens, correlation_id=correlation_id, json_mode=False)


MAX_PASSAGE_CHARS = 1200
RAG_RULES = ("\nAnswer the QUESTION using only the numbered SOURCES. Cite every statement as [n]. If the sources do not contain the answer, say that "
             "they do not. Keep numbers and units exactly as in the sources.")


def build_rag_messages(question: str, passages: list[dict]) -> list[dict]:
    """The one prompt format for answering from sources. Comparison runs and the training-data export both use it, so a fine-tuned model
    is trained on exactly what it will be asked. passages: [{"text": ...}] (numbered in order)."""
    from . import llm_policy
    src = "\n".join(f"[{i}] {p['text'][:MAX_PASSAGE_CHARS]}" for i, p in enumerate(passages, 1)) or "(no sources)"
    return [{"role": "system", "content": llm_policy.SYSTEM_RULES + RAG_RULES}, {"role": "user", "content": f"SOURCES:\n{src}\n\nQUESTION: {question}"}]


def _chat(s: Settings, model: str, messages: list[dict], *, max_tokens: int, correlation_id: str | None, json_mode: bool) -> dict:
    bad = problem(s)
    if bad:
        raise OpenWeightError(bad)
    if model not in models(s):
        raise OpenWeightError("That model is not in the allow-list.")
    payload = {"model": served_name(s, model), "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if s.openweight_disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}   # Qwen3.5 and Gemma 4 both honour this on vLLM
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "cnr-ml-lab/" + s.service_version}
    if _key(s):
        headers["Authorization"] = "Bearer " + _key(s)
    if correlation_id:
        headers["X-Request-Id"] = correlation_id[:64]
    req = urllib.request.Request(base_url(s) + PATH, data=json.dumps(payload).encode(), method="POST", headers=headers)
    limit = s.llm_max_response_kb * 1024
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=s.openweight_timeout_s) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            body = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            raise OpenWeightError(f"The model server answered HTTP {e.code} (a redirect). Redirects are not followed; check OPENWEIGHT_API_URL.")
        hint = {400: "the request was refused", 401: "the API key was refused", 403: "the API key is not allowed", 404: "the model or route was not found",
                413: "the request was too large", 429: "rate limited", 500: "the server could not complete the request", 503: "the server is unavailable"}.get(e.code, "unexpected answer")
        raise OpenWeightError(f"The model server answered HTTP {e.code}: {hint}.")
    except Exception as e:
        raise OpenWeightError(f"No usable answer from the model server ({type(e).__name__}).")
    if len(body) > limit:
        raise OpenWeightError(f"The answer is larger than {s.llm_max_response_kb} KB.")
    if "json" not in ctype:
        raise OpenWeightError("The model server did not return JSON.")
    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise OpenWeightError("The model server returned an answer in an unexpected shape.")
    if not isinstance(content, str):
        raise OpenWeightError("The model server returned an answer in an unexpected shape.")
    content = _THINK.sub("", content).strip()
    u = data.get("usage") if isinstance(data, dict) else None
    ok = lambda v: v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 10**8 else 0   # noqa: E731
    return {"content": content, "tokens_in": ok(u.get("prompt_tokens") if isinstance(u, dict) else None),
            "tokens_out": ok(u.get("completion_tokens") if isinstance(u, dict) else None)}
