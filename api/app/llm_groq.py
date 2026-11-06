"""The one place that knows the Groq address and API key (LLM phase 4: clinical document summarisation).

Same rules as `llm_gateway` (Chakudya), applied to a second provider:
- The key comes from the server's environment (GROQ_API_KEY) and is held as a SecretStr. No request, header, body or dashboard setting
  can supply a key or change the address. A request may choose a model only from the allow-list in GROQ_MODELS.
- The key is sent only as a Bearer header to the configured address, never logged, audited, returned or put in an error message.
  Redirects are refused, so the key cannot follow one to another host. In production the address must be https and a groq.com host.
- One route only: the OpenAI-compatible chat completions route. Callers hand over a prompt that holds extracted facts, never the note.
  This module never inspects, stores or logs request or response bodies.
"""
import json
import urllib.error
import urllib.parse
import urllib.request

from .config import Settings
from .integration import _NoRedirect

PATH = "/chat/completions"
MIN_KEY_CHARS = 16


class GroqError(RuntimeError):
    """A short message that never contains the key, a URL path or any request or response text."""


def models(s: Settings) -> list[str]:
    allowed = [m.strip() for m in (s.groq_models or "").split(",") if m.strip()]
    if s.groq_model.strip() and s.groq_model.strip() not in allowed:
        allowed.insert(0, s.groq_model.strip())
    return allowed


def default_model(s: Settings) -> str:
    return s.groq_model.strip()


def _key(s: Settings) -> str:
    return s.groq_api_key.get_secret_value().strip()


def base_url(s: Settings) -> str:
    return (s.groq_api_url or "").strip().rstrip("/")


def key_problem(s: Settings) -> str | None:
    """None when the model can be called, otherwise a short reason (never contains the key)."""
    if not s.summary_enabled:
        return "Summarisation is off (SUMMARY_ENABLED=false)."
    url = base_url(s)
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("https", "http") or not u.hostname:
        return "The Groq API address must be an http(s) address."
    if u.username or u.password or u.query or u.fragment:
        return "The Groq API address must not contain credentials, a query or a fragment."
    if s.is_production and (u.scheme != "https" or not (u.hostname == "groq.com" or u.hostname.endswith(".groq.com"))):
        return "In production the Groq API address must be an https groq.com address."
    k = _key(s)
    if len(k) < MIN_KEY_CHARS or any(c.isspace() for c in k):
        return "GROQ_API_KEY is not set (or is too short)."
    if not default_model(s):
        return "GROQ_MODEL is not set."
    return None


def public_status(s: Settings) -> dict:
    """What a caller may see: enabled/ready, the host name, the model names and whether a key exists. Never the key or a path."""
    u = urllib.parse.urlsplit(base_url(s))
    return {"enabled": s.summary_enabled, "ready": key_problem(s) is None, "problem": key_problem(s), "provider": "groq", "host": u.hostname,
            "key_configured": len(_key(s)) >= MIN_KEY_CHARS, "default_model": default_model(s), "models": models(s)}


def chat_json(s: Settings, model: str, messages: list[dict], *, max_tokens: int, correlation_id: str | None = None) -> dict:
    """One authenticated chat completion that must answer with a JSON object. Returns {"content": str, "tokens_in": int, "tokens_out": int}.
    Raises GroqError with a key-free, text-free message."""
    bad = key_problem(s)
    if bad:
        raise GroqError(bad)
    if model not in models(s):
        raise GroqError("That model is not in the allow-list.")
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    headers = {"Authorization": "Bearer " + _key(s), "Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": "cnr-ml-lab/" + s.service_version}
    if correlation_id:
        headers["X-Request-Id"] = correlation_id[:64]
    req = urllib.request.Request(base_url(s) + PATH, data=json.dumps(payload).encode(), method="POST", headers=headers)
    limit = s.llm_max_response_kb * 1024
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=s.summary_timeout_s) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            body = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            raise GroqError(f"Groq answered HTTP {e.code} (a redirect). Redirects are not followed; check the configured address.")
        hint = {400: "the request was refused", 401: "the API key was refused", 403: "the API key is not allowed to use this model",
                404: "the model or route was not found", 413: "the request was too large", 429: "rate limited by Groq",
                500: "Groq could not complete the request", 503: "Groq is unavailable"}.get(e.code, "unexpected answer")
        raise GroqError(f"Groq answered HTTP {e.code}: {hint}.")
    except Exception as e:      # DNS, TLS, timeout, refused
        raise GroqError(f"No usable answer from Groq ({type(e).__name__}).")
    if len(body) > limit:
        raise GroqError(f"The answer is larger than {s.llm_max_response_kb} KB.")
    if "json" not in ctype:
        raise GroqError("Groq did not return JSON.")
    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise GroqError("Groq returned an answer in an unexpected shape.")
    if not isinstance(content, str):
        raise GroqError("Groq returned an answer in an unexpected shape.")
    u = data.get("usage") if isinstance(data, dict) else None
    ti = u.get("prompt_tokens") if isinstance(u, dict) else None
    to = u.get("completion_tokens") if isinstance(u, dict) else None
    ok = lambda v: v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 10**8 else 0   # noqa: E731
    return {"content": content, "tokens_in": ok(ti), "tokens_out": ok(to)}
