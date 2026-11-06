"""The one place that knows Chakudya's address and consumer key for LLM features.

- Both come from the server's environment (CHAKUDYA_API_URL or CNR_API_BASE_URL, and CHAKUDYA_CONSUMER_KEY). No request, header,
  body or dashboard setting can change the address or supply a key, so the lab cannot be steered to another host.
- Key choice: CHAKUDYA_CONSUMER_KEY when set. Otherwise, only if LLM_USE_SURVEILLANCE_KEY=true, the CNR admin key already held for
  the sync (SURVEILLANCE_API_KEY). That key is broader than a consumer key, so it is sent only to the same host as CNR_API_BASE_URL,
  the host it is already sent to, never to a different CHAKUDYA_API_URL.
- The key is read only here, held as a SecretStr, sent only as a Bearer header to that address, and never logged, audited,
  returned or included in an error message. Redirects are refused, so the key cannot follow one to another host.
- Only the two RAG routes of the contract may be called (ALLOWED_PATHS). Callers hand over text that has already passed
  llm_policy.prepare_for_external; this module never inspects, stores or logs request or response bodies.
Phase 1 provides the transport and its rules; the endpoints that use it arrive in phase 2."""
import json
import urllib.error
import urllib.parse
import urllib.request

from .config import Settings
from .integration import _NoRedirect

ALLOWED_PATHS = ("/rag/ask", "/rag/retrieve")
MIN_KEY_CHARS = 16


class GatewayError(RuntimeError):
    """A short message that never contains the key, a URL path or any request or response text."""


def base_url(s: Settings) -> str:
    return (s.chakudya_api_url or s.cnr_api_base_url or "").strip().rstrip("/")


def _secret(s: Settings) -> tuple[str, str | None]:
    """(key, source). source is "consumer", "surveillance_admin" or None. The only place either key is read for LLM features."""
    consumer = s.chakudya_consumer_key.get_secret_value().strip()
    if consumer:
        return consumer, "consumer"
    if s.llm_use_surveillance_key:
        admin = (s.surveillance_api_key or "").strip()
        if admin:
            return admin, "surveillance_admin"
    return "", None


def _host(url: str) -> str:
    return (urllib.parse.urlsplit((url or "").strip()).netloc or "").lower()


def config_problem(s: Settings) -> str | None:
    """None when LLM features can call Chakudya, otherwise a short reason (shown to admins; never contains the key)."""
    if not s.llm_enabled:
        return "LLM features are off (LLM_ENABLED=false)."
    url = base_url(s)
    if not url:
        return "CHAKUDYA_API_URL (or CNR_API_BASE_URL) is not set."
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("https", "http") or not u.hostname:
        return "The Chakudya API address must be an http(s) address."
    if u.username or u.password:
        return "The Chakudya API address must not contain credentials."
    if u.query or u.fragment:
        return "The Chakudya API address must not contain a query or fragment."
    if s.is_production and u.scheme != "https":
        return "The Chakudya API address must use https in production."
    key, source = _secret(s)
    if len(key) < MIN_KEY_CHARS or any(c.isspace() for c in key):
        return ("SURVEILLANCE_API_KEY is not set (or is too short)." if s.llm_use_surveillance_key and source is None
                else "CHAKUDYA_CONSUMER_KEY is not set (or is too short).")
    if source == "surveillance_admin" and _host(url) != _host(s.cnr_api_base_url):
        return "The CNR admin key may only be used for the CNR address (CNR_API_BASE_URL); set CHAKUDYA_CONSUMER_KEY for a different host."
    return None


def public_status(s: Settings) -> dict:
    """What an admin may see: no key, no path, no query; only the host name and whether a key exists."""
    host = urllib.parse.urlsplit(base_url(s)).hostname if base_url(s) else None
    key, source = _secret(s)
    return {"enabled": s.llm_enabled, "ready": config_problem(s) is None, "problem": config_problem(s), "chakudya_host": host,
            "consumer_key_configured": len(s.chakudya_consumer_key.get_secret_value().strip()) >= MIN_KEY_CHARS,
            "key_source": source, "uses_admin_key": source == "surveillance_admin",
            "note": "Using the CNR admin key (broader than a consumer key). Only the two RAG routes can be called with it." if source == "surveillance_admin" else None}


def post_json(s: Settings, path: str, payload: dict, *, correlation_id: str | None = None) -> dict:
    """One authenticated POST to an allowed Chakudya route. Raises GatewayError with a key-free message."""
    if path not in ALLOWED_PATHS:
        raise GatewayError("That Chakudya route is not part of the LLM contract.")
    bad = config_problem(s)
    if bad:
        raise GatewayError(bad)
    headers = {"Authorization": "Bearer " + _secret(s)[0], "Content-Type": "application/json",
               "Accept": "application/json", "User-Agent": "cnr-ml-lab/" + s.service_version}
    if correlation_id:
        headers["X-Request-Id"] = correlation_id[:64]
    req = urllib.request.Request(base_url(s) + path, data=json.dumps(payload).encode(), method="POST", headers=headers)
    limit = s.llm_max_response_kb * 1024
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=s.llm_timeout_s) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            body = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            raise GatewayError(f"Chakudya answered HTTP {e.code} (a redirect). Redirects are not followed; check the configured address.")
        hint = {400: "the request was refused", 401: "the consumer key was refused", 403: "the consumer key is not allowed to use this route",
                404: "the route was not found", 429: "rate limited by Chakudya", 502: "Chakudya could not complete the request",
                503: "Chakudya is unavailable"}.get(e.code, "unexpected answer")
        raise GatewayError(f"Chakudya answered HTTP {e.code}: {hint}.")
    except Exception as e:   # DNS, TLS, timeout, refused
        raise GatewayError(f"No usable answer from the Chakudya API ({type(e).__name__}).")
    if len(body) > limit:
        raise GatewayError(f"The answer is larger than {s.llm_max_response_kb} KB.")
    if "json" not in ctype:
        raise GatewayError("Chakudya did not return JSON.")
    try:
        data = json.loads(body)
    except ValueError:
        raise GatewayError("Chakudya returned invalid JSON.")
    if not isinstance(data, dict):
        raise GatewayError("Chakudya returned an unexpected JSON shape.")
    return data
