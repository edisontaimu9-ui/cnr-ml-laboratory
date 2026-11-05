"""CNR integration helpers.

The laboratory and CNR talk only over authenticated HTTPS. CNR calls the lab with a `service` key
(see routers/integration.py). This module covers the other direction in the one way that needs no
CNR secrets: a reachability check of the configured CNR API address. The address comes from the
environment, never from a request, and no credentials or patient data are ever sent to it."""
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import Settings

CONTRACT = "cnr-ml-contract/1"
CASE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
CORRELATION = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def case_ref_problem(ref: str) -> str | None:
    """case_ref must be an opaque reference CNR can map back itself, never something identifying."""
    if not isinstance(ref, str) or not CASE_REF.match(ref):
        return "case_ref must be 1-64 letters, digits or . _ : - characters."
    if re.search(r"\d{9,}", ref):
        return "case_ref looks like a phone or ID number. Send an opaque reference instead."
    return None


def config_problem(s: Settings) -> str | None:
    """None when the configured CNR address is usable, otherwise a short reason."""
    url = (s.cnr_api_base_url or "").strip()
    if not url:
        return "CNR_API_BASE_URL is not set."
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("https", "http") or not u.hostname:
        return "CNR_API_BASE_URL must be an http(s) address."
    if u.username or u.password:
        return "CNR_API_BASE_URL must not contain credentials."
    if s.is_production and u.scheme != "https":
        return "CNR_API_BASE_URL must use https in production."
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):   # a redirect is reported, never followed
        return None


def probe(s: Settings) -> dict:
    """One GET to the configured CNR address. Any HTTP answer means the service is reachable.
    No Authorization header and no body are sent."""
    bad = config_problem(s)
    if bad:
        return {"reachable": False, "reason": bad}
    path = s.cnr_api_health_path if s.cnr_api_health_path.startswith("/") else "/" + s.cnr_api_health_path
    url = s.cnr_api_base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "cnr-ml-lab/" + s.service_version})
    opener = urllib.request.build_opener(_NoRedirect)
    t0 = time.monotonic()
    try:
        with opener.open(req, timeout=s.cnr_api_timeout_s) as r:
            code = r.status
    except urllib.error.HTTPError as e:     # an error status is still an answer
        code = e.code
    except Exception as e:                  # DNS, TLS, timeout, refused
        return {"reachable": False, "reason": "No answer from the CNR API (" + type(e).__name__ + ").",
                "latency_ms": round((time.monotonic() - t0) * 1000)}
    return {"reachable": True, "http_status": code, "latency_ms": round((time.monotonic() - t0) * 1000)}
