"""Lab -> CNR model-status events (Phase 6).

When a model is deployed, rolled back or withdrawn, the lab tells CNR so it can stop offering (or start offering)
the estimate. The event holds only the model slug, version number and what happened: no patient data, no
evaluation, no free text. It is signed with HMAC-SHA256 over "<timestamp>.<body>" using a secret shared with CNR,
so CNR can reject forged or replayed calls. Delivery is best effort in a background thread (a slow or unreachable
CNR never delays a deploy) with three attempts; the outcome is audited as `integration.notify`.
CNR also learns the truth on every assessment (it gets a 409 when nothing is deployed), so a lost event is not unsafe."""
import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .config import Settings, get_settings
from .integration import CONTRACT, _NoRedirect
from .models import utcnow

log = logging.getLogger("cnr-ml")
EVENTS = {"deployed": "model.deployed", "rollback": "model.rolled_back", "withdrawn": "model.retired"}
ATTEMPTS = 3
_BACKOFF_S = (1.0, 2.0)


def config_problem(s: Settings) -> str | None:
    """None when the webhook is usable, a short reason otherwise. An empty URL means 'off', reported as such."""
    url = (s.cnr_webhook_url or "").strip()
    if not url:
        return "CNR_WEBHOOK_URL is not set (events are off)."
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("https", "http") or not u.hostname:
        return "CNR_WEBHOOK_URL must be an http(s) address."
    if u.username or u.password:
        return "CNR_WEBHOOK_URL must not contain credentials."
    if s.is_production and u.scheme != "https":
        return "CNR_WEBHOOK_URL must use https in production."
    if len(s.cnr_webhook_secret or "") < 32:
        return "CNR_WEBHOOK_SECRET must be at least 32 characters."
    return None


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def build_event(kind: str, slug: str, version: int, version_id: int, *, reason: str | None = None,
                superseded_version: int | None = None) -> dict:
    return {"event_id": uuid.uuid4().hex, "event": EVENTS[kind], "occurred_at": utcnow().isoformat() + "Z",
            "contract": CONTRACT,
            "model": {"slug": slug, "version": version, "version_id": version_id},
            "detail": {k: v for k, v in {"reason": reason, "superseded_version": superseded_version}.items() if v is not None}}


def _post(s: Settings, event: dict) -> int:
    """One signed POST. Returns the HTTP status; raises on network errors. Redirects are not followed."""
    body = json.dumps(event, separators=(",", ":")).encode()
    ts = str(int(time.time()))
    req = urllib.request.Request(s.cnr_webhook_url, data=body, method="POST", headers={
        "Content-Type": "application/json", "User-Agent": "cnr-ml-lab/" + s.service_version,
        "X-ML-Lab-Timestamp": ts, "X-ML-Lab-Event-Id": event["event_id"],
        "X-ML-Lab-Signature": sign(s.cnr_webhook_secret, ts, body)})
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=s.cnr_api_timeout_s) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _deliver(event: dict) -> None:
    from .db import SessionLocal            # local import: this runs in its own thread with its own session
    from . import audit
    s = get_settings()
    status, error = None, None
    for i in range(ATTEMPTS):
        try:
            status = _post(s, event)
            error = None
            if 200 <= status < 300:
                break
            if 400 <= status < 500 and status != 429:   # CNR refused it; retrying will not help
                break
        except Exception as e:
            error = type(e).__name__
        if i < ATTEMPTS - 1:
            time.sleep(_BACKOFF_S[i])
    try:
        with SessionLocal() as db:
            audit.record(db, "integration.notify", resource_type="model", resource_id=event["model"]["slug"],
                         detail={"event": event["event"], "event_id": event["event_id"], "version": event["model"]["version"],
                                 "delivered": bool(status and 200 <= status < 300), "http_status": status, "error": error})
    except Exception:
        log.exception("Could not audit the CNR notification")


def _spawn(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True).start()


def emit(kind: str, slug: str, version: int, version_id: int, **detail) -> bool:
    """Queue one event for CNR. Never raises and never blocks. False when events are off."""
    try:
        if config_problem(get_settings()) is not None:
            return False
        _spawn(_deliver, build_event(kind, slug, version, version_id, **detail))
        return True
    except Exception:
        log.exception("Could not queue the CNR notification")
        return False
