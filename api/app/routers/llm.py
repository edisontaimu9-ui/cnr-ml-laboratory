"""LLM intelligence, phase 1: the contract and the safety gate.

`POST /v1/llm/precheck` runs exactly the checks every later LLM endpoint runs before any text leaves the laboratory: the per-key
rate limit, the decision-support boundary (for questions) and the identifier gate. It calls no external service and stores nothing.
`GET /v1/llm/contract` states the rules; `GET /v1/llm/status` (admin) says whether Chakudya is configured, without revealing the key.
Audit entries (`llm.precheck`, `llm.rate_limited`) hold counts and fixed codes only."""
import threading
import uuid
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import llm_audit, llm_gateway, llm_limits, llm_policy, nlp_lab
from ..config import get_settings
from ..db import get_db
from ..models import utcnow
from ..ratelimit import SlidingWindow
from ..security import Principal, require

router = APIRouter(prefix="/v1/llm", tags=["llm"])
_audit_throttle = SlidingWindow(1, 60)     # at most one rate-limit audit row per key per minute, so a flood cannot flood the log
_throttle_lock = threading.Lock()


class PrecheckRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    kind: Literal["question", "document"] = "question"     # "document": identifiers only (a note may legitimately say "diagnosed with")
    identifiers: Literal["mask", "reject"] | None = None   # None -> LLM_IDENTIFIER_POLICY


@dataclass
class Gated:
    principal: Principal
    remaining: dict

    @property
    def headers(self) -> dict:
        return {"X-RateLimit-Remaining-Minute": str(self.remaining["minute"]), "X-RateLimit-Remaining-Hour": str(self.remaining["hour"])}


def gate(request: Request, p: Principal = Depends(require("use:llm")), db: Session = Depends(get_db)) -> Gated:
    """Authentication and role first (401/403 do not use the allowance), then the per-key rate limit. Reused by every LLM endpoint."""
    try:
        return Gated(p, llm_limits.acquire(p.key_id))
    except llm_limits.RateLimited as e:
        with _throttle_lock:
            first = _audit_throttle.hit(str(p.key_id))
        if first:
            llm_audit.record(db, "llm.rate_limited", principal=p, request=request, scope=e.scope, retry_after_s=e.retry_after)
        raise HTTPException(429, f"Too many requests this {e.scope}. Wait and try again.", headers={"Retry-After": str(e.retry_after)})


def _mode(requested: str | None) -> str:
    chosen = requested or get_settings().llm_identifier_policy
    return chosen if chosen in ("mask", "reject") else "reject"     # anything unexpected falls back to the stricter rule


@router.get("/contract", dependencies=[Depends(require("read:llm"))])
def contract():
    s = get_settings()
    return {"contract": llm_policy.CONTRACT, "policy": llm_policy.POLICY_VERSION, "rules": llm_policy.RULES, "limitations": llm_policy.LIMITATIONS,
            "refusal_categories": llm_policy.CATEGORIES,
            "identifiers": {"detector": nlp_lab.ENGINE_VERSION, "detects": ["EMAIL", "PHONE", "RECORD_ID", "ID_NUMBER", "NAME (titled)"],
                            "default": _mode(None), "choices": ["reject", "mask"], "max_chars": s.llm_max_text_chars},
            "limits": {"per_minute_per_key": s.llm_rate_limit_per_min, "per_hour_per_key": s.llm_rate_limit_per_hour},
            "audit": "Counts and fixed codes only; never text, questions, answers or identifiers.",
            "stored": False, "endpoints": ["GET /v1/llm/contract", "POST /v1/llm/precheck", "GET /v1/llm/status (admin)"], "notice": llm_policy.NOTICE}


@router.get("/status", dependencies=[Depends(require("read:integration"))])
def status():
    s = get_settings()
    st = llm_gateway.public_status(s)
    st.update({"identifier_policy": _mode(None), "identifier_policy_valid": s.llm_identifier_policy in ("mask", "reject"),
               "limits": {"per_minute_per_key": s.llm_rate_limit_per_min, "per_hour_per_key": s.llm_rate_limit_per_hour,
                          "max_text_chars": s.llm_max_text_chars}})
    return st


@router.post("/precheck")
def precheck(body: PrecheckRequest, request: Request, g: Gated = Depends(gate), db: Session = Depends(get_db)):
    """Would this text be allowed to go to an external service, and in what form? Nothing is sent and nothing is stored."""
    s, rid, mode = get_settings(), str(uuid.uuid4()), _mode(body.identifiers)
    audit_base = dict(kind=body.kind, mode=mode, chars=len(body.text), policy=llm_policy.POLICY_CODE)

    def refuse(category: str, message: str, status: int = 422, **extra) -> JSONResponse:
        llm_audit.record(db, "llm.precheck", principal=g.principal, request=request, decision="refused", category=category, refused=True,
                         **audit_base, **({"identifiers_found": extra["found"], "identifier_types": extra["types"]} if "found" in extra else {}))
        return JSONResponse(status_code=status, headers=g.headers, content=llm_policy.envelope(
            request_id=rid, detail=message, decision="refused", category=category, **{k: v for k, v in extra.items() if k in ("found", "types", "limit")}))

    if body.kind == "question":
        category = llm_policy.screen_request(body.text)
        if category:
            return refuse(category, llm_policy.CATEGORIES[category])
    try:
        prepared = llm_policy.prepare_for_external(body.text, mode, max_chars=s.llm_max_text_chars)
    except llm_policy.PolicyRefusal as e:
        return refuse(e.category, e.message, **e.details)
    decision = "masked" if prepared.masked else "allowed"
    llm_audit.record(db, "llm.precheck", principal=g.principal, request=request, decision=decision, masked=prepared.masked, refused=False,
                     identifiers_found=prepared.identifiers_found, identifier_types=prepared.identifier_types, **audit_base)
    out = llm_policy.envelope(request_id=rid, generated_at=utcnow().isoformat() + "Z", decision=decision, kind=body.kind, chars=len(body.text),
                              identifiers={"found": prepared.identifiers_found, "types": prepared.identifier_types}, warnings=prepared.warnings)
    if prepared.masked:
        out["masked_text"] = prepared.text
    return JSONResponse(out, headers=g.headers)
