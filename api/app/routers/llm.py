"""LLM intelligence, phase 1: the contract and the safety gate.

`POST /v1/llm/precheck` runs exactly the checks every later LLM endpoint runs before any text leaves the laboratory: the per-key
rate limit, the decision-support boundary (for questions) and the identifier gate. It calls no external service and stores nothing.
`GET /v1/llm/contract` states the rules; `GET /v1/llm/status` (admin) says whether Chakudya is configured, without revealing the key.
Audit entries (`llm.precheck`, `llm.rate_limited`) hold counts and fixed codes only.

Phase 2 adds `POST /v1/llm/ask` (Chakudya `/rag/ask`: grounded answer with citations) and `POST /v1/llm/retrieve` (Chakudya `/rag/retrieve`:
source chunks only). Both go through the phase 1 gate: rate limit, decision-support screen, identifier gate, then the server-side gateway.
No session id is forwarded and nothing is stored. The answer is screened again before it is returned (`llm_rag.shape_ask`). No new
language model is involved: the answer is Chakudya's own."""
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import llm_audit, llm_gateway, llm_limits, llm_policy, llm_rag, nlp_lab
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


def _gate_for(permission: str):
    def dep(request: Request, p: Principal = Depends(require(permission)), db: Session = Depends(get_db)) -> Gated:
        try:
            return Gated(p, llm_limits.acquire(p.key_id))
        except llm_limits.RateLimited as e:
            with _throttle_lock:
                first = _audit_throttle.hit(str(p.key_id))
            if first:
                llm_audit.record(db, "llm.rate_limited", principal=p, request=request, scope=e.scope, retry_after_s=e.retry_after)
            raise HTTPException(429, f"Too many requests this {e.scope}. Wait and try again.", headers={"Retry-After": str(e.retry_after)})
    return dep


# Authentication and role first (401/403 do not use the allowance), then the per-key rate limit. Reused by every LLM endpoint.
gate = _gate_for("use:llm")
gate_summarise = _gate_for("summarise:llm")      # phase 4: same limiter, a different role (clinical reviewer, admin)


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
            "stored": False, "endpoints": ["GET /v1/llm/contract", "POST /v1/llm/precheck", "POST /v1/llm/ask", "POST /v1/llm/retrieve", "POST /v1/llm/summarise", "POST /v1/llm/summarise/review", "GET /v1/llm/summarise/status", "GET /v1/llm/status (admin)"], "notice": llm_policy.NOTICE}


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


# ---------------------------------------------------------------- phase 2: ask and retrieve
class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    context: Literal["clinical", "general", "both"] = "both"
    top_k: int = Field(default=6, ge=1, le=10)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)    # only the last 6 turns are used
    identifiers: Literal["mask", "reject"] | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    context: Literal["clinical", "general", "both"] = "both"
    top_k: int = Field(default=5, ge=1, le=20)
    identifiers: Literal["mask", "reject"] | None = None


def _refusal(db, g, request, action, rid, base, category, message, status=422, **extra) -> JSONResponse:
    found = {"identifiers_found": extra["found"], "identifier_types": extra["types"]} if "found" in extra else {}
    llm_audit.record(db, action, principal=g.principal, request=request, decision="refused", category=category, refused=True, **base, **found)
    return JSONResponse(status_code=status, headers=g.headers, content=llm_policy.envelope(
        request_id=rid, detail=message, decision="refused", category=category, **{k: v for k, v in extra.items() if k in ("found", "types", "limit")}))


def _failure(db, g, request, action, rid, base, reason, message, status) -> JSONResponse:
    llm_audit.record(db, action, principal=g.principal, request=request, decision="error", reason=reason, refused=False, http_status=status, **base)
    return JSONResponse(status_code=status, headers=g.headers, content=llm_policy.envelope(request_id=rid, detail=message, decision="error"))


def _call(db, g, request, action, rid, base, path, payload, shape):
    """Gateway call and response shaping with key-free, text-free failures. Returns (Shaped, None) or (None, JSONResponse)."""
    s = get_settings()
    bad = llm_gateway.config_problem(s)
    if bad:
        return None, _failure(db, g, request, action, rid, base, "not_configured", bad, 503)
    t0 = time.monotonic()
    try:
        raw = llm_gateway.post_json(s, path, payload, correlation_id=rid)
    except llm_gateway.GatewayError as e:
        return None, _failure(db, g, request, action, rid, base, "upstream_error", str(e), 502)
    ms = int((time.monotonic() - t0) * 1000)
    try:
        return shape(raw, ms), None
    except llm_rag.ShapeError as e:
        return None, _failure(db, g, request, action, rid, base, "bad_shape", str(e), 502)


def _prepare_question(db, g, request, action, rid, base, text, mode):
    """The decision-support screen, then the identifier gate. Returns (Prepared, None) or (None, refusal response)."""
    category = llm_policy.screen_request(text)
    if category:
        return None, _refusal(db, g, request, action, rid, base, category, llm_policy.CATEGORIES[category])
    try:
        return llm_policy.prepare_for_external(text, mode, max_chars=get_settings().llm_max_text_chars), None
    except llm_policy.PolicyRefusal as e:
        return None, _refusal(db, g, request, action, rid, base, e.category, e.message, **e.details)


@router.post("/ask")
def ask(body: AskRequest, request: Request, g: Gated = Depends(gate), db: Session = Depends(get_db)):
    """A grounded, cited answer from Chakudya's RAG pipeline. Decision support only; nothing is stored and no session is created."""
    rid, mode = str(uuid.uuid4()), _mode(body.identifiers)
    base = dict(kind="question", mode=mode, chars=len(body.question), policy=llm_policy.POLICY_CODE, context=body.context, top_k=body.top_k)
    q, refusal = _prepare_question(db, g, request, "llm.ask", rid, base, body.question, mode)
    if refusal:
        return refusal

    # Earlier turns are used by Chakudya only to resolve a short follow-up, but they leave the laboratory too, so they pass the same gate.
    turns = [t for t in body.history[-llm_rag.MAX_HISTORY:] if t.text.strip()]
    sent_history, found, types = [], q.identifiers_found, dict(q.identifier_types)
    masked = q.masked
    for t in turns:
        clipped = t.text[:llm_rag.MAX_HISTORY_CHARS]
        if t.role == "user":
            category = llm_policy.screen_request(clipped)
            if category:
                return _refusal(db, g, request, "llm.ask", rid, base, category, llm_policy.CATEGORIES[category])
        try:
            p = llm_policy.prepare_for_external(clipped, mode, max_chars=llm_rag.MAX_HISTORY_CHARS)
        except llm_policy.PolicyRefusal as e:
            return _refusal(db, g, request, "llm.ask", rid, base, e.category, e.message, **e.details)
        found += p.identifiers_found
        masked = masked or p.masked
        for k, v in p.identifier_types.items():
            types[k] = types.get(k, 0) + v
        sent_history.append({"role": t.role, "text": p.text})

    payload = {"query": q.text, "context": body.context, "top_k": body.top_k}
    if sent_history:
        payload["history"] = sent_history
    base["history_turns"] = len(sent_history)
    shaped, failure = _call(db, g, request, "llm.ask", rid, base, "/rag/ask", payload,
                            lambda raw, ms: llm_rag.shape_ask(raw, context=body.context, top_k=body.top_k, history_turns=len(sent_history), latency_ms=ms))
    if failure:
        return failure
    llm_audit.record(db, "llm.ask", principal=g.principal, request=request, decision="masked" if masked else "allowed", masked=masked, refused=False,
                     mode=mode, chars=len(body.question), policy=llm_policy.POLICY_CODE, identifiers_found=found, identifier_types=types, **shaped.audit)
    out = llm_policy.envelope(request_id=rid, generated_at=utcnow().isoformat() + "Z", decision="masked" if masked else "allowed",
                              identifiers={"found": found, "types": types}, **shaped.body)
    out["warnings"] = list(dict.fromkeys(q.warnings + shaped.body["warnings"]))
    if q.masked:
        out["masked_question"] = q.text
    return JSONResponse(out, headers=g.headers)


@router.post("/retrieve")
def retrieve(body: RetrieveRequest, request: Request, g: Gated = Depends(gate), db: Session = Depends(get_db)):
    """Matching knowledge-base chunks, without a generated answer. Same gate as /ask."""
    rid, mode = str(uuid.uuid4()), _mode(body.identifiers)
    base = dict(kind="question", mode=mode, chars=len(body.query), policy=llm_policy.POLICY_CODE, context=body.context, top_k=body.top_k)
    q, refusal = _prepare_question(db, g, request, "llm.retrieve", rid, base, body.query, mode)
    if refusal:
        return refusal
    shaped, failure = _call(db, g, request, "llm.retrieve", rid, base, "/rag/retrieve", {"query": q.text, "context": body.context, "top_k": body.top_k},
                            lambda raw, ms: llm_rag.shape_retrieve(raw, context=body.context, top_k=body.top_k, latency_ms=ms))
    if failure:
        return failure
    decision = "masked" if q.masked else "allowed"
    llm_audit.record(db, "llm.retrieve", principal=g.principal, request=request, decision=decision, masked=q.masked, refused=False, mode=mode,
                     chars=len(body.query), policy=llm_policy.POLICY_CODE, identifiers_found=q.identifiers_found, identifier_types=q.identifier_types,
                     **shaped.audit)
    out = llm_policy.envelope(request_id=rid, generated_at=utcnow().isoformat() + "Z", decision=decision,
                              identifiers={"found": q.identifiers_found, "types": q.identifier_types}, warnings=q.warnings, **shaped.body)
    if q.masked:
        out["masked_question"] = q.text
    return JSONResponse(out, headers=g.headers)
