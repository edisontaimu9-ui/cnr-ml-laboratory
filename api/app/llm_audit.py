"""Audit entries for LLM features: counts, booleans and short fixed codes only.

The general audit helper stores whatever dict it is given. For anything that touches clinical text that is too trusting, so LLM code
goes through this allow-list instead: a field is kept only if its name is listed below and its value has the listed shape. A value
that does not fit is dropped (never stored, never logged; only its field name is logged), so free text cannot reach the audit log by
mistake, whatever a caller passes."""
import logging
import re

from fastapi import Request
from sqlalchemy.orm import Session

from . import audit

log = logging.getLogger("cnr-ml")
_CODE = re.compile(r"^[a-z0-9_.:-]{1,40}$")
_TYPE = re.compile(r"^[A-Z_]{1,20}$")
# field -> "int" (a non-negative integer), "bool", "code" (fixed lower-case code), "counts" (dict of UPPER_CASE type -> int)
ALLOWED = {
    "chars": "int", "identifiers_found": "int", "identifier_types": "counts", "decision": "code", "category": "code", "kind": "code",
    "mode": "code", "masked": "bool", "refused": "bool", "scope": "code", "retry_after_s": "int", "policy": "code", "reason": "code",
    # phase 2 (ask / retrieve): counts and fixed codes about the exchange, never its text
    "context": "code", "intent": "code", "cache": "code", "top_k": "int", "history_turns": "int", "sources": "int", "cited": "int",
    "invalid_citations": "int", "latency_ms": "int", "output_blocked": "bool", "http_status": "int",
    # phase 3 (evaluation): counts and fixed codes about sets, runs and ratings, never question, answer or comment text
    "questions": "int", "completed": "int", "errors": "int", "split": "code", "expected_sources": "int", "expected_facts": "int",
    "should_refuse": "bool", "accuracy": "int", "supported": "bool", "safe": "bool", "comment_chars": "int", "name_chars": "int",
    # phase 4 (document summarisation): counts and fixed codes about extraction, validation and review, never note or summary text
    "facts": "int", "sentences": "int", "sentences_valid": "int", "sentences_rejected": "int", "numbers_checked": "int",
    "reject_reasons": "counts", "model_used": "bool", "model": "code", "provider": "code", "prompt": "code", "tokens_in": "int", "tokens_out": "int",
    "examples": "int", "review": "code", "attest_facts": "bool", "attest_numbers": "bool", "edited": "bool", "age_s": "int", "flagged_sentences": "int",
}


def clean(fields: dict) -> dict:
    out = {}
    for name, value in fields.items():
        shape = ALLOWED.get(name)
        ok = False
        if shape == "int":
            ok = isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 10**9
        elif shape == "bool":
            ok = isinstance(value, bool)
        elif shape == "code":
            ok = isinstance(value, str) and bool(_CODE.match(value))
        elif shape == "counts":
            ok = isinstance(value, dict) and len(value) <= 12 and all(
                isinstance(k, str) and _TYPE.match(k) and isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 10**6 for k, v in value.items())
        if ok:
            out[name] = value
        else:
            log.warning("LLM audit field %r dropped (not an allowed shape).", name)   # the value is never logged
    return out


def record(db: Session, action: str, *, principal=None, request: Request | None = None, resource_id=None, **fields) -> None:
    audit.record(db, action, principal=principal, request=request, resource_type="llm", resource_id=resource_id, detail=clean(fields))
