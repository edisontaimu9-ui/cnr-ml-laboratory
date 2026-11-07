"""LLM intelligence, phase 4: clinical document summarisation.

`POST /v1/llm/summarise` takes a de-identified clinical note and returns the facts the rule engine extracted and, optionally, a summary written
by a Groq-hosted open-weight model that was shown only those facts. The note passes the phase 1 identifier gate first; every sentence of the
summary is validated against the facts and the note (`llm_summary.validate_output`) and a sentence that fails is dropped. The result is
"pending clinician review": `POST /v1/llm/summarise/review` records the clinician's decision (counts and codes only).
The note, the facts and the summary are never stored: not in the database, not in the audit log, not in a log file.
"""
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import llm_audit, llm_extract, llm_groq, llm_openweight, llm_policy, llm_summary, nlp_lab
from ..config import get_settings
from ..db import get_db
from ..models import utcnow
from ..security import Principal, require
from . import llm as base                       # the phase 1 gate, refusal and failure helpers

router = APIRouter(prefix="/v1/llm/summarise", tags=["llm"])

NOTICE = ("Decision support only. The facts come from a rule-based extractor and the summary (if requested) from a language model that was shown only "
          "those facts. A summary sentence is shown only if its numbers, measurements and terms match the note, which does not make it complete or "
          "clinically correct. A clinician must review it before it is used. Nothing is stored.")


class SummariseRequest(BaseModel):
    note: str = Field(min_length=1, max_length=20000)
    identifiers: Literal["mask", "reject"] | None = None
    use_model: bool = True
    model: str | None = Field(default=None, max_length=60)
    provider: Literal["groq", "openweight"] = "groq"


class ReviewRequest(BaseModel):
    request_id: str = Field(min_length=36, max_length=36)
    decision: Literal["accepted", "accepted_with_edits", "rejected"]
    facts_checked: bool = False
    numbers_checked: bool = False


@router.get("/status", dependencies=[Depends(require("read:llm"))])
def status():
    s = get_settings()
    return {"summarisation": llm_groq.public_status(s), "openweight": llm_openweight.public_status(s), "prompt": llm_summary.PROMPT_VERSION, "extractor": nlp_lab.ENGINE_VERSION,
            "limits": {"max_note_chars": s.summary_max_chars, "max_facts": llm_extract.MAX_FACTS, "max_sentences": llm_summary.MAX_SENTENCES,
                       "review_ttl_s": s.summary_review_ttl_s},
            "review_checklist": llm_summary.REVIEW_CHECKLIST, "extraction_limits": llm_extract.EXTRACTION_LIMITS,
            "stored": False, "notice": NOTICE}


@router.post("")
def summarise(body: SummariseRequest, request: Request, g: base.Gated = Depends(base.gate_summarise), db: Session = Depends(get_db)):
    s, rid, mode = get_settings(), str(uuid.uuid4()), base._mode(body.identifiers)
    prov = llm_openweight if body.provider == "openweight" else llm_groq
    perr = (llm_openweight.OpenWeightError, llm_groq.GroqError)
    model = (body.model or prov.default_model(s)) if body.use_model else None
    audit_base = dict(kind="document", mode=mode, chars=len(body.note), policy=llm_policy.POLICY_CODE, model_used=bool(body.use_model))
    if not s.summary_enabled:
        return base._failure(db, g, request, "llm.summarise", rid, audit_base, "not_configured", "Summarisation is off (SUMMARY_ENABLED=false).", 503)
    if body.use_model:
        if model not in prov.models(s):
            raise HTTPException(422, "That model is not in the allow-list. Choose one of: " + ", ".join(prov.models(s)) + ".")
        bad = prov.key_problem(s)
        if bad:
            return base._failure(db, g, request, "llm.summarise", rid, audit_base, "not_configured", bad + " Untick the model option to extract facts only.", 503)

    # 1. identifier gate (a note may legitimately say "diagnosed with", so no request screen; the OUTPUT is screened per sentence)
    try:
        note = llm_policy.prepare_for_external(body.note, mode, max_chars=s.summary_max_chars)
    except llm_policy.PolicyRefusal as e:
        return base._refusal(db, g, request, "llm.summarise", rid, audit_base, e.category, e.message, **e.details)

    # 2. facts from the rule engine (on the sanitised or masked text, never the original)
    ex = llm_extract.extract(note.text)
    facts = ex["facts"]
    summary, validation, usage, ms = None, None, {"tokens_in": 0, "tokens_out": 0}, 0
    warnings = list(dict.fromkeys(note.warnings + ex["warnings"]))
    if not facts:
        warnings.append("No facts were found in the note, so there is nothing to summarise.")

    # 3-4. model summary restricted to the facts, then validation
    if body.use_model and facts:
        t0 = time.monotonic()
        try:
            r = prov.chat_json(s, model, llm_summary.build_messages(facts), max_tokens=s.summary_max_output_tokens, correlation_id=rid)
            ms = int((time.monotonic() - t0) * 1000)
            validation = llm_summary.validate_output(r["content"], facts, note.text)
            usage = r
        except perr as e:
            return base._failure(db, g, request, "llm.summarise", rid, audit_base, "upstream_error", str(e), 502)
        except llm_summary.BadOutput as e:
            return base._failure(db, g, request, "llm.summarise", rid, audit_base, "bad_output", str(e), 502)
        flagged = sum(1 for x in validation.accepted if x.flags)
        cov = llm_summary.coverage(facts, validation.accepted)
        if validation.rejected:
            warnings.append(f"{validation.rejected} sentence(s) written by the model were dropped because they did not match the note and the facts.")
        if not validation.accepted:
            warnings.append("No sentence passed validation, so there is no summary. The extracted facts are still shown.")
        if flagged:
            warnings.append(f"{flagged} sentence(s) contain words that are not in the note or the facts; they are marked for the reviewer.")
        summary = {"generated_by": {"provider": body.provider, "model": model, "prompt": llm_summary.PROMPT_VERSION},
                   "sentences": [{"text": x.text, "facts": x.facts, "flags": x.flags} for x in validation.accepted],
                   "text": " ".join(x.text for x in validation.accepted), "rejected": {"count": validation.rejected, "reasons": validation.audit_reasons},
                   "coverage": cov, "latency_ms": ms}
    elif body.use_model:
        summary = {"generated_by": {"provider": body.provider, "model": model, "prompt": llm_summary.PROMPT_VERSION}, "sentences": [], "text": "",
                   "rejected": {"count": 0, "reasons": {}}, "coverage": {"facts": 0, "used": 0, "not_mentioned": []}, "latency_ms": 0}

    # 5. pending clinician review (id and counts only)
    llm_summary.pending.ttl = s.summary_review_ttl_s
    n_valid = len(validation.accepted) if validation else 0
    llm_summary.pending.add(rid, g.principal.key_id, facts=len(facts), sentences=(n_valid + validation.rejected) if validation else 0, sentences_valid=n_valid)

    decision = "masked" if note.masked else "allowed"
    audit = dict(decision=decision, masked=note.masked, refused=False, identifiers_found=note.identifiers_found, identifier_types=note.identifier_types,
                 facts=len(facts), model_used=bool(body.use_model), latency_ms=ms, tokens_in=usage["tokens_in"], tokens_out=usage["tokens_out"],
                 prompt=llm_summary.PROMPT_CODE, **{k: v for k, v in audit_base.items() if k != "model_used"})
    if model:
        audit["model"] = model
        audit["provider"] = body.provider
    if validation:
        audit.update(sentences=n_valid + validation.rejected, sentences_valid=n_valid, sentences_rejected=validation.rejected,
                     numbers_checked=validation.numbers_checked, reject_reasons=validation.audit_reasons, flagged_sentences=flagged)
    llm_audit.record(db, "llm.summarise", principal=g.principal, request=request, **audit)

    out = llm_policy.envelope(request_id=rid, generated_at=utcnow().isoformat() + "Z", decision=decision, notice=NOTICE,
                              identifiers={"found": note.identifiers_found, "types": note.identifier_types}, warnings=warnings,
                              extraction={"engine": ex["engine"], "language": ex["language"], "counts": ex["counts"], "facts": facts, "truncated": ex["truncated"],
                                          "limits": llm_extract.EXTRACTION_LIMITS},
                              summary=summary,
                              review={"required": True, "status": "pending_clinician_review", "expires_in_s": s.summary_review_ttl_s,
                                      "checklist": llm_summary.REVIEW_CHECKLIST, "endpoint": "POST /v1/llm/summarise/review"},
                              usable_for_patient=False)
    if note.masked:
        out["masked_note"] = note.text
    return JSONResponse(out, headers=g.headers)


@router.post("/review")
def review(body: ReviewRequest, request: Request, p: Principal = Depends(require("review:llm")), db: Session = Depends(get_db)):
    """The clinician's decision on one summary. Counts and codes only; the text is not sent here, kept or audited."""
    accepted = body.decision.startswith("accepted")
    if accepted and not (body.facts_checked and body.numbers_checked):
        raise HTTPException(422, "To accept, confirm that you checked the facts and the numbers against the note.")
    entry = llm_summary.pending.take(body.request_id, p.key_id)
    if entry is None:
        raise HTTPException(404, "This summary is unknown, has expired, was already reviewed, or was made with another key. Generate it again.")
    llm_audit.record(db, "llm.summary.reviewed", principal=p, request=request, resource_id=body.request_id[:36], review=body.decision,
                     attest_facts=body.facts_checked, attest_numbers=body.numbers_checked, edited=body.decision == "accepted_with_edits",
                     age_s=entry["age_s"], facts=entry["facts"], sentences=entry["sentences"], sentences_valid=entry["sentences_valid"])
    return llm_policy.envelope(request_id=body.request_id, review=body.decision, reviewed_at=utcnow().isoformat() + "Z", usable_for_patient=False,
                               detail=("Recorded. The summary is decision support only; you remain responsible for any use of it." if accepted
                                       else "Recorded as rejected. Do not use this summary."))
