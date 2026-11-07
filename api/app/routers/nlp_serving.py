"""Serving endpoints for deployed NLP models. Decision support only; inputs and outputs are never stored and the audit log keeps counts
only. Needs `predict:models` (the CNR service key has it). See nlp_serving for the safety rules."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, nlp_corpus, nlp_lab, nlp_serving
from ..config import get_settings
from ..db import get_db
from ..models import NlpModelVersion
from ..ratelimit import SlidingWindow
from ..security import Principal, require

router = APIRouter(prefix="/v1/nlp/serving", tags=["nlp"])
_limiter = SlidingWindow(get_settings().rate_limit_predict_per_min, 60)


class PredictBody(BaseModel):
    text: str = Field(min_length=1, max_length=nlp_lab.MAX_TEXT_CHARS)
    lang: str | None = Field(None, description="en, ny, tum, yao or mixed. Left out, the rule engine guesses it.")


@router.get("", dependencies=[Depends(require("predict:models"))])
def available(db: Session = Depends(get_db)):
    """The NLP models that are deployed right now."""
    rows = db.scalars(select(NlpModelVersion).where(NlpModelVersion.status == "deployed").order_by(NlpModelVersion.name)).all()
    return {"notice": nlp_serving.NOTICE, "models": [nlp_serving.block(v) for v in rows]}


@router.post("/{name}/predict")
def predict(name: str, body: PredictBody, request: Request, p: Principal = Depends(require("predict:models")), db: Session = Depends(get_db)):
    if not _limiter.hit(str(p.key_id)):
        raise HTTPException(429, "Too many requests. Wait a moment.", headers={"Retry-After": str(_limiter.retry_after(str(p.key_id)))})
    lang = (body.lang or "").strip().lower() or None
    if lang is not None and lang not in nlp_corpus.LANG_CODES:
        raise HTTPException(422, "lang must be one of " + ", ".join(sorted(nlp_corpus.LANG_CODES)))
    try:
        v = nlp_serving.deployed(db, name)
        result = nlp_serving.predict(v, body.text, lang)
    except nlp_serving.NlpServingError as e:
        audit.record(db, "nlp.serve_refused", principal=p, request=request, detail={"name": name, "status": e.status})
        raise HTTPException(e.status, e.message)
    audit.record(db, "nlp.served", principal=p, request=request, resource_type="nlp_model_version", resource_id=v.id,
                 detail={"name": v.name, "version": v.version, "model_used": result.get("model_used"), "chars": len(body.text)})
    return {"model": nlp_serving.block(v), "result": result, "notice": nlp_serving.NOTICE}
