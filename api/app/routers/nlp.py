"""NLP laboratory: extract structured nutrition information from free text (clinical notes, reports, research text,
dietary records) in English and Chichewa.

Safety rules, same spirit as predictions: text is processed in memory and never stored or logged (the audit log keeps counts
only); identifiers are masked, or the request is refused with `identifiers: reject`; the output describes what the text
says and makes no diagnosis. The engine is a rule-based baseline, and every response says so."""
import uuid
from collections import Counter
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from .. import audit, nlp_lab, nlp_lexicon
from ..config import get_settings
from ..db import get_db
from ..models import utcnow
from ..nlp_gold import run_benchmark
from ..ratelimit import SlidingWindow
from ..security import Principal, require

router = APIRouter(prefix="/v1/nlp", tags=["nlp"])
_limiter = SlidingWindow(get_settings().rate_limit_predict_per_min, 60)

NOTICE = ("Rule-based extraction: it reports what the text says and makes no diagnosis or nutrition-status decision. Terminology lists "
          "are developer-written and not yet clinically reviewed; terms that are not listed are not found. Chichewa support covers listed "
          "words only. Text is not stored. Do not submit identifiable patient notes.")


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=nlp_lab.MAX_TEXTS)
    identifiers: Literal["mask", "reject"] = "mask"


@router.get("/info", dependencies=[Depends(require("read:nlp"))])
def info():
    ents = nlp_lexicon.entries()
    return {"engine": nlp_lab.ENGINE_VERSION, "kind": "rule-based baseline", "labels": nlp_lexicon.LABELS, "languages": ["en", "ny"],
            "limits": {"texts_per_request": nlp_lab.MAX_TEXTS, "chars_per_text": nlp_lab.MAX_TEXT_CHARS,
                       "requests_per_minute": get_settings().rate_limit_predict_per_min},
            "lexicon": {"terms": len(ents), "by_label": dict(Counter(e["label"] for e in ents)),
                        "by_language": dict(Counter(e["lang"] for e in ents)), "clinically_reviewed": 0},
            "notice": NOTICE}


@router.get("/lexicon", dependencies=[Depends(require("read:nlp"))])
def lexicon(label: str | None = Query(None, max_length=20), lang: str | None = Query(None, max_length=4),
            q: str | None = Query(None, max_length=60), limit: int = Query(300, ge=1, le=500)):
    rows = nlp_lexicon.entries()
    if label:
        rows = [r for r in rows if r["label"] == label.upper()]
    if lang:
        rows = [r for r in rows if r["lang"] in (lang, "both")]
    if q and q.strip():
        needle = q.strip().lower()
        rows = [r for r in rows if needle in r["term"].lower() or needle in r["norm"].lower()]
    return {"total": len(rows), "terms": rows[:limit]}


@router.get("/benchmark", dependencies=[Depends(require("read:nlp"))])
def benchmark():
    return run_benchmark()


@router.post("/analyze")
def analyze(body: AnalyzeRequest, request: Request, p: Principal = Depends(require("run:nlp")), db: Session = Depends(get_db)):
    if not _limiter.hit(str(p.key_id)):
        raise HTTPException(429, "Too many requests. Wait a moment.", headers={"Retry-After": str(_limiter.retry_after(str(p.key_id)))})
    for i, t in enumerate(body.texts):
        if not t.strip():
            raise HTTPException(422, f"Text {i + 1} is empty.")
        if len(t) > nlp_lab.MAX_TEXT_CHARS:
            raise HTTPException(422, f"Text {i + 1} is longer than {nlp_lab.MAX_TEXT_CHARS} characters.")
    results = [nlp_lab.analyze(t) for t in body.texts]
    flagged = [i + 1 for i, r in enumerate(results) if r["identifiers"]["found"]]
    n_ents, n_phi = sum(len(r["entities"]) for r in results), sum(r["identifiers"]["found"] for r in results)
    # counts only: never the text, the entities or the identifiers
    audit.record(db, "nlp.analyze", principal=p, request=request, resource_type="nlp",
                 detail={"texts": len(body.texts), "chars": sum(len(t) for t in body.texts), "entities": n_ents,
                         "identifiers_found": n_phi, "mode": body.identifiers, "refused": bool(flagged and body.identifiers == "reject")})
    if flagged and body.identifiers == "reject":
        raise HTTPException(422, "Identifiers were found in text " + ", ".join(map(str, flagged)) + ". Remove them and send the text again.")
    return {"request_id": str(uuid.uuid4()), "generated_at": utcnow().isoformat() + "Z", "engine": nlp_lab.ENGINE_VERSION,
            "stored": False, "notice": NOTICE, "results": results}
