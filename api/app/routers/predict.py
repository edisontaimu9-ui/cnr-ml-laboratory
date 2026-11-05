"""Prediction endpoints (Phase 5). Decision support only: every response names the model version and
carries the safety notice. Inputs and outputs are never stored; the audit log records counts only."""
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, serving
from ..catalog import SAFETY_NOTICE
from ..config import get_settings
from ..db import get_db
from ..models import ModelDefinition, ModelVersion, utcnow
from ..ratelimit import SlidingWindow
from ..security import Principal, require

router = APIRouter(prefix="/v1/predict", tags=["predict"])
_limiter = SlidingWindow(get_settings().rate_limit_predict_per_min, 60)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=50)


def _model_block(d: ModelDefinition, v: ModelVersion) -> dict:
    return {"slug": d.slug, "name": d.name, "task": d.task_type, "version": v.version, "version_id": v.id,
            "deployed_at": v.deployed_at, "approved_by": v.approved_by,
            "clinical_validation_ref": v.clinical_validation_ref, "intended_use": v.intended_use,
            "evaluation": serving.headline(v.metrics_json or {})}


def _live(db: Session, slug: str) -> tuple[ModelDefinition, ModelVersion]:
    try:
        d, v = serving.deployed(db, slug)
    except serving.ServingError as e:
        raise HTTPException(e.status, e.message)
    if v is None:
        raise HTTPException(409, "No approved model is deployed for this target, so no prediction can be made.")
    return d, v


@router.get("", dependencies=[Depends(require("predict:models"))])
def available(db: Session = Depends(get_db)):
    """Models that are currently deployed and can be called."""
    rows = db.execute(select(ModelDefinition, ModelVersion).join(
        ModelVersion, ModelVersion.model_definition_id == ModelDefinition.id)
        .where(ModelVersion.status == "deployed").order_by(ModelDefinition.id)).all()
    return {"safety_notice": SAFETY_NOTICE, "models": [_model_block(d, v) for d, v in rows]}


@router.get("/{slug}", dependencies=[Depends(require("predict:models"))])
def schema(slug: str, db: Session = Depends(get_db)):
    """What the deployed model expects: feature names, types and the categories it knows."""
    d, v = _live(db, slug)
    try:
        b = serving.load_bundle(v, slug)
    except serving.ServingError as e:
        raise HTTPException(e.status, e.message)
    cats = serving.categories(b)
    return {**_model_block(d, v), "safety_notice": SAFETY_NOTICE,
            "features": {"numeric": b["numeric"], "categorical": [{"name": c, "values": cats.get(c, [])} for c in b["categorical"]]},
            "classes": b.get("classes"),
            "note": "Send only these fields. Leave out anything you do not know; missing values lower confidence."}


def run_prediction(db: Session, request: Request, p: Principal, slug: str, records: list[dict[str, Any]], *,
                   audit_action: str = "prediction.made", extra: dict | None = None) -> dict:
    """Shared by the direct endpoint and the CNR integration endpoint, so both enforce the same rules:
    per-key rate limit, deployed version only, hash check, strict input, audit of counts only."""
    if not _limiter.hit(str(p.key_id)):
        wait = _limiter.retry_after(str(p.key_id))
        raise HTTPException(429, "Too many prediction requests. Wait a moment and try again.",
                            headers={"Retry-After": str(wait)})
    rid = uuid.uuid4().hex[:12]
    try:
        d, v = _live(db, slug)
    except HTTPException as e:
        if e.status_code == 409:
            audit.record(db, "prediction.refused", principal=p, request=request, resource_type="model",
                         resource_id=slug, detail={"reason": "no deployed version", "request_id": rid, **(extra or {})})
        raise
    try:
        bundle = serving.load_bundle(v, slug)
        results = serving.predict(bundle, records)
    except serving.ServingError as e:
        if e.status == 503:
            audit.record(db, "prediction.refused", principal=p, request=request, resource_type="model_version",
                         resource_id=v.id, detail={"reason": "artifact check failed", "request_id": rid, **(extra or {})})
        raise HTTPException(e.status, e.message)
    audit.record(db, audit_action, principal=p, request=request, resource_type="model_version", resource_id=v.id,
                 detail={"model": slug, "version": v.version, "records": len(results), "request_id": rid,
                         "low_confidence": sum(1 for r in results if r["risk_indicators"]["low_confidence"]),
                         **(extra or {})})
    m = v.metrics_json or {}
    caveats = list(m.get("limitations", []))[:3]
    if (m.get("warnings") or []):
        caveats.insert(0, f"This model was approved with {len(m['warnings'])} evaluation warning(s). See the registry.")
    return {"request_id": rid, "generated_at": utcnow().isoformat() + "Z", "decision_support_only": True,
            "safety_notice": SAFETY_NOTICE, "model": _model_block(d, v), "caveats": caveats, "results": results}


@router.post("/{slug}")
def make_prediction(slug: str, body: PredictRequest, request: Request,
                    p: Principal = Depends(require("predict:models")), db: Session = Depends(get_db)):
    return run_prediction(db, request, p, slug, body.records)
