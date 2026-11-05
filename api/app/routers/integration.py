"""CNR integration endpoints.

CNR calls these with a `service` key. They add no new modelling rules: assessments go through the same
code as /v1/predict (deployed version only, hash check, strict features, rate limit, counts-only audit).
What they add is a stable contract for CNR: opaque case references echoed back, a correlation ID, and a
per-case result envelope. Case references and features are never stored or logged."""
import urllib.parse
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import audit, integration, serving, webhook
from ..catalog import SAFETY_NOTICE
from ..config import get_settings
from ..db import get_db
from ..models import ApiKey, AuditLog, ModelDefinition, ModelVersion, utcnow
from ..ratelimit import SlidingWindow
from ..security import Principal, require
from . import predict as predict_router

router = APIRouter(prefix="/v1/integration/cnr", tags=["integration"])
_probe_limiter = SlidingWindow(6, 60)


class Case(BaseModel):
    case_ref: str = Field(min_length=1, max_length=64)
    features: dict = Field(default_factory=dict)


class AssessRequest(BaseModel):
    cases: list[Case] = Field(min_length=1, max_length=50)


def _correlation(request: Request) -> str | None:
    raw = request.headers.get("x-request-id")
    return raw if raw and integration.CORRELATION.match(raw) else None


@router.get("/models", dependencies=[Depends(require("predict:models"))])
def models(db: Session = Depends(get_db)):
    """Deployed models with the feature contract CNR must follow when sending cases."""
    rows = db.execute(select(ModelDefinition, ModelVersion).join(
        ModelVersion, ModelVersion.model_definition_id == ModelDefinition.id)
        .where(ModelVersion.status == "deployed").order_by(ModelDefinition.id)).all()
    out = []
    for d, v in rows:
        block = predict_router._model_block(d, v)
        try:
            b = serving.load_bundle(v, d.slug)
            cats = serving.categories(b)
            block["features"] = {"numeric": b["numeric"],
                                 "categorical": [{"name": c, "values": cats.get(c, [])} for c in b["categorical"]]}
            block["classes"] = b.get("classes")
            block["available"] = True
        except serving.ServingError as e:
            block["available"], block["problem"] = False, e.message
        out.append(block)
    return {"contract": integration.CONTRACT, "safety_notice": SAFETY_NOTICE, "models": out}


@router.post("/assess/{slug}")
def assess(slug: str, body: AssessRequest, request: Request,
           p: Principal = Depends(require("predict:models")), db: Session = Depends(get_db)):
    refs = [c.case_ref for c in body.cases]
    for r in refs:
        problem = integration.case_ref_problem(r)
        if problem:
            raise HTTPException(422, problem)
    if len(set(refs)) != len(refs):
        raise HTTPException(422, "Each case_ref may appear only once per request.")
    corr = _correlation(request)
    extra = {"channel": "cnr", **({"correlation_id": corr} if corr else {})}
    resp = predict_router.run_prediction(db, request, p, slug, [c.features for c in body.cases],
                                         audit_action="integration.assess", extra=extra)
    resp["contract"] = integration.CONTRACT
    if corr:
        resp["correlation_id"] = corr
    resp["results"] = [{"case_ref": ref, **r} for ref, r in zip(refs, resp["results"])]
    return resp


@router.get("/status")
def status(request: Request, probe: bool = False, p: Principal = Depends(require("read:integration")),
           db: Session = Depends(get_db)):
    """Admin view: is the CNR address configured, are service keys in place, is CNR reachable, and how
    much has CNR used the lab recently. `probe=true` makes one outbound check (limited to 6 per minute)."""
    s = get_settings()
    problem = integration.config_problem(s)
    keys = db.scalars(select(ApiKey).where(ApiKey.role == "service", ApiKey.revoked_at.is_(None))).all()
    since = utcnow() - timedelta(hours=24)
    count = lambda *w: db.scalar(select(func.count()).select_from(AuditLog).where(AuditLog.ts >= since, *w)) or 0  # noqa: E731
    notified = db.scalars(select(AuditLog.detail).where(AuditLog.ts >= since, AuditLog.action == "integration.notify")).all()
    deployed = db.scalar(select(func.count()).select_from(ModelVersion).where(ModelVersion.status == "deployed")) or 0
    out = {
        "contract": integration.CONTRACT,
        "cnr_api": {"configured": problem is None, "problem": problem,
                    "base_url": s.cnr_api_base_url if problem is None else None,
                    "health_path": s.cnr_api_health_path},
        "webhook": {"configured": webhook.config_problem(s) is None, "problem": webhook.config_problem(s),
                    "host": urllib.parse.urlsplit(s.cnr_webhook_url).hostname if webhook.config_problem(s) is None else None,
                    "last_24h": {"delivered": sum(1 for d in notified if (d or {}).get("delivered")),
                                 "attempted": len(notified)}},
        "service_keys": [{"id": k.id, "name": k.name, "prefix": k.prefix, "created_at": k.created_at,
                          "last_used_at": k.last_used_at} for k in keys],
        "deployed_models": deployed,
        "last_24h": {"assessments": count(AuditLog.action == "integration.assess"),
                     "refused": count(AuditLog.action == "prediction.refused")},
        "probe": None,
    }
    if probe:
        if not _probe_limiter.hit(str(p.key_id)):
            raise HTTPException(429, "Too many checks. Wait a minute.", headers={"Retry-After": "60"})
        out["probe"] = integration.probe(s)
        audit.record(db, "integration.probe", principal=p, request=request, resource_type="cnr_api",
                     detail={"reachable": out["probe"]["reachable"], "http_status": out["probe"].get("http_status")})
    return out
