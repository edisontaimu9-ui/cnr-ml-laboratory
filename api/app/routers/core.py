from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ..catalog import SAFETY_NOTICE
from ..config import get_settings
from ..db import get_db
from ..models import Dataset, DatasetVersion, Experiment, Exploration, ModelDefinition, ModelVersion
from ..schemas import MeOut
from ..security import Principal, get_principal

router = APIRouter()


@router.get("/health", tags=["system"])
def health(db: Session = Depends(get_db)):
    """Public liveness probe. Reveals nothing beyond up/down and version."""
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "unavailable"
    return {"status": "ok" if database == "ok" else "degraded", "database": database,
            "version": get_settings().service_version}


@router.get("/v1/me", response_model=MeOut, tags=["system"])
def me(p: Principal = Depends(get_principal)):
    return MeOut(name=p.name, role=p.role, permissions=sorted(p.permissions))


@router.get("/v1/overview", tags=["system"])
def overview(_: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    count = lambda m, *w: db.scalar(select(func.count()).select_from(m).where(*w)) or 0  # noqa: E731
    by_status = dict(db.execute(select(ModelDefinition.status, func.count())
                                .group_by(ModelDefinition.status)).all())
    s = get_settings()
    return {
        "service": {"version": s.service_version, "environment": s.environment},
        "counts": {
            "model_definitions": count(ModelDefinition),
            "datasets": count(Dataset),
            "dataset_versions": count(DatasetVersion),
            "experiments": count(Experiment),
            "explorations": count(Exploration),
            "model_versions": count(ModelVersion),
            "deployed_models": count(ModelVersion, ModelVersion.status == "deployed"),
        },
        "catalog_status": by_status,
        "safety_notice": SAFETY_NOTICE,
    }
