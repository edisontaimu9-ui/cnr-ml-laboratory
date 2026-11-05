import io
import logging
import os
from dataclasses import asdict

import joblib
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, storage, training
from ..db import SessionLocal, get_db
from ..models import DatasetVersion, Experiment, ModelDefinition, utcnow
from ..schemas import ExperimentCreate, ExperimentDetail, ExperimentOut
from ..security import Principal, require

log = logging.getLogger("cnr-ml")
router = APIRouter(prefix="/v1/experiments", tags=["experiments"])
STALE_MINUTES = 30


def _summary(e: Experiment) -> dict:
    m = e.metrics_json or {}
    if e.status == "failed":
        return {"error": m.get("error")}
    if e.status != "completed":
        return {}
    t, b = m.get("test", {}), m.get("baseline", {})
    key = "macro_f1" if m.get("task") == "classification" else "r2"
    out = {"task": m.get("task"), "metric": key, "test": t.get(key), "baseline": b.get(key),
           "warnings": len(m.get("warnings", [])), "algorithm": m.get("algorithm", "xgboost")}
    c = m.get("comparison")
    if c:
        out["vs_xgboost"] = {"difference": c.get("difference"), "verdict": c.get("verdict")}
    return out


def _out(db: Session, e: Experiment, detail: bool = False) -> dict:
    d = db.get(ModelDefinition, e.model_definition_id)
    base = {"id": e.id, "model_definition_id": e.model_definition_id, "model_slug": d.slug if d else None,
            "dataset_version_id": e.dataset_version_id, "framework": e.framework, "status": e.status,
            "summary": _summary(e), "created_by": e.created_by, "created_at": e.created_at, "finished_at": e.finished_at}
    if detail:
        base.update(model_name=d.name if d else None, params_json=e.params_json or {}, metrics_json=e.metrics_json or {})
    return base


@router.get("", response_model=list[ExperimentOut], dependencies=[Depends(require("read:experiments"))])
def list_experiments(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return [_out(db, e) for e in db.scalars(select(Experiment).order_by(Experiment.id.desc()).limit(limit))]


@router.get("/{exp_id}", response_model=ExperimentDetail, dependencies=[Depends(require("read:experiments"))])
def get_experiment(exp_id: int, db: Session = Depends(get_db)):
    e = db.get(Experiment, exp_id)
    if e is None:
        raise HTTPException(404, "Unknown experiment.")
    return _out(db, e, detail=True)


@router.post("", response_model=ExperimentDetail, status_code=201)
def create_experiment(body: ExperimentCreate, request: Request, background: BackgroundTasks,
                      p: Principal = Depends(require("write:experiments")), db: Session = Depends(get_db)):
    d = db.scalar(select(ModelDefinition).where(ModelDefinition.slug == body.model_slug))
    if d is None:
        raise HTTPException(404, "Unknown model definition.")
    q = select(DatasetVersion).where(DatasetVersion.dataset_id == body.dataset_id)
    q = q.where(DatasetVersion.version == body.version) if body.version else q.order_by(DatasetVersion.version.desc())
    v = db.scalars(q.limit(1)).first()
    if v is None or not v.storage_uri:
        raise HTTPException(404, "Unknown dataset version.")
    cols = {c["name"] for c in (v.schema_info_cols())}
    for name in [body.target, *(body.features or []), *body.exclude]:
        if name not in cols:
            raise HTTPException(422, f"Column not in this dataset version: {name[:80]}")
    cutoff = utcnow().replace(microsecond=0)
    busy = db.scalars(select(Experiment).where(Experiment.status.in_(["queued", "running"]))).all()
    if any((cutoff - b.created_at).total_seconds() < STALE_MINUTES * 60 for b in busy):
        raise HTTPException(409, "Another experiment is still running. Wait for it to finish.")
    cfg = training.Config(target=body.target, task=d.task_type, features=body.features, exclude=body.exclude,
                          test_size=body.test_size, seed=body.seed, n_estimators=body.n_estimators,
                          max_depth=body.max_depth, learning_rate=body.learning_rate,
                          balance_classes=body.balance_classes, algorithm=body.algorithm,
                          hidden_layers=body.hidden_layers, alpha=body.alpha, max_iter=body.max_iter,
                          compare_xgboost=body.compare_xgboost)
    e = Experiment(model_definition_id=d.id, dataset_version_id=v.id, framework=body.algorithm, status="queued",
                   params_json={"config": asdict(cfg), "dataset_id": v.dataset_id, "dataset_version": v.version,
                                "model_slug": d.slug},
                   git_commit=(os.environ.get("RENDER_GIT_COMMIT") or "")[:40] or None, created_by=p.name)
    db.add(e)
    db.commit()
    audit.record(db, "experiment.created", principal=p, request=request, resource_type="experiment", resource_id=e.id,
                 detail={"model": d.slug, "dataset": v.dataset_id, "version": v.version, "target": body.target,
                             "algorithm": body.algorithm})
    background.add_task(run_experiment, e.id)
    return _out(db, e, detail=True)


def run_experiment(exp_id: int) -> None:
    """Runs in a worker thread after the response is sent. Uses its own DB session."""
    with SessionLocal() as db:
        e = db.get(Experiment, exp_id)
        if e is None:
            return
        e.status = "running"
        db.commit()
        try:
            v = db.get(DatasetVersion, e.dataset_version_id)
            cfg = training.Config(**e.params_json["config"])
            result, bundle = training.run(storage.load(v.storage_uri), cfg)
            buf = io.BytesIO()
            joblib.dump(bundle, buf)
            result["artifact_uri"] = storage.save_bytes(f"experiments/exp{e.id}.joblib", buf.getvalue())
            e.metrics_json, e.status = result, "completed"
            d = db.get(ModelDefinition, e.model_definition_id)
            if d and d.status == "planned":
                d.status = "in_development"
        except training.TrainingError as err:
            e.metrics_json, e.status = {"error": str(err)}, "failed"
        except Exception:
            log.exception("Experiment %s failed", exp_id)
            e.metrics_json, e.status = {"error": "Unexpected error while training. See the service logs."}, "failed"
        e.finished_at = utcnow()
        db.commit()
        audit.record(db, "experiment." + e.status, resource_type="experiment", resource_id=e.id,
                     detail={"summary": _summary(e)})
