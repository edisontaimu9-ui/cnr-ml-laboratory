"""Explorations: unsupervised analysis (clustering, anomaly detection) kept apart from model development.

This router deliberately imports nothing from the registry, serving, predict, webhook or integration code and never
writes a model file. An exploration is stored as written results only, so it cannot be registered, reviewed, deployed,
served or announced to CNR. `tests/test_explorations.py` enforces this."""
import logging
from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, exploration, storage
from ..db import SessionLocal, get_db
from ..jobs import another_job_running
from ..models import DatasetVersion, Exploration, utcnow
from ..schemas import ExplorationCreate, ExplorationDetail, ExplorationOut
from ..security import Principal, require

log = logging.getLogger("cnr-ml")
router = APIRouter(prefix="/v1/explorations", tags=["explorations"])


def _summary(x: Exploration) -> dict:
    r = x.results_json or {}
    if x.status == "failed":
        return {"error": r.get("error")}
    if x.status != "completed":
        return {}
    c, a = r.get("clustering"), r.get("anomaly")
    out = {"methods": r.get("methods", []), "n_rows": r.get("n_rows"), "warnings": len(r.get("warnings", []))}
    if c:
        out["clustering"] = {"k": c["k"], "structure": c["structure"], "silhouette": c["quality"]["silhouette"]}
    if a:
        out["anomaly"] = {"flagged": a["n_flagged"], "flag_share": a["flag_share"]}
    return out


def _out(x: Exploration, detail: bool = False) -> dict:
    base = {"id": x.id, "kind": "exploration", "deployable": False, "dataset_version_id": x.dataset_version_id,
            "status": x.status, "summary": _summary(x), "created_by": x.created_by, "created_at": x.created_at,
            "finished_at": x.finished_at}
    if detail:
        base.update(params_json=x.params_json or {}, results_json=x.results_json or {})
    return base


@router.get("", response_model=list[ExplorationOut], dependencies=[Depends(require("read:explorations"))])
def list_explorations(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return [_out(x) for x in db.scalars(select(Exploration).order_by(Exploration.id.desc()).limit(limit))]


@router.get("/{xid}", response_model=ExplorationDetail, dependencies=[Depends(require("read:explorations"))])
def get_exploration(xid: int, db: Session = Depends(get_db)):
    x = db.get(Exploration, xid)
    if x is None:
        raise HTTPException(404, "Unknown exploration.")
    return _out(x, detail=True)


@router.post("", response_model=ExplorationDetail, status_code=201)
def create_exploration(body: ExplorationCreate, request: Request, background: BackgroundTasks,
                       p: Principal = Depends(require("write:explorations")), db: Session = Depends(get_db)):
    q = select(DatasetVersion).where(DatasetVersion.dataset_id == body.dataset_id)
    q = q.where(DatasetVersion.version == body.version) if body.version else q.order_by(DatasetVersion.version.desc())
    v = db.scalars(q.limit(1)).first()
    if v is None or not v.storage_uri:
        raise HTTPException(404, "Unknown dataset version.")
    cols = {c["name"] for c in v.schema_info_cols()}
    for name in [*(body.features or []), *body.exclude]:
        if name not in cols:
            raise HTTPException(422, f"Column not in this dataset version: {name[:80]}")
    if another_job_running(db):
        raise HTTPException(409, "Another experiment or exploration is still running. Wait for it to finish.")
    cfg = exploration.Config(features=body.features, exclude=body.exclude, methods=body.methods, k=body.k,
                             k_min=body.k_min, k_max=body.k_max, flag_share=body.flag_share,
                             report_top=body.report_top, seed=body.seed)
    x = Exploration(dataset_version_id=v.id, status="queued", created_by=p.name,
                    params_json={"config": asdict(cfg), "dataset_id": v.dataset_id, "dataset_version": v.version,
                                 "note": body.note.strip()})
    db.add(x)
    db.commit()
    audit.record(db, "exploration.created", principal=p, request=request, resource_type="exploration", resource_id=x.id,
                 detail={"dataset": v.dataset_id, "version": v.version, "methods": body.methods})
    background.add_task(run_exploration, x.id)
    return _out(x, detail=True)


def run_exploration(xid: int) -> None:
    """Runs in a worker thread after the response is sent. Uses its own DB session. Stores written results only."""
    with SessionLocal() as db:
        x = db.get(Exploration, xid)
        if x is None:
            return
        x.status = "running"
        db.commit()
        try:
            v = db.get(DatasetVersion, x.dataset_version_id)
            cfg = exploration.Config(**x.params_json["config"])
            x.results_json, x.status = exploration.run(storage.load(v.storage_uri), cfg), "completed"
        except exploration.ExplorationError as err:
            x.results_json, x.status = {"error": str(err)}, "failed"
        except Exception:
            log.exception("Exploration %s failed", xid)
            x.results_json, x.status = {"error": "Unexpected error during the analysis. See the service logs."}, "failed"
        x.finished_at = utcnow()
        db.commit()
        audit.record(db, "exploration." + x.status, resource_type="exploration", resource_id=x.id,
                     detail={"summary": _summary(x)})
