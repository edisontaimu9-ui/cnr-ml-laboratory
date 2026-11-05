"""District hotspot / prevalence-trend analyses of AGGREGATED counts, kept apart from model development.

Like the exploration router, this one deliberately imports nothing from the registry, serving, predict, webhook or
integration code and never writes a model file. An analysis is stored as written results only, so it cannot be registered,
reviewed, deployed, served or announced to CNR. `tests/test_hotspots.py` enforces this."""
import logging
from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, hotspot, storage
from ..db import SessionLocal, get_db
from ..jobs import another_job_running
from ..models import DatasetVersion, HotspotAnalysis, utcnow
from ..schemas import HotspotCreate, HotspotDetail, HotspotOut
from ..security import Principal, require

log = logging.getLogger("cnr-ml")
router = APIRouter(prefix="/v1/hotspots", tags=["hotspots"])


def _summary(x: HotspotAnalysis) -> dict:
    r = x.results_json or {}
    if x.status == "failed":
        return {"error": r.get("error")}
    if x.status != "completed":
        return {}
    d, t, sp = r.get("data", {}), r.get("trend"), r.get("spatial")
    out = {"areas": d.get("areas"), "periods": d.get("periods"), "hotspots": len(r.get("hotspots", [])),
           "warnings": len(r.get("warnings", [])), "smoothing": (r.get("smoothing") or {}).get("method")}
    if t:
        out["trend"] = {"model": t["model"], "direction": t["overall"]["direction"],
                        "rate_ratio_per_unit": t["overall"]["rate_ratio_per_unit"], "unit": t["unit"]}
    if sp:
        out["spatial"] = {"verdict": sp["global"]["verdict"], "morans_i": sp["global"]["i"], "p_value": sp["global"]["p_value"]}
    return out


def _out(x: HotspotAnalysis, detail: bool = False) -> dict:
    base = {"id": x.id, "kind": "hotspot_analysis", "deployable": False, "dataset_version_id": x.dataset_version_id,
            "status": x.status, "summary": _summary(x), "created_by": x.created_by, "created_at": x.created_at,
            "finished_at": x.finished_at}
    if detail:
        base.update(params_json=x.params_json or {}, results_json=x.results_json or {})
    return base


@router.get("", response_model=list[HotspotOut], dependencies=[Depends(require("read:hotspots"))])
def list_hotspots(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return [_out(x) for x in db.scalars(select(HotspotAnalysis).order_by(HotspotAnalysis.id.desc()).limit(limit))]


@router.get("/{hid}", response_model=HotspotDetail, dependencies=[Depends(require("read:hotspots"))])
def get_hotspot(hid: int, db: Session = Depends(get_db)):
    x = db.get(HotspotAnalysis, hid)
    if x is None:
        raise HTTPException(404, "Unknown hotspot analysis.")
    return _out(x, detail=True)


@router.post("", response_model=HotspotDetail, status_code=201)
def create_hotspot(body: HotspotCreate, request: Request, background: BackgroundTasks,
                   p: Principal = Depends(require("write:hotspots")), db: Session = Depends(get_db)):
    q = select(DatasetVersion).where(DatasetVersion.dataset_id == body.dataset_id)
    q = q.where(DatasetVersion.version == body.version) if body.version else q.order_by(DatasetVersion.version.desc())
    v = db.scalars(q.limit(1)).first()
    if v is None or not v.storage_uri:
        raise HTTPException(404, "Unknown dataset version.")
    cols = {c["name"] for c in v.schema_info_cols()}
    for name in (body.area_column, body.cases_column, body.denominator_column, body.period_column,
                 body.latitude_column, body.longitude_column):
        if name and name not in cols:
            raise HTTPException(422, f"Column not in this dataset version: {name[:80]}")
    if another_job_running(db):
        raise HTTPException(409, "Another experiment, exploration or analysis is still running. Wait for it to finish.")
    cfg = hotspot.Config(area_col=body.area_column, cases_col=body.cases_column, denominator_col=body.denominator_column,
                         period_col=body.period_column, lat_col=body.latitude_column, lon_col=body.longitude_column,
                         smoothing=body.smoothing, rr_threshold=body.rr_threshold, exceedance_cutoff=body.exceedance_cutoff,
                         k_neighbours=body.k_neighbours, permutations=body.permutations,
                         min_periods_trend=body.min_periods_trend, report_top=body.report_top, seed=body.seed)
    x = HotspotAnalysis(dataset_version_id=v.id, status="queued", created_by=p.name,
                        params_json={"config": asdict(cfg), "dataset_id": v.dataset_id, "dataset_version": v.version,
                                     "note": body.note.strip()})
    db.add(x)
    db.commit()
    audit.record(db, "hotspot.created", principal=p, request=request, resource_type="hotspot_analysis", resource_id=x.id,
                 detail={"dataset": v.dataset_id, "version": v.version, "smoothing": body.smoothing,
                         "trend": bool(body.period_column), "spatial": bool(body.latitude_column)})
    background.add_task(run_hotspot, x.id)
    return _out(x, detail=True)


def run_hotspot(hid: int) -> None:
    """Runs in a worker thread after the response is sent. Uses its own DB session. Stores written results only."""
    with SessionLocal() as db:
        x = db.get(HotspotAnalysis, hid)
        if x is None:
            return
        x.status = "running"
        db.commit()
        try:
            v = db.get(DatasetVersion, x.dataset_version_id)
            cfg = hotspot.Config(**x.params_json["config"])
            x.results_json, x.status = hotspot.run(storage.load(v.storage_uri), cfg), "completed"
        except hotspot.HotspotError as err:
            x.results_json, x.status = {"error": str(err)}, "failed"
        except Exception:
            log.exception("Hotspot analysis %s failed", hid)
            x.results_json, x.status = {"error": "Unexpected error during the analysis. See the service logs."}, "failed"
        x.finished_at = utcnow()
        db.commit()
        audit.record(db, "hotspot." + x.status, resource_type="hotspot_analysis", resource_id=x.id,
                     detail={"summary": _summary(x)})
