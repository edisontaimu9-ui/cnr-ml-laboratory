"""Status and manual trigger for the automatic CNR surveillance sync (see app/surveillance_sync.py). Admin only."""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .. import audit, surveillance_sync, training_sync
from ..config import get_settings
from ..db import get_db
from ..security import Principal, require

router = APIRouter(prefix="/v1/sync", tags=["sync"])


@router.get("/surveillance", dependencies=[Depends(require("read:integration"))])
def surveillance_status():
    return surveillance_sync.status(get_settings())


@router.post("/surveillance/run", status_code=202)
def surveillance_run(request: Request, background: BackgroundTasks,
                     p: Principal = Depends(require("manage:sync")), db: Session = Depends(get_db)):
    """Starts a sync now (the same work the schedule does). Poll GET /v1/sync/surveillance for the outcome."""
    s = get_settings()
    bad = surveillance_sync.config_problem(s)
    if bad:
        raise HTTPException(409, bad)
    if surveillance_sync._lock.locked():
        raise HTTPException(409, "A sync is already running.")
    audit.record(db, "surveillance_sync.started", principal=p, request=request, detail={"trigger": "manual"})
    background.add_task(surveillance_sync.run_once, s, "manual", p.name)
    return {"status": "started"}


@router.get("/training", dependencies=[Depends(require("read:integration"))])
def training_status(db: Session = Depends(get_db)):
    """Every catalogue model with what CNR can feed it (or why not), the dataset it fills and the last run."""
    return training_sync.status(get_settings(), db)


@router.post("/training/run", status_code=202)
def training_run(request: Request, background: BackgroundTasks,
                 p: Principal = Depends(require("manage:sync")), db: Session = Depends(get_db)):
    """Starts a training-data sync now (the same work the schedule does). Poll GET /v1/sync/training for the outcome."""
    s = get_settings()
    bad = training_sync.config_problem(s)
    if bad:
        raise HTTPException(409, bad)
    if training_sync._lock.locked():
        raise HTTPException(409, "A training-data sync is already running.")
    audit.record(db, "training_sync.started", principal=p, request=request, detail={"trigger": "manual"})
    background.add_task(training_sync.run_once, s, "manual", p.name)
    return {"status": "started"}
