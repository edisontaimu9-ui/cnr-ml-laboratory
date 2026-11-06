import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from . import surveillance_sync, training_sync
from .catalog import CATALOG, LABEL_NOTE, LABEL_NOTES
from .config import get_settings
from .db import Base, SessionLocal, add_missing_columns, engine
from .models import ApiKey, Exploration, Experiment, HotspotAnalysis, ModelDefinition, utcnow
from .routers import admin, core, datasets, experiments, explorations, hotspots, integration, matching, predict, registry, sync
from .security import hash_key

log = logging.getLogger("cnr-ml")
settings = get_settings()


def _seed() -> None:
    with SessionLocal() as db:
        existing = {m.slug for m in db.scalars(select(ModelDefinition))}
        for slug, name, task, desc in CATALOG:
            if slug not in existing:  # never overwrite a status changed since
                db.add(ModelDefinition(slug=slug, name=name, task_type=task,
                                       description=desc, label_source=LABEL_NOTES.get(slug, LABEL_NOTE)))
        key = settings.bootstrap_admin_key
        if key and len(key) >= 32:
            row = db.scalar(select(ApiKey).where(ApiKey.prefix == "boot"))
            if row is None:
                db.add(ApiKey(name="bootstrap-admin", role="admin", prefix="boot",
                              key_hash=hash_key(key), created_by="system"))
            elif row.key_hash != hash_key(key):  # rotation via env var
                row.key_hash, row.revoked_at = hash_key(key), None
        elif key:
            log.warning("BOOTSTRAP_ADMIN_KEY is shorter than 32 chars; ignored.")
        db.commit()


def _fail_interrupted() -> None:
    """Training runs in-process; a restart kills them. Mark them failed instead of leaving them 'running'."""
    with SessionLocal() as db:
        for e in db.scalars(select(Experiment).where(Experiment.status.in_(["queued", "running"]))):
            e.status, e.finished_at = "failed", utcnow()
            e.metrics_json = {"error": "Interrupted by a service restart. Start it again."}
        for x in db.scalars(select(Exploration).where(Exploration.status.in_(["queued", "running"]))):
            x.status, x.finished_at = "failed", utcnow()
            x.results_json = {"error": "Interrupted by a service restart. Start it again."}
        for h in db.scalars(select(HotspotAnalysis).where(HotspotAnalysis.status.in_(["queued", "running"]))):
            h.status, h.finished_at = "failed", utcnow()
            h.results_json = {"error": "Interrupted by a service restart. Start it again."}
        db.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    add_missing_columns()
    _seed()
    _fail_interrupted()
    tasks = []
    if surveillance_sync.config_problem(settings) is None:
        tasks.append(asyncio.create_task(surveillance_sync.scheduler(settings)))
    if training_sync.config_problem(settings) is None:
        tasks.append(asyncio.create_task(training_sync.scheduler(settings)))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()


app = FastAPI(
    title="CNR Machine Learning Laboratory API",
    version=settings.service_version,
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
    openapi_url=None if settings.is_production else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_list, allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    try:
        resp = await call_next(request)
    except Exception:
        # Unhandled errors bypass CORSMiddleware, which browsers report as "cannot reach the API".
        # Log the real cause and return a JSON error that the dashboard can display.
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        resp = JSONResponse({"detail": "Server error. Check the service logs."}, status_code=500)
        origin = request.headers.get("origin")
        if origin and origin in settings.cors_list:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "no-store"
    return resp


app.include_router(core.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(explorations.router)
app.include_router(hotspots.router)
app.include_router(matching.router)
app.include_router(registry.router)
app.include_router(predict.router)
app.include_router(integration.router)
app.include_router(sync.router)
app.include_router(admin.router)
