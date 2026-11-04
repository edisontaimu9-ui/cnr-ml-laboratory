import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from .catalog import CATALOG, LABEL_NOTE
from .config import get_settings
from .db import Base, SessionLocal, engine
from .models import ApiKey, ModelDefinition
from .routers import admin, core, registry
from .security import hash_key

log = logging.getLogger("cnr-ml")
settings = get_settings()


def _seed() -> None:
    with SessionLocal() as db:
        existing = {m.slug for m in db.scalars(select(ModelDefinition))}
        for slug, name, task, desc in CATALOG:
            if slug not in existing:  # never overwrite status changed by later phases
                db.add(ModelDefinition(slug=slug, name=name, task_type=task,
                                       description=desc, label_source=LABEL_NOTE))
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    _seed()
    yield


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
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "no-store"
    return resp


app.include_router(core.router)
app.include_router(registry.router)
app.include_router(admin.router)
