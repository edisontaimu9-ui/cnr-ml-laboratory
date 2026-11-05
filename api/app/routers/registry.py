"""Model registry and review gate.

Lifecycle:  candidate -> in_review -> approved | rejected
            approved -> deployed -> retired

"deployed" means *designated as the version to serve*; the prediction endpoints only ever load the
deployed version. The lab also tells CNR (signed webhook) when a deployed version changes.
The registry never loads (unpickles) an artifact; it only copies, hashes and compares bytes."""
import hashlib
from sqlalchemy.exc import IntegrityError
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, storage, webhook
from ..config import get_settings
from ..db import get_db
from ..models import Experiment, ModelDefinition, ModelVersion, utcnow
from ..schemas import (ApproveVersion, DeployVersion, ModelDefinitionOut, ModelVersionDetail, ModelVersionOut,
                       RegisterVersion, RejectVersion, RetireVersion, SubmitVersion)
from ..security import Principal, require

router = APIRouter(prefix="/v1", tags=["registry"])


# ---------- catalogue ----------

@router.get("/catalog", response_model=list[ModelDefinitionOut],
            dependencies=[Depends(require("read:catalog"))])
def catalog(db: Session = Depends(get_db)):
    return db.scalars(select(ModelDefinition).order_by(ModelDefinition.id)).all()


@router.get("/catalog/{slug}", response_model=ModelDefinitionOut,
            dependencies=[Depends(require("read:catalog"))])
def catalog_item(slug: str, db: Session = Depends(get_db)):
    row = db.scalar(select(ModelDefinition).where(ModelDefinition.slug == slug))
    if row is None:
        raise HTTPException(404, "Unknown model definition.")
    return row


# ---------- helpers ----------

def _headline(m: dict) -> dict:
    t, b = m.get("test", {}), m.get("baseline", {})
    key = "macro_f1" if m.get("task") == "classification" else "r2"
    return {"task": m.get("task"), "metric": key, "test": t.get(key), "baseline": b.get(key),
            "n_rows": m.get("n_rows"), "warnings": len(m.get("warnings", []))}


def _out(db: Session, v: ModelVersion, detail: bool = False) -> dict:
    d = db.get(ModelDefinition, v.model_definition_id)
    base = {"id": v.id, "model_definition_id": v.model_definition_id, "model_slug": d.slug if d else None,
            "model_name": d.name if d else None, "version": v.version, "status": v.status,
            "experiment_id": v.experiment_id, "dataset_version_id": v.dataset_version_id,
            "headline": _headline(v.metrics_json or {}), "created_by": v.created_by, "created_at": v.created_at,
            "submitted_at": v.submitted_at, "reviewed_at": v.reviewed_at, "deployed_at": v.deployed_at,
            "approved_by": v.approved_by}
    if detail:
        base.update(intended_use=v.intended_use, clinical_validation_ref=v.clinical_validation_ref,
                    review_notes=v.review_notes, warnings_acknowledged=bool(v.warnings_acknowledged),
                    submitted_by=v.submitted_by, deployed_by=v.deployed_by, retired_at=v.retired_at,
                    retired_reason=v.retired_reason, artifact_sha256=v.artifact_sha256,
                    artifact_bytes=v.artifact_bytes, metrics_json=v.metrics_json or {},
                    history=v.history_json or [])
    return base


def _get(db: Session, vid: int) -> ModelVersion:
    v = db.get(ModelVersion, vid)
    if v is None:
        raise HTTPException(404, "Unknown model version.")
    return v


def _log(v: ModelVersion, action: str, by: str, note: str | None = None) -> None:
    # Reassign (not append) so SQLAlchemy sees the JSON column change.
    v.history_json = [*(v.history_json or []),
                      {"ts": utcnow().isoformat() + "Z", "action": action, "by": by, "note": note or None}]


def _need(v: ModelVersion, *allowed: str) -> None:
    if v.status not in allowed:
        raise HTTPException(409, f"This version is {v.status}; this action needs it to be {' or '.join(allowed)}.")


def _sync_catalog(db: Session, definition_id: int) -> None:
    """Catalogue status mirrors the best state any version has reached. Never lowers 'planned'
    unless a version exists; 'deployed' and 'approved' fall back when versions are retired or rejected."""
    d = db.get(ModelDefinition, definition_id)
    if d is None:
        return
    db.flush()  # the session has autoflush off; make pending status changes visible to the query below
    states = {s for (s,) in db.execute(select(ModelVersion.status).where(ModelVersion.model_definition_id == definition_id))}
    d.status = "deployed" if "deployed" in states else "approved" if "approved" in states else "in_development"


def _verify_artifact(v: ModelVersion) -> None:
    try:
        ok = bool(v.artifact_uri) and storage.exists(v.artifact_uri) and \
            hashlib.sha256(storage.load_bytes(v.artifact_uri)).hexdigest() == v.artifact_sha256
    except Exception:
        ok = False
    if not ok:
        raise HTTPException(409, "The stored model file is missing or does not match its recorded hash. "
                                 "It will not be deployed. Investigate before continuing.")


def _separate(v: ModelVersion, p: Principal) -> None:
    if get_settings().require_separate_reviewer and p.key_id in {v.created_by_key_id, v.submitted_by_key_id}:
        raise HTTPException(403, "A different person must review this model than the one who registered or submitted it. "
                                 "Ask a clinical reviewer to sign in with their own key.")


def _deploy(db: Session, v: ModelVersion, p: Principal, request: Request, note: str, action: str) -> None:
    _verify_artifact(v)
    now = utcnow()
    replaced = None
    for cur in db.scalars(select(ModelVersion).where(ModelVersion.model_definition_id == v.model_definition_id,
                                                    ModelVersion.status == "deployed")):
        replaced = cur.version
        cur.status, cur.retired_at = "retired", now
        cur.retired_reason = "rolled back" if action == "rollback" else "superseded"
        _log(cur, "retired", p.name, cur.retired_reason)
    v.status, v.deployed_at, v.deployed_by, v.retired_at, v.retired_reason = "deployed", now, p.name, None, None
    _log(v, action, p.name, note)
    _sync_catalog(db, v.model_definition_id)
    db.commit()
    audit.record(db, "model." + action, principal=p, request=request, resource_type="model_version", resource_id=v.id,
                 detail={"model_definition_id": v.model_definition_id, "version": v.version})
    d = db.get(ModelDefinition, v.model_definition_id)
    webhook.emit("rollback" if action == "rollback" else "deployed", d.slug, v.version, v.id, superseded_version=replaced)


# ---------- read ----------

@router.get("/models/versions", response_model=list[ModelVersionOut], dependencies=[Depends(require("read:models"))])
def model_versions(model: str | None = None, status: str | None = None,
                   limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    q = select(ModelVersion).order_by(ModelVersion.id.desc()).limit(limit)
    if model:
        q = q.join(ModelDefinition, ModelDefinition.id == ModelVersion.model_definition_id).where(ModelDefinition.slug == model)
    if status:
        q = q.where(ModelVersion.status == status)
    return [_out(db, v) for v in db.scalars(q)]


@router.get("/models/versions/{vid}", response_model=ModelVersionDetail, dependencies=[Depends(require("read:models"))])
def model_version(vid: int, db: Session = Depends(get_db)):
    return _out(db, _get(db, vid), detail=True)


# ---------- lifecycle ----------

@router.post("/models/versions", response_model=ModelVersionDetail, status_code=201)
def register(body: RegisterVersion, request: Request, p: Principal = Depends(require("write:models")),
             db: Session = Depends(get_db)):
    e = db.get(Experiment, body.experiment_id)
    if e is None:
        raise HTTPException(404, "Unknown experiment.")
    if e.status != "completed":
        raise HTTPException(409, "Only a completed experiment can be registered.")
    uri = (e.metrics_json or {}).get("artifact_uri")
    if not uri or not storage.exists(uri):
        raise HTTPException(409, "This experiment's model file is not available. Retrain it.")
    if db.scalar(select(ModelVersion.id).where(ModelVersion.experiment_id == e.id)):
        raise HTTPException(409, "This experiment is already registered.")
    data = storage.load_bytes(uri)
    last = db.scalar(select(ModelVersion.version).where(ModelVersion.model_definition_id == e.model_definition_id)
                     .order_by(ModelVersion.version.desc()).limit(1)) or 0
    metrics = {k: val for k, val in (e.metrics_json or {}).items() if k != "artifact_uri"}  # keep paths out of the record
    v = ModelVersion(model_definition_id=e.model_definition_id, version=last + 1, experiment_id=e.id,
                     dataset_version_id=e.dataset_version_id, status="candidate", metrics_json=metrics,
                     intended_use=body.intended_use.strip(), created_by=p.name, created_by_key_id=p.key_id,
                     artifact_sha256=hashlib.sha256(data).hexdigest(), artifact_bytes=len(data), history_json=[])
    d = db.get(ModelDefinition, e.model_definition_id)
    try:
        v.artifact_uri = storage.save_bytes(f"models/{d.slug}/v{v.version}.joblib", data)   # immutable copy
        _log(v, "registered", p.name, f"from experiment #{e.id}")
        db.add(v)
        db.flush()
        _sync_catalog(db, v.model_definition_id)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Another version was registered at the same moment. Try again.")
    audit.record(db, "model.registered", principal=p, request=request, resource_type="model_version", resource_id=v.id,
                 detail={"model": d.slug, "version": v.version, "experiment": e.id})
    return _out(db, v, detail=True)


@router.post("/models/versions/{vid}/submit", response_model=ModelVersionDetail)
def submit(vid: int, body: SubmitVersion, request: Request, p: Principal = Depends(require("write:models")),
           db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "candidate")
    v.status, v.submitted_at, v.submitted_by, v.submitted_by_key_id = "in_review", utcnow(), p.name, p.key_id
    _log(v, "submitted for clinical review", p.name, body.note)
    db.commit()
    audit.record(db, "model.submitted", principal=p, request=request, resource_type="model_version", resource_id=v.id)
    return _out(db, v, detail=True)


@router.post("/models/versions/{vid}/approve", response_model=ModelVersionDetail)
def approve(vid: int, body: ApproveVersion, request: Request, p: Principal = Depends(require("approve:models")),
            db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "in_review")
    _separate(v, p)
    n_warn = len((v.metrics_json or {}).get("warnings", []))
    if n_warn and not body.acknowledge_warnings:
        raise HTTPException(422, f"This model carries {n_warn} evaluation warning(s). Read them and confirm you have "
                                 "considered them (acknowledge_warnings) before approving.")
    v.status, v.approved_by, v.reviewed_by_key_id, v.reviewed_at = "approved", p.name, p.key_id, utcnow()
    v.clinical_validation_ref, v.review_notes = body.clinical_validation_ref.strip(), body.review_notes.strip()
    v.warnings_acknowledged = bool(n_warn and body.acknowledge_warnings)
    _log(v, "approved", p.name, body.review_notes)
    _sync_catalog(db, v.model_definition_id)
    db.commit()
    audit.record(db, "model.approved", principal=p, request=request, resource_type="model_version", resource_id=v.id,
                 detail={"validation_ref": v.clinical_validation_ref, "warnings": n_warn})
    return _out(db, v, detail=True)


@router.post("/models/versions/{vid}/reject", response_model=ModelVersionDetail)
def reject(vid: int, body: RejectVersion, request: Request, p: Principal = Depends(require("approve:models")),
           db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "in_review")
    _separate(v, p)
    v.status, v.reviewed_by_key_id, v.reviewed_at, v.review_notes = "rejected", p.key_id, utcnow(), body.review_notes.strip()
    _log(v, "rejected", p.name, body.review_notes)
    _sync_catalog(db, v.model_definition_id)
    db.commit()
    audit.record(db, "model.rejected", principal=p, request=request, resource_type="model_version", resource_id=v.id)
    return _out(db, v, detail=True)


@router.post("/models/versions/{vid}/deploy", response_model=ModelVersionDetail)
def deploy(vid: int, body: DeployVersion, request: Request, p: Principal = Depends(require("deploy:models")),
           db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "approved", "retired")
    if not v.clinical_validation_ref:
        raise HTTPException(409, "This version has no recorded clinical validation reference.")
    if v.status == "retired" and v.retired_reason != "superseded":
        raise HTTPException(409, "A version that was rolled back or withdrawn cannot be redeployed. Register and review a new one.")
    _deploy(db, v, p, request, body.note, "deployed")
    return _out(db, v, detail=True)


@router.post("/models/versions/{vid}/retire", response_model=ModelVersionDetail)
def retire(vid: int, body: RetireVersion, request: Request, p: Principal = Depends(require("deploy:models")),
           db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "approved", "deployed")
    was_deployed = v.status == "deployed"
    v.status, v.retired_at, v.retired_reason = "retired", utcnow(), "withdrawn: " + body.reason.strip()
    _log(v, "retired", p.name, body.reason)
    _sync_catalog(db, v.model_definition_id)
    db.commit()
    audit.record(db, "model.retired", principal=p, request=request, resource_type="model_version", resource_id=v.id)
    if was_deployed:        # only a version that was being served matters to CNR
        webhook.emit("withdrawn", db.get(ModelDefinition, v.model_definition_id).slug, v.version, v.id, reason="withdrawn")
    return _out(db, v, detail=True)


@router.post("/models/{slug}/rollback", response_model=ModelVersionDetail)
def rollback(slug: str, body: DeployVersion, request: Request, p: Principal = Depends(require("deploy:models")),
             db: Session = Depends(get_db)):
    d = db.scalar(select(ModelDefinition).where(ModelDefinition.slug == slug))
    if d is None:
        raise HTTPException(404, "Unknown model definition.")
    cur = db.scalar(select(ModelVersion).where(ModelVersion.model_definition_id == d.id, ModelVersion.status == "deployed"))
    if cur is None:
        raise HTTPException(409, "Nothing is deployed for this model, so there is nothing to roll back.")
    prev = db.scalars(select(ModelVersion).where(
        ModelVersion.model_definition_id == d.id, ModelVersion.status == "retired",
        ModelVersion.retired_reason == "superseded", ModelVersion.deployed_at.is_not(None))
        .order_by(ModelVersion.deployed_at.desc()).limit(1)).first()
    if prev is None:
        raise HTTPException(409, "There is no earlier deployed version to go back to. Use Retire to withdraw this one.")
    _deploy(db, prev, p, request, body.note or f"rolled back from version {cur.version}", "rollback")
    return _out(db, prev, detail=True)
