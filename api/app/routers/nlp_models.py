"""NLP model registry and review gate (stage 3).

Lifecycle:  candidate -> in_review -> approved | rejected;  approved -> deployed -> retired.

Only a completed, non-trial training run can be registered. Registration copies the model file and freezes the evaluation. Approval is
per language (see nlp_registry), needs a validation reference and written notes, and must be done by someone other than the person who
registered or submitted the model. Deploying (an administrator's decision, one version per name) designates the version the serving
endpoints load (`/v1/nlp/serving`); rollback returns to the version it replaced, and retiring withdraws it. The registry itself never
loads a model file; it only copies, hashes and compares bytes. The audit log keeps ids and counts only."""
import hashlib

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import audit, nlp_registry, nlp_serving, storage
from ..config import get_settings
from ..db import get_db
from ..models import NlpCorpusVersion, NlpModelVersion, NlpTrainingRun, utcnow
from ..security import Principal, require

router = APIRouter(prefix="/v1/nlp/models", tags=["nlp"])
NOTICE = ("Approval records that a reviewer read the evaluation and accepted the model for the languages listed. It does not make the model "
          "clinically validated. Only a deployed version is served, and only for its approved languages.")


class RegisterBody(BaseModel):
    training_run_id: int
    name: str = Field(min_length=3, max_length=64, description="Lower-case letters, digits and hyphens, for example nlp-ner-xx.")
    intended_use: str = Field(min_length=20, max_length=2000)


class SubmitBody(BaseModel):
    note: str = Field(default="", max_length=1000)


class ApproveBody(BaseModel):
    approved_languages: list[str] = Field(min_length=1, max_length=8, description="Languages a reviewer signs off (en, ny, tum, yao, mixed).")
    clinical_validation_ref: str = Field(min_length=5, max_length=500)
    review_notes: str = Field(min_length=20, max_length=3000)
    acknowledge_warnings: bool = False


class RejectBody(BaseModel):
    review_notes: str = Field(min_length=10, max_length=3000)


class RetireBody(BaseModel):
    reason: str = Field(min_length=10, max_length=300)


class DeployBody(BaseModel):
    note: str = Field(default="", max_length=500)


def _out(v: NlpModelVersion, detail: bool = False) -> dict:
    m = v.metrics_json or {}
    base = {"id": v.id, "name": v.name, "version": v.version, "status": v.status, "training_run_id": v.training_run_id,
            "headline": nlp_registry.headline(m), "approved_languages": v.approved_languages_json or [], "created_by": v.created_by,
            "created_at": v.created_at.isoformat() + "Z", "submitted_at": v.submitted_at.isoformat() + "Z" if v.submitted_at else None,
            "reviewed_at": v.reviewed_at.isoformat() + "Z" if v.reviewed_at else None, "approved_by": v.approved_by,
            "deployed_at": v.deployed_at.isoformat() + "Z" if v.deployed_at else None, "deployed_by": v.deployed_by}
    if detail:
        base.update(intended_use=v.intended_use, clinical_validation_ref=v.clinical_validation_ref, review_notes=v.review_notes,
                    warnings_acknowledged=bool(v.warnings_acknowledged), submitted_by=v.submitted_by,
                    retired_at=v.retired_at.isoformat() + "Z" if v.retired_at else None, retired_reason=v.retired_reason,
                    artifact_sha256=v.artifact_sha256, artifact_bytes=v.artifact_bytes, language_review=nlp_registry.language_flags(m),
                    metrics_json=m, history=v.history_json or [], notice=NOTICE)
    return base


def _get(db: Session, vid: int) -> NlpModelVersion:
    v = db.get(NlpModelVersion, vid)
    if v is None:
        raise HTTPException(404, "Unknown NLP model version.")
    return v


def _log(v: NlpModelVersion, action: str, by: str, note: str | None = None) -> None:
    v.history_json = [*(v.history_json or []), {"ts": utcnow().isoformat() + "Z", "action": action, "by": by, "note": note or None}]


def _need(v: NlpModelVersion, *allowed: str) -> None:
    if v.status not in allowed:
        raise HTTPException(409, f"This version is {v.status}; this action needs it to be {' or '.join(allowed)}.")


def _separate(v: NlpModelVersion, p: Principal) -> None:
    if get_settings().require_separate_reviewer and p.key_id in {v.created_by_key_id, v.submitted_by_key_id}:
        raise HTTPException(403, "A different person must review this model than the one who registered or submitted it. "
                                 "Ask a clinical reviewer to sign in with their own key.")


def _verify_artifact(v: NlpModelVersion) -> None:
    try:
        ok = bool(v.artifact_uri) and storage.exists(v.artifact_uri) and hashlib.sha256(storage.load_bytes(v.artifact_uri)).hexdigest() == v.artifact_sha256
    except Exception:
        ok = False
    if not ok:
        raise HTTPException(409, "The stored model file is missing or does not match its recorded hash. It will not be approved. Investigate before continuing.")


@router.get("", dependencies=[Depends(require("read:models"))])
def list_versions(name: str | None = Query(None, max_length=64), status: str | None = Query(None, max_length=32),
                  limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    q = select(NlpModelVersion).order_by(NlpModelVersion.id.desc()).limit(limit)
    if name:
        q = q.where(NlpModelVersion.name == name)
    if status:
        q = q.where(NlpModelVersion.status == status)
    return [_out(v) for v in db.scalars(q)]


@router.get("/{vid}", dependencies=[Depends(require("read:models"))])
def get_version(vid: int, db: Session = Depends(get_db)):
    return _out(_get(db, vid), detail=True)


@router.post("", status_code=201)
def register(body: RegisterBody, request: Request, p: Principal = Depends(require("write:models")), db: Session = Depends(get_db)):
    name = body.name.strip().lower()
    if not nlp_registry.NAME.match(name):
        raise HTTPException(422, "name must be 3-63 characters: lower-case letters, digits and hyphens, starting with a letter or digit.")
    run = db.get(NlpTrainingRun, body.training_run_id)
    if run is None:
        raise HTTPException(404, "Unknown training run.")
    if run.status != "completed":
        raise HTTPException(409, "Only a completed training run can be registered.")
    if run.trial or (run.results_json or {}).get("trial"):
        raise HTTPException(409, "A trial run cannot be registered: its data was too small for a trustworthy model. Train again on a corpus that meets the minimums.")
    if not run.artifact_uri or not storage.exists(run.artifact_uri):
        raise HTTPException(409, "This run's model file is not available. Train again.")
    if db.scalar(select(NlpModelVersion.id).where(NlpModelVersion.training_run_id == run.id)):
        raise HTTPException(409, "This training run is already registered.")
    data = storage.load_bytes(run.artifact_uri)
    last = db.scalar(select(NlpModelVersion.version).where(NlpModelVersion.name == name).order_by(NlpModelVersion.version.desc()).limit(1)) or 0
    cv = db.get(NlpCorpusVersion, run.corpus_version_id)
    v = NlpModelVersion(name=name, version=last + 1, training_run_id=run.id, corpus_version_id=run.corpus_version_id, status="candidate",
                        metrics_json=dict(run.results_json or {}), intended_use=body.intended_use.strip(), created_by=p.name,
                        created_by_key_id=p.key_id, artifact_sha256=hashlib.sha256(data).hexdigest(), artifact_bytes=len(data), history_json=[])
    try:
        v.artifact_uri = storage.save_bytes(f"nlp_models/{name}/v{v.version}.zip", data)          # immutable copy
        _log(v, "registered", p.name, f"from training run #{run.id}" + (f", corpus {cv.corpus_id} version {cv.version}" if cv else ""))
        db.add(v)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Another version was registered at the same moment. Try again.")
    audit.record(db, "nlp.model_registered", principal=p, request=request, resource_type="nlp_model_version", resource_id=v.id,
                 detail={"name": name, "version": v.version, "training_run": run.id})
    return _out(v, detail=True)


@router.post("/{vid}/submit")
def submit(vid: int, body: SubmitBody, request: Request, p: Principal = Depends(require("write:models")), db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "candidate")
    v.status, v.submitted_at, v.submitted_by, v.submitted_by_key_id = "in_review", utcnow(), p.name, p.key_id
    _log(v, "submitted for review", p.name, body.note)
    db.commit()
    audit.record(db, "nlp.model_submitted", principal=p, request=request, resource_type="nlp_model_version", resource_id=v.id)
    return _out(v, detail=True)


@router.post("/{vid}/approve")
def approve(vid: int, body: ApproveBody, request: Request, p: Principal = Depends(require("approve:models")), db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "in_review")
    _separate(v, p)
    langs = sorted({lang.strip().lower() for lang in body.approved_languages})
    problems = nlp_registry.approval_problems(v.metrics_json or {}, langs, body.acknowledge_warnings)
    if problems:
        raise HTTPException(422, " ".join(problems))
    _verify_artifact(v)
    flagged = any(nlp_registry.language_flags(v.metrics_json or {}).get(lang, {}).get("flags") for lang in langs) or bool((v.metrics_json or {}).get("skipped_misaligned"))
    v.status, v.approved_by, v.reviewed_by_key_id, v.reviewed_at = "approved", p.name, p.key_id, utcnow()
    v.clinical_validation_ref, v.review_notes = body.clinical_validation_ref.strip(), body.review_notes.strip()
    v.approved_languages_json, v.warnings_acknowledged = langs, bool(flagged and body.acknowledge_warnings)
    _log(v, "approved for " + ", ".join(langs), p.name, body.review_notes)
    db.commit()
    audit.record(db, "nlp.model_approved", principal=p, request=request, resource_type="nlp_model_version", resource_id=v.id,
                 detail={"validation_ref": v.clinical_validation_ref, "languages": langs, "acknowledged": v.warnings_acknowledged})
    return _out(v, detail=True)


@router.post("/{vid}/reject")
def reject(vid: int, body: RejectBody, request: Request, p: Principal = Depends(require("approve:models")), db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "in_review")
    _separate(v, p)
    v.status, v.reviewed_by_key_id, v.reviewed_at, v.review_notes = "rejected", p.key_id, utcnow(), body.review_notes.strip()
    _log(v, "rejected", p.name, body.review_notes)
    db.commit()
    audit.record(db, "nlp.model_rejected", principal=p, request=request, resource_type="nlp_model_version", resource_id=v.id)
    return _out(v, detail=True)


@router.post("/{vid}/retire")
def retire(vid: int, body: RetireBody, request: Request, p: Principal = Depends(require("approve:models")), db: Session = Depends(get_db)):
    v = _get(db, vid)
    _need(v, "approved", "deployed")
    was_live = v.status == "deployed"
    v.status, v.retired_at, v.retired_reason = "retired", utcnow(), "withdrawn: " + body.reason.strip()
    _log(v, "retired", p.name, body.reason)
    db.commit()
    nlp_serving.clear_cache()
    audit.record(db, "nlp.model_retired", principal=p, request=request, resource_type="nlp_model_version", resource_id=v.id, detail={"was_deployed": was_live})
    return _out(v, detail=True)


def _deploy(db: Session, v: NlpModelVersion, p: Principal, request: Request, note: str, action: str) -> int | None:
    """Makes `v` the one served version for its name; the version it replaces is retired as 'superseded' (rollback can bring it back)."""
    _verify_artifact(v)
    replaced = None
    for other in db.scalars(select(NlpModelVersion).where(NlpModelVersion.name == v.name, NlpModelVersion.status == "deployed", NlpModelVersion.id != v.id)):
        other.status, other.retired_at, other.retired_reason = "retired", utcnow(), "superseded"
        _log(other, "superseded by version " + str(v.version), p.name)
        replaced = other.version
    v.status, v.deployed_at, v.deployed_by, v.retired_at, v.retired_reason = "deployed", utcnow(), p.name, None, None
    _log(v, action, p.name, note)
    db.commit()
    nlp_serving.clear_cache()
    audit.record(db, "nlp.model_" + action, principal=p, request=request, resource_type="nlp_model_version", resource_id=v.id,
                 detail={"name": v.name, "version": v.version, "replaced_version": replaced, "languages": v.approved_languages_json})
    return replaced


@router.post("/{vid}/deploy")
def deploy(vid: int, body: DeployBody, request: Request, p: Principal = Depends(require("deploy:models")), db: Session = Depends(get_db)):
    """Designates an approved version as the one to serve. Needs at least one approved language; checks the model file's hash again."""
    v = _get(db, vid)
    if v.status == "retired" and (v.retired_reason or "").startswith("withdrawn"):
        raise HTTPException(409, "A version that was withdrawn cannot be redeployed. Register and review a new one.")
    _need(v, "approved")
    if not (v.approved_languages_json or []):
        raise HTTPException(409, "No language is approved for this version, so there is nothing it may serve.")
    _deploy(db, v, p, request, body.note, "deployed")
    return _out(v, detail=True)


@router.post("/by-name/{name}/rollback")
def rollback(name: str, body: DeployBody, request: Request, p: Principal = Depends(require("deploy:models")), db: Session = Depends(get_db)):
    """Returns to the version that the deployed one replaced."""
    cur = db.scalar(select(NlpModelVersion).where(NlpModelVersion.name == name, NlpModelVersion.status == "deployed"))
    if cur is None:
        raise HTTPException(409, "Nothing is deployed under this name, so there is nothing to roll back.")
    prev = db.scalars(select(NlpModelVersion).where(NlpModelVersion.name == name, NlpModelVersion.status == "retired", NlpModelVersion.retired_reason == "superseded",
                                                    NlpModelVersion.deployed_at.is_not(None)).order_by(NlpModelVersion.deployed_at.desc()).limit(1)).first()
    if prev is None:
        raise HTTPException(409, "There is no earlier deployed version to go back to. Use Retire to withdraw this one.")
    cur.status, cur.retired_at, cur.retired_reason = "retired", utcnow(), "withdrawn: rolled back"
    _log(cur, "rolled back", p.name, body.note)
    db.flush()                                              # sessions here do not autoflush: the replaced version must not be seen as deployed
    _deploy(db, prev, p, request, body.note or f"rolled back from version {cur.version}", "rollback")
    return _out(prev, detail=True)
