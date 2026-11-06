"""Food-name matcher: registering an evaluated matcher as a candidate and serving the deployed one.

The matcher goes through the same review gate as every model (candidate, in review, approved, deployed), with one
difference: a clinical reviewer or the food registry's data owner can approve it. It is suggest-only. A response is a
ranked list of candidate food codes for a person to confirm; nothing is ever assigned automatically, and the similarity
score is not a probability. Query text is never stored or logged (the audit log keeps counts only) and never echoed back."""
import hashlib
import io
import threading
import uuid

import joblib
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import audit, matching, serving, storage
from ..config import get_settings
from ..db import get_db
from ..jobs import another_job_running
from ..models import ModelDefinition, ModelVersion, utcnow
from ..ratelimit import SlidingWindow
from ..schemas import MatchingCandidate, MatchSuggest
from ..security import Principal, require
from .matching import MAX_QUERY_ROWS, MAX_REGISTRY_ROWS, _version

router = APIRouter(prefix="/v1/matching", tags=["matching"])
SLUG = "food-name-matching"
SUGGEST_ONLY = " Suggest-only: candidates are shown for a person to confirm; nothing is assigned automatically."
SCORE_IS = "cosine similarity of character n-grams, not a probability"
_limiter = SlidingWindow(get_settings().rate_limit_predict_per_min, 60)
_index: dict[str, tuple[int, str, matching.Index]] = {}
_lock = threading.Lock()


def _definition(db: Session) -> ModelDefinition:
    d = db.scalar(select(ModelDefinition).where(ModelDefinition.slug == SLUG))
    if d is None:
        raise HTTPException(404, "Unknown model definition.")
    return d


@router.post("/candidates", status_code=201)
def register_candidate(body: MatchingCandidate, request: Request, p: Principal = Depends(require("write:models")),
                       db: Session = Depends(get_db)):
    d = _definition(db)
    vq = _version(db, body.queries_dataset_id, body.queries_version, "labelled queries",
                  [body.query_column, body.food_code_column, body.group_column])
    vr = _version(db, body.registry_dataset_id, body.registry_version, "registry",
                  [body.registry_code_column, body.registry_label_column, body.registry_aliases_column])
    if vq.row_count > MAX_QUERY_ROWS or vr.row_count > MAX_REGISTRY_ROWS:
        raise HTTPException(422, f"Too large for an in-request evaluation (limits: {MAX_QUERY_ROWS} labelled queries, "
                                 f"{MAX_REGISTRY_ROWS} registry foods). Use a subset.")
    if another_job_running(db):
        raise HTTPException(409, "Another experiment or exploration is still running. Wait for it to finish.")
    cfg = matching.Config(query_col=body.query_column, food_col=body.food_code_column,
                          registry_id_col=body.registry_code_column, registry_name_col=body.registry_label_column,
                          registry_alias_col=body.registry_aliases_column, group_col=body.group_column,
                          test_size=body.test_size, seed=body.seed, top_k=body.top_k, target_accuracy=body.target_accuracy)
    try:
        res, bundle = matching.run(storage.load(vq.storage_uri), storage.load(vr.storage_uri), cfg, bundle=True)
    except matching.MatchingError as err:
        raise HTTPException(422, str(err))
    buf = io.BytesIO()
    joblib.dump(bundle, buf)                         # plain lists and numbers only
    data = buf.getvalue()
    last = db.scalar(select(ModelVersion.version).where(ModelVersion.model_definition_id == d.id)
                     .order_by(ModelVersion.version.desc()).limit(1)) or 0
    v = ModelVersion(model_definition_id=d.id, version=last + 1, experiment_id=None, dataset_version_id=vq.id,
                     status="candidate", metrics_json={**res, "task": "matching"},
                     intended_use=body.intended_use.strip() + SUGGEST_ONLY, created_by=p.name, created_by_key_id=p.key_id,
                     artifact_sha256=hashlib.sha256(data).hexdigest(), artifact_bytes=len(data), history_json=[])
    try:
        v.artifact_uri = storage.save_bytes(f"models/{SLUG}/v{v.version}.joblib", data)      # immutable copy
        v.history_json = [{"ts": utcnow().isoformat() + "Z", "action": "registered", "by": p.name,
                           "note": f"evaluated on queries dataset version {vq.id}, registry dataset version {vr.id}"}]
        db.add(v)
        db.flush()
        if d.status == "planned":
            d.status = "in_development"
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Another version was registered at the same moment. Try again.")
    audit.record(db, "model.registered", principal=p, request=request, resource_type="model_version", resource_id=v.id,
                 detail={"model": SLUG, "version": v.version, "learned_top1": res["learned"]["top1"],
                         "verdict": res["learning_effect"]["verdict"]})
    return {"id": v.id, "version": v.version, "status": v.status, "model_slug": SLUG, "results": res}


def _live(db: Session) -> tuple[ModelDefinition, ModelVersion]:
    try:
        d, v = serving.deployed(db, SLUG)
    except serving.ServingError as e:
        raise HTTPException(e.status, e.message)
    if v is None:
        raise HTTPException(409, "No approved food-name matcher is deployed, so no suggestions can be made.")
    return d, v


def _block(d: ModelDefinition, v: ModelVersion) -> dict:
    return {"slug": d.slug, "name": d.name, "version": v.version, "version_id": v.id, "deployed_at": v.deployed_at,
            "approved_by": v.approved_by, "clinical_validation_ref": v.clinical_validation_ref,
            "intended_use": v.intended_use, "evaluation": serving.headline(v.metrics_json or {})}


@router.get("/deployed", dependencies=[Depends(require("predict:models"))])
def deployed(db: Session = Depends(get_db)):
    d, v = serving.deployed(db, SLUG)
    return {"deployed": v is not None, "suggestion_only": True, "score_is": SCORE_IS,
            "model": _block(d, v) if v else None}


@router.post("/suggest")
def suggest(body: MatchSuggest, request: Request, p: Principal = Depends(require("predict:models")),
            db: Session = Depends(get_db)):
    if not _limiter.hit(str(p.key_id)):
        raise HTTPException(429, "Too many requests. Wait a moment and try again.",
                            headers={"Retry-After": str(_limiter.retry_after(str(p.key_id)))})
    rid = uuid.uuid4().hex[:12]
    try:
        d, v = _live(db)
    except HTTPException as e:
        if e.status_code == 409:
            audit.record(db, "matching.refused", principal=p, request=request, resource_type="model", resource_id=SLUG,
                         detail={"reason": "no deployed version", "request_id": rid})
        raise
    try:
        bundle = serving.load_bundle(v, SLUG)          # hash-checked against the registered SHA-256 before loading
        if bundle.get("kind") != "food_matcher":
            raise serving.ServingError(503, "The deployed file is not a food-name matcher. Suggestions are paused.")
    except serving.ServingError as e:
        if e.status == 503:
            audit.record(db, "matching.refused", principal=p, request=request, resource_type="model_version",
                         resource_id=v.id, detail={"reason": "artifact check failed", "request_id": rid})
        raise HTTPException(e.status, e.message)
    with _lock:
        hit = _index.get(SLUG)
        if hit and hit[0] == v.id and hit[1] == v.artifact_sha256:
            idx = hit[2]
        else:
            idx = matching.Index.from_bundle(bundle)
            _index[SLUG] = (v.id, v.artifact_sha256, idx)
    thr = bundle.get("threshold")
    results = matching.suggest(idx, body.queries, body.top_k, thr)
    audit.record(db, "matching.suggested", principal=p, request=request, resource_type="model_version", resource_id=v.id,
                 detail={"model": SLUG, "version": v.version, "queries": len(results), "request_id": rid,
                         "below_review_threshold": sum(1 for r in results if r["below_review_threshold"])})
    return {"request_id": rid, "generated_at": utcnow().isoformat() + "Z", "suggestion_only": True,
            "score_is": SCORE_IS, "review_threshold": thr,
            "note": "Suggestions for a person to confirm. Nothing is assigned. Results are in the same order as the queries; query text is not stored or returned.",
            "model": _block(d, v), "results": results}
