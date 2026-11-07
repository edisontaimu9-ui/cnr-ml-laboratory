"""NLP laboratory stage 2: train a spaCy entity model on a stored corpus version and compare it with the rule baseline.

Runs one at a time with the other heavy jobs. Needs the same permissions as experiments (`write:experiments` to start,
`read:experiments` to read). The training text is read from the corpus store and never copied into the results; the audit log keeps
counts only. Nothing is registered, deployed or served: stage 3 adds review and approval."""
import json
import logging
import zipfile
import io

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, nlp_training, storage
from ..db import SessionLocal, get_db
from ..jobs import another_job_running
from ..models import NlpCorpus, NlpCorpusVersion, NlpTrainingRun, utcnow
from ..security import Principal, require

log = logging.getLogger("cnr-ml")
router = APIRouter(prefix="/v1/nlp/training", tags=["nlp"])


class TrainingCreate(BaseModel):
    corpus_id: int
    version: int | None = Field(None, ge=1, description="Corpus version; the latest if left out.")
    n_iter: int = Field(30, ge=5, le=100, description="Passes over the training documents. Fixed up front; nothing is tuned on the test set.")
    dropout: float = Field(0.2, ge=0.0, le=0.5)
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    seed: int = Field(1, ge=0, le=2**31 - 1)
    trial: bool = Field(False, description="Allow a tiny corpus so the pipeline can be tried. The run is marked trial and its results are not for use.")


def _summary(r: NlpTrainingRun) -> dict:
    res = r.results_json or {}
    if r.status == "failed":
        return {"error": res.get("error")}
    if r.status != "completed":
        return {}
    cmp_ = res.get("comparison", {})
    return {"macro_f1_model": (res.get("model") or {}).get("macro_f1"), "macro_f1_rules": (res.get("rules") or {}).get("macro_f1"),
            "overall_verdict": (cmp_.get("overall") or {}).get("verdict"), "test_documents": (res.get("split") or {}).get("test_documents"),
            "warnings": len(res.get("warnings", [])), "trial": bool(res.get("trial"))}


def _out(db: Session, r: NlpTrainingRun, detail: bool = False) -> dict:
    v = db.get(NlpCorpusVersion, r.corpus_version_id)
    c = db.get(NlpCorpus, v.corpus_id) if v else None
    base = {"id": r.id, "corpus_id": c.id if c else None, "corpus_name": c.name if c else None, "corpus_version": v.version if v else None,
            "status": r.status, "trial": bool(r.trial), "summary": _summary(r), "created_by": r.created_by,
            "created_at": r.created_at.isoformat() + "Z", "finished_at": r.finished_at.isoformat() + "Z" if r.finished_at else None}
    if detail:
        base.update(params=r.params_json or {}, results=r.results_json or {})
    return base


@router.get("", dependencies=[Depends(require("read:experiments"))])
def list_runs(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return [_out(db, r) for r in db.scalars(select(NlpTrainingRun).order_by(NlpTrainingRun.id.desc()).limit(limit))]


@router.get("/limits", dependencies=[Depends(require("read:experiments"))])
def limits():
    return {"engine": "spacy-ner-xx", "baseline": "nlp-rules-1",
            "minimums": {"examples_per_label": nlp_training.MIN_EXAMPLES_PER_LABEL, "documents": nlp_training.MIN_DOCS,
                         "trial_examples_per_label": nlp_training.TRIAL_MIN_EXAMPLES_PER_LABEL, "trial_documents": nlp_training.TRIAL_MIN_DOCS},
            "verdict": {"min_test_entities": nlp_training.MIN_SUPPORT, "min_f1_difference": nlp_training.MARGIN},
            "notice": nlp_training.NOTICE}


MAX_RESULTS_BYTES, MAX_MODEL_BYTES = 2 * 1024 * 1024, 300 * 1024 * 1024
EXTERNAL_ENGINE = "spacy-ner-external-gpu"
_RESULT_KEYS = ("engine", "base_model", "spacy", "baseline_engine", "trial", "config", "labels", "split", "loss_by_epoch", "model", "rules",
                "skipped_misaligned", "warnings", "notice", "corpus")


def _version(db: Session, corpus_id: int, version: int | None) -> NlpCorpusVersion:
    q = select(NlpCorpusVersion).where(NlpCorpusVersion.corpus_id == corpus_id)
    q = q.where(NlpCorpusVersion.version == version) if version else q.order_by(NlpCorpusVersion.version.desc())
    v = db.scalars(q.limit(1)).first()
    if v is None or not v.storage_uri:
        raise HTTPException(404, "Unknown corpus version.")
    return v


@router.get("/export/{corpus_id}")
def export_corpus(corpus_id: int, request: Request, version: int | None = Query(None, ge=1),
                  p: Principal = Depends(require("write:experiments")), db: Session = Depends(get_db)):
    """The stored, de-identified corpus version exactly as kept (gzip JSON Lines), for training on an external GPU."""
    v = _version(db, corpus_id, version)
    audit.record(db, "nlp.corpus_exported", principal=p, request=request, resource_type="nlp_corpus", resource_id=corpus_id, detail={"version": v.version})
    return Response(storage.load_bytes(v.storage_uri), media_type="application/gzip",
                    headers={"Content-Disposition": f'attachment; filename="corpus{corpus_id}-v{v.version}.jsonl.gz"', "X-Corpus-SHA256": v.sha256})


def _safe_zip(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            if not names or any(n.startswith(("/", "\\")) or ".." in n.split("/") for n in names):
                raise ValueError
            if z.testzip() is not None:
                raise ValueError
    except (zipfile.BadZipFile, ValueError):
        raise HTTPException(422, "The model file is not a valid spaCy model zip.")


@router.post("/import", status_code=201)
async def import_run(request: Request, corpus_id: int = Form(...), version: int | None = Form(None),
                     results: UploadFile = File(...), model: UploadFile | None = File(None),
                     p: Principal = Depends(require("write:experiments")), db: Session = Depends(get_db)):
    """Records a run trained outside this server (external GPU). The results must name this exact corpus version by SHA-256. The verdicts
    are recomputed here, never taken from the file. The run is marked imported; it still needs review before a model can be approved."""
    v = _version(db, corpus_id, version)
    raw = await results.read(MAX_RESULTS_BYTES + 1)
    if len(raw) > MAX_RESULTS_BYTES:
        raise HTTPException(413, "results.json is too large.")
    try:
        res = json.loads(raw)
        assert isinstance(res, dict) and res.get("engine") == EXTERNAL_ENGINE
        assert (res.get("corpus") or {}).get("sha256") == v.sha256
        for k in ("labels", "split", "model", "rules", "config"):
            assert k in res
    except Exception:
        raise HTTPException(422, "results.json is not from train_ner_gpu.py for this corpus version (engine or corpus SHA-256 does not match).")
    clean = {k: res[k] for k in _RESULT_KEYS if k in res}
    try:
        clean["comparison"] = nlp_training.compare(clean["model"], clean["rules"])
    except Exception:
        raise HTTPException(422, "The model and rules results are incomplete.")
    data = None
    if model is not None and model.filename:
        data = await model.read(MAX_MODEL_BYTES + 1)
        if len(data) > MAX_MODEL_BYTES:
            raise HTTPException(413, "The model file is larger than " + str(MAX_MODEL_BYTES // 2**20) + " MB.")
        _safe_zip(data)
    trial = bool(clean.get("trial"))
    clean["imported"] = {"external_gpu": True, "by": p.name, "model_file_included": data is not None}
    clean.setdefault("warnings", []).append("Trained outside this server and imported. Scores were computed by the uploaded script, not here; only the verdicts were recomputed.")
    params = {**(clean.get("config") or {}), "corpus_id": corpus_id, "corpus_version": v.version, "trial": trial, "external": True}
    r = NlpTrainingRun(corpus_version_id=v.id, params_json=params, trial=trial, status="completed", results_json=clean, created_by=p.name, finished_at=utcnow())
    db.add(r)
    db.flush()
    if data is not None:
        r.artifact_uri = storage.save_bytes(f"nlp_training/run{r.id}.zip", data)
    db.commit()
    audit.record(db, "nlp.training_imported", principal=p, request=request, resource_type="nlp_training", resource_id=r.id,
                 detail={"corpus": corpus_id, "version": v.version, "model_file": data is not None, "trial": trial})
    return _out(db, r, detail=True)


@router.get("/{run_id}", dependencies=[Depends(require("read:experiments"))])
def get_run(run_id: int, db: Session = Depends(get_db)):
    r = db.get(NlpTrainingRun, run_id)
    if r is None:
        raise HTTPException(404, "Unknown training run.")
    return _out(db, r, detail=True)


@router.post("", status_code=201)
def start_run(body: TrainingCreate, request: Request, background: BackgroundTasks,
              p: Principal = Depends(require("write:experiments")), db: Session = Depends(get_db)):
    q = select(NlpCorpusVersion).where(NlpCorpusVersion.corpus_id == body.corpus_id)
    q = q.where(NlpCorpusVersion.version == body.version) if body.version else q.order_by(NlpCorpusVersion.version.desc())
    v = db.scalars(q.limit(1)).first()
    if v is None or not v.storage_uri:
        raise HTTPException(404, "Unknown corpus version.")
    # refuse too-small corpora now, so the person hears it in the response instead of finding a failed run later
    try:
        nlp_training.check_enough(nlp_training.read_corpus(storage.load_bytes(v.storage_uri)), body.trial)
    except nlp_training.TrainingError as e:
        audit.record(db, "nlp.training_refused", principal=p, request=request, resource_type="nlp_corpus", resource_id=v.corpus_id, detail={"reason": str(e)[:200]})
        raise HTTPException(422, str(e))
    if another_job_running(db):
        raise HTTPException(409, "Another experiment, exploration or training run is still running. Wait for it to finish.")
    params = body.model_dump()
    params["corpus_version"] = v.version
    r = NlpTrainingRun(corpus_version_id=v.id, params_json=params, trial=body.trial, status="queued", created_by=p.name)
    db.add(r)
    db.commit()
    audit.record(db, "nlp.training_created", principal=p, request=request, resource_type="nlp_training", resource_id=r.id,
                 detail={"corpus": v.corpus_id, "version": v.version, "trial": body.trial, "n_iter": body.n_iter})
    background.add_task(run_training, r.id)
    return _out(db, r, detail=True)


def run_training(run_id: int) -> None:
    """Runs in a worker thread after the response is sent. Uses its own DB session."""
    with SessionLocal() as db:
        r = db.get(NlpTrainingRun, run_id)
        if r is None:
            return
        r.status = "running"
        db.commit()
        try:
            v = db.get(NlpCorpusVersion, r.corpus_version_id)
            docs = nlp_training.read_corpus(storage.load_bytes(v.storage_uri))
            pr = r.params_json
            results, model = nlp_training.run(docs, n_iter=pr["n_iter"], dropout=pr["dropout"], test_size=pr["test_size"], seed=pr["seed"], trial=pr["trial"])
            results["corpus"] = {"corpus_id": v.corpus_id, "version": v.version, "sha256": v.sha256}
            r.artifact_uri = storage.save_bytes(f"nlp_training/run{r.id}.zip", model)
            r.results_json, r.status = results, "completed"
        except nlp_training.TrainingError as err:
            r.results_json, r.status = {"error": str(err)}, "failed"
        except Exception:
            log.exception("NLP training run %s failed", run_id)
            r.results_json, r.status = {"error": "Unexpected error while training. See the service logs."}, "failed"
        r.finished_at = utcnow()
        db.commit()
        audit.record(db, "nlp.training_" + r.status, resource_type="nlp_training", resource_id=r.id, detail={"summary": _summary(r)})
