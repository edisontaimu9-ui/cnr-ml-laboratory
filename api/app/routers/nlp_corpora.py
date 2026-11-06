"""Annotated text corpora for NLP training (stage 1: validate and store; training comes later).

Two steps, like datasets: `inspect` validates a JSONL file and stores nothing; creating a corpus or a version validates again and
stores only the cleaned, identifier-free documents. A file with identifiers or invalid lines is refused whole. The audit log keeps
counts only, never text."""
import gzip
import hashlib

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import audit, nlp_corpus, storage
from ..db import get_db
from ..models import NlpCorpus, NlpCorpusVersion
from ..security import Principal, require
from .datasets import _read_upload

router = APIRouter(prefix="/v1/nlp/corpora", tags=["nlp"])

FORMAT = {"format": "JSON Lines, UTF-8, one document per line",
          "example": {"text": "Mwana ali ndi kutupa.", "lang": "ny", "entities": [{"start": 0, "end": 5, "label": "POPULATION"}]},
          "lang": sorted(nlp_corpus.LANG_CODES), "labels": sorted(nlp_corpus.TRAIN_LABELS),
          "rules": ["Offsets are character positions, end exclusive.", "Entities may not overlap.",
                    "Files with identifiers (names with titles, phone numbers, emails, record or ID numbers) are refused whole.",
                    "Only text, lang and entities are kept; other fields are dropped."]}


def _version_out(v: NlpCorpusVersion) -> dict:
    return {"version": v.version, "documents": v.doc_count, "entities": v.entity_count, "sha256": v.sha256, "approval_ref": v.approval_ref,
            "created_by": v.created_by, "created_at": v.created_at.isoformat() + "Z", "report": v.report_json}


def _check_ref(ref: str) -> str:
    ref = ref.strip()
    if len(ref) < 5 or len(ref) > 300:
        raise HTTPException(422, "Give the data-sharing or ethics approval reference (5-300 characters).")
    return ref


def _validate(raw: bytes, db: Session, request: Request, p: Principal, action: str) -> dict:
    try:
        res = nlp_corpus.parse(raw)
    except nlp_corpus.CorpusError as e:
        audit.record(db, "nlp.corpus_rejected", principal=p, request=request, detail={"reason": str(e)[:200]})
        raise HTTPException(422, str(e))
    audit.record(db, action, principal=p, request=request, resource_type="nlp_corpus",
                 detail={"valid": res["valid"], "documents": res["report"]["documents"], "errors": res["errors_total"],
                         "identifier_lines": res["identifier_lines_total"]})
    return res


def _refuse(res: dict):
    raise HTTPException(422, {"message": "The file was not stored because it has problems. Fix them and upload again.",
                              "errors": res["errors"], "errors_total": res["errors_total"],
                              "identifier_lines": res["identifier_lines"], "identifier_lines_total": res["identifier_lines_total"],
                              "identifier_types": res["identifier_types"]})


def _store(db: Session, corpus: NlpCorpus, res: dict, raw: bytes, ref: str, p: Principal) -> NlpCorpusVersion:
    source_hash = hashlib.sha256(raw).hexdigest()
    dup = db.scalar(select(NlpCorpusVersion).where(NlpCorpusVersion.corpus_id == corpus.id, NlpCorpusVersion.source_sha256 == source_hash))
    if dup is not None:
        raise HTTPException(409, f"This exact file was already stored as version {dup.version}.")
    version = (db.scalar(select(func.max(NlpCorpusVersion.version)).where(NlpCorpusVersion.corpus_id == corpus.id)) or 0) + 1
    blob = gzip.compress(nlp_corpus.serialise(res["docs"]), mtime=0)
    uri = storage.save_bytes(f"nlp{corpus.id}/v{version}.jsonl.gz", blob)
    v = NlpCorpusVersion(corpus_id=corpus.id, version=version, sha256=hashlib.sha256(blob).hexdigest(), source_sha256=source_hash,
                         doc_count=res["report"]["documents"], entity_count=res["report"]["entities"], storage_uri=uri,
                         approval_ref=ref, report_json=res["report"], created_by=p.name)
    db.add(v)
    db.commit()
    return v


@router.get("/format", dependencies=[Depends(require("read:datasets"))])
def corpus_format():
    return FORMAT


@router.post("/inspect")
async def inspect(request: Request, file: UploadFile = File(...), p: Principal = Depends(require("write:datasets")),
                  db: Session = Depends(get_db)):
    """Validates a file and returns the report. Nothing is stored. Problems are listed by line number, never by text."""
    res = _validate(await _read_upload(file), db, request, p, "nlp.corpus_inspected")
    return {k: res[k] for k in ("valid", "errors", "errors_total", "identifier_lines", "identifier_lines_total", "identifier_types", "report")}


@router.post("", status_code=201)
async def create_corpus(request: Request, file: UploadFile = File(...), name: str = Form(..., min_length=2, max_length=160),
                        description: str = Form("", max_length=2000), approval_ref: str = Form(...),
                        p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    ref = _check_ref(approval_ref)
    raw = await _read_upload(file)
    res = _validate(raw, db, request, p, "nlp.corpus_checked")
    if not res["valid"]:
        _refuse(res)
    corpus = NlpCorpus(name=name.strip(), description=description, created_by=p.name)
    db.add(corpus)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "A corpus with that name already exists. Add a version to it instead.")
    v = _store(db, corpus, res, raw, ref, p)
    audit.record(db, "nlp.corpus_created", principal=p, request=request, resource_type="nlp_corpus", resource_id=corpus.id,
                 detail={"name": corpus.name, "version": v.version, "documents": v.doc_count})
    return {"id": corpus.id, "name": corpus.name, "description": corpus.description, "version": _version_out(v)}


@router.post("/{corpus_id}/versions", status_code=201)
async def add_version(corpus_id: int, request: Request, file: UploadFile = File(...), approval_ref: str = Form(...),
                      p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    corpus = db.get(NlpCorpus, corpus_id)
    if corpus is None:
        raise HTTPException(404, "Unknown corpus.")
    ref = _check_ref(approval_ref)
    raw = await _read_upload(file)
    res = _validate(raw, db, request, p, "nlp.corpus_checked")
    if not res["valid"]:
        _refuse(res)
    v = _store(db, corpus, res, raw, ref, p)
    audit.record(db, "nlp.corpus_version_added", principal=p, request=request, resource_type="nlp_corpus", resource_id=corpus.id,
                 detail={"version": v.version, "documents": v.doc_count})
    return {"id": corpus.id, "name": corpus.name, "version": _version_out(v)}


@router.get("", dependencies=[Depends(require("read:datasets"))])
def list_corpora(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    out = []
    for c in db.scalars(select(NlpCorpus).order_by(NlpCorpus.id.desc()).limit(limit)):
        n = db.scalar(select(func.count()).select_from(NlpCorpusVersion).where(NlpCorpusVersion.corpus_id == c.id)) or 0
        out.append({"id": c.id, "name": c.name, "description": c.description, "versions": n, "created_by": c.created_by,
                    "created_at": c.created_at.isoformat() + "Z"})
    return out


@router.get("/{corpus_id}", dependencies=[Depends(require("read:datasets"))])
def get_corpus(corpus_id: int, db: Session = Depends(get_db)):
    c = db.get(NlpCorpus, corpus_id)
    if c is None:
        raise HTTPException(404, "Unknown corpus.")
    versions = db.scalars(select(NlpCorpusVersion).where(NlpCorpusVersion.corpus_id == c.id).order_by(NlpCorpusVersion.version.desc()))
    return {"id": c.id, "name": c.name, "description": c.description, "created_by": c.created_by,
            "versions": [_version_out(v) for v in versions]}
