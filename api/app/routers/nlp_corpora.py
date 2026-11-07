"""Annotated text corpora for NLP training (stage 1: validate and store; training comes later).

Accepts JSONL, JSON, CSV, BIO/IOB2, CoNLL, spaCy DocBin and BRAT standoff (see nlp_formats). Two steps, like datasets: `inspect` validates a JSONL file and stores nothing; creating a corpus or a version validates again and
stores only the cleaned, identifier-free documents. A file with identifiers or invalid lines is refused whole. The audit log keeps
counts only, never text."""
import gzip
import hashlib
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import audit, nlp_corpus, nlp_formats, nlp_prelabel, storage
from ..config import get_settings
from ..db import get_db
from ..models import NlpCorpus, NlpCorpusVersion
from ..ratelimit import SlidingWindow
from ..security import Principal, require
from .datasets import _read_upload

router = APIRouter(prefix="/v1/nlp/corpora", tags=["nlp"])
_limiter = SlidingWindow(get_settings().rate_limit_predict_per_min, 60)

FORMAT = {"formats": nlp_formats.FORMATS,
          "choose": "Set `format` to one of the codes above, or leave it as auto to go by the file extension (.jsonl .json .csv .bio .iob .iob2 .conll .spacy .zip).",
          "canonical": {"text": "Mwana ali ndi kutupa.", "lang": "ny", "entities": [{"start": 0, "end": 5, "label": "POPULATION"}]},
          "lang": sorted(nlp_corpus.LANG_CODES), "labels": sorted(nlp_corpus.TRAIN_LABELS),
          "options": {"lang": "Language for documents that do not carry their own (required for BIO, CoNLL, spaCy and BRAT unless the file says it).",
                      "label_map": 'JSON that renames labels from another dataset, e.g. {"FOOD_ITEM": "FOOD", "PER": ""}. A label mapped to "" is dropped.'},
          "notes": {"bio": "One `token TAG` per line; add `# lang = ny` above a sentence to set its language. IOB2 needs B- to start every entity.",
                    "conll": "The last column is the tag; -DOCSTART- lines are skipped; IOB1 tags are accepted.",
                    "csv": "Either one row per entity (columns text, start, end, label, optional doc_id and lang), or one row per document "
                           "with an entities cell like `Mwana|POPULATION; kutupa|SYMPTOM` (each listed word is matched as a whole word, every time).",
                    "json": "Documents may carry entities as objects or as [start, end, label] lists, and spaCy-style [text, {entities: [...]}] pairs work too.",
                    "brat": "ZIP of name.txt and name.ann pairs. Only entity (T) lines are used. A folder named en, ny, tum, yao or mixed sets the language."},
          "rules": ["Offsets are character positions, end exclusive.", "Entities may not overlap.",
                    "Files with identifiers (names with titles, phone numbers, emails, record or ID numbers) are refused whole.",
                    "Whatever the input format, only text, lang and entities are kept, stored as one canonical JSON Lines file."]}


def _version_out(v: NlpCorpusVersion) -> dict:
    return {"version": v.version, "documents": v.doc_count, "entities": v.entity_count, "sha256": v.sha256, "approval_ref": v.approval_ref,
            "created_by": v.created_by, "created_at": v.created_at.isoformat() + "Z", "report": v.report_json}


def _check_ref(ref: str) -> str:
    ref = ref.strip()
    if len(ref) < 5 or len(ref) > 300:
        raise HTTPException(422, "Give the data-sharing or ethics approval reference (5-300 characters).")
    return ref


def _opts(file: UploadFile, format: str, lang: str | None, label_map: str | None) -> dict:
    lang = (lang or "").strip().lower() or None
    if lang is not None and lang not in nlp_corpus.LANG_CODES:
        raise HTTPException(422, "lang must be one of " + ", ".join(sorted(nlp_corpus.LANG_CODES)))
    lmap = None
    if label_map and label_map.strip():
        try:
            lmap = json.loads(label_map)
        except json.JSONDecodeError:
            lmap = None
        if not isinstance(lmap, dict) or not all(isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in lmap.items()):
            raise HTTPException(422, 'label_map must be a JSON object of label names, for example {"FOOD_ITEM": "FOOD"}.')
    return {"filename": file.filename or "", "fmt": format, "lang": lang, "label_map": lmap}


def _validate(raw: bytes, opts: dict, db: Session, request: Request, p: Principal, action: str) -> dict:
    try:
        res = nlp_corpus.parse(raw, **opts)
    except nlp_corpus.CorpusError as e:
        audit.record(db, "nlp.corpus_rejected", principal=p, request=request, detail={"reason": str(e)[:200]})
        raise HTTPException(422, str(e))
    audit.record(db, action, principal=p, request=request, resource_type="nlp_corpus",
                 detail={"valid": res["valid"], "format": res["report"]["source_format"], "documents": res["report"]["documents"], "errors": res["errors_total"],
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


@router.get("/template", dependencies=[Depends(require("read:datasets"))])
def annotation_template():
    """A blank annotation sheet with one example row each for English, Chichewa, Tumbuka and Yao. Delete the example rows before uploading."""
    return Response(nlp_prelabel.template_csv(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="cnr-nlp-annotation-template.csv"'})


@router.post("/prelabel")
async def prelabel(request: Request, file: UploadFile = File(...), lang: str | None = Form(None),
                   p: Principal = Depends(require("run:nlp")), db: Session = Depends(get_db)):
    """Upload plain sentences (.txt, one per line, or a .csv with a text column) or a document (.docx .pptx .odt .pdf .html .md) and get back an annotation CSV holding the engine's
    guesses for speakers to correct. Nothing is stored. Identifiers are replaced by placeholders such as [PHONE]. lang may be a code or a pair such as en+ny."""
    if not _limiter.hit(str(p.key_id)):
        raise HTTPException(429, "Too many requests. Wait a moment.", headers={"Retry-After": str(_limiter.retry_after(str(p.key_id)))})
    lang = (lang or "").strip().lower() or None
    try:
        res = nlp_prelabel.prelabel(await _read_upload(file), file.filename or "", lang)
    except (nlp_prelabel.PrelabelError, nlp_formats.FormatError) as e:
        audit.record(db, "nlp.prelabel_rejected", principal=p, request=request, detail={"reason": str(e)[:200]})
        raise HTTPException(422, str(e))
    audit.record(db, "nlp.prelabel", principal=p, request=request, resource_type="nlp_corpus", detail=res["report"])
    return Response(res["csv"], media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="cnr-nlp-prelabel.csv"',
                             "X-Prelabel-Documents": str(res["report"]["documents"]),
                             "X-Prelabel-Guessed-Entities": str(res["report"]["guessed_entities"]),
                             "X-Prelabel-Identifiers-Removed": str(res["report"]["identifier_lines_removed"])})


@router.post("/inspect")
async def inspect(request: Request, file: UploadFile = File(...), format: str = Form("auto"), lang: str | None = Form(None),
                  label_map: str | None = Form(None), p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    """Validates a file in any supported format and returns the report. Nothing is stored. Problems are listed by location, never by text."""
    res = _validate(await _read_upload(file), _opts(file, format, lang, label_map), db, request, p, "nlp.corpus_inspected")
    return {k: res[k] for k in ("valid", "errors", "errors_total", "identifier_lines", "identifier_lines_total", "identifier_types", "report")}


@router.post("", status_code=201)
async def create_corpus(request: Request, file: UploadFile = File(...), name: str = Form(..., min_length=2, max_length=160),
                        description: str = Form("", max_length=2000), approval_ref: str = Form(...), format: str = Form("auto"),
                        lang: str | None = Form(None), label_map: str | None = Form(None), p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    ref = _check_ref(approval_ref)
    raw = await _read_upload(file)
    res = _validate(raw, _opts(file, format, lang, label_map), db, request, p, "nlp.corpus_checked")
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
                      format: str = Form("auto"), lang: str | None = Form(None), label_map: str | None = Form(None), p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    corpus = db.get(NlpCorpus, corpus_id)
    if corpus is None:
        raise HTTPException(404, "Unknown corpus.")
    ref = _check_ref(approval_ref)
    raw = await _read_upload(file)
    res = _validate(raw, _opts(file, format, lang, label_map), db, request, p, "nlp.corpus_checked")
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
