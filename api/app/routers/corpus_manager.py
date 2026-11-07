"""Corpus Manager & Dataset Builder (modular workflow).

corpora -> uploads -> annotation -> quality checks -> review and approval (frozen snapshot) -> merge approved corpora into a versioned
master dataset -> export, or build a task dataset (NER, classification, language id).

Every step is a separate call. Merging never trains a model and nothing here is connected to the training endpoints. Individual corpora
stay as they are: a merge reads frozen snapshots and writes a new master version. The audit log keeps counts, never text."""
import gzip
import hashlib
import re
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import audit, corpus_manager as cm, storage
from ..config import get_settings
from ..db import get_db
from ..models import CmCorpus, CmMaster, CmRecord, CmSnapshot, CmUpload, utcnow
from ..security import Principal, require
from .datasets import _read_upload

router = APIRouter(prefix="/v1/cm", tags=["corpus-manager"])
SLUG = re.compile(r"[^A-Za-z0-9._-]+")


# ------------------------------------------------------------------ helpers
def _iso(d: datetime | None) -> str | None:
    return d.isoformat() + "Z" if d else None


def _corpus(db: Session, corpus_id: int) -> CmCorpus:
    c = db.get(CmCorpus, corpus_id)
    if c is None:
        raise HTTPException(404, "Unknown corpus.")
    return c


def _rec(r: CmRecord) -> dict:
    return {"id": r.id, "text": r.text, "norm_hash": r.norm_hash, "lang": r.lang, "category": r.category, "entities": r.entities_json,
            "parse_note": r.parse_note, "status": r.status, "source": r.source}


def _records(db: Session, corpus_id: int) -> list[dict]:
    return [_rec(r) for r in db.scalars(select(CmRecord).where(CmRecord.corpus_id == corpus_id).order_by(CmRecord.id))]


def _others(db: Session, corpus_id: int) -> dict[str, list[str]]:
    rows = db.execute(select(CmRecord.norm_hash, CmCorpus.name).join(CmCorpus, CmCorpus.id == CmRecord.corpus_id)
                      .where(CmRecord.corpus_id != corpus_id, CmRecord.status != "rejected").distinct())
    out: dict[str, list[str]] = {}
    for h, name in rows:
        out.setdefault(h, []).append(name)
    return out


def _quality(db: Session, c: CmCorpus) -> dict:
    return cm.quality(_records(db, c.id), c.tasks_json or list(cm.TASKS), _others(db, c.id))


def _counts(db: Session, corpus_id: int) -> dict:
    rows = db.execute(select(CmRecord.status, func.count()).where(CmRecord.corpus_id == corpus_id).group_by(CmRecord.status)).all()
    by = {k: n for k, n in rows}
    return {"records": sum(by.values()), "by_status": by}


def _corpus_out(db: Session, c: CmCorpus, detail: bool = False) -> dict:
    snaps = db.scalars(select(CmSnapshot).where(CmSnapshot.corpus_id == c.id).order_by(CmSnapshot.number.desc())).all()
    out = {"id": c.id, "name": c.name, "domain": c.domain, "description": c.description, "source": c.source, "default_lang": c.default_lang,
           "tasks": c.tasks_json, "state": c.state, "review_note": c.review_note, "reviewed_by": c.reviewed_by, "reviewed_at": _iso(c.reviewed_at),
           "last_edited_by": c.last_edited_by, "last_edited_at": _iso(c.last_edited_at), "created_by": c.created_by, "created_at": _iso(c.created_at),
           "latest_snapshot": snaps[0].number if snaps else None, **_counts(db, c.id)}
    if detail:
        ups = db.scalars(select(CmUpload).where(CmUpload.corpus_id == c.id).order_by(CmUpload.id.desc())).all()
        out["uploads"] = [{"id": u.id, "filename": u.filename, "format": u.format, "records": u.record_count, "source_sha256": u.source_sha256,
                           "created_by": u.created_by, "created_at": _iso(u.created_at), "report": u.report_json} for u in ups]
        out["snapshots"] = [{"id": s.id, "number": s.number, "documents": s.doc_count, "sha256": s.sha256, "approved_by": s.approved_by,
                             "approval_note": s.approval_note, "created_at": _iso(s.created_at)} for s in snaps]
    return out


def _touch(c: CmCorpus, p: Principal) -> bool:
    """Marks an edit. Any edit withdraws an earlier approval or review: the corpus is a draft again and must be approved again."""
    reopened = c.state != "draft"
    c.state = "draft"
    c.last_edited_by, c.last_edited_key_id, c.last_edited_at = p.name, p.key_id, utcnow()
    return reopened


def _tasks(v: list[str] | None) -> list[str]:
    tasks = [t for t in cm.TASKS if t in (v or list(cm.TASKS))]
    if not tasks or set(v or []) - set(cm.TASKS):
        raise HTTPException(422, "tasks must be a non-empty list from: " + ", ".join(cm.TASKS))
    return tasks


def _check_domain(domain: str) -> str:
    if domain not in cm.DOMAINS:
        raise HTTPException(422, "domain must be one of " + ", ".join(cm.DOMAINS))
    return domain


def _check_lang(lang: str | None) -> str | None:
    lang = cm.clean_lang(lang)
    if lang is not None and lang not in cm.LANG_CODES:
        raise HTTPException(422, "language must be one of " + ", ".join(sorted(cm.LANG_CODES)))
    return lang


def _blob(uri: str, sha: str) -> bytes:
    """Reads a stored file and checks its hash, so a damaged or swapped file is never used."""
    if not storage.exists(uri):
        raise HTTPException(409, "A stored file is missing. Contact an administrator.")
    blob = storage.load_bytes(uri)
    if hashlib.sha256(blob).hexdigest() != sha:
        raise HTTPException(409, "A stored file failed its integrity check. Contact an administrator.")
    return blob


def _save(rel: str, docs: list[dict]) -> tuple[str, str]:
    blob = gzip.compress(cm.to_jsonl(docs), mtime=0)
    return storage.save_bytes(rel, blob), hashlib.sha256(blob).hexdigest()


def _upload_error(e: Exception, db, request, p):
    audit.record(db, "cm.upload_rejected", principal=p, request=request, detail={"reason": str(e)[:200]})
    raise HTTPException(422, str(e))


# ------------------------------------------------------------------ info
@router.get("/info", dependencies=[Depends(require("read:datasets"))])
def info():
    return {"domains": cm.DOMAINS, "tasks": cm.TASKS, "languages": sorted(cm.LANG_CODES), "entity_labels": sorted(cm.ENTITY_LABELS),
            "categories": sorted(cm.CATEGORIES), "statuses": list(cm.STATUSES), "upload_formats": cm.UPLOAD_FORMATS, "upload_accept": cm.ACCEPT, "export_formats": list(cm.EXPORT_FORMATS),
            "conflict_policies": cm.CONFLICT_POLICIES, "limits": {"records_per_upload": cm.MAX_UPLOAD_RECORDS, "records_per_corpus": cm.MAX_CORPUS_RECORDS,
                                                                 "characters_per_text": cm.MAX_CHARS},
            "rules": ["Text with identifiers (titled names, phone numbers, emails, record or ID numbers) is refused whole.",
                      "Only text, language, category, entities, source and id are kept; other columns are dropped.",
                      "Records are never deleted: an unwanted record is rejected and kept.",
                      "Editing a corpus withdraws its approval. A master is built from frozen snapshots, so later edits never change it.",
                      "Merging builds a dataset only. It never trains a model."]}


# ------------------------------------------------------------------ corpora
class CorpusIn(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    domain: str
    description: str = Field("", max_length=2000)
    source: str | None = Field(None, max_length=300)
    default_lang: str | None = None
    tasks: list[str] | None = None


class CorpusPatch(BaseModel):
    name: str | None = Field(None, min_length=2, max_length=160)
    domain: str | None = None
    description: str | None = Field(None, max_length=2000)
    source: str | None = Field(None, max_length=300)
    default_lang: str | None = None
    tasks: list[str] | None = None


@router.post("/corpora", status_code=201)
def create_corpus(body: CorpusIn, request: Request, p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    c = CmCorpus(name=body.name.strip(), domain=_check_domain(body.domain), description=body.description, source=(body.source or "").strip() or None,
                 default_lang=_check_lang(body.default_lang), tasks_json=_tasks(body.tasks), created_by=p.name,
                 last_edited_by=p.name, last_edited_key_id=p.key_id, last_edited_at=utcnow())
    db.add(c)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "A corpus with that name already exists.")
    audit.record(db, "cm.corpus_created", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id, detail={"name": c.name, "domain": c.domain})
    return _corpus_out(db, c)


@router.get("/corpora", dependencies=[Depends(require("read:datasets"))])
def list_corpora(state: str | None = None, domain: str | None = None, limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    q = select(CmCorpus).order_by(CmCorpus.id.desc()).limit(limit)
    if state:
        q = q.where(CmCorpus.state == state)
    if domain:
        q = q.where(CmCorpus.domain == domain)
    return [_corpus_out(db, c) for c in db.scalars(q)]


@router.get("/corpora/{corpus_id}", dependencies=[Depends(require("read:datasets"))])
def get_corpus(corpus_id: int, db: Session = Depends(get_db)):
    return _corpus_out(db, _corpus(db, corpus_id), detail=True)


@router.patch("/corpora/{corpus_id}")
def update_corpus(corpus_id: int, body: CorpusPatch, request: Request, p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    c = _corpus(db, corpus_id)
    if body.name is not None:
        c.name = body.name.strip()
    if body.domain is not None:
        c.domain = _check_domain(body.domain)
    if body.description is not None:
        c.description = body.description
    if body.source is not None:
        c.source = body.source.strip() or None
    if body.default_lang is not None:
        c.default_lang = _check_lang(body.default_lang)
    if body.tasks is not None:
        c.tasks_json = _tasks(body.tasks)
    reopened = _touch(c, p)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "A corpus with that name already exists.")
    audit.record(db, "cm.corpus_updated", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id, detail={"reopened": reopened})
    return _corpus_out(db, c)


# ------------------------------------------------------------------ uploads
def _read(raw: bytes, file: UploadFile, fmt: str, lang: str | None, source: str | None, c: CmCorpus) -> dict:
    return cm.parse_upload(raw, filename=file.filename or "", fmt=fmt, lang=lang or c.default_lang, source=source or c.source)


def _public(res: dict) -> dict:
    return {k: res[k] for k in ("valid", "skipped", "skipped_total", "identifier_lines", "identifier_lines_total", "identifier_types", "report")}


@router.post("/corpora/{corpus_id}/inspect")
async def inspect_upload(corpus_id: int, request: Request, file: UploadFile = File(...), format: str = Form("auto"), lang: str | None = Form(None),
                         source: str | None = Form(None), p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    """Checks a file against a corpus and stores nothing."""
    c = _corpus(db, corpus_id)
    try:
        res = _read(await _read_upload(file), file, format, lang, source, c)
    except cm.CorpusManagerError as e:
        _upload_error(e, db, request, p)
    audit.record(db, "cm.upload_inspected", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id,
                 detail={"valid": res["valid"], "records": res["report"]["records"], "identifier_lines": res["identifier_lines_total"]})
    return _public(res)


@router.post("/corpora/{corpus_id}/uploads", status_code=201)
async def add_upload(corpus_id: int, request: Request, file: UploadFile = File(...), format: str = Form("auto"), lang: str | None = Form(None),
                     source: str | None = Form(None), p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    c = _corpus(db, corpus_id)
    raw = await _read_upload(file)
    try:
        res = _read(raw, file, format, lang, source, c)
    except cm.CorpusManagerError as e:
        _upload_error(e, db, request, p)
    if not res["valid"]:
        audit.record(db, "cm.upload_rejected", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id,
                     detail={"reason": "identifiers", "identifier_lines": res["identifier_lines_total"]})
        raise HTTPException(422, {"message": "The file was not stored because it contains identifiers. Remove them and upload again.", **_public(res)})
    existing = db.scalar(select(func.count()).select_from(CmRecord).where(CmRecord.corpus_id == c.id)) or 0
    if existing + len(res["rows"]) > cm.MAX_CORPUS_RECORDS:
        raise HTTPException(422, f"A corpus holds at most {cm.MAX_CORPUS_RECORDS} records.")
    src_hash = hashlib.sha256(raw).hexdigest()
    if db.scalar(select(CmUpload).where(CmUpload.corpus_id == c.id, CmUpload.source_sha256 == src_hash)) is not None:
        raise HTTPException(409, "This exact file was already uploaded to this corpus.")
    up = CmUpload(corpus_id=c.id, filename=(file.filename or "")[:300], format=res["report"]["source_format"], source_sha256=src_hash, sha256="",
                  record_count=len(res["rows"]), report_json=res["report"], created_by=p.name)
    db.add(up)
    db.flush()
    original = [{"row": r["where"], "text": r["text"], "lang": r["lang"], "category": r["category"], "entities": r["entities"],
                 "source": r["source"], "status": r["status"]} for r in res["rows"]]
    up.storage_uri, up.sha256 = _save(f"cm/corpus{c.id}/upload{up.id}.jsonl.gz", original)
    for r in res["rows"]:
        db.add(CmRecord(corpus_id=c.id, upload_id=up.id, text=r["text"], norm_hash=cm.text_hash(r["text"]), lang=r["lang"], category=r["category"],
                        entities_json=r["entities"], parse_note=r["parse_note"], status=r["status"], source=r["source"], ext_id=r["ext_id"],
                        row_ref=r["where"], annotated_by=p.name if r["status"] == "annotated" else None,
                        annotated_key_id=p.key_id if r["status"] == "annotated" else None))
    reopened = _touch(c, p)
    db.commit()
    audit.record(db, "cm.upload_added", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id,
                 detail={"upload": up.id, "format": up.format, "records": up.record_count, "skipped": res["skipped_total"], "reopened": reopened})
    return {"upload_id": up.id, "corpus": _corpus_out(db, c), **_public(res)}


@router.get("/corpora/{corpus_id}/uploads/{upload_id}/original", dependencies=[Depends(require("read:dataset_rows"))])
def download_original(corpus_id: int, upload_id: int, db: Session = Depends(get_db)):
    """The records as they were ingested, before any annotation edit (gzip JSON Lines)."""
    up = db.get(CmUpload, upload_id)
    if up is None or up.corpus_id != corpus_id:
        raise HTTPException(404, "Unknown upload.")
    return Response(_blob(up.storage_uri, up.sha256), media_type="application/gzip",
                    headers={"Content-Disposition": f'attachment; filename="corpus{corpus_id}-upload{upload_id}-original.jsonl.gz"'})


# ------------------------------------------------------------------ records and annotation
@router.get("/corpora/{corpus_id}/records", dependencies=[Depends(require("read:dataset_rows"))])
def list_records(corpus_id: int, status: str | None = None, lang: str | None = None, category: str | None = None,
                 problem: str | None = Query(None, description="missing, invalid, duplicate or any"), q: str | None = Query(None, max_length=100),
                 limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    c = _corpus(db, corpus_id)
    base = select(CmRecord).where(CmRecord.corpus_id == c.id)
    if status:
        base = base.where(CmRecord.status == status)
    if lang:
        base = base.where(CmRecord.lang == lang)
    if category:
        base = base.where(CmRecord.category == category)
    if q:
        base = base.where(CmRecord.text.ilike("%" + q.replace("%", r"\%").replace("_", r"\_") + "%", escape="\\"))
    tasks = c.tasks_json or list(cm.TASKS)
    dupe_hashes = set(db.scalars(select(CmRecord.norm_hash).where(CmRecord.corpus_id == c.id, CmRecord.status != "rejected")
                                 .group_by(CmRecord.norm_hash).having(func.count() > 1)))

    def issues(r) -> list[dict]:
        if r.status == "rejected":
            return []
        out = [{"code": k, "problem": m} for k, m in cm.record_issues(_rec(r), tasks)]
        if r.norm_hash in dupe_hashes:
            out.append({"code": "duplicate", "problem": "the same text appears more than once"})
        return out

    if problem:                                               # filtering by problem needs every record checked; plain listing checks only the page
        keep = {r.id for r in db.scalars(select(CmRecord).where(CmRecord.corpus_id == c.id)) if any(x["code"].startswith(problem) or problem == "any" for x in issues(r))}
        base = base.where(CmRecord.id.in_(keep or {-1}))
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = db.scalars(base.order_by(CmRecord.id).limit(limit).offset(offset)).all()
    return {"total": total, "limit": limit, "offset": offset,
            "records": [{"id": r.id, "text": r.text, "lang": r.lang, "category": r.category, "entities": r.entities_json, "status": r.status,
                         "source": r.source, "annotated_by": r.annotated_by, "reviewed_by": r.reviewed_by, "issues": issues(r)} for r in rows]}


def _dupe_groups(recs: list[dict]) -> list[list[int]]:
    g: dict[str, list[int]] = {}
    for r in recs:
        if r["status"] != "rejected":
            g.setdefault(r["norm_hash"], []).append(r["id"])
    return [v for v in g.values() if len(v) > 1]


class Annotation(BaseModel):
    lang: str | None = None
    category: str | None = None
    entities: list | None = None
    clear_category: bool = False
    clear_entities: bool = False
    status: str | None = None


def _apply(r: CmRecord, c: CmCorpus, a: Annotation, p: Principal) -> None:
    """Applies one annotation edit with strict checks. Invalid values are refused here: they can only enter through an upload."""
    if a.lang is not None:
        r.lang = _check_lang(a.lang) or None
    if a.category is not None:
        cat = cm.clean_category(a.category)
        if cat not in cm.CATEGORIES:
            raise HTTPException(422, "category must be one of " + ", ".join(sorted(cm.CATEGORIES)))
        r.category = cat
    if a.clear_category:
        r.category = None
    if a.entities is not None:
        ents, note = cm.clean_entities(a.entities)
        if ents is None:
            raise HTTPException(422, note or "The entities could not be read.")
        problems = cm.check_entities(r.text, ents)
        if problems:
            raise HTTPException(422, "; ".join(problems[:3]))
        r.entities_json, r.parse_note = sorted(ents, key=lambda e: (e["start"], e["end"])), None
    if a.clear_entities:
        r.entities_json, r.parse_note = None, None
    status = cm.clean_status(a.status) if a.status else None
    if a.status and status is None:
        raise HTTPException(422, "status must be one of " + ", ".join(cm.STATUSES))
    changed_content = any(x is not None for x in (a.lang, a.category, a.entities)) or a.clear_category or a.clear_entities
    if status is None and changed_content and r.status in ("reviewed", "annotated"):
        status = "annotated"                                  # content changed after review: it needs to be reviewed again
    if status in ("annotated", "reviewed"):
        problems = [m for _, m in cm.record_issues(_rec(r), c.tasks_json or list(cm.TASKS))]
        if problems:
            raise HTTPException(422, "Cannot mark this record " + status + ": " + "; ".join(problems[:3]) + ".")
        if status == "reviewed":
            if get_settings().require_separate_reviewer and p.key_id == r.annotated_key_id:
                raise HTTPException(403, "A record must be reviewed by someone other than its annotator.")
            r.reviewed_by = p.name
        else:
            r.annotated_by, r.annotated_key_id, r.reviewed_by = p.name, p.key_id, None
    elif status == "unannotated":
        r.reviewed_by = None
    if status:
        r.status = status
    r.updated_at = utcnow()


def _record_out(r: CmRecord) -> dict:
    return {"id": r.id, "text": r.text, "lang": r.lang, "category": r.category, "entities": r.entities_json, "status": r.status,
            "annotated_by": r.annotated_by, "reviewed_by": r.reviewed_by}


@router.patch("/corpora/{corpus_id}/records/{record_id}")
def annotate_record(corpus_id: int, record_id: int, body: Annotation, request: Request, p: Principal = Depends(require("write:datasets")),
                    db: Session = Depends(get_db)):
    c = _corpus(db, corpus_id)
    r = db.get(CmRecord, record_id)
    if r is None or r.corpus_id != c.id:
        raise HTTPException(404, "Unknown record.")
    _apply(r, c, body, p)
    reopened = _touch(c, p)
    db.commit()
    audit.record(db, "cm.record_annotated", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id,
                 detail={"record": r.id, "status": r.status, "reopened": reopened})
    return _record_out(r)


class Bulk(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=1000)
    set: Annotation


@router.post("/corpora/{corpus_id}/records/bulk")
def bulk_annotate(corpus_id: int, body: Bulk, request: Request, p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    """Applies the same change to up to 1000 records (for example set the language, or mark records reviewed). All or none."""
    c = _corpus(db, corpus_id)
    recs = db.scalars(select(CmRecord).where(CmRecord.corpus_id == c.id, CmRecord.id.in_(body.ids))).all()
    if len(recs) != len(set(body.ids)):
        raise HTTPException(404, "Some records do not belong to this corpus.")
    if body.set.entities is not None:
        raise HTTPException(422, "Entities are set one record at a time (offsets differ per text).")
    try:
        for r in recs:
            _apply(r, c, body.set, p)
    except HTTPException as e:
        db.rollback()
        raise HTTPException(e.status_code, f"Record {r.id}: {e.detail}")
    reopened = _touch(c, p)
    db.commit()
    audit.record(db, "cm.records_bulk", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id,
                 detail={"records": len(recs), "status": body.set.status, "reopened": reopened})
    return {"updated": len(recs), "corpus": _corpus_out(db, c)}


@router.post("/corpora/{corpus_id}/resolve-duplicates")
def resolve_duplicates(corpus_id: int, request: Request, p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    """Keeps one copy of every duplicated text (the best-annotated, then the oldest) and rejects the others. Nothing is deleted."""
    c = _corpus(db, corpus_id)
    ids = cm.duplicate_extras(_records(db, c.id))
    for r in db.scalars(select(CmRecord).where(CmRecord.id.in_(ids or [-1]))):
        r.status, r.reviewed_by, r.updated_at = "rejected", None, utcnow()
    if ids:
        _touch(c, p)
    db.commit()
    audit.record(db, "cm.duplicates_resolved", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id, detail={"rejected": len(ids)})
    return {"rejected": len(ids), "corpus": _corpus_out(db, c)}


# ------------------------------------------------------------------ quality, review, approval
@router.get("/corpora/{corpus_id}/quality", dependencies=[Depends(require("read:datasets"))])
def quality_report(corpus_id: int, db: Session = Depends(get_db)):
    return _quality(db, _corpus(db, corpus_id))


@router.post("/corpora/{corpus_id}/submit")
def submit_for_review(corpus_id: int, request: Request, p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    c = _corpus(db, corpus_id)
    if c.state not in ("draft", "rejected"):
        raise HTTPException(409, f"The corpus is {c.state}; only a draft or rejected corpus can be submitted.")
    q = _quality(db, c)
    if not q["ready_for_approval"]:
        raise HTTPException(422, {"message": "The corpus is not ready for review.", "blockers": q["blockers"]})
    c.state, c.review_note = "in_review", None                  # submitting is not an edit: the editor stays on record, so they cannot approve
    db.commit()
    audit.record(db, "cm.corpus_submitted", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id, detail={"records": q["usable"]})
    return _corpus_out(db, c)


class Review(BaseModel):
    note: str = Field(min_length=5, max_length=2000)


@router.post("/corpora/{corpus_id}/approve")
def approve_corpus(corpus_id: int, body: Review, request: Request, p: Principal = Depends(require("approve:corpora")), db: Session = Depends(get_db)):
    """Approves a corpus in review and freezes a snapshot. The reviewer must not be the person who last edited it."""
    c = _corpus(db, corpus_id)
    if c.state != "in_review":
        raise HTTPException(409, "Only a corpus in review can be approved. Submit it first.")
    if get_settings().require_separate_reviewer and p.key_id == c.last_edited_key_id:
        raise HTTPException(403, "A corpus must be approved by someone other than the person who last edited it.")
    recs = _records(db, c.id)
    q = cm.quality(recs, c.tasks_json or list(cm.TASKS), _others(db, c.id))
    if not q["ready_for_approval"]:
        raise HTTPException(422, {"message": "The corpus is no longer ready.", "blockers": q["blockers"]})
    n = (db.scalar(select(func.max(CmSnapshot.number)).where(CmSnapshot.corpus_id == c.id)) or 0) + 1
    docs = cm.snapshot_docs(recs)
    uri, sha = _save(f"cm/corpus{c.id}/snapshot{n}.jsonl.gz", docs)
    s = CmSnapshot(corpus_id=c.id, number=n, sha256=sha, storage_uri=uri, doc_count=len(docs), report_json=q, approved_by=p.name, approval_note=body.note.strip())
    db.add(s)
    c.state, c.reviewed_by, c.reviewed_at, c.review_note = "approved", p.name, utcnow(), body.note.strip()
    db.commit()
    audit.record(db, "cm.corpus_approved", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id, detail={"snapshot": n, "documents": len(docs)})
    return _corpus_out(db, c, detail=True)


@router.post("/corpora/{corpus_id}/reject")
def reject_corpus(corpus_id: int, body: Review, request: Request, p: Principal = Depends(require("approve:corpora")), db: Session = Depends(get_db)):
    c = _corpus(db, corpus_id)
    if c.state != "in_review":
        raise HTTPException(409, "Only a corpus in review can be rejected.")
    c.state, c.reviewed_by, c.reviewed_at, c.review_note = "rejected", p.name, utcnow(), body.note.strip()
    db.commit()
    audit.record(db, "cm.corpus_rejected", principal=p, request=request, resource_type="cm_corpus", resource_id=c.id)
    return _corpus_out(db, c)


# ------------------------------------------------------------------ master datasets
class MasterIn(BaseModel):
    name: str = Field("cnr-nutrition-master", min_length=2, max_length=120)
    corpus_ids: list[int] = Field(min_length=1, max_length=50)
    on_conflict: str = "drop_conflicts"
    note: str | None = Field(None, max_length=2000)


def _gather(db: Session, ids: list[int]) -> list[dict]:
    """The latest snapshot of each chosen corpus, in the order given (earlier means higher priority). Every corpus must be approved now."""
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "A corpus can be listed only once.")
    out = []
    for cid in ids:
        c = _corpus(db, cid)
        if c.state != "approved":
            raise HTTPException(422, f"Corpus \"{c.name}\" is {c.state}. Only approved corpora can be merged.")
        s = db.scalar(select(CmSnapshot).where(CmSnapshot.corpus_id == c.id).order_by(CmSnapshot.number.desc()))
        if s is None:
            raise HTTPException(409, f"Corpus \"{c.name}\" has no snapshot.")
        out.append({"corpus_id": c.id, "name": c.name, "domain": c.domain, "snapshot_id": s.id, "snapshot_number": s.number,
                    "snapshot_sha256": s.sha256, "docs": cm.from_jsonl(gzip.decompress(_blob(s.storage_uri, s.sha256)))})
    return out


def _run_merge(db: Session, body: MasterIn) -> tuple[dict, list[dict]]:
    srcs = _gather(db, body.corpus_ids)
    try:
        return cm.merge(srcs, body.on_conflict), srcs
    except cm.MergeConflict as e:
        raise HTTPException(422, {"message": str(e), "conflicts": e.conflicts})
    except cm.CorpusManagerError as e:
        raise HTTPException(422, str(e))


@router.post("/masters/preview", dependencies=[Depends(require("write:datasets"))])
def preview_master(body: MasterIn, db: Session = Depends(get_db)):
    """Runs the merge and returns its report. Nothing is stored."""
    res, _ = _run_merge(db, body)
    return {"report": res["report"]}


@router.post("/masters", status_code=201)
def build_master(body: MasterIn, request: Request, p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    """Builds the next version of a named master dataset from the latest snapshots of the chosen approved corpora. Trains nothing."""
    name = SLUG.sub("-", body.name.strip()).strip("-")
    if len(name) < 2:
        raise HTTPException(422, "Give the master dataset a name (letters, digits, - _ .).")
    res, srcs = _run_merge(db, body)
    if not res["rows"]:
        raise HTTPException(422, "The merge would leave no records.")
    version = (db.scalar(select(func.max(CmMaster.version)).where(CmMaster.name == name)) or 0) + 1
    m = CmMaster(name=name, version=version, sources_json=[{k: s[k] for k in ("corpus_id", "name", "snapshot_id", "snapshot_number", "snapshot_sha256")} for s in srcs],
                 on_conflict=body.on_conflict, note=(body.note or "").strip() or None, record_count=len(res["rows"]), sha256="", storage_uri="",
                 report_json=res["report"], created_by=p.name)
    db.add(m)
    try:
        db.flush()
        m.storage_uri, m.sha256 = _save(f"cm/master{m.id}/v{version}.jsonl.gz", res["rows"])
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Another version was built at the same time. Try again.")
    audit.record(db, "cm.master_built", principal=p, request=request, resource_type="cm_master", resource_id=m.id,
                 detail={"name": name, "version": version, "corpora": len(srcs), "records": m.record_count, "conflicts": res["report"]["conflicts"]})
    return _master_out(m, detail=True)


def _master_out(m: CmMaster, detail: bool = False) -> dict:
    out = {"id": m.id, "name": m.name, "version": m.version, "records": m.record_count, "sha256": m.sha256, "on_conflict": m.on_conflict, "note": m.note,
           "sources": m.sources_json, "created_by": m.created_by, "created_at": _iso(m.created_at)}
    if detail:
        out["report"] = m.report_json
    return out


def _master(db: Session, master_id: int) -> CmMaster:
    m = db.get(CmMaster, master_id)
    if m is None:
        raise HTTPException(404, "Unknown master dataset.")
    return m


def _master_rows(m: CmMaster) -> list[dict]:
    return cm.from_jsonl(gzip.decompress(_blob(m.storage_uri, m.sha256)))


@router.get("/masters", dependencies=[Depends(require("read:datasets"))])
def list_masters(name: str | None = None, limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    q = select(CmMaster).order_by(CmMaster.id.desc()).limit(limit)
    if name:
        q = q.where(CmMaster.name == name)
    return [_master_out(m) for m in db.scalars(q)]


@router.get("/masters/{master_id}", dependencies=[Depends(require("read:datasets"))])
def get_master(master_id: int, db: Session = Depends(get_db)):
    return _master_out(_master(db, master_id), detail=True)


def _file(data: bytes, fmt: str, stem: str, extra: dict | None = None) -> Response:
    return Response(data, media_type=cm.EXPORT_FORMATS[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{stem}.{fmt}"', "X-Content-SHA256": hashlib.sha256(data).hexdigest(), **(extra or {})})


@router.get("/masters/{master_id}/export")
def export_master(master_id: int, request: Request, format: str = Query("csv"), spreadsheet_safe: bool = False,
                  p: Principal = Depends(require("read:dataset_rows")), db: Session = Depends(get_db)):
    """The full master dataset as CSV, JSONL or Parquet."""
    m = _master(db, master_id)
    try:
        data = cm.export_rows(_master_rows(m), cm.MASTER_COLUMNS, format, spreadsheet_safe=spreadsheet_safe)
    except cm.CorpusManagerError as e:
        raise HTTPException(422, str(e))
    audit.record(db, "cm.master_exported", principal=p, request=request, resource_type="cm_master", resource_id=m.id,
                 detail={"format": format, "records": m.record_count})
    return _file(data, format, f"{m.name}-v{m.version}", {"X-Master-Version": str(m.version), "X-Master-SHA256": m.sha256})


def _task(m: CmMaster, task: str, test_size: float, val_size: float, seed: int, include_mixed: bool) -> dict:
    try:
        return cm.build_task(_master_rows(m), task, test_size=test_size, val_size=val_size, seed=seed, include_mixed=include_mixed)
    except cm.CorpusManagerError as e:
        raise HTTPException(422, str(e))


@router.get("/masters/{master_id}/tasks/{task}/summary", dependencies=[Depends(require("read:datasets"))])
def task_summary(master_id: int, task: str, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42, include_mixed: bool = False,
                 db: Session = Depends(get_db)):
    """What a task dataset would contain (counts, classes, split, warnings). No text is returned."""
    return _task(_master(db, master_id), task, test_size, val_size, seed, include_mixed)["report"]


@router.get("/masters/{master_id}/tasks/{task}/export")
def export_task(master_id: int, task: str, request: Request, format: str = Query("jsonl"), test_size: float = 0.1, val_size: float = 0.1, seed: int = 42,
                include_mixed: bool = False, spreadsheet_safe: bool = False, p: Principal = Depends(require("read:dataset_rows")), db: Session = Depends(get_db)):
    """A task dataset (ner, classification or langid) from a master version, with a repeatable stratified train/validation/test split.
    Built on request from the stored master, so the same master version, options and seed always give the same file. Nothing is trained."""
    m = _master(db, master_id)
    res = _task(m, task, test_size, val_size, seed, include_mixed)
    if not res["rows"]:
        raise HTTPException(422, "No records qualify for this task. " + " ".join(res["report"]["warnings"]))
    try:
        data = cm.export_rows(res["rows"], res["columns"], format, spreadsheet_safe=spreadsheet_safe)
    except cm.CorpusManagerError as e:
        raise HTTPException(422, str(e))
    audit.record(db, "cm.task_dataset_exported", principal=p, request=request, resource_type="cm_master", resource_id=m.id,
                 detail={"task": task, "format": format, "records": res["report"]["records"], "seed": seed})
    return _file(data, format, f"{m.name}-v{m.version}-{task}", {"X-Master-Version": str(m.version), "X-Master-SHA256": m.sha256,
                                                                   "X-Task-Records": str(res["report"]["records"])})
