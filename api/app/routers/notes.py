"""Laboratory memory: notes (decisions, findings, lessons, warnings, questions) attached to datasets, experiments,
explorations, hotspot analyses, models and model versions, or to the lab in general.

Notes are written by people and read back by people (and later by an assistant). They hold no row-level data: text that
looks like an email, phone or ID number is refused. Notes are archived, never deleted, and edits keep the earlier
wording. The audit log records who did what and to which note, never the note text."""
import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .. import audit
from ..db import get_db
from ..ingest import EMAIL, PHONE
from ..models import (Dataset, DatasetVersion, Experiment, Exploration, HotspotAnalysis, LabNote, ModelDefinition,
                      ModelVersion, utcnow)
from ..schemas import NoteCreate, NoteHistoryOut, NoteOut, NoteUpdate
from ..security import Principal, require

router = APIRouter(prefix="/v1/notes", tags=["notes"])

LONG_NUMBER = re.compile(r"(?<!\d)\d{9,13}(?!\d)")
TAG = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,29}$")
MAX_HISTORY = 20
_NUMERIC = {"dataset": Dataset, "dataset_version": DatasetVersion, "experiment": Experiment,
            "exploration": Exploration, "hotspot": HotspotAnalysis, "model_version": ModelVersion}


def _check_text(*texts: str) -> None:
    for t in texts:
        if t and (EMAIL.search(t) or PHONE.search(t) or LONG_NUMBER.search(t)):
            raise HTTPException(422, "A note must not contain emails, phone numbers, ID numbers or patient-level data. "
                                     "Describe it in words instead.")


def _tags(raw: list[str]) -> list[str]:
    out: list[str] = []
    for t in raw:
        t = t.strip().lower()
        if not TAG.match(t):
            raise HTTPException(422, "Tags use letters, digits, spaces, _ and -, up to 30 characters.")
        if t not in out:
            out.append(t)
    return out


def _check_subject(db: Session, stype: str, sid: str | None) -> str | None:
    if stype == "general":
        if sid:
            raise HTTPException(422, "A general note has no subject id.")
        return None
    if not sid:
        raise HTTPException(422, "subject_id is required for this subject type.")
    if stype == "model":
        ok = db.scalar(select(ModelDefinition.id).where(ModelDefinition.slug == sid)) is not None
    else:
        ok = sid.isdigit() and db.get(_NUMERIC[stype], int(sid)) is not None
    if not ok:
        raise HTTPException(404, "Unknown subject for this note.")
    return sid


def _out(n: LabNote, history: bool = False) -> dict:
    d = {"id": n.id, "subject_type": n.subject_type, "subject_id": n.subject_id, "kind": n.kind, "title": n.title,
         "body": n.body, "tags": n.tags_json or [], "pinned": bool(n.pinned), "created_by": n.created_by,
         "created_at": n.created_at, "updated_at": n.updated_at, "archived_at": n.archived_at,
         "edits": len(n.history_json or [])}
    if history:
        d["history"] = n.history_json or []
    return d


def _get(db: Session, nid: int) -> LabNote:
    n = db.get(LabNote, nid)
    if n is None:
        raise HTTPException(404, "Unknown note.")
    return n


def _may_change(n: LabNote, p: Principal) -> None:
    if p.role != "admin" and n.created_by != p.name:
        raise HTTPException(403, "Only the author or an administrator can change this note.")


def _ordered(q):
    return q.order_by(LabNote.pinned.desc(), LabNote.id.desc())


@router.get("", response_model=list[NoteOut], dependencies=[Depends(require("read:notes"))])
def list_notes(subject_type: str | None = Query(None, max_length=32), subject_id: str | None = Query(None, max_length=60),
               kind: str | None = Query(None, max_length=24), tag: str | None = Query(None, max_length=30),
               q: str | None = Query(None, max_length=80), pinned: bool | None = None,
               include_archived: bool = False, limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    sel = select(LabNote)
    if not include_archived:
        sel = sel.where(LabNote.archived_at.is_(None))
    if subject_type:
        sel = sel.where(LabNote.subject_type == subject_type)
    if subject_id:
        sel = sel.where(LabNote.subject_id == subject_id)
    if kind:
        sel = sel.where(LabNote.kind == kind)
    if pinned is not None:
        sel = sel.where(LabNote.pinned == pinned)
    if q and q.strip():
        like = "%" + q.strip().replace("%", r"\%").replace("_", r"\_") + "%"
        sel = sel.where(or_(LabNote.title.ilike(like, escape="\\"), LabNote.body.ilike(like, escape="\\")))
    rows = db.scalars(_ordered(sel).limit(500)).all()
    if tag:
        rows = [n for n in rows if tag.strip().lower() in (n.tags_json or [])]
    return [_out(n) for n in rows[:limit]]


@router.get("/recall", dependencies=[Depends(require("read:notes"))])
def recall(subject_type: str = Query(..., max_length=32), subject_id: str | None = Query(None, max_length=60),
           db: Session = Depends(get_db)):
    """What the lab remembers when working on one thing: the pinned notes across the lab, every warning and decision
    about this subject, and its latest other notes. Same content a person sees on the subject's page."""
    live = LabNote.archived_at.is_(None)
    pinned = db.scalars(_ordered(select(LabNote).where(live, LabNote.pinned.is_(True))).limit(10)).all()
    about = []
    if subject_type != "general" and subject_id:
        about = db.scalars(_ordered(select(LabNote).where(live, LabNote.subject_type == subject_type,
                                                          LabNote.subject_id == subject_id)).limit(20)).all()
    return {"subject_type": subject_type, "subject_id": subject_id,
            "pinned": [_out(n) for n in pinned], "about_subject": [_out(n) for n in about]}


@router.get("/{nid}", response_model=NoteHistoryOut, dependencies=[Depends(require("read:notes"))])
def get_note(nid: int, db: Session = Depends(get_db)):
    return _out(_get(db, nid), history=True)


@router.post("", response_model=NoteOut, status_code=201)
def create_note(body: NoteCreate, request: Request, p: Principal = Depends(require("write:notes")),
                db: Session = Depends(get_db)):
    title, text = body.title.strip(), body.body.strip()
    if len(title) < 2 or not text:
        raise HTTPException(422, "A note needs a title and some text.")
    _check_text(title, text, *body.tags)
    sid = _check_subject(db, body.subject_type, body.subject_id.strip() if body.subject_id else None)
    n = LabNote(subject_type=body.subject_type, subject_id=sid, kind=body.kind, title=title, body=text,
                tags_json=_tags(body.tags), pinned=body.pinned, created_by=p.name)
    db.add(n)
    db.commit()
    audit.record(db, "note.create", principal=p, request=request, resource_type="note", resource_id=n.id,
                 detail={"subject_type": n.subject_type, "subject_id": n.subject_id, "kind": n.kind})
    return _out(n)


@router.patch("/{nid}", response_model=NoteOut)
def update_note(nid: int, body: NoteUpdate, request: Request, p: Principal = Depends(require("write:notes")),
                db: Session = Depends(get_db)):
    n = _get(db, nid)
    _may_change(n, p)
    if n.archived_at is not None:
        raise HTTPException(409, "This note is archived.")
    title = body.title.strip() if body.title is not None else None
    text = body.body.strip() if body.body is not None else None
    if title is not None and len(title) < 2 or text is not None and not text:
        raise HTTPException(422, "A note needs a title and some text.")
    _check_text(title or "", text or "", *(body.tags or []))
    changed = []
    if (title is not None and title != n.title) or (text is not None and text != n.body):
        n.history_json = [*(n.history_json or []), {"at": utcnow().isoformat() + "Z", "by": p.name,
                                                     "title": n.title, "body": n.body}][-MAX_HISTORY:]
        changed.append("text")
    if title is not None:
        n.title = title
    if text is not None:
        n.body = text
    if body.kind is not None and body.kind != n.kind:
        n.kind = body.kind
        changed.append("kind")
    if body.tags is not None:
        n.tags_json = _tags(body.tags)
        changed.append("tags")
    if body.pinned is not None and body.pinned != n.pinned:
        n.pinned = body.pinned
        changed.append("pin")
    if changed:
        n.updated_at = utcnow()
        db.commit()
        audit.record(db, "note.update", principal=p, request=request, resource_type="note", resource_id=n.id,
                     detail={"changed": changed})
    return _out(n)


@router.delete("/{nid}", status_code=204)
def archive_note(nid: int, request: Request, p: Principal = Depends(require("write:notes")),
                 db: Session = Depends(get_db)):
    """Archives the note. It stops showing in lists and recall but is kept, and the audit log records who archived it."""
    n = _get(db, nid)
    _may_change(n, p)
    if n.archived_at is None:
        n.archived_at, n.pinned = utcnow(), False
        db.commit()
        audit.record(db, "note.archive", principal=p, request=request, resource_type="note", resource_id=n.id)
