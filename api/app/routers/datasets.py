import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import audit, ingest, storage
from ..config import get_settings
from ..db import get_db
from ..models import Dataset, DatasetVersion
from ..schemas import DatasetCreate, DatasetDetail, DatasetOut, DatasetVersionOut, IngestPlan
from ..security import Principal, require

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])


def _out(db: Session, d: Dataset) -> dict:
    n = db.scalar(select(func.count()).select_from(DatasetVersion).where(DatasetVersion.dataset_id == d.id)) or 0
    return {**DatasetOut.model_validate(d).model_dump(), "version_count": n}


async def _read_upload(file: UploadFile) -> bytes:
    limit = get_settings().max_upload_mb * 1024 * 1024
    buf = bytearray()
    while chunk := await file.read(1024 * 1024):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(413, f"File is larger than {get_settings().max_upload_mb} MB.")
    if not buf:
        raise HTTPException(400, "The uploaded file is empty.")
    return bytes(buf)


def _parse(raw: bytes, filename: str, db: Session, request: Request, p: Principal):
    try:
        return ingest.parse_upload(raw, filename)
    except ingest.IngestError as e:
        audit.record(db, "dataset.upload_rejected", principal=p, request=request, detail={"reason": str(e)[:200]})
        raise HTTPException(422, str(e))


@router.get("", response_model=list[DatasetOut], dependencies=[Depends(require("read:datasets"))])
def list_datasets(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return [_out(db, d) for d in db.scalars(select(Dataset).order_by(Dataset.id.desc()).limit(limit))]


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(body: DatasetCreate, request: Request, p: Principal = Depends(require("write:datasets")),
                   db: Session = Depends(get_db)):
    row = Dataset(name=body.name.strip(), description=body.description, sensitivity=body.sensitivity, created_by=p.name)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "A dataset with that name already exists.")
    audit.record(db, "dataset.created", principal=p, request=request, resource_type="dataset", resource_id=row.id,
                 detail={"name": row.name, "sensitivity": row.sensitivity})
    return _out(db, row)


@router.post("/inspect")
async def inspect_file(request: Request, file: UploadFile = File(...),
                       p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    """Step 1. Parse and analyse an upload and propose a de-identification plan.
    Nothing is stored. Flagged columns are not profiled or echoed back."""
    raw = await _read_upload(file)
    df, fmt, warnings = _parse(raw, file.filename or "", db, request, p)
    df, notes = ingest.clean(df)
    reports = ingest.analyse(df)
    audit.record(db, "dataset.inspected", principal=p, request=request,
                 detail={"rows": len(df), "columns": df.shape[1], "flagged": sum(r.category != "none" for r in reports),
                         "source_sha256": ingest.sha256(raw)})
    return {
        "file_format": fmt, "rows": int(len(df)), "source_sha256": ingest.sha256(raw), "warnings": warnings,
        "pseudonymisation_available": len(get_settings().pseudonym_key) >= 32,
        "date_assumptions": notes["date_assumptions"], "coerced_to_missing": notes["coerced_to_missing"],
        "columns": [{"name": r.name, "kind": r.kind, "category": r.category, "reason": r.reason,
                     "suggested": r.suggested, "can_keep": r.can_keep, "quasi": r.quasi, "profile": r.profile}
                    for r in reports],
    }


@router.get("/{dataset_id}", response_model=DatasetDetail, dependencies=[Depends(require("read:datasets"))])
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    d = db.get(Dataset, dataset_id)
    if d is None:
        raise HTTPException(404, "Unknown dataset.")
    vs = db.scalars(select(DatasetVersion).where(DatasetVersion.dataset_id == d.id)
                    .order_by(DatasetVersion.version.desc())).all()
    return {**_out(db, d), "versions": [DatasetVersionOut.model_validate(v).model_dump() for v in vs]}


@router.post("/{dataset_id}/versions", response_model=DatasetVersionOut, status_code=201)
async def create_version(dataset_id: int, request: Request, file: UploadFile = File(...),
                         plan: str = Form("{}"), approval_ref: str = Form(...),
                         p: Principal = Depends(require("write:datasets")), db: Session = Depends(get_db)):
    """Step 2. Re-parse the file, apply the plan, store only the de-identified table as a new version."""
    d = db.get(Dataset, dataset_id)
    if d is None:
        raise HTTPException(404, "Unknown dataset.")
    approval_ref = approval_ref.strip()
    if len(approval_ref) < 5 or len(approval_ref) > 300:
        raise HTTPException(422, "Give the data-sharing or ethics approval reference (5-300 characters).")
    try:
        plan_obj = IngestPlan.model_validate(json.loads(plan or "{}"))
    except (json.JSONDecodeError, ValidationError):
        raise HTTPException(422, "The de-identification plan is not valid.")

    raw = await _read_upload(file)
    source_hash = ingest.sha256(raw)
    dup = db.scalar(select(DatasetVersion).where(DatasetVersion.dataset_id == d.id,
                                                 DatasetVersion.source_sha256 == source_hash))
    if dup is not None:
        raise HTTPException(409, f"This exact file was already ingested as version {dup.version}.")

    df, fmt, warnings = _parse(raw, file.filename or "", db, request, p)
    df, notes = ingest.clean(df)
    reports = ingest.analyse(df)
    known = {r.name for r in reports}
    actions = {r.name: (r.suggested, None) for r in reports}
    for c in plan_obj.columns:
        if c.name not in known:
            raise HTTPException(422, f"The plan refers to a column that is not in the file: {c.name[:80]}")
        actions[c.name] = (c.action, c.override_reason)
    try:
        out, report, schema = ingest.apply_plan(df, reports, actions, plan_obj.drop_duplicates, notes)
    except ingest.IngestError as e:
        audit.record(db, "dataset.upload_rejected", principal=p, request=request, resource_type="dataset",
                     resource_id=d.id, detail={"reason": str(e)[:200]})
        raise HTTPException(422, str(e))
    report["warnings"] = warnings + report["warnings"]

    version = (db.scalar(select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == d.id)) or 0) + 1
    uri, stored = storage.save(d.id, version, out)
    row = DatasetVersion(dataset_id=d.id, version=version, sha256=ingest.sha256(stored), source_sha256=source_hash,
                         row_count=len(out), column_count=out.shape[1], file_format=fmt, storage_uri=uri,
                         schema_json=schema, report_json=report, deidentified=True, approval_ref=approval_ref,
                         created_by=p.name)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:  # concurrent upload took this version number
        db.rollback()
        raise HTTPException(409, "Another upload finished first. Please try again.")
    audit.record(db, "dataset.version_created", principal=p, request=request, resource_type="dataset",
                 resource_id=d.id, detail={"version": version, "rows": len(out), "columns": out.shape[1],
                                           "dropped": len(report["dropped"]), "pseudonymised": len(report["pseudonymised"]),
                                           "overrides": [o["column"] for o in report["overrides"]],
                                           "source_sha256": source_hash, "approval_ref": approval_ref})
    return row


@router.get("/{dataset_id}/versions/{version}/preview")
def preview(dataset_id: int, version: int, request: Request, rows: int = Query(20, ge=1, le=50),
            p: Principal = Depends(require("read:dataset_rows")), db: Session = Depends(get_db)):
    """First rows of the de-identified table, for checking the result. Audited."""
    v = db.scalar(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id, DatasetVersion.version == version))
    if v is None or not v.storage_uri:
        raise HTTPException(404, "Unknown dataset version.")
    df = storage.load(v.storage_uri, nrows=rows)
    audit.record(db, "dataset.preview_viewed", principal=p, request=request, resource_type="dataset",
                 resource_id=dataset_id, detail={"version": version, "rows": len(df)})
    return {"columns": list(df.columns), "rows": json.loads(df.to_json(orient="values", date_format="iso"))}
