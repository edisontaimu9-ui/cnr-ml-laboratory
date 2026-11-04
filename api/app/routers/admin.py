from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit
from ..db import get_db
from ..models import ApiKey, AuditLog, utcnow
from ..schemas import ApiKeyCreate, ApiKeyCreated, ApiKeyOut, AuditOut
from ..security import Principal, generate_key, hash_key, require

router = APIRouter(prefix="/v1", tags=["admin"])


@router.get("/audit", response_model=list[AuditOut])
def audit_log(action: str | None = None, limit: int = Query(100, ge=1, le=500),
              offset: int = Query(0, ge=0), _: Principal = Depends(require("read:audit")),
              db: Session = Depends(get_db)):
    q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit).offset(offset)
    if action:
        q = q.where(AuditLog.action == action)
    return db.scalars(q).all()


@router.get("/admin/keys", response_model=list[ApiKeyOut])
def list_keys(_: Principal = Depends(require("manage:keys")), db: Session = Depends(get_db)):
    return db.scalars(select(ApiKey).order_by(ApiKey.id)).all()


@router.post("/admin/keys", response_model=ApiKeyCreated, status_code=201)
def create_key(body: ApiKeyCreate, request: Request,
               p: Principal = Depends(require("manage:keys")), db: Session = Depends(get_db)):
    raw, prefix = generate_key()
    row = ApiKey(name=body.name, role=body.role, prefix=prefix,
                 key_hash=hash_key(raw), created_by=p.name)
    db.add(row)
    db.commit()
    audit.record(db, "key.created", principal=p, request=request, resource_type="api_key",
                 resource_id=row.id, detail={"name": row.name, "role": row.role})
    return ApiKeyCreated.model_validate({**ApiKeyOut.model_validate(row).model_dump(), "key": raw})


@router.delete("/admin/keys/{key_id}", status_code=204)
def revoke_key(key_id: int, request: Request, p: Principal = Depends(require("manage:keys")),
               db: Session = Depends(get_db)):
    row = db.get(ApiKey, key_id)
    if row is None:
        raise HTTPException(404, "Unknown key.")
    if row.id == p.key_id:
        raise HTTPException(400, "You cannot revoke the key you are using.")
    if row.revoked_at is None:
        row.revoked_at = utcnow()
        db.commit()
        audit.record(db, "key.revoked", principal=p, request=request,
                     resource_type="api_key", resource_id=row.id, detail={"name": row.name})
