from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from .models import AuditLog

# Never store secrets or request bodies in audit detail.
_FORBIDDEN = {"key", "authorization", "password", "token", "secret"}


def _client_ip(request: Request | None) -> str | None:
    if request is None:
        return None
    fwd = request.headers.get("x-forwarded-for")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else None))


def record(db: Session, action: str, *, principal=None, request: Request | None = None,
           resource_type: str | None = None, resource_id: Any = None,
           detail: dict | None = None) -> None:
    clean = {k: v for k, v in (detail or {}).items() if k.lower() not in _FORBIDDEN}
    db.add(AuditLog(
        actor_key_id=getattr(principal, "key_id", None),
        actor_name=getattr(principal, "name", None),
        role=getattr(principal, "role", None),
        action=action,
        resource_type=resource_type,
        resource_id=None if resource_id is None else str(resource_id),
        detail=clean,
        ip=_client_ip(request),
    ))
    db.commit()
