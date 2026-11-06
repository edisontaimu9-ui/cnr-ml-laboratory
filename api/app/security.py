import hashlib
import hmac
import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import audit
from .config import get_settings
from .ratelimit import SlidingWindow
from .db import get_db
from .models import ApiKey, utcnow

_VIEW = {"read:catalog", "read:models", "read:experiments", "read:datasets", "read:explorations", "read:hotspots"}
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer": _VIEW,
    "data_scientist": _VIEW | {"write:datasets", "write:experiments", "write:explorations", "write:hotspots", "read:dataset_rows", "write:models", "predict:models"},
    "clinical_reviewer": _VIEW | {"approve:models", "read:audit", "predict:models"},
    # Food registry data owner: may review food-name matchers only (enforced in the registry router), nothing else.
    "registry_owner": _VIEW | {"approve:matchers"},
    # Machine key for the CNR integration: can list the catalogue and call predictions, nothing else.
    "service": {"read:catalog", "read:models", "predict:models"},
    "admin": _VIEW | {"write:datasets", "write:experiments", "write:explorations", "write:hotspots", "read:dataset_rows", "write:models", "approve:models",
                      "read:audit", "manage:keys", "deploy:models", "predict:models", "read:integration", "manage:sync"},
}


_s = get_settings()
# Counts wrong-key attempts per client address. It keeps the audit log and database from being flooded;
# it is not what protects the keys (they are 256-bit random values). Valid keys are never blocked by it.
fail_limiter = SlidingWindow(_s.auth_fail_limit, _s.auth_fail_window_s)


def _addr(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[-1].strip() if fwd else (request.client.host if request.client else "unknown")


@dataclass
class Principal:
    key_id: int
    name: str
    role: str

    @property
    def permissions(self) -> set[str]:
        return ROLE_PERMISSIONS.get(self.role, set())


def hash_key(raw: str) -> str:
    # Keys are 256-bit random values, so a plain SHA-256 digest is sufficient.
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_key() -> tuple[str, str]:
    prefix = secrets.token_hex(4)
    return f"cnrml_{prefix}_{secrets.token_urlsafe(32)}", prefix


def _extract(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def get_principal(request: Request, db: Session = Depends(get_db),
                  authorization: str | None = Header(default=None),
                  x_api_key: str | None = Header(default=None)) -> Principal:
    raw = _extract(authorization, x_api_key)
    row = db.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(raw))) if raw else None
    if row is None or row.revoked_at is not None or not hmac.compare_digest(row.key_hash, hash_key(raw or "")):
        addr = _addr(request)
        if not fail_limiter.hit(addr):
            raise HTTPException(429, "Too many failed attempts. Wait a few minutes.",
                                headers={"Retry-After": str(fail_limiter.retry_after(addr))})
        audit.record(db, "auth.denied", request=request, detail={"path": request.url.path})
        raise HTTPException(401, "Missing or invalid API key.", headers={"WWW-Authenticate": "Bearer"})
    row.last_used_at = utcnow()
    db.commit()
    return Principal(key_id=row.id, name=row.name, role=row.role)


def require_any(*permissions: str):
    """Allow a principal holding at least one of the permissions. Callers narrow further where needed."""
    def dep(request: Request, principal: Principal = Depends(get_principal),
            db: Session = Depends(get_db)) -> Principal:
        if not set(permissions) & principal.permissions:
            audit.record(db, "auth.forbidden", principal=principal, request=request,
                         detail={"permission": " or ".join(permissions), "path": request.url.path})
            raise HTTPException(403, f"Your role cannot perform this action ({' or '.join(permissions)}).")
        return principal
    return dep


def require(permission: str):
    def dep(request: Request, principal: Principal = Depends(get_principal),
            db: Session = Depends(get_db)) -> Principal:
        if permission not in principal.permissions:
            audit.record(db, "auth.forbidden", principal=principal, request=request,
                         detail={"permission": permission, "path": request.url.path})
            raise HTTPException(403, f"Your role cannot perform this action ({permission}).")
        return principal
    return dep
