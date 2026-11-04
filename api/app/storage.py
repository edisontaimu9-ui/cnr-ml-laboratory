"""Private storage for de-identified dataset versions.

Only the *processed, de-identified* table is ever stored. The raw upload is
parsed in memory and discarded. Backend is local disk (a persistent volume in
production); swap this module for object storage later without touching callers."""
import io
import os
from pathlib import Path

import pandas as pd

from .config import get_settings


def _root() -> Path:
    root = Path(get_settings().dataset_storage_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path(uri: str) -> Path:
    if not uri.startswith("local://"):
        raise ValueError("Unsupported storage URI.")
    p = (_root() / uri[len("local://"):]).resolve()
    if _root() not in p.parents:  # path traversal guard
        raise ValueError("Invalid storage URI.")
    return p


def save(dataset_id: int, version: int, df: pd.DataFrame) -> tuple[str, bytes]:
    """Write df as gzip CSV. Returns (uri, bytes) so the caller can hash the exact stored content."""
    rel = f"ds{dataset_id}/v{version}.csv.gz"
    buf = io.BytesIO()
    df.to_csv(buf, index=False, compression="gzip")
    data = buf.getvalue()
    target = _path("local://" + rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, target)
    return "local://" + rel, data


def load(uri: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(_path(uri), compression="gzip", nrows=nrows)
