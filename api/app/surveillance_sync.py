"""Automatic pull of district-by-month surveillance counts from the CNR API into the laboratory.

CNR's `GET /surveillance/export` returns AGGREGATED counts (one row per month, district, group and indicator; cells under
the minimum size are already left out). This module fetches that CSV on a schedule, stores it as a new de-identified dataset
version through the same pipeline as a manual upload, and starts a hotspot analysis on it. Nothing individual is ever involved.

Rules it keeps:
- The CNR address comes from the environment (CNR_API_BASE_URL), never from a request. Redirects are refused.
- The CNR key (SURVEILLANCE_API_KEY) is sent only to that address, only as a Bearer header, and is never logged, audited or returned.
- An unchanged export (same SHA-256) creates no new version and no new analysis.
- One heavy job at a time: if a training run, exploration or analysis is running, the analysis is skipped and tried next time.
"""
import asyncio
import logging
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from . import audit, hotspot, ingest, storage
from .config import Settings
from .db import SessionLocal
from .integration import _NoRedirect, config_problem as cnr_address_problem
from .jobs import another_job_running
from .models import Dataset, DatasetVersion, HotspotAnalysis, utcnow

log = logging.getLogger("cnr-ml")

ACTOR = "surveillance-sync"
TARGET = re.compile(r"^([a-z0-9_]{1,40}):([a-z0-9_]{1,60})$")
GROUPS = {"under5", "school_age", "adolescent", "adult", "pregnant_lactating"}
INITIAL_DELAY_S = 120

_lock = threading.Lock()
LAST_RUN: dict = {}


class SyncError(RuntimeError):
    pass


def parse_targets(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        m = TARGET.match(part)
        if not m or m.group(1) not in GROUPS:
            raise SyncError(f"Bad target '{part[:60]}'. Use population_group:indicator, with a group from {sorted(GROUPS)}.")
        if (m.group(1), m.group(2)) not in out:
            out.append((m.group(1), m.group(2)))
    return out


def config_problem(s: Settings) -> str | None:
    """None when automatic sync can run, otherwise a short reason (shown to admins; never contains the key)."""
    if not s.surveillance_sync_enabled:
        return "Automatic surveillance sync is off (SURVEILLANCE_SYNC_ENABLED=false)."
    if len((s.surveillance_api_key or "").strip()) < 8:
        return "SURVEILLANCE_API_KEY is not set."
    bad = cnr_address_problem(s)
    if bad:
        return bad
    try:
        if not parse_targets(s.surveillance_sync_targets):
            return "SURVEILLANCE_SYNC_TARGETS is empty."
    except SyncError as e:
        return str(e)
    if not 1 <= s.surveillance_sync_months <= 36:
        return "SURVEILLANCE_SYNC_MONTHS must be from 1 to 36."
    return None


def export_url(s: Settings, group: str, indicator: str) -> str:
    q = urllib.parse.urlencode({"population_group": group, "indicator": indicator,
                                "months": s.surveillance_sync_months, "format": "csv"})
    return s.cnr_api_base_url.rstrip("/") + "/surveillance/export?" + q


def fetch_csv(s: Settings, group: str, indicator: str) -> bytes:
    """One authenticated GET of the CNR export. Raises SyncError with a short, key-free message."""
    req = urllib.request.Request(export_url(s, group, indicator), method="GET", headers={
        "Authorization": "Bearer " + s.surveillance_api_key.strip(), "Accept": "text/csv",
        "User-Agent": "cnr-ml-lab/" + s.service_version})
    limit = s.surveillance_fetch_max_mb * 1024 * 1024
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=s.surveillance_fetch_timeout_s) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            body = r.read(limit + 1)
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            raise SyncError(f"CNR answered HTTP {e.code} (a redirect). Redirects are not followed; check CNR_API_BASE_URL.")
        hint = {401: "the key was refused", 403: "the key is not an admin key", 404: "the export route was not found (is the CNR API updated?)",
                429: "rate limited", 502: "CNR could not read its data", 503: "surveillance storage is not set up on CNR"}.get(e.code, "unexpected answer")
        raise SyncError(f"CNR answered HTTP {e.code}: {hint}.")
    except Exception as e:  # DNS, TLS, timeout, refused, redirect
        raise SyncError(f"No usable answer from the CNR API ({type(e).__name__}).")
    if len(body) > limit:
        raise SyncError(f"The export is larger than {s.surveillance_fetch_max_mb} MB.")
    if "text/csv" not in ctype:
        raise SyncError("CNR did not return a CSV file.")
    return body


def _csv_rows(raw: bytes) -> int:
    return max(0, len([ln for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip()]) - 1)


def _dataset(db, group: str, indicator: str) -> Dataset:
    name = f"cnr-surveillance-{group}-{indicator}"
    d = db.scalar(select(Dataset).where(Dataset.name == name))
    if d is None:
        d = Dataset(name=name, sensitivity="internal", created_by=ACTOR, description=(
            f"District-by-month counts of {indicator} for {group}, pulled automatically from the CNR surveillance export. "
            "Aggregated counts only; cells under the minimum size were left out by CNR. Programme screening data, not a random survey."))
        db.add(d)
        db.commit()
    return d


def _store_version(db, d: Dataset, raw: bytes, approval_ref: str) -> DatasetVersion:
    """Same path as a manual upload: parse, clean, analyse, apply the default plan, store only the de-identified table."""
    df, fmt, warnings = ingest.parse_upload(raw, "cnr_surveillance_export.csv")
    df, notes = ingest.clean(df)
    reports = ingest.analyse(df)
    actions = {r.name: (r.suggested, None) for r in reports}
    out, report, schema = ingest.apply_plan(df, reports, actions, True, notes)
    report["warnings"] = warnings + report["warnings"]
    version = (db.scalar(select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == d.id)) or 0) + 1
    uri, stored = storage.save(d.id, version, out)
    row = DatasetVersion(dataset_id=d.id, version=version, sha256=ingest.sha256(stored), source_sha256=ingest.sha256(raw),
                         row_count=len(out), column_count=out.shape[1], file_format=fmt, storage_uri=uri, schema_json=schema,
                         report_json=report, deidentified=True, approval_ref=approval_ref, created_by=ACTOR)
    db.add(row)
    db.commit()
    return row


def _start_analysis(db, v: DatasetVersion, group: str, indicator: str) -> dict:
    """Runs a hotspot analysis on a version now (in this worker thread). Stores written results only."""
    from .routers.hotspots import run_hotspot  # local import: the hotspot code itself never imports this module
    if another_job_running(db):
        return {"analysis": "skipped", "analysis_note": "another run was in progress; will be tried at the next sync"}
    cfg = hotspot.Config(area_col="district", cases_col="cases", denominator_col="examined", period_col="month")
    x = HotspotAnalysis(dataset_version_id=v.id, status="queued", created_by=ACTOR,
                        params_json={"config": asdict(cfg), "dataset_id": v.dataset_id, "dataset_version": v.version,
                                     "note": f"Automatic: {indicator} in {group}"})
    db.add(x)
    db.commit()
    analysis_id = x.id
    run_hotspot(analysis_id)
    db.expire_all()
    done = db.get(HotspotAnalysis, analysis_id)
    return {"analysis": done.status, "analysis_id": analysis_id}


def sync_target(db, s: Settings, group: str, indicator: str, fetch=None) -> dict:
    res: dict = {"population_group": group, "indicator": indicator}
    try:
        raw = (fetch or fetch_csv)(s, group, indicator)
        if _csv_rows(raw) == 0:
            return {**res, "status": "no_data", "note": "CNR returned no rows (no data yet, or every cell is under the minimum size)."}
        d = _dataset(db, group, indicator)
        res["dataset_id"] = d.id
        src = ingest.sha256(raw)
        known = db.scalar(select(DatasetVersion).where(DatasetVersion.dataset_id == d.id, DatasetVersion.source_sha256 == src))
        if known is not None:
            v, res["status"] = known, "unchanged"
        else:
            try:
                v = _store_version(db, d, raw, s.surveillance_sync_approval_ref)
            except IntegrityError:
                db.rollback()
                raise SyncError("Another upload took the version number; will retry at the next sync.")
            res["status"] = "new_version"
        res.update(version=v.version, rows=v.row_count)
        if s.surveillance_sync_run_analysis:
            latest = db.scalar(select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == d.id))
            has = db.scalar(select(HotspotAnalysis.id).where(HotspotAnalysis.dataset_version_id == v.id).limit(1))
            if v.version == latest and has is None:   # new data, or an earlier analysis was skipped
                res.update(_start_analysis(db, v, group, indicator))
        return res
    except SyncError as e:
        return {**res, "status": "error", "error": str(e)}
    except ingest.IngestError as e:
        return {**res, "status": "error", "error": "The export was rejected by the dataset checks: " + str(e)[:160]}
    except Exception:
        log.exception("Surveillance sync failed for %s:%s", group, indicator)
        db.rollback()
        return {**res, "status": "error", "error": "Unexpected error. See the service logs."}


def run_once(s: Settings, trigger: str = "schedule", actor: str | None = None, fetch=None) -> dict | None:
    """One full sync of every target. Returns None when another sync is already running."""
    if not _lock.acquire(blocking=False):
        return None
    try:
        bad = config_problem(s)
        started = utcnow()
        LAST_RUN.clear()
        LAST_RUN.update(state="running", trigger=trigger, started_at=started, finished_at=None, results=[])
        if bad:
            LAST_RUN.update(state="not_configured", finished_at=utcnow(), error=bad)
            return dict(LAST_RUN)
        results = []
        with SessionLocal() as db:
            for group, indicator in parse_targets(s.surveillance_sync_targets):
                results.append(sync_target(db, s, group, indicator, fetch))
                LAST_RUN["results"] = list(results)
            failed = sum(r["status"] == "error" for r in results)
            summary = {"trigger": trigger, "actor": actor or ACTOR, "targets": len(results), "failed": failed,
                       "new_versions": sum(r["status"] == "new_version" for r in results),
                       "unchanged": sum(r["status"] == "unchanged" for r in results),
                       "no_data": sum(r["status"] == "no_data" for r in results)}
            audit.record(db, "surveillance_sync.completed", detail=summary)
        LAST_RUN.update(state="completed" if not failed else "completed_with_errors", finished_at=utcnow(), summary=summary)
        return dict(LAST_RUN)
    finally:
        _lock.release()


def status(s: Settings) -> dict:
    bad = config_problem(s)
    try:
        targets = [f"{g}:{i}" for g, i in parse_targets(s.surveillance_sync_targets)]
    except SyncError:
        targets = []
    return {"enabled": bool(s.surveillance_sync_enabled), "configured": bad is None, "problem": bad,
            "source": (s.cnr_api_base_url.rstrip("/") + "/surveillance/export") if s.cnr_api_base_url else None,
            "interval_hours": s.surveillance_sync_interval_hours, "months": s.surveillance_sync_months,
            "targets": targets, "run_analysis": s.surveillance_sync_run_analysis,
            "running": _lock.locked(), "last_run": dict(LAST_RUN) or None}


async def scheduler(s: Settings) -> None:
    """Background loop started with the app. Sleeps first so a restart does not hit CNR immediately."""
    interval = max(1.0, float(s.surveillance_sync_interval_hours)) * 3600
    log.info("Surveillance sync on: every %.1f h, %d target(s)", interval / 3600, len(parse_targets(s.surveillance_sync_targets)))
    try:
        await asyncio.sleep(INITIAL_DELAY_S)
        while True:
            try:
                await asyncio.to_thread(run_once, s, "schedule")
            except Exception:
                log.exception("Scheduled surveillance sync failed")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
