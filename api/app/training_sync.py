"""Automatic pull of de-identified, record-level TRAINING data from the CNR API into the laboratory, for every catalogue model.

The surveillance sync (surveillance_sync.py) brings in aggregated district-by-month counts for hotspot analysis. This module is its
counterpart for model training: it fetches CNR's `GET /surveillance/training-export` (one row per screened person, coarse columns
and classified indicators only; groups smaller than CNR's minimum size are already left out) and stores each change as a new dataset
version, ready for an experiment. It never trains, registers or deploys anything: a synced dataset is only data.

Which models can be fed?  SOURCES lists every model in the catalogue. A model has a source only when CNR really holds the data it
needs. Today CNR keeps one kind of individual-level health data (coarse, classified screening rows), so only the childhood
malnutrition model has a source; the others are listed with the reason, and gain a source by one entry here once CNR stores the data.

Rules it keeps (the same as the surveillance sync, plus two for record-level data):
- The CNR address comes from the environment (CNR_API_BASE_URL), never from a request. Redirects are refused. The CNR key
  (SURVEILLANCE_API_KEY) is sent only to that address, only as a Bearer header, and is never logged, audited or returned.
- An unchanged export (same SHA-256) creates no new version.
- Repeated rows are KEPT. Coarse records repeat by nature; removing repeats would erase the frequencies a model learns from.
- The label column and the columns that define it are named for the person training (`target`, `exclude`), so a model is not
  trained to "predict" a status from the measurements it was derived from.
"""
import asyncio
import csv
import io
import logging
import threading
import urllib.parse
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from . import audit, ingest
from .catalog import CATALOG
from .config import Settings
from .db import SessionLocal
from .integration import config_problem as cnr_address_problem
from .models import Dataset, DatasetVersion, utcnow
from .surveillance_sync import SyncError, _csv_rows, _store_version, get_csv
from .training import MIN_PER_CLASS, MIN_ROWS

log = logging.getLogger("cnr-ml")

ACTOR = "training-sync"
INITIAL_DELAY_S = 180
DATASET_PREFIX = "cnr-training-"

_lock = threading.Lock()
LAST_RUN: dict = {}


@dataclass(frozen=True)
class Source:
    """What CNR can supply for one catalogue model. population_group is None when it cannot (yet); `reason` then says why."""
    population_group: str | None = None
    target: str | None = None
    exclude: tuple[str, ...] = ()
    note: str = ""
    reason: str = ""


_NO_INDIVIDUAL = ("CNR holds no individual-level {what}. Its only individual-level health data is coarse, classified screening rows "
                  "(month, district, sex, age band, setting, indicator statuses); it keeps no raw measurements and no clinical records.")

_SCHOOL_AGE = ("CNR keeps school-age stunting and overweight statuses, but none of the household, diet, activity or early-life context this "
                "model predicts them from. Training on district, sex and age band alone would be a different, much weaker model.")

SOURCES: dict[str, Source] = {
    "child-malnutrition-risk": Source(
        population_group="under5", target="any_undernutrition",
        # The label is derived from stunting, wasting and underweight; MUAC, acute malnutrition, oedema and overweight are statuses
        # from the same measurements. Training on any of them would be reading the answer, so they are held out by default.
        exclude=("stunting", "wasting", "underweight", "muac_gam", "acute_malnutrition", "bilateral_oedema", "overweight"),
        note=("One row per screened child under 5. any_undernutrition is yes when stunting, wasting or underweight is moderate or severe "
              "(CNR's WHO classification, not recomputed here). Features left after the default exclusions: month, district, region, sex, "
              "age band, setting, anaemia, dietary diversity, household food insecurity, recent diarrhoea and breastfeeding. CNR discards "
              "raw weight, length and MUAC, so anthropometry is not available as a feature. Programme screening data, not a random survey.")),
    "acute-malnutrition-classification": Source(
        reason=("CNR keeps only the classified status, not the weight, length, MUAC or oedema measurements this model classifies from, so "
                "the label would also be its own feature. It needs measurements stored with consent on the CNR side.")),
    "diabetes-risk": Source(reason=_NO_INDIVIDUAL.format(what="clinical, anthropometric or dietary records for adults")),
    "ckd-nutrition-risk": Source(reason=_NO_INDIVIDUAL.format(what="renal records")),
    "hospital-malnutrition-risk": Source(reason=_NO_INDIVIDUAL.format(what="hospital admission or screening records")),
    "food-insecurity-prediction": Source(
        reason=("CNR records whether a screened household was food insecure, but none of the socioeconomic or agricultural context this "
                "model predicts it from.")),
    "nutritional-outcome-prediction": Source(
        reason=("CNR holds no follow-up: screening rows are anonymous and cannot be linked over time, so there are no outcomes after an "
                "intervention to learn from.")),
    "hypertension-risk": Source(reason=_NO_INDIVIDUAL.format(what="blood pressure, waist, diet or activity records")),
    "metabolic-syndrome-screen": Source(reason=_NO_INDIVIDUAL.format(what="waist, blood pressure or metabolic records")),
    "prediabetes-screen": Source(reason=_NO_INDIVIDUAL.format(what="HbA1c or glucose results")),
    "low-birth-weight-risk": Source(reason=_NO_INDIVIDUAL.format(what="maternal, antenatal or birth-weight records")),
    "preterm-growth-faltering": Source(reason=_NO_INDIVIDUAL.format(what="neonatal unit or Fenton growth records")),
    "school-age-overweight-risk": Source(reason=_SCHOOL_AGE),
    "school-age-stunting-risk": Source(reason=_SCHOOL_AGE),
    "refeeding-syndrome-risk": Source(reason=_NO_INDIVIDUAL.format(what="hospital, electrolyte or feeding records")),
    "malnutrition-prolonged-stay-risk": Source(reason=_NO_INDIVIDUAL.format(what="hospital admission or length-of-stay records")),
    "malnutrition-readmission-risk": Source(reason=_NO_INDIVIDUAL.format(what="discharge or readmission records")),
    "weight-loss-trajectory": Source(reason=_NO_INDIVIDUAL.format(what="follow-up weight records")),
    "food-name-matching": Source(
        reason=("The labelled queries this matcher learns from (typed food names with the correct registry food code) are not part of CNR's "
                "screening data. They come from food registry search logs reviewed by a person and are uploaded as a dataset, then evaluated "
                "and registered under /v1/matching rather than trained as an experiment.")),
}


def parse_models(raw: str) -> list[str]:
    """'all' (every catalogue model) or comma-separated slugs. Unknown slugs are an error, not silently ignored."""
    known = [slug for slug, *_ in CATALOG]
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts or parts == ["all"]:
        return known
    out: list[str] = []
    for p in parts:
        if p not in SOURCES:
            raise SyncError(f"Unknown model '{p[:60]}' in TRAINING_SYNC_MODELS. Use 'all' or slugs from the catalogue.")
        if p not in out:
            out.append(p)
    return out


def config_problem(s: Settings) -> str | None:
    """None when the training sync can run, otherwise a short reason (shown to admins; never contains the key)."""
    if not s.training_sync_enabled:
        return "Automatic training-data sync is off (TRAINING_SYNC_ENABLED=false)."
    if len((s.surveillance_api_key or "").strip()) < 8:
        return "SURVEILLANCE_API_KEY (the CNR admin key, shared with the surveillance sync) is not set."
    bad = cnr_address_problem(s)
    if bad:
        return bad
    try:
        slugs = parse_models(s.training_sync_models)
    except SyncError as e:
        return str(e)
    if not any(SOURCES[x].population_group for x in slugs):
        return "None of the selected models has a CNR training source yet."
    if not 1 <= s.training_sync_months <= 36:
        return "TRAINING_SYNC_MONTHS must be from 1 to 36."
    return None


def export_url(s: Settings, src: Source) -> str:
    q = urllib.parse.urlencode({"population_group": src.population_group, "months": s.training_sync_months, "format": "csv"})
    return s.cnr_api_base_url.rstrip("/") + "/surveillance/training-export?" + q


def fetch_training_csv(s: Settings, src: Source) -> bytes:
    return get_csv(s, export_url(s, src), max_mb=s.training_fetch_max_mb)


def _header(raw: bytes) -> list[str]:
    first = raw.decode("utf-8-sig", "replace").splitlines()[0] if raw.strip() else ""
    return next(csv.reader([first]), [])


def _label_counts(raw: bytes, target: str) -> dict[str, int]:
    col = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, usecols=[target])[target].str.strip()
    counts = col.replace("", "(missing)").value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def _ready(rows: int, counts: dict[str, int]) -> tuple[bool, str | None]:
    """The same floor the training code enforces, so the admin sees before starting whether an experiment would be refused."""
    labelled = {k: v for k, v in counts.items() if k != "(missing)"}
    if sum(labelled.values()) < MIN_ROWS:
        return False, f"Fewer than {MIN_ROWS} labelled rows."
    if len(labelled) < 2:
        return False, "The label has only one class so far."
    small = [k for k, v in labelled.items() if v < MIN_PER_CLASS]
    if small:
        return False, f"Each class needs at least {MIN_PER_CLASS} examples (too few: {', '.join(sorted(small))})."
    return True, None


def _dataset(db, slug: str, src: Source) -> Dataset:
    name = DATASET_PREFIX + slug
    d = db.scalar(select(Dataset).where(Dataset.name == name))
    if d is None:
        d = Dataset(name=name, sensitivity="confidential", created_by=ACTOR, description=(
            f"De-identified, record-level training data for '{slug}', pulled automatically from CNR's training export "
            f"({src.population_group}). Suggested target: {src.target}. Suggested columns to leave out: {', '.join(src.exclude) or 'none'}. "
            f"{src.note}"))
        db.add(d)
        db.commit()
    return d


def sync_model(db, s: Settings, slug: str, fetch=None) -> dict:
    src = SOURCES[slug]
    res: dict = {"model_slug": slug}
    if src.population_group is None:
        return {**res, "status": "no_source", "note": src.reason}
    try:
        raw = (fetch or fetch_training_csv)(s, src)
        if _csv_rows(raw) == 0:
            return {**res, "status": "no_data", "note": "CNR returned no rows (no data yet, or every group is under the minimum size)."}
        header = _header(raw)
        if src.target not in header:
            raise SyncError(f"The export has no '{src.target}' column. Is the CNR API up to date (it needs GET /surveillance/training-export)?")
        if not [c for c in header if c != src.target and c not in src.exclude]:
            raise SyncError("The export has no feature columns left after the default exclusions.")
        d = _dataset(db, slug, src)
        res["dataset_id"] = d.id
        counts = _label_counts(raw, src.target)
        ready, why = _ready(sum(counts.values()), counts)
        res.update(label_counts=counts, ready_to_train=ready, **({"not_ready_reason": why} if why else {}))
        digest = ingest.sha256(raw)
        known = db.scalar(select(DatasetVersion).where(DatasetVersion.dataset_id == d.id, DatasetVersion.source_sha256 == digest))
        if known is not None:
            v, res["status"] = known, "unchanged"
        else:
            try:
                v = _store_version(db, d, raw, s.training_sync_approval_ref, actor=ACTOR, drop_duplicates=False,
                                   filename="cnr_training_export.csv")
            except IntegrityError:
                db.rollback()
                raise SyncError("Another upload took the version number; will retry at the next sync.")
            res["status"] = "new_version"
        res.update(version=v.version, rows=v.row_count)
        return res
    except SyncError as e:
        return {**res, "status": "error", "error": str(e)}
    except ingest.IngestError as e:
        return {**res, "status": "error", "error": "The export was rejected by the dataset checks: " + str(e)[:160]}
    except Exception:
        log.exception("Training sync failed for %s", slug)
        db.rollback()
        return {**res, "status": "error", "error": "Unexpected error. See the service logs."}


def run_once(s: Settings, trigger: str = "schedule", actor: str | None = None, fetch=None) -> dict | None:
    """One full sync of every selected model. Returns None when another training sync is already running."""
    if not _lock.acquire(blocking=False):
        return None
    try:
        bad = config_problem(s)
        LAST_RUN.clear()
        LAST_RUN.update(state="running", trigger=trigger, started_at=utcnow(), finished_at=None, results=[])
        if bad:
            LAST_RUN.update(state="not_configured", finished_at=utcnow(), error=bad)
            return dict(LAST_RUN)
        results: list[dict] = []
        with SessionLocal() as db:
            for slug in parse_models(s.training_sync_models):
                results.append(sync_model(db, s, slug, fetch))
                LAST_RUN["results"] = list(results)
            failed = sum(r["status"] == "error" for r in results)
            summary = {"trigger": trigger, "actor": actor or ACTOR, "models": len(results), "failed": failed,
                       "new_versions": sum(r["status"] == "new_version" for r in results),
                       "unchanged": sum(r["status"] == "unchanged" for r in results),
                       "no_data": sum(r["status"] == "no_data" for r in results),
                       "no_source": sum(r["status"] == "no_source" for r in results)}
            audit.record(db, "training_sync.completed", detail=summary)
        LAST_RUN.update(state="completed" if not failed else "completed_with_errors", finished_at=utcnow(), summary=summary)
        return dict(LAST_RUN)
    finally:
        _lock.release()


def source_info(slug: str) -> dict:
    """What a model can be fed with, for the catalogue and the status page. No names or values beyond the registry."""
    src = SOURCES.get(slug)
    if src is None:
        return {"available": False, "reason": "No source registered."}
    if src.population_group is None:
        return {"available": False, "reason": src.reason}
    return {"available": True, "population_group": src.population_group, "target": src.target, "exclude": list(src.exclude), "note": src.note}


def status(s: Settings, db=None) -> dict:
    bad = config_problem(s)
    try:
        selected = set(parse_models(s.training_sync_models))
    except SyncError:
        selected = set()
    last = {r["model_slug"]: r for r in (LAST_RUN.get("results") or [])}
    models = []
    for slug, name, *_ in CATALOG:
        row = {"slug": slug, "name": name, "selected": slug in selected, **source_info(slug)}
        if db is not None and SOURCES[slug].population_group:
            d = db.scalar(select(Dataset).where(Dataset.name == DATASET_PREFIX + slug))
            if d is not None:
                v = db.scalar(select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == d.id))
                row.update(dataset_id=d.id, dataset_name=d.name, latest_version=v)
        if slug in last:
            row["last_result"] = last[slug]
        models.append(row)
    return {"enabled": bool(s.training_sync_enabled), "configured": bad is None, "problem": bad,
            "source": (s.cnr_api_base_url.rstrip("/") + "/surveillance/training-export") if s.cnr_api_base_url else None,
            "interval_hours": s.training_sync_interval_hours, "months": s.training_sync_months,
            "running": _lock.locked(), "models": models, "last_run": dict(LAST_RUN) or None}


async def scheduler(s: Settings) -> None:
    """Background loop started with the app. Sleeps first so a restart does not hit CNR immediately."""
    interval = max(1.0, float(s.training_sync_interval_hours)) * 3600
    log.info("Training-data sync on: every %.1f h", interval / 3600)
    try:
        await asyncio.sleep(INITIAL_DELAY_S)
        while True:
            try:
                await asyncio.to_thread(run_once, s, "schedule")
            except Exception:
                log.exception("Scheduled training-data sync failed")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
