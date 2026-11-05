"""The service is small, so only one heavy background job (training run, exploration or hotspot analysis) may run at a time."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Exploration, Experiment, HotspotAnalysis, utcnow

STALE_MINUTES = 30   # a job older than this is treated as dead (for example, lost in a restart)


def another_job_running(db: Session) -> bool:
    now = utcnow().replace(microsecond=0)
    for table in (Experiment, Exploration, HotspotAnalysis):
        for row in db.scalars(select(table).where(table.status.in_(["queued", "running"]))):
            if (now - row.created_at).total_seconds() < STALE_MINUTES * 60:
                return True
    return False
