"""Food-name matching evaluation.

Like explorations, this router deliberately imports nothing from the registry, serving, predict, webhook or
integration code and never writes a model file. Results are returned and audited as counts only; nothing is
stored, so nothing can be registered, reviewed, deployed, served or announced to CNR.
`tests/test_matching.py` enforces this."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import audit, matching, storage
from ..db import get_db
from ..jobs import another_job_running
from ..models import DatasetVersion
from ..schemas import MatchingEvaluate
from ..security import Principal, require

log = logging.getLogger("cnr-ml")
router = APIRouter(prefix="/v1/matching", tags=["matching"])

MAX_QUERY_ROWS = 50_000     # the evaluation runs inside the request, so it is capped
MAX_REGISTRY_ROWS = 20_000


def _version(db: Session, dataset_id: int, version: int | None, what: str, columns: list[str | None]) -> DatasetVersion:
    q = select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id)
    q = q.where(DatasetVersion.version == version) if version else q.order_by(DatasetVersion.version.desc())
    v = db.scalars(q.limit(1)).first()
    if v is None or not v.storage_uri:
        raise HTTPException(404, f"Unknown {what} dataset version.")
    have = {c["name"] for c in v.schema_info_cols()}
    for c in columns:
        if c and c not in have:
            raise HTTPException(422, f"Column not in the {what} dataset version: {c[:80]}")
    return v


@router.post("/evaluations", status_code=200)
def evaluate(body: MatchingEvaluate, request: Request, p: Principal = Depends(require("write:experiments")),
             db: Session = Depends(get_db)):
    vq = _version(db, body.queries_dataset_id, body.queries_version, "labelled queries",
                  [body.query_column, body.food_code_column, body.group_column])
    vr = _version(db, body.registry_dataset_id, body.registry_version, "registry",
                  [body.registry_code_column, body.registry_label_column, body.registry_aliases_column])
    if vq.row_count > MAX_QUERY_ROWS or vr.row_count > MAX_REGISTRY_ROWS:
        raise HTTPException(422, f"Too large for an in-request evaluation (limits: {MAX_QUERY_ROWS} labelled queries, "
                                 f"{MAX_REGISTRY_ROWS} registry foods). Use a subset.")
    if another_job_running(db):
        raise HTTPException(409, "Another experiment or exploration is still running. Wait for it to finish.")
    cfg = matching.Config(query_col=body.query_column, food_col=body.food_code_column,
                          registry_id_col=body.registry_code_column, registry_name_col=body.registry_label_column,
                          registry_alias_col=body.registry_aliases_column, group_col=body.group_column,
                          test_size=body.test_size, seed=body.seed, top_k=body.top_k, target_accuracy=body.target_accuracy)
    try:
        res = matching.run(storage.load(vq.storage_uri), storage.load(vr.storage_uri), cfg)
    except matching.MatchingError as err:
        raise HTTPException(422, str(err))
    except Exception:
        log.exception("Matching evaluation failed")
        raise HTTPException(500, "Unexpected error during the evaluation. See the service logs.")
    audit.record(db, "matching.evaluated", principal=p, request=request, resource_type="dataset", resource_id=vq.dataset_id,
                 detail={"queries_version": vq.version, "registry_dataset": vr.dataset_id, "registry_version": vr.version,
                         "n_test": res["n_test"], "learned_top1": res["learned"]["top1"],
                         "verdict": res["learning_effect"]["verdict"]})
    return {"kind": "matching_evaluation", "deployable": False, "queries_dataset_version_id": vq.id,
            "registry_dataset_version_id": vr.id, "results": res}
