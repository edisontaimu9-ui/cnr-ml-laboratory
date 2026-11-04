from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Dataset, Experiment, ModelDefinition, ModelVersion
from ..schemas import DatasetOut, ExperimentOut, ModelDefinitionOut, ModelVersionOut
from ..security import require

router = APIRouter(prefix="/v1", tags=["registry"])


@router.get("/catalog", response_model=list[ModelDefinitionOut],
            dependencies=[Depends(require("read:catalog"))])
def catalog(db: Session = Depends(get_db)):
    return db.scalars(select(ModelDefinition).order_by(ModelDefinition.id)).all()


@router.get("/catalog/{slug}", response_model=ModelDefinitionOut,
            dependencies=[Depends(require("read:catalog"))])
def catalog_item(slug: str, db: Session = Depends(get_db)):
    row = db.scalar(select(ModelDefinition).where(ModelDefinition.slug == slug))
    if row is None:
        raise HTTPException(404, "Unknown model definition.")
    return row


@router.get("/models/versions", response_model=list[ModelVersionOut],
            dependencies=[Depends(require("read:models"))])
def model_versions(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return db.scalars(select(ModelVersion).order_by(ModelVersion.id.desc()).limit(limit)).all()


@router.get("/experiments", response_model=list[ExperimentOut],
            dependencies=[Depends(require("read:experiments"))])
def experiments(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return db.scalars(select(Experiment).order_by(Experiment.id.desc()).limit(limit)).all()


@router.get("/datasets", response_model=list[DatasetOut],
            dependencies=[Depends(require("read:datasets"))])
def datasets(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    return db.scalars(select(Dataset).order_by(Dataset.id.desc()).limit(limit)).all()
