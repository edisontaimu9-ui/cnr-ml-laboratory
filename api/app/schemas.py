from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer


def _iso(d: datetime | None) -> str | None:
    return d.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if d else None


UTC = Annotated[datetime | None, PlainSerializer(_iso, return_type=str | None)]
Role = Literal["admin", "data_scientist", "clinical_reviewer", "viewer"]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ModelDefinitionOut(ORM):
    slug: str
    name: str
    task_type: str
    description: str
    label_source: str
    status: str


class DatasetOut(ORM):
    id: int
    name: str
    description: str
    sensitivity: str
    created_by: str | None = None
    created_at: UTC
    version_count: int = 0


class DatasetCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    description: str = Field(default="", max_length=2000)
    sensitivity: Literal["restricted", "confidential", "internal"] = "restricted"


class DatasetVersionOut(ORM):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    id: int
    dataset_id: int
    version: int
    row_count: int
    column_count: int
    sha256: str
    source_sha256: str | None
    file_format: str | None
    deidentified: bool
    approval_ref: str | None
    created_by: str | None
    created_at: UTC
    schema_info: dict[str, Any] = Field(validation_alias="schema_json")
    report_json: dict[str, Any]


class DatasetDetail(DatasetOut):
    versions: list[DatasetVersionOut] = []


class ColumnAction(BaseModel):
    name: str = Field(max_length=200)
    action: Literal["keep", "drop", "pseudonymise", "year_only"]
    override_reason: str | None = Field(default=None, max_length=300)


class IngestPlan(BaseModel):
    columns: list[ColumnAction] = []
    drop_duplicates: bool = True


class ExperimentOut(ORM):
    id: int
    model_definition_id: int
    framework: str
    status: str
    metrics_json: dict[str, Any]
    created_by: str | None
    created_at: UTC


class ModelVersionOut(ORM):
    id: int
    model_definition_id: int
    version: int
    status: str
    approved_by: str | None
    created_at: UTC


class AuditOut(ORM):
    id: int
    ts: UTC
    actor_name: str | None
    role: str | None
    action: str
    resource_type: str | None
    resource_id: str | None
    detail: dict[str, Any]


class ApiKeyOut(ORM):
    id: int
    name: str
    role: str
    prefix: str
    created_by: str | None
    created_at: UTC
    last_used_at: UTC
    revoked_at: UTC


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    role: Role


class ApiKeyCreated(ApiKeyOut):
    key: str  # shown exactly once


class MeOut(BaseModel):
    name: str
    role: str
    permissions: list[str]
