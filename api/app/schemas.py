from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, model_validator


def _iso(d: datetime | None) -> str | None:
    return d.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if d else None


UTC = Annotated[datetime | None, PlainSerializer(_iso, return_type=str | None)]
Role = Literal["admin", "data_scientist", "clinical_reviewer", "viewer", "service"]


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
    model_slug: str | None = None
    dataset_version_id: int | None = None
    framework: str
    status: str
    summary: dict[str, Any] = {}
    created_by: str | None
    created_at: UTC
    finished_at: UTC


class ExperimentDetail(ExperimentOut):
    model_name: str | None = None
    params_json: dict[str, Any] = {}
    metrics_json: dict[str, Any] = {}


class ExperimentCreate(BaseModel):
    model_slug: str = Field(max_length=80)
    dataset_id: int
    version: int | None = None
    target: str = Field(max_length=200)
    features: list[str] | None = Field(default=None, max_length=300)
    exclude: list[str] = Field(default_factory=list, max_length=300)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    n_estimators: int = Field(default=200, ge=10, le=1000)
    max_depth: int = Field(default=4, ge=1, le=10)
    learning_rate: float = Field(default=0.1, ge=0.01, le=0.5)
    balance_classes: bool = True
    algorithm: Literal["xgboost", "mlp", "logistic"] = "xgboost"
    # Neural network (mlp) settings. The XGBoost settings above are ignored for mlp runs and vice versa.
    hidden_layers: list[Annotated[int, Field(ge=2, le=256)]] = Field(default_factory=lambda: [64, 32], min_length=1, max_length=3)
    alpha: float = Field(default=1e-3, ge=1e-6, le=1.0)
    max_iter: int = Field(default=500, ge=50, le=1000)
    compare_xgboost: bool = True              # mlp and logistic
    # Logistic regression settings (classification only). max_iter and compare_xgboost above also apply.
    logistic_c: float = Field(default=1.0, ge=0.001, le=100.0)
    calibration: Literal["none", "sigmoid", "isotonic"] = "none"   # recalibration is opt-in: it was not better in testing


class ExplorationCreate(BaseModel):
    """Unsupervised analysis. There is no target and no model slug: an exploration is not tied to a catalogue model."""
    dataset_id: int
    version: int | None = None
    features: list[str] | None = Field(default=None, max_length=300)
    exclude: list[str] = Field(default_factory=list, max_length=300)
    methods: list[Literal["clustering", "anomaly"]] = Field(default_factory=lambda: ["clustering", "anomaly"], min_length=1, max_length=2)
    k: int | None = Field(default=None, ge=2, le=12)           # fix the number of groups instead of searching
    k_min: int = Field(default=2, ge=2, le=12)
    k_max: int = Field(default=8, ge=2, le=12)
    flag_share: float = Field(default=0.05, ge=0.005, le=0.2)  # share of rows flagged as most unusual, by design
    report_top: int = Field(default=20, ge=5, le=100)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    note: str = Field(default="", max_length=500)
    preset: Literal["cardiometabolic"] | None = None           # cardiometabolic phenotypes: clustering of measurements, never a model

    @model_validator(mode="after")
    def _check(self):
        if self.preset and "methods" not in self.model_fields_set:
            self.methods = ["clustering"]                       # phenotypes are about groups; anomaly detection stays opt-in
        if self.k_min > self.k_max:
            raise ValueError("k_min cannot be larger than k_max")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not repeat")
        return self


class ExplorationOut(ORM):
    id: int
    kind: str = "exploration"
    deployable: bool = False          # always false: explorations can never be registered, deployed or sent to CNR
    dataset_version_id: int | None = None
    status: str
    summary: dict[str, Any] = {}
    created_by: str | None
    created_at: UTC
    finished_at: UTC


class ExplorationDetail(ExplorationOut):
    params_json: dict[str, Any] = {}
    results_json: dict[str, Any] = {}


class HotspotCreate(BaseModel):
    """District hotspot / prevalence-trend analysis of aggregated counts. Not tied to a catalogue model."""
    dataset_id: int
    version: int | None = None
    area_column: str = Field(min_length=1, max_length=120)             # district, facility or community
    cases_column: str = Field(min_length=1, max_length=120)            # count of children with the condition
    denominator_column: str = Field(min_length=1, max_length=120)      # count of children examined
    period_column: str | None = Field(default=None, max_length=120)    # month / year; needed for trends
    latitude_column: str | None = Field(default=None, max_length=120)  # centre point of the area; needed for spatial parts
    longitude_column: str | None = Field(default=None, max_length=120)
    smoothing: Literal["auto", "global", "spatial"] = "auto"
    rr_threshold: float = Field(default=1.0, ge=0.5, le=5.0)           # hotspot = above overall prevalence x this ratio
    exceedance_cutoff: float = Field(default=0.95, gt=0.5, lt=1.0)
    k_neighbours: int = Field(default=4, ge=2, le=10)
    permutations: int = Field(default=999, ge=199, le=9999)
    min_periods_trend: int = Field(default=4, ge=3, le=60)
    report_top: int = Field(default=20, ge=5, le=100)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    note: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _check(self):
        if bool(self.latitude_column) != bool(self.longitude_column):
            raise ValueError("Give both latitude_column and longitude_column, or neither")
        roles = [c for c in (self.area_column, self.cases_column, self.denominator_column, self.period_column,
                             self.latitude_column, self.longitude_column) if c]
        if len(set(roles)) != len(roles):
            raise ValueError("Each role needs a different column")
        if self.smoothing == "spatial" and not self.latitude_column:
            raise ValueError("Spatial smoothing needs latitude_column and longitude_column")
        return self


class HotspotOut(ORM):
    id: int
    kind: str = "hotspot_analysis"
    deployable: bool = False          # always false: never registered, deployed or sent to CNR
    dataset_version_id: int | None = None
    status: str
    summary: dict[str, Any] = {}
    created_by: str | None
    created_at: UTC
    finished_at: UTC


class HotspotDetail(HotspotOut):
    params_json: dict[str, Any] = {}
    results_json: dict[str, Any] = {}


class ModelVersionOut(ORM):
    id: int
    model_definition_id: int
    model_slug: str | None = None
    model_name: str | None = None
    version: int
    status: str
    experiment_id: int | None = None
    dataset_version_id: int | None = None
    headline: dict[str, Any] = {}
    created_by: str | None = None
    created_at: UTC
    submitted_at: UTC = None
    reviewed_at: UTC = None
    deployed_at: UTC = None
    approved_by: str | None = None


class ModelVersionDetail(ModelVersionOut):
    intended_use: str | None = None
    clinical_validation_ref: str | None = None
    review_notes: str | None = None
    warnings_acknowledged: bool = False
    submitted_by: str | None = None
    deployed_by: str | None = None
    retired_at: UTC = None
    retired_reason: str | None = None
    artifact_sha256: str | None = None
    artifact_bytes: int | None = None
    metrics_json: dict[str, Any] = {}
    history: list[dict[str, Any]] = []


class RegisterVersion(BaseModel):
    experiment_id: int
    intended_use: str = Field(min_length=20, max_length=2000)


class SubmitVersion(BaseModel):
    note: str = Field(default="", max_length=1000)


class ApproveVersion(BaseModel):
    clinical_validation_ref: str = Field(min_length=5, max_length=500)
    review_notes: str = Field(min_length=20, max_length=3000)
    acknowledge_warnings: bool = False


class RejectVersion(BaseModel):
    review_notes: str = Field(min_length=10, max_length=3000)


class DeployVersion(BaseModel):
    note: str = Field(default="", max_length=1000)


class RetireVersion(BaseModel):
    reason: str = Field(min_length=10, max_length=300)


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
