"""Metadata schema. Tables are prefixed `mlab_` so they can never collide with
CNR tables if the same Postgres instance is shared. The full
skeleton is created at start-up; the write paths sit in the routers."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    # Stored as naive UTC (portable across SQLite/Postgres); serialised with "Z".
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ApiKey(Base):
    __tablename__ = "mlab_api_keys"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(32))
    prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ModelDefinition(Base):
    __tablename__ = "mlab_model_definitions"
    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    task_type: Mapped[str] = mapped_column(String(32))  # classification | regression
    description: Mapped[str] = mapped_column(Text)
    label_source: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="planned")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Dataset(Base):
    __tablename__ = "mlab_datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    sensitivity: Mapped[str] = mapped_column(String(32), default="restricted")
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class DatasetVersion(Base):
    __tablename__ = "mlab_dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("mlab_datasets.id"))
    version: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    schema_json: Mapped[dict] = mapped_column(JSON, default=dict)
    deidentified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # Dataset ingestion
    column_count: Mapped[int] = mapped_column(Integer, default=0)
    source_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_format: Mapped[str | None] = mapped_column(String(8), nullable=True)
    storage_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    approval_ref: Mapped[str | None] = mapped_column(String(300), nullable=True)
    report_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)

    def schema_info_cols(self) -> list[dict]:
        return (self.schema_json or {}).get("columns", [])


class Experiment(Base):
    __tablename__ = "mlab_experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_definition_id: Mapped[int] = mapped_column(ForeignKey("mlab_model_definitions.id"))
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("mlab_dataset_versions.id"), nullable=True)
    framework: Mapped[str] = mapped_column(String(40))
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Exploration(Base):
    """Unsupervised analysis (clustering, anomaly detection) of one dataset version.

    Deliberately NOT an Experiment: it has no model definition, no label and no stored model file, and it has no
    foreign key to `mlab_model_versions`. There is therefore nothing the registry could register, review, deploy
    or serve, and nothing CNR could ever be sent. Only the written results are kept."""
    __tablename__ = "mlab_explorations"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("mlab_dataset_versions.id"), nullable=True)
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    results_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class HotspotAnalysis(Base):
    """District hotspot and prevalence-trend analysis of AGGREGATED counts (one dataset version).

    Like `Exploration`, deliberately NOT an Experiment: no model definition, no label, no stored model file and no
    foreign key to `mlab_model_versions`, so the registry cannot pick it up and nothing can be served or sent to CNR.
    Only the written results are kept, and they hold area-level figures, never individual rows."""
    __tablename__ = "mlab_hotspot_analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int | None] = mapped_column(ForeignKey("mlab_dataset_versions.id"), nullable=True)
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    results_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ModelVersion(Base):
    """A registered, immutable copy of a trained model plus its review trail.
    Status: candidate -> in_review -> approved | rejected; approved -> deployed -> retired."""
    __tablename__ = "mlab_model_versions"
    __table_args__ = (UniqueConstraint("model_definition_id", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    model_definition_id: Mapped[int] = mapped_column(ForeignKey("mlab_model_definitions.id"))
    version: Mapped[int] = mapped_column(Integer)
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("mlab_experiments.id"), nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="candidate")
    approved_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    clinical_validation_ref: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # Registry
    dataset_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)      # frozen copy of the evaluation
    intended_use: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_by_key_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submitted_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    submitted_by_key_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reviewed_by_key_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    review_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    deployed_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    retired_reason: Mapped[str | None] = mapped_column(String(300), nullable=True)
    history_json: Mapped[list] = mapped_column(JSON, default=list)


class AuditLog(Base):
    __tablename__ = "mlab_audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    actor_key_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actor_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(80), index=True)
    resource_type: Mapped[str | None] = mapped_column(String(60), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(60), nullable=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)


class LabNote(Base):
    """Laboratory memory: a short written note (decision, finding, lesson, warning, question) attached to one
    thing in the lab, or to the lab in general. Notes are text written by people; the lab never writes row-level data
    into them. Notes are archived, never hard-deleted, and edits keep their earlier wording in `history_json`."""
    __tablename__ = "mlab_notes"
    id: Mapped[int] = mapped_column(primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(32), index=True)
    subject_id: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(24), default="note", index=True)
    title: Mapped[str] = mapped_column(String(160))
    body: Mapped[str] = mapped_column(Text)
    tags_json: Mapped[list] = mapped_column(JSON, default=list)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    history_json: Mapped[list] = mapped_column(JSON, default=list)


class NlpCorpus(Base):
    """A named collection of annotated text for training the NLP laboratory. Versions hold the validated, identifier-free text."""
    __tablename__ = "mlab_nlp_corpora"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class NlpCorpusVersion(Base):
    __tablename__ = "mlab_nlp_corpus_versions"
    __table_args__ = (UniqueConstraint("corpus_id", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    corpus_id: Mapped[int] = mapped_column(ForeignKey("mlab_nlp_corpora.id"))
    version: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    source_sha256: Mapped[str] = mapped_column(String(64))
    doc_count: Mapped[int] = mapped_column(Integer, default=0)
    entity_count: Mapped[int] = mapped_column(Integer, default=0)
    storage_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    approval_ref: Mapped[str | None] = mapped_column(String(300), nullable=True)
    report_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class NlpTrainingRun(Base):
    """A spaCy NER training run on one NLP corpus version (NLP lab stage 2).

    Like `Exploration`, deliberately NOT an Experiment and with no foreign key to `mlab_model_versions`: until stage 3 adds review and
    approval for NLP models, nothing here can be registered, deployed, served or sent to CNR. The model file is a private artifact."""
    __tablename__ = "mlab_nlp_training_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    corpus_version_id: Mapped[int] = mapped_column(ForeignKey("mlab_nlp_corpus_versions.id"))
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    results_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    trial: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
