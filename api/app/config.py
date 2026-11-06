from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "sqlite:///./data/mlab.db"
    bootstrap_admin_key: str = ""
    cors_origins: str = "http://localhost:8080"
    cnr_api_base_url: str = ""
    # Outbound reachability check of the CNR API (fixed URL from the environment, never from a request)
    cnr_api_health_path: str = "/"
    cnr_api_timeout_s: float = 5.0
    # Lab -> CNR: signed model-status events. Off until both are set. The secret must be >= 32 chars and
    # equal to ML_LAB_WEBHOOK_SECRET on the CNR API.
    cnr_webhook_url: str = ""
    cnr_webhook_secret: str = ""
    # CNR -> lab: automatic pull of district-by-month surveillance counts (GET {CNR_API_BASE_URL}/surveillance/export).
    # Off until enabled AND the key is set. The key is an admin key of the CNR API; it is only ever sent to CNR_API_BASE_URL.
    surveillance_sync_enabled: bool = False
    surveillance_api_key: str = ""
    surveillance_sync_interval_hours: float = 24.0
    surveillance_sync_months: int = 12
    # "population_group:indicator" pairs, comma separated. One dataset (and one hotspot analysis) per pair.
    surveillance_sync_targets: str = "under5:stunting,under5:wasting,under5:acute_malnutrition"
    surveillance_sync_run_analysis: bool = True
    surveillance_sync_approval_ref: str = "CNR surveillance aggregate export (automatic)"
    surveillance_fetch_timeout_s: float = 60.0
    surveillance_fetch_max_mb: int = 5
    # CNR -> lab: automatic pull of record-level, de-identified TRAINING data for catalogue models
    # (GET {CNR_API_BASE_URL}/surveillance/training-export). Uses the same CNR admin key (SURVEILLANCE_API_KEY).
    # "all" syncs every model that has a CNR source; or list slugs, comma separated.
    training_sync_enabled: bool = False
    training_sync_models: str = "all"
    training_sync_interval_hours: float = 24.0
    training_sync_months: int = 12
    training_sync_approval_ref: str = "CNR de-identified training export (automatic)"
    training_fetch_max_mb: int = 20
    # LLM intelligence, phase 1 (contract and safety rules). The Chakudya address and consumer key live ONLY here, on the server:
    # they are never sent to the dashboard, returned by an endpoint, logged or audited. The key is a SecretStr so that even
    # printing the settings object shows nothing. Off until LLM_ENABLED=true AND the key is set.
    llm_enabled: bool = False
    chakudya_api_url: str = ""                  # empty -> CNR_API_BASE_URL
    chakudya_consumer_key: SecretStr = SecretStr("")
    # Opt-in: with no consumer key set, reuse the CNR admin key already held for the sync (SURVEILLANCE_API_KEY). It is a broader
    # key than a consumer key, so it is never used silently, and only for the CNR address it is already sent to (see llm_gateway).
    llm_use_surveillance_key: bool = False
    llm_timeout_s: float = 30.0
    llm_max_response_kb: int = 256
    # Text may leave the lab only after the NLP identifier detector has run. "reject" refuses identifiable text (default);
    # "mask" replaces what the detector finds with [EMAIL], [PHONE]... A request may choose either, never "off".
    llm_identifier_policy: str = "reject"
    llm_max_text_chars: int = 4000
    # Per API key (not per address). In-memory, so per process, like the other limiters.
    llm_rate_limit_per_min: int = 20
    llm_rate_limit_per_hour: int = 300
    service_version: str = "0.9.0"
    # Dataset ingestion
    dataset_storage_dir: str = "./data/datasets"
    pseudonym_key: str = ""          # >= 32 chars; enables the "pseudonymise" action
    max_upload_mb: int = 20
    max_rows: int = 200_000
    max_columns: int = 300
    # Governance. A model must be reviewed by a different key than the one that registered/submitted it.
    require_separate_reviewer: bool = True
    # Limits (per API key for predictions; per client address for repeated wrong keys)
    rate_limit_predict_per_min: int = 60
    auth_fail_limit: int = 20
    auth_fail_window_s: int = 300

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sqlalchemy_url(self) -> str:
        url = self.database_url
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg2://" + url[len(prefix):]
        return url

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
