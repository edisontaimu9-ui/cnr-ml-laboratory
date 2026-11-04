from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "sqlite:///./data/mlab.db"
    bootstrap_admin_key: str = ""
    cors_origins: str = "http://localhost:8080"
    cnr_api_base_url: str = ""
    service_version: str = "0.2.0"
    # Phase 2: dataset ingestion
    dataset_storage_dir: str = "./data/datasets"
    pseudonym_key: str = ""          # >= 32 chars; enables the "pseudonymise" action
    max_upload_mb: int = 20
    max_rows: int = 200_000
    max_columns: int = 300

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
