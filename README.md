# CNR Machine Learning Laboratory

An independent machine-learning platform for the Chakudya Nutrition Registry (CNR): datasets, training, evaluation, a model registry and a prediction API, with governance built in.

**Status: Phase 1 of 6 — foundation and dashboard.** No datasets have been ingested, no model has been trained, and nothing here makes predictions yet. Predictions will never be presented as confirmed diagnoses.

## What Phase 1 delivers
- **ML API** (`api/`, FastAPI): API-key authentication, four roles, audit log, full metadata schema (datasets, versions, experiments, model versions), a catalogue of the seven planned model targets, `/health`.
- **Dashboard** (`dashboard/`, static HTML/JS, no build step): overview with build progress, models, experiments, datasets, audit log and key management. Pages show honest empty states until later phases fill them.
- **Deployment**: Dockerfiles, `docker-compose.yml`, `render.yaml`, CI, Supabase RLS script.
- It does not touch CNR's production API. The two will talk only through authenticated endpoints (Phase 6).

## Roles
| Role | Can |
|---|---|
| viewer | read catalogue, models, experiments, datasets |
| data_scientist | viewer + create datasets/experiments (Phase 2+) |
| clinical_reviewer | viewer + approve models (Phase 4+), read audit log |
| admin | everything, including keys and deployment |

## Run locally
```bash
cp .env.example .env           # set BOOTSTRAP_ADMIN_KEY (>= 32 chars)
docker compose up --build      # API :8000, dashboard :8080
# or without Docker:
cd api && pip install -r requirements-dev.txt && pytest -q
BOOTSTRAP_ADMIN_KEY=<key> uvicorn app.main:app --reload
```
Open the dashboard and sign in with the bootstrap key, then create per-person keys under **Access**.

## Deploy
See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Design and governance: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Roadmap
1. Foundation and dashboard (this release) 2. Dataset ingestion and preprocessing 3. First XGBoost model and evaluation 4. Model registry and versioning 5. Approved model served via FastAPI 6. Integration with the CNR API
