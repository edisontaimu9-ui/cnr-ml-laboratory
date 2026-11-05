# CNR Machine Learning Laboratory

An independent machine-learning platform for the Chakudya Nutrition Registry (CNR): datasets, training, evaluation, a model registry and a prediction API, with governance built in.

**Status: Phase 4 of 6 — model registry and review gate.** You can upload and de-identify datasets, run training experiments with honest hold-out evaluation, and register, review, approve and deploy-designate model versions. Nothing serves predictions yet; that is Phase 5. Predictions will never be presented as confirmed diagnoses.

## What Phase 1 delivers
- **ML API** (`api/`, FastAPI): API-key authentication, four roles, audit log, full metadata schema (datasets, versions, experiments, model versions), a catalogue of the seven planned model targets, `/health`.
- **Dashboard** (`dashboard/`, static HTML/JS, no build step): overview with build progress, models, experiments, datasets, audit log and key management. Pages show honest empty states until later phases fill them.
- **Deployment**: Dockerfiles, `docker-compose.yml`, `render.yaml`, CI, Supabase RLS script.
- It does not touch CNR's production API. The two will talk only through authenticated endpoints (Phase 6).

## What Phase 2 adds
- **Two-step upload** (`.csv` or `.xlsx`, 20 MB, 200k rows): *inspect* proposes a de-identification plan without storing anything; *create version* applies the reviewed plan and stores only the de-identified table.
- **Identifier detection** by column name and by value (emails, phone numbers, ID-like numbers, free text). Flagged columns are dropped by default and their values are never shown or logged. Direct identifiers can only be kept with a written reason; columns whose values contain personal data can never be kept.
- **Actions per column:** keep, drop, pseudonymise (keyed HMAC, stable within a column), year only (dates).
- **Cleaning:** header normalisation, missing-value tokens, numeric/date typing, empty-row and duplicate removal. No imputation or encoding here; that happens inside training (Phase 3) to avoid leakage.
- **Governance:** every version records a source hash, stored-file hash, approval reference, uploader, a de-identification report and a k-anonymity check on quasi-identifiers. Uploads, rejections and previews are audited.
- **Endpoints:** `POST /v1/datasets`, `POST /v1/datasets/inspect`, `GET /v1/datasets/{id}`, `POST /v1/datasets/{id}/versions`, `GET /v1/datasets/{id}/versions/{n}/preview`.

## What Phase 3 adds
- **Experiments:** pick a catalogue model, a dataset version, a target column and columns to leave out; the API trains an XGBoost model in the background and records the result.
- **Honest evaluation:** stratified hold-out split, cross-validation on the training part only, a trivial baseline next to every score, a bootstrap 95% range, per-class results, confusion matrix and feature importance. All preprocessing sits inside the pipeline, so nothing leaks from the test set.
- **Guards:** at least 50 rows and 10 examples per class; warnings for small datasets, a model that does not beat the baseline, and suspiciously strong results (likely label leakage, for example MUAC when predicting a nutrition status derived from it).
- **Governance:** a run never makes a model available. The first completed run only moves the catalogue entry from `planned` to `in_development`. The fitted pipeline is saved privately for Phase 4 and is never served.
- **Endpoints:** `POST /v1/experiments`, `GET /v1/experiments`, `GET /v1/experiments/{id}`. One experiment runs at a time.

## What Phase 4 adds
- **Registry:** a completed experiment can be registered as an immutable model version. The model file is copied, hashed (SHA-256) and the evaluation is frozen with it, together with the written intended use.
- **Review gate:** `candidate → in review → approved | rejected`. Approval needs a clinical reviewer, a reference to the external validation evidence, written review notes and, when the evaluation carries warnings, an explicit acknowledgement. Rejection is final for that version.
- **Four eyes:** the key that registered or submitted a version cannot approve or reject it (`REQUIRE_SEPARATE_REVIEWER`, on by default).
- **Deploy designation:** only an administrator can deploy an approved version. One version per model is deployed at a time; the previous one is retired as *superseded* and can be rolled back to. The stored file is re-hashed at deploy time and refused if it changed. Rolled-back and withdrawn versions can never be redeployed.
- **Audit and history:** every step is written to the audit log and to the version's own history. The catalogue status follows (`in_development` → `approved` → `deployed`).
- **Endpoints:** `POST /v1/models/versions`, `GET /v1/models/versions[/{id}]`, `POST /v1/models/versions/{id}/submit|approve|reject|deploy|retire`, `POST /v1/models/{slug}/rollback`.
- **Dashboard:** new **Registry** page with a version page for each model (evaluation, warnings, review form, actions, history), and a *Register* panel on completed experiments.
- "Deployed" only designates which version Phase 5 will serve. Registering, approving or deploying never exposes a prediction endpoint.

## Roles
| Role | Can |
|---|---|
| viewer | read catalogue, models, experiments, datasets |
| data_scientist | viewer + create datasets, upload versions, preview rows, create experiments, register and submit model versions |
| clinical_reviewer | viewer + approve or reject models in review, read audit log |
| admin | everything, including keys, deploy, roll back and withdraw (still cannot review their own registrations) |

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
1. Foundation and dashboard ✅ 2. Dataset ingestion and preprocessing ✅ 3. First XGBoost model and evaluation ✅ 4. Model registry and versioning ✅ 5. Approved model served via FastAPI 6. Integration with the CNR API
