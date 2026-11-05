# CNR Machine Learning Laboratory

An independent machine-learning platform for the Chakudya Nutrition Registry (CNR): datasets, training, evaluation, a model registry and a prediction API, with governance built in.

You can upload and de-identify datasets, train and evaluate models, review and approve them, serve the deployed version, and let CNR request assessments through a dedicated, authenticated contract. Predictions are never presented as confirmed diagnoses.

## Platform
- **ML API** (`api/`, FastAPI): API-key authentication, four roles, audit log, full metadata schema (datasets, versions, experiments, model versions), a catalogue of the seven planned model targets, `/health`.
- **Dashboard** (`dashboard/`, static HTML/JS, no build step): installable as an app (PWA), with overview, models, experiments, exploration, datasets, registry, prediction, CNR integration, audit log and key management. Pages show honest empty states until there is data.
- **Deployment**: Dockerfiles, `docker-compose.yml`, `render.yaml`, CI, Supabase RLS script.
- It does not touch CNR's production API. The two will talk only through authenticated endpoints (see CNR integration below).

## Datasets
- **Two-step upload** (`.csv` or `.xlsx`, 20 MB, 200k rows): *inspect* proposes a de-identification plan without storing anything; *create version* applies the reviewed plan and stores only the de-identified table.
- **Identifier detection** by column name and by value (emails, phone numbers, ID-like numbers, free text). Flagged columns are dropped by default and their values are never shown or logged. Direct identifiers can only be kept with a written reason; columns whose values contain personal data can never be kept.
- **Actions per column:** keep, drop, pseudonymise (keyed HMAC, stable within a column), year only (dates).
- **Cleaning:** header normalisation, missing-value tokens, numeric/date typing, empty-row and duplicate removal. No imputation or encoding here; that happens inside training to avoid leakage.
- **Governance:** every version records a source hash, stored-file hash, approval reference, uploader, a de-identification report and a k-anonymity check on quasi-identifiers. Uploads, rejections and previews are audited.
- **Endpoints:** `POST /v1/datasets`, `POST /v1/datasets/inspect`, `GET /v1/datasets/{id}`, `POST /v1/datasets/{id}/versions`, `GET /v1/datasets/{id}/versions/{n}/preview`.

## Experiments
- **Experiments:** pick a catalogue model, a dataset version, a target column and columns to leave out; the API trains the model in the background and records the result. Three algorithms are available: **XGBoost** (default), a **neural network (MLP)** and an interpretable **logistic regression**.
- **Honest evaluation:** stratified hold-out split, cross-validation on the training part only, a trivial baseline next to every score, a bootstrap 95% range, per-class results, confusion matrix and feature importance. All preprocessing sits inside the pipeline, so nothing leaks from the test set.
- **Neural network vs XGBoost:** an MLP run (`algorithm: mlp`, with `hidden_layers`, `alpha`, `max_iter`) also fits XGBoost with its default settings on the identical training rows, folds, class weights and hold-out rows (`compare_xgboost`, on by default). The result holds both sets of scores side by side, the gap with a paired-bootstrap 95% range, and a plain verdict: `mlp_better`, `xgboost_better` or `no_clear_difference`. The comparison is a lead to confirm, not proof, because choosing the winner on the same hold-out split flatters it. Feature importance for the MLP is by permutation on the hold-out set.
- **Logistic regression (interpretable):** `algorithm: logistic`, classification models only (a numeric target is refused with a 422). Settings: `logistic_c` (inverse L2 penalty), `calibration` (`none` default, `sigmoid`, `isotonic`), `max_iter`. Numeric columns are median-imputed and standardised inside the pipeline, so every coefficient is per standard deviation. The result carries a `coefficients` block: log-odds and, for a two-class target, odds ratios (also per one original unit), largest first; one set per class for more classes. Category levels seen in fewer than 10 training rows are never named. Feature importance is the size of the standardised coefficients.
- **Calibrated probabilities:** every logistic run gets a `calibration` block on the hold-out set: Brier score, log loss, expected calibration error and a reliability table. Recalibration (sigmoid or isotonic, cross-fitted on the training part only) is **opt-in**, and when it is on the result shows the scores before and after with a plain verdict. In testing, recalibration did not improve probabilities that logistic regression already produces well, so the default is off. Class balancing makes probabilities describe equal-sized classes; the run warns, and turning `balance_classes` off gives probabilities that match real frequencies.
- **Logistic regression as a baseline:** a logistic run also fits XGBoost on the identical split, folds, class weights and hold-out rows (`compare_xgboost`, on by default) with the same paired-bootstrap verdict (`logistic_better`, `xgboost_better`, `no_clear_difference`). When the simple model is not clearly worse, the result says so; that is the check on whether the heavier models are worth their lost explainability.
- **Guards:** at least 50 rows and 10 examples per class; warnings for small datasets, a model that does not beat the baseline, and suspiciously strong results (likely label leakage, for example MUAC when predicting a nutrition status derived from it).
- **Governance:** a run never makes a model available. The first completed run only moves the catalogue entry from `planned` to `in_development`. The fitted pipeline is saved privately for the registry and is never served.
- **Endpoints:** `POST /v1/experiments`, `GET /v1/experiments`, `GET /v1/experiments/{id}`. One experiment runs at a time.

## Exploration (unsupervised analysis)
- **A separate type, not a model.** An *exploration* looks for groups of similar rows (**clustering**, k-means) and unusual rows (**anomaly detection**, isolation forest) in a de-identified dataset version. There is no target column and no catalogue model. It is for understanding data and checking data quality.
- **Never deployed, never sent to CNR.** Explorations live in their own table (`mlab_explorations`) with no link to model versions, and nothing is saved as a model file, so there is nothing to register, review, deploy, serve or announce. Every result carries `kind: "exploration"` and `deployable: false`, the registry and CNR integration code cannot see them, and the `service` role (CNR's machine key) cannot read them. Tests enforce the isolation, including a static check that the exploration code imports no registry, serving, predict, webhook or integration module.
- **Honest about structure:** k-means always returns groups, so every clustering result has a verdict (*clear*, *weak* or *no clear structure*) built from the silhouette score, stability across refits on 80% subsamples (adjusted Rand index) and a reference run on column-shuffled data. The number of groups is searched over a range and the full search is shown.
- **Honest about anomalies:** exactly the chosen share of rows is flagged by design, and the result says so. It reports how much the flagged set changes with the random seed, a score for every listed row, and why each stands out (standardised deviation per column, or a rare category). Reasons never contain raw values or category names; rows are identified by position in the stored dataset version.
- **Privacy:** group profiles are aggregates, and groups under 5 rows are not described.
- **Guards:** at least 50 rows and two usable columns; identifier-like and high-cardinality columns are skipped with a note; warnings for heavy missingness, tiny groups and sampling (large tables are fitted on a 20,000-row sample, every row is still assigned and scored). Only one training run or exploration runs at a time.
- **Endpoints:** `POST /v1/explorations`, `GET /v1/explorations`, `GET /v1/explorations/{id}`. Reading needs `read:explorations` (viewer, data scientist, clinical reviewer, admin); starting one needs `write:explorations` (data scientist, admin).
- **Dashboard:** an **Exploration** page marked *exploration · never deployed*, with the structure verdict, group profiles, flagged rows and a two-dimensional picture.

## Model registry
- **Registry:** a completed experiment can be registered as an immutable model version. The model file is copied, hashed (SHA-256) and the evaluation is frozen with it, together with the written intended use.
- **Review gate:** `candidate → in review → approved | rejected`. Approval needs a clinical reviewer, a reference to the external validation evidence, written review notes and, when the evaluation carries warnings, an explicit acknowledgement. Rejection is final for that version.
- **Four eyes:** the key that registered or submitted a version cannot approve or reject it (`REQUIRE_SEPARATE_REVIEWER`, on by default).
- **Deploy designation:** only an administrator can deploy an approved version. One version per model is deployed at a time; the previous one is retired as *superseded* and can be rolled back to. The stored file is re-hashed at deploy time and refused if it changed. Rolled-back and withdrawn versions can never be redeployed.
- **Audit and history:** every step is written to the audit log and to the version's own history. The catalogue status follows (`in_development` → `approved` → `deployed`).
- **Endpoints:** `POST /v1/models/versions`, `GET /v1/models/versions[/{id}]`, `POST /v1/models/versions/{id}/submit|approve|reject|deploy|retire`, `POST /v1/models/{slug}/rollback`.
- **Dashboard:** new **Registry** page with a version page for each model (evaluation, warnings, review form, actions, history), and a *Register* panel on completed experiments.
- "Deployed" only designates which version is served. Registering, approving or deploying never exposes a prediction endpoint.

## Serving predictions
- **Serving:** `POST /v1/predict/{slug}` answers only from the *deployed* version of a model. No deployed version means a clear 409 refusal, never a guess. Retiring or rolling back takes effect on the next request.
- **Integrity:** the model file is hashed against the SHA-256 recorded at registration before it is ever loaded; a changed file pauses predictions (503) and is audited.
- **Every response** carries the model name and version, approval and validation reference, intended use, evaluation headline, timestamp, request ID, a caveats list, `decision_support_only: true` and the safety notice. Results are called *risk estimates*, with per-class scores, a margin and a low-confidence flag; sparse or unfamiliar input lowers confidence and says so.
- **Explanations (logistic regression models):** each result also carries an `explanation`: the features that pushed that estimate up or down (value times coefficient on the standardised inputs, summed over a column's category levels), toward the predicted class, plus the features that were not given and so not scored. Raw input values are never echoed back. The CNR assessment endpoint passes it through. Other model types have no `explanation`.
- **Strict input:** only the model's own features are accepted (unknown fields such as names are rejected), numbers must be numbers, up to 50 records per call. Inputs and outputs are not stored; the audit log records counts, model version and request ID only.
- **Limits:** 60 prediction requests per key per minute (`RATE_LIMIT_PREDICT_PER_MIN`), and repeated wrong-key attempts from one address are throttled (`AUTH_FAIL_LIMIT`) so they cannot flood the audit log. Valid keys are never blocked by the second limit.
- **New role `service`:** can read the catalogue and call predictions, nothing else. Intended for the CNR integration.
- **Endpoints:** `GET /v1/predict`, `GET /v1/predict/{slug}` (features and known categories), `POST /v1/predict/{slug}`.
- **Dashboard:** a **Predict** page that builds the entry form from the deployed model.

## CNR integration
- **CNR contract (`cnr-ml-contract/1`):** `GET /v1/integration/cnr/models` lists deployed models with the exact features CNR must send. `POST /v1/integration/cnr/assess/{slug}` takes up to 50 cases, each an opaque `case_ref` plus `features`, and returns one result per case with the `case_ref` echoed back, the model version, approval reference, caveats and the safety notice. An optional `X-Request-Id` header is echoed as `correlation_id`.
- **Same safety rules as predictions:** it runs through the same code as `/v1/predict`: deployed version only (409 otherwise), hash check, strict features, the per-key rate limit. Case references that look like emails or phone/ID numbers are rejected. Case references and features are never stored; the audit log (`integration.assess`) holds counts, model version, request ID and correlation ID only.
- **Admin status and reachability:** `GET /v1/integration/cnr/status` shows whether the CNR address is configured, the active service keys and their last use, deployed models and 24-hour usage. `?probe=true` makes one outbound check of `CNR_API_BASE_URL` (no credentials or data sent, redirects not followed, 6 per minute, audited).
- **Dashboard:** new **CNR integration** page (admin).
- **Docs:** [docs/CNR_INTEGRATION.md](docs/CNR_INTEGRATION.md) has the contract and a worked example for the calling side.
- **Lab -> CNR events:** when a model is deployed, rolled back or withdrawn, the lab sends CNR a signed webhook (`CNR_WEBHOOK_URL`, `CNR_WEBHOOK_SECRET`; HMAC-SHA256, three attempts, audited as `integration.notify`, no patient data). The CNR side is `chakudya-api` v1.83.0: `GET /ml/status`, `GET /ml/models`, `POST /ml/assess/{slug}` and `POST /hooks/ml-lab`.
- CNR's own nutrition calculations stay in CNR. The lab neither pulls patient records from CNR nor pushes anything to it.

## Roles
| Role | Can |
|---|---|
| viewer | read catalogue, models, experiments, explorations, datasets (cannot call predictions) |
| data_scientist | viewer + create datasets, upload versions, preview rows, create experiments and explorations, register and submit model versions |
| clinical_reviewer | viewer + approve or reject models in review, read audit log |
| service | read catalogue, call predictions and CNR assessments only (machine key for CNR) |
| admin | everything, including keys, deploy, roll back, withdraw and the CNR integration status (still cannot review their own registrations) |

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
