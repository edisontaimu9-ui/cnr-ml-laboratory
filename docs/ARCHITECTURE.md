# Architecture and governance

## Principles
- **Independent of CNR production.** Separate repo, services and database. CNR calls the lab (and vice versa) only through authenticated HTTPS endpoints. Nutrition calculations (z-scores, BMI-for-age, etc.) stay in CNR; the lab consumes their outputs as features or labels and never re-implements them.
- **No fabricated data.** The lab ships with no datasets and no pre-trained models. A model is "available" only when it has a dataset version, an evaluation report and reviewer sign-off.
- **Decision support only.** Every prediction response must carry the model version, timestamp, risk indicators and the safety notice.

## Services
| Service | Host | Hosting |
|---|---|---|
| ML API | ml.chakudya.xyz | Render Web Service (Docker) |
| Dashboard | ml-lab.chakudya.xyz | Render Static Site |
| Metadata | — | Supabase Postgres (tables prefixed `mlab_`) |
| DNS / routing | — | Cloudflare |

## Data model
`mlab_api_keys`, `mlab_model_definitions`, `mlab_datasets`, `mlab_dataset_versions`, `mlab_experiments`, `mlab_explorations`, `mlab_hotspot_analyses`, `mlab_model_versions`, `mlab_audit_logs`.
Model artifacts and datasets are **not** stored in Postgres or git. Each de-identified dataset version as a gzip CSV on private disk (`DATASET_STORAGE_DIR`, a persistent volume in production) behind `app/storage.py`, which can be swapped for object storage without touching callers. `mlab_dataset_versions` gained columns over time; the API adds them to existing databases automatically at start-up (additive only, never destructive).

## Training and evaluation
- `POST /v1/experiments` validates the request, records an `experiments` row (`queued`) and starts training in a background thread of the API process. The row moves to `running`, then `completed` or `failed`; the dashboard polls it. Only one run is allowed at a time (the service is small), and runs interrupted by a restart are marked `failed` at start-up.
- `app/training.py` builds one sklearn `Pipeline`: numeric columns pass through (XGBoost handles missing values), categorical columns are filled and one-hot encoded, then `XGBClassifier` or `XGBRegressor`. With `algorithm: mlp` the pipeline instead imputes (median) and scales numeric columns, one-hot encodes categorical ones, and fits `MLPClassifier` or `MLPRegressor` (the regression target is standardised). With `algorithm: logistic` the pipeline imputes (median) and scales numeric columns, one-hot encodes categorical ones and fits an L2 `LogisticRegression`, optionally wrapped in `CalibratedClassifierCV(ensemble=False)` so the readable coefficients stay a single set (read from the fitted base model). All pipelines use the same step names (`pre`, `clf`), so registry and serving treat them alike. The test split is taken first; cross-validation and fitting use the training part only. Class imbalance is handled with balanced sample weights.
- Everything shown in the dashboard comes from `experiments.metrics_json`: test metrics, baseline, bootstrap range, CV mean and spread, per-class table, confusion matrix, top features, warnings, limitations. Failures store a user-safe message.
- The fitted pipeline is stored with `joblib` under `DATASET_STORAGE_DIR/experiments/` and never exposed over the API. The registry stores approved artifacts; only load files this service wrote itself.
- A completed experiment does not make a model available. Availability still requires the review and sign-off added by the registry and serving.

## Exploration (unsupervised analysis)
Explorations are a deliberately separate kind of work from experiments. They answer "what structure is in this data?" rather than "can we predict this label?", so they are never candidates for the registry.
- **Isolation by construction.** `mlab_explorations` has a foreign key to a dataset version only. It has no model definition, no label and no link to `mlab_model_versions`. `app/exploration.py` returns written results and no fitted object; no joblib file is written. The registry registers rows of `mlab_experiments` only, and serving and the CNR integration load only deployed model versions, so an exploration cannot reach either. The API output always states `kind: "exploration"` and `deployable: false`.
- **Enforced by tests.** `tests/test_explorations.py` checks that running explorations changes nothing the registry, catalogue, overview counts or CNR contract can see (and writes no `.joblib` file), that `exploration.py`, `jobs.py` and `routers/explorations.py` import no registry, serving, predict, webhook, integration, catalog, joblib or pickle code, and that the registry router does not mention explorations.
- **Access.** `read:explorations` for viewer, data scientist, clinical reviewer and admin; `write:explorations` for data scientist and admin. The `service` role (CNR) has neither.
- **Method.** One shared preprocessing step (median-imputed, standardised numbers; one-hot categories; identifier-like and high-cardinality columns skipped). Clustering: k-means over a searched range of k, chosen by silhouette, with Davies-Bouldin and Calinski-Harabasz, stability (adjusted Rand index over refits on 80% subsamples) and a reference silhouette from column-shuffled data, combined into a verdict. Anomaly detection: isolation forest, a fixed share of rows flagged by design, agreement between two seeds, and per-row reasons as standardised deviations or rare-category flags. A PCA projection of up to 600 rows is returned for plotting.
- **Scale.** Fitting, the k search, stability and the reference use at most 20,000 rows (sampled with the run's seed); all rows are assigned to groups and scored. Explorations run in a background thread like training, and `app/jobs.py` allows one heavy job at a time across both kinds. Runs interrupted by a restart are marked failed.
- **Privacy.** Group profiles are aggregates and are suppressed for groups under 5 rows. Flagged rows are identified by position in the stored dataset version and explained without raw values or category names. The audit log records the dataset, version and methods, and a result summary; never rows or values.

## Hotspots and prevalence trends (aggregated analysis)
A hotspot analysis answers "where is prevalence higher than expected, and how is it changing?" from counts per area and period. It is not a prediction model and is kept apart from the registry in the same way as explorations.
- **Isolation by construction.** `mlab_hotspot_analyses` has a foreign key to a dataset version only: no model definition, no label, no link to `mlab_model_versions`, no model file. `app/hotspot.py` returns written results and no fitted object. Results state `kind: "hotspot_analysis"` and `deployable: false`. `tests/test_hotspots.py` checks that running analyses changes nothing the registry, catalogue, overview model counts or CNR contract can see, that `hotspot.py` and `routers/hotspots.py` import no registry, serving, predict, webhook, integration, catalog, joblib or pickle code, and that the registry router does not mention hotspots.
- **Aggregated input only.** One row per area and period: cases, children examined, optional period and area centre point. Area names are never written to the audit log. Counts under 5 (or leaving fewer than 5 children in a group) are hidden in results; smoothed prevalence is still shown.
- **Methods** (`statsmodels`, `scipy`): Poisson GLM with area effects, a common linear trend and a log-denominator offset, with a negative binomial comparison when Pearson dispersion exceeds 1.5; per-area trends with Benjamini-Hochberg control and a Cochran Q test for different trends; Poisson-gamma empirical-Bayes smoothing (global, or towards nearest neighbours by haversine distance); exceedance probabilities from the gamma posterior; Moran's I (permutation test) and local Gi* (exact conditional sampling moments, Benjamini-Hochberg control) on standardised residuals. With no evidence of between-area differences (Poisson homogeneity test) all areas are pooled and none is flagged.
- **Access.** `read:hotspots` for viewer, data scientist, clinical reviewer and admin; `write:hotspots` for data scientist and admin. The `service` role (CNR) has neither. Runs happen in a background thread; `app/jobs.py` allows one heavy job at a time across training, explorations and hotspot analyses, and interrupted runs are marked failed on restart.

## Registry and review gate
- **Register** (`write:models`): only a `completed` experiment whose model file still exists, once. The file is copied to `models/<slug>/v<n>.joblib` under the private storage directory (immutable from then on) and its SHA-256 and size are recorded. The evaluation (`metrics_json`) is copied into the version so later edits or re-runs cannot change what was reviewed. Storage paths are never returned by the API. The registry never loads a model file; it only reads bytes.
- **States:** `candidate → in_review → approved | rejected`; `approved → deployed → retired`. Every other transition is refused with 409. `rejected` is final.
- **Approve** (`approve:models`): `clinical_validation_ref` (evidence reference, 5+ chars) and review notes (20+ chars) are mandatory. If the frozen evaluation holds warnings (small data, baseline not beaten, suspiciously strong result) the reviewer must set `acknowledge_warnings`. With `REQUIRE_SEPARATE_REVIEWER=true` (default) the reviewing key must differ from the key that registered and the key that submitted.
- **Deploy** (`deploy:models`, admin): from `approved` or from `retired` with reason *superseded*. Re-verifies the stored file against its recorded hash, retires the current deployed version of the same model (*superseded*) and records who and when. **Rollback** re-deploys the most recent superseded version and retires the current one as *rolled back*; a rolled-back or withdrawn version can never be redeployed. **Retire** withdraws a version with a written reason.
- **Catalogue status** is derived: `deployed` if any version is deployed, else `approved` if any is approved, else `in_development`.
- **"Deployed" is a designation, not serving.** The prediction endpoints may load only the deployed version, after re-checking its hash.
- Additive columns are added to `mlab_model_versions` automatically at start-up; no manual SQL migration is needed.

## Serving
- **Only the deployed version is served**, looked up on every request, so a retire or rollback applies immediately. With none deployed the API refuses (409) and audits `prediction.refused`.
- **Load path:** `serving.load_bundle` reads the artifact bytes, compares SHA-256 with the value recorded at registration and only then deserialises them. A mismatch returns 503 and is audited. The verified model is cached in memory per (version, hash); a restart re-verifies. This assumes the persistent disk is writable only by the API; protect it accordingly.
- **Input contract:** features are exactly those the model was trained on. Unknown fields are rejected so identifiers cannot be sent by mistake. Missing values are allowed (XGBoost or the pipeline's imputer handle them) but more than half missing, or a margin under 0.15, or a top score under 0.5, sets `low_confidence`. Unseen categories are treated as unknown and flagged.
- **Output contract:** never a diagnosis. `kind: risk_estimate`, `decision_support_only: true`, the safety notice, model/version/approval reference, caveats (including that the model was approved with warnings, if so) and a request ID. Class scores are not calibrated probabilities and the dashboard says so.
- **Privacy:** request bodies and results are not stored or logged. Audit detail: model, version, record count, low-confidence count, request ID.
- **Limits:** per-key prediction limit and a per-address wrong-key counter, both in process memory (fine for one instance; use shared storage if scaled out). The wrong-key limiter protects the audit log and database from flooding; the keys themselves are 256-bit random values.
- **Role `service`** is the machine identity used by the CNR integration.

## CNR integration
- **Direction and trust.** CNR is the caller. It holds a `service` key (read catalogue, predict, nothing else) as a secret and calls the lab over HTTPS. The lab never receives CNR credentials and sends CNR nothing but a reachability request.
- **Contract `cnr-ml-contract/1`.** `GET /v1/integration/cnr/models` (feature contract of each deployed model) and `POST /v1/integration/cnr/assess/{slug}` (cases of `case_ref` + `features`, max 50). Results come back in request order with the `case_ref` echoed. The version string changes only for breaking changes.
- **One code path.** Assessments call `routers/predict.run_prediction`, the same function as `/v1/predict`: per-key rate limit, deployed version only, SHA-256 check before loading, strict features, counts-only audit. The only differences are the audit action (`integration.assess`, with `channel: cnr` and an optional correlation ID) and the per-case envelope.
- **Privacy.** `case_ref` is an opaque reference that only CNR can map back to a person (1-64 chars of `A-Za-z0-9._:-`; anything that looks like an email or a 9+ digit number is rejected). Neither references nor features are stored or logged. Unknown feature names are rejected, so names or IDs cannot be sent by accident. A `case_ref` is not a substitute for de-identification: CNR must not derive it from a name, phone or national ID.
- **Outbound check.** `integration.probe` makes one GET to `CNR_API_BASE_URL` + `CNR_API_HEALTH_PATH`. The address comes only from the environment (no request can change it, so there is no server-side request forgery surface), must be https in production, may not embed credentials, sends no Authorization header or body and does not follow redirects. Any HTTP answer counts as reachable. It is admin-only, limited to 6 per minute and audited.
- **Status** (`read:integration`, admin): configuration problems, active service keys with last use, deployed models, 24-hour counts of assessments and refusals.
- **Lab -> CNR events.** `app/webhook.py` sends `model.deployed`, `model.rolled_back` and `model.retired` to `CNR_WEBHOOK_URL`, signed with HMAC-SHA256 over `<timestamp>.<body>` (secret `CNR_WEBHOOK_SECRET`, 32+ chars, https required in production, no embedded credentials, redirects not followed). Payload: slug, version, event ID, time, and a reason category only (never free text). Delivery is a background thread with three attempts and never blocks or fails a deploy; the outcome is audited as `integration.notify` and counted on the status page. Events are hints: the lab remains the authority through the 409 it returns when nothing is deployed.
- **Out of scope.** No bulk transfer of CNR records into the lab and no calls from the lab into CNR's data endpoints; that would need a separate de-identification and approval design. Treat these as open decisions with the data owner.
- **Known limits.** The rate limit is per process (single instance). There is no per-caller allow-list of IP addresses; the service key is the only credential, so keep it in CNR's secret store and rotate it by revoking and re-creating it under Access.

## Dataset ingestion
1. **Inspect** (`/v1/datasets/inspect`): the file is parsed in memory, typed, scanned for identifiers and profiled. Nothing is stored. Values of flagged columns are never returned.
2. **Create version** (`/v1/datasets/{id}/versions`): the file is parsed again, the reviewed plan is applied, and only the resulting table is stored. The raw upload is discarded. Re-uploading an identical file is rejected.
3. **Rules enforced server-side:** direct identifiers need a written reason to keep; columns whose values contain emails or phone numbers can never be kept; pseudonymisation uses an HMAC key from the environment; categorical counts under 5 are suppressed in profiles; a k-anonymity check is run over quasi-identifiers (sex, district, age, dates) and warned on when the smallest group is under 5.
4. **Limits of automated detection:** name- and pattern-based detection will miss identifiers it cannot recognise. A person must review each plan, and every upload needs an approval reference (ethics or data-sharing agreement).
5. Dates are read day-first for `dd/mm/yyyy` and as ISO for `yyyy-mm-dd`; the assumption is shown and recorded.

## Security
- Keys are 256-bit random values, stored only as SHA-256 hashes, shown once at creation, individually revocable.
- Denied requests, key changes and uploads, training runs and predictions are written to the audit log. Secrets and request bodies are never logged.
- `/docs` and `/openapi.json` are disabled when `ENVIRONMENT=production`. CORS is restricted to the dashboard origin.
- `read:dataset_rows` (data_scientist and admin only) allows row previews; viewers see only metadata. Audit detail for uploads holds counts, hashes and the approval reference, never rows, file names or cell values.
- Known limits: rate limits are per process, and the dashboard keeps the key in `sessionStorage` (cleared on tab close). Per-user login remains open.

## Deployment gate
Candidate → clinical review (external validation evidence recorded) → approved → deployed, with rollback to the previous approved version. Human oversight remains mandatory for any clinical decision.
