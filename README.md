# CNR Machine Learning Laboratory

An independent machine-learning platform for the Chakudya Nutrition Registry (CNR): datasets, training, evaluation, a model registry and a prediction API, with governance built in.

You can upload and de-identify datasets, train and evaluate models, review and approve them, serve the deployed version, and let CNR request assessments through a dedicated, authenticated contract. Predictions are never presented as confirmed diagnoses.

## Platform
- **ML API** (`api/`, FastAPI): API-key authentication, four roles, audit log, full metadata schema (datasets, versions, experiments, model versions), a catalogue of the eighteen planned model targets, `/health`.
- **Dashboard** (`dashboard/`, static HTML/JS, no build step): installable as an app (PWA), with overview, models, experiments, exploration, hotspots, datasets, registry, prediction, CNR integration, audit log and key management. Pages show honest empty states until there is data.
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

## Hypertension risk model (`hypertension-risk`)
- **Purpose:** estimate the chance that someone has raised blood pressure from age, BMI, waist circumference, diet (salt, alcohol) and physical activity, for community screening where a cuff is not always available. A high estimate means *measure blood pressure*, never *has hypertension*.
- **Label:** from measured blood pressure (for example the average of repeated seated readings at or above 140/90 mmHg, or the threshold agreed with the data owner). The label note on the catalogue entry says so.
- **Leakage guard:** blood pressure readings and diagnosis or treatment columns (names containing systolic, diastolic, SBP, DBP, BP, HTN, hypertension, antihypertensive) are **refused as features** for this model: the run fails with a message naming the column. Exclude them next to the target and run again. Other models are unaffected.
- **Training:** the same experiment flow as every model (XGBoost, MLP or interpretable logistic regression with odds ratios and calibration). Logistic regression is a good first choice here because the odds ratios are readable by clinicians.

## Metabolic syndrome screen (`metabolic-syndrome-screen`)
- **Purpose:** screen for metabolic syndrome from cheap measurements (waist, BMI, age, diet) where blood tests are not available. A positive result means *send for blood tests*, never *has metabolic syndrome*.
- **Label:** the harmonised definition (IDF/AHA/NHLBI 2009, or the one agreed with the data owner): three or more of raised waist, raised triglycerides, low HDL, raised blood pressure, raised fasting glucose, or treatment for them.
- **Leakage guard:** triglycerides, HDL, glucose, blood pressure, diabetes, statin or lipid-treatment and syndrome columns are **refused as features** (the run fails and names the column). Exclude them next to the target.
- **Waist overlap:** waist circumference is both a criterion and the main cheap predictor. It is allowed, but every run that uses it carries a warning that scores overstate what the other features can do. Train once without waist and compare.

## Prediabetes / raised HbA1c finder (`prediabetes-screen`)
- **Purpose:** identify people most likely to need confirmatory testing when lab results are unavailable. A positive result means *arrange confirmatory testing*, never *has prediabetes*. It is a separate target from `diabetes-risk`.
- **Label:** measured HbA1c at or above 5.7% (ADA threshold; includes the diabetes range because the aim is who needs a confirmatory test), or the definition agreed with the data owner. People with known diabetes should be left out of the dataset or analysed separately.
- **Leakage guard:** HbA1c, glucose (fasting, random, OGTT), diabetes or prediabetes diagnosis and diabetes-treatment columns (metformin, insulin, sulfonylureas) are **refused as features**: the run fails and names the column. Exclude them next to the target.
- **Training:** the usual flow. Logistic regression gives readable odds ratios; the calibration check matters here because the output is used to decide who gets tested.

## Low birth weight risk (`low-birth-weight-risk`)
- **Purpose:** estimate the risk of a baby being born under 2,500 g from maternal and antenatal context, to prioritise extra antenatal and nutrition support. A high estimate means *offer extra support*, never *the baby will be small*.
- **Label:** measured birth weight under 2,500 g, taken within an hour of birth. Very low birth weight or small-for-gestational-age are different labels and must be agreed separately.
- **Leakage guard:** birth weight and anything derived from it or measured at or after birth (LBW/VLBW/SGA flags, birth length, head circumference, Apgar) are **refused as features**; the run fails and names the column.
- **Allowed with a warning:** gestational age at delivery and preterm birth are only known at delivery. They are allowed for a model used at birth, but every run that uses them warns that scores overstate antenatal performance. Train once without them and compare.

## Preterm infant growth faltering (`preterm-growth-faltering`)
- **Purpose:** estimate which preterm infants are likely to fall away from their growth trajectory on the Fenton reference, from early-life context, so feeding and growth can be reviewed closely. Never a diagnosis.
- **Label:** calculated from Fenton reference values by CNR or the Chakudya API's deterministic Fenton calculations, not re-implemented here: for example a fall in weight z-score of at least 1.0 from birth to discharge or 36 weeks postmenstrual age, or weight under the 10th percentile at that point. The data owner agrees the definition and time point first, and every row must use the same one.
- **Leakage guard:** discharge or follow-up weights, later z-scores or percentiles, z-score change, growth velocity, weight gain, and faltering/EUGR columns are **refused as features**. Birth weight, gestational age and birth z-score are allowed (they are known early). Birth z-score is expected to matter because a larger baby has further to fall.
- **Small samples:** neonatal-unit datasets are usually small; the usual small-dataset, baseline and confidence-range warnings apply and should be taken seriously.

## Adolescent / school-age overweight and stunting risk (`school-age-overweight-risk`, `school-age-stunting-risk`)
- **Purpose:** estimate which children and adolescents aged 5 to 19 are at risk of overweight or of stunting, from household, diet, activity and early-life context. Two separate models because the labels, and the measurements that give them away, differ. A positive result means *review growth and diet*, never a diagnosis.
- **Labels:** BMI-for-age z-score above +1 SD (overweight, includes obesity) or height-for-age z-score below -2 SD, on the WHO 2007 reference for 61 to 228 months. They are calculated by CNR or the Chakudya API's deterministic growth calculations, not re-implemented here. Younger children need the under-five tools.
- **Leakage guard:** the child's own weight, height, BMI, z-scores, percentiles and nutrition-status columns are **refused as features** for the overweight model; height, height stand-ins (knee height, ulna, demi-span, arm span), z-scores and stunting columns for the stunting model. Mother's and parents' measurements and birth weight are exempt (name contains mother, maternal, father, paternal, parent, caregiver, sibling or birth).
- **Allowed with a warning:** waist and similar body-size measures (overweight model); weight, BMI, MUAC, waist and head circumference (stunting model, only if height is not measured where the model is used).
- **Naming:** the upload step treats column names containing mother, father, guardian or caregiver as possible personal names and drops them unless kept with a reason. Name parent measurements `maternal_bmi`, `parental_height_cm` and so on.
- **Name matching:** column-name fragments of three characters or fewer (such as `bmi`) match whole words only, so a column like `submission_date` is not caught.

## Refeeding syndrome risk (`refeeding-syndrome-risk`)
- **Purpose:** estimate which patients are at risk of refeeding syndrome when nutrition support is started, from what is known before the first feed, so electrolytes are checked and feeding starts with care. A positive result means *check electrolytes and start feeding with care*, never *has refeeding syndrome*.
- **Label:** measured electrolytes after feeding is started or substantially increased, with one definition for every row (for example the ASPEN 2020 consensus criteria: a fall in phosphate, potassium or magnesium within 5 days of starting or increasing energy, graded by size of fall and organ dysfunction), or the one the data owner agrees. Adults and children (including severe acute malnutrition on F-75 or RUTF) use different definitions: one population per dataset.
- **Leakage guard:** values after feeding starts (day 1 to 7, 24/48/72 h, nadir, post-feed), electrolyte falls, electrolyte replacement or repletion, the diagnosis and its complications are **refused as features**.
- **Timing guard:** phosphate, potassium, magnesium and similar columns are allowed **only if the name says they were measured before feeding** (`baseline_`, `admission_`, `prefeeding_`, for example `baseline_phosphate`). A bare `phosphate` is refused with a message saying so, because the same measurement after feeding is the label.
- **Allowed with a warning:** how fast feeding was started (starting kcal/kg, feeding rate) and thiamine given in advance. High-risk patients are fed more slowly, so these hide the risk they respond to.
- **Labelling bias:** only patients whose electrolytes were rechecked have a label. Rows without a label are dropped and counted in the run warnings; they are mostly the patients clinicians judged low-risk.

## Malnutrition-related prolonged stay (`malnutrition-prolonged-stay-risk`)
- **Purpose:** estimate which admitted patients will stay longer than an agreed threshold, from nutrition status and context known at admission, to prioritise early nutrition support. A positive result means *review nutrition early*.
- **Label:** length of stay above a threshold fixed before training and applied to every row (for example more than 7 days), not a cut-off derived from this dataset. Decide how in-hospital deaths and transfers are handled; leaving them out is simplest.
- **Leakage guard:** length of stay and the days it is counted in, discharge date, weight, destination or diagnosis, readmission, death, and complications during the stay are **refused as features**. Short names such as `los` match whole words only.
- **Allowed with a warning:** nutrition care given during the stay (dietitian review, supplements, tube or parenteral feeding). A longer stay gives more time for it to happen, so it can look predictive only because it takes time.
- **Casemix:** stays differ by ward and diagnosis, so include them as features or train per ward.

## Malnutrition-related readmission (`malnutrition-readmission-risk`)
- **Purpose:** estimate which patients are at risk of unplanned readmission within 30 days of discharge, from nutrition status and context known at discharge, to target nutrition follow-up. A positive result means *arrange nutrition follow-up*.
- **Label:** unplanned readmission within 30 days of the discharge date, counted the same way for every row. The cohort is patients discharged alive. Readmission to another facility, or loss to follow-up, makes the label miss events, so say how follow-up was ascertained.
- **Leakage guard:** anything after discharge (readmission, return or attendance, follow-up weights or visits, death after discharge, emergency visits) is **refused as a feature**.
- **Different from the stay model on purpose:** length of stay, discharge weight and discharge nutrition status are known at discharge, so they are allowed here and refused for the stay model.
- **Allowed with a warning:** a supplement, therapeutic food or nutrition plan arranged at discharge, because it is decided by how much at risk the patient looks.

## Weight-loss trajectory (`weight-loss-trajectory`)
- **Purpose:** estimate which patients will lose a clinically significant amount of weight over a fixed follow-up window, from baseline context. A positive result means *review nutrition early and re-weigh soon*.
- **Label:** percentage weight change between a baseline weight and the end of a fixed window, the same for every row: for example 5% or more lost within 3 months (yes or no), or bands (under 5%, 5 to under 10%, 10% or more). The window and cut-offs are agreed with the data owner.
- **Leakage guard:** follow-up and final weights, weight change, velocity or slope, and loss or trajectory flags are **refused as features**.
- **Timing guard:** weight, BMI, MUAC, cachexia and similar columns are allowed **only if the name says when they were taken** (`baseline_`, `admission_`, `enrolment_`, `prior_`, `previous_`, for example `baseline_weight_kg` or `weight_loss_prior_6m_pct`). Loss before baseline is a legitimate feature; a bare `weight_3m_kg` is refused.
- **Allowed with a warning:** nutrition counselling, supplements or other care given during the window.
- **Attrition:** patients who die or are lost before the end weight have no label. Dropping them removes the fastest losers; the run reports how many were dropped.
- **Naming:** the platform's weight-change outcome model (`nutritional-outcome-prediction`, regression) is separate; this one is a classification of a fixed-window loss.

## Exploration (unsupervised analysis)
- **A separate type, not a model.** An *exploration* looks for groups of similar rows (**clustering**, k-means) and unusual rows (**anomaly detection**, isolation forest) in a de-identified dataset version. There is no target column and no catalogue model. It is for understanding data and checking data quality.
- **Never deployed, never sent to CNR.** Explorations live in their own table (`mlab_explorations`) with no link to model versions, and nothing is saved as a model file, so there is nothing to register, review, deploy, serve or announce. Every result carries `kind: "exploration"` and `deployable: false`, the registry and CNR integration code cannot see them, and the `service` role (CNR's machine key) cannot read them. Tests enforce the isolation, including a static check that the exploration code imports no registry, serving, predict, webhook or integration module.
- **Honest about structure:** k-means always returns groups, so every clustering result has a verdict (*clear*, *weak* or *no clear structure*) built from the silhouette score, stability across refits on 80% subsamples (adjusted Rand index) and a reference run on column-shuffled data. The number of groups is searched over a range and the full search is shown.
- **Honest about anomalies:** exactly the chosen share of rows is flagged by design, and the result says so. It reports how much the flagged set changes with the random seed, a score for every listed row, and why each stands out (standardised deviation per column, or a rare category). Reasons never contain raw values or category names; rows are identified by position in the stored dataset version.
- **Privacy:** group profiles are aggregates, and groups under 5 rows are not described.
- **Guards:** at least 50 rows and two usable columns; identifier-like and high-cardinality columns are skipped with a note; warnings for heavy missingness, tiny groups and sampling (large tables are fitted on a 20,000-row sample, every row is still assigned and scored). Only one training run or exploration runs at a time.
- **Cardiometabolic phenotypes (preset):** send `"preset": "cardiometabolic"` to cluster waist, BMI, blood pressure, glucose or HbA1c, lipids and diet/activity measurements into descriptive patterns. It is the same k-means exploration, not a catalogue model: no label, nothing saved as a model, never deployed or sent to CNR, and the `service` role cannot read it. The preset defaults to clustering only (anomaly detection stays opt-in) and adds a `preset` block to the result: which columns were recognised as which kind of measurement, and notes that the groups are not disease classes or a syndrome definition. It warns when a column looks like an assigned diagnosis, label, outcome or treatment (groups would just reproduce it), when one kind of measurement dominates the columns (k-means weights every column equally), when columns move almost together (rho 0.9 or more, e.g. weight and BMI), and when no cardiometabolic measurements are recognised. Unlike the prediction models, blood pressure, glucose and lipids are *wanted* here. The groups still carry the usual structure verdict (silhouette, stability, shuffled-data reference).
- **Endpoints:** `POST /v1/explorations`, `GET /v1/explorations`, `GET /v1/explorations/{id}`. Reading needs `read:explorations` (viewer, data scientist, clinical reviewer, admin); starting one needs `write:explorations` (data scientist, admin).
- **Dashboard:** an **Exploration** page marked *exploration · never deployed*, with the structure verdict, group profiles, flagged rows and a two-dimensional picture.

## District hotspots and prevalence trends (aggregated analysis)
- **A separate type, not a model.** A *hotspot analysis* works on **counts per area (and period)**, never on individual children: the number with the condition, the number examined, and optionally a month or year and a centre point per area. Areas can be districts, facilities or communities. It supports the hotspot dashboard; no individual-level data is needed or accepted.
- **Never deployed, never sent to CNR.** Same isolation as explorations: its own table (`mlab_hotspot_analyses`) with no link to model versions, written results only, no model file. Every result carries `kind: "hotspot_analysis"` and `deployable: false`; the `service` role cannot read it; tests enforce the isolation, including a static check that the code imports no registry, serving, predict, webhook or integration module.
- **Prevalence trend:** Poisson regression with area effects, a common time trend and a log(children examined) offset. If counts vary more than Poisson allows, a negative binomial model is fitted too and chosen on AIC; otherwise Poisson is kept with standard errors widened to match. The result gives the rate ratio per period (and per 12 months for monthly data) with a 95% range, plus a trend for each area (false discovery rate controlled) and a test of whether areas move differently.
- **Smoothing:** empirical-Bayes (Poisson-gamma) shrinkage pulls small, noisy areas towards the overall rate. With coordinates, each area is pulled towards its nearest neighbours instead (spatial smoothing). If the data show no more variation between areas than chance, every area is pooled to the overall rate, nothing is flagged, and the result says so.
- **Hotspots:** the probability that an area's true prevalence exceeds the overall prevalence times a chosen ratio (default 1.0). *Hotspot* needs at least 95% (configurable); *watch* at least 80%. Intervals are 95% ranges after smoothing.
- **Spatial clustering check** (needs coordinates and at least 8 areas): global Moran's I with a permutation test, and local Getis-Ord Gi* per area with false discovery rate control, on standardised differences from the overall rate. Areas are linked to their nearest neighbours by distance between centre points. With fewer than 15 areas the result warns that power is low.
- **Honest by design:** in simulations with nothing planted, the analysis flags no hotspots and few false trends or clusters (checked in tests); with four planted hotspots it finds all four. Areas expecting very few cases are marked low precision. Counts under 5, and groups leaving fewer than 5 children, are hidden (smoothed prevalence is still shown). Area names never go to the audit log.
- **Coordinates:** latitude and longitude columns are address-like, so upload removes them unless you keep them with a written reason (use area centre points, never households). Results carry centres rounded to about 1 km.
- **Endpoints:** `POST /v1/hotspots`, `GET /v1/hotspots`, `GET /v1/hotspots/{id}`. Reading needs `read:hotspots` (viewer, data scientist, clinical reviewer, admin); starting one needs `write:hotspots` (data scientist, admin). One training run, exploration or analysis runs at a time.
- **Dashboard:** a **Hotspots** page marked *aggregated analysis · never deployed*, with a schematic map of area centres, the hotspot table, the observed and fitted trend, the spatial verdict and a table of all areas.

## Automatic surveillance sync (CNR to laboratory)
- **What it does:** every `SURVEILLANCE_SYNC_INTERVAL_HOURS` (default 24) the API fetches CNR's `GET /surveillance/export` (district-by-month counts: examined, cases, severe cases) for each target in `SURVEILLANCE_SYNC_TARGETS`, stores the CSV as a new de-identified dataset version through the normal upload pipeline, and runs a hotspot analysis on it (`district`, `cases`, `examined`, `month`). One dataset per target, named `cnr-surveillance-<group>-<indicator>`.
- **Aggregates only:** CNR already leaves out cells under its minimum size. Datasets are marked `internal`; nothing individual is pulled.
- **Quiet when nothing changed:** an identical export (same SHA-256) makes no new version and no new analysis. If a run was skipped because a training run, exploration or analysis was busy, the analysis is done at the next sync.
- **Safe by design:** the CNR address comes only from `CNR_API_BASE_URL`; redirects are refused; the key (`SURVEILLANCE_API_KEY`, a CNR admin key) is sent only as a Bearer header to that address and never logged, audited or shown. Off until `SURVEILLANCE_SYNC_ENABLED=true` and the key is set.
- **Endpoints (admin):** `GET /v1/sync/surveillance` (settings, last run per target), `POST /v1/sync/surveillance/run` (sync now, 202). Audited as `surveillance_sync.started` and `surveillance_sync.completed`.
- **Dashboard:** a **Surveillance sync from CNR** section on the CNR integration page with the last run and a *Sync now* button.
- **Needs:** the CNR API with `GET /surveillance/export` deployed, and an always-on API service (the schedule runs inside the API; the first run is 2 minutes after start).

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
- CNR's own nutrition calculations stay in CNR. The lab pulls no patient records from CNR, only aggregated surveillance counts (see *Automatic surveillance sync*), and pushes it nothing except signed model-status events.

## Roles
| Role | Can |
|---|---|
| viewer | read catalogue, models, experiments, explorations, hotspot analyses, datasets (cannot call predictions) |
| data_scientist | viewer + create datasets, upload versions, preview rows, create experiments, explorations and hotspot analyses, register and submit model versions |
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
