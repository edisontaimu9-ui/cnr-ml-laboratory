# Deployment

## 1. Supabase
Prefer a **new Supabase project** for the lab (keeps it independent of CNR). Copy the *pooler* connection string (Project → Connect). After the API's first start creates the tables, run `sql/enable_rls.sql` in the SQL editor. The script covers every `mlab_` table and is safe to run again: re-run it after an update that adds a table (for example `mlab_explorations`, created automatically at the next API start).

## 2. Render
1. New → Blueprint → select this repo (`render.yaml`).
2. Set the secrets on `cnr-ml-api`: `DATABASE_URL` (Supabase string) and `BOOTSTRAP_ADMIN_KEY` (`python -c "import secrets;print(secrets.token_urlsafe(48))"`).
3. Also set `PSEUDONYM_KEY` (another 32+ random characters; keep it stable, or pseudonyms from different versions will not match).
4. Deploy. Check `https://<service>.onrender.com/health` returns `{"status":"ok","database":"ok",...}`.
Training uses XGBoost and scikit-learn (the neural network and logistic regression options are scikit-learn's MLP and LogisticRegression, so they add no dependency; scikit-learn 1.7 or newer is required for class weights) (the Docker image installs `libgomp1` for XGBoost). Training runs inside the API process, so a 512 MB Starter instance can run out of memory on larger datasets; if the service restarts during a run, move to a larger instance. De-identified datasets are stored on a **persistent disk** (`mlab-data`, mounted at `/srv/data`, declared in `render.yaml`), which needs a paid instance. The container starts as root only to fix the disk's ownership and then runs as an unprivileged user. Back the disk up before relying on it.

The model registry needs no extra service or manual SQL. New columns are added to `mlab_model_versions` on the API's first start. Registered model files are stored on the same persistent disk, under `DATASET_STORAGE_DIR` in `models/` (next to `experiments/`), so include them in your disk backup. Before the first approval, create a **clinical_reviewer** key under Access for the reviewer; with `REQUIRE_SEPARATE_REVIEWER=true` (default) the admin key that registered a model cannot approve it.

Serving needs no extra service, disk or SQL. Predictions are served by the same API process from the model file on the persistent disk. Before connecting CNR, create a **service** key under Access for the CNR integration and store it only as a secret in the calling service. Keep `/docs` off in production (already the default). The first request after a deploy or restart loads the model, so it can be slower.

The CNR integration needs no extra service, disk or SQL. Steps: (1) create a **service** key under Access, named for the caller (e.g. `cnr-production`), and store it only as a secret in CNR; (2) confirm `CNR_API_BASE_URL` on the API (already in `render.yaml`) and, if CNR has a lightweight health route, set `CNR_API_HEALTH_PATH` to it; (3) open **CNR integration** in the dashboard and press *Check CNR API*. See [CNR_INTEGRATION.md](CNR_INTEGRATION.md) for the calling side. **Two-way link.** Generate one shared secret (`python -c "import secrets;print(secrets.token_urlsafe(48))"`). On Render set `CNR_WEBHOOK_SECRET` to it (`CNR_WEBHOOK_URL` is in `render.yaml`). On the CNR Worker run `npx wrangler secret put ML_LAB_WEBHOOK_SECRET` with the same value, and `npx wrangler secret put ML_LAB_KEY` with the service key; `ML_LAB_URL` is in `wrangler.toml`. Then deploy the Worker (`npx wrangler deploy`) and check `GET /ml/status` on CNR and the **CNR integration** page in the lab.

To rotate the key, create a new service key, update CNR's secret, then revoke the old one.

**Automatic surveillance sync.** To feed the hotspot analysis from CNR without uploading files: deploy the CNR API with `GET /surveillance/export` (`npx wrangler deploy`), then on Render set `SURVEILLANCE_API_KEY` to a CNR admin key (`SURVEILLANCE_SYNC_ENABLED` is already `true` in `render.yaml`; `CNR_API_BASE_URL` is the CNR address). Redeploy, open **CNR integration** and press *Sync now* under *Surveillance sync from CNR*. Targets, window and interval are `SURVEILLANCE_SYNC_TARGETS`, `SURVEILLANCE_SYNC_MONTHS` and `SURVEILLANCE_SYNC_INTERVAL_HOURS`.

**Automatic training-data sync (opt-in).** To fetch training data for models from CNR: run `sql/050_add_surveillance_training_rows.sql` in the CNR Supabase project (after 037), deploy the CNR API with `GET /surveillance/training-export`, then on Render set `TRAINING_SYNC_ENABLED=true` (it reuses `SURVEILLANCE_API_KEY`). Redeploy, open **CNR integration** and press *Sync now* under *Training data sync from CNR*. `TRAINING_SYNC_MODELS` (default `all`; models without a CNR source are reported, not fetched), `TRAINING_SYNC_MONTHS` and `TRAINING_SYNC_INTERVAL_HOURS` tune it. CNR's `SURVEILLANCE_MIN_CELL` sets the smallest group it will release.

**LLM intelligence (phases 1 to 3).** No new service, disk or SQL. `render.yaml` declares `LLM_ENABLED=false`, `CHAKUDYA_API_URL` and `LLM_IDENTIFIER_POLICY`, and `LLM_USE_SURVEILLANCE_KEY=true`, so the lab reuses the CNR admin key already set as `SURVEILLANCE_API_KEY` and needs no extra secret. To use a dedicated key instead, create a consumer key in the Chakudya developer console and set `CHAKUDYA_CONSUMER_KEY` (a secret; never commit it); it takes precedence. Nothing calls Chakudya until `LLM_ENABLED=true`; set it in Render to switch on Ask CNR (phase 2: `/v1/llm/ask`, `/v1/llm/retrieve` and the dashboard page). Until then those endpoints answer `503`. Check `GET /v1/llm/status` as admin: it shows the host name and whether a key exists, never the key. Rotate the key by creating a new consumer key, updating the secret and revoking the old one. Rate limits (`LLM_RATE_LIMIT_PER_MIN`, `LLM_RATE_LIMIT_PER_HOUR`) are per API key and per process. Phase 3 (RAG evaluation) adds no setting: its five `mlab_llm_eval_*` tables are created at start-up, and evaluation runs use the same Chakudya configuration. Each question in a run makes two calls to Chakudya and every call counts against the starting key's `LLM_RATE_LIMIT_PER_MIN` and `LLM_RATE_LIMIT_PER_HOUR`, so a 50-question run takes several minutes at the defaults; raise the limits only if Chakudya's own limits allow it.

## 3. Cloudflare DNS
Add CNAME records (proxied): `ml` → your API's `*.onrender.com` host, `ml-lab` → the dashboard's `*.onrender.com` host. Add the same custom domains in each Render service's Settings → Custom Domains.

## 4. First sign-in
Open https://ml-lab.chakudya.xyz, sign in with `BOOTSTRAP_ADMIN_KEY`, create a named admin key and role-specific keys under **Access**. Keep the bootstrap key as break-glass only; rotate it by changing the env var.

## Environment variables
See `.env.example`. Never commit `.env`.
