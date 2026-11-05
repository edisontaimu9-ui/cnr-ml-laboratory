# Deployment

## 1. Supabase
Prefer a **new Supabase project** for the lab (keeps it independent of CNR). Copy the *pooler* connection string (Project → Connect). After the API's first start creates the tables, run `sql/enable_rls.sql` in the SQL editor.

## 2. Render
1. New → Blueprint → select this repo (`render.yaml`).
2. Set the secrets on `cnr-ml-api`: `DATABASE_URL` (Supabase string) and `BOOTSTRAP_ADMIN_KEY` (`python -c "import secrets;print(secrets.token_urlsafe(48))"`).
3. Also set `PSEUDONYM_KEY` (another 32+ random characters; keep it stable, or pseudonyms from different versions will not match).
4. Deploy. Check `https://<service>.onrender.com/health` returns `{"status":"ok","database":"ok",...}`.
Phase 3 adds XGBoost and scikit-learn (the Docker image installs `libgomp1` for XGBoost). Training runs inside the API process, so a 512 MB Starter instance can run out of memory on larger datasets; if the service restarts during a run, move to a larger instance. Phase 2 stores de-identified datasets on a **persistent disk** (`mlab-data`, mounted at `/srv/data`, declared in `render.yaml`), which needs a paid instance. The container starts as root only to fix the disk's ownership and then runs as an unprivileged user. Back the disk up before relying on it.

Phase 4 needs no new service or manual SQL. New columns are added to `mlab_model_versions` on the API's first start. Registered model files are stored on the same persistent disk, under `DATASET_STORAGE_DIR` in `models/` (next to `experiments/`), so include them in your disk backup. Before the first approval, create a **clinical_reviewer** key under Access for the reviewer; with `REQUIRE_SEPARATE_REVIEWER=true` (default) the admin key that registered a model cannot approve it.

## 3. Cloudflare DNS
Add CNAME records (proxied): `ml` → your API's `*.onrender.com` host, `ml-lab` → the dashboard's `*.onrender.com` host. Add the same custom domains in each Render service's Settings → Custom Domains.

## 4. First sign-in
Open https://ml-lab.chakudya.xyz, sign in with `BOOTSTRAP_ADMIN_KEY`, create a named admin key and role-specific keys under **Access**. Keep the bootstrap key as break-glass only; rotate it by changing the env var.

## Environment variables
See `.env.example`. Never commit `.env`.
