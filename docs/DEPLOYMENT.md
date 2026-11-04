# Deployment

## 1. Supabase
Prefer a **new Supabase project** for the lab (keeps it independent of CNR). Copy the *pooler* connection string (Project → Connect). After the API's first start creates the tables, run `sql/enable_rls.sql` in the SQL editor.

## 2. Render
1. New → Blueprint → select this repo (`render.yaml`).
2. Set the secrets on `cnr-ml-api`: `DATABASE_URL` (Supabase string) and `BOOTSTRAP_ADMIN_KEY` (`python -c "import secrets;print(secrets.token_urlsafe(48))"`).
3. Deploy. Check `https://<service>.onrender.com/health` returns `{"status":"ok","database":"ok",...}`.
Note: Render's free web services sleep when idle; use a paid instance for persistent serving once models are live.

## 3. Cloudflare DNS
Add CNAME records (proxied): `ml` → your API's `*.onrender.com` host, `ml-lab` → the dashboard's `*.onrender.com` host. Add the same custom domains in each Render service's Settings → Custom Domains.

## 4. First sign-in
Open https://ml-lab.chakudya.xyz, sign in with `BOOTSTRAP_ADMIN_KEY`, create a named admin key and role-specific keys under **Access**. Keep the bootstrap key as break-glass only; rotate it by changing the env var.

## Environment variables
See `.env.example`. Never commit `.env`.
