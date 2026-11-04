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

## Data model (created in Phase 1)
`mlab_api_keys`, `mlab_model_definitions`, `mlab_datasets`, `mlab_dataset_versions`, `mlab_experiments`, `mlab_model_versions`, `mlab_audit_logs`.
Model artifacts and raw datasets are **not** stored in Postgres or git; Phase 2/4 will add private object storage.

## Security
- Keys are 256-bit random values, stored only as SHA-256 hashes, shown once at creation, individually revocable.
- Denied requests, key changes and (from Phase 2) uploads, training runs and predictions are written to the audit log. Secrets and request bodies are never logged.
- `/docs` and `/openapi.json` are disabled when `ENVIRONMENT=production`. CORS is restricted to the dashboard origin.
- Phase 1 limits: no rate limiting yet, and the dashboard keeps the key in `sessionStorage` (cleared on tab close). Rate limiting and per-user login are candidates for Phase 4.

## Deployment gate (enforced from Phase 4)
Candidate → clinical review (external validation evidence recorded) → approved → deployed, with rollback to the previous approved version. Human oversight remains mandatory for any clinical decision.
