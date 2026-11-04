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

## Data model (created in Phase 1, extended in Phase 2)
`mlab_api_keys`, `mlab_model_definitions`, `mlab_datasets`, `mlab_dataset_versions`, `mlab_experiments`, `mlab_model_versions`, `mlab_audit_logs`.
Model artifacts and datasets are **not** stored in Postgres or git. Phase 2 stores each de-identified dataset version as a gzip CSV on private disk (`DATASET_STORAGE_DIR`, a persistent volume in production) behind `app/storage.py`, which can be swapped for object storage without touching callers. `mlab_dataset_versions` gained columns in Phase 2; the API adds them to existing databases automatically at start-up (additive only, never destructive).

## Dataset ingestion (Phase 2)
1. **Inspect** (`/v1/datasets/inspect`): the file is parsed in memory, typed, scanned for identifiers and profiled. Nothing is stored. Values of flagged columns are never returned.
2. **Create version** (`/v1/datasets/{id}/versions`): the file is parsed again, the reviewed plan is applied, and only the resulting table is stored. The raw upload is discarded. Re-uploading an identical file is rejected.
3. **Rules enforced server-side:** direct identifiers need a written reason to keep; columns whose values contain emails or phone numbers can never be kept; pseudonymisation uses an HMAC key from the environment; categorical counts under 5 are suppressed in profiles; a k-anonymity check is run over quasi-identifiers (sex, district, age, dates) and warned on when the smallest group is under 5.
4. **Limits of automated detection:** name- and pattern-based detection will miss identifiers it cannot recognise. A person must review each plan, and every upload needs an approval reference (ethics or data-sharing agreement).
5. Dates are read day-first for `dd/mm/yyyy` and as ISO for `yyyy-mm-dd`; the assumption is shown and recorded.

## Security
- Keys are 256-bit random values, stored only as SHA-256 hashes, shown once at creation, individually revocable.
- Denied requests, key changes and (from Phase 2) uploads, training runs and predictions are written to the audit log. Secrets and request bodies are never logged.
- `/docs` and `/openapi.json` are disabled when `ENVIRONMENT=production`. CORS is restricted to the dashboard origin.
- Phase 2 adds `read:dataset_rows` (data_scientist and admin only) for row previews; viewers see only metadata. Audit detail for uploads holds counts, hashes and the approval reference, never rows, file names or cell values.
- Known limits: no rate limiting yet, and the dashboard keeps the key in `sessionStorage` (cleared on tab close). Rate limiting and per-user login are candidates for Phase 4.

## Deployment gate (enforced from Phase 4)
Candidate → clinical review (external validation evidence recorded) → approved → deployed, with rollback to the previous approved version. Human oversight remains mandatory for any clinical decision.
