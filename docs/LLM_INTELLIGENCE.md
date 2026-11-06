# LLM intelligence roadmap

Six sequential phases. Each is implemented, tested, documented, pushed and **approved before the next one starts**.

| Phase | Scope | Status |
|---|---|---|
| 1 | Contract and safety rules | Implemented, awaiting approval |
| 2 | Ask CNR: RAG and natural-language interaction (`/v1/llm/ask`, `/v1/llm/retrieve`, chat page) | Not started |
| 3 | RAG evaluation laboratory (gold-standard questions, hit rate, citation validity, groundedness, refusals, latency, reviewer ratings) | Not started |
| 4 | Clinical document summarisation (de-identified notes, extracted facts only, number and entity validation, clinician review) | Not started |
| 5 | Open-weight model comparison (same gold set, prompts and retrieval configurations, registry entries with review) | Not started |
| 6 | Fine-tuning, gated (reviewed instruction data, JSONL export, external LoRA training, deploy only if it beats base model plus RAG on held-out data) | Not started |

## Phase 1: contract and safety rules

Contract name: `cnr-llm-contract/1`. Policy: `cnr-llm-policy/1`. Nothing in this phase calls a language model or sends any text to Chakudya.
It provides the rules and the transport that phases 2 to 6 must go through.

### What is enforced, and where

| Rule | Code | How it is enforced |
|---|---|---|
| Chakudya URL and consumer key are server-side only | `llm_gateway.py`, `config.py` | Read from the environment only (`CHAKUDYA_API_URL`, falling back to `CNR_API_BASE_URL`, and `CHAKUDYA_CONSUMER_KEY`). The key (a consumer key, or the CNR admin key if opted in) is read in one module, sent only as a Bearer header to the configured host, never returned, logged, audited or put in an error message. Redirects are refused. Only `/rag/ask` and `/rag/retrieve` can be called. No request field can change the address. |
| Never diagnose | `llm_policy.py` | Questions asking the service to diagnose a person, or to prescribe or dose for a patient, are refused (`screen_request`). Answers that state a diagnosis or prescription about a person are replaced by a fixed refusal text (`enforce_output`). `SYSTEM_RULES` is the text later phases put in front of any model. Criteria, guideline text and documented findings stay allowed. |
| No identifiable text leaves the lab | `llm_policy.prepare_for_external` | Runs the NLP identifier detector (`nlp_lab.screen_identifiers`, the same detector and overlap rule as `/v1/nlp/analyze`). Identifiable text is refused (default) or masked; masked text is checked again. Text is first folded (full-width characters) and stripped of zero-width and control characters, so identifiers cannot hide behind look-alikes. There is no mode that skips the check. |
| Audit holds counts and metadata only | `llm_audit.py` | An allow-list: integers, booleans, short fixed codes and identifier-type counts. Anything else is dropped, and only the field name is logged. |
| Rate limits per API key | `llm_limits.py`, `routers/llm.py` | Per-minute and per-hour windows keyed by API key. Refused requests count too. Failed authentication does not use a key's allowance. One `llm.rate_limited` audit row per key per minute. |

### Endpoints

- `GET /v1/llm/contract` (`read:llm`: viewer, data scientist, clinical reviewer, registry owner, admin; not `service`): the rules, limitations, refusal categories, limits and identifier policy.
- `GET /v1/llm/status` (`read:integration`, admin): whether LLM features are enabled and configured, the Chakudya **host name** and whether a key exists. Never the key or a path.
- `POST /v1/llm/precheck` (`use:llm`: data scientist, clinical reviewer, admin): runs the same gate every later endpoint will run, calls nothing external and stores nothing.

`POST /v1/llm/precheck`
```json
{"text": "What is the MUAC cut-off for SAM?", "kind": "question", "identifiers": "reject"}
```
- `kind`: `question` (default) applies the diagnosis and prescribing screen as well as the identifier gate; `document` (for a clinical note) applies the identifier gate only, because a note may legitimately say "diagnosed with".
- `identifiers`: `reject` or `mask`; omitted means `LLM_IDENTIFIER_POLICY` (default `reject`). There is no `off`.
- 200: `decision` is `allowed` or `masked` (with `masked_text`), plus `identifiers` counts, `warnings`, `decision_support_only: true`, `stored: false`. Headers `X-RateLimit-Remaining-Minute` and `-Hour`.
- 422: `decision: "refused"` with `category` (`diagnosis_request`, `prescribing_request`, `identifiers_found`, `empty`, `too_long`). The text is never echoed.
- 429: rate limit, with `Retry-After`.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_ENABLED` | `false` | Master switch for calls to Chakudya (phase 2 onward). |
| `CHAKUDYA_API_URL` | empty | Chakudya address; empty falls back to `CNR_API_BASE_URL`. https required in production. |
| `CHAKUDYA_CONSUMER_KEY` | empty | Optional dedicated Chakudya consumer key (at least 16 characters). Secret: set it in the Render dashboard, never in the repo. Wins over the admin key when both exist. |
| `LLM_USE_SURVEILLANCE_KEY` | `false` (`true` in `render.yaml`) | When no consumer key is set, reuse the CNR admin key already held as `SURVEILLANCE_API_KEY`. It is broader than a consumer key, so it is opt-in, sent only to the `CNR_API_BASE_URL` host (a different `CHAKUDYA_API_URL` host is refused), and the gateway can still call only `/rag/ask` and `/rag/retrieve` with it. `GET /v1/llm/status` shows `key_source` and `uses_admin_key`, never the key. |
| `LLM_IDENTIFIER_POLICY` | `reject` | Default for identifiable text: `reject` or `mask`. An invalid value falls back to `reject`. |
| `LLM_MAX_TEXT_CHARS` | `4000` | Longest text accepted. |
| `LLM_RATE_LIMIT_PER_MIN` / `LLM_RATE_LIMIT_PER_HOUR` | `20` / `300` | Allowance per API key. |
| `LLM_TIMEOUT_S`, `LLM_MAX_RESPONSE_KB` | `30`, `256` | Gateway timeout and response cap. |

### Known limitations (stated by the contract itself)
- The identifier detector finds e-mail addresses, phone numbers, record and ID numbers and **titled** names. It does not find untitled personal names, addresses, dates of birth, spelled-out numbers or obfuscated e-mail addresses ("banda at gmail dot com"). "No identifiers found" is therefore not proof that text is safe; de-identification at the source remains required.
- The same detector over-refuses: a word such as "ID" or "registration number" followed by another word can be read as a record number, and a titled name is refused even if it is a public figure. This is the safe direction; use `identifiers: mask` or reword.
- The diagnosis and prescribing screens are English-only pattern rules. No Chichewa, Tumbuka or Yao patterns have been written or clinically reviewed.
- Rules reduce the chance of a boundary breach and do not remove it; clinician review stays mandatory.
- Rate limits are in memory, per process.

### Tests
`api/tests/test_llm_phase1.py`: boundary (requests and answers), identifier gate (mask, refuse, evasion), audit allow-list and a sentinel test over real requests, per-key limits over HTTP, roles, secrecy of the key (settings repr, every response, audit rows, dashboard files, `render.yaml`, `.env.example`), and the gateway transport against a local HTTP server (Bearer header only, route allow-list, redirects refused, key-free errors, size and content-type checks).
