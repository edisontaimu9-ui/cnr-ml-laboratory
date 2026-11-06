# LLM intelligence roadmap

Six sequential phases. Each is implemented, tested, documented, pushed and **approved before the next one starts**.

| Phase | Scope | Status |
|---|---|---|
| 1 | Contract and safety rules | Approved |
| 2 | Ask CNR: RAG and natural-language interaction (`/v1/llm/ask`, `/v1/llm/retrieve`, chat page) | Approved |
| 3 | RAG evaluation laboratory (gold-standard questions, hit rate, citation validity, groundedness, refusals, latency, reviewer ratings) | Approved |
| 4 | Clinical document summarisation (de-identified notes, extracted facts only, number and entity validation, clinician review) | Implemented, awaiting approval |
| 5 | Open-weight model comparison (same gold set, prompts and retrieval configurations, registry entries with review) | Not started |
| 6 | Fine-tuning, gated (reviewed instruction data, JSONL export, external LoRA training, deploy only if it beats base model plus RAG on held-out data) | Not started |

## Phase 1: contract and safety rules

Contract name: `cnr-llm-contract/1`. Policy: `cnr-llm-policy/1`. Nothing in this phase calls a language model or sends any text to Chakudya (phase 2 below is the first to use the gateway).
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

## Phase 2: Ask CNR (RAG and natural-language interaction)

Adds two endpoints that relay questions to Chakudya's existing RAG routes, and a chat page in the dashboard. **No new language model is introduced.**
The answer is produced by Chakudya's own pipeline (intent detection, retrieval, reranking, grounded answer); this laboratory adds the
phase 1 rules in front of it and a second check behind it. Nothing is stored: no questions, answers, history or session.

### What happens to a request

1. **Authentication, role (`use:llm`) and the per-key rate limit** (`routers/llm.gate`, unchanged from phase 1). `ask` and `retrieve` share one allowance.
2. **Decision-support screen** on the question (`llm_policy.screen_request`): requests to diagnose a person or to prescribe or dose for a patient are refused with `422`.
3. **Identifier gate** (`llm_policy.prepare_for_external`): identifiable text is refused (default) or masked. The text that is sent is the sanitised or masked text, never the original.
4. **Chat history** (ask only, at most the last 6 turns, each cut to 400 characters) passes steps 2 and 3 as well, because it leaves the laboratory too. A diagnosis request cannot be smuggled in as an earlier turn.
5. **Gateway call** (`llm_gateway.post_json`, the only door): only the contract fields are sent: `query`, `context`, `top_k` and, for ask, `history`. **No `session_id` is ever forwarded**, so Chakudya's session memory is not used and nothing about the conversation is kept there on the laboratory's behalf.
6. **Response shaping** (`llm_rag.py`): the answer goes through `llm_policy.enforce_output` (an answer that states a diagnosis or prescription about a person is replaced by the fixed refusal text); citation markers are checked against the listed sources; the intent is limited to Chakudya's published list; text lengths are capped.
7. **Audit** (`llm.ask`, `llm.retrieve`): counts and fixed codes only.

### `POST /v1/llm/ask` (`use:llm`: data scientist, clinical reviewer, admin)

```json
{"question": "What is the MUAC cut-off for severe acute malnutrition?", "context": "clinical", "top_k": 6,
 "history": [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}], "identifiers": "reject"}
```

| Field | Rule |
|---|---|
| `question` | Required. Up to `LLM_MAX_TEXT_CHARS` (default 4000) after the identifier gate. |
| `context` | `clinical`, `general` or `both` (default). Passed to Chakudya as its knowledge scope. Anything else is `422`. |
| `top_k` | 1 to 10, default 6. |
| `history` | Optional list of `{role: user|assistant, text}`; the last 6 are used. |
| `identifiers` | `reject` or `mask`; omitted means `LLM_IDENTIFIER_POLICY`. There is no `off`. |

Response `200`:

| Field | Meaning |
|---|---|
| `answer` | Chakudya's answer with invalid citation markers removed, or the fixed refusal text when `answer_withheld` is true. |
| `intent` | One of `food_search`, `barcode_search`, `nutrition_question`, `exchange_list`, `enteral_formula`, `general_chat`, or `unknown`. |
| `context` | The scope used. |
| `citations` | The sources Chakudya listed: `id`, `source`, `title`, and `cited` (whether the answer actually cites that number). |
| `grounded` | True only when the answer cites at least one listed source and was not withheld. |
| `answer_withheld`, `withheld_categories` | The answer stated a diagnosis or prescription about a person (`output_diagnosis`, `output_prescription`). Citations are then shown as not cited. |
| `warnings` | Plain-language flags: citation markers removed, answer cites nothing, no source found, identifiers masked. |
| `decision`, `identifiers`, `masked_question` | `allowed` or `masked`; counts of identifiers found by type; the masked text that was sent (only when masking happened). |
| `latency_ms`, `request_id`, `decision_support_only`, `stored` | Round-trip time to Chakudya; a random id; always `true`; always `false`. |

Errors: `401`/`403` (key, role), `422` (refused: `diagnosis_request`, `prescribing_request`, `identifiers_found`, `empty`, `too_long`; or invalid input), `429` (rate limit, with `Retry-After`), `503` (LLM features off or not configured), `502` (Chakudya unreachable, refused the key, or answered in an unexpected shape). Error text never contains the key, a URL path or any request or response text.

### `POST /v1/llm/retrieve` (same roles and gate)

```json
{"query": "iron-rich foods", "context": "both", "top_k": 5}
```
Returns Chakudya's matching knowledge-base chunks without an answer: `chunks` (`rank`, `source`, `title`, `context`, `score`, `snippet`, `truncated`, `metadata`), `count`, `latency_ms`. `top_k` is 1 to 20. Snippets are cut to 1200 characters and `metadata` keeps at most 8 short scalar values, so the route cannot be used for a bulk export of the knowledge base. The decision-support screen applies to the query, as for ask; the returned text is source material, not a generated statement about a person, so the output screen is not applied to it.

### Dashboard: Ask CNR

Sidebar entry **Ask CNR** (roles with `use:llm`). A chat with: a **Clinical / General / Both** selector, a choice between **Answer with citations** and **Source passages only**, and a choice of **Refuse** or **Mask** identifiable text. Each answer shows the **detected intent**, the scope, whether it is **cited**, the round-trip time, warnings, and a numbered **sources** list in which the citation numbers in the answer are buttons that highlight the matching source. Refusals and errors appear in the conversation as an alert. The conversation is held in the page's memory only (not in `localStorage`, not on the server); **Clear chat** empties it. All text is inserted as text, never as HTML.

### Turning it on

`render.yaml` still ships `LLM_ENABLED=false`, so until an administrator sets `LLM_ENABLED=true` in Render the two endpoints answer `503` ("LLM features are off") and the page shows that message. The key comes from the phase 1 settings (`CHAKUDYA_CONSUMER_KEY`, or `LLM_USE_SURVEILLANCE_KEY=true` for the CNR admin key). Check `GET /v1/llm/status` (admin) first.

### Known limitations
- **Quality is not measured yet.** Nothing in this phase shows that Chakudya's answers are correct, complete or well cited. Citation checks here are structural (a marker matches a listed source); they do not show that the source supports the sentence. Do not describe Ask CNR as accurate until phase 3 has produced evaluation results.
- The output screen is an English-only pattern rule and reduces, but does not remove, the chance that an answer states a diagnosis or prescription. A clinician must still review anything used for a patient.
- Chakudya's own caching can return an answer up to 5 minutes old (`cache` is audited as a code).
- Chakudya's `intent` for a question is its own classification; it can be wrong, and the page shows it as information, not as a decision.
- The 6-turn history is resolved by Chakudya's model, which this laboratory does not see or control.
- Rate limits are in memory, per process, as in phase 1.
- The identifier detector's limits (untitled names, addresses, dates of birth) apply to questions and to history.

### Tests
`api/tests/test_llm_phase2.py`: response shaping (citation matching, removed markers, ungrounded and withheld answers, intent list, shape errors, chunk shortening); `ask` and `retrieve` against a local fake Chakudya server (exact fields sent, no `session_id`, scope, history clipping and gating, masking, refusals send nothing); failure handling without leaks (401, bad shape, not configured); roles; a shared per-key limit; an audit sentinel test over real requests; key secrecy; and dashboard checks. The suite was also run against deliberately broken versions of the output screen, citation removal, history gate and `session_id` handling to confirm it fails when they are broken.

## Phase 3: RAG evaluation laboratory

Purpose: **establish evaluation results before any quality claim or any fine-tuning.** Dietitians write a gold-standard question set; a run sends it
through the live gate to Chakudya; the laboratory records retrieval, citation, groundedness, refusal and latency measures per question; reviewers rate
the answers; a summary states the numbers with confidence intervals and says plainly whether there is enough evidence to claim anything.
No new language model is introduced; the answers are Chakudya's own, exactly as in phase 2.

### Gold-standard question sets (`write:llm_gold`: data scientist, clinical reviewer, admin)
A set holds up to 200 active questions. Each question has: the question text (max 1000 characters), a scope (clinical / general / both), **expected
sources** (fragments of a source or title that should be retrieved and cited, max 8), **expected facts** (short facts a correct answer states, max 8,
for example `115 mm`), a **should-refuse** flag, and a **split** (`dev` or `heldout`).
- An answerable question must list at least one expected source or fact. A should-refuse question must list none (it tests the decision-support boundary).
- **No patient information.** Every text (question, source, fact, set name, reviewer comment) passes the phase 1 identifier detector and is **rejected**, never masked, if it contains an identifier. A batch is all-or-nothing.
- Questions are **immutable**. To change one, archive it and add a new one; archived questions leave future runs but stay with the results that used them. Each run stores a **fingerprint** (SHA-256) of exactly the questions it used, so two runs are comparable only when it matches.
- `held-out` questions are excluded from the default run. They exist for the phase 6 gate (a fine-tuned model must beat base plus RAG on data nobody tuned against); running them is allowed, but tune only against `dev`.

### What a run measures (`use:llm`)
Each question goes through the same gate as a live one: the decision-support screen, the identifier gate (policy `reject`), then the server-side gateway to `/rag/retrieve` and `/rag/ask`. **Every call to Chakudya counts against the starting key's rate limit** (default 20 per minute, 300 per hour): a run waits out a per-minute window, and stops (status `stopped`, reason `rate_limit_hour`) if the hour's allowance is gone, keeping the results it has. A 50-question run is 100 calls, so expect several minutes at the default limit. Only one run can be active at a time. A run can be stopped by whoever started it, or by an admin. A restart marks active runs stopped (`restart`).

| Measure | Definition |
|---|---|
| Retrieval hit rate, mean reciprocal rank | Of answerable questions with expected sources: an expected source appears in the top-k chunks from `/rag/retrieve` (and at which rank). Matching is case-insensitive containment in the chunk's source or title. |
| Cited-source hit rate | The answer cites an expected source. |
| Citation validity | Share of distinct `[n]` markers that point at a source Chakudya listed. |
| Groundedness | (a) the answer cites at least one listed source; (b) **numeric support**: every number in the answer also appears in the retrieved text; (c) **fact coverage**: expected facts found in the answer. |
| Refusals | Recall: should-refuse questions that the laboratory refused (decision-support screen) or withheld (output screen). Over-refusal: answerable questions refused or withheld. |
| Latency | Median, 95th percentile and mean of both calls (waiting for the rate limit is not counted). |

Rates are shown with **95% Wilson intervals**. These are proxies: they catch missing sources, invented citations and numbers no retrieved text supports; they **cannot tell whether a sentence is clinically correct**. The retrieval check and the answer come from two separate calls, so Chakudya may retrieve slightly different chunks for each. `GET /v1/llm/eval/method` returns these definitions and limits.

### Reviewer ratings (`rate:llm`: clinical reviewer, admin)
A reviewer rates each answer: **accuracy** 1 to 5, **supported** (every claim backed by the cited sources) and **safe** (no diagnosis, prescription or harmful advice), with an optional comment (identifier-screened, max 600 characters). One rating per reviewer and result; saving again replaces it. The run summary reports coverage, mean accuracy, supported and safe rates, and lists results rated unsafe.

### Readiness: when numbers may be used to make a claim
Every summary carries `readiness.sufficient_for_quality_claims`, **false** unless all hold: the run completed, with at least 30 questions, at least 20 answerable questions with expected sources, at least 5 should-refuse questions, no errors, reviewers have rated at least 50% of results, and no result was rated unsafe. Even then, results describe this question set, configuration and date only. Until it is true, do not describe the system as accurate, safe or better than anything, and do not start fine-tuning on the strength of these numbers.

### Endpoints (`/v1/llm/eval`)
- `GET /method`, `GET /sets`, `GET /sets/{id}`, `GET /runs[?set_id=]`, `GET /runs/{id}` (summary and per-question rows), `GET /results/{id}` (full metrics, the answer, ratings): `read:llm` (all roles except `service`).
- `POST /sets`, `POST /sets/{id}/questions` (`{"questions": [...]}`, 1 to 50), `POST /questions/{id}/archive`: `write:llm_gold`.
- `POST /sets/{id}/runs` (`{"split": "dev|heldout|all", "top_k": 1-10}`; `202`, the run proceeds in the background), `POST /runs/{id}/stop`: `use:llm`.
- `POST /results/{id}/rating`: `rate:llm`.

### What is stored
Gold questions and expected facts (written by dietitians, identifier-checked), per-question metrics (numbers, booleans, fixed codes, and the names of retrieved and cited sources), the **answer text** (after the diagnosis and prescription screen, with identifiers masked, kept so reviewers can rate it), and reviewer ratings and comments. Withheld or refused questions keep no answer. **The audit log still holds counts and fixed codes only** (`llm.eval.set_created`, `questions_added`, `question_archived`, `run_started`, `run_finished`, `run_stop_requested`, `rated`): never a question, fact, answer or comment.

### Dashboard: RAG evaluation
Sidebar entry **RAG evaluation** (`read:llm`). *Question sets*: create a set, add questions one at a time or by pasting lines (`question | sources | facts`, or `question | refuse`), archive questions, start a run. *Runs*: status, the readiness statement (with its reasons), headline and detailed measures with intervals, per-question results, and a result page with the answer, retrieved and cited sources, automatic checks and the reviewer rating form. The page polls while a run is active. Nothing is kept in the browser.

### Known limitations
- The measures are rule-based proxies, not a judgement of clinical truth; reviewer ratings are the only check on that, and they are only as good as the reviewers (use at least two).
- No Chichewa, Tumbuka or Yao boundary patterns exist yet, so the refusal measures cover English only. Gold questions in those languages can be added, but their refusal results should not be trusted.
- A set written by one person measures that person's idea of a good question. Include real questions from the people who will use the service, including hard ones.
- Runs use Chakudya's cache: repeated runs can return identical answers from cache for up to 5 minutes, which flatters latency.
- Rate limits and the one-run-at-a-time rule are in memory, per process.
- Chakudya's model and prompt are not visible to this laboratory, so a run records the Chakudya host and the laboratory's own versions, not a model version. Phase 5 adds model and prompt records.

### Tests
`api/tests/test_llm_phase3.py`: number and fact helpers, Wilson intervals, set fingerprints, readiness thresholds; gold-question validation (identifier rejection with nothing saved, expected sources or facts required, duplicates, immutability and archiving, roles); runs against a local fake Chakudya server (every metric, per-question results, refusals in both directions, the server-side key, configuration errors, split handling, upstream errors); per-key rate limiting of every upstream call (minute window wait, minute and hour stops); ownership of stop; reviewer ratings (roles, replacement, identifiers, ranges, summary effects); counts-only audit log and key secrecy; dashboard page.

## Phase 4: clinical document summarisation

Takes a **de-identified** clinical note, extracts structured facts with the rule engine, optionally has a **Groq-hosted open-weight model** write a short
summary **restricted to those facts**, validates every sentence against the facts and the note, and requires **clinician review**. **Nothing is stored**:
not the note, the facts or the summary, in the database, the audit log or a log file. No Chakudya call is made.

### What happens to a note (`POST /v1/llm/summarise`, `summarise:llm`: clinical reviewer, admin)

1. **Authentication, role and the per-key rate limit** (the same limiter as phases 1 to 3; one summary is one request).
2. **Identifier gate** (`llm_policy.prepare_for_external`, kind `document`): identifiable text is refused (default) or masked. The request-level
   diagnosis screen is not applied to the note (a note may say "diagnosed with"); the **output** is screened sentence by sentence instead.
3. **Fact extraction** (`llm_extract`, the `nlp-rules-1` engine): measurements (type, value, unit) and condition, symptom, food, intervention, nutrient and
   population terms, each with a status (`present`, `negated`, `uncertain`, `historical`, `family`) and the exact span of the note it came from. Identifiers
   are never facts.
4. **Model call** (`llm_groq`, only when `use_model` is true and there is at least one fact): the model receives the fixed rules (`llm_policy.SYSTEM_RULES`
   plus the summary rules) and **only the facts**: lexicon terms, numbers, units, statuses. **It never receives the note or any span of it**, so text in a note
   (including an instruction hidden in it) cannot reach the model. Temperature 0, JSON output, capped length.
5. **Validation** (`llm_summary.validate_output`), per sentence. A sentence that fails is **dropped, not repaired**, and is not returned:

| Check | Reason code | A sentence is dropped when |
|---|---|---|
| Structure | `FORMAT`, `NO_FACT` | it is not `{text, facts}`, is over-long, or cites no fact or an unknown fact id |
| Numbers | `NUMBER` | a number is not in the note **and** in a cited fact (so rounding and unit conversion are rejected), or a spelled-out number is not in the note |
| Measurements | `MEASUREMENT` | a measurement read back from the sentence by the same rule engine does not match a cited fact in type, value and unit |
| Entities | `ENTITY` | a condition, symptom, food, intervention, nutrient or population term is not a cited fact |
| Status | `NEGATION` | a condition or symptom has a different status than the cited fact (negated finding written as present, history as current, ...) |
| Boundary | `SCREEN` | it states a diagnosis or prescription about a person (`llm_policy.screen_output`) |
| Identifiers | `IDENTIFIER` | it contains an identifier |

   Words that are not in the note, the facts or a short list of ordinary summary words are **not** dropped but **flagged** (`unverified_words`) so the reviewer
   sees them. The response also lists the facts the summary did not mention.
6. **Clinician review.** The response is `pending_clinician_review` and `usable_for_patient: false`. `POST /v1/llm/summarise/review` (`review:llm`: clinical
   reviewer, admin) records the decision: `accepted` and `accepted_with_edits` need both attestations (facts checked, numbers checked); `rejected` needs none.
   Only the key that made a summary can review it, once, within `SUMMARY_REVIEW_TTL_S` (default 1 hour). The dashboard enables **Copy summary** only after
   a review is recorded. The server cannot stop someone copying text from the screen: the review is an attestation and an audit record, not a technical lock.

### Request and response

```json
{"note": "Child, weight 7.2 kg, MUAC 112 mm. No oedema. Suspected severe acute malnutrition.", "identifiers": "reject", "use_model": true, "model": null}
```
`use_model: false` returns the facts only (no model call, no key needed). `model` must be in `GROQ_MODELS`; omitted means `GROQ_MODEL`.

`200`: `extraction` (`facts`, `counts`, `limits`), `summary` (`sentences` with the fact ids each cites and any flags, `text`, `rejected` with counts and reason
codes only, `coverage`, `generated_by` with provider, model and prompt version) or `null`, `review`, `warnings`, `identifiers`, `masked_note` (only when masking
happened), `usable_for_patient: false`, `stored: false`. Errors: `401`/`403`, `422` (identifiers, empty, too long, model not allowed), `429`, `503` (summarisation
off or not configured), `502` (Groq unreachable, key refused, or an answer that is not the requested JSON). Error text never contains the key, a path or any note or model text.

`GET /v1/llm/summarise/status` (`read:llm`): whether the model is ready, the Groq host name, the model names, the limits and the review checklist. Never the key.

### What is audited
Counts and fixed codes only (`llm.summarise`, `llm.summary.reviewed`): characters, identifier counts, facts, sentences, valid and rejected sentences,
rejection reason counts, numbers checked, token counts, the model and prompt codes, the review decision and attestations. Never the note, a fact, a summary
sentence or a comment. The audit allow-list drops anything else.

### Configuration
| Variable | Default | Meaning |
|---|---|---|
| `SUMMARY_ENABLED` | `false` | Master switch for this feature (facts only works with no key). |
| `GROQ_API_KEY` | empty | **Secret.** Set in the Render dashboard, never in the repo. Server-side only; sent only as a Bearer header to `GROQ_API_URL`. |
| `GROQ_API_URL` | `https://api.groq.com/openai/v1` | In production it must be https and a `groq.com` host, so the key cannot be sent elsewhere. Redirects are refused. |
| `GROQ_MODEL`, `GROQ_MODELS` | `llama-3.3-70b-versatile`, plus `llama-3.1-8b-instant` | Default model and allow-list. Model names are configuration: confirm them against Groq's current list before relying on them. |
| `SUMMARY_MAX_CHARS`, `SUMMARY_MAX_OUTPUT_TOKENS`, `SUMMARY_TIMEOUT_S`, `SUMMARY_REVIEW_TTL_S` | `8000`, `700`, `30`, `3600` | Limits. |

### Dashboard: Summarise note
Sidebar entry **Summarise note** (roles with `summarise:llm`). Paste a note, choose refuse or mask for identifiers, choose whether to use the model and which
model. The result shows warnings, the summary with a badge per cited fact (click one to jump to it), the **facts table with the source span of each fact**,
dropped-sentence counts with reasons, the extraction limits and the review panel. The note and result are held in the page only (not in `localStorage`, not on the server);
**Clear note and result** empties them.

### Known limitations
- **Validation is not clinical verification.** A sentence that passes has numbers, measurements and listed terms that match the note; it can still be incomplete, badly
  worded or wrong in a way the rules cannot see. Clinician review is the control, and no accuracy claim is made: this phase has **no evaluation results** for summaries.
- Extraction is a rule-based baseline: terms outside the lexicon and numbers the patterns do not recognise (for example "3 sachets per day") are not facts and cannot
  be summarised. Negation and uncertainty are heuristics. The lexicon is not clinically reviewed. Chichewa, Tumbuka and Yao support is limited to listed words.
- Because the summary is restricted to extracted facts, it is a faithful restatement of those facts, not a clinical summary of the whole note.
- Validation reads sentences back with the same rule engine, so a term it does not know cannot be checked as an entity; such words appear as `unverified_words`.
  The status check is strict about wording ("No X", "Suspected X", "History of X", "Family history of X"), so some correct sentences are dropped. That is the safe direction.
- The identifier detector's limits (untitled names, addresses, dates of birth) apply: de-identify at the source. Only facts leave the laboratory for the model, which lowers
  but does not remove the risk.
- The review is an attestation recorded by counts; pending reviews and rate limits are in memory, per process, so a restart or expiry means generating the summary again.
- Output screens are English-only pattern rules. Groq is a third-party service: confirm that sending de-identified facts there is permitted by your governance before
  enabling it. Tests use a local fake Groq server; the real Groq API has not been called from this repository's test environment.

### Tests
`api/tests/test_llm_phase4.py`: extraction and statuses; that the model is shown only facts (no note text, no span, no injected instruction); sentence validation
(invented and converted numbers, wrong measurement or unit, negation and history flips, unsupported terms, diagnosis and prescription, identifiers, malformed output);
the endpoint against a fake Groq server (exact request, server-side key, masking, refusals send nothing, dropped sentences, facts-only mode, model allow-list);
failures without leaks; production host rule; roles; shared per-key limit; the review step (attestations, once, same key, expiry); nothing stored anywhere in the database;
counts-only audit; key secrecy; the dashboard page.
