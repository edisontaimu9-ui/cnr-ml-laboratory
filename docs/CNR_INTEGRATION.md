# CNR integration guide (contract `cnr-ml-contract/1`)

The link runs both ways. **CNR -> lab:** CNR asks for risk estimates. **Lab -> CNR:** the lab tells CNR when a model is deployed, rolled back or withdrawn. The lab never reads CNR's patient data. CNR's side is implemented in `chakudya-api` v1.83.0 (`src/mlLab.js`): admin routes `GET /ml/status`, `GET /ml/models`, `POST /ml/assess/{slug}` and the receiver `POST /hooks/ml-lab`.

## Before you start
- An administrator creates a key with the **service** role (Access page) and gives it to CNR once. Store it as a secret (for a Cloudflare Worker: `wrangler secret put ML_LAB_KEY`). Never put it in client-side code or in a repository.
- API base URL: `https://ml.chakudya.xyz`. Send the key as `X-API-Key: <key>` or `Authorization: Bearer <key>`.
- Call the lab from CNR's **server side** only.

## 1. Discover what is available
`GET /v1/integration/cnr/models`

```json
{"contract": "cnr-ml-contract/1", "safety_notice": "...",
 "models": [{"slug": "child-malnutrition-risk", "version": 2, "available": true,
             "features": {"numeric": ["age_months", "weight_kg"],
                          "categorical": [{"name": "district", "values": ["Zomba", "Blantyre"]}]},
             "classes": ["moderate", "normal", "severe"], "intended_use": "..."}]}
```
Models are listed only while a version is deployed. Send exactly the features listed; unknown fields are rejected.

## 2. Request assessments
`POST /v1/integration/cnr/assess/{slug}` (up to 50 cases)

```json
{"cases": [{"case_ref": "cnr-8f3a21", "features": {"age_months": 20, "weight_kg": 6.0, "district": "Zomba"}}]}
```
Optional header `X-Request-Id: <your id>` (1-64 of `A-Za-z0-9._:-`) is echoed as `correlation_id`.

Response (abridged): `request_id`, `contract`, `correlation_id`, `decision_support_only: true`, `safety_notice`, `model` (name, version, approval and validation reference, evaluation headline), `caveats`, and `results` in the same order, each with your `case_ref`, `kind: "risk_estimate"`, `predicted_class` or `predicted_value`, per-class `probabilities` (scores, not calibrated probabilities), `risk_indicators.low_confidence` and `notes`.

## Rules CNR must follow
- **`case_ref` is opaque.** Generate it in CNR (random or an internal row key) so only CNR can link it to a person. Never derive it from a name, phone or national ID. Emails and 9+ digit numbers are rejected.
- **Send measurements only.** No names, phone numbers, addresses or ID numbers.
- **Show it as decision support.** Display the safety notice, the model version and the low-confidence warning with every estimate. Never present a result as a diagnosis. Keep nutrition calculations (z-scores, classifications) in CNR; use the lab only for what its models were approved for (`intended_use`).
- **Do not store the lab's estimate as a clinical finding.** If CNR records it, record the model name, version and `request_id` with it.

## Errors
| Status | Meaning | What to do |
|---|---|---|
| 401 / 403 | Missing, revoked or wrong-role key | Check the secret; ask an administrator |
| 404 | Unknown model slug | Use a slug from `/models` |
| 409 | No approved model is deployed | Hide the feature; do not retry in a loop |
| 422 | Invalid case or feature (message says which) | Fix the request; values are not echoed back |
| 429 | Rate limit (60/min per key) | Wait `Retry-After` seconds |
| 503 | Model file failed its integrity check or is missing | Predictions are paused; tell an administrator |

## Example (Cloudflare Worker, server side)
```js
async function assess(env, slug, cases, requestId) {
  const res = await fetch(`https://ml.chakudya.xyz/v1/integration/cnr/assess/${slug}`, {
    method: "POST",
    headers: { "X-API-Key": env.ML_LAB_KEY, "Content-Type": "application/json", "X-Request-Id": requestId },
    body: JSON.stringify({ cases }),
  });
  if (res.status === 409) return { unavailable: true };          // no deployed model
  if (!res.ok) throw new Error("ML lab " + res.status);           // log status only, never the body or inputs
  return res.json();
}
```

## Operating it
Admins see key usage, 24-hour counts and a reachability check under **CNR integration**. Every call is audited as `integration.assess` with counts, model version, request ID and correlation ID, never cases or values.

## Lab -> CNR events (webhook)
When an administrator deploys, rolls back or withdraws a model, the lab sends one signed `POST` to `CNR_WEBHOOK_URL` (`https://api.chakudya.xyz/hooks/ml-lab`) in the background, so a slow CNR never delays a deploy. Three attempts; a 4xx from CNR is not retried. Each outcome is audited as `integration.notify`.

```json
{"event_id": "9f1c...", "event": "model.deployed", "occurred_at": "2026-11-05T10:00:00Z", "contract": "cnr-ml-contract/1",
 "model": {"slug": "child-malnutrition-risk", "version": 2, "version_id": 9}, "detail": {"superseded_version": 1}}
```
Events: `model.deployed`, `model.rolled_back`, `model.retired` (only for a version that was being served). No patient data, no evaluation and no free-text reasons are sent.

**Verify every event.** Headers `X-ML-Lab-Timestamp` (unix seconds) and `X-ML-Lab-Signature: sha256=<hex>`, the HMAC-SHA256 of `"<timestamp>.<raw body>"` with the shared secret. Reject a timestamp more than 5 minutes old, compare signatures in constant time, and ignore a repeated `event_id`. The CNR implementation does all four and is tested against the lab's signer.

**Shared secret.** One 32+ character value, set as `CNR_WEBHOOK_SECRET` on the lab and `ML_LAB_WEBHOOK_SECRET` on CNR. Rotate by setting the new value on CNR first, then on the lab.

**An event is a hint, not the truth.** The lab answers 409 whenever nothing is deployed, whatever CNR last heard, so a lost or delayed event is never unsafe.

## What CNR never sees: explorations
Unsupervised explorations (clustering, anomaly detection) are for the laboratory's own use. They are not models, are never registered or deployed, and are not part of the `cnr-ml-contract/1` contract: no endpoint, webhook event or status field refers to them, and the `service` key used by CNR cannot read them.
