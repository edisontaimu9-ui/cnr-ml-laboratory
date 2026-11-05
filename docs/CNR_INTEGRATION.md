# CNR integration guide (contract `cnr-ml-contract/1`)

For the team that builds the CNR side. CNR calls the laboratory; the lab does not call CNR's data endpoints.

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
