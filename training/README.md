# Open-weight models and LoRA fine-tuning (external GPU)

Models (first choice Qwen3.5, second choice Gemma 4). Hugging Face ids are in `configs/*.json` and `api/app/llm_openweight.py`; confirm them before a long run.

| Id | Hugging Face | QLoRA VRAM |
|---|---|---|
| `qwen3.5-9b` (1st) | `Qwen/Qwen3.5-9B` | ~24 GB |
| `qwen3.5-27b` (1st) | `Qwen/Qwen3.5-27B` | ~48 GB |
| `gemma-4-e4b` (2nd) | `google/gemma-4-E4B-it` | ~16 GB |
| `gemma-4-12b` (2nd) | `google/gemma-4-12B` | ~24 GB |

## 1. Serve a model (inference for the lab)
On any GPU host (RunPod, Vast, Lambda, a university server):
```
pip install vllm
vllm serve Qwen/Qwen3.5-9B --port 8000 --api-key YOUR_SECRET --reasoning-parser qwen3
```
Put it behind https. Then set on the API (Render dashboard, never in the repo):
`OPENWEIGHT_ENABLED=true`, `SUMMARY_ENABLED=true`, `OPENWEIGHT_API_URL=https://your-host/v1`, `OPENWEIGHT_API_KEY=YOUR_SECRET`.
Call `POST /v1/llm/summarise` with `"provider": "openweight"` and optionally `"model": "gemma-4-e4b"`. `GET /v1/llm/summarise/status` shows readiness.

## 2. Fine-tune with LoRA (training, external GPU)
1. Prepare `train.jsonl` (and optionally `val.jsonl`): reviewed, de-identified chats (see `sample.jsonl`). No patient information.
2. On the GPU machine (Colab/Kaggle notebook: `!git clone`, `!pip install -r training/requirements.txt`):
```
export HF_TOKEN=...   # only if pushing the adapter
python training/train_lora.py --config training/configs/qwen3.5-9b.json --data train.jsonl --val val.jsonl --out out/cnr-qwen-lora --push-to-hub YOURUSER/cnr-qwen-lora
```
3. Output: the adapter plus `train_report.json` (data SHA-256, settings, loss, GPU). Loss is not quality.

## 3. Use the adapter
```
vllm serve Qwen/Qwen3.5-9B --enable-lora --lora-modules cnr-qwen-lora=YOURUSER/cnr-qwen-lora --api-key YOUR_SECRET
```
On the API set `OPENWEIGHT_MODELS=qwen3.5-9b,cnr-qwen` and `OPENWEIGHT_SERVED=cnr-qwen=cnr-qwen-lora`, then request `"model": "cnr-qwen"`.

## Rules
- Compare the adapter with the base model on questions it was not trained on before trusting it. Training on clinical text can make answers fluent and wrong.
- Keep held-out gold questions out of the training data.
- Output screens, identifier gate, fact validation and clinician review still apply to every model.
