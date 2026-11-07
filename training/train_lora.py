#!/usr/bin/env python3
"""LoRA / QLoRA fine-tuning for the CNR open-weight models. Run it on an EXTERNAL GPU (Colab, Kaggle, RunPod...), not on the API server.

  python train_lora.py --config configs/qwen3.5-9b.json --data train.jsonl --out out/cnr-qwen-lora [--val val.jsonl] [--push-to-hub user/name]

Data: JSONL, one chat per line: {"messages": [{"role":"system|user|assistant","content":"..."}, ...]} with a final assistant message.
Train only on reviewed, de-identified text. Never put patient information in the data: it ends up in the adapter and on the GPU host.
Output: a LoRA adapter (small) plus train_report.json with the data hash, settings and final loss.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|(?<!\d)(?:\+?265|0)[\s-]?\d[\d\s-]{7,}(?!\d)")


def load(path: str) -> list[dict]:
    rows, bad = [], 0
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            m = json.loads(line)["messages"]
            assert isinstance(m, list) and m and m[-1]["role"] == "assistant"
            assert all(x["role"] in ("system", "user", "assistant") and isinstance(x["content"], str) and x["content"].strip() for x in m)
        except Exception:
            sys.exit(f"{path} line {n}: not a valid chat record.")
        if PII.search(" ".join(x["content"] for x in m)):
            bad += 1
        rows.append({"messages": m})
    if bad:
        sys.exit(f"{path}: {bad} record(s) look like they contain an e-mail address or phone number. Remove them and run again.")
    if len(rows) < 20:
        print(f"WARNING: only {len(rows)} examples. LoRA on so little data mostly teaches format and can hurt. Aim for hundreds, reviewed.")
    return rows


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--config", required=True)
    a.add_argument("--data", required=True)
    a.add_argument("--val")
    a.add_argument("--out", required=True)
    a.add_argument("--push-to-hub", help="private Hugging Face repo for the adapter (needs HF_TOKEN)")
    a.add_argument("--epochs", type=float)
    a.add_argument("--no-4bit", action="store_true", help="bf16 LoRA instead of QLoRA (needs much more VRAM)")
    args = a.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.no_4bit:
        cfg["load_in_4bit"] = False
    train, val = load(args.data), (load(args.val) if args.val else None)
    sha = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        sys.exit("No GPU found. Run this on a GPU machine (Colab, Kaggle, RunPod...).")
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    quant = None
    if cfg.get("load_in_4bit", True):
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto")
    except (ValueError, KeyError):      # some multimodal releases only register the image-text class
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(cfg["model_id"], quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], task_type="CAUSAL_LM",
                      target_modules="all-linear", exclude_modules=r".*(visual|vision|audio|mm_projector|multi_modal).*")   # language layers only
    sft = SFTConfig(output_dir=args.out, num_train_epochs=cfg["epochs"], learning_rate=cfg["learning_rate"],
                    per_device_train_batch_size=cfg["per_device_batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
                    max_length=cfg["max_seq_len"], bf16=True, logging_steps=5, save_strategy="epoch", gradient_checkpointing=True,
                    lr_scheduler_type="cosine", warmup_ratio=0.05, report_to="none", eval_strategy="epoch" if val else "no")
    t0 = time.time()
    trainer = SFTTrainer(model=model, args=sft, train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(val) if val else None,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    logs = trainer.state.log_history
    report = {"base_model": cfg["model_id"], "method": "qlora" if quant else "lora", "config": cfg, "train_examples": len(train),
              "val_examples": len(val or []), "train_data_sha256": sha,
              "final_train_loss": next((x["loss"] for x in reversed(logs) if "loss" in x), None),
              "final_eval_loss": next((x["eval_loss"] for x in reversed(logs) if "eval_loss" in x), None),
              "gpu": torch.cuda.get_device_name(0), "minutes": round((time.time() - t0) / 60, 1),
              "note": "Loss is not quality. Compare the adapter with the base model on held-out questions before using it."}
    (Path(args.out) / "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub, private=True)
        tok.push_to_hub(args.push_to_hub, private=True)
        print("Adapter pushed (private) to", args.push_to_hub)


if __name__ == "__main__":
    main()
