"""Train a spaCy entity model on an external GPU (Colab, Kaggle, rented box) and score it on held-out documents.

Input: a JSONL corpus, one document per line:
  {"text": "...", "entities": [[start, end, "LABEL"], ...], "language": "ny"}   (entities may also be {"start","end","label"}; "language" optional)
Usage:
  pip install -r requirements-ner.txt
  python train_ner_gpu.py --data corpus.jsonl --out ner_out --epochs 30
  python train_ner_gpu.py --data corpus.jsonl --out ner_out --model xlm-roberta-base   # transformer, needs a GPU
  python train_ner_gpu.py --data corpus.jsonl --out ner_out --cpu                      # small model, no GPU
Nothing is tuned on the test split. Results go to ner_out/results.json and the model to ner_out/model-best (zipped as ner_model.zip).
Use only de-identified text. Do not upload real patient text to a shared GPU service.
"""
import argparse, json, random, shutil, sys
from pathlib import Path


def load(path):
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        ents = [(e["start"], e["end"], e["label"]) if isinstance(e, dict) else tuple(e) for e in r.get("entities", [])]
        rows.append({"text": r["text"], "ents": ents, "lang": r.get("language") or "unknown"})
    if not rows:
        sys.exit("The corpus is empty.")
    return rows


def docbin(nlp, rows, path):
    import spacy
    from spacy.tokens import DocBin
    db, skipped = DocBin(), 0
    for r in rows:
        doc = nlp.make_doc(r["text"])
        spans = [doc.char_span(s, e, label=l, alignment_mode="strict") for s, e, l in r["ents"]]
        if any(s is None for s in spans):
            skipped += 1  # an entity boundary falls inside a word: skip the document, as the dashboard trainer does
            continue
        doc.ents = spacy.util.filter_spans(spans)
        db.add(doc)
    db.to_disk(path)
    return skipped


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True); a.add_argument("--out", default="ner_out")
    a.add_argument("--model", default="xlm-roberta-base", help="Hugging Face encoder; ignored with --cpu")
    a.add_argument("--epochs", type=int, default=30); a.add_argument("--dropout", type=float, default=0.2)
    a.add_argument("--test-size", type=float, default=0.2); a.add_argument("--seed", type=int, default=1)
    a.add_argument("--cpu", action="store_true")
    a = a.parse_args()

    import spacy
    from spacy.cli.init_config import init_config
    from spacy.cli.train import train
    from spacy.scorer import Scorer
    from spacy.training import Example

    rows = load(a.data)
    random.Random(a.seed).shuffle(rows)
    n_test = max(1, int(len(rows) * a.test_size))
    test, trainset = rows[:n_test], rows[n_test:]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blank = spacy.blank("xx")
    sk = docbin(blank, trainset, out / "train.spacy") + docbin(blank, test, out / "dev.spacy")

    gpu = not a.cpu
    if gpu:
        import torch
        if not torch.cuda.is_available():
            sys.exit("No GPU found. Switch the runtime to a GPU, or pass --cpu.")
    cfg = init_config(lang="xx", pipeline=["transformer", "ner"] if gpu else ["ner"], optimize="accuracy" if gpu else "efficiency", gpu=gpu)
    cfg["paths"]["train"], cfg["paths"]["dev"] = str(out / "train.spacy"), str(out / "dev.spacy")
    cfg["training"]["max_epochs"], cfg["training"]["dropout"], cfg["training"]["seed"] = a.epochs, a.dropout, a.seed
    if gpu:
        cfg["components"]["transformer"]["model"]["name"] = a.model
    cfg.to_disk(out / "config.cfg")
    train(out / "config.cfg", out / "model", use_gpu=0 if gpu else -1)

    nlp = spacy.load(out / "model" / "model-best")
    def score(subset):
        exs = [Example(nlp.make_doc(r["text"]), blank.make_doc(r["text"])) for r in subset]
        gold = []
        for ex, r in zip(exs, subset):
            ref = blank.make_doc(r["text"])
            spans = [ref.char_span(s, e, label=l, alignment_mode="strict") for s, e, l in r["ents"]]
            ref.ents = spacy.util.filter_spans([s for s in spans if s is not None])
            gold.append(Example(nlp(r["text"]), ref))
        s = Scorer.score_spans(gold, "ents")
        return {"precision": s["ents_p"], "recall": s["ents_r"], "f1": s["ents_f"], "support": sum(len(r["ents"]) for r in subset), "per_label": s["ents_per_type"]}
    langs = sorted({r["lang"] for r in test})
    res = {"engine": "spacy-ner-external-gpu", "base_model": None if a.cpu else a.model, "epochs": a.epochs, "dropout": a.dropout, "seed": a.seed,
           "train_documents": len(trainset), "test_documents": len(test), "skipped_misaligned": sk,
           "overall": score(test), "per_language": {l: score([r for r in test if r["lang"] == l]) for l in langs}}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    shutil.make_archive(str(out / "ner_model"), "zip", out / "model" / "model-best")
    o = res["overall"]
    print(f"\nTest F1 {o['f1']:.3f} (P {o['precision']:.3f}, R {o['recall']:.3f}) on {len(test)} documents. Results: {out/'results.json'}")
    if len(test) < 30:
        print("Warning: fewer than 30 test documents. Treat this score as a trial, not evidence.")


if __name__ == "__main__":
    main()
