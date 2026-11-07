"""Train the NLP entity model on an external GPU and produce results the dashboard can import.

It reuses the laboratory's own split, scoring, rule baseline and minimum-data checks, so the numbers mean the same as a run on the server.
Nothing is tuned on the test documents: a separate slice of the training documents picks the best epoch.

1. Dashboard > NLP > Corpora > open the corpus > "Train on an external GPU" > Download corpus (corpusN-vM.jsonl.gz).
2. On the GPU machine, with this repo cloned or uploaded:
     pip install -r training/requirements-ner.txt -r api/requirements.txt
     python training/train_ner_gpu.py --repo . --data corpus1-v1.jsonl.gz --out ner_out
3. Back on the dashboard, same card: import ner_out/results.json (and ner_out/ner_model.zip).
Options: --model xlm-roberta-base (transformer encoder, needs a GPU) | --cpu (small model) | --epochs 30 | --test-size 0.2 | --seed 1 | --trial
"""
import argparse, gzip, hashlib, json, shutil, sys
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="folder that contains api/app")
    ap.add_argument("--data", required=True, help="the exported corpus, corpusN-vM.jsonl.gz")
    ap.add_argument("--out", default="ner_out")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.repo) / "api"))
    try:
        from app import nlp_training as nt
    except Exception as e:
        sys.exit(f"Cannot load the laboratory code from {a.repo}/api ({e}). Clone the repo here and install api/requirements.txt.")
    import spacy
    from spacy.cli.init_config import init_config
    from spacy.cli.train import train
    from spacy.tokens import DocBin

    raw = Path(a.data).read_bytes()
    if not a.data.endswith(".gz"):
        sys.exit("Use the file downloaded from the dashboard (.jsonl.gz), so the results can be tied to that exact corpus version.")
    sha = hashlib.sha256(raw).hexdigest()
    docs = nt.read_corpus(raw)
    warnings = nt.check_enough(docs, a.trial)
    labels = sorted({e["label"] for d in docs for e in d["entities"]})
    train_all, test_docs = nt.split(docs, a.test_size, a.seed)
    train_docs, dev_docs = nt.split(train_all, 0.1, a.seed)          # dev picks the best epoch; the test documents are never used for that
    blank = spacy.blank("xx")

    def build(subset):
        keep, db = [], DocBin()
        for d in subset:
            doc = blank.make_doc(d["text"])
            spans = [doc.char_span(e["start"], e["end"], label=e["label"], alignment_mode="strict") for e in d["entities"]]
            if any(s is None for s in spans):
                continue
            doc.set_ents(spans); db.add(doc); keep.append(d)
        return keep, db

    train_k, train_db = build(train_docs); dev_k, dev_db = build(dev_docs); test_k, _ = build(test_docs)
    lost = len(train_docs) + len(dev_docs) + len(test_docs) - len(train_k) - len(dev_k) - len(test_k)
    if lost / len(docs) > nt.MAX_MISALIGNED_SHARE:
        sys.exit(f"{lost} of {len(docs)} documents have entity boundaries that do not fall on word boundaries. Check the annotation.")
    if lost:
        warnings.append(f"{lost} documents were skipped because an entity does not start and end on word boundaries.")
    if not train_k or not dev_k or not test_k:
        sys.exit("After the split there are no training, development or test documents. Add more documents.")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train_db.to_disk(out / "train.spacy"); dev_db.to_disk(out / "dev.spacy")

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
    gold = [[(e["start"], e["end"], e["label"]) for e in d["entities"]] for d in test_k]
    langs = [d["lang"] for d in test_k]
    pred = [[(e.start_char, e.end_char, e.label_) for e in doc.ents] for doc in nlp.pipe([d["text"] for d in test_k])]
    model = nt.score(gold, pred, langs, labels)
    rules = nt.score(gold, nt.rule_predictions(test_k, set(labels)), langs, labels)
    for lang in sorted({d["lang"] for d in docs} - set(langs)):
        warnings.append(f"No {lang} documents in the test set, so {lang} was not evaluated.")
    res = {"engine": "spacy-ner-external-gpu", "base_model": None if a.cpu else a.model, "spacy": spacy.__version__, "baseline_engine": nt.nlp_lab.ENGINE_VERSION,
           "trial": a.trial, "config": {"n_iter": a.epochs, "dropout": a.dropout, "test_size": a.test_size, "seed": a.seed}, "labels": labels,
           "split": {"train_documents": len(train_k) + len(dev_k), "test_documents": len(test_k),
                     "train_by_language": dict(Counter(d["lang"] for d in train_docs + dev_docs)), "test_by_language": dict(Counter(langs)),
                     "train_by_label": dict(Counter(e["label"] for d in train_docs + dev_docs for e in d["entities"])),
                     "test_by_label": dict(Counter(e["label"] for d in test_k for e in d["entities"]))},
           "loss_by_epoch": [], "model": model, "rules": rules, "skipped_misaligned": lost, "warnings": warnings, "notice": nt.NOTICE,
           "corpus": {"sha256": sha}}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    shutil.make_archive(str(out / "ner_model"), "zip", out / "model" / "model-best")
    o = model["overall"]
    print(f"\nTest F1 {o['f1']} (P {o['precision']}, R {o['recall']}) vs rules {rules['overall']['f1']} on {len(test_k)} documents.")
    print(f"Import {out/'results.json'} (and {out/'ner_model.zip'}) on the dashboard. Transformer models are large; the server accepts up to 300 MB.")


if __name__ == "__main__":
    main()
