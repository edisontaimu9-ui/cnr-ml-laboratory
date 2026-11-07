"""NLP laboratory stage 2: train a spaCy named-entity model on a stored corpus version and compare it, label by label and language by
language, with the rule-based engine on the same held-out documents.

One multilingual model is trained from scratch (`spacy.blank("xx")`, no pretrained vectors, because English, Chichewa, Tumbuka and
Yao have no shared pretrained pipeline). The corpus is split by language so every language is represented in the test set, the model
never sees the test documents, and nothing is tuned on them: the number of passes is fixed up front. Results are reported as
precision, recall and F1 on exact span-and-label matches, together with the support (how many gold entities the number rests on).

Safety rules, in the spirit of the rest of the laboratory:
  * Too little data is refused, not trained on. A label needs MIN_EXAMPLES_PER_LABEL examples and the corpus MIN_DOCS. A `trial` run
    lowers those limits so the pipeline can be tried on a small file, and its results are marked `trial` and must not be registered.
  * Small numbers are flagged, not hidden: a language or label with fewer than MIN_SUPPORT test entities gets no verdict.
  * Only the written results and the model file are kept; the training text stays in the corpus store, and the audit log keeps counts only.
  * Nothing here deploys or serves a model. The model file is a private artifact until stage 3 adds review and approval."""
import gzip
import io
import random
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from . import nlp_corpus, nlp_lab

MIN_EXAMPLES_PER_LABEL = nlp_corpus.MIN_EXAMPLES_PER_LABEL          # 20
MIN_DOCS = nlp_corpus.MIN_DOCS_PER_LANG                           # 50
TRIAL_MIN_EXAMPLES_PER_LABEL = 5
TRIAL_MIN_DOCS = 10
MIN_SUPPORT = 20                  # gold entities needed in a test cell before it gets a verdict
MARGIN = 0.05                     # F1 difference needed to call one system better
MAX_MISALIGNED_SHARE = 0.10       # more documents than this lost to tokenisation mismatches means the annotation does not fit
NOTICE = ("A rule-based baseline and a small model trained from scratch on one corpus, scored on documents held out from training. "
          "Scores are only as good as the annotation and say nothing about text unlike the corpus. Annotators' disagreements were not "
          "measured, and no clinical review has taken place. Do not rely on a result without expert review.")


class TrainingError(ValueError):
    pass


# ---------------------------------------------------------------- data
def read_corpus(blob: bytes) -> list[dict]:
    import json
    return [json.loads(line) for line in gzip.decompress(blob).decode("utf-8").splitlines() if line.strip()]


def split(docs: list[dict], test_size: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified by language: each language with two or more documents puts at least one in the test set."""
    rng = random.Random(seed)
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_lang[d["lang"]].append(d)
    train, test = [], []
    for lang in sorted(by_lang):
        group = sorted(by_lang[lang], key=lambda d: d["text"])      # sorted first so the shuffle depends on the seed only
        rng.shuffle(group)
        n_test = min(len(group) - 1, max(1, round(len(group) * test_size))) if len(group) >= 2 else 0
        test += group[:n_test]
        train += group[n_test:]
    return train, test


def check_enough(docs: list[dict], trial: bool) -> list[str]:
    """Refuses a corpus that is too small to train on. Returns warnings for what was allowed through."""
    min_label, min_docs = (TRIAL_MIN_EXAMPLES_PER_LABEL, TRIAL_MIN_DOCS) if trial else (MIN_EXAMPLES_PER_LABEL, MIN_DOCS)
    labels = Counter(e["label"] for d in docs for e in d["entities"])
    if not labels:
        raise TrainingError("The corpus has no entities, so there is nothing to learn from.")
    problems = []
    if len(docs) < min_docs:
        problems.append(f"it has {len(docs)} documents and at least {min_docs} are needed")
    short = {k: v for k, v in sorted(labels.items()) if v < min_label}
    if short:
        problems.append("labels with too few examples (need " + str(min_label) + "): " + ", ".join(f"{k} {v}" for k, v in short.items()))
    if problems:
        raise TrainingError("Not trained: " + "; ".join(problems) + ". Annotate more text, or start a trial run to try the pipeline on small data.")
    warnings = []
    if trial:
        warnings.append("TRIAL RUN: the data is far too small for a trustworthy model. These results only show that the pipeline works and must not be registered or used.")
    for lang, n in sorted(Counter(d["lang"] for d in docs).items()):
        if n < nlp_corpus.MIN_DOCS_PER_LANG:
            warnings.append(f"Only {n} {lang} documents; results for {lang} are unreliable.")
    return warnings


# ---------------------------------------------------------------- spaCy
def _spacy():
    try:
        import spacy
        from spacy.training import Example
        from spacy.util import compounding, fix_random_seed, minibatch
    except ImportError as e:                                                    # pragma: no cover
        raise TrainingError("spaCy is not installed on this server.") from e
    return spacy, Example, compounding, fix_random_seed, minibatch


def _examples(nlp, docs: list[dict], Example) -> tuple[list, int]:
    """Gold examples from corpus documents. A document whose entity boundaries do not fall on token boundaries is skipped and counted."""
    out, skipped = [], 0
    for d in docs:
        doc = nlp.make_doc(d["text"])
        spans = [doc.char_span(e["start"], e["end"], label=e["label"], alignment_mode="strict") for e in d["entities"]]
        if any(s is None for s in spans):
            skipped += 1
            continue
        doc.set_ents(spans)
        out.append(Example(nlp.make_doc(d["text"]), doc))
    return out, skipped


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    rd = lambda x: None if x is None else round(x, 3)  # noqa: E731
    return {"precision": rd(p), "recall": rd(r), "f1": rd(f1), "support": tp + fn, "predicted": tp + fp}


def score(gold: list[list[tuple]], pred: list[list[tuple]], langs: list[str], labels: list[str]) -> dict:
    """Exact span-and-label scoring. Each entity is (start, end, label). Returns overall, per-label, per-language and per-language-per-label."""
    cells: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])     # (lang, label) -> tp, fp, fn
    for g, p, lang in zip(gold, pred, langs):
        gs, ps = set(g), set(p)
        for e in gs & ps:
            cells[(lang, e[2])][0] += 1
        for e in ps - gs:
            cells[(lang, e[2])][1] += 1
        for e in gs - ps:
            cells[(lang, e[2])][2] += 1

    def agg(pred_fn):
        tp = fp = fn = 0
        for (lang, label), v in cells.items():
            if pred_fn(lang, label):
                tp, fp, fn = tp + v[0], fp + v[1], fn + v[2]
        return _prf(tp, fp, fn)

    present = sorted({lab for (_, lab) in cells} | set(labels))
    per_label = {lab: agg(lambda lg, lb, lab=lab: lb == lab) for lab in present}
    per_lang = {lg: agg(lambda l2, lb, lg=lg: l2 == lg) for lg in sorted(set(langs))}
    per_cell = {lg: {lab: _prf(*cells[(lg, lab)]) for lab in present if (lg, lab) in cells} for lg in sorted(set(langs))}
    f1s = [v["f1"] for v in per_label.values() if v["f1"] is not None and v["support"]]
    return {"overall": agg(lambda *_: True), "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else None,
            "per_label": per_label, "per_language": per_lang, "per_language_label": per_cell}


def _verdict(model: dict, rules: dict) -> dict:
    mf, rf, n = model.get("f1"), rules.get("f1"), model.get("support", 0)
    if n < MIN_SUPPORT:
        return {"verdict": "too few examples", "difference": None, "support": n}
    if mf is None or rf is None:
        return {"verdict": "not comparable", "difference": None, "support": n}
    diff = round(mf - rf, 3)
    return {"verdict": "model better" if diff >= MARGIN else "rules better" if diff <= -MARGIN else "no clear difference",
            "difference": diff, "support": n}


def compare(model: dict, rules: dict) -> dict:
    """Model minus rule baseline in F1, with a verdict only where the test set has enough entities."""
    out = {"overall": _verdict(model["overall"], rules["overall"]),
           "per_label": {k: _verdict(model["per_label"][k], rules["per_label"].get(k, {})) for k in model["per_label"]},
           "per_language": {k: _verdict(model["per_language"][k], rules["per_language"].get(k, {})) for k in model["per_language"]}}
    out["note"] = f"A verdict needs at least {MIN_SUPPORT} gold entities in the test set and a difference of {MARGIN} F1 or more; otherwise no claim is made."
    return out


def rule_predictions(docs: list[dict], labels: set[str]) -> list[list[tuple]]:
    return [[(e["start"], e["end"], e["label"]) for e in nlp_lab.analyze(d["text"])["entities"] if e["label"] in labels] for d in docs]


# ---------------------------------------------------------------- run
def run(docs: list[dict], *, n_iter: int = 30, dropout: float = 0.2, test_size: float = 0.2, seed: int = 1, trial: bool = False) -> tuple[dict, bytes]:
    """Trains and evaluates. Returns (results, model_bytes). Raises TrainingError for a corpus that cannot or should not be trained on."""
    warnings = check_enough(docs, trial)
    spacy, Example, compounding, fix_random_seed, minibatch = _spacy()
    fix_random_seed(seed)
    labels = sorted({e["label"] for d in docs for e in d["entities"]})
    train_docs, test_docs = split(docs, test_size, seed)
    nlp = spacy.blank("xx")
    ner = nlp.add_pipe("ner")
    for lab in labels:
        ner.add_label(lab)
    train_ex, skipped_train = _examples(nlp, train_docs, Example)
    test_ex, skipped_test = _examples(nlp, test_docs, Example)
    lost = skipped_train + skipped_test
    if lost / len(docs) > MAX_MISALIGNED_SHARE:
        raise TrainingError(f"{lost} of {len(docs)} documents have entity boundaries that do not fall on word boundaries, "
                            "so the annotation does not fit the tokeniser. Check the spans (no stray punctuation or part-words).")
    if lost:
        warnings.append(f"{lost} documents were skipped because an entity does not start and end on word boundaries.")
    if not train_ex or not test_ex:
        raise TrainingError("After the split there are no training or no test documents. Add more documents.")
    nlp.initialize(lambda: train_ex)
    rng = random.Random(seed)
    losses_by_epoch = []
    for _ in range(n_iter):
        rng.shuffle(train_ex)
        losses: dict = {}
        for batch in minibatch(train_ex, size=compounding(4.0, 32.0, 1.001)):
            nlp.update(batch, drop=dropout, losses=losses)
        losses_by_epoch.append(round(float(losses.get("ner", 0.0)), 2))
    # evaluation on held-out documents only
    kept_texts = {ex.reference.text for ex in test_ex}
    held = [d for d in test_docs if d["text"] in kept_texts]
    gold = [[(e["start"], e["end"], e["label"]) for e in d["entities"]] for d in held]
    langs = [d["lang"] for d in held]
    model_pred = [[(e.start_char, e.end_char, e.label_) for e in doc.ents] for doc in nlp.pipe([d["text"] for d in held])]
    model = score(gold, model_pred, langs, labels)
    rules = score(gold, rule_predictions(held, set(labels)), langs, labels)
    split_info = {"train_documents": len(train_ex), "test_documents": len(held),
                  "train_by_language": dict(Counter(d["lang"] for d in train_docs)), "test_by_language": dict(Counter(langs)),
                  "train_by_label": dict(Counter(e["label"] for d in train_docs for e in d["entities"])),
                  "test_by_label": dict(Counter(e["label"] for d in held for e in d["entities"]))}
    for lang in sorted({d["lang"] for d in docs} - set(langs)):
        warnings.append(f"No {lang} documents in the test set, so {lang} was not evaluated.")
    results = {"engine": "spacy-ner-xx", "spacy": spacy.__version__, "baseline_engine": nlp_lab.ENGINE_VERSION, "trial": trial,
               "config": {"n_iter": n_iter, "dropout": dropout, "test_size": test_size, "seed": seed},
               "labels": labels, "split": split_info, "loss_by_epoch": losses_by_epoch, "model": model, "rules": rules,
               "comparison": compare(model, rules), "skipped_misaligned": lost, "warnings": warnings, "notice": NOTICE}
    return results, model_to_bytes(nlp)


# ---------------------------------------------------------------- model file
def model_to_bytes(nlp) -> bytes:
    """The pipeline as a ZIP of its spaCy directory."""
    with tempfile.TemporaryDirectory() as tmp:
        nlp.to_disk(tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(Path(tmp).rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(tmp).as_posix())
        return buf.getvalue()


def load_model(data: bytes):
    """Loads a model made by `model_to_bytes`. Only our own stored artifacts are ever passed here."""
    spacy, *_ = _spacy()
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = Path(tmp).resolve()
            for name in z.namelist():                                          # refuse paths that climb out of the folder
                if (root / name).resolve().parent != root and root not in (root / name).resolve().parents:
                    raise TrainingError("The model file is not valid.")
            z.extractall(tmp)
        return spacy.load(tmp)
