"""NLP laboratory: language identification and document-type classification, trained on the same corpus as the entity model.

Two small text models, both character and word n-gram TF-IDF with a logistic regression on top (scikit-learn, no pretrained weights, so
they work for Chichewa, Tumbuka and Yao as well as English):

  * `langid`    predicts the corpus `lang` of a text (en, ny, tum, yao, mixed).
  * `classify`  predicts the corpus `category` of a text (clinical_note, lab_report, dietary_record, research_text, other). Only documents
                that carry a category are used.

Same discipline as the entity model: a stratified train / validation / test split (the validation share chooses the regularisation
strength, the test set is never used for any choice), exact scores on the held-out documents only, the rule engine scored on the very same
documents as the baseline, the majority-class rate shown for context, small test cells flagged instead of judged. Results use the same
shape as the entity results (overall, per label = per class, per language) so the registry, the per-language review and the dashboard
treat all three model kinds alike. The model file is a ZIP holding the fitted pipeline (`model.joblib`) and `meta.json`; it is only ever
produced and read by this server, and its hash is checked before approval and before it is loaded."""
import io
import json
import zipfile
from collections import Counter

from . import nlp_corpus, nlp_lab, nlp_training as nt

TASKS = {
    "langid": {"field": "lang", "engine": "sklearn-langid", "baseline": "rule-based language detector (nlp-rules-1)",
               "what": "the language of a text"},
    "classify": {"field": "category", "engine": "sklearn-textcat", "baseline": "rule-based document-type cues (nlp-rules-1)",
                 "what": "the document type of a text"},
}
C_GRID = (0.3, 1.0, 3.0, 10.0)
MIN_PER_CLASS = nlp_corpus.MIN_EXAMPLES_PER_LABEL        # documents of every class a normal run needs
TRIAL_MIN_PER_CLASS = nt.TRIAL_MIN_EXAMPLES_PER_LABEL
NOTICE = ("A small n-gram model trained from scratch on one corpus and scored on documents held out from training, next to the rule "
          "engine on the same documents. It says nothing about text unlike the corpus, annotators' disagreements were not measured, and "
          "no clinical review has taken place. Do not rely on a result without expert review.")


class TextModelError(nt.TrainingError):
    pass


def labelled(docs: list[dict], task: str) -> list[dict]:
    field = TASKS[task]["field"]
    return [d for d in docs if d.get(field)]


def check_enough(docs: list[dict], task: str, trial: bool) -> list[str]:
    if task not in TASKS:
        raise TextModelError("task must be one of " + ", ".join(sorted(TASKS)) + ", or ner.")
    field = TASKS[task]["field"]
    use = labelled(docs, task)
    counts = Counter(d[field] for d in use)
    if not use:
        raise TextModelError("No document carries a " + field + (" (add a category column or field to the corpus)" if task == "classify" else "") + ", so there is nothing to learn from.")
    min_class, min_docs = (TRIAL_MIN_PER_CLASS, nt.TRIAL_MIN_DOCS) if trial else (MIN_PER_CLASS, nt.MIN_DOCS)
    problems = []
    if len(counts) < 2:
        problems.append(f"it has only one {field} ({next(iter(counts))}); at least two are needed")
    if len(use) < min_docs:
        problems.append(f"it has {len(use)} documents with a {field} and at least {min_docs} are needed")
    short = {k: v for k, v in sorted(counts.items()) if v < min_class}
    if short:
        problems.append(f"{field} values with too few documents (need {min_class}): " + ", ".join(f"{k} {v}" for k, v in short.items()))
    if problems:
        raise TextModelError("Not trained: " + "; ".join(problems) + ". Annotate more text, or start a trial run to try the pipeline on small data.")
    warnings = []
    if trial:
        warnings.append("TRIAL RUN: the data is far too small for a trustworthy model. These results only show that the pipeline works and must not be registered or used.")
    if len(use) < len(docs):
        warnings.append(f"{len(docs) - len(use)} documents without a {field} were left out.")
    for lang, n in sorted(Counter(d["lang"] for d in use).items()):
        if n < nlp_corpus.MIN_DOCS_PER_LANG:
            warnings.append(f"Only {n} {lang} documents; results for {lang} are unreliable.")
    return warnings


def _pipeline(task: str, C: float, seed: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True)
    feats = chars if task == "langid" else FeatureUnion([("char", chars), ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True))])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed))])


def _macro_f1(y, p) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", zero_division=0))


def rule_labels(docs: list[dict], task: str, classes: list[str]) -> list[str]:
    """The rule engine's answer for each document, mapped onto the corpus classes (anything it cannot say counts as 'other' or 'unknown')."""
    out = []
    for d in docs:
        a = nlp_lab.analyze(d["text"])
        if task == "langid":
            out.append(a["language"]["overall"])
        else:
            lab = a["classification"]["document_type"]["label"]
            out.append("other" if lab == "unclear" else lab)
    return out


def score(y: list[str], pred: list[str], langs: list[str], classes: list[str]) -> dict:
    """Same shape as nlp_training.score. 'overall' is accuracy (micro F1 of single-label prediction); per_label is per class;
    per_language is the accuracy within each language, with the number of test documents as support."""
    from sklearn.metrics import precision_recall_fscore_support
    rd = lambda x: None if x is None else round(float(x), 3)  # noqa: E731
    n = len(y)
    acc = sum(a == b for a, b in zip(y, pred)) / n if n else None
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    predicted = Counter(pred)
    per_label = {c: {"precision": rd(p[i]), "recall": rd(r[i]), "f1": rd(f[i]), "support": int(sup[i]), "predicted": predicted.get(c, 0)}
                 for i, c in enumerate(classes)}
    per_lang = {}
    for lg in sorted(set(langs)):
        idx = [i for i, x in enumerate(langs) if x == lg]
        a = sum(y[i] == pred[i] for i in idx) / len(idx)
        per_lang[lg] = {"precision": rd(a), "recall": rd(a), "f1": rd(a), "support": len(idx), "predicted": len(idx)}
    f1s = [v["f1"] for v in per_label.values() if v["support"] and v["f1"] is not None]
    matrix = [[sum(1 for a, b in zip(y, pred) if a == t and b == c) for c in classes] for t in classes]
    return {"overall": {"precision": rd(acc), "recall": rd(acc), "f1": rd(acc), "accuracy": rd(acc), "support": n, "predicted": n},
            "macro_f1": round(sum(f1s) / len(f1s), 3) if f1s else None, "per_label": per_label, "per_language": per_lang,
            "per_language_label": {}, "confusion": {"labels": classes, "rows_are": "true class", "matrix": matrix}}


def run(docs: list[dict], task: str, *, test_size: float = 0.2, val_size: float = 0.15, seed: int = 1, trial: bool = False) -> tuple[dict, bytes]:
    """Trains and evaluates. Returns (results, model_zip_bytes)."""
    import joblib
    import sklearn
    warnings = check_enough(docs, task, trial)
    field = TASKS[task]["field"]
    use = labelled(docs, task)
    classes = sorted({d[field] for d in use})
    train, val, test = nt.split3(use, max(val_size, 0.0), test_size, seed, key=lambda d: d[field])
    if not train or not test:
        raise TextModelError("After the split there are no training or no test documents. Add more documents.")
    missing = [c for c in classes if not any(d[field] == c for d in train)]
    if missing:
        raise TextModelError("Some classes have no training documents after the split: " + ", ".join(missing) + ". Add more documents.")
    X = lambda ds: [d["text"] for d in ds]          # noqa: E731
    y = lambda ds: [d[field] for d in ds]           # noqa: E731
    by_c: dict[str, float] = {}
    if val:
        for C in C_GRID:
            m = _pipeline(task, C, seed).fit(X(train), y(train))
            by_c[str(C)] = round(_macro_f1(y(val), m.predict(X(val))), 3)
        chosen = float(max(by_c, key=lambda k: (by_c[k], -float(k))))          # ties go to the stronger regularisation (smaller C)
    else:
        chosen = 1.0
        warnings.append("No validation documents (the corpus is too small), so the regularisation strength was not tuned.")
    model = _pipeline(task, chosen, seed).fit(X(train), y(train))
    pred = list(model.predict(X(test)))
    langs = [d["lang"] for d in test]
    scored = score(y(test), pred, langs, classes)
    rules = score(y(test), rule_labels(test, task, classes), langs, classes)
    majority = Counter(y(train)).most_common(1)[0][0]
    maj_acc = round(sum(a == majority for a in y(test)) / len(test), 3)
    for lang in sorted({d["lang"] for d in use} - set(langs)):
        warnings.append(f"No {lang} documents in the test set, so {lang} was not evaluated.")
    split_info = {"train_documents": len(train), "validation_documents": len(val), "test_documents": len(test),
                  "train_by_language": dict(Counter(d["lang"] for d in train)), "validation_by_language": dict(Counter(d["lang"] for d in val)),
                  "test_by_language": dict(Counter(langs)), "train_by_label": dict(Counter(y(train))), "test_by_label": dict(Counter(y(test)))}
    results = {"task": task, "engine": TASKS[task]["engine"], "sklearn": sklearn.__version__, "baseline_engine": nlp_lab.ENGINE_VERSION,
               "baseline_description": TASKS[task]["baseline"], "trial": trial,
               "config": {"test_size": test_size, "val_size": val_size, "seed": seed, "C_grid": list(C_GRID), "chosen_C": chosen},
               "validation": {"documents": len(val), "macro_f1_by_C": by_c, "chosen_C": chosen,
                              "note": "The validation set chose the regularisation strength. The test set was not used for any choice."},
               "labels": classes, "split": split_info, "model": scored, "rules": rules, "comparison": nt.compare(scored, rules),
               "majority_class": {"class": majority, "test_accuracy": maj_acc,
                                  "note": "Accuracy of always answering the most common training class: a model must beat this to have learned anything."},
               "skipped_misaligned": 0, "warnings": warnings, "notice": NOTICE}
    meta = {"task": task, "engine": TASKS[task]["engine"], "classes": classes, "sklearn": sklearn.__version__}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        mb = io.BytesIO()
        joblib.dump(model, mb)
        z.writestr("model.joblib", mb.getvalue())
        z.writestr("meta.json", json.dumps(meta))
    return results, buf.getvalue()


def load_model(data: bytes):
    """Returns (pipeline, meta) from a model ZIP made by `run`. Reads two named members only; nothing is extracted to disk."""
    import joblib
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            meta = json.loads(z.read("meta.json"))
            model = joblib.load(io.BytesIO(z.read("model.joblib")))
    except Exception as e:
        raise TextModelError("The model file is not valid.") from e
    return model, meta


def predict(model, meta: dict, text: str, top: int = 3) -> dict:
    probs = model.predict_proba([text])[0]
    order = sorted(range(len(probs)), key=lambda i: -probs[i])[:top]
    cls = list(model.classes_)
    ranked = [{"label": str(cls[i]), "probability": round(float(probs[i]), 3)} for i in order]
    return {"label": ranked[0]["label"], "probability": ranked[0]["probability"], "ranked": ranked}
