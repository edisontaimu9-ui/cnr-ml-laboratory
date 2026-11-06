"""NLP model registry rules (stage 3): what a reviewer is told about each language, and which languages may be approved.

A registered NLP model is a frozen copy of one completed, non-trial training run. Review is per language, because one multilingual
model can be good in English and useless in Yao. From the frozen evaluation this module works out, for every language the model was
trained on, whether it was tested on held-out text and what a reviewer must look at before signing it off:

  * not evaluated        no held-out documents in that language: it can never be approved (blocking)
  * few documents        fewer than MIN_DOCS_PER_LANG documents in the whole corpus (needs acknowledgement)
  * too few test entities  too few gold entities in the test set for any verdict (needs acknowledgement)
  * rules better         the rule baseline scored clearly higher in that language (needs acknowledgement)"""
import re

from . import nlp_corpus

NAME = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")


def language_flags(results: dict) -> dict[str, dict]:
    split = results.get("split") or {}
    train, test = split.get("train_by_language") or {}, split.get("test_by_language") or {}
    per_lang = (results.get("model") or {}).get("per_language") or {}
    rules_lang = (results.get("rules") or {}).get("per_language") or {}
    verdicts = (results.get("comparison") or {}).get("per_language") or {}
    out: dict[str, dict] = {}
    for lang in sorted(set(train) | set(test)):
        total_docs = train.get(lang, 0) + test.get(lang, 0)
        flags, blocking = [], False
        if not test.get(lang):
            flags.append("not evaluated: no held-out documents in this language")
            blocking = True
        else:
            if total_docs < nlp_corpus.MIN_DOCS_PER_LANG:
                flags.append(f"few documents: {total_docs} in the corpus, {nlp_corpus.MIN_DOCS_PER_LANG} are advised")
            verdict = (verdicts.get(lang) or {}).get("verdict")
            if verdict == "too few examples":
                flags.append("too few test entities for a verdict")
            elif verdict == "rules better":
                flags.append("the rule baseline scored clearly higher")
        m, r = per_lang.get(lang) or {}, rules_lang.get(lang) or {}
        out[lang] = {"documents": total_docs, "test_documents": test.get(lang, 0), "test_entities": m.get("support", 0),
                     "f1_model": m.get("f1"), "f1_rules": r.get("f1"), "verdict": (verdicts.get(lang) or {}).get("verdict"),
                     "flags": flags, "blocking": blocking}
    return out


def headline(results: dict) -> dict:
    cmp_ = (results.get("comparison") or {}).get("overall") or {}
    return {"macro_f1_model": (results.get("model") or {}).get("macro_f1"), "macro_f1_rules": (results.get("rules") or {}).get("macro_f1"),
            "overall_verdict": cmp_.get("verdict"), "test_documents": (results.get("split") or {}).get("test_documents"),
            "labels": results.get("labels"), "languages": sorted((results.get("split") or {}).get("train_by_language") or {})}


def approval_problems(results: dict, approved: list[str], acknowledged: bool) -> list[str]:
    """Empty when the approval may go ahead. Messages are for the reviewer."""
    flags = language_flags(results)
    problems = []
    unknown = [lang for lang in approved if lang not in flags]
    if unknown:
        problems.append("The model was not trained on: " + ", ".join(unknown) + ".")
    blocked = [lang for lang in approved if flags.get(lang, {}).get("blocking")]
    if blocked:
        problems.append("Not evaluated on held-out text, so they cannot be approved: " + ", ".join(blocked) + ".")
    flagged = {lang: flags[lang]["flags"] for lang in approved if lang in flags and flags[lang]["flags"] and not flags[lang]["blocking"]}
    skipped = results.get("skipped_misaligned", 0)
    if (flagged or skipped) and not acknowledged:
        bits = [f"{lang}: " + "; ".join(f) for lang, f in flagged.items()]
        if skipped:
            bits.append(f"{skipped} training or test documents were skipped for misaligned entity boundaries")
        problems.append("These need your explicit acknowledgement (acknowledge_warnings): " + " | ".join(bits) + ".")
    return problems
