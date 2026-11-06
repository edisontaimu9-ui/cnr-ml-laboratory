"""Regression benchmark for the NLP engine: a small set of synthetic, de-identified sentences with the entities a reader
would mark. It catches regressions when terminology or patterns change. It is NOT a measure of accuracy on real notes:
the sentences were written by the developers, and real clinical text is messier."""
from collections import Counter

from . import nlp_lab

# text, expected (label, surface) pairs, surfaces that must be flagged negated, uncertain, family
GOLD = [
    ("A 14 month old child presented with severe acute malnutrition. MUAC 10.8 cm, weight 6.2 kg and bilateral pitting oedema.",
     [("MEASUREMENT", "14 month old"), ("POPULATION", "child"), ("CONDITION", "severe acute malnutrition"), ("MEASUREMENT", "MUAC 10.8 cm"),
      ("MEASUREMENT", "weight 6.2 kg"), ("SYMPTOM", "pitting oedema")], [], [], []),
    ("No fever or vomiting. Started RUTF and F-75.",
     [("SYMPTOM", "fever"), ("SYMPTOM", "vomiting"), ("INTERVENTION", "RUTF"), ("INTERVENTION", "F-75")], ["fever", "vomiting"], [], []),
    ("Hb 7.9 g/dL, albumin 28 g/L, RBS 9.4 mmol/L.",
     [("MEASUREMENT", "Hb 7.9 g/dL"), ("MEASUREMENT", "albumin 28 g/L"), ("MEASUREMENT", "RBS 9.4 mmol/L")], [], [], []),
    ("Mwana ali ndi kutupa ndipo palibe kusanza.",
     [("POPULATION", "Mwana"), ("SYMPTOM", "kutupa"), ("SYMPTOM", "kusanza")], ["kusanza"], [], []),
    ("Amadya nsima ndi nyemba ndi masamba, nthawi zina nsomba.",
     [("FOOD", "nsima"), ("FOOD", "nyemba"), ("FOOD", "masamba"), ("FOOD", "nsomba")], [], [], []),
    ("Ate 2 cups of rice with beans and drank tea for breakfast.",
     [("MEASUREMENT", "2 cups"), ("FOOD", "rice"), ("FOOD", "beans"), ("FOOD", "tea")], [], [], []),
    ("Pregnant woman, 26 years old, with anaemia and low iron intake. Given iron and folic acid.",
     [("POPULATION", "Pregnant"), ("MEASUREMENT", "26 years old"), ("CONDITION", "anaemia"), ("NUTRIENT", "iron"),
      ("INTERVENTION", "iron and folic acid")], [], [], []),
    ("Mayi amadya Likuni Phala ndipo mwana amadya mkaka.",
     [("FOOD", "Likuni Phala"), ("POPULATION", "mwana"), ("FOOD", "mkaka")], [], [], []),
    ("Family history of diabetes. Has hypertension and obesity, BMI 31.4.",
     [("CONDITION", "diabetes"), ("CONDITION", "hypertension"), ("CONDITION", "obesity"), ("MEASUREMENT", "BMI 31.4")], [], [], ["diabetes"]),
    ("HIV negative. Possible kwashiorkor.",
     [("CONDITION", "HIV"), ("CONDITION", "kwashiorkor")], ["HIV"], ["kwashiorkor"], []),
    ("Contact Mr Phiri on 0999123456.", [("PHI", "[redacted]"), ("PHI", "[redacted]")], [], [], []),
    ("Mwana akutsegula m'mimba, ndipo wapatsidwa ORS.", [("POPULATION", "Mwana"), ("SYMPTOM", "akutsegula m'mimba"), ("INTERVENTION", "ORS")], [], [], []),
]


def run_benchmark() -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    misses, extras, ctx_total, ctx_ok = [], [], 0, 0
    for i, (text, expected, neg, unc, fam) in enumerate(GOLD):
        r = nlp_lab.analyze(text)
        got = Counter((e["label"], e["text"].lower()) for e in r["entities"])
        want = Counter((lab, s.lower()) for lab, s in expected)
        for k in set(got) | set(want):
            a, b = got[k], want[k]
            tp[k[0]] += min(a, b)
            if b > a:
                fn[k[0]] += b - a
                misses.append({"sentence": i + 1, "label": k[0], "text": k[1]})
            if a > b:
                fp[k[0]] += a - b
                extras.append({"sentence": i + 1, "label": k[0], "text": k[1]})
        flags = {"negated": neg, "uncertain": unc, "family": fam}
        for e in r["entities"]:
            if e["label"] not in ("CONDITION", "SYMPTOM"):
                continue
            for flag, surfaces in flags.items():
                ctx_total += 1
                ctx_ok += (e["text"].lower() in [s.lower() for s in surfaces]) == bool(e["attributes"][flag])
    per = {}
    for lab in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else None
        rc = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else None
        f1 = 2 * p * rc / (p + rc) if p and rc else 0.0 if (p is not None and rc is not None) else None
        per[lab] = {"expected": tp[lab] + fn[lab], "found": tp[lab] + fp[lab], "precision": p and round(p, 3), "recall": rc and round(rc, 3),
                    "f1": f1 and round(f1, 3)}
    T, P, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p, rc = T / (T + P) if T + P else 0.0, T / (T + N) if T + N else 0.0
    return {"engine": nlp_lab.ENGINE_VERSION, "sentences": len(GOLD), "exact_match": True,
            "overall": {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0},
            "per_label": per, "context_flag_accuracy": round(ctx_ok / ctx_total, 3) if ctx_total else None,
            "missed": misses, "extra": extras,
            "notice": "A regression check on a few synthetic sentences written by the developers. It does not measure accuracy on real "
                      "clinical text, which is more varied. Validate on real, de-identified, expert-annotated text before relying on any result."}
