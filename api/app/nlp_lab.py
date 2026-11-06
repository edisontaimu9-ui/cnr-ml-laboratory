"""NLP laboratory engine: nutrition entity recognition, measurement extraction, language identification (English /
Chichewa), identifier masking and document classification for free text.

This is a transparent, rule-based baseline (terminology lists and patterns), not a trained model. It extracts what the text
says and does not interpret it: no diagnosis is made and no nutrition status is assigned. Negation, uncertainty, history
and family context are heuristics. Text is processed in memory and never stored or logged."""
import re
from collections import Counter, defaultdict

from . import nlp_lexicon as lx
from .ingest import EMAIL, PHONE

ENGINE_VERSION = "nlp-rules-1"
MAX_TEXT_CHARS = 8000
MAX_TEXTS = 20
_PRIORITY = {"PHI": 0, "MEASUREMENT": 1}          # everything else (lexicon) is 2

# ---------------------------------------------------------------- lexicon matching
def _key(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("'", "")
    return re.sub(r"[\s\-]+", " ", s).strip()


def _pattern(term: str, plural: bool) -> str:
    out = []
    for ch in term:
        if ch in " -":
            out.append(r"[\s\-]*" if ch == "-" else r"\s+")
        elif ch in "'’":
            out.append("['’]?")
        else:
            out.append(re.escape(ch))
    return "".join(out) + (r"(?:s|es)?" if plural else "")


class _Matcher:
    def __init__(self, items: list[dict], flags: int, plural: bool):
        self.by_key = {(_key(i["term"]).lower() if flags & re.I else _key(i["term"])): i for i in items}
        ordered = sorted(items, key=lambda i: -len(i["term"]))
        alt = "|".join(_pattern(i["term"], plural and i["lang"] in ("en", "both")) for i in ordered)
        self.rx = re.compile(r"(?<!\w)(?:" + alt + r")(?!\w)", flags) if alt else None
        self.cs = not flags & re.I

    def find(self, text: str):
        if self.rx is None:
            return
        for m in self.rx.finditer(text):
            k = _key(m.group(0))
            k = k if self.cs else k.lower()
            item = self.by_key.get(k) or self.by_key.get(k[:-1]) or self.by_key.get(k[:-2])
            if item:
                yield m.start(), m.end(), item


_ENTRIES = lx.entries()
_MATCHERS = [_Matcher([e for e in _ENTRIES if not e["case_sensitive"]], re.I, True),
             _Matcher([e for e in _ENTRIES if e["case_sensitive"]], 0, False)]

# ---------------------------------------------------------------- measurements
_NUM = r"(\d{1,4}(?:[.,]\d{1,2})?)"
_LEAD = r"\s*(?:of|is|was|=|:)?\s*"
_B, _E = r"(?<![\w])", r"(?![\w/])"


def _f(x: str) -> float:
    return float(x.replace(",", "."))


# (type, regex, group index of the value, group index of the unit or None)
_MEAS = [
    ("muac", _B + r"MUAC" + _LEAD + _NUM + r"\s*(mm|cm)?" + _E, 1, 2),
    ("bmi", _B + r"BMI" + _LEAD + _NUM + r"(?:\s*kg\s*/\s*m(?:2|²))?" + _E, 1, None),
    ("haemoglobin", _B + r"(?:Hb|Hgb|haemoglobin|hemoglobin)" + _LEAD + _NUM + r"\s*(g\s*/\s*dl|g\s*/\s*l)?" + _E, 1, 2),
    ("hba1c", _B + r"HbA1c" + _LEAD + _NUM + r"\s*(%|mmol\s*/\s*mol)?", 1, 2),
    ("glucose", _B + r"(?:RBS|FBS|random blood sugar|fasting blood sugar|blood glucose|blood sugar|glucose)" + _LEAD + _NUM
     + r"\s*(mmol\s*/\s*l|mg\s*/\s*dl)?" + _E, 1, 2),
    ("albumin", _B + r"albumin" + _LEAD + _NUM + r"\s*(g\s*/\s*dl|g\s*/\s*l)?" + _E, 1, 2),
    ("z_score", _B + r"(?:WAZ|HAZ|WHZ|WFH|WFL|BAZ|LAZ|z[\s-]?score)" + _LEAD + r"(-?\s?\d{1,2}(?:[.,]\d{1,2})?)", 1, None),
    ("weight", _B + r"(?:weight|weighs|weighed|wt|wgt)" + _LEAD + _NUM + r"\s*(kg|kgs|g)?" + _E, 1, 2),
    ("height", _B + r"(?:height|length|ht)" + _LEAD + _NUM + r"\s*(cm|mm|m)?" + _E, 1, 2),
    ("age", _B + r"(?:aged?" + _LEAD + _NUM + r"\s*(years?|yrs?|months?|mths?|weeks?|wks?)|"
     + r"(?P<n2>\d{1,3})[\s-]*(?P<u2>year|yr|month|mth|week|wk)s?[\s-]*old)", None, None),
    ("energy", _B + _NUM + r"\s*(kcal|kj|calories)" + _E, 1, 2),
    ("nutrient_amount", _B + _NUM + r"\s*g\s*(?:of\s+)?(?:protein|fat|carbohydrates?|fib(?:re|er))" + _E, 1, None),
    ("portion", _B + _NUM + r"\s*(cups?|plates?|bowls?|tablespoons?|teaspoons?|spoons?|pieces?|slices?|glasses)" + _E, 1, 2),
    ("duration", _B + r"(?:for|since|over|past|last)\s+(\d{1,3})\s*(days?|weeks?|months?|years?)" + _E, 1, 2),
    ("mass", _B + _NUM + r"\s*(kg|kgs|kilograms?)" + _E, 1, 2),
    ("length", _B + _NUM + r"\s*(cm)" + _E, 1, 2),
    ("dose_or_volume", _B + _NUM + r"\s*(mg|mcg|µg|ug|iu|ml)" + _E, 1, 2),
]
_MEAS = [(t, re.compile(rx, re.I), v, u) for t, rx, v, u in _MEAS]

# plausible ranges in canonical units: (low, high)
_RANGE = {"muac": (60, 400), "bmi": (8, 80), "haemoglobin": (2, 25), "hba1c": (3, 20), "glucose": (0.5, 60), "albumin": (5, 60),
          "z_score": (-10, 10), "weight": (0.3, 300), "height": (30, 250), "mass": (0.001, 300), "length": (1, 250),
          "energy": (1, 10000)}


def _canon(kind: str, value: float, unit: str | None) -> tuple[float, str | None, bool]:
    """Returns (value, unit, unit_was_guessed)."""
    u = re.sub(r"\s+", "", unit.lower()) if unit else None
    guessed = u is None
    if kind == "muac":
        if u is None:
            u = "mm" if value >= 40 else "cm"
        return (round(value * 10, 1), "mm", guessed) if u == "cm" else (value, "mm", guessed)
    if kind == "height":
        u = u or "cm"
        return (value * 100 if u == "m" else value / 10 if u == "mm" else value, "cm", guessed)
    if kind == "weight":
        u = u or "kg"
        return (value / 1000 if u == "g" else value, "kg", guessed)
    if kind == "mass":
        return value, "kg", False
    if kind == "haemoglobin":
        if u is None:
            u = "g/dl" if value <= 25 else "g/l"
        return (value / 10 if u == "g/l" else value, "g/dL", guessed)
    if kind == "albumin":
        if u is None:
            u = "g/dl" if value <= 10 else "g/l"
        return (value * 10 if u == "g/dl" else value, "g/L", guessed)
    if kind == "glucose":
        if u is None:
            u = "mg/dl" if value > 30 else "mmol/l"
        return (round(value / 18.0, 2) if u == "mg/dl" else value, "mmol/L", guessed)
    if kind == "hba1c":
        return value, (u or "%"), guessed
    if kind == "energy":
        return (value / 4.184 if u == "kj" else value), "kcal", False
    if kind == "bmi":
        return value, "kg/m²", False
    if kind in ("z_score", "nutrient_amount"):
        return value, ("SD" if kind == "z_score" else "g"), False
    return value, u, False


def _measurements(text: str) -> list[dict]:
    out = []
    for kind, rx, vi, ui in _MEAS:
        for m in rx.finditer(text):
            if kind == "age":
                raw = m.group(1) or m.group("n2")
                unit = (m.group(2) or m.group("u2") or "").lower()
                unit = {"yr": "years", "yrs": "years", "year": "years", "years": "years", "mth": "months", "mths": "months",
                        "month": "months", "months": "months", "wk": "weeks", "wks": "weeks", "week": "weeks", "weeks": "weeks"}.get(unit, unit)
                out.append(_ent(m.start(), m.end(), text, "MEASUREMENT", "age", "measurement", "regex", mtype="age",
                                value=_f(raw), unit=unit, plausible=0 < _f(raw) < 130))
                continue
            raw = m.group(vi).replace(" ", "")
            val, unit, guessed = _canon(kind, _f(raw), m.group(ui) if ui else None)
            lo, hi = _RANGE.get(kind, (None, None))
            ent = _ent(m.start(), m.end(), text, "MEASUREMENT", kind, "measurement", "regex", mtype=kind, value=round(val, 3),
                       unit=unit, plausible=True if lo is None else lo <= val <= hi)
            if guessed and kind in ("muac", "haemoglobin", "albumin", "glucose"):
                ent["unit_guessed"] = True
            out.append(ent)
    return out


# ---------------------------------------------------------------- identifiers
_TITLE_NAME = re.compile(r"(?<!\w)(?:Mr|Mrs|Ms|Miss|Dr|Prof|Mayi|Bambo|Agogo|Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
_LONG = re.compile(r"(?<!\d)\d{9,13}(?!\d)")
_HOSP_ID = re.compile(r"(?<!\w)(?:MRN|ID|Reg(?:istration)?\s*(?:no|number)|File\s*no|OPD\s*no)\s*[:#.]?\s*[A-Z0-9/-]{4,}", re.I)


def _identifiers(text: str) -> list[dict]:
    out = []
    for rx, kind in ((EMAIL, "EMAIL"), (PHONE, "PHONE"), (_HOSP_ID, "RECORD_ID"), (_LONG, "ID_NUMBER"), (_TITLE_NAME, "NAME")):
        for m in rx.finditer(text):
            out.append(_ent(m.start(), m.end(), text, "PHI", kind, "identifier", "regex"))
    return out


def _ent(start: int, end: int, text: str, label: str, norm: str, group: str, source: str, lang: str | None = None, **extra) -> dict:
    e = {"start": start, "end": end, "text": text[start:end] if label != "PHI" else "[redacted]", "label": label, "norm": norm,
         "group": group, "source": source}
    if lang:
        e["lang"] = lang
    e.update(extra)
    return e


# ---------------------------------------------------------------- language
_SENT_END = re.compile(r"[.!?;](?=\s|$)|\n")
_TOKEN = re.compile(r"[A-Za-zŵŴ']+")


def _sentences(text: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for m in _SENT_END.finditer(text):
        if text[start:m.end()].strip():
            out.append((start, m.end()))
        start = m.end()
    if text[start:].strip():
        out.append((start, len(text)))
    return out


def _lang_scores(span_text: str, hits: Counter) -> Counter:
    toks = [t.lower().strip("'") for t in _TOKEN.findall(span_text.replace("’", "'"))]
    score = Counter({"en": sum(t in lx.EN_WORDS for t in toks), "ny": sum(t in lx.NY_WORDS for t in toks)})
    for lang, words in lx.MARKERS.items():
        score[lang] += sum(t in words for t in toks)
    for lang, n in hits.items():
        score[lang] += n
    return +score                                   # drops zero counts


def _lang_of(span_text: str, hits: Counter) -> str:
    """One of en, ny, tum, yao, mixed (two languages in about equal measure) or unknown."""
    ranked = _lang_scores(span_text, hits).most_common()
    if not ranked:
        return "unknown"
    if len(ranked) > 1 and ranked[1][1] / ranked[0][1] >= 0.4:
        return "mixed"
    return ranked[0][0]


# ---------------------------------------------------------------- context (negation, uncertainty, history, family)
def _context(text: str, ent: dict, sent: tuple[int, int]) -> dict:
    before = text[sent[0]:ent["start"]].lower().replace("’", "'")
    after = text[ent["end"]:sent[1]].lower()
    toks = re.findall(r"[a-z/?']+|\?", before)
    window = []
    for t in reversed(toks):            # nearest tokens first, stop at a scope breaker
        if t in lx.SCOPE_BREAKERS:
            break
        window.append(t)
        if len(window) == 6:
            break
    w = " ".join(reversed(window))
    has = lambda cues, hay: any(re.search(r"(?<![a-z])" + re.escape(c) + r"(?![a-z])", hay) for c in cues)  # noqa: E731
    a = " ".join(after.split()[:4])
    return {"negated": has(lx.NEGATION_BEFORE, w) or has(lx.NEGATION_AFTER, a),
            "uncertain": has(lx.UNCERTAIN_BEFORE, w) or "?" in before[-4:],
            "historical": has(lx.HISTORICAL_BEFORE, w),
            "family": has(lx.FAMILY_BEFORE, w)}


# ---------------------------------------------------------------- classification
_DOC_CUES = {
    "clinical_note": [r"\bpatient\b", r"\badmitted|admission\b", r"\bdiagnos[ie]s\b", r"presented with|presenting", r"\bc/o\b|complains? of",
                      r"on examination|\bo/e\b", r"\bplan\b", r"\bward\b", r"discharge[d]?\b", r"\bhistory of\b", r"\bassessment\b", r"\bclinic\b"],
    "lab_report": [r"reference range|normal range", r"\bspecimen\b", r"\bresults?\b", r"mmol\s*/\s*l|g\s*/\s*dl|mg\s*/\s*dl", r"\blaborator(?:y|ies)\b",
                   r"\bsample\b", r"\bflag(?:ged)?\b"],
    "dietary_record": [r"\bbreakfast\b", r"\blunch\b", r"\bdinner\b|\bsupper\b", r"\bsnacks?\b", r"24[\s-]?hours? recall|food diary|food record",
                       r"\bate\b|\beaten\b|\bdrank\b", r"\bmeals?\b", r"\bportions?\b", r"\b(?:ndinadya|tinadya|adadya|anadya|amadya|akudya|ndikudya)\b"],
    "research_text": [r"\bmethods?\b", r"\bconclusions?\b", r"\bparticipants?\b|\brespondents?\b", r"cross[\s-]sectional|\bcohort\b|randomi[sz]ed",
                      r"\bp\s?[<=]\s?0?\.\d+", r"95\s?%\s?ci|odds ratio", r"\bprevalence\b", r"\bn\s?=\s?\d+", r"\bobjective\b|\baim\b|\bbackground\b",
                      r"\bquestionnaire\b|sample size", r"\bfindings\b|\bstudy\b"],
}
_DOC_CUES = {k: [re.compile(p, re.I) for p in v] for k, v in _DOC_CUES.items()}

_TOPICS = {
    "acute_malnutrition": {"norms": {"severe acute malnutrition", "moderate acute malnutrition", "acute malnutrition", "malnutrition",
                                     "undernutrition", "kwashiorkor", "marasmus", "marasmic kwashiorkor", "wasting", "underweight",
                                     "protein-energy malnutrition", "swelling", "oedema", "pitting oedema", "RUTF", "RUSF", "F-75", "F-100",
                                     "ReSoMal", "therapeutic feeding"}, "meas": {"muac"}},
    "stunting_and_growth": {"norms": {"stunting"}, "meas": {"z_score"}},
    "overweight_and_ncd": {"norms": {"overweight", "obesity", "diabetes", "type 2 diabetes", "type 1 diabetes", "gestational diabetes",
                                     "hypertension", "hyperglycaemia", "high-protein diet", "low-salt diet", "diabetic diet"},
                           "meas": {"bmi", "hba1c", "glucose"}},
    "micronutrient_status": {"norms": {"anaemia", "iron deficiency", "vitamin A deficiency", "zinc deficiency", "iodine deficiency",
                                       "folate deficiency", "pellagra", "rickets", "scurvy", "goitre", "iron supplementation",
                                       "vitamin A supplementation", "zinc supplementation", "iron and folic acid", "multivitamin"},
                             "meas": {"haemoglobin"}},
    "maternal_and_infant_feeding": {"norms": {"breastfeeding", "exclusive breastfeeding", "complementary feeding", "breast milk",
                                              "infant formula", "pregnant", "lactating", "breastfeeding mother", "infant", "newborn",
                                              "neonate"}, "meas": set()},
    "infection_and_nutrition": {"norms": {"HIV", "tuberculosis", "malaria", "pneumonia", "sepsis", "diarrhoea"}, "meas": set()},
    "dietary_intake": {"norms": set(), "meas": {"portion", "energy", "nutrient_amount"}, "foods": 2},
}


def _classify(text: str, ents: list[dict]) -> dict:
    scores, evidence = {}, {}
    for kind, cues in _DOC_CUES.items():
        hit = [c.pattern for c in cues if c.search(text)]
        s = float(sum(min(len(c.findall(text)), 3) for c in cues))
        mt = Counter(e["mtype"] for e in ents if e["label"] == "MEASUREMENT")
        foods = sum(e["label"] == "FOOD" for e in ents)
        if kind == "lab_report":
            s += sum(mt[k] for k in ("haemoglobin", "albumin", "glucose", "hba1c"))
        if kind == "dietary_record":
            s += 0.5 * foods + 1.5 * mt["portion"]
        if kind == "clinical_note":
            s += 0.5 * sum(e["label"] in ("CONDITION", "SYMPTOM") for e in ents) + 0.5 * sum(mt[k] for k in ("muac", "weight", "height", "z_score", "bmi"))
        scores[kind] = round(s, 1)
        evidence[kind] = len(hit)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]
    label = top[0] if top[1] >= 2 and top[1] - second[1] >= 1 else "unclear"
    live = [e for e in ents if not e.get("attributes", {}).get("negated")]
    topics = []
    for name, spec in _TOPICS.items():
        norms = {e["norm"] for e in live if e["label"] in ("CONDITION", "SYMPTOM", "INTERVENTION", "POPULATION", "FOOD", "NUTRIENT")
                 and e["norm"] in spec["norms"]}
        meas = {e["mtype"] for e in live if e["label"] == "MEASUREMENT" and e["mtype"] in spec["meas"]}
        n = len(norms) + len(meas)
        if name == "dietary_intake":
            n += len({e["norm"] for e in live if e["label"] == "FOOD"}) // (spec["foods"] + 1)
        if n:
            topics.append({"topic": name, "strength": "strong" if n >= 3 else "moderate" if n == 2 else "weak",
                           "evidence": sorted(norms | meas)})
    topics.sort(key=lambda t: {"strong": 0, "moderate": 1, "weak": 2}[t["strength"]])
    return {"document_type": {"label": label, "scores": dict(ranked), "note": "Cue counts, not probabilities."}, "topics": topics}


# ---------------------------------------------------------------- analysis
def _resolve(cands: list[dict]) -> list[dict]:
    cands = sorted(cands, key=lambda e: (_PRIORITY.get(e["label"], 2), -(e["end"] - e["start"]), e["start"]))
    kept: list[dict] = []
    for c in cands:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda e: e["start"])


def _mask(text: str, phi: list[dict]) -> str:
    out, pos = [], 0
    for e in phi:
        out += [text[pos:e["start"]], "[" + e["norm"] + "]"]
        pos = e["end"]
    out.append(text[pos:])
    return "".join(out)


def _fold(text: str) -> str:
    """Typographic quotes, minus signs and dashes to ASCII. Same length, so offsets stay valid."""
    return text.replace("’", "'").replace("‘", "'").replace("−", "-").replace("–", "-")


def screen_identifiers(text: str) -> list[dict]:
    """Identifiers only (e-mail, phone, record and ID numbers, titled names), without the rest of the analysis. It is the same
    detector, folding and overlap rule that `analyze` uses, so it finds exactly what `analyze` reports as PHI. Used by the LLM
    safety gate to decide whether text may leave the laboratory. Offsets refer to `text`."""
    return _resolve(_identifiers(_fold(text)))


def mask_identifiers(text: str, phi: list[dict]) -> str:
    return _mask(text, phi)


def analyze(text: str) -> dict:
    t = _fold(text)
    cands = _identifiers(t) + _measurements(t)
    for m in _MATCHERS:
        for s, e, item in m.find(t):
            cands.append(_ent(s, e, t, item["label"], item["norm"], item["group"], "lexicon", item["lang"]))
    ents = _resolve(cands)
    sents = _sentences(t)
    for e in ents:
        e["text"] = text[e["start"]:e["end"]] if e["label"] != "PHI" else "[redacted]"
        if e["label"] in ("CONDITION", "SYMPTOM"):
            sent = next((s for s in sents if s[0] <= e["start"] < s[1]), (0, len(t)))
            e["attributes"] = _context(t, e, sent)
    # portions point at the food that follows them ("2 cups of rice")
    for i, e in enumerate(ents):
        if e["label"] == "MEASUREMENT" and e.get("mtype") == "portion" and i + 1 < len(ents):
            n = ents[i + 1]
            if n["label"] == "FOOD" and re.fullmatch(r"\s*(?:of\s+)?", t[e["end"]:n["start"]], re.I):
                e["food"] = n["norm"]
    hits_by_sent: dict[int, Counter] = defaultdict(Counter)
    for e in ents:
        if e.get("lang") in ("ny", "tum", "yao"):
            hits_by_sent[next((k for k, s in enumerate(sents) if s[0] <= e["start"] < s[1]), 0)][e["lang"]] += 1
    sent_langs = [{"start": s[0], "end": s[1], "language": _lang_of(t[s[0]:s[1]], hits_by_sent[k])} for k, s in enumerate(sents)]
    total_hits = sum(hits_by_sent.values(), Counter())
    overall = _lang_of(t, total_hits)
    detected = [{"language": k, "evidence": n} for k, n in _lang_scores(t, total_hits).most_common()]
    seen = {s["language"] for s in sent_langs} - {"unknown"}
    phi = [e for e in ents if e["label"] == "PHI"]
    rec = _record(ents)
    warnings = []
    if phi:
        warnings.append("Identifiers were found and masked in `masked_text`. Do not process identifiable notes; remove identifiers at the source.")
    if any(e["label"] == "MEASUREMENT" and not e.get("plausible", True) for e in ents):
        warnings.append("Some measurements are outside plausible ranges and may be typing errors or wrong units.")
    if any(e.get("unit_guessed") for e in ents):
        warnings.append("Some units were missing and were guessed from the value (flagged `unit_guessed`). Check them.")
    warnings.append("Personal names cannot be detected reliably; only titled names (Mr, Dr, Mayi, Bambo, ...) are flagged.")
    return {"engine": ENGINE_VERSION, "n_chars": len(text), "language": {"overall": overall, "code_switching": len(seen) > 1 or "mixed" in seen,
                                                                       "detected": detected, "sentences": sent_langs},
            "entities": ents, "record": rec, "classification": _classify(t, ents),
            "identifiers": {"found": len(phi), "types": dict(Counter(e["norm"] for e in phi)), "masked_text": _mask(text, phi) if phi else None},
            "warnings": warnings}


def _record(ents: list[dict]) -> dict:
    def collect(label: str):
        pres, neg, unc = Counter(), Counter(), Counter()
        for e in ents:
            if e["label"] != label:
                continue
            a = e.get("attributes", {})
            (neg if a.get("negated") else unc if a.get("uncertain") else pres)[e["norm"]] += 1
        return pres, neg, unc

    cp, cn, cu = collect("CONDITION")
    sp, sn, su = collect("SYMPTOM")
    foods = Counter(e["norm"] for e in ents if e["label"] == "FOOD")
    groups = Counter(e["group"] for e in ents if e["label"] == "FOOD")
    meas = defaultdict(list)
    for e in ents:
        if e["label"] == "MEASUREMENT":
            item = {"value": e["value"], "unit": e["unit"], "text": e["text"], "plausible": e.get("plausible", True)}
            if e.get("food"):
                item["food"] = e["food"]
            meas[e["mtype"]].append(item)
    lst = lambda c: [{"term": k, "count": v} for k, v in c.most_common()]  # noqa: E731
    return {"conditions": {"present": lst(cp), "negated": lst(cn), "uncertain": lst(cu)},
            "symptoms": {"present": lst(sp), "negated": lst(sn), "uncertain": lst(su)},
            "foods": lst(foods), "food_groups": lst(groups),
            "interventions": lst(Counter(e["norm"] for e in ents if e["label"] == "INTERVENTION")),
            "nutrients": lst(Counter(e["norm"] for e in ents if e["label"] == "NUTRIENT")),
            "populations": lst(Counter(e["norm"] for e in ents if e["label"] == "POPULATION")),
            "measurements": dict(meas)}
