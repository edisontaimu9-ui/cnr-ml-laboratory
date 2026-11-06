"""LLM intelligence, phase 4: structured facts from a de-identified clinical note, by the rule engine.

The facts come from `nlp_lab.analyze` (the same transparent, rule-based engine as /v1/nlp/analyze), not from a language model. Each fact is
one thing the note says: a measurement (type, value, unit), or a condition, symptom, food, intervention, nutrient or population term with
its assertion status (present, negated, uncertain, historical, family). Facts carry the exact span of the note they came from (`evidence`),
so a clinician can check each one against the source.

Only what the engine found can be summarised. A term that is not in the lexicon, or a number the patterns do not recognise (for example
"3 sachets per day"), is simply not a fact; the summary cannot mention it and the response says so (`extraction_limits`).
"""
import re
from collections import Counter

from . import nlp_lab

MAX_FACTS = 60
_KIND = {"MEASUREMENT": "measurement", "CONDITION": "condition", "SYMPTOM": "symptom", "FOOD": "food", "INTERVENTION": "intervention",
         "NUTRIENT": "nutrient", "POPULATION": "population"}
_ORDER = {"measurement": 0, "condition": 1, "symptom": 2, "intervention": 3, "nutrient": 4, "food": 5, "population": 6}

EXTRACTION_LIMITS = [
    "Only terms in the rule engine's lexicon and measurements its patterns recognise are extracted; anything else in the note is not a fact and cannot appear in the summary.",
    "Negation, uncertainty, history and family context are heuristics and can be wrong; check each fact's status against the note.",
    "The lexicon and patterns are developer-written and not clinically reviewed.",
]


def numbers(text: str) -> set[float]:
    """Every number written in digits in the text, as floats ('1,5' is read as 1.5, as the measurement patterns do)."""
    out = set()
    for m in re.finditer(r"(?<![\w.])(\d+(?:[.,]\d+)?)", text or ""):
        try:
            out.add(float(m.group(1).replace(",", ".")))
        except ValueError:
            pass
    return out


def _status(attrs: dict | None) -> str:
    a = attrs or {}
    if a.get("negated"):
        return "negated"
    if a.get("uncertain"):
        return "uncertain"
    if a.get("family"):
        return "family"
    if a.get("historical"):
        return "historical"
    return "present"


def _num(v: float):
    return int(v) if float(v).is_integer() else round(float(v), 2)


def extract(text: str) -> dict:
    """Facts from one note (identifiers must already have been refused or masked). Returns {facts, counts, language, warnings, truncated}."""
    a = nlp_lab.analyze(text)
    raw = []
    for e in a["entities"]:
        kind = _KIND.get(e["label"])
        if not kind:
            continue                                  # PHI never becomes a fact
        f = {"kind": kind, "term": e["norm"], "evidence": e["text"][:120], "group": e.get("group")}
        if kind == "measurement":
            f.update(mtype=e.get("mtype"), value=_num(e["value"]), unit=e.get("unit"), plausible=bool(e.get("plausible", True)))
            if e.get("unit_guessed"):
                f["unit_guessed"] = True
            if e.get("food"):
                f["food"] = e["food"]
            f["status"] = "present"
        else:
            f["status"] = _status(e.get("attributes"))
        raw.append(f)
    # one fact per distinct (kind, term, status, value, unit); repeated mentions are counted
    seen: dict[tuple, dict] = {}
    for f in raw:
        k = (f["kind"], f["term"], f["status"], f.get("mtype"), f.get("value"), f.get("unit"))
        if k in seen:
            seen[k]["mentions"] += 1
        else:
            seen[k] = {**f, "mentions": 1}
    facts = sorted(seen.values(), key=lambda f: _ORDER[f["kind"]])
    truncated = len(facts) > MAX_FACTS
    facts = facts[:MAX_FACTS]
    for i, f in enumerate(facts, start=1):
        f["id"] = f"F{i}"
    warnings = [w for w in a["warnings"] if not w.startswith("Identifiers were found")]
    if truncated:
        warnings.append(f"More than {MAX_FACTS} facts were found; only the first {MAX_FACTS} are kept.")
    if any(f.get("plausible") is False for f in facts):
        warnings.append("A measurement is outside its plausible range: check the note for a typing error or a wrong unit.")
    return {"facts": facts, "counts": dict(Counter(f["kind"] for f in facts)), "language": a["language"]["overall"], "warnings": warnings,
            "truncated": truncated, "engine": nlp_lab.ENGINE_VERSION}


def model_view(facts: list[dict]) -> list[dict]:
    """What the model is shown: lexicon-controlled terms, numbers, units and statuses. Never the note, and never an evidence span, so no free
    text from the note (and so no instruction hidden in it) can reach the model."""
    out = []
    for f in facts:
        v = {"id": f["id"], "kind": f["kind"], "term": f["term"], "status": f["status"]}
        if f["kind"] == "measurement":
            v.update(type=f.get("mtype"), value=f.get("value"), unit=f.get("unit"))
            if f.get("food"):
                v["food"] = f["food"]
        out.append(v)
    return out
