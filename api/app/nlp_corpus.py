"""Annotated text corpora for training the NLP laboratory (stage 1: upload, validation, safe storage).

Format: JSON Lines, one document per line:
    {"text": "Mwana ali ndi kutupa.", "lang": "ny", "entities": [{"start": 0, "end": 5, "label": "POPULATION"}, ...]}
Offsets are character positions in `text` (Python style, end exclusive). Only `text`, `lang` and `entities` are kept; any other
field is dropped so stray identifiers in extra columns never reach storage. A file that contains identifiers or any invalid line
is refused as a whole, and the report names line numbers and identifier types only, never the text."""
import hashlib
import json
from collections import Counter

from . import nlp_lab, nlp_lexicon as lx

LANG_CODES = {"en", "ny", "tum", "yao", "mixed"}
TRAIN_LABELS = {k for k in lx.LABELS if k != "PHI"}          # PHI is detected and refused, never a training label
MAX_DOCS = 50_000
MAX_ERRORS_SHOWN = 25
MIN_EXAMPLES_PER_LABEL = 50                                   # below this a trained model is not trustworthy (a warning, not a refusal)
MIN_DOCS_PER_LANG = 100


class CorpusError(ValueError):
    pass


def _check_entities(text: str, ents, where: str, errors: list[str]) -> list[dict] | None:
    if not isinstance(ents, list):
        errors.append(f"{where}: entities must be a list")
        return None
    out, ok = [], True
    for k, e in enumerate(ents, start=1):
        if not isinstance(e, dict) or not isinstance(e.get("start"), int) or not isinstance(e.get("end"), int) or isinstance(e.get("start"), bool) or isinstance(e.get("end"), bool):
            errors.append(f"{where}: entity {k} needs integer start and end")
            ok = False
            continue
        s, en, label = e["start"], e["end"], str(e.get("label", "")).upper()
        if label not in TRAIN_LABELS:
            errors.append(f"{where}: entity {k} label must be one of " + ", ".join(sorted(TRAIN_LABELS)))
            ok = False
        elif not 0 <= s < en <= len(text):
            errors.append(f"{where}: entity {k} offsets {s}-{en} are outside the text (length {len(text)})")
            ok = False
        elif not text[s:en].strip() or text[s:en] != text[s:en].strip():
            errors.append(f"{where}: entity {k} is empty or has spaces at its edges")
            ok = False
        else:
            out.append({"start": s, "end": en, "label": label})
    if not ok:
        return None
    out.sort(key=lambda x: (x["start"], x["end"]))
    for a, b in zip(out, out[1:]):
        if b["start"] < a["end"]:
            errors.append(f"{where}: entities overlap at characters {b['start']}-{min(a['end'], b['end'])}")
            return None
    return out


def parse(raw: bytes) -> dict:
    """Validates a JSONL corpus. Returns {valid, errors, identifier_lines, docs, report}. `docs` holds the cleaned, de-duplicated
    documents and is empty unless the file is valid. Never raises for bad content, only for an unreadable file."""
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        raise CorpusError("The file is not UTF-8 text.")
    lines = [(n, ln) for n, ln in enumerate(lines, start=1) if ln.strip()]
    if not lines:
        raise CorpusError("The file has no documents.")
    if len(lines) > MAX_DOCS:
        raise CorpusError(f"The file has more than {MAX_DOCS} documents. Split it.")
    errors: list[str] = []
    id_types: Counter = Counter()
    id_lines: list[int] = []
    docs, seen, dupes = [], set(), 0
    for n, ln in lines:
        where = f"line {n}"
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            errors.append(f"{where}: not valid JSON")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
            errors.append(f"{where}: needs a text string")
            continue
        text, lang = obj["text"], obj.get("lang")
        if not text.strip() or len(text) > nlp_lab.MAX_TEXT_CHARS:
            errors.append(f"{where}: text is empty or longer than {nlp_lab.MAX_TEXT_CHARS} characters")
            continue
        if lang not in LANG_CODES:
            errors.append(f"{where}: lang must be one of " + ", ".join(sorted(LANG_CODES)))
            continue
        phi = nlp_lab._identifiers(text)
        if phi:
            id_lines.append(n)
            id_types.update(e["norm"] for e in phi)
            continue
        ents = _check_entities(text, obj.get("entities", []), where, errors)
        if ents is None:
            continue
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        docs.append({"text": text, "lang": lang, "entities": ents})
    by_lang = Counter(d["lang"] for d in docs)
    by_label = Counter(e["label"] for d in docs for e in d["entities"])
    label_by_lang: dict[str, Counter] = {}
    for d in docs:
        label_by_lang.setdefault(d["lang"], Counter()).update(e["label"] for e in d["entities"])
    warnings = []
    for lang, n in sorted(by_lang.items()):
        if n < MIN_DOCS_PER_LANG:
            warnings.append(f"Only {n} {lang} documents; aim for at least {MIN_DOCS_PER_LANG} per language before training.")
    for label in sorted(TRAIN_LABELS):
        if by_label[label] < MIN_EXAMPLES_PER_LABEL:
            warnings.append(f"{label} has {by_label[label]} examples; at least {MIN_EXAMPLES_PER_LABEL} are needed for a trustworthy model.")
    if dupes:
        warnings.append(f"{dupes} duplicate texts were removed.")
    valid = not errors and not id_lines
    report = {"lines_read": len(lines), "documents": len(docs), "duplicates_removed": dupes, "entities": sum(by_label.values()),
              "by_language": dict(by_lang), "by_label": dict(by_label),
              "label_by_language": {k: dict(v) for k, v in label_by_lang.items()}, "warnings": warnings}
    return {"valid": valid, "errors": errors[:MAX_ERRORS_SHOWN], "errors_total": len(errors),
            "identifier_lines": id_lines[:MAX_ERRORS_SHOWN], "identifier_lines_total": len(id_lines),
            "identifier_types": dict(id_types), "docs": docs if valid else [], "report": report}


def serialise(docs: list[dict]) -> bytes:
    return ("\n".join(json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in docs) + "\n").encode("utf-8")
