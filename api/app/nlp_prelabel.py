"""Pre-label export and annotation template for the NLP laboratory.

Speakers and clinicians correct guesses faster than they write annotations from nothing. `prelabel` reads plain sentences (a .txt file
with one sentence per line, or a CSV with a `text` column) or a document (.docx, .pptx, .odt, .pdf with selectable text, .html, .md,
split into sentences by `nlp_extract`), runs the rule-based engine on each, and returns a CSV in the
one-row-per-document corpus format (`text`, `lang`, `entities` as `surface|LABEL; surface|LABEL`). After correction the same file can
be uploaded to /v1/nlp/corpora as it is: extra columns are ignored and never stored.

The guesses are only guesses. Texts are processed in memory and not stored. Identifiers (titled names, phones, e-mails, record
and ID numbers) are removed by the tool itself: each one is replaced by a placeholder such as [PHONE] before anything is guessed or
written, and the notes column and the report name the line and the identifier type only, never the text. Corpus upload still refuses
a file that holds identifiers, so the CSV that leaves this tool is safe to upload.

`lang` may be a single code (en, ny, tum, yao, mixed) or a pair joined by `+`, for example `en+ny`: each sentence is then assigned to
one of the two by the engine's evidence."""
import csv
import io
import re
from collections import Counter

from . import nlp_corpus, nlp_extract, nlp_formats, nlp_lab

MAX_LINES = 5000
COLUMNS = ["doc_id", "lang", "text", "entities", "reviewed", "notes"]


class PrelabelError(ValueError):
    pass


def read_sentences(raw: bytes, filename: str = "") -> list[tuple[str, str]]:
    """Returns [(doc_id, text)] from a .txt (one sentence per line), a .csv (a `text` column, optional `doc_id`) or a document."""
    out: list[tuple[str, str]] = []
    if nlp_extract.is_document(filename):
        try:
            sentences = nlp_extract.split_sentences(nlp_extract.extract_text(raw, filename))
        except nlp_extract.ExtractError as e:
            raise PrelabelError(str(e))
        out = [(f"s-{n}", s) for n, s in enumerate(sentences, start=1)]
        if not out:
            raise PrelabelError("No sentences were found in the document.")
        if len(out) > MAX_LINES:
            raise PrelabelError(f"The document has {len(out)} sentences and the limit is {MAX_LINES}. Split it into parts.")
        return out
    body = nlp_formats._decode(raw)
    if filename.lower().endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(body, newline="")))
        if not rows or "text" not in {(k or "").strip().lower() for k in rows[0]}:
            raise PrelabelError("The CSV needs a text column.")
        for n, r in enumerate(rows, start=2):
            r = {(k or "").strip().lower(): (v or "") for k, v in r.items()}
            if r["text"].strip():
                out.append(((r.get("doc_id") or "").strip() or f"row-{n}", r["text"].strip()))
    else:
        for n, line in enumerate(body.splitlines(), start=1):
            if line.strip():
                out.append((f"line-{n}", line.strip()))
    if not out:
        raise PrelabelError("The file has no sentences.")
    if len(out) > MAX_LINES:
        raise PrelabelError(f"The file has more than {MAX_LINES} sentences. Split it.")
    return out


def _guess_entities(text: str, ents: list[dict]) -> tuple[str, list[str]]:
    """Builds the `entities` cell. The corpus CSV format matches each listed word as a whole word, every time, so a word that would
    produce overlapping or conflicting spans is left out and named in the notes instead of being written wrongly."""
    notes: list[str] = []
    seen: dict[str, tuple[str, str]] = {}            # lower-case surface -> (surface as written, label)
    for e in ents:
        if e["label"] == "PHI":
            continue
        surface = text[e["start"]:e["end"]].strip()
        if not surface or ";" in surface:
            notes.append(f'"{surface[:30]}" skipped (contains a semicolon)')
            continue
        key = surface.lower()
        if key in seen and seen[key][1] != e["label"]:
            notes.append(f'"{surface[:30]}" guessed as both {seen[key][1]} and {e["label"]}: kept {seen[key][1]}')
        seen.setdefault(key, (surface, e["label"]))
    items = sorted(seen.values(), key=lambda sl: -len(sl[0]))
    kept: list[tuple[str, str]] = []
    taken: list[tuple[int, int]] = []
    for surface, label in items:
        spans = nlp_formats._surface_spans(text, surface)
        if any(not (b <= s or a >= e) for s, e in taken for a, b in spans):
            notes.append(f'"{surface[:30]}|{label}" left out: it also occurs inside a longer term, add it by hand if needed')
            continue
        taken.extend(spans)
        kept.append((surface, label))
    kept.sort(key=lambda sl: text.lower().find(sl[0].lower()))
    return "; ".join(f"{s}|{lab}" for s, lab in kept), notes


_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


def parse_lang(lang: str | None) -> tuple[str | None, tuple[str, ...]]:
    """Returns (single code or None, allowed codes for a pair). Raises PrelabelError for an unknown code."""
    lang = (lang or "").strip().lower()
    if not lang:
        return None, ()
    parts = tuple(dict.fromkeys(p.strip() for p in lang.split("+") if p.strip()))
    bad = [p for p in parts if p not in nlp_corpus.LANG_CODES]
    if bad or not parts or len(parts) > 3:
        raise PrelabelError("lang must be one of " + ", ".join(sorted(nlp_corpus.LANG_CODES)) + ", or a pair such as en+ny")
    return (parts[0], ()) if len(parts) == 1 else (None, parts)


def _redact(text: str) -> tuple[str, Counter]:
    """Replaces identifiers with [TYPE] placeholders. Repeats until none are left (a placeholder never matches a detector)."""
    types: Counter = Counter()
    for _ in range(3):
        phi = nlp_lab.screen_identifiers(text)
        if not phi:
            return text, types
        types.update(e["norm"] for e in phi)
        text = nlp_lab.mask_identifiers(nlp_lab._fold(text), phi)
    return (text, types) if not nlp_lab.screen_identifiers(text) else ("", types)


def _pick_lang(a: dict, single: str | None, pair: tuple[str, ...]) -> tuple[str, str | None]:
    """Returns (code, note)."""
    if single:
        return single, None
    guessed = a["language"]["overall"]
    if pair:
        if guessed in pair:
            return guessed, None
        ev = {d["language"]: d["evidence"] for d in a["language"]["detected"]}
        best = max(pair, key=lambda c: ev.get(c, 0))
        if ev.get(best, 0) > 0:
            return best, "language mix detected: check lang" if guessed == "mixed" else None
        return "", f"language could not be guessed: fill in lang ({' or '.join(pair)})"
    use = guessed if guessed in nlp_corpus.LANG_CODES else ""
    if not use:
        return "", "language could not be guessed: fill in lang (en, ny, tum, yao or mixed)"
    return use, "language mix detected: check lang" if a["language"]["code_switching"] else None


def prelabel(raw: bytes, filename: str = "", lang: str | None = None) -> dict:
    """Returns {csv, report}. Identifiers are removed (replaced by placeholders) before anything else happens. Raises PrelabelError
    for an unreadable file or an unknown language."""
    single, pair = parse_lang(lang)
    sentences = read_sentences(raw, filename)
    red_lines, red_types, dropped = [], Counter(), []
    analysed = []
    for doc_id, text in sentences:
        if len(text) > nlp_lab.MAX_TEXT_CHARS:
            raise PrelabelError(f"{doc_id}: text is longer than {nlp_lab.MAX_TEXT_CHARS} characters.")
        clean, types = _redact(text)
        if types:
            red_lines.append(doc_id)
            red_types.update(types)
        if not clean.strip():
            dropped.append(doc_id)
            continue
        a = nlp_lab.analyze(clean)
        lang_a = nlp_lab.analyze(_PLACEHOLDER.sub(" ", clean)) if types and (not single) else a
        analysed.append((doc_id, clean, a, lang_a, dict(types)))
    if not analysed:
        raise PrelabelError("Nothing was left after identifiers were removed.")
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(COLUMNS)
    langs, with_ents, total_ents = Counter(), 0, 0
    for doc_id, text, a, lang_a, types in analysed:
        use, lnote = _pick_lang(lang_a, single, pair)
        cell, notes = _guess_entities(text, a["entities"])
        if types:
            notes.insert(0, "identifiers removed: " + ", ".join(sorted(types)) + " (check the sentence still reads correctly)")
        if lnote:
            notes.append(lnote)
        langs[use or "unset"] += 1
        n = cell.count("|")
        with_ents += bool(n)
        total_ents += n
        w.writerow([doc_id, use, text, cell, "", " | ".join(notes)])
    report = {"engine": nlp_lab.ENGINE_VERSION, "documents": len(analysed), "with_guesses": with_ents, "guessed_entities": total_ents,
              "by_language": dict(langs), "identifier_lines_removed": len(red_lines), "identifier_types_removed": dict(red_types),
              "lines_dropped": len(dropped)}
    return {"csv": out.getvalue(), "report": report}


# ---------------------------------------------------------------- template
# Example rows only. Chichewa and English are written by the developers; the Tumbuka row uses only terms already in the starter list
# (unreviewed); there is no Yao example sentence because nobody who speaks Yao has supplied one. Every example is for deleting.
_TEMPLATE_ROWS = [
    ["example-en", "en", "The child ate rice and beans.", "child|POPULATION; rice|FOOD; beans|FOOD", "", "EXAMPLE: delete this row"],
    ["example-ny", "ny", "Mwana ali ndi kutupa.", "Mwana|POPULATION; kutupa|SYMPTOM", "", "EXAMPLE: delete this row"],
    ["example-tum", "tum", "Ŵana wakulya somba.", "Ŵana|POPULATION; somba|FOOD", "", "EXAMPLE: delete this row (Tumbuka terms are unreviewed)"],
    ["example-yao", "yao", "<<replace with a real Yao sentence>>", "", "", "EXAMPLE: a Yao speaker must supply the sentence and its terms"],
]


def template_csv() -> str:
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(COLUMNS)
    w.writerows(_TEMPLATE_ROWS)
    return out.getvalue()
