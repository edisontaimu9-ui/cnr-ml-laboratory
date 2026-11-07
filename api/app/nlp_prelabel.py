"""Pre-label export and annotation template for the NLP laboratory.

Speakers and clinicians correct guesses faster than they write annotations from nothing. `prelabel` reads plain sentences (a .txt file
with one sentence per line, or a CSV with a `text` column) or a document (.docx, .pptx, .odt, .pdf with selectable text, .html, .md,
split into sentences by `nlp_extract`), runs the rule-based engine on each, and returns a CSV in the
one-row-per-document corpus format (`text`, `lang`, `entities` as `surface|LABEL; surface|LABEL`). After correction the same file can
be uploaded to /v1/nlp/corpora as it is: extra columns are ignored and never stored.

The guesses are only guesses. Texts are processed in memory and not stored. A file with identifiers is refused whole, in line with
corpus upload, and the refusal names line numbers and identifier types only, never the text."""
import csv
import io
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


def prelabel(raw: bytes, filename: str = "", lang: str | None = None) -> dict:
    """Returns {csv, report}. Raises PrelabelError for an unreadable file or when identifiers are found."""
    sentences = read_sentences(raw, filename)
    id_lines, id_types = [], Counter()
    analysed = []
    for doc_id, text in sentences:
        if len(text) > nlp_lab.MAX_TEXT_CHARS:
            raise PrelabelError(f"{doc_id}: text is longer than {nlp_lab.MAX_TEXT_CHARS} characters.")
        a = nlp_lab.analyze(text)
        if a["identifiers"]["found"]:
            id_lines.append(doc_id)
            id_types.update(a["identifiers"]["types"])
        analysed.append((doc_id, text, a))
    if id_lines:
        raise PrelabelError("Identifiers were found, so nothing was exported. Remove them at the source and upload again. "
                            f"Lines: {', '.join(id_lines[:25])}{' ...' if len(id_lines) > 25 else ''}. Types: {dict(id_types)}.")
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(COLUMNS)
    langs, with_ents, total_ents = Counter(), 0, 0
    for doc_id, text, a in analysed:
        guessed = a["language"]["overall"]
        use = lang or (guessed if guessed in nlp_corpus.LANG_CODES else "")
        cell, notes = _guess_entities(text, a["entities"])
        if not use:
            notes.append("language could not be guessed: fill in lang (en, ny, tum, yao or mixed)")
        elif not lang and a["language"]["code_switching"]:
            notes.append("language mix detected: check lang")
        langs[use or "unset"] += 1
        n = cell.count("|")
        with_ents += bool(n)
        total_ents += n
        w.writerow([doc_id, use, text, cell, "", " | ".join(notes)])
    report = {"engine": nlp_lab.ENGINE_VERSION, "documents": len(analysed), "with_guesses": with_ents, "guessed_entities": total_ents,
              "by_language": dict(langs)}
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
