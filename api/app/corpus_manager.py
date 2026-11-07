"""Corpus Manager & Dataset Builder: the pure logic (no database, no HTTP), so every rule can be tested on its own.

Workflow (each step is separate and nothing trains a model):
  1. A corpus is created for one slice of nutrition text (English, Chichewa, clinical nutrition, foods and nutrients, community
     nutrition, research documents). CSV, JSONL and TXT files are added to it as uploads.
  2. People annotate it (language, document category, entities). Quality checks find duplicates and missing or invalid annotations.
  3. A reviewer approves it. Approval freezes a snapshot; later edits never change a snapshot.
  4. Approved snapshots are merged, on request, into one versioned master dataset. Corpora are never modified by a merge.
  5. A master version is exported (CSV, JSONL, Parquet) or turned into a task dataset (NER, text classification, language id).

Privacy follows the rest of the lab: text with identifiers (titled names, phones, emails, record or ID numbers) is refused whole and
reports give record locations and types, never text. Only the fields the lab uses are kept, so stray columns never reach storage.
"""
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict

from . import nlp_corpus, nlp_extract, nlp_formats, nlp_lab

DOMAINS = {
    "english": "English-language nutrition text",
    "chichewa": "Chichewa nutrition text",
    "clinical_nutrition": "Clinical nutrition: notes, reports, feeding and treatment records",
    "food_nutrients": "Foods, dishes and nutrients",
    "community_nutrition": "Community nutrition: screening, counselling, health talks",
    "research_documents": "Research documents: articles, reports, protocols",
    "other": "Another kind of nutrition text",
}
TASKS = {"ner": "Entity recognition", "classification": "Text classification (document category)", "langid": "Language identification"}
STATUSES = ("unannotated", "annotated", "reviewed", "rejected")
CORPUS_STATES = ("draft", "in_review", "approved", "rejected")
UPLOAD_FORMATS = {
    "csv": "Columns: text (required), lang, category, entities, source, id. `entities` is `word|LABEL; word|LABEL` or a JSON list.",
    "tsv": "Same columns as CSV, separated by tabs.",
    "xlsx": "Excel workbook: the first sheet, with a header row and the same columns as CSV.",
    "jsonl": "One JSON object per line: text (required), lang, category, entities, source, id. Entities are objects or [start, end, label].",
    "json": "One JSON array of such objects, or an object with a `documents` (or `records`) list.",
    "txt": "One text per line. No annotations: choose the language for the upload, and annotate afterwards.",
    "document": "Word (.docx), PDF with selectable text, PowerPoint (.pptx), OpenDocument (.odt), HTML (.html) or Markdown (.md). The text is split into sentences, each becoming an unannotated record.",
    "bio": "BIO / IOB2 tagging: one `token TAG` per line, blank line between sentences (annotation tools such as doccano and Label Studio export this).",
    "conll": "CoNLL-2003 style columns: the last column is the tag.",
    "docbin": "spaCy DocBin (.spacy) with entity annotations.",
    "brat": "BRAT standoff: a ZIP of paired name.txt and name.ann files.",
}
EXTENSIONS = {".csv": "csv", ".tsv": "tsv", ".tab": "tsv", ".xlsx": "xlsx", ".xlsm": "xlsx", ".jsonl": "jsonl", ".ndjson": "jsonl", ".json": "json",
              ".txt": "txt", ".bio": "bio", ".iob": "bio", ".iob2": "bio", ".conll": "conll", ".conll03": "conll", ".spacy": "docbin", ".zip": "brat",
              **{e: "document" for e in nlp_extract.DOCUMENT_EXTENSIONS}}
FORMAT_ALIASES = {"ndjson": "jsonl", "iob": "bio", "iob2": "bio", "conll2003": "conll", "spacy": "docbin", "excel": "xlsx", "tab": "tsv",
                  **{e[1:]: "document" for e in nlp_extract.DOCUMENT_EXTENSIONS}}
ACCEPT = ",".join(sorted(EXTENSIONS))
LANG_CODES = nlp_corpus.LANG_CODES
ENTITY_LABELS = nlp_corpus.TRAIN_LABELS
CATEGORIES = nlp_corpus.CATEGORIES
LANG_ALIASES = {"english": "en", "eng": "en", "chichewa": "ny", "chewa": "ny", "nya": "ny", "chinyanja": "ny", "tumbuka": "tum", "chitumbuka": "tum",
                "chiyao": "yao", "yawo": "yao"}
MAX_UPLOAD_RECORDS = 50_000
MAX_CORPUS_RECORDS = 200_000
MIN_CHARS = 2
MAX_CHARS = nlp_lab.MAX_TEXT_CHARS
MAX_SHOWN = 25
MIN_EXAMPLES_PER_LABEL = nlp_corpus.MIN_EXAMPLES_PER_LABEL
MIN_DOCS_PER_CLASS = 20
MIN_DOCS_PER_LANG = nlp_corpus.MIN_DOCS_PER_LANG
FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


class CorpusManagerError(ValueError):
    pass


class MergeConflict(CorpusManagerError):
    def __init__(self, message: str, conflicts: list[dict]):
        super().__init__(message)
        self.conflicts = conflicts


# ------------------------------------------------------------------ small helpers
def normalise_text(text: str) -> str:
    """Case, punctuation and spacing folded away, so 'Rice.' and ' rice ' count as the same text."""
    t = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", t)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def clean_lang(v) -> str | None:
    s = str(v).strip().lower() if v not in (None, "") else ""
    return LANG_ALIASES.get(s, s)[:16] or None


def clean_category(v) -> str | None:
    s = str(v).strip().lower().replace(" ", "_") if v not in (None, "") else ""
    return s[:64] or None


def clean_status(v) -> str | None:
    s = str(v).strip().lower() if v not in (None, "") else ""
    return s if s in STATUSES else None


def _clean_source(v) -> str | None:
    s = str(v).strip() if v not in (None, "") else ""
    return s[:120] or None


def clean_entities(raw) -> tuple[list[dict] | None, str | None]:
    """Reads an entity list into [{start, end, label}]. Returns (None, note) when it cannot be read at all. Whether the entities are
    *valid* (labels, offsets, overlaps) is judged separately by `check_entities`, so a bad annotation is kept and flagged, not lost."""
    if raw is None:
        return None, None
    if not isinstance(raw, list):
        return None, "the entities could not be read"
    out = []
    for e in raw:
        if isinstance(e, (list, tuple)) and len(e) == 3:
            e = {"start": e[0], "end": e[1], "label": e[2]}
        if (not isinstance(e, dict) or isinstance(e.get("start"), bool) or isinstance(e.get("end"), bool)
                or not isinstance(e.get("start"), int) or not isinstance(e.get("end"), int)):
            return None, "an entity needs whole-number start and end"
        out.append({"start": e["start"], "end": e["end"], "label": str(e.get("label", "")).strip().upper()[:40]})
    return out, None


def check_entities(text: str, ents: list[dict]) -> list[str]:
    """Why a (readable) entity list is not valid. Empty list means valid."""
    problems = []
    for k, e in enumerate(ents, start=1):
        s, en, label = e["start"], e["end"], e["label"]
        if label not in ENTITY_LABELS:
            problems.append(f"entity {k}: label {label or '(empty)'} is not one of " + ", ".join(sorted(ENTITY_LABELS)))
        elif not 0 <= s < en <= len(text):
            problems.append(f"entity {k}: offsets {s}-{en} are outside the text (length {len(text)})")
        elif not text[s:en].strip() or text[s:en] != text[s:en].strip():
            problems.append(f"entity {k}: empty or has spaces at its edges")
    if not problems:
        ordered = sorted(ents, key=lambda x: (x["start"], x["end"]))
        for a, b in zip(ordered, ordered[1:]):
            if b["start"] < a["end"]:
                problems.append(f"entities overlap at characters {b['start']}-{min(a['end'], b['end'])}")
                break
    return problems


# ------------------------------------------------------------------ record checks
def record_issues(rec: dict, tasks) -> list[tuple[str, str]]:
    """(code, message) for everything wrong or missing in one record. `rec` needs text, lang, category, entities, parse_note.
    Codes starting `missing_` are gaps in the annotation; codes starting `invalid_` are annotations that are wrong."""
    out = []
    tasks = set(tasks)
    lang = rec.get("lang")
    if not lang:
        out.append(("missing_language", "no language"))
    elif lang not in LANG_CODES:
        out.append(("invalid_language", f"language {lang} is not one of " + ", ".join(sorted(LANG_CODES))))
    cat = rec.get("category")
    if cat is None:
        if "classification" in tasks:
            out.append(("missing_category", "no document category"))
    elif cat not in CATEGORIES:
        out.append(("invalid_category", f"category {cat} is not one of " + ", ".join(sorted(CATEGORIES))))
    ents = rec.get("entities")
    if rec.get("parse_note"):
        out.append(("invalid_entities", rec["parse_note"]))
    elif ents is None:
        if "ner" in tasks:
            out.append(("missing_entities", "entities not annotated (an empty list confirms there are none)"))
    else:
        for msg in check_entities(rec["text"], ents):
            out.append(("invalid_entities", msg))
    return out


def is_complete(rec: dict, tasks) -> bool:
    return not record_issues(rec, tasks)


# ------------------------------------------------------------------ reading uploads
def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise CorpusManagerError("The file is not UTF-8 text.")


def detect_format(fmt: str | None, filename: str) -> str:
    fmt = (fmt or "auto").strip().lower()
    if fmt != "auto":
        fmt = FORMAT_ALIASES.get(fmt, fmt)
        if fmt not in UPLOAD_FORMATS:
            raise CorpusManagerError("format must be auto or one of " + ", ".join(UPLOAD_FORMATS) + ".")
        return fmt
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in EXTENSIONS:
        return EXTENSIONS[ext]
    raise CorpusManagerError("This file type is not supported. Use " + ", ".join(sorted(EXTENSIONS)) + ", or set format.")


def _entities_from_cell(text: str, cell: str) -> tuple[list[dict] | None, str | None]:
    cell = cell.strip()
    if not cell:
        return None, None
    if cell.startswith("["):
        try:
            return clean_entities(json.loads(cell))
        except json.JSONDecodeError:
            return None, "the entities cell is not valid JSON"
    ents = []
    for item in (c for c in cell.split(";") if c.strip()):
        surface, _, label = item.strip().rpartition("|")
        if not surface.strip() or not label.strip():
            return None, 'the entities cell needs "text|LABEL" items separated by semicolons'
        spans = nlp_formats._surface_spans(text, surface.strip())
        if not spans:
            return None, f'"{surface.strip()[:30]}" was not found in the text'
        ents += [{"start": s, "end": e, "label": label.strip().upper()[:40]} for s, e in spans]
    return ents, None


def _row(where: str, obj: dict) -> dict:
    return {"where": where, "text": obj["text"], "lang": clean_lang(obj.get("lang")), "category": clean_category(obj.get("category")),
            "entities": obj.get("entities"), "parse_note": obj.get("parse_note"), "source": _clean_source(obj.get("source")),
            "ext_id": _clean_source(obj.get("id")), "status": clean_status(obj.get("status"))}


def _read_txt(raw: bytes, **_):
    return [(f"line {n}", {"text": ln.strip()}, None) for n, ln in enumerate(_decode(raw).splitlines(), start=1) if ln.strip()], []


def _obj_record(where: str, obj):
    """One JSON object (from JSONL or JSON) as a row."""
    if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
        return (where, None, "needs a JSON object with a text string")
    item = {"text": obj["text"], "lang": obj.get("lang"), "source": obj.get("source"), "id": obj.get("id"), "status": obj.get("status"),
            "category": obj.get("category", obj.get("label") if isinstance(obj.get("label"), str) else None)}
    # `entities` must be a list (anything else is an unreadable annotation, kept and flagged); doccano-style `label`/`labels` count only when they are lists
    key = next((k for k in ("entities", "labels", "label") if k in obj and (k == "entities" or isinstance(obj[k], list))), None)
    item["entities"], item["parse_note"] = clean_entities(obj[key]) if key else (None, None)
    return (where, item, None)


def _read_jsonl(raw: bytes, **_):
    out = []
    for n, ln in enumerate(_decode(raw).splitlines(), start=1):
        if ln.strip():
            try:
                out.append(_obj_record(f"line {n}", json.loads(ln)))
            except json.JSONDecodeError:
                out.append((f"line {n}", None, "not valid JSON"))
    return out, []


def _read_json(raw: bytes, **_):
    try:
        data = json.loads(_decode(raw))
    except json.JSONDecodeError:
        raise CorpusManagerError("The file is not valid JSON.")
    if isinstance(data, dict):
        data = next((data[k] for k in ("documents", "records", "data") if isinstance(data.get(k), list)), None)
    if not isinstance(data, list):
        raise CorpusManagerError("The JSON must be an array of documents, or an object with a documents list.")
    return [_obj_record(f"item {n}", o) for n, o in enumerate(data, start=1)], []


def _from_table(rows: list[dict]):
    """Rows keyed by column name (CSV, TSV, Excel) as records."""
    if not rows:
        return [], []
    if "text" not in {(k or "").strip().lower() for k in rows[0]}:
        raise CorpusManagerError("The table needs a text column.")
    out = []
    for n, r in enumerate(rows, start=2):
        low = {(k or "").strip().lower(): ("" if v is None else str(v)) for k, v in r.items() if k}
        text = low.get("text", "")
        item = {"text": text, "lang": low.get("lang") or low.get("language"), "source": low.get("source"),
                "id": low.get("id") or low.get("doc_id"), "status": None, "category": low.get("category") or low.get("class")}
        item["entities"], item["parse_note"] = _entities_from_cell(text, low.get("entities", ""))
        out.append((f"row {n}", item, None))
    return out, []


def _read_delimited(raw: bytes, delimiter: str = ",", **_):
    try:
        return _from_table(list(csv.DictReader(io.StringIO(_decode(raw), newline=""), delimiter=delimiter)))
    except csv.Error as e:
        raise CorpusManagerError(f"The file could not be read: {str(e)[:120]}")


def _read_xlsx(raw: bytes, **_):
    try:
        nlp_extract._open_zip(raw, "Excel")                         # size and bomb limits
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        it = wb.worksheets[0].iter_rows(values_only=True)
        head = next(it, None)
        if not head:
            return [], []
        names = [("" if h is None else str(h)).strip() for h in head]
        rows = []
        for vals in it:
            if all(v is None or str(v).strip() == "" for v in vals):
                continue
            rows.append(dict(zip(names, vals)))
            if len(rows) > MAX_UPLOAD_RECORDS:
                break
        wb.close()
    except nlp_extract.ExtractError as e:
        raise CorpusManagerError(str(e))
    except CorpusManagerError:
        raise
    except Exception:
        raise CorpusManagerError("The Excel file could not be read. Save it as .xlsx and try again.")
    return _from_table(rows)


def _read_document(raw: bytes, filename: str = "", fmt_hint: str = "", **_):
    ext = nlp_extract.extension(filename)
    if ext not in nlp_extract.DOCUMENT_EXTENSIONS:
        ext = "." + fmt_hint if "." + fmt_hint in nlp_extract.DOCUMENT_EXTENSIONS else ext
    try:
        sentences = nlp_extract.split_sentences(nlp_extract.extract_text(raw, "x" + ext))
    except nlp_extract.ExtractError as e:
        raise CorpusManagerError(str(e))
    return [(f"sentence {n}", {"text": t}, None) for n, t in enumerate(sentences, start=1)], \
           ["The document was split into sentences. Check that the splits are right before annotating."]


def _read_annotation_tool(fmt: str):
    def read(raw: bytes, lang: str | None = None, **_):
        try:
            recs, notes = nlp_formats.load(raw, fmt, lang)
        except nlp_formats.FormatError as e:
            raise CorpusManagerError(str(e))
        out = []
        for where, obj, err in recs:
            parts = None if err else nlp_corpus._doc_parts(obj, lang)
            if parts is None:
                out.append((where, None, err or "needs a text string"))
                continue
            text, dlang, ents, cat = parts
            item = {"text": text, "lang": dlang, "category": cat}
            item["entities"], item["parse_note"] = clean_entities(ents)
            out.append((where, item, None))
        return out, list(notes)
    return read


READERS = {"txt": _read_txt, "jsonl": _read_jsonl, "json": _read_json, "csv": _read_delimited, "tsv": lambda raw, **k: _read_delimited(raw, "\t"),
           "xlsx": _read_xlsx, "document": _read_document, **{f: _read_annotation_tool(f) for f in ("bio", "conll", "docbin", "brat")}}


def parse_upload(raw: bytes, *, filename: str = "", fmt: str = "auto", lang: str | None = None, source: str | None = None) -> dict:
    """Reads and checks an upload. Returns {valid, rows, skipped, identifier_lines, identifier_types, report}.
    Only identifiers (or an unreadable file) make it invalid: rows with other problems are skipped and listed, and annotation problems
    are kept on the row so the quality check can show them. `lang` is the default for rows without a language."""
    fmt_in = (fmt or "auto").strip().lower()
    fmt = detect_format(fmt, filename)
    default_lang = clean_lang(lang)
    if default_lang is not None and default_lang not in LANG_CODES:
        raise CorpusManagerError("lang must be one of " + ", ".join(sorted(LANG_CODES)))
    hint = (fmt_in if fmt_in in FORMAT_ALIASES and FORMAT_ALIASES[fmt_in] == "document" else "")
    records, notes = READERS[fmt](raw, filename=filename, lang=default_lang, fmt_hint=hint)
    if not records:
        raise CorpusManagerError("The file has no records.")
    if len(records) > MAX_UPLOAD_RECORDS:
        raise CorpusManagerError(f"The file has more than {MAX_UPLOAD_RECORDS} records. Split it.")
    rows, skipped, id_lines, id_types = [], [], [], Counter()
    for where, obj, err in records:
        if err:
            skipped.append(f"{where}: {err}")
            continue
        text = obj["text"]
        if len(text.strip()) < MIN_CHARS or not normalise_text(text) or len(text) > MAX_CHARS:
            skipped.append(f"{where}: text is empty, has no letters or digits, or is longer than {MAX_CHARS} characters")
            continue
        phi = nlp_lab._identifiers(text)
        if phi:
            id_lines.append(where)
            id_types.update(e["norm"] for e in phi)
            continue
        r = _row(where, obj)
        r["text"] = text.strip() if fmt in ("txt", "document") else text
        r["lang"] = r["lang"] or default_lang
        r["source"] = r["source"] or _clean_source(source)
        rows.append(r)
    valid = not id_lines
    for r in rows:
        complete_by_data = r["entities"] is not None or r["category"] is not None
        r["status"] = r["status"] if r["status"] in ("unannotated", "annotated") else ("annotated" if complete_by_data and not r["parse_note"] else "unannotated")
    hashes = Counter(text_hash(r["text"]) for r in rows)
    warnings = list(notes)
    if skipped:
        warnings.append(f"{len(skipped)} record(s) were skipped (see the list).")
    if any(r["parse_note"] for r in rows):
        warnings.append(f"{sum(1 for r in rows if r['parse_note'])} record(s) have an annotation that could not be read. They are kept and flagged.")
    if fmt in ("txt", "document"):
        warnings.append("This file has no annotations: every record starts as unannotated.")
    report = {"source_format": fmt, "records_read": len(records), "records": len(rows), "skipped": len(skipped),
              "repeated_in_file": sum(n - 1 for n in hashes.values() if n > 1),
              "by_language": dict(Counter(r["lang"] or "(missing)" for r in rows)),
              "by_category": dict(Counter(r["category"] for r in rows if r["category"])),
              "by_label": dict(Counter(e["label"] for r in rows for e in (r["entities"] or []))),
              "annotated": sum(1 for r in rows if r["status"] == "annotated"), "unannotated": sum(1 for r in rows if r["status"] == "unannotated"),
              "warnings": warnings}
    return {"valid": valid, "rows": rows if valid else [], "skipped": skipped[:MAX_SHOWN], "skipped_total": len(skipped),
            "identifier_lines": id_lines[:MAX_SHOWN], "identifier_lines_total": len(id_lines), "identifier_types": dict(id_types), "report": report}


# ------------------------------------------------------------------ quality report
def quality(records: list[dict], tasks, other_corpora: dict[str, list[str]] | None = None) -> dict:
    """Quality report for one corpus. `records` need id, text, norm_hash, lang, category, entities, parse_note, status.
    `other_corpora` maps a text hash to the names of other corpora that hold the same text. Rejected records are listed but never count."""
    tasks = [t for t in TASKS if t in set(tasks)]
    live = [r for r in records if r["status"] != "rejected"]
    by_issue: dict[str, list[dict]] = defaultdict(list)
    for r in live:
        for code, msg in record_issues(r, tasks):
            by_issue[code].append({"id": r["id"], "problem": msg})
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in live:
        groups[r["norm_hash"]].append(r)
    dup = [g for g in groups.values() if len(g) > 1]
    conflicting = [g for g in dup if len({(x["lang"], x["category"], json.dumps(x["entities"], sort_keys=True)) for x in g}) > 1]
    overlaps: Counter = Counter()
    for r in live:
        for name in (other_corpora or {}).get(r["norm_hash"], []):
            overlaps[name] += 1
    missing = {k: v for k, v in by_issue.items() if k.startswith("missing_")}
    invalid = {k: v for k, v in by_issue.items() if k.startswith("invalid_")}
    n_missing = len({x["id"] for v in missing.values() for x in v})
    n_invalid = len({x["id"] for v in invalid.values() for x in v})
    by_lang = Counter(r["lang"] or "(missing)" for r in live)
    by_cat = Counter(r["category"] for r in live if r["category"])
    by_label = Counter(e["label"] for r in live for e in (r["entities"] or []))
    warnings = []
    def thin(counts: dict, names, floor: int) -> str:
        return ", ".join(f"{k} {counts.get(k, 0)}" for k in sorted(names) if counts.get(k, 0) < floor)

    if "ner" in tasks and (low := thin(by_label, ENTITY_LABELS, MIN_EXAMPLES_PER_LABEL)):
        warnings.append(f"Entity labels with fewer than {MIN_EXAMPLES_PER_LABEL} examples (a trustworthy entity model needs at least that many of each): {low}.")
    if "classification" in tasks and (low := thin(by_cat, by_cat, MIN_DOCS_PER_CLASS)):
        warnings.append(f"Categories with fewer than {MIN_DOCS_PER_CLASS} documents: {low}.")
    if low := thin(by_lang, [lg for lg in by_lang if lg in LANG_CODES], MIN_DOCS_PER_LANG):
        warnings.append(f"Languages with fewer than {MIN_DOCS_PER_LANG} documents: {low}.")
    short = sum(1 for r in live if len(r["text"].split()) < 3)
    if short:
        warnings.append(f"{short} record(s) have fewer than 3 words.")
    if overlaps:
        warnings.append("Texts also found in other corpora: " + ", ".join(f"{k} ({v})" for k, v in sorted(overlaps.items())) + ". A merge keeps one copy.")
    blockers = []
    if not live:
        blockers.append("The corpus has no usable records.")
    if n_missing:
        blockers.append(f"{n_missing} record(s) are missing an annotation the corpus needs ({', '.join(sorted(missing))}).")
    if n_invalid:
        blockers.append(f"{n_invalid} record(s) have an invalid annotation.")
    if dup:
        blockers.append(f"{len(dup)} duplicate group(s): resolve them (keep one copy, reject the rest).")
    return {"tasks": tasks, "records": len(records), "usable": len(live), "rejected": len(records) - len(live),
            "by_status": dict(Counter(r["status"] for r in records)), "by_language": dict(by_lang), "by_category": dict(by_cat), "by_label": dict(by_label),
            "label_by_language": {lg: dict(Counter(e["label"] for r in live if (r["lang"] or "(missing)") == lg for e in (r["entities"] or []))) for lg in by_lang},
            "missing": {"records": n_missing, **{k: {"count": len(v), "sample": v[:MAX_SHOWN]} for k, v in missing.items()}},
            "invalid": {"records": n_invalid, **{k: {"count": len(v), "sample": v[:MAX_SHOWN]} for k, v in invalid.items()}},
            "duplicates": {"groups": len(dup), "extra_records": sum(len(g) - 1 for g in dup), "conflicting_groups": len(conflicting),
                           "sample": [[x["id"] for x in g] for g in dup[:MAX_SHOWN]]},
            "also_in_other_corpora": dict(overlaps), "warnings": warnings, "blockers": blockers, "ready_for_approval": not blockers}


def duplicate_extras(records: list[dict]) -> list[int]:
    """Ids to reject so that one copy of every duplicated text stays: the best-annotated copy, then the oldest."""
    rank = {"reviewed": 0, "annotated": 1, "unannotated": 2}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["status"] != "rejected":
            groups[r["norm_hash"]].append(r)
    out = []
    for g in groups.values():
        if len(g) > 1:
            g.sort(key=lambda r: (rank.get(r["status"], 3), r["id"]))
            out += [r["id"] for r in g[1:]]
    return sorted(out)


# ------------------------------------------------------------------ snapshots and serialisation
def snapshot_docs(records: list[dict]) -> list[dict]:
    """What an approval freezes: every non-rejected record, with its annotation."""
    return [{"id": r["id"], "text": r["text"], "lang": r["lang"], "category": r["category"], "entities": r["entities"],
             "status": r["status"], "source": r.get("source")} for r in sorted(records, key=lambda r: r["id"]) if r["status"] != "rejected"]


def to_jsonl(docs: list[dict]) -> bytes:
    return ("\n".join(json.dumps(d, ensure_ascii=False, separators=(",", ":")) for d in docs) + "\n").encode("utf-8") if docs else b""


def from_jsonl(raw: bytes) -> list[dict]:
    return [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]


# ------------------------------------------------------------------ merging
CONFLICT_POLICIES = {"drop_conflicts": "Leave out any text whose annotations disagree between corpora (default, safest).",
                     "keep_first": "Keep the version from the corpus listed first and report the disagreement.",
                     "refuse": "Stop and list the disagreements, so they can be fixed in the corpora first."}


def _entity_key(text, ents):
    return (text, tuple(sorted((e["start"], e["end"], e["label"]) for e in ents)))


def merge(sources: list[dict], on_conflict: str = "drop_conflicts") -> dict:
    """Combines frozen corpus snapshots into master rows. `sources` is in priority order: each has corpus_id, name, domain, snapshot_id and docs.
    The same text (after folding case, punctuation and spacing) found in several corpora becomes one row that lists all its corpora.
    Annotations that complement each other (one corpus has a category, another does not) are combined; annotations that disagree are a conflict."""
    if on_conflict not in CONFLICT_POLICIES:
        raise CorpusManagerError("on_conflict must be one of " + ", ".join(CONFLICT_POLICIES))
    if not sources:
        raise CorpusManagerError("Choose at least one approved corpus.")
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for pri, s in enumerate(sources):
        for d in s["docs"]:
            h = text_hash(d["text"])
            if h not in groups:
                groups[h] = []
                order.append(h)
            groups[h].append({**d, "_corpus": s["name"], "_corpus_id": s["corpus_id"], "_domain": s["domain"], "_pri": pri})
    rows, conflicts, left_out = [], [], 0
    for h in order:
        g = groups[h]
        best, clash = {}, []
        for field in ("lang", "category", "entities"):
            vals = [x for x in g if x[field] is not None]
            keys = {_entity_key(x["text"], x["entities"]) for x in vals} if field == "entities" else {x[field] for x in vals}
            if len(keys) > 1:
                clash.append(field)
            best[field] = vals[0] if vals else None
        if clash:
            conflicts.append({"fields": clash, "corpora": sorted({x["_corpus"] for x in g}), "records": [[x["_corpus_id"], x["id"]] for x in g]})
            if on_conflict == "drop_conflicts":
                left_out += len(g)
                continue
            if on_conflict == "keep_first":
                top = min(x["_pri"] for x in g)
                left_out += sum(1 for x in g if x["_pri"] != top)
                g = [x for x in g if x["_pri"] == top][:1]
                best = {f: (g[0] if g[0][f] is not None else None) for f in ("lang", "category", "entities")}
        ent_src = best["entities"]
        primary = ent_src or g[0]                       # the text that the entity offsets belong to
        rows.append({"text": primary["text"], "lang": best["lang"]["lang"] if best["lang"] else None,
                     "category": best["category"]["category"] if best["category"] else None,
                     "entities": ent_src["entities"] if ent_src else None,
                     "annotation_status": "reviewed" if all(x["status"] == "reviewed" for x in g) else "annotated",
                     "source": primary.get("source"), "corpora": sorted({x["_corpus"] for x in g}), "domains": sorted({x["_domain"] for x in g}),
                     "origin": [[x["_corpus_id"], x["id"]] for x in g]})
    if conflicts and on_conflict == "refuse":
        raise MergeConflict(f"{len(conflicts)} text(s) have annotations that disagree between corpora. Fix them, or choose another policy.", conflicts[:MAX_SHOWN])
    for n, r in enumerate(rows, start=1):
        r["uid"] = f"m{n:07d}"
    total_in = sum(len(s["docs"]) for s in sources)
    report = {"records_in": total_in, "records_out": len(rows), "duplicates_merged": sum(len(r["origin"]) - 1 for r in rows),
              "conflicts": len(conflicts), "conflict_policy": on_conflict, "records_left_out_for_conflict": left_out, "conflict_sample": conflicts[:MAX_SHOWN],
              "per_source": [{"corpus_id": s["corpus_id"], "name": s["name"], "snapshot_id": s["snapshot_id"], "records": len(s["docs"])} for s in sources],
              **summarise(rows)}
    return {"rows": rows, "report": report}


def summarise(rows: list[dict]) -> dict:
    return {"by_language": dict(Counter(r["lang"] or "(missing)" for r in rows)), "by_category": dict(Counter(r["category"] for r in rows if r["category"])),
            "by_label": dict(Counter(e["label"] for r in rows for e in (r["entities"] or []))),
            "by_domain": dict(Counter(d for r in rows for d in r["domains"])), "by_corpus": dict(Counter(c for r in rows for c in r["corpora"])),
            "task_ready": {"ner": sum(1 for r in rows if r["entities"] is not None), "classification": sum(1 for r in rows if r["category"] in CATEGORIES),
                           "langid": sum(1 for r in rows if r["lang"] in LANG_CODES)}}


# ------------------------------------------------------------------ task datasets
TASK_COLUMNS = {"ner": ["uid", "text", "lang", "entities", "split"],
                "classification": ["uid", "text", "label", "lang", "split"],
                "langid": ["uid", "text", "lang", "split"]}


def assign_splits(items: list[dict], key, test_size: float, val_size: float, seed: int) -> None:
    """Adds item['split'] (train, validation, test), stratified by `key(item)` and repeatable: the same data and seed always give the same split.
    A group of fewer than 3 documents stays in train (a split on it would not be meaningful)."""
    strata: dict = defaultdict(list)
    for it in items:
        strata[key(it)].append(it)
    for members in strata.values():
        members.sort(key=lambda it: hashlib.sha256(f"{seed}:{it['uid']}".encode()).hexdigest())
        n = len(members)
        n_test = round(n * test_size) if n >= 3 else 0
        n_val = round(n * val_size) if n >= 3 else 0
        if n >= 3 and test_size > 0:
            n_test = max(n_test, 1)
        if n >= 3 and val_size > 0:
            n_val = max(n_val, 1)
        while n_test + n_val >= n:
            n_val, n_test = (n_val - 1, n_test) if n_val >= n_test and n_val > 0 else (n_val, n_test - 1)
        for i, it in enumerate(members):
            it["split"] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"


def build_task(rows: list[dict], task: str, *, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42, include_mixed: bool = False) -> dict:
    """A task dataset from master rows: {rows, columns, report}. Nothing is trained."""
    if task not in TASKS:
        raise CorpusManagerError("task must be one of " + ", ".join(TASKS))
    if not (0 <= test_size < 0.5 and 0 <= val_size < 0.5 and test_size + val_size < 0.8):
        raise CorpusManagerError("test_size and val_size must each be under 0.5 and add up to under 0.8 (use 0 for no split).")
    out, excluded = [], Counter()
    for r in rows:
        if task == "ner":
            if r["entities"] is None:
                excluded["no entity annotation"] += 1
                continue
            if r["lang"] not in LANG_CODES:
                excluded["no valid language"] += 1
                continue
            out.append({"uid": r["uid"], "text": r["text"], "lang": r["lang"], "entities": r["entities"]})
        elif task == "classification":
            if r["category"] not in CATEGORIES:
                excluded["no document category"] += 1
                continue
            out.append({"uid": r["uid"], "text": r["text"], "label": r["category"], "lang": r["lang"]})
        else:
            if r["lang"] not in LANG_CODES:
                excluded["no valid language"] += 1
                continue
            if r["lang"] == "mixed" and not include_mixed:
                excluded["mixed-language (not a single class; set include_mixed to keep)"] += 1
                continue
            out.append({"uid": r["uid"], "text": r["text"], "lang": r["lang"]})
    stratum = {"ner": lambda it: it["lang"], "classification": lambda it: it["label"], "langid": lambda it: it["lang"]}[task]
    if test_size or val_size:
        assign_splits(out, stratum, test_size, val_size, seed)
    else:
        for it in out:
            it["split"] = "train"
    classes = Counter(stratum(it) for it in out)
    warnings = []
    if len(out) == 0:
        warnings.append("No records qualify for this task.")
    if task in ("classification", "langid") and len(classes) < 2 and out:
        warnings.append("Only one class: a classifier needs at least 2.")
    floor = MIN_DOCS_PER_LANG if task == "langid" else MIN_DOCS_PER_CLASS
    if task != "ner":
        if low := ", ".join(f"{c} {n}" for c, n in sorted(classes.items()) if n < floor):
            warnings.append(f"Classes with fewer than {floor} documents (too few to train and test on): {low}.")
    else:
        lab = Counter(e["label"] for it in out for e in it["entities"])
        if low := ", ".join(f"{k} {lab[k]}" for k in sorted(ENTITY_LABELS) if lab[k] < MIN_EXAMPLES_PER_LABEL):
            warnings.append(f"Entity labels with fewer than {MIN_EXAMPLES_PER_LABEL} examples: {low}.")
    rep = {"task": task, "records": len(out), "excluded": dict(excluded), "excluded_total": sum(excluded.values()), "classes": dict(classes),
           "by_split": dict(Counter(it["split"] for it in out)), "seed": seed, "test_size": test_size, "val_size": val_size, "warnings": warnings,
           **({"entities": sum(len(it["entities"]) for it in out), "by_label": dict(Counter(e["label"] for it in out for e in it["entities"]))} if task == "ner" else {})}
    return {"rows": out, "columns": TASK_COLUMNS[task], "report": rep}


# ------------------------------------------------------------------ export
EXPORT_FORMATS = {"csv": "text/csv; charset=utf-8", "jsonl": "application/x-ndjson; charset=utf-8", "parquet": "application/vnd.apache.parquet"}
MASTER_COLUMNS = ["uid", "text", "lang", "category", "entities", "annotation_status", "source", "corpora", "domains"]


def _flat(row: dict, columns: list[str]) -> dict:
    """One row with list fields written as text (entities as JSON, corpora and domains joined by `;`), for CSV and Parquet."""
    out = {}
    for c in columns:
        v = row.get(c)
        out[c] = (json.dumps(v, ensure_ascii=False, separators=(",", ":")) if c == "entities" and v is not None
                  else ";".join(v) if isinstance(v, list) else v)
    return out


def export_rows(rows: list[dict], columns: list[str], fmt: str, *, spreadsheet_safe: bool = False) -> bytes:
    """Serialises rows. JSONL keeps entities as real lists. CSV and Parquet write entities as a JSON string. With `spreadsheet_safe`, CSV cells
    that would start a spreadsheet formula get a leading apostrophe (this changes the text, so it is off by default)."""
    if fmt not in EXPORT_FORMATS:
        raise CorpusManagerError("format must be one of " + ", ".join(EXPORT_FORMATS))
    if fmt == "jsonl":
        return to_jsonl([{c: r.get(c) for c in columns} for r in rows])
    flat = [_flat(r, columns) for r in rows]
    if fmt == "csv":
        buf = io.StringIO(newline="")
        w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in flat:
            if spreadsheet_safe:
                r = {k: ("'" + v if isinstance(v, str) and v.startswith(FORMULA_START) else v) for k, v in r.items()}
            w.writerow(r)
        return buf.getvalue().encode("utf-8")
    import pandas as pd                                         # parquet needs pyarrow, which is only imported when asked for
    try:
        df = pd.DataFrame(flat, columns=columns)
        out = io.BytesIO()
        df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    except ImportError:
        raise CorpusManagerError("Parquet export needs the pyarrow package on the server.")
    return out.getvalue()
