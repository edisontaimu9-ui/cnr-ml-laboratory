"""Readers for annotated-text formats. Every reader turns a file into the same plain records, which `nlp_corpus.validate` then checks
(identifiers, offsets, overlaps, labels) and stores as one canonical JSON Lines form. Nothing here stores or logs text.

Supported (format code, typical file):
  jsonl   JSON Lines, one document per line          .jsonl .ndjson
  json    One JSON array or {"documents": [...]}     .json
  csv     Spreadsheet annotation                     .csv
  bio     BIO / IOB2 tagged tokens (strict IOB2)     .bio .iob .iob2
  conll   CoNLL-2003 style columns (IOB1 or IOB2)    .conll
  docbin  spaCy DocBin                               .spacy
  brat    BRAT standoff pairs (.txt + .ann) in a ZIP .zip

A record is (where, obj, error). `obj` is a dict (or spaCy-style [text, {"entities": [...]}] pair) that still has to be validated;
`error` is a message about the record that could not be read at all."""
import csv
import io
import json
import re
import zipfile
import zlib
from pathlib import Path

MAX_TEXT_CHARS = 8000
MAX_DOCS = 50_000
MAX_DOCBIN_BYTES = 256 * 1024 * 1024
MAX_ZIP_MEMBERS = 100_000
MAX_ZIP_MEMBER_BYTES = 5 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 100 * 1024 * 1024

FORMATS = {
    "jsonl": "JSON Lines: one document per line.",
    "json": "One JSON array of documents, or an object with a documents list.",
    "csv": "Spreadsheet: one row per entity (text, start, end, label) or one row per document with an entities cell.",
    "bio": "BIO / IOB2: one token and tag per line, blank line between sentences.",
    "conll": "CoNLL-2003 style columns: the last column is the tag (IOB1 or IOB2), blank line between sentences.",
    "docbin": "spaCy DocBin (.spacy) with entity annotations.",
    "brat": "BRAT standoff: a ZIP of paired name.txt and name.ann files.",
}
ALIASES = {"ndjson": "jsonl", "iob": "bio", "iob2": "bio", "conll2003": "conll", "spacy": "docbin"}
EXTENSIONS = {".jsonl": "jsonl", ".ndjson": "jsonl", ".json": "json", ".csv": "csv", ".bio": "bio", ".iob": "bio", ".iob2": "bio",
              ".conll": "conll", ".conll03": "conll", ".spacy": "docbin", ".zip": "brat"}
LANGS = {"en", "ny", "tum", "yao", "mixed"}

Record = tuple[str, object, str | None]


class FormatError(ValueError):
    pass


def detect(fmt: str | None, filename: str) -> str:
    fmt = (fmt or "auto").strip().lower()
    if fmt != "auto":
        fmt = ALIASES.get(fmt, fmt)
        if fmt not in FORMATS:
            raise FormatError("format must be auto or one of " + ", ".join(FORMATS))
        return fmt
    if not filename:
        return "jsonl"
    ext = Path(filename).suffix.lower()
    if ext in EXTENSIONS:
        return EXTENSIONS[ext]
    raise FormatError("The format cannot be told from the file name. Set format to one of " + ", ".join(FORMATS) + ".")


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise FormatError("The file is not UTF-8 text.")


def load(raw: bytes, fmt: str, lang: str | None) -> tuple[list[Record], list[str]]:
    """Returns (records, notes). Raises FormatError when the file cannot be read as that format at all."""
    return {"jsonl": _jsonl, "json": _json, "csv": _csv, "bio": lambda r, la: _tagged(r, la, strict=True),
            "conll": lambda r, la: _tagged(r, la, strict=False), "docbin": _docbin, "brat": _brat}[fmt](raw, lang)


# ---------------------------------------------------------------- JSON
def _jsonl(raw: bytes, lang):
    out = []
    for n, ln in enumerate(_decode(raw).splitlines(), start=1):
        if not ln.strip():
            continue
        try:
            out.append((f"line {n}", json.loads(ln), None))
        except json.JSONDecodeError:
            out.append((f"line {n}", None, "not valid JSON"))
    return out, []


def _json(raw: bytes, lang):
    try:
        data = json.loads(_decode(raw))
    except json.JSONDecodeError:
        raise FormatError("The file is not valid JSON. For one document per line, use the jsonl format.")
    if isinstance(data, dict):
        for k in ("documents", "docs", "data", "examples", "annotations"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise FormatError("The JSON must be a list of documents or an object with a documents list.")
    return [(f"item {n}", obj, None) for n, obj in enumerate(data, start=1)], []


# ---------------------------------------------------------------- CSV
def _int(v):
    try:
        return int(str(v).strip())
    except ValueError:
        return None


def _surface_spans(text: str, surface: str) -> list[tuple[int, int]]:
    rx = re.compile(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", re.I)
    return [m.span() for m in rx.finditer(text)]


def _csv(raw: bytes, lang):
    rows = list(csv.DictReader(io.StringIO(_decode(raw), newline="")))
    if not rows:
        return [], []
    head = {(k or "").strip().lower() for k in rows[0]}
    if "text" not in head:
        raise FormatError("The CSV needs a text column.")
    norm = [({(k or "").strip().lower(): (v or "").strip() if k and (k or "").strip().lower() != "text" else (v or "")
              for k, v in r.items()}, n) for n, r in enumerate(rows, start=2)]
    notes, out = [], []
    if {"start", "end", "label"} <= head:                                   # one row per entity
        order: list = []
        docs: dict = {}
        for r, n in norm:
            key = r.get("doc_id") or r["text"]
            if key not in docs:
                docs[key] = {"row": n, "text": r["text"], "lang": r.get("lang") or None, "entities": [], "err": None}
                order.append(key)
            d = docs[key]
            if d["text"] != r["text"] and not d["err"]:
                d["err"] = (n, "has different text from earlier rows with the same doc_id")
            if r.get("start") or r.get("end") or r.get("label"):
                s, e = _int(r.get("start")), _int(r.get("end"))
                if s is None or e is None:
                    d["err"] = d["err"] or (n, "start and end must be whole numbers")
                else:
                    d["entities"].append({"start": s, "end": e, "label": r.get("label", "")})
        for key in order:
            d = docs[key]
            if d["err"]:
                out.append((f"row {d['err'][0]}", None, d["err"][1]))
            else:
                out.append((f"row {d['row']}", {"text": d["text"], "lang": d["lang"], "entities": d["entities"]}, None))
    elif "entities" in head:                                                # one row per document: "surface|LABEL; surface|LABEL"
        for r, n in norm:
            ents, err = [], None
            for cell in (c for c in r.get("entities", "").split(";") if c.strip()):
                surface, _, label = cell.strip().rpartition("|")
                if not surface.strip() or not label.strip():
                    err = f'entities cell needs "text|LABEL" items separated by semicolons'
                    break
                spans = _surface_spans(r["text"], surface.strip())
                if not spans:
                    err = f'"{surface.strip()[:30]}" was not found in the text'
                    break
                ents += [{"start": s, "end": e, "label": label.strip()} for s, e in spans]
            out.append((f"row {n}", None if err else {"text": r["text"], "lang": r.get("lang") or None, "entities": ents}, err))
        notes.append("Entities were found by matching each listed word in the text as a whole word, every time it appears.")
    else:
        notes.append("The CSV has no entity columns (start, end, label or entities), so its documents have no entities.")
        out = [(f"row {n}", {"text": r["text"], "lang": r.get("lang") or None, "entities": []}, None) for r, n in norm]
    return out, notes


# ---------------------------------------------------------------- BIO / CoNLL
_TAG = re.compile(r"^([BI])-(.+)$", re.I)
_LANG_COMMENT = re.compile(r"#\s*lang(?:uage)?\s*[:=]\s*([A-Za-z]+)", re.I)


def _tagged(raw: bytes, lang, strict: bool):
    out: list[Record] = []
    block: dict = {"first": None, "tokens": [], "tags": [], "lang": None, "err": False}

    def flush():
        b = block
        if b["tokens"] and not b["err"]:
            text, cur, spans = "", None, []
            for tok, (kind, label) in zip(b["tokens"], b["tags"]):
                start = len(text) + (1 if text else 0)
                text = (text + " " if text else "") + tok
                end = len(text)
                if kind == "O":
                    cur = None
                elif kind == "B" or cur is None or cur[0] != label:
                    cur = [label, start, end]
                    spans.append(cur)
                else:
                    cur[2] = end
            ents = [{"start": s, "end": e, "label": label} for label, s, e in spans]
            out.append((f"line {b['first']}", {"text": text, "lang": b["lang"] or lang, "entities": ents}, None))
        block.update(first=None, tokens=[], tags=[], lang=None, err=False)

    for n, ln in enumerate(_decode(raw).splitlines(), start=1):
        s = ln.strip()
        if not s:
            flush()
            continue
        if s.startswith("#"):
            m = _LANG_COMMENT.match(s)
            if m:
                block["lang"] = m.group(1).lower()
            continue
        if s.upper().startswith("-DOCSTART-"):
            flush()
            continue
        parts = s.split()
        if len(parts) < 2:
            out.append((f"line {n}", None, "needs a token and a tag separated by a space or tab"))
            block["err"] = True
            continue
        if block["first"] is None:
            block["first"] = n
        tag = parts[-1]
        if tag.upper() == "O":
            kind, label = "O", None
        else:
            m = _TAG.match(tag)
            if not m:
                out.append((f"line {n}", None, f"tag {tag[:20]!r} is not O, B-LABEL or I-LABEL"))
                block["err"] = True
                kind, label = "O", None
            else:
                kind, label = m.group(1).upper(), m.group(2)
                prev = block["tags"][-1] if block["tags"] else ("O", None)
                if strict and kind == "I" and (prev[0] == "O" or prev[1] != label):
                    out.append((f"line {n}", None, "I- tag without a B- tag of the same label before it (IOB2 needs B- to start an entity)"))
                    block["err"] = True
        block["tokens"].append(parts[0])
        block["tags"].append((kind, label))
    flush()
    return out, ["Text was rebuilt by joining tokens with single spaces."]


# ---------------------------------------------------------------- spaCy DocBin
def _docbin(raw: bytes, lang):
    try:
        import spacy
        from spacy.tokens import DocBin
    except ImportError:
        raise FormatError("spaCy is not installed on this server, so .spacy files cannot be read. Convert to JSONL and upload that.")
    try:
        d = zlib.decompressobj()
        d.decompress(raw, MAX_DOCBIN_BYTES)
        if d.unconsumed_tail:
            raise FormatError("The DocBin is too large once unpacked.")
        db = DocBin().from_bytes(raw)
        if len(db) > MAX_DOCS:
            raise FormatError(f"The DocBin has more than {MAX_DOCS} documents. Split it.")
        docs = list(db.get_docs(spacy.blank("xx").vocab))
    except FormatError:
        raise
    except Exception:
        raise FormatError("The file could not be read as a spaCy DocBin.")
    out = []
    for n, doc in enumerate(docs, start=1):
        ud = doc.user_data if isinstance(doc.user_data, dict) else {}
        ents = [{"start": e.start_char, "end": e.end_char, "label": e.label_} for e in doc.ents]
        out.append((f"doc {n}", {"text": doc.text, "lang": ud.get("lang") or lang, "entities": ents}, None))
    notes = []
    if any(o[1]["lang"] is None for o in out):
        notes.append("spaCy DocBin files carry no language per document unless saved with a lang entry in user data; set lang for the upload.")
    return out, notes


# ---------------------------------------------------------------- BRAT standoff
def _chunks(text: str):
    if len(text) <= MAX_TEXT_CHARS:
        yield 0, text
        return
    start, cur = 0, 0
    for line in text.splitlines(keepends=True):
        if cur - start + len(line) > MAX_TEXT_CHARS and cur > start:
            yield start, text[start:cur]
            start = cur
        cur += len(line)
    if cur > start:
        yield start, text[start:cur]


def _brat(raw: bytes, lang):
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise FormatError("BRAT files must be uploaded as one ZIP containing name.txt and name.ann pairs.")
    infos = [i for i in zf.infolist() if not i.is_dir() and "__MACOSX" not in i.filename and not Path(i.filename).name.startswith(".")
             and Path(i.filename).suffix.lower() in (".txt", ".ann")]
    if len(infos) > MAX_ZIP_MEMBERS or sum(i.file_size for i in infos) > MAX_ZIP_TOTAL_BYTES or any(i.file_size > MAX_ZIP_MEMBER_BYTES for i in infos):
        raise FormatError("The ZIP is too large (more than 100 MB unpacked, or a file over 5 MB). Split it.")
    pairs: dict[str, dict] = {}
    for i in infos:
        p = Path(i.filename)
        pairs.setdefault(str(p.with_suffix("")), {})[p.suffix.lower()] = i
    if not pairs:
        raise FormatError("The ZIP has no .txt and .ann files.")
    out, ignored = [], 0
    for stem in sorted(pairs):
        parts = pairs[stem]
        if ".txt" not in parts or ".ann" not in parts:
            out.append((stem, None, "needs both a .txt and a .ann file"))
            continue
        try:
            text = zf.read(parts[".txt"]).decode("utf-8")
            ann = zf.read(parts[".ann"]).decode("utf-8-sig")
        except UnicodeDecodeError:
            out.append((stem, None, "is not UTF-8 text"))
            continue
        folder = [c.lower() for c in Path(stem).parent.parts]
        dlang = next((c for c in reversed(folder) if c in LANGS), None) or lang
        ents, bad = [], False
        for ln in ann.splitlines():
            if not ln.strip():
                continue
            cols = ln.split("\t")
            if not cols[0].startswith("T"):
                ignored += 1
                continue
            try:
                label, spans = cols[1].split(" ", 1)
                if ";" in spans:
                    raise ValueError("discontinuous")
                s, e = (int(x) for x in spans.split())
            except (IndexError, ValueError) as ex:
                msg = "has a discontinuous entity, which is not supported" if str(ex) == "discontinuous" else f"{cols[0]} could not be read"
                out.append((stem + ".ann", None, msg))
                bad = True
                continue
            if len(cols) > 2 and " ".join(cols[2].split()) != " ".join(text[s:e].split()):
                out.append((stem + ".ann", None, f"{cols[0]} text does not match its offsets in the .txt file"))
                bad = True
                continue
            ents.append({"start": s, "end": e, "label": label})
        if bad:
            continue
        chunks = list(_chunks(text))
        for k, (c0, chunk) in enumerate(chunks, start=1):
            where = stem if len(chunks) == 1 else f"{stem} part {k}"
            mine = [{"start": e["start"] - c0, "end": e["end"] - c0, "label": e["label"]} for e in ents if c0 <= e["start"] and e["end"] <= c0 + len(chunk)]
            if chunk.strip() or mine:
                out.append((where, {"text": chunk, "lang": dlang, "entities": mine}, None))
        if len(chunks) > 1:
            inside = sum(1 for e in ents for c0, ch in chunks if c0 <= e["start"] and e["end"] <= c0 + len(ch))
            if inside < len(ents):
                out.append((stem, None, "has an entity that crosses a split point of a long text; shorten the text or move the entity"))
    notes = [f"{ignored} BRAT annotations that are not entities (relations, attributes, notes) were ignored."] if ignored else []
    notes.append("A folder named en, ny, tum, yao or mixed inside the ZIP sets the language of the files in it.")
    return out, notes
