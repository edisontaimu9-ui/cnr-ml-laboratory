"""Text extraction for pre-label: turns a document into plain sentences, in memory only.

Pre-label used to read only a .txt (one sentence per line) or a .csv with a `text` column. This module adds the document formats
people actually have, so a lecture note, guideline or report can be dropped in as it is:

  .docx   Word                       paragraphs and tables, read straight from the ZIP/XML (no extra library)
  .pptx   PowerPoint                 text of every slide, in slide order
  .odt    OpenDocument text          paragraphs and headings
  .pdf    PDF with selectable text   needs `pypdf`; a scanned PDF has no text and is refused with a clear message
  .html   Web page                   scripts, styles and navigation are skipped
  .md     Markdown                   markup is stripped

Nothing is stored or logged here. Limits keep a hostile file from using the server's memory: capped file members, capped total
size, capped pages, and XML that declares entities or a DOCTYPE is refused instead of parsed."""
import io
import re
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import PurePath
from xml.etree import ElementTree as ET

MAX_MEMBER_BYTES = 20 * 1024 * 1024        # one unpacked XML part
MAX_TOTAL_BYTES = 60 * 1024 * 1024         # all unpacked parts together
MAX_PDF_PAGES = 600
MAX_CHARS = 3_000_000                      # extracted text kept
MAX_SENTENCE_CHARS = 1000                  # longer runs are cut at the last clause break
MIN_SENTENCE_CHARS = 10
MIN_SENTENCE_WORDS = 2

DOCUMENT_EXTENSIONS = (".docx", ".pptx", ".odt", ".pdf", ".html", ".htm", ".md", ".markdown")
ACCEPT = ".txt,.csv," + ",".join(DOCUMENT_EXTENSIONS)


class ExtractError(ValueError):
    pass


def extension(filename: str) -> str:
    return PurePath(filename or "").suffix.lower()


def is_document(filename: str) -> bool:
    return extension(filename) in DOCUMENT_EXTENSIONS


# ---------------------------------------------------------------- ZIP and XML helpers
def _open_zip(raw: bytes, what: str) -> zipfile.ZipFile:
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        raise ExtractError(f"This is not a valid {what} file. Open it, save it again as {what}, and upload that.")
    infos = z.infolist()
    if len(infos) > 5000 or sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
        raise ExtractError(f"The {what} file is too large once unpacked.")
    return z


def _part(z: zipfile.ZipFile, name: str) -> bytes:
    try:
        info = z.getinfo(name)
    except KeyError:
        raise ExtractError("The file is missing its main text part, so it cannot be read.")
    if info.file_size > MAX_MEMBER_BYTES:
        raise ExtractError("A part of the file is too large to read.")
    return z.read(name)


def _xml(data: bytes) -> ET.Element:
    head = data[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in head:
        raise ExtractError("The file contains unsafe XML and was refused.")
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        raise ExtractError("The file's text could not be read. Save it again from the original program and retry.")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _paragraph_text(p: ET.Element, text_tags: set[str], space_tags: set[str], break_tags: set[str]) -> str:
    out: list[str] = []
    for el in p.iter():
        t = _local(el.tag)
        if t in text_tags and el.text:
            out.append(el.text)
        elif t in space_tags:
            out.append(" ")
        elif t in break_tags:
            out.append(" ")
    return "".join(out)


# ---------------------------------------------------------------- formats
def _docx(raw: bytes) -> str:
    z = _open_zip(raw, ".docx")
    root = _xml(_part(z, "word/document.xml"))
    paras = [_paragraph_text(p, {"t"}, {"tab", "noBreakHyphen"}, {"br", "cr"}) for p in root.iter() if _local(p.tag) == "p"]
    return "\n\n".join(p for p in paras if p.strip())


def _pptx(raw: bytes) -> str:
    z = _open_zip(raw, ".pptx")
    slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                    key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
    if not slides:
        raise ExtractError("No slides were found in the file.")
    out: list[str] = []
    for n in slides:
        root = _xml(_part(z, n))
        out.extend(_paragraph_text(p, {"t"}, set(), {"br"}) for p in root.iter() if _local(p.tag) == "p")
    return "\n\n".join(p for p in out if p.strip())


def _odt(raw: bytes) -> str:
    z = _open_zip(raw, ".odt")
    root = _xml(_part(z, "content.xml"))
    out: list[str] = []
    for p in root.iter():
        if _local(p.tag) in ("p", "h"):
            out.append(_odt_text(p))
    return "\n\n".join(x for x in out if x.strip())


def _odt_text(el: ET.Element) -> str:
    parts = [el.text or ""]
    for child in el:
        t = _local(child.tag)
        if t in ("s", "tab", "line-break"):
            parts.append(" ")
        else:
            parts.append(_odt_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:
        raise ExtractError("PDF support needs the pypdf package on the server (pip install pypdf).")
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise ExtractError("The PDF is password protected. Remove the password and upload it again.")
        pages = reader.pages
        if len(pages) > MAX_PDF_PAGES:
            raise ExtractError(f"The PDF has more than {MAX_PDF_PAGES} pages. Split it into parts.")
        chunks = [(pg.extract_text() or "") for pg in pages]
    except ExtractError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError, TypeError):
        raise ExtractError("The PDF could not be read. Try saving or printing it to a new PDF and upload that.")
    text = "\n\n".join(chunks)
    if len(re.findall(r"[^\W\d_]{2,}", text)) < 5:
        raise ExtractError("The PDF has no selectable text. It is probably a scan or a photo. Run OCR on it first "
                           "(for example with Google Drive or Adobe), then upload the result.")
    return _unwrap(text)


class _Html(HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg", "head", "nav", "footer", "form", "button"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "ul", "ol", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n\n" if tag != "br" else "\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def _html(raw: bytes) -> str:
    try:
        body = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        body = raw.decode("latin-1")
    p = _Html()
    p.feed(body)
    p.close()
    return re.sub(r"[ \t]+", " ", "".join(p.out))


def _markdown(raw: bytes) -> str:
    try:
        t = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ExtractError("The file is not UTF-8 text.")
    t = re.sub(r"```.*?```", "", t, flags=re.S)                      # fenced code
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)                       # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)                   # links keep their words
    t = re.sub(r"^\s{0,3}(#{1,6}|>|[-*+]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*([-*_]\s*){3,}$", "", t, flags=re.M)            # rules
    t = re.sub(r"(\*\*|__|\*|_|`)", "", t)
    return t


_READERS = {".docx": _docx, ".pptx": _pptx, ".odt": _odt, ".pdf": _pdf, ".html": _html, ".htm": _html,
            ".md": _markdown, ".markdown": _markdown}


def extract_text(raw: bytes, filename: str) -> str:
    ext = extension(filename)
    if ext not in _READERS:
        raise ExtractError("Unsupported file type. Use " + ", ".join(("txt", "csv") + tuple(e[1:] for e in DOCUMENT_EXTENSIONS)) + ".")
    text = unicodedata.normalize("NFC", _READERS[ext](raw))
    if len(text) > MAX_CHARS:
        raise ExtractError("The document has too much text. Split it into parts.")
    return text


# ---------------------------------------------------------------- sentences
_ABBREV = {"dr", "mr", "mrs", "ms", "prof", "fig", "figs", "vs", "etc", "e.g", "i.e", "approx", "no", "nos", "st", "govt", "dept", "ph",
           "al", "eg", "ie", "cf", "viz", "vol", "pp", "ed", "eds", "sr", "jr", "mt", "inc", "ltd", "co", "ref", "tab", "spp", "sp"}
_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")


def _unwrap(text: str) -> str:
    """Joins lines that were only wrapped (PDF and plain-text layouts) and mends words split by a line-end hyphen."""
    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", ""))
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)                 # mal-\nnutrition -> malnutrition
    text = re.sub(r"\n{2,}", "\x00", text)                              # real paragraph breaks
    text = re.sub(r"\n(?=[•●▪◦\-*] )", "\x00", text)                    # bullets start a new unit
    text = re.sub(r"(?<=[.!?:;])\n(?=[A-Z0-9])", "\x00", text)          # line ends a sentence, next starts a new one
    text = re.sub(r"\n", " ", text)
    return text.replace("\x00", "\n\n")


def _cut_long(s: str) -> list[str]:
    out = []
    while len(s) > MAX_SENTENCE_CHARS:
        cut = max(s.rfind(c, 0, MAX_SENTENCE_CHARS) for c in ("; ", ": ", ", ", " "))
        cut = cut if cut > 200 else MAX_SENTENCE_CHARS
        out.append(s[:cut + 1].strip())
        s = s[cut + 1:].strip()
    if s:
        out.append(s)
    return out


def split_sentences(text: str) -> list[str]:
    """One sentence per item. Paragraph and list breaks always split; full stops split unless they follow an abbreviation or a single
    initial. Page numbers, bare numbers and fragments with fewer than two words are dropped, and so are exact repeats (page headers)."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        pieces = _SPLIT.split(para)
        cur = ""
        for piece in pieces:
            cur = f"{cur} {piece}".strip() if cur else piece
            last = re.findall(r"([A-Za-z.]+)[.!?][\"')\]]*$", cur)
            tail = last[-1].lower().rstrip(".") if last else ""
            if cur.endswith(".") and (tail in _ABBREV or re.fullmatch(r"[a-z]", tail or "xx")) :
                continue                                              # abbreviation or initial: keep joining
            units.extend(_cut_long(cur))
            cur = ""
        if cur:
            units.extend(_cut_long(cur))
    out, seen = [], set()
    for u in units:
        words = re.findall(r"[^\W\d_]+", u)
        if len(u) < MIN_SENTENCE_CHARS or len(words) < MIN_SENTENCE_WORDS:
            continue
        key = u.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out
