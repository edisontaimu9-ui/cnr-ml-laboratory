"""Pre-label from documents: .docx .pptx .odt .pdf .html .md are turned into sentences in memory, hostile files are refused, and the
result goes through the same identifier check and CSV format as a .txt upload."""
import csv
import io
import zipfile

import pytest

from app import nlp_extract as X
from app import nlp_prelabel

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
O = 'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'


def _zip(name, data):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, data)
    return b.getvalue()


def docx(paras):
    body = "".join(f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paras)
    return _zip("word/document.xml", f"<w:document {W}><w:body>{body}</w:body></w:document>")


def pdf(lines):
    content = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
    objs = ["<</Type/Catalog/Pages 2 0 R>>", "<</Type/Pages/Kids[3 0 R]/Count 1>>",
            "<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
            f"<</Length {len(content)}>>\nstream\n{content}\nendstream", "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    return out + f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF".encode()


def sentences(raw, name):
    return X.split_sentences(X.extract_text(raw, name))


def test_docx_paragraphs_abbreviations_and_decimals():
    s = sentences(docx(["PEM is common in children. See Fig. 2 for details.", "MUAC below 11.5 cm is severe, e.g. in kwashiorkor.",
                        "Mwana ali ndi kutupa."]), "a.docx")
    assert s == ["PEM is common in children.", "See Fig. 2 for details.", "MUAC below 11.5 cm is severe, e.g. in kwashiorkor.",
                 "Mwana ali ndi kutupa."]


def test_pptx_slides_in_order():
    def slide(t):
        return f'<p:sld xmlns:p="p" {A}><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:sld>'
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("ppt/slides/slide10.xml", slide("Tenth slide of the deck."))
        z.writestr("ppt/slides/slide2.xml", slide("Second slide of the deck."))
        z.writestr("ppt/slides/slide1.xml", slide("First slide of the deck."))
    assert sentences(b.getvalue(), "d.pptx") == ["First slide of the deck.", "Second slide of the deck.", "Tenth slide of the deck."]


def test_odt_spans_and_spaces():
    raw = _zip("content.xml", f"<c {O}><text:p>Mwana adadya <text:span>nsima</text:span> dzulo.<text:s/>Anamwa mkaka.</text:p></c>")
    assert sentences(raw, "o.odt") == ["Mwana adadya nsima dzulo.", "Anamwa mkaka."]


def test_html_skips_scripts_and_navigation():
    raw = (b"<html><head><title>x</title></head><body><nav>Menu items</nav><h1>Diet notes</h1>"
           b"<p>The child ate nsima. She refused beans.</p><script>var a = 1;</script></body></html>")
    assert sentences(raw, "a.html") == ["Diet notes", "The child ate nsima.", "She refused beans."]


def test_markdown_markup_is_stripped():
    s = sentences(b"# Title here\n\n- **Marasmus**: no oedema. See [guide](http://x).\n", "a.md")
    assert s == ["Title here", "Marasmus: no oedema.", "See guide."]


def test_pdf_joins_wrapped_lines_and_mends_hyphens():
    raw = pdf(["Malnutrition is a range of conditions occurring when the", "intake of one or more nutrients does not meet the require-",
               "ments. PEM affects preschool children in Malawi."])
    assert sentences(raw, "n.pdf") == ["Malnutrition is a range of conditions occurring when the intake of one or more nutrients does not "
                                       "meet the requirements.", "PEM affects preschool children in Malawi."]


def test_scanned_pdf_is_refused_with_advice():
    with pytest.raises(X.ExtractError, match="no selectable text"):
        X.extract_text(pdf(["12"]), "scan.pdf")


@pytest.mark.parametrize("raw,name,msg", [
    (b"hello", "x.docx", "not a valid .docx"),
    (_zip("other.xml", "<a/>"), "x.docx", "missing its main text"),
    (_zip("word/document.xml", f'<!DOCTYPE x [<!ENTITY a "b">]><w:document {W}/>'), "x.docx", "unsafe XML"),
    (b"%PDF-1.4 broken", "x.pdf", "could not be read"),
])
def test_bad_files_are_refused(raw, name, msg):
    with pytest.raises(X.ExtractError, match=msg):
        X.extract_text(raw, name)


def test_repeats_and_fragments_are_dropped_and_long_runs_are_cut():
    s = X.split_sentences("Page 12\n\nThe child ate nsima.\n\nThe child ate nsima.\n\n7\n\nOk.\n\n" + ("word " * 400).strip() + ".")
    assert "Page 12" not in s and s[0] == "The child ate nsima." and s.count("The child ate nsima.") == 1 and "Ok." not in s and "7" not in s
    assert all(len(x) <= X.MAX_SENTENCE_CHARS for x in s)


def test_prelabel_reads_a_document_and_returns_the_corpus_csv():
    raw = docx(["The child ate rice and beans.", "Mwana ali ndi kutupa."])
    res = nlp_prelabel.prelabel(raw, "notes.docx")
    rows = list(csv.DictReader(io.StringIO(res["csv"])))
    assert [r["doc_id"] for r in rows] == ["s-1", "s-2"]
    assert "rice|FOOD" in rows[0]["entities"] and "kutupa|SYMPTOM" in rows[1]["entities"]
    assert res["report"]["documents"] == 2


def test_prelabel_refuses_identifiers_inside_a_document():
    with pytest.raises(nlp_prelabel.PrelabelError, match="Identifiers were found"):
        nlp_prelabel.prelabel(docx(["Call the mother on 0999 123 456 about the child."]), "n.docx")


def test_prelabel_txt_and_csv_behave_as_before():
    assert [t for _, t in nlp_prelabel.read_sentences(b"One sentence here.\nAnother sentence here.\n", "a.txt")] == \
        ["One sentence here.", "Another sentence here."]
    assert nlp_prelabel.read_sentences(b"text\nHello there friend.\n", "a.csv") == [("row-2", "Hello there friend.")]
