"""Ces tests exercent les lecteurs réels. Le dernier fait également tourner l'OCR."""

import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import pytest
from docx import Document
from PIL import Image

from app.config import Settings
from app.errors import RagError
from app.extraction import extract_file


DEMO = Path(__file__).resolve().parents[1] / "demo"


def test_markdown_et_texte_utf8():
    original = (DEMO / "atelier.md").read_bytes()
    assert extract_file("atelier.MARKDOWN", original, Settings()) == [(None, original.decode().strip())]
    assert extract_file("texte.txt", b"\xef\xbb\xbfAtelier", Settings()) == [(None, "Atelier")]


def test_docx_conserve_ordre_des_paragraphes_et_tableaux():
    doc = Document()
    doc.add_paragraph("Section A")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Armoire"
    table.cell(0, 1).text = "Indigo"
    doc.add_paragraph("Section B")
    stream = BytesIO()
    doc.save(stream)
    assert extract_file("notes.docx", stream.getvalue(), Settings()) == [
        (None, "Section A\nArmoire\tIndigo\nSection B"),
    ]


@pytest.mark.parametrize(("filename", "content", "status"), [
    ("vide.md", b"  \n ", 422),
    ("absent.txt", b"", 422),
    ("binaire.txt", b"a\x00b", 422),
    ("encodage.txt", b"\xff", 422),
    ("livre.docx", b"archive invalide", 422),
    ("manuel.pdf", b"document invalide", 422),
    ("dessin.jpeg", b"image invalide", 422),
    ("archive.zip", b"contenu", 415),
])
def test_fichiers_non_exploitables(filename, content, status):
    with pytest.raises(RagError) as error:
        extract_file(filename, content, Settings())
    assert error.value.status == status


@pytest.mark.parametrize(("filename", "config"), [
    ("manuel.pdf", {"max_pages": 1}),
    ("atelier.md", {"max_file_bytes": 16}),
    ("atelier.md", {"max_extracted_chars": 16}),
    ("consigne.png", {"max_image_pixels": 100}),
])
def test_plafonds_extraction(filename, config):
    with pytest.raises(RagError) as error:
        extract_file(filename, (DEMO / filename).read_bytes(), Settings(**config))
    assert error.value.status == 413


def test_extension_image_incoherente():
    with pytest.raises(RagError) as error:
        extract_file("renommee.jpg", (DEMO / "consigne.png").read_bytes(), Settings())
    assert error.value.status == 422


def test_pdf_mixte_declenche_ocr_seulement_sur_page_scannee(monkeypatch):
    calls = []

    def simulate(command, timeout, data=None):
        calls.append(command)
        return b"PNG" if command[0] == "pdftoppm" else "Casier Opale".encode()

    monkeypatch.setattr("app.extraction._run_tool", simulate)
    first, second = extract_file("manuel.pdf", (DEMO / "manuel.pdf").read_bytes(), Settings())
    assert first[0] == 1 and "bouton Ambre" in first[1]
    assert second == (2, "Casier Opale")
    render, ocr = calls
    assert render[render.index("-f") + 1] == "2"
    assert ocr[0] == "tesseract"


@pytest.mark.parametrize(("problem", "status"), [
    (FileNotFoundError(), 503),
    (subprocess.TimeoutExpired("tesseract", 1), 504),
    (subprocess.CompletedProcess([], 1, b"", b"Failed loading language"), 503),
    (subprocess.CompletedProcess([], 1, b"", b"unreadable"), 422),
    (subprocess.CompletedProcess([], 0, b"  ", b""), 422),
])
def test_erreurs_ocr(monkeypatch, problem, status):
    def simulate(*args, **kwargs):
        if isinstance(problem, Exception):
            raise problem
        return problem

    monkeypatch.setattr(subprocess, "run", simulate)
    with pytest.raises(RagError) as error:
        extract_file("consigne.png", (DEMO / "consigne.png").read_bytes(), Settings())
    assert error.value.status == status


@pytest.mark.skipif(not shutil.which("tesseract") or not shutil.which("pdftoppm"),
                    reason="Tesseract et Poppler sont installés dans l’image Docker.")
def test_ocr_reel_png_jpeg_et_pdf_mixte():
    image_bytes = (DEMO / "consigne.png").read_bytes()
    jpeg = BytesIO()
    with Image.open(BytesIO(image_bytes)) as image:
        image.save(jpeg, "JPEG")
    for name, data in [("consigne.png", image_bytes), ("consigne.jpg", jpeg.getvalue())]:
        pages = extract_file(name, data, Settings())
        assert len(pages) == 1 and pages[0][0] is None
        assert "cour jasmin" in pages[0][1].lower()
    first, second = extract_file("manuel.pdf", (DEMO / "manuel.pdf").read_bytes(), Settings())
    assert first[0] == 1 and "bouton Ambre" in first[1]
    assert second[0] == 2 and "casier opale" in second[1].lower()
