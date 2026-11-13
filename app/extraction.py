"""Extraction du texte avant le découpage. Les pages PDF sont numérotées dès 1."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Iterable, Iterator
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from docx import Document
from docx.table import Table
from PIL import Image, ImageOps, UnidentifiedImageError
from pypdf import PdfReader

from .config import Settings
from .errors import RagError


_FORMATS = {
    ".txt": "txt", ".md": "md", ".markdown": "md", ".docx": "docx",
    ".pdf": "pdf", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg",
}


def source_type(name: str) -> str:
    """Normalise l'extension avant de vérifier le contenu avec son lecteur."""
    kind = _FORMATS.get(Path(name).suffix.lower())
    if kind is None:
        raise RagError(415, "Format non pris en charge (TXT, MD, DOCX, PDF, PNG, JPEG).")
    return kind


def _run_tool(command: list[str], timeout: float, data: bytes | None = None) -> bytes:
    """Appelle un outil sans shell et transmet uniquement des erreurs lisibles à l'API."""
    try:
        process = subprocess.run(command, input=data, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RagError(503, f"Outil d’extraction indisponible : {command[0]}.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RagError(504, f"Délai d’extraction dépassé pour {command[0]}.") from exc
    if b"Failed loading language" in process.stderr or b"Error opening data file" in process.stderr:
        raise RagError(503, "Les langues OCR configurées ne sont pas installées.")
    if process.returncode:
        raise RagError(422, f"{command[0]} n’a pas pu traiter le fichier.")
    return process.stdout


def _ocr(png: bytes, settings: Settings) -> str:
    # « stdin » et « stdout » évitent d'exposer des noms de fichiers utilisateur.
    output = _run_tool(
        ["tesseract", "stdin", "stdout", "-l", settings.ocr_languages],
        settings.ocr_timeout,
        data=png,
    )
    return output.decode("utf-8", errors="replace").strip()


def _image_text(data: bytes, kind: str, settings: Settings) -> str:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != kind.upper():
                raise RagError(422, "L’image ne correspond pas à son extension.")
            if image.width * image.height > settings.max_image_pixels:
                raise RagError(413, "L’image dépasse la limite de pixels.")
            # Une photo de téléphone peut être orientée par ses métadonnées EXIF.
            # Le fond blanc permet à l'OCR de lire aussi les PNG transparents.
            oriented = ImageOps.exif_transpose(image).convert("RGBA")
            background = Image.new("RGB", oriented.size, "white")
            background.paste(oriented, mask=oriented.getchannel("A"))
            with BytesIO() as buffer:
                background.save(buffer, format="PNG")
                png = buffer.getvalue()
    except Image.DecompressionBombError as exc:
        raise RagError(413, "L’image dépasse la limite de pixels.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise RagError(422, "Image invalide ou endommagée.") from exc
    return _ocr(png, settings)


def _pdf_pages(data: bytes, settings: Settings) -> Iterator[tuple[int, str]]:
    """Lit chaque page indépendamment pour conserver la pagination du PDF.

    Un PDF peut combiner une page de texte et une page scannée. Si la lecture
    directe rend une chaîne vide pour une page, Poppler dessine cette page et
    Tesseract reconnaît les caractères dans l'image obtenue.
    """
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise RagError(422, "Les fichiers PDF chiffrés ne sont pas pris en charge.")
        if len(reader.pages) > settings.max_pages:
            raise RagError(413, "Le PDF dépasse la limite de pages.")
        with TemporaryDirectory() as folder:
            # Le fichier temporaire n'est créé que si une page nécessite un rendu OCR.
            path = Path(folder) / "document.pdf"
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    if not path.exists():
                        path.write_bytes(data)
                    # Le PDF exprime sa taille en points (72 par pouce). On vise
                    # 200 pixels par pouce pour l'OCR, puis on limite le grand côté
                    # pour tenir dans le budget de pixels, même si la page est carrée.
                    longest = max(float(page.mediabox.width), float(page.mediabox.height))
                    natural_pixels = max(1, math.ceil(longest * 200 / 72))
                    side = min(3500, math.isqrt(settings.max_image_pixels), natural_pixels)
                    rendered = _run_tool(
                        ["pdftoppm", "-f", str(number), "-l", str(number), "-singlefile",
                         "-scale-to", str(side), "-png", str(path)],
                        settings.ocr_timeout,
                    )
                    text = _ocr(rendered, settings)
                yield number, text
    except RagError:
        raise
    except Exception as exc:
        # Les PDF endommagés peuvent échouer dès l'ouverture ou à la lecture d'une page.
        raise RagError(422, "PDF invalide ou endommagé.") from exc


def _docx_parts(data: bytes, settings: Settings) -> Iterator[str]:
    try:
        # DOCX est une archive ZIP. Le plafond décompressé protège aussi les fichiers
        # qui contiendraient une pièce jointe volumineuse hors du corps à indexer.
        with ZipFile(BytesIO(data)) as archive:
            if sum(member.file_size for member in archive.infolist()) > 4 * settings.max_file_bytes:
                raise RagError(413, "Le DOCX dépasse la limite après décompression.")
        document = Document(BytesIO(data))
        for block in document.iter_inner_content():
            if isinstance(block, Table):
                for row in block.rows:
                    yield "\t".join(cell.text for cell in row.cells)
            else:
                yield block.text
    except RagError:
        raise
    except Exception as exc:
        raise RagError(422, "Document DOCX invalide ou endommagé.") from exc


def _plain_text(data: bytes) -> str:
    try:
        # utf-8-sig accepte aussi la marque d'encodage (BOM) présente en tête
        # de certains exports TXT. Elle ne se retrouvera pas dans le passage.
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RagError(422, "Le fichier texte doit être encodé en UTF-8.") from exc
    if any(ord(char) < 32 and char not in "\r\n\t\f" for char in text):
        raise RagError(422, "Le fichier texte contient des données binaires.")
    return text


def extract_file(name: str, data: bytes, settings: Settings) -> list[tuple[int | None, str]]:
    """Fournit au découpage des couples (numéro de page, texte exploitable).

    Par exemple, un passage issu de la deuxième page d'un PDF gardera le numéro 2.
    Pour une image, un fichier texte ou un DOCX, la source est le fichier entier
    et le numéro de page vaut None.
    """
    kind = source_type(name)
    if not data:
        raise RagError(422, "Le fichier est vide.")
    if len(data) > settings.max_file_bytes:
        raise RagError(413, "Le fichier dépasse la taille maximale.")

    pages: Iterable[tuple[int | None, str]]
    if kind == "pdf":
        pages = _pdf_pages(data, settings)
    elif kind == "docx":
        # Word détermine les sauts de page à l'affichage. Conserver l'ordre du corps
        # (paragraphes et tableaux) permet de citer le fichier sans inventer de page.
        pages = [(None, "\n".join(_docx_parts(data, settings)))]
    elif kind in ("png", "jpeg"):
        pages = [(None, _image_text(data, kind, settings))]
    else:
        pages = [(None, _plain_text(data))]

    extracted = []
    total_chars = 0
    for number, text in pages:
        text = text.strip()
        if not text:
            continue
        total_chars += len(text)
        if total_chars > settings.max_extracted_chars:
            raise RagError(413, "Le texte extrait dépasse le nombre de caractères autorisé.")
        extracted.append((number, text))
    if not extracted:
        raise RagError(422, "Aucun texte exploitable n’a pu être extrait du fichier.")
    return extracted
