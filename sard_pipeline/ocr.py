\
"""OCR helpers (Tesseract)."""

from __future__ import annotations

from typing import List

from PIL import Image

from .utils import log_debug, require

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]


def ocr_zones(
    extracted_zones: List[List[Image.Image]],
    *,
    lang: str,
    config: str,
    device: str = "cpu",
    debug: bool = False,
) -> List[List[str]]:
    """Run Tesseract OCR on a list of cropped zones (per page)."""
    require(pytesseract, "pytesseract")

    if device != "cpu":
        log_debug(
            "Tesseract OCR fonctionne uniquement sur CPU; le paramètre device est ignoré.",
            debug,
            tags=["ocr"],
        )

    out: List[List[str]] = []
    for page_zones in extracted_zones:
        page_texts: List[str] = []
        for zone_img in page_zones:
            page_texts.append(pytesseract.image_to_string(zone_img, lang=lang, config=config))
        out.append(page_texts)

    return out
