\
"""Helpers to decode a base64 (image or PDF) into PIL images."""

from __future__ import annotations

import base64
import io
import re
from typing import Any, List, Optional, Tuple, Literal

from PIL import Image

from .utils import require

try:
    import fitz  # pymupdf
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]


_DATAURL_RE = re.compile(r"data:(?P<mime>[\w/-]+)?(;base64)?,(?P<data>.+)", re.IGNORECASE)


def _try_decode_base64(s: Any) -> Tuple[Optional[bytes], Optional[str]]:
    if not isinstance(s, str) or not s.strip():
        return None, None

    candidate = s.strip()

    # data URL (data:application/pdf;base64,....)
    m = _DATAURL_RE.match(candidate)
    if m:
        try:
            return base64.b64decode(m.group("data"), validate=False), (m.group("mime") or "").lower()
        except Exception:
            return None, None

    # raw base64
    try:
        return base64.b64decode(candidate, validate=False), None
    except Exception:
        return None, None


def _load_pil_from_bytes(img_bytes: bytes, *, convert: str = "L") -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes))
    return img.convert(convert)


def _load_pil_from_pdf(
    raw_bytes: bytes,
    *,
    page_mode: Literal["first_page_only", "all_pages"] = "all_pages",
    dpi: int = 200,
    convert: str = "L",
) -> List[Image.Image]:
    require(fitz, "pymupdf (fitz)")

    images: List[Image.Image] = []
    with fitz.open(stream=raw_bytes, filetype="pdf") as doc:  # type: ignore[union-attr]
        if doc.page_count < 1:
            raise ValueError("PDF vide (0 page).")

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # type: ignore[union-attr]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes()  # PNG bytes
            images.append(_load_pil_from_bytes(img_bytes, convert=convert))
            if page_mode == "first_page_only":
                break

    return images


def load_images_from_base64(
    base64_data: str,
    *,
    page_mode: Literal["first_page_only", "all_pages"] = "all_pages",
    page_dpi: int = 200,
    page_convert: Literal["L", "RGB", "1"] = "L",
) -> List[Image.Image]:
    """Decode a base64 payload (image or PDF) into PIL images."""
    raw_bytes, mime = _try_decode_base64(base64_data)
    if raw_bytes is None:
        raise ValueError("Base64 invalide: impossible de décoder.")

    is_pdf = (mime == "application/pdf") or (mime is None and raw_bytes[:4] == b"%PDF")
    if is_pdf:
        return _load_pil_from_pdf(raw_bytes, page_mode=page_mode, dpi=page_dpi, convert=page_convert)

    return [_load_pil_from_bytes(raw_bytes, convert=page_convert)]
