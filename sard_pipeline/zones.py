\
"""Utilities to crop detected zones out of page images."""

from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image


def extract_zones(
    images: List[Image.Image],
    zones: List[List[Dict[str, Any]]],
    *,
    exclude_zone_classes: List[str],
) -> List[List[Image.Image]]:
    """Crop zones from page images.

    Parameters
    ----------
    images:
        One image per page.
    zones:
        One list per page, with items containing "bbox" and "type".
    exclude_zone_classes:
        Any zone with ``zone["type"]`` in this list is skipped.
    """
    out: List[List[Image.Image]] = []

    for page_img, page_zones in zip(images, zones):
        page_extracted: List[Image.Image] = []
        for zone in page_zones:
            if zone.get("type") in exclude_zone_classes:
                continue
            x1, y1, x2, y2 = zone["bbox"]
            page_extracted.append(page_img.crop((x1, y1, x2, y2)))
        out.append(page_extracted)

    return out
