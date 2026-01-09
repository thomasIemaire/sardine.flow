\
"""End-to-end pipeline that goes from base64 -> pages -> zones -> OCR -> text classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import PipelineConfig
from .image_loading import load_images_from_base64
from .models_yolo import classify_images, detect_zones
from .zones import extract_zones
from .ocr import ocr_zones
from .models_gliner2 import classify_texts
from .utils import time_call


@dataclass
class PipelineResult:
    images: List[Image.Image]
    page_classes: List[Any]
    zones: List[List[Dict[str, Any]]]
    extracted_zones: List[List[Image.Image]]
    raw_texts: List[List[str]]
    cleaned_texts: List[List[str]]
    classified_texts: List[Any]
    logs: List[Dict[str, Any]]


def _clean_text(s: str) -> str:
    return s.strip().replace("\n", " ").replace("\r", " ")


def run_pipeline(base64_data: str, config: PipelineConfig) -> PipelineResult:
    """Run the full pipeline and return both outputs and timing logs."""
    logs: List[Dict[str, Any]] = []

    images, log = time_call(
        load_images_from_base64,
        base64_data,
        page_mode=config.page.mode,
        page_dpi=config.page.dpi,
        page_convert=config.page.convert,
        debug=config.debug,
    )
    logs.append(log)

    page_classes, log = time_call(
        classify_images,
        images,
        model_path=config.yolo_cls.model_path,
        page_mode=config.page.mode,
        device=config.yolo_cls.device,
        debug=config.debug,
    )
    logs.append(log)

    zones, log = time_call(
        detect_zones,
        images,
        model_path=config.yolo_det.model_path,
        page_mode=config.page.mode,
        device=config.yolo_det.device,
        confidence=config.yolo_det.confidence,
        padding=config.yolo_det.padding,
        debug=config.debug,
    )
    logs.append(log)

    extracted, log = time_call(
        extract_zones,
        images,
        zones,
        exclude_zone_classes=config.ocr.exclude_zone_classes,
        debug=config.debug,
    )
    logs.append(log)

    raw_texts, log = time_call(
        ocr_zones,
        extracted,
        lang=config.ocr.lang,
        config=config.ocr.config,
        debug=config.debug,
    )
    logs.append(log)

    cleaned_texts = [[_clean_text(t) for t in page] for page in raw_texts]
    flat_texts = [t for page in cleaned_texts for t in page]

    labels = [{ "unlabeled": "Unlabeled" }]
    for a in config.gliner2.agents:
        if a.get("target_zone"):
            labels.append({ a.get("reference"): a.get("description") })

    classified, log = time_call(
        classify_texts,
        flat_texts,
        model_id=config.gliner2.model_id,
        labels=labels,
        multi_label=config.gliner2.multi_label,
        threshold=config.gliner2.threshold,
        include_confidence=config.gliner2.include_confidence,
        debug=config.debug,
    )
    logs.append(log)

    return PipelineResult(
        images=images,
        page_classes=page_classes,
        zones=zones,
        extracted_zones=extracted,
        raw_texts=raw_texts,
        cleaned_texts=cleaned_texts,
        classified_texts=classified,
        logs=logs,
    )


def run_logs_only(base64_data: str, config: PipelineConfig) -> List[Dict[str, Any]]:
    """Backward-compatible helper that returns only the logs (like your old `run`)."""
    return run_pipeline(base64_data, config).logs
