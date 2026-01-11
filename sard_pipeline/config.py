"""Configuration for the document pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Sequence, Union


PageMode = Literal["first_page_only", "all_pages"]


@dataclass(frozen=True)
class PageConfig:
    mode: PageMode = "first_page_only"
    dpi: int = 300
    convert: Literal["L", "RGB", "1"] = "RGB"


@dataclass(frozen=True)
class YoloClassificationConfig:
    model_path: str = "../sardine.agents/sard-cls/best.pt"
    device: str = "cpu"


@dataclass(frozen=True)
class YoloDetectionConfig:
    model_path: str = "../sardine.agents/sard-det/best.pt"
    device: str = "cpu"
    confidence: float = 0.4
    iou: float = 0.5
    padding: int = 8


@dataclass(frozen=True)
class OcrConfig:
    exclude_zone_classes: List[str] = field(default_factory=lambda: ["logo", "signature"])
    lang: str = "fra+eng"
    config: str = "--oem 1 --psm 6"
    device: str = "cpu"


LabelSpec = Union[str, Dict[str, str]]


@dataclass(frozen=True)
class Gliner2Config:
    """Config pour la CLASSIFICATION de texte (via gliner2)."""
    model_id: str = "fastino/gliner2-large-2907"
    multi_label: bool = False
    threshold: float = 0.2
    include_confidence: bool = False
    device: str = "cpu"


@dataclass(frozen=True)
class GlinerExtractionConfig:
    """Config pour l'EXTRACTION d'entités (via gliner standard)."""
    model_id: str = "urchade/gliner_multi-v2.1"
    # Les agents définissent quelles entités chercher
    agents: List[Dict[str, Any]] = field(default_factory=list)
    threshold: float = 0.3
    device: str = "cpu"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""
    page: PageConfig = field(default_factory=PageConfig)
    yolo_cls: YoloClassificationConfig = field(default_factory=YoloClassificationConfig)
    yolo_det: YoloDetectionConfig = field(default_factory=YoloDetectionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    # Classification (GLiNER2)
    gliner2: Gliner2Config = field(default_factory=Gliner2Config)
    # Extraction (GLiNER standard)
    gliner_extract: GlinerExtractionConfig = field(default_factory=GlinerExtractionConfig)
    debug: bool = False