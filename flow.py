\
"""CLI entrypoint + backward-compatible functional API.

Historically, this file contained the whole pipeline. It is now a thin wrapper
around the `sard_pipeline` package, while keeping the same function names.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Literal, Sequence, Union

from PIL import Image

from helpers import _log_debug, _timed_function
from sard_pipeline import (
    PipelineConfig,
    PageConfig,
    YoloClassificationConfig,
    YoloDetectionConfig,
    OcrConfig,
    Gliner2Config,
    run_logs_only,
)
from sard_pipeline.image_loading import load_images_from_base64
from sard_pipeline.models_yolo import classify_images, detect_zones
from sard_pipeline.zones import extract_zones
from sard_pipeline.ocr import ocr_zones
from sard_pipeline.models_gliner2 import classify_texts


# ---------------------------------------------------------------------------
# Defaults (kept to match your original script's CLI defaults)
# ---------------------------------------------------------------------------

_DEFAULT_AGENTS = [
    {
        "name": "Siren",
        "reference": "siren",
        "description": "Siren d'une entreprise.",
        "target_zone": True,
        "root": "",
        "mapper": {
            "siren": { 
                "type": "str",
                "description": "Numéro SIREN de l'entreprise.",
                "requirements": { "rule": "regex", "pattern": r"(\d{3}\s*){3}" }
            }
        }
    },
    {
        "name": "Numéro de TVA",
        "reference": "vat-number",
        "description": "Numéro de TVA intracommunautaire.",
        "target_zone": True,
        "root": "",
        "mapper": {
            "vat.number": { 
                "type": "str",
                "description": "Numéro de TVA intracommunautaire.",
                "requirements": { "rule": "regex", "pattern": r"[A-Z]{2}\s*\d{2}\s*(\d{3}\s*){3}" }
            }
        }
    },
    {
        "name": "Montants",
        "reference": "amounts",
        "description": "Montants présents dans le document.",
        "target_zone": True,
        "root": "",
        "mapper": {
            "amounts.ttc": { 
                "type": "float",
                "description": "Montant toutes taxes comprises."
            },
            "amounts.ht": { 
                "type": "float",
                "description": "Montant hors taxes."
            },
            "amounts.tva": { 
                "type": "float",
                "description": "Montant de la TVA."
            }
        }
    },
    {
        "name": "Devise",
        "reference": "currency",
        "description": "Devise utilisée pour les montants.",
        "target_zone": False,
        "root": "",
        "mapper": {
            "currency": { 
                "type": "str",
                "description": "Devise utilisée (ex: EUR, USD)."
            }
        }
    },
    {
        "name": "Adresse postale",
        "reference": "address",
        "description": "Adresse postale complète.",
        "target_zone": True,
        "root": "",
        "mapper": {
            "address.name": {
                "type": "str",
                "description": "Nom de la personne ou de l'entreprise."
            },
            "address.street": { 
                "type": "str",
                "description": "Adresse postale complète."
            },
            "address.zip_code": { 
                "type": "str",
                "description": "Code postal.",
                "requirements": { "rule": "regex", "pattern": r"\d{5}" }
            },
            "address.city": { 
                "type": "str",
                "description": "Ville."
            },
            "address.country": { 
                "type": "str",
                "description": "Pays."
            }
        }
    }
]

_DEFAULT_PAGE_MODE: Literal["first_page_only", "all_pages"] = "first_page_only"
_DEFAULT_DPI = 300
_DEFAULT_PAGE_CONVERT: Literal["L", "RGB", "1"] = "RGB"

_DEFAULT_DEVICE = "cpu"
_DEFAULT_SARD_CLS_MODEL_PATH = "../sardine.agents/sard-cls/best.pt"
_DEFAULT_SARD_DET_MODEL_PATH = "../sardine.agents/sard-det/best.pt"
_DEFAULT_SARD_DET_CONFIDENCE = 0.5
_DEFAULT_SARD_DET_PADDING = 8

_DEFAULT_EXCLUDE_ZONES_CLASSES = ["logo", "signature"]

_DEFAULT_OCR_LANG = "fra+eng"
_DEFAULT_OCR_CONFIG = "--oem 1 --psm 6"

_DEFAULT_GLINER2_MODEL_ID = "fastino/gliner2-large-2907"
_DEFAULT_GLIN_CLS_MULTI_LABEL = False
_DEFAULT_GLIN_CLS_THRESHOLD = 0.2
_DEFAULT_GLIN_CLS_INCLUDE_CONFIDENCE = False


# ---------------------------------------------------------------------------
# Backward-compatible "run" (same signature, returns logs)
# ---------------------------------------------------------------------------

def run(
    base64_data: str,
    agents: List[Dict[str, Any]],
    page_mode: str = _DEFAULT_PAGE_MODE,
    page_dpi: int = _DEFAULT_DPI,
    page_convert: str = _DEFAULT_PAGE_CONVERT,
    device: str = _DEFAULT_DEVICE,
    cls_model_path: str = _DEFAULT_SARD_CLS_MODEL_PATH,
    det_model_path: str = _DEFAULT_SARD_DET_MODEL_PATH,
    det_confidence: float = _DEFAULT_SARD_DET_CONFIDENCE,
    det_padding: int = _DEFAULT_SARD_DET_PADDING,
    exclude_zones_classes: List[str] = _DEFAULT_EXCLUDE_ZONES_CLASSES,
    ocr_lang: str = _DEFAULT_OCR_LANG,
    ocr_config: str = _DEFAULT_OCR_CONFIG,
    glin_model_id: str = _DEFAULT_GLINER2_MODEL_ID,
    glin_multi_label: bool = _DEFAULT_GLIN_CLS_MULTI_LABEL,
    glin_threshold: float = _DEFAULT_GLIN_CLS_THRESHOLD,
    glin_include_confidence: bool = _DEFAULT_GLIN_CLS_INCLUDE_CONFIDENCE,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Backward-compatible pipeline runner returning only the timing logs."""
    cfg = PipelineConfig(
        page=PageConfig(mode=page_mode, dpi=page_dpi, convert=page_convert),  # type: ignore[arg-type]
        yolo_cls=YoloClassificationConfig(model_path=cls_model_path, device=device),
        yolo_det=YoloDetectionConfig(
            model_path=det_model_path,
            device=device,
            confidence=det_confidence,
            padding=det_padding,
        ),
        ocr=OcrConfig(
            exclude_zone_classes=exclude_zones_classes,
            lang=ocr_lang,
            config=ocr_config,
            device=device,
        ),
        gliner2=Gliner2Config(
            model_id=glin_model_id,
            agents=agents,  # type: ignore[arg-type]
            multi_label=glin_multi_label,
            threshold=glin_threshold,
            include_confidence=glin_include_confidence,
            device=device,
        ),
        debug=debug,
    )
    return run_logs_only(base64_data, cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_base64_arg(value: str) -> str:
    # Backward-compat: historically --base64 was actually a file path.
    if os.path.exists(value):
        with open(value, "r", encoding="utf-8") as f:
            return f.read()
    return value


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--base64", type=str, required=True, help="Base64 string or path to a text file containing base64")
    ap.add_argument("--agents", type=str, default=json.dumps(_DEFAULT_AGENTS), help="JSON string defining the agents")

    ap.add_argument("--page_mode", type=str, choices=["first_page_only", "all_pages"], default=_DEFAULT_PAGE_MODE)
    ap.add_argument("--page_dpi", type=int, default=_DEFAULT_DPI)
    ap.add_argument("--page_convert", type=str, choices=["L", "RGB", "1"], default=_DEFAULT_PAGE_CONVERT)

    ap.add_argument("--device", type=str, default=_DEFAULT_DEVICE, choices=["cpu", "cuda"])
    ap.add_argument("--cls_model_path", type=str, default=_DEFAULT_SARD_CLS_MODEL_PATH)
    ap.add_argument("--det_model_path", type=str, default=_DEFAULT_SARD_DET_MODEL_PATH)
    ap.add_argument("--det_confidence", type=float, default=_DEFAULT_SARD_DET_CONFIDENCE)
    ap.add_argument("--det_padding", type=int, default=_DEFAULT_SARD_DET_PADDING)

    ap.add_argument("--ocr_exclude_zones_classes", type=str, nargs="*", default=_DEFAULT_EXCLUDE_ZONES_CLASSES)

    ap.add_argument("--ocr_lang", type=str, default=_DEFAULT_OCR_LANG)
    ap.add_argument("--ocr_config", type=str, default=_DEFAULT_OCR_CONFIG)

    ap.add_argument("--glin_model_id", type=str, default=_DEFAULT_GLINER2_MODEL_ID)
    ap.add_argument("--glin_multi_label", action="store_true", default=_DEFAULT_GLIN_CLS_MULTI_LABEL)
    ap.add_argument("--glin_threshold", type=float, default=_DEFAULT_GLIN_CLS_THRESHOLD)
    ap.add_argument("--glin_include_confidence", action="store_true", default=_DEFAULT_GLIN_CLS_INCLUDE_CONFIDENCE)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save-logs", action="store_true", help="Save timing logs to logs.csv")
    ap.add_argument("--runs", type=int, default=1, help="Number of runs to execute")

    args = ap.parse_args()

    base64_data = _read_base64_arg(args.base64)

    # Run N times (useful for warmup / benchmarking)
    last_outer_log: Dict[str, Any] | None = None
    last_inner_logs: List[Dict[str, Any]] = []

    for i in range(args.runs):
        inner_logs, outer_log = _timed_function(
            run,
            base64_data,
            json.loads(args.agents),
            page_mode=args.page_mode,
            page_dpi=args.page_dpi,
            page_convert=args.page_convert,
            device=args.device,
            cls_model_path=args.cls_model_path,
            det_model_path=args.det_model_path,
            det_confidence=args.det_confidence,
            det_padding=args.det_padding,
            exclude_zones_classes=args.ocr_exclude_zones_classes,
            ocr_lang=args.ocr_lang,
            ocr_config=args.ocr_config,
            glin_model_id=args.glin_model_id,
            glin_multi_label=args.glin_multi_label,
            glin_threshold=args.glin_threshold,
            glin_include_confidence=args.glin_include_confidence,
            debug=args.debug,
        )

        last_outer_log = outer_log
        last_inner_logs = inner_logs

        _log_debug(f"============= Run {i+1} completed.", args.debug, tags=["main", "run"])

    if args.save_logs and last_outer_log is not None:
        logs = [last_outer_log, *last_inner_logs]

        with open("logs.csv", "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = logs[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)


if __name__ == "__main__":
    main()
