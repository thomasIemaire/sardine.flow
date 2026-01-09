\
"""YOLO helpers (classification + detection) with an in-memory cache."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Literal, Optional, TypedDict

from PIL import Image
import torch

from .utils import log_debug, require

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


class Zone(TypedDict, total=False):
    type: str
    class_id: int
    conf: Optional[float]
    bbox: List[int]  # [x1,y1,x2,y2]


_YOLO_CACHE: Dict[str, Any] = {}
_YOLO_LOCK = threading.Lock()


def get_cached_model(model_path: str) -> Any:
    """Load a YOLO model only once (thread-safe)."""
    require(YOLO, "ultralytics")

    abs_path = os.path.abspath(model_path)
    if abs_path in _YOLO_CACHE:
        return _YOLO_CACHE[abs_path]

    with _YOLO_LOCK:
        if abs_path not in _YOLO_CACHE:
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Modèle YOLO introuvable: {abs_path}")
            _YOLO_CACHE[abs_path] = YOLO(abs_path)  # type: ignore[misc]

    return _YOLO_CACHE[abs_path]


def _select_pages(images: List[Image.Image], page_mode: Literal["first_page_only", "all_pages"]) -> List[Image.Image]:
    if page_mode == "first_page_only":
        return images[:1]
    return images


def _resolve_device(device: str, debug: bool) -> str:
    requested = (device or "cpu").lower()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            log_debug(
                "CUDA indisponible; exécution sur CPU.",
                debug,
                tags=["yolo", "device"],
            )
            return "cpu"
        capability = torch.cuda.get_device_capability()
        arch = f"sm_{capability[0]}{capability[1]}"
        supported = torch.cuda.get_arch_list()
        if supported and arch not in supported:
            log_debug(
                f"Architecture GPU {arch} non supportée par cette version de PyTorch; "
                "exécution sur CPU.",
                debug,
                tags=["yolo", "device"],
            )
            return "cpu"
    return device


def classify_images(
    images: List[Image.Image],
    *,
    model_path: str,
    page_mode: Literal["first_page_only", "all_pages"] = "first_page_only",
    device: str = "cpu",
    debug: bool = False,
) -> List[Any]:
    """Run image-level classification on each selected page."""
    model = get_cached_model(model_path)
    pages = _select_pages(images, page_mode)
    resolved_device = _resolve_device(device, debug)

    # ultralytics accepts either a single image or a list. We always pass a list for consistency.
    predictions = model.predict(source=pages, device=resolved_device, save=False, verbose=False)

    results: List[Any] = []
    for res in predictions:
        if getattr(res, "probs", None) is not None:
            results.append(res.probs.top1)  # type: ignore[attr-defined]
        else:
            results.append("unknown")
    return results


def detect_zones(
    images: List[Image.Image],
    *,
    model_path: str,
    page_mode: Literal["first_page_only", "all_pages"] = "first_page_only",
    device: str = "cpu",
    confidence: float = 0.4,
    padding: int = 8,
    debug: bool = False,
) -> List[List[Zone]]:
    """Detect zones (text, table, logo, signature, ...) on each selected page."""
    require(np, "numpy")

    DEFAULT_ZONE_NAMES = {0: "text", 1: "text-column", 2: "table", 3: "logo", 4: "signature"}

    model = get_cached_model(model_path)
    pages = _select_pages(images, page_mode)
    resolved_device = _resolve_device(device, debug)

    det_results = model.predict(source=pages, device=resolved_device, conf=confidence, iou=.5, save=False, verbose=False)

    out: List[List[Zone]] = []
    for img, res in zip(pages, det_results):
        w, h = img.size
        names = getattr(res, "names", None) or getattr(model, "names", None) or DEFAULT_ZONE_NAMES

        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            out.append([])
            continue

        xyxy = boxes.xyxy
        cls = boxes.cls
        confs = getattr(boxes, "conf", None)

        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy.numpy()
        cls = cls.cpu().numpy() if hasattr(cls, "cpu") else cls.numpy()
        confs_np = None
        if confs is not None:
            confs_np = confs.cpu().numpy() if hasattr(confs, "cpu") else confs.numpy()

        xyxy = xyxy.astype(int)
        cls = cls.astype(int)

        # Reading order: top-to-bottom then left-to-right
        order = np.lexsort((xyxy[:, 0], xyxy[:, 1]))
        xyxy = xyxy[order]
        cls = cls[order]
        if confs_np is not None:
            confs_np = confs_np[order]

        page_items: List[Zone] = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            x1p, y1p = max(0, int(x1) - padding), max(0, int(y1) - padding)
            x2p, y2p = min(w, int(x2) + padding), min(h, int(y2) + padding)
            if x2p <= x1p or y2p <= y1p:
                continue

            c = int(cls[i])
            cconf = float(confs_np[i]) if confs_np is not None else None
            zone_type = names.get(c, str(c)) if isinstance(names, dict) else str(c)

            page_items.append(
                {
                    "type": str(zone_type),
                    "class_id": c,
                    "conf": cconf,
                    "bbox": [x1p, y1p, x2p, y2p],
                }
            )

        out.append(page_items)

    return out
