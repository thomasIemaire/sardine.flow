\
"""GLiNER2 helpers with an in-memory cache."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Sequence, Union

from .config import LabelSpec
from .utils import require

try:
    from gliner2 import GLiNER2  # type: ignore
except Exception:  # pragma: no cover
    GLiNER2 = None  # type: ignore[assignment]


_GLINER2_CACHE: Dict[str, Any] = {}
_GLINER2_LOCK = threading.Lock()


def get_cached_gliner2_model(model_id: str) -> Any:
    """Load GLiNER2 only once (thread-safe)."""
    require(GLiNER2, "gliner2")

    key = str(model_id).strip()
    if not key:
        raise ValueError("model_id GLiNER2 vide")

    if key in _GLINER2_CACHE:
        return _GLINER2_CACHE[key]

    with _GLINER2_LOCK:
        if key not in _GLINER2_CACHE:
            _GLINER2_CACHE[key] = GLiNER2.from_pretrained(key)

    return _GLINER2_CACHE[key]


def _normalize_labels(labels: Sequence[LabelSpec]) -> List[str]:
    """Turn [{key: desc}, ...] into ["key :: desc", ...] (or just strings)."""
    sep = " :: "
    out: List[str] = []
    for l in labels:
        if isinstance(l, dict) and l:
            k = next(iter(l.keys()))
            out.append(f"{k}{sep}{l[k]}")
        else:
            out.append(str(l))
    return out


def classify_texts(
    texts: List[str],
    *,
    model_id: str,
    labels: Sequence[LabelSpec],
    multi_label: bool,
    threshold: float,
    include_confidence: bool,
) -> List[Any]:
    """Classify short texts into high-level labels."""
    model = get_cached_gliner2_model(model_id)

    sep = " :: "
    task_name = "text_classification"
    tasks = {
        task_name: {
            "labels": _normalize_labels(labels),
            "multi_label": multi_label,
            "cls_threshold": threshold,
        }
    }

    raw = model.batch_classify_text(
        texts,
        tasks=tasks,
        threshold=threshold,
        format_results=True,
        include_confidence=include_confidence,
        include_spans=False,
    )

    flat = [r.get(task_name) for r in raw]

    # Remove label descriptions (keep only the key).
    for r in flat:
        for e in r:
            e["label"] = e["label"].split(sep)[0] if sep in e["label"] else e["label"]

    return flat
