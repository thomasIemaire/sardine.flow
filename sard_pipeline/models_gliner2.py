\
"""GLiNER2 helpers with an in-memory cache."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union

from .config import LabelSpec
from .utils import _supports_kwarg, require

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


@dataclass(frozen=True)
class _LabelInfo:
    key: str
    description: str
    label: str


def _normalize_label_specs(labels: Sequence[LabelSpec]) -> List[_LabelInfo]:
    sep = " :: "
    out: List[_LabelInfo] = []
    for l in labels:
        if isinstance(l, dict) and l:
            k = next(iter(l.keys()))
            desc = str(l[k])
            label = f"{k}{sep}{desc}" if desc else str(k)
            out.append(_LabelInfo(str(k), desc, label))
        else:
            key = str(l)
            out.append(_LabelInfo(key, "", key))
    return out


def _build_label_map(label_specs: Sequence[_LabelInfo]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for spec in label_specs:
        label_map[spec.key] = spec.key
        label_map[spec.label] = spec.key
    return label_map


def _call_with_optional_kwargs(func: Any, *args: Any, **kwargs: Any) -> Any:
    call_kwargs = {k: v for k, v in kwargs.items() if _supports_kwarg(func, k)}
    return func(*args, **call_kwargs)


def _normalize_entity_payload(payload: Any, label_map: Dict[str, str]) -> List[Dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
        items = payload["entities"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return []

    sep = " :: "
    entities: List[Dict[str, Any]] = []

    def _coerce_label(label: Any) -> str | None:
        if label is None:
            return None
        label_str = str(label)
        if label_str in label_map:
            return label_map[label_str]
        if sep in label_str:
            return label_str.split(sep)[0]
        return label_str

    for item in items:
        if not isinstance(item, dict):
            entities.append({"label": _coerce_label(item), "text": str(item)})
            continue

        if isinstance(item.get("spans"), list):
            for span in item["spans"]:
                if not isinstance(span, dict):
                    continue
                label = _coerce_label(item.get("label") or span.get("label"))
                entities.append(
                    {
                        "label": label,
                        "text": span.get("text"),
                        "start": span.get("start"),
                        "end": span.get("end"),
                        "confidence": span.get("score") or span.get("confidence"),
                    }
                )
            continue

        span = item.get("span") if isinstance(item.get("span"), dict) else {}
        label = _coerce_label(item.get("label") or item.get("type") or item.get("entity"))
        entities.append(
            {
                "label": label,
                "text": item.get("text") or span.get("text"),
                "start": item.get("start") or span.get("start"),
                "end": item.get("end") or span.get("end"),
                "confidence": item.get("confidence")
                or item.get("score")
                or item.get("prob")
                or span.get("score")
                or span.get("confidence"),
            }
        )

    return entities


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


def extract_entities(
    texts: List[str],
    *,
    model_id: str,
    labels: Sequence[LabelSpec],
    threshold: float,
    include_confidence: bool = True,
) -> List[List[Dict[str, Any]]]:
    """Extract entities from each text with confidence scores."""
    model = get_cached_gliner2_model(model_id)

    label_specs = _normalize_label_specs(labels)
    label_strings = [spec.label for spec in label_specs]
    label_map = _build_label_map(label_specs)

    if hasattr(model, "batch_predict_entities"):
        raw = _call_with_optional_kwargs(
            model.batch_predict_entities,
            texts,
            labels=label_strings,
            threshold=threshold,
            include_confidence=include_confidence,
        )
    elif hasattr(model, "batch_ner"):
        raw = _call_with_optional_kwargs(
            model.batch_ner,
            texts,
            labels=label_strings,
            threshold=threshold,
            include_confidence=include_confidence,
        )
    elif hasattr(model, "predict_entities"):
        raw = [
            _call_with_optional_kwargs(
                model.predict_entities,
                text,
                labels=label_strings,
                threshold=threshold,
                include_confidence=include_confidence,
            )
            for text in texts
        ]
    else:
        task_name = "text_classification"
        tasks = {
            task_name: {
                "labels": label_strings,
                "multi_label": False,
                "cls_threshold": threshold,
            }
        }
        raw = model.batch_classify_text(
            texts,
            tasks=tasks,
            threshold=threshold,
            format_results=True,
            include_confidence=include_confidence,
            include_spans=True,
        )
        raw = [r.get(task_name) for r in raw]

    return [_normalize_entity_payload(item, label_map) for item in raw]
