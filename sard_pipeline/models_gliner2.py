\
"""GLiNER2 helpers with an in-memory cache, plus GLiNER entity extraction."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Sequence, Union

from .config import LabelSpec
from .utils import _supports_kwarg, require

try:
    from gliner2 import GLiNER2  # type: ignore
except Exception:  # pragma: no cover
    GLiNER2 = None  # type: ignore[assignment]

try:
    from gliner import GLiNER  # type: ignore
except Exception:  # pragma: no cover
    GLiNER = None  # type: ignore[assignment]


_GLINER2_CACHE: Dict[str, Any] = {}
_GLINER2_LOCK = threading.Lock()
_GLINER_CACHE: Dict[str, Any] = {}
_GLINER_LOCK = threading.Lock()


def get_cached_gliner2_model(model_id: str, *, device: str = "cpu") -> Any:
    """Load GLiNER2 only once (thread-safe)."""
    require(GLiNER2, "gliner2")

    key = str(model_id).strip()
    if not key:
        raise ValueError("model_id GLiNER2 vide")

    cache_key = f"{key}::{device}"
    if cache_key in _GLINER2_CACHE:
        return _GLINER2_CACHE[cache_key]

    with _GLINER2_LOCK:
        if cache_key not in _GLINER2_CACHE:
            kwargs: Dict[str, Any] = {}
            if device and _supports_kwarg(GLiNER2.from_pretrained, "device"):
                kwargs["device"] = device
            model = GLiNER2.from_pretrained(key, **kwargs)
            if device and not kwargs and hasattr(model, "to"):
                model = model.to(device)
            _GLINER2_CACHE[cache_key] = model

    return _GLINER2_CACHE[cache_key]


def get_cached_gliner_model(model_id: str, *, device: str = "cpu") -> Any:
    """Load GLiNER only once (thread-safe)."""
    require(GLiNER, "gliner")

    key = str(model_id).strip()
    if not key:
        raise ValueError("model_id GLiNER vide")

    cache_key = f"{key}::{device}"
    if cache_key in _GLINER_CACHE:
        return _GLINER_CACHE[cache_key]

    with _GLINER_LOCK:
        if cache_key not in _GLINER_CACHE:
            kwargs: Dict[str, Any] = {}
            if device and _supports_kwarg(GLiNER.from_pretrained, "device"):
                kwargs["device"] = device
            model = GLiNER.from_pretrained(key, **kwargs)
            if device and not kwargs and hasattr(model, "to"):
                model = model.to(device)
            _GLINER_CACHE[cache_key] = model

    return _GLINER_CACHE[cache_key]


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


def get_labels_from_agents(agents: Sequence[Dict[str, Any]]) -> List[LabelSpec]:
    """Extract GLiNER2 labels from agent configs."""
    labels: List[LabelSpec] = []
    for a in agents:
        if a.get("target_zone"):
            labels.append({ a.get("reference"): a.get('description') })
    labels.append({ "other": f"Other / Unrelated, not corresponding to ({', '.join([a.get('reference') for a in agents if a.get('target_zone')])})" })
    return labels


def classify_texts(
    texts: List[List[str]],
    *,
    model_id: str,
    labels: Sequence[LabelSpec],
    multi_label: bool,
    threshold: float,
    include_confidence: bool,
    device: str = "cpu",
) -> List[List[Dict[str, Any]]]:
    """Classify short texts into high-level labels."""
    model = get_cached_gliner2_model(model_id, device=device)

    sep = " :: "
    task_name = "text_classification"
    tasks = {
        task_name: {
            "labels": _normalize_labels(labels),
            "multi_label": multi_label,
            "cls_threshold": threshold,
        }
    }

    out: List[List[Dict[str, Any]]] = []

    for page in texts:
        raw = model.batch_classify_text(
            page,
            tasks=tasks,
            threshold=threshold,
            format_results=True,
            include_confidence=include_confidence,
            include_spans=False,
        )

        flat = [{"text": page[i], "class": r.get(task_name)} for i, r in enumerate(raw)]

        # Remove label descriptions (keep only the key).
        for r in flat:
            for e in r["class"]:
                e["label"] = e["label"].split(sep)[0] if sep in e["label"] else e["label"]

        out.append(flat)

    return out


def get_entities_from_mapper(entities: Dict[str, Any], *, all_entities: List[str] = []) -> List[str]:
    """Extract entity names from a mapper config."""
    sep = " :: "
    out: List[str] = []

    for k, v in entities.items():
        description = v.get("description", "")
        if description:
            description = f"{sep}{description}"
        out.append(f"{k}{description} (Ne surtout pas confondre avec : {', '.join([e for e in all_entities if e != k])})")

    return out


def _strip_entity_label(label: str) -> str:
    label = label.split(" :: ", 1)[0]
    return label.split(" (", 1)[0]


def _group_gliner_entities(
    raw_entities: Sequence[Dict[str, Any]],
    *,
    include_confidence: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for entity in raw_entities:
        label = entity.get("label")
        if not label:
            continue
        entry: Dict[str, Any] = {
            "start": entity.get("start"),
            "end": entity.get("end"),
            "text": entity.get("text"),
        }
        score = entity.get("score")
        if include_confidence and score is not None:
            entry["score"] = score
        grouped.setdefault(_strip_entity_label(str(label)), []).append(entry)

    return grouped


def extract_entities(
    agents: Sequence[Dict[str, Any]],
    classified_texts: List[List[Dict[str, Any]]],
    *,
    model_id: str,
    threshold: float,
    include_confidence: bool,
    device: str = "cpu",
) -> List[List[Dict[str, Any]]]:
    """Extract entities from short texts."""
    model = get_cached_gliner_model(model_id, device=device)
    all_entities = [e for agent in agents for e in list(agent.get("mapper", {}).keys())]
    out = []

    for page_idx, page in enumerate(classified_texts):
        for agent in agents:
            mapper = agent.get("mapper", {})
            entities = get_entities_from_mapper(mapper, all_entities=all_entities)
            
            if not entities:
                continue
            
            raw = [
                model.predict_entities(text, entities, threshold=threshold)
                for text in [r["text"] for r in page]
            ]

            for i, r in enumerate(raw):
                if page_idx >= len(out):
                    out.append([])

                entry: Dict[str, Any] = {
                    "text": page[i]["text"],
                    "agent_reference": agent.get("reference"),
                    "entities": _group_gliner_entities(r, include_confidence=include_confidence),
                }

                out[page_idx].append(entry)
    
    return out
