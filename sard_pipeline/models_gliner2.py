\
"""GLiNER helpers with an in-memory cache."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Sequence, Union

from .config import LabelSpec
from .utils import _supports_kwarg, require

try:
    from gliner import GLiNER  # type: ignore
except Exception:  # pragma: no cover
    GLiNER = None  # type: ignore[assignment]


_GLINER_CACHE: Dict[str, Any] = {}
_GLINER_LOCK = threading.Lock()


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


def _batch_predict_entities(
    model: Any,
    texts: List[str],
    labels: Sequence[str],
    *,
    threshold: float,
) -> List[List[Dict[str, Any]]]:
    if hasattr(model, "batch_predict_entities"):
        return model.batch_predict_entities(texts, labels, threshold=threshold)
    return [model.predict_entities(text, labels, threshold=threshold) for text in texts]


def _score_from_entity(entity: Dict[str, Any]) -> float:
    for key in ("score", "confidence", "probability"):
        value = entity.get(key)
        if value is not None:
            return float(value)
    return 0.0


def get_labels_from_agents(agents: Sequence[Dict[str, Any]]) -> List[LabelSpec]:
    """Extract GLiNER labels from agent configs."""
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
    model = get_cached_gliner_model(model_id, device=device)

    sep = " :: "
    normalized_labels = _normalize_labels(labels)
    out: List[List[Dict[str, Any]]] = []

    for page in texts:
        raw = _batch_predict_entities(
            model,
            page,
            normalized_labels,
            threshold=threshold,
        )

        flat: List[Dict[str, Any]] = []
        for i, entities in enumerate(raw):
            label_scores: Dict[str, float] = {}
            for entity in entities:
                label = entity.get("label")
                if not label:
                    continue
                score = _score_from_entity(entity)
                label_scores[label] = max(score, label_scores.get(label, 0.0))

            classes = [{"label": label, "score": score} for label, score in label_scores.items()]
            classes.sort(key=lambda item: item["score"], reverse=True)
            if not multi_label and classes:
                classes = [classes[0]]
            if not include_confidence:
                for entry in classes:
                    entry.pop("score", None)

            # Remove label descriptions (keep only the key).
            for entry in classes:
                entry["label"] = entry["label"].split(sep)[0] if sep in entry["label"] else entry["label"]

            flat.append({"text": page[i], "class": classes})

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
            
            raw = _batch_predict_entities(
                model,
                [r["text"] for r in page],
                entities,
                threshold=threshold,
            )

            for i, r in enumerate(raw):
                if page_idx >= len(out):
                    out.append([])

                grouped_entities: Dict[str, List[Dict[str, Any]]] = {}
                for entity in r:
                    label = entity.get("label")
                    if not label:
                        continue
                    grouped_entities.setdefault(label, []).append(
                        {
                            "text": entity.get("text"),
                            "start": entity.get("start"),
                            "end": entity.get("end"),
                            "score": _score_from_entity(entity),
                        }
                    )
                if not include_confidence:
                    for group in grouped_entities.values():
                        for item in group:
                            item.pop("score", None)

                entry: Dict[str, Any] = {
                    "text": page[i]["text"],
                    "agent_reference": agent.get("reference"),
                    "entities": grouped_entities,
                }

                entities = entry.get("entities")
                if entities:
                    entry["entities"] = {
                        (k.split(" :: ", 1)[0] if " :: " in k else k): v
                        for k, v in entities.items()
                    }

                out[page_idx].append(entry)
    
    return out
