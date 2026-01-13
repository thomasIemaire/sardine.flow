"""GLiNER helpers handling both Classification (GLiNER2) and Extraction (GLiNER standard)."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Sequence, Union

from .config import LabelSpec
from .utils import _supports_kwarg, require, resolve_device

# Import conditionnel pour GLiNER2 (Classification)
try:
    from gliner2 import GLiNER2  # type: ignore
except Exception:  # pragma: no cover
    GLiNER2 = None  # type: ignore[assignment]

# Import conditionnel pour GLiNER standard (Extraction)
try:
    from gliner import GLiNER  # type: ignore
except Exception:
    GLiNER = None


_GLINER_CACHE: Dict[str, Any] = {}
_GLINER_LOCK = threading.Lock()


def get_cached_gliner2_model(model_id: str, *, device: str = "cpu", debug: bool = False) -> Any:
    """Load GLiNER2 (for classification) only once."""
    require(GLiNER2, "gliner2")

    key = str(model_id).strip()
    if not key:
        raise ValueError("model_id GLiNER2 vide")

    # Resolve device securely (fallback to CPU if arch mismatch)
    safe_device = resolve_device(device, debug=debug, log_tag="gliner2")

    cache_key = f"gliner2::{key}::{safe_device}"
    if cache_key in _GLINER_CACHE:
        return _GLINER_CACHE[cache_key]

    with _GLINER_LOCK:
        if cache_key not in _GLINER_CACHE:
            kwargs: Dict[str, Any] = {}
            if safe_device and _supports_kwarg(GLiNER2.from_pretrained, "device"):
                kwargs["device"] = safe_device
            
            model = GLiNER2.from_pretrained(key, **kwargs)
            
            # Manual .to() if not handled by from_pretrained
            if safe_device and not kwargs and hasattr(model, "to"):
                model = model.to(safe_device)
                
            _GLINER_CACHE[cache_key] = model

    return _GLINER_CACHE[cache_key]


def get_cached_gliner_std_model(model_id: str, *, device: str = "cpu", debug: bool = False) -> Any:
    """Load standard GLiNER (for extraction) only once."""
    require(GLiNER, "gliner")

    key = str(model_id).strip()
    if not key:
        raise ValueError("model_id GLiNER vide")

    # Resolve device securely (fallback to CPU if arch mismatch)
    safe_device = resolve_device(device, debug=debug, log_tag="gliner_std")

    cache_key = f"gliner_std::{key}::{safe_device}"
    if cache_key in _GLINER_CACHE:
        return _GLINER_CACHE[cache_key]

    with _GLINER_LOCK:
        if cache_key not in _GLINER_CACHE:
            # GLiNER standard usage
            model = GLiNER.from_pretrained(key)
            if safe_device:
                model = model.to(safe_device)
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
    """Extract GLiNER2 labels from agent configs (used for classification)."""
    labels: List[LabelSpec] = []
    for a in agents:
        if a.get("target_zone"):
            labels.append({ a.get("reference"): a.get('description') })
    labels.append({ "other": f"Other / Unrelated. Not corresponding to {', '.join([a.get('reference') for a in agents if a.get('target_zone')])}" })
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
    debug: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Classify short texts into high-level labels using GLiNER2."""
    model = get_cached_gliner2_model(model_id, device=device, debug=debug)

    sep = " :: "
    labels_normalized = _normalize_labels(labels)

    task_name = "text_classification"

    schema = model.create_schema()
    schema.classification(task_name, labels_normalized, multi_label=multi_label, cls_threshold=threshold)
    schema.entities(labels_normalized)

    out: List[List[Dict[str, Any]]] = []

    for page in texts:
        raw = model.batch_extract(page, schema, threshold=threshold,
                         include_spans=True, include_confidence=include_confidence)

        flat = [{"text": page[i], "class": r.get(task_name), "entities": r.get("entities")} for i, r in enumerate(raw)]

        # Remove label descriptions (keep only the key).
        for r in flat:
            for e in r["class"]:
                e["label"] = e["label"].split(sep)[0] if sep in e["label"] else e["label"]
        
            entities = {k.split(sep)[0] if sep in k else k: v for k, v in r["entities"].items()}
            r["entities"] = entities

        out.append(flat)

    return out


def get_entities_labels_from_mapper(mapper: Dict[str, Any]) -> List[str]:
    """Extract simple label keys from a mapper config for GLiNER standard."""
    return [f"{k} :: {v.get('description')}" for k, v in mapper.items()]


def extract_entities(
    agents: Sequence[Dict[str, Any]],
    classified_texts: List[List[Dict[str, Any]]],
    *,
    model_id: str,
    threshold: float,
    include_confidence: bool = False,
    device: str = "cpu",
    debug: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Extract entities using standard GLiNER with batch processing."""
    # Propagate debug flag so resolving device can log warnings
    model = get_cached_gliner_std_model(model_id, device=device, debug=debug)
    out: List[List[Dict[str, Any]]] = []

    for page_idx, page in enumerate(classified_texts):
        if page_idx >= len(out):
            out.append([])

        texts_content = [t["text"] for t in page]
        if not texts_content:
            continue

        # We'll store results by text index to preserve the original (Text -> Agents) order in the output
        # page_results[i] will contain a list of agent extractions for text i
        page_results: List[List[Dict[str, Any]]] = [[] for _ in range(len(texts_content))]

        for agent in agents:
            mapper = agent.get("mapper", {})
            if not mapper:
                continue
            
            labels = get_entities_labels_from_mapper(mapper)
            if not labels:
                continue

            # Batch prediction: process ALL texts on the page for THIS agent at once
            # This is significantly faster than looping over texts.
            batch_preds = model.batch_predict_entities(texts_content, labels, threshold=threshold)

            # batch_preds matches the order of texts_content
            for i, preds in enumerate(batch_preds):
                if not preds:
                    continue
                
                text = texts_content[i]
                
                extracted_map = {}
                for p in preds:
                    label = p["label"]
                    score = p["score"]
                    extracted_text = p["text"]
                    extracted_map[label] = extracted_text
                
                if extracted_map:
                    entry: Dict[str, Any] = {
                        "text": text,
                        "agent_reference": agent.get("reference"),
                        "entities": extracted_map,
                    }
                    # Store in the bucket corresponding to the text index
                    page_results[i].append(entry)

        # Flatten results back into the page output, maintaining text order
        for text_entries in page_results:
            out[page_idx].extend(text_entries)

    return out