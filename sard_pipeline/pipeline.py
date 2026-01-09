\
"""End-to-end pipeline that goes from base64 -> pages -> zones -> OCR -> text classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import re

from PIL import Image

from .config import PipelineConfig
from .image_loading import load_images_from_base64
from .models_yolo import classify_images, detect_zones
from .zones import extract_zones
from .ocr import ocr_zones
from .models_gliner2 import classify_texts, extract_entities
from .utils import time_call


@dataclass
class PipelineResult:
    images: List[Image.Image]
    page_classes: List[Any]
    zones: List[List[Dict[str, Any]]]
    extracted_zones: List[List[Image.Image]]
    raw_texts: List[List[str]]
    classified_texts: List[Any]
    extracted_entities: Dict[str, Any]
    logs: List[Dict[str, Any]]


def _clean_text(s: str) -> str:
    return s.strip().replace("\n", " ").replace("\r", " ")


def _zone_label_confidence(zone_result: Sequence[Dict[str, Any]], label: str) -> float:
    for item in zone_result:
        if item.get("label") == label:
            return float(item.get("confidence", 0.0))
    return 0.0


def _zone_max_confidence(zone_result: Sequence[Dict[str, Any]]) -> float:
    confidences = [float(item.get("confidence", 0.0)) for item in zone_result]
    return max(confidences, default=0.0)


def _build_label_specs(mapper: Dict[str, Any]) -> List[Dict[str, str] | str]:
    labels: List[Dict[str, str] | str] = []
    for key, spec in mapper.items():
        description = ""
        if isinstance(spec, dict):
            description = str(spec.get("description", "")).strip()
        if description:
            labels.append({key: description})
        else:
            labels.append(key)
    return labels


def _requirements_match(value: str, requirements: Optional[Dict[str, Any]]) -> bool:
    if not requirements:
        return True
    rule = requirements.get("rule")
    if rule != "regex":
        return True
    pattern = requirements.get("pattern")
    if not pattern:
        return True
    return re.fullmatch(pattern, value.strip()) is not None


def _set_nested_value(target: Dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    if not parts:
        return
    current = target
    for part in parts[:-1]:
        if part not in current or not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _extract_for_agent(
    texts: List[str],
    classified: List[Sequence[Dict[str, Any]]],
    agent: Dict[str, Any],
    *,
    model_id: str,
    threshold: float,
    device: str,
) -> Dict[str, Any]:
    mapper = agent.get("mapper", {})
    if not mapper:
        return {}

    labels = _build_label_specs(mapper)
    target_zone = bool(agent.get("target_zone", False))
    reference = str(agent.get("reference", "")).strip()
    root = str(agent.get("root", "")).strip()

    remaining: Set[str] = set(mapper.keys())
    if not remaining:
        return {}

    same_type_indices: List[Tuple[int, float]] = []
    other_indices: List[Tuple[int, float]] = []
    other_label_indices: List[Tuple[int, float]] = []
    non_other_indices: List[Tuple[int, float]] = []

    for idx, zone_result in enumerate(classified):
        zone_conf = _zone_label_confidence(zone_result, reference)
        zone_max = _zone_max_confidence(zone_result)
        if reference and zone_conf > 0.0:
            same_type_indices.append((idx, zone_conf))
        else:
            other_indices.append((idx, zone_max))

        other_conf = _zone_label_confidence(zone_result, "other")
        if other_conf > 0.0:
            other_label_indices.append((idx, other_conf))
        else:
            non_other_indices.append((idx, zone_max))

    candidate_indices: List[int] = []
    if target_zone:
        same_type_indices.sort(key=lambda x: x[1], reverse=True)
        other_indices.sort(key=lambda x: x[1], reverse=True)
        candidate_indices.extend([idx for idx, _ in same_type_indices])
        candidate_indices.extend([idx for idx, _ in other_indices if idx not in candidate_indices])
    else:
        other_label_indices.sort(key=lambda x: x[1], reverse=True)
        non_other_indices.sort(key=lambda x: x[1])
        candidate_indices.extend([idx for idx, _ in other_label_indices])
        candidate_indices.extend([idx for idx, _ in non_other_indices if idx not in candidate_indices])

    if not candidate_indices:
        return {}

    candidate_texts = [_clean_text(texts[idx]) for idx in candidate_indices]
    entity_batches = extract_entities(
        candidate_texts,
        model_id=model_id,
        labels=labels,
        threshold=threshold,
        device=device,
    )

    extracted: Dict[str, Any] = {}
    for entities in entity_batches:
        if not remaining:
            break
        for ent in entities:
            label = ent.get("label")
            if label not in remaining:
                continue
            value = ent.get("text")
            if value is None:
                continue
            value_str = str(value)
            requirements = mapper.get(label, {}).get("requirements") if isinstance(mapper.get(label), dict) else None
            if not _requirements_match(value_str, requirements):
                continue
            full_path = f"{root}.{label}" if root else label
            _set_nested_value(extracted, full_path, value_str)
            remaining.discard(label)

    return extracted


def extract_entities_for_agents(
    texts: List[str],
    classified: List[Sequence[Dict[str, Any]]],
    agents: List[Dict[str, Any]],
    *,
    model_id: str,
    threshold: float,
    device: str,
) -> Dict[str, Any]:
    """Extract entities for all agents following zone-classification rules."""
    final_mapping: Dict[str, Any] = {}
    for agent in agents:
        agent_mapping = _extract_for_agent(
            texts,
            classified,
            agent,
            model_id=model_id,
            threshold=threshold,
            device=device,
        )
        for key, value in agent_mapping.items():
            if key not in final_mapping:
                final_mapping[key] = value
            elif isinstance(final_mapping.get(key), dict) and isinstance(value, dict):
                final_mapping[key].update(value)
    return final_mapping


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
        device=config.ocr.device,
        debug=config.debug,
    )
    logs.append(log)

    flat_texts = [t for page in raw_texts for t in page]

    labels = []
    for a in config.gliner2.agents:
        if a.get("target_zone"):
            labels.append({ a.get("reference"): a.get('description') })
    labels.append({ "other": f"Other / Unrelated, not corresponding to ({', '.join([a.get('reference') for a in config.gliner2.agents if a.get('target_zone')])})" })

    classified, log = time_call(
        classify_texts,
        flat_texts,
        model_id=config.gliner2.model_id,
        labels=labels,
        multi_label=config.gliner2.multi_label,
        threshold=config.gliner2.threshold,
        include_confidence=config.gliner2.include_confidence,
        device=config.gliner2.device,
        debug=config.debug,
    )
    logs.append(log)

    extracted_entities, log = time_call(
        extract_entities_for_agents,
        flat_texts,
        classified,
        config.gliner2.agents,
        model_id=config.gliner2.model_id,
        threshold=config.gliner2.threshold,
        device=config.gliner2.device,
        debug=config.debug,
    )
    logs.append(log)

    return PipelineResult(
        images=images,
        page_classes=page_classes,
        zones=zones,
        extracted_zones=extracted,
        raw_texts=raw_texts,
        classified_texts=classified,
        extracted_entities=extracted_entities,
        logs=logs,
    )


def run_logs_only(base64_data: str, config: PipelineConfig) -> List[Dict[str, Any]]:
    """Backward-compatible helper that returns only the logs (like your old `run`)."""
    return run_pipeline(base64_data, config).logs
