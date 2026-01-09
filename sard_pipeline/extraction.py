"""Entity extraction helpers based on classified text zones."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple


ZoneClassification = Iterable[Mapping[str, Any]]


def _clean_text(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\r", " ")


def _parse_float(raw: str) -> Optional[float]:
    normalized = raw.replace(" ", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _extract_number_iter(text: str) -> Iterator[str]:
    return iter(re.findall(r"\d+(?:[.,]\d+)?", text))


def _match_regex(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    if not match:
        return None
    if match.lastindex:
        return match.group(1)
    return match.group(0)


def _set_nested_value(target: Dict[str, Any], path: str, value: Any) -> bool:
    parts = [p for p in path.split(".") if p]
    if not parts:
        return False
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    if parts[-1] in current:
        return False
    current[parts[-1]] = value
    return True


def _extract_from_text(text: str, mapper: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned = _clean_text(text)
    if not cleaned:
        return {}

    extracted: Dict[str, Any] = {}
    number_iter = _extract_number_iter(cleaned)

    for key, spec in mapper.items():
        if not isinstance(spec, Mapping):
            continue
        value: Optional[Any] = None
        requirements = spec.get("requirements")
        if isinstance(requirements, Mapping) and requirements.get("rule") == "regex":
            pattern = requirements.get("pattern")
            if pattern:
                value = _match_regex(str(pattern), cleaned)

        if value is None:
            value_type = spec.get("type")
            if value_type == "float":
                try:
                    value = next(number_iter)
                except StopIteration:
                    value = None
            else:
                value = cleaned

        if value is None:
            continue

        if spec.get("type") == "float" and not isinstance(value, float):
            parsed = _parse_float(str(value))
            if parsed is None:
                continue
            value = parsed

        extracted[str(key)] = value

    return extracted


def _label_confidence(items: ZoneClassification, label: str) -> Optional[float]:
    for item in items:
        if item.get("label") == label:
            return float(item.get("confidence", 0.0))
    return None


def _max_confidence(items: ZoneClassification) -> float:
    return max((float(item.get("confidence", 0.0)) for item in items), default=0.0)


def _select_zone_order(
    zones: List[Tuple[str, ZoneClassification]],
    *,
    label: Optional[str],
    descending: bool,
) -> List[Tuple[str, ZoneClassification, float]]:
    scored: List[Tuple[str, ZoneClassification, float]] = []
    for text, classified in zones:
        if label is None:
            score = _max_confidence(classified)
            scored.append((text, classified, score))
            continue
        score = _label_confidence(classified, label)
        if score is None:
            continue
        scored.append((text, classified, score))
    return sorted(scored, key=lambda item: item[2], reverse=descending)


def _try_extract_zones(
    zones: List[Tuple[str, ZoneClassification, float]],
    *,
    mapper: Mapping[str, Any],
    root: str,
    target: Dict[str, Any],
) -> bool:
    found_any = False
    for text, _classified, _score in zones:
        extracted = _extract_from_text(text, mapper)
        if not extracted:
            continue
        for key, value in extracted.items():
            path = f"{root}.{key}" if root else key
            if _set_nested_value(target, path, value):
                found_any = True
    return found_any


def extract_entities(
    texts: List[str],
    classified: List[ZoneClassification],
    agents: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Extract mapped entities based on classification and agent configuration."""
    zones = list(zip(texts, classified))
    output: Dict[str, Any] = {}

    for agent in agents:
        mapper = agent.get("mapper") or {}
        if not isinstance(mapper, Mapping) or not mapper:
            continue
        root = str(agent.get("root") or "")
        reference = str(agent.get("reference") or "")
        target_zone = bool(agent.get("target_zone"))

        if target_zone:
            same_type = _select_zone_order(zones, label=reference, descending=True)
            found = _try_extract_zones(same_type, mapper=mapper, root=root, target=output)
            if not found:
                other_type = _select_zone_order(zones, label=None, descending=True)
                other_type = [zone for zone in other_type if _label_confidence(zone[1], reference) is None]
                _try_extract_zones(other_type, mapper=mapper, root=root, target=output)
        else:
            other_zones = _select_zone_order(zones, label="other", descending=True)
            found = _try_extract_zones(other_zones, mapper=mapper, root=root, target=output)
            if not found:
                non_other = _select_zone_order(zones, label=None, descending=False)
                non_other = [zone for zone in non_other if _label_confidence(zone[1], "other") is None]
                _try_extract_zones(non_other, mapper=mapper, root=root, target=output)

    return output
