from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from core.utils import resource_path


@dataclass(frozen=True)
class CalibrationConfig:
    exact_match_ad_score: float = 1.0
    exact_match_promote_to: float = 0.75
    exact_match_min_model_confidence: float = 0.55
    similar_high_tanimoto: float = 0.85
    similar_high_ad_floor: float = 0.85
    similar_high_promote_to: float = 0.75
    similar_medium_tanimoto: float = 0.70
    similar_medium_ad_floor: float = 0.65
    similar_medium_confidence_cap: float = 0.70
    borderline_cap: float = 0.60
    out_of_domain_cap: float = 0.35
    unknown_cap: float = 0.55
    in_domain_model_weight: float = 0.75
    in_domain_ad_weight: float = 0.25
    ratio_borderline: float = 1.0
    ratio_out: float = 1.25
    sim_borderline: float = 0.45
    sim_out: float = 0.30


def _float_or_default(value: Any, default: float) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    if value != value:
        return default
    return value


def _unit_interval(value: Any, default: float) -> float:
    value = _float_or_default(value, default)
    return max(0.0, min(1.0, value))


@lru_cache(maxsize=1)
def load_calibration_config() -> CalibrationConfig:
    defaults = asdict(CalibrationConfig())
    path = resource_path("config/calibration.json")
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}

    values: dict[str, float] = {}
    for key, default in defaults.items():
        if key in {"ratio_borderline", "ratio_out"}:
            values[key] = max(0.0, _float_or_default(raw.get(key), default))
        else:
            values[key] = _unit_interval(raw.get(key), default)

    if values["similar_high_tanimoto"] < values["similar_medium_tanimoto"]:
        values["similar_high_tanimoto"] = values["similar_medium_tanimoto"]
    if values["ratio_out"] < values["ratio_borderline"]:
        values["ratio_out"] = values["ratio_borderline"]
    if values["sim_borderline"] < values["sim_out"]:
        values["sim_borderline"] = values["sim_out"]

    weight_sum = values["in_domain_model_weight"] + values["in_domain_ad_weight"]
    if weight_sum <= 0:
        values["in_domain_model_weight"] = defaults["in_domain_model_weight"]
        values["in_domain_ad_weight"] = defaults["in_domain_ad_weight"]

    return CalibrationConfig(**values)
