from __future__ import annotations

import math
from typing import Any, List, Optional


def is_valid_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value)

def extract_result_value(results: Any, objective: str) -> Optional[float]:
    if not results:
        return None
    if hasattr(results, 'get'):
        value = results.get(objective)
        if value is None and hasattr(results, 'keys'):
            keys = list(results.keys())
            value = results[keys[0]] if keys else None
    else:
        value = None

    if value is None or not is_valid_numeric(value):
        return None
    return float(value)

def extract_raw_objective_series(exp: Any, objective: str) -> List[float]:
    values: List[float] = []
    for conf in getattr(exp, 'measured_configurations', []):
        results = getattr(conf, 'averaged_result', None) or getattr(conf, 'results', None) or {}
        value = extract_result_value(results, objective)
        if value is not None:
            values.append(value)
    return values

def extract_best_so_far_series(
    exp: Any,
    objective: str,
    minimize: bool = True,
    only_enabled_improves: bool = True,
) -> List[float]:
    values: List[float] = []
    current_best: Optional[float] = None

    for conf in getattr(exp, 'measured_configurations', []):
        results = getattr(conf, 'averaged_result', None) or getattr(conf, 'results', None) or {}
        value = extract_result_value(results, objective)
        if value is None:
            continue

        if current_best is None:
            current_best = value
        else:
            can_improve = getattr(conf, 'is_enabled', True) if only_enabled_improves else True
            if can_improve:
                current_best = min(current_best, value) if minimize else max(current_best, value)
        values.append(current_best)

    return values


def extract_baseline_trajectory(
    baseline: Any,
    objective: str,
    prefer_cached: bool = True,
    best_so_far_fallback: bool = True,
    minimize: bool = True,
) -> List[float]:
    cached = getattr(baseline, 'trajectory', None)
    if prefer_cached and cached and not all(v == float('inf') for v in cached):
        return [float(v) for v in cached if v is not None and is_valid_numeric(v)]

    raw_experiment = getattr(baseline, 'raw_experiment', None)
    if raw_experiment is None:
        return []

    if best_so_far_fallback:
        return extract_best_so_far_series(raw_experiment, objective, minimize=minimize, only_enabled_improves=False)
    return extract_raw_objective_series(raw_experiment, objective)


