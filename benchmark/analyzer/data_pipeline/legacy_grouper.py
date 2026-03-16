from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from analyzer.config.benchmark_config import CustomGroupingConfig, MatchCondition
from analyzer.data_pipeline.experiment_metadata import ExperimentMetadata

_DUPLICATE_SUFFIX_RE = re.compile(r'(\(\d+\))+$')


class ExperimentGrouper:
    """Domain-agnostic grouper for any BRISE serialised Experiment"""

    def __init__(self, config: CustomGroupingConfig):
        self.config = config

    def group(self, experiments: List[Any]) -> Dict[str, List[Any]]:
        """Return ``{group_name: [exp, ...]}`` for all experiments."""
        if self.config.auto_group_by:
            return self._group_by_auto(experiments)
        return self._group_by_name(experiments)

    @staticmethod
    def filter(experiments: List[Any], conditions: List[MatchCondition]) -> List[Any]:
        """Return only experiments whose metadata satisfies ALL given conditions"""
        if not conditions:
            return experiments
        return [
            exp for exp in experiments
            if all(
                ExperimentGrouper._condition_matches(ExperimentMetadata.extract(exp), cond)
                for cond in conditions
            )
        ]

    @staticmethod
    def _condition_matches(meta: Dict[str, Any], cond: MatchCondition) -> bool:
        extracted = ExperimentMetadata.get(meta, cond.path)

        if isinstance(extracted, list):
            extracted_str = ",".join(str(v) for v in extracted)
        else:
            extracted_str = str(extracted) if extracted is not None else ""

        if cond.pattern is not None:
            return bool(re.search(cond.pattern, extracted_str))

        if cond.value is None:
            return bool(extracted_str)

        if cond.contains:
            return cond.value in extracted_str

        return extracted_str == cond.value

    def _group_by_auto(self, experiments: List[Any]) -> Dict[str, List[Any]]:
        groups: Dict[str, List[Any]] = {}
        for exp in experiments:
            meta = ExperimentMetadata.extract(exp)
            parts = []
            for dim in self.config.auto_group_by:
                val = ExperimentMetadata.get(meta, dim.path)
                if val is None:
                    val = "?"
                elif isinstance(val, list):
                    val = "+".join(str(v).split(".")[-1] for v in val)
                else:
                    val = str(val)

                if dim.transform == "basename":
                    val = os.path.basename(val)

                prefix = dim.label or dim.path.split(".")[-1]
                parts.append(f"{prefix}={val}")

            label = " | ".join(parts) if parts else "group"
            groups.setdefault(label, []).append(exp)
        return groups

    @staticmethod
    def _group_by_name(experiments: List[Any]) -> Dict[str, List[Any]]:
        groups: Dict[str, List[Any]] = {}
        for exp in experiments:
            fname = getattr(exp, "_source_filename", "") or ""
            base = fname[:-4] if fname.endswith(".pkl") else fname
            base = _DUPLICATE_SUFFIX_RE.sub("", base)
            groups.setdefault(base, []).append(exp)
        return groups

LegacyExperimentGrouper = ExperimentGrouper
