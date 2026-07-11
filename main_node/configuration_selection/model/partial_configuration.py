from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Set, Tuple

import pandas as pd

from core_entities.configuration import Configuration
from core_entities.search_space import Hyperparameter


@dataclass
class PartialConfiguration:
    """
    One point of a multi-point proposal, while it is still being assembled.

    The Predictor walks the tree-shaped search space level by level. Each point
    keeps its own lineage: the parameters it has decided on so far, and the
    regions it activated on the previous level and has yet to expand. This is
    what keeps a point on a single root-to-leaf branch when sibling points
    descend into other branches.
    """
    parameters: Dict[str, Any] = field(default_factory=dict)
    predicted_result: List[float] = field(default_factory=list)
    pending_regions: Set[Tuple[Hyperparameter]] = field(default_factory=set)
    type: Configuration.Type = Configuration.Type.PREDICTED

    def absorb(self, candidate: pd.Series, region: Tuple[Hyperparameter]) -> None:
        """Take one candidate row of a region: its parameters and its objective values."""
        region_hp_names = [hp.name for hp in region]
        # to_dict() unboxes the numpy scalars of the row into native Python types,
        # which the database can encode.
        for name, value in candidate.to_dict().items():
            if name in region_hp_names:
                self.parameters[name] = value
            else:
                self.predicted_result.append(value)

    def to_configuration(self, experiment_id: str, prediction_info: Mapping) -> Configuration:
        configuration = Configuration(self.parameters, self.type, experiment_id, prediction_info=prediction_info)
        configuration.predicted_result = self.predicted_result
        return configuration
