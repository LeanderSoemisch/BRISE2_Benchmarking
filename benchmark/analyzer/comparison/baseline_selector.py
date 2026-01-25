import json
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SelectableExperiment:
    """Represents an experiment that can be selected as a baseline"""
    name: str
    display_name: str
    description: str
    objectives: List[str]
    iterations: int
    pkl_file: str


class BaselineSelector:
    """Manages interactive baseline selection from experiment pool"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.selection_file = self.results_dir.parent / "baseline_selection.json"

    def get_selectable_experiments(self, experiments: List[Any]) -> List[SelectableExperiment]:
        from analyzer.data_pipeline import ExperimentParser
        parser = ExperimentParser()

        selectable = []
        for exp in experiments:
            name = getattr(exp, 'name', None) or getattr(exp, 'ed_id', None) or 'experiment'
            objectives = self._get_objectives(exp)
            iterations = len(getattr(exp, 'measured_configurations', []))

            if iterations > 0 and objectives:
                display_name = parser.build_display_name(name)
                description = self._build_description(name, objectives, iterations)
                selectable.append(SelectableExperiment(
                    name=name,
                    display_name=display_name,
                    description=description,
                    objectives=objectives,
                    iterations=iterations,
                    pkl_file=getattr(exp, '_source_filename', f"{name}.pkl")
                ))

        return selectable

    def save_selection(self, selected_names: List[str]):
        with open(self.selection_file, 'w') as f:
            json.dump({'selected_baselines': selected_names, 'timestamp': datetime.now().isoformat()}, f, indent=2)
        logger.info(f"Saved baseline selection: {selected_names}")

    def load_selection(self) -> Optional[List[str]]:
        if not self.selection_file.exists():
            return None
        try:
            with open(self.selection_file, 'r') as f:
                return json.load(f).get('selected_baselines', [])
        except Exception as e:
            logger.error(f"Failed to load baseline selection: {e}")
            return None

    def clear_selection(self):
        if self.selection_file.exists():
            self.selection_file.unlink()

    def has_selection(self) -> bool:
        return self.selection_file.exists()

    @staticmethod
    def _get_objectives(exp: Any) -> List[str]:
        measured_configs = getattr(exp, 'measured_configurations', [])
        if not measured_configs:
            return []
        first_config = measured_configs[0]
        results = getattr(first_config, 'averaged_result', None) or getattr(first_config, 'results', {})
        return list(results.keys()) if hasattr(results, 'keys') else []

    @staticmethod
    def _build_description(full_name: str, objectives: List[str], iterations: int) -> str:
        parts = full_name.replace('exp_', '').split('_')
        skip = {'test', 'case', 'wo', 'dch'}
        descriptors = [
            p.upper() if len(p) <= 4 else p.capitalize()
            for p in parts if not p.isdigit() and p not in skip and len(p) > 1
        ]
        obj_str = f"{len(objectives)} objective{'s' if len(objectives) != 1 else ''} ({', '.join(objectives[:3])}{'...' if len(objectives) > 3 else ''})"
        components = [' • '.join(descriptors[:5]), obj_str, f"{iterations} iterations"]
        return ' | '.join(components)
