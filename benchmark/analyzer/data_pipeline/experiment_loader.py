import os
import pickle
import re
from typing import List, Any, Dict


class ExperimentLoader:
    """Loads and groups serialized experiment files"""

    REPETITION_PATTERN = re.compile(r'_(\d+)$')

    def __init__(self, folder: str):
        self.folder = folder

    def load_all_experiments(self) -> List[Any]:
        pkl_files = self._get_pkl_files()
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {self.folder}")

        experiments = [self._load_experiment(f) for f in pkl_files]
        return self._sort_by_start_time(experiments)

    def _get_pkl_files(self) -> List[str]:
        return sorted(f for f in os.listdir(self.folder) if f.endswith('.pkl'))

    def _load_experiment(self, filename: str) -> Any:
        filepath = os.path.join(self.folder, filename)
        with open(filepath, 'rb') as f:
            exp = pickle.load(f)
        exp._source_filename = filename
        return exp

    def _sort_by_start_time(self, experiments: List[Any]) -> List[Any]:
        try:
            return sorted(experiments, key=lambda e: getattr(e, 'start_time', 0))
        except Exception:
            return experiments

    def group_experiments(self, experiments: List[Any]) -> Dict[str, List[Any]]:
        """Group experiments by base name without repetition index

        Examples:
        - exp_test_gpr_sobol_acceptableerrorbased_timebased_test_case_0
        - exp_test_gpr_sobol_acceptableerrorbased_timebased_test_case_0_2
        - exp_test_gpr_sobol_acceptableerrorbased_timebased_test_case_0_3
        All grouped under: exp_test_gpr_sobol_acceptableerrorbased_timebased_test_case_0
        """
        groups = {}
        for exp in experiments:
            name = self._get_experiment_name(exp)
            base_name = self._remove_repetition_suffix(name)

            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(exp)

        return groups

    def _get_experiment_name(self, exp: Any) -> str:
        return getattr(exp, 'name', None) or getattr(exp, 'ed_id', None) or 'experiment'

    def _remove_repetition_suffix(self, name: str) -> str:
        """Remove trailing _N where N is a number"""
        return self.REPETITION_PATTERN.sub('', name)
