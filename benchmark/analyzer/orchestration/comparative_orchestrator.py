import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from analyzer.data_pipeline import ExperimentParser
from analyzer.data_pipeline.experiment_metadata import ExperimentMetadata

from analyzer.config import BenchmarkConfig
from analyzer.comparison import BaselineManager, ComparisonProcessor

logger = logging.getLogger(__name__)


class ComparativeAnalysisOrchestrator:
    """Orchestrates comparative benchmarking analysis"""

    def __init__(self, config: BenchmarkConfig, benchmark_id: Optional[str] = None):
        self.config = config
        self.benchmark_id = benchmark_id
        self.results_dir = Path(config.results_folder)
        self.parser = ExperimentParser()
        self.baseline_manager = BaselineManager(self.results_dir, benchmark_id)
        self.comparison_processor = ComparisonProcessor(config)

    def load_baseline_experiments(self) -> Dict[str, Any]:
        baselines = {}
        for baseline_type in self.baseline_manager.get_available_baseline_types():
            baseline = self.baseline_manager.load_baseline(baseline_type)
            if baseline:
                baselines[baseline_type] = baseline
                logger.info(f"Loaded baseline: {baseline_type}")
        return baselines

    def compute_comparative_metrics(self, experiments: List[Any], baselines: Dict[str, Any]) -> Dict[str, List[Any]]:
        if not baselines:
            return {}

        from analyzer.data_pipeline import MetricExtractor
        extractor = MetricExtractor()

        objectives = extractor.discover_objectives(experiments)
        if self.config.objectives_to_measure:
            objectives = [obj for obj in self.config.objectives_to_measure if obj in objectives]
        else:
            objectives = list(objectives)

        comparisons = {}

        for objective in objectives:
            objective_comparisons = []

            for exp in experiments:
                try:
                    exp_name = getattr(exp, 'name', None) or getattr(exp, 'ed_id', 'unknown')
                    exp_trajectory = extractor.extract_objective_series(exp, objective)

                    if not exp_trajectory:
                        continue

                    experiment_data = {
                        'name': exp_name,
                        'display_name': self.parser.build_display_name(exp_name),
                        'trajectory': {objective: exp_trajectory},
                        'objective_values': {objective: exp_trajectory},
                        'raw_experiment': exp,
                        'runtime': extractor.extract_runtime(exp)
                    }

                    matching_baselines = self._select_matching_baselines(exp, exp_name, baselines)

                    for baseline_key, baseline in matching_baselines:
                        try:
                            known_optimum = None
                            regret_config = self.config.comparative_analysis.regret_analysis
                            if regret_config:
                                if regret_config.optimum_per_objective and objective in regret_config.optimum_per_objective:
                                    known_optimum = regret_config.optimum_per_objective[objective]
                                elif regret_config.known_optimum is not None:
                                    known_optimum = regret_config.known_optimum

                            result = self.comparison_processor.process_experiment_comparison(
                                experiment_data=experiment_data,
                                baseline=baseline,
                                objective=objective,
                                known_optimum=known_optimum
                            )
                            objective_comparisons.append(result)
                        except Exception as e:
                            logger.error(f"Comparison failed for {exp_name} vs {baseline_key}: {e}")
                except Exception as e:
                    logger.error(f"Failed to process experiment: {e}")

            if objective_comparisons:
                comparisons[objective] = objective_comparisons

        return comparisons

    def _select_matching_baselines(self, exp: Any, exp_name: str, baselines: Dict[str, Any]) -> List[Any]:
        exp_meta = ExperimentMetadata.extract(exp)
        exp_task = self._normalize_key(exp_meta.get("task_name", ""))
        exp_display = self._normalize_key(self.parser.build_display_name(exp_name))
        exp_identifier = self._normalize_key(exp_name)

        matched = []
        for baseline_key, baseline in baselines.items():
            baseline_exp = getattr(baseline, 'raw_experiment', None)
            baseline_meta = ExperimentMetadata.extract(baseline_exp) if baseline_exp is not None else {}
            baseline_task = self._normalize_key(baseline_meta.get("task_name", ""))

            baseline_name = self._baseline_name(baseline_key, baseline)
            baseline_identifier = self._normalize_key(baseline_name)
            baseline_display = self._normalize_key(self.parser.build_display_name(baseline_name))

            if exp_task and baseline_task and exp_task == baseline_task:
                matched.append((baseline_key, baseline))
                continue

            if exp_identifier and baseline_identifier and exp_identifier == baseline_identifier:
                matched.append((baseline_key, baseline))
                continue

            if exp_display and baseline_display and exp_display == baseline_display:
                matched.append((baseline_key, baseline))

        if matched:
            return matched

        logger.warning(
            "No baseline matched experiment '%s' by task/id/display; using all selected baselines",
            exp_name,
        )
        return list(baselines.items())

    @staticmethod
    def _baseline_name(baseline_key: str, baseline: Any) -> str:
        raw_experiment = getattr(baseline, 'raw_experiment', None)
        if raw_experiment is None:
            return baseline_key
        return getattr(raw_experiment, 'name', None) or getattr(raw_experiment, 'ed_id', None) or baseline_key

    @staticmethod
    def _normalize_key(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        if not text:
            return ""
        return re.sub(r'[^a-z0-9]+', '_', text).strip('_')

