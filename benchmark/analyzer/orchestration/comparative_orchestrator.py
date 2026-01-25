import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from analyzer.data_pipeline import ExperimentParser

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

                    task_name = getattr(exp, 'description', {}).get("TaskConfiguration", {}).get("TaskName", "unknown")
                    matching_baselines = [(k, v) for k, v in baselines.items() if task_name in k or k.startswith(task_name)]
                    if not matching_baselines:
                        matching_baselines = list(baselines.items())

                    for baseline_key, baseline in matching_baselines:
                        try:
                            known_optimum = None
                            regret_config = self.config.comparative_metrics.regret_analysis
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

