import logging
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objs as go

from analyzer.config import BenchmarkConfig, NormalizationType
from analyzer.data_pipeline import (ExperimentLoader, ExperimentParser, MetricExtractor, DataProcessor,
                                    ExperimentGrouper)
from analyzer.visualization import (PlotGenerator, TableGenerator, ReportGenerator)
from analyzer.visualization.comparative_plots import ComparativePlotGenerator
from analyzer.orchestration.comparative_orchestrator import ComparativeAnalysisOrchestrator
from analyzer.orchestration.analysis_services import (
    ObjectivePartitionService,
    ComparativeTableService,
    ExportService,
)

logger = logging.getLogger(__name__)


class BenchmarkAnalyzer:
    """Main analyzer orchestrator - delegates to appropriate pipelines based on configuration"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = ExperimentLoader(config.results_folder)
        self.parser = ExperimentParser()
        self.extractor = MetricExtractor()
        self.processor = DataProcessor()
        self.plotter = PlotGenerator()
        self.table_gen = TableGenerator(self.parser, self.extractor)
        self.report_gen = ReportGenerator(config, self.table_gen)
        self.comparative_plotter = ComparativePlotGenerator()
        self.partition_service = ObjectivePartitionService(self.extractor)
        self.export_service = ExportService()

        self.comparative_orchestrator = None
        if self.config.comparative_analysis.is_active():
            logger.info("Initializing comparative analysis mode...")
            self.comparative_orchestrator = ComparativeAnalysisOrchestrator(config)

    def analyze(self, output_html: str, output_csv: str):
        """Main analysis entry point"""
        experiments = self._load_experiments()

        baselines = self.comparative_orchestrator.load_baseline_experiments() \
            if self.comparative_orchestrator else {}

        # Experiments selected as baselines must not also appear as regular experiments
        # (they come from the same results folder and would otherwise be counted twice).
        filtered_experiments = self._exclude_baselines(experiments, baselines)

        # Partition experiments per objective (result key or problem instance name)
        objective_partitions = self._partition_by_objective(filtered_experiments)
        logger.info(f"Objectives: {list(objective_partitions.keys())}")

        objective_plots = self._generate_plots(objective_partitions, baselines)
        tables_by_objective = self._build_tables(objective_partitions)

        comparative_results = {}
        comparative_plots = {}
        comparative_tables = {}
        performance_profile_plot = None

        if self.comparative_orchestrator and self.config.comparative_analysis.is_active():
            logger.info("Computing comparative metrics...")
            comparative_results = self.comparative_orchestrator.compute_comparative_metrics(
                filtered_experiments, baselines
            )
            if comparative_results:
                logger.info("Generating comparative plots...")
                comparative_plots = self._generate_comparative_plots(comparative_results)
                comparative_tables = self._build_comparative_tables(comparative_results)
                if self.config.comparative_analysis.performance_profile is not None:
                    logger.info("Computing global performance profile...")
                    performance_profile_plot = self._generate_global_performance_profile(comparative_results)

        csv_files, comparative_csv_files, zip_file = self._save_csv_files(
            tables_by_objective,
            comparative_tables,
            output_csv,
        )
        self._generate_html_report(
            objective_plots=objective_plots,
            tables_by_objective=tables_by_objective,
            csv_files=csv_files,
            comparative_csv_files=comparative_csv_files,
            zip_file=zip_file,
            output_html=output_html,
            output_csv=output_csv,
            comparative_results=comparative_results,
            comparative_plots=comparative_plots,
            comparative_tables=comparative_tables,
            performance_profile_plot=performance_profile_plot,
        )

    def _load_experiments(self) -> List[Any]:
        logger.info(f"Loading experiments from {self.config.results_folder}...")
        experiments = self.loader.load_all_experiments()
        logger.info(f"Loaded {len(experiments)} experiments")
        return experiments

    @staticmethod
    def _exclude_baselines(experiments: List[Any], baselines: Dict[str, Any]) -> List[Any]:
        """Return *experiments* with any baseline experiments removed.

        Experiments selected as baselines live in the same results folder.
        Without this filter they would appear both as regular experiments and
        as baselines, leading to double-counting in plots and tables.
        """
        baseline_names = {
            getattr(bl.raw_experiment, 'name', None) or getattr(bl.raw_experiment, 'ed_id', None)
            for bl in baselines.values()
            if getattr(bl, 'raw_experiment', None) is not None
        }
        if not baseline_names:
            return experiments
        filtered = [e for e in experiments
                    if (getattr(e, 'name', None) or getattr(e, 'ed_id', None)) not in baseline_names]
        logger.info(f"Excluded {len(experiments) - len(filtered)} baseline experiment(s)")
        return filtered

    def _partition_by_objective(self, experiments: List[Any]) -> Dict[str, List[Any]]:
        return self.partition_service.partition(self.config.objectives_to_measure, experiments)

    def _generate_plots(self, objective_partitions: Dict[str, List[Any]],
                        baselines: Dict[str, Any]) -> Dict[str, List[go.Figure]]:
        """Return ``{objective: [figures]}`` — one entry per report tab."""
        objective_plots: Dict[str, List[go.Figure]] = {}

        for objective, experiments in objective_partitions.items():
            obj_baselines = self._filter_baselines_for_objective(baselines, objective)
            result_key = self._resolve_result_key(objective, experiments)
            figures = self._make_figures(result_key, experiments, obj_baselines, objective)
            if figures:
                objective_plots[objective] = figures

        return objective_plots

    def _resolve_result_key(self, objective: str, experiments: List[Any]) -> str:
        return self.partition_service.resolve_result_key(objective, experiments)

    def _make_figures(self, result_key: str, experiments: List[Any],
                      baselines: Dict[str, Any], partition_key: str = "") -> List[go.Figure]:
        """Produce one figure per configured Plot_N block"""
        known_optimum = self.config.known_optima.get(partition_key) if partition_key else None

        figures = []
        for plot_config in self.config.plots:
            if not plot_config.should_plot_objective(result_key):
                continue

            plot_exps = ExperimentGrouper.filter(experiments, plot_config.filter_conditions)
            grouping = plot_config.plot_grouping  # per-plot CustomGrouping (may be None)
            title_suffix = self._build_plot_title_suffix(plot_config)

            if plot_config.enable_grouping:
                groups = self._build_plot_groups(plot_exps, grouping)
                logger.info(f"  [{partition_key or result_key}] plot '{title_suffix or 'default'}': "
                            f"{len(plot_exps)} experiments -> {len(groups)} groups")
                fig = self.plotter.create_grouped_plot(
                    result_key, groups, plot_config, self.extractor, baselines,
                    title_suffix=title_suffix, known_optimum=known_optimum,
                )
            else:
                fig = self._create_convergence_plot(
                    plot_exps, result_key, plot_config, baselines,
                    known_optimum=known_optimum, title_suffix=title_suffix,
                )
            if fig:
                figures.append(fig)
        return figures

    def _build_plot_groups(self, experiments: List[Any], grouping) -> Dict[str, List[Any]]:
        if grouping is not None and grouping.is_configured:
            return ExperimentGrouper(grouping).group(experiments)
        return self.loader.group_experiments(experiments)

    @staticmethod
    def _build_plot_title_suffix(plot_config) -> str:
        """Build a easy readable title suffix from the plot's title or filter conditions"""
        if plot_config.title:
            return f" – {plot_config.title}"
        if plot_config.filter_conditions:
            parts = [
                f"{c.path.split('.')[-1]}={c.value}"
                for c in plot_config.filter_conditions if c.value
            ]
            return f" [{', '.join(parts)}]" if parts else ""
        return ""


    def _filter_baselines_for_objective(self, baselines: Dict[str, Any], objective: str) -> Dict[str, Any]:
        """Filter baselines to only those relevant for the current objective"""
        filtered = {}
        for baseline_key, baseline_result in baselines.items():
            # Check if this baseline's objectives include the current one
            if hasattr(baseline_result, 'metadata') and baseline_result.metadata:
                if objective in baseline_result.metadata.objectives:
                    filtered[baseline_key] = baseline_result
            # Also include if we can't check (be permissive)
            elif baseline_result:
                filtered[baseline_key] = baseline_result
        return filtered

    def _generate_comparative_plots(self, comparative_results: Dict[str, List[Any]]) -> Dict[str, List[go.Figure]]:
        """Generate comparative analysis plots from comparison results."""
        comparative_plots = {}

        for objective, comparison_list in comparative_results.items():
            if not comparison_list:
                continue

            plots = []
            regret_type_labels = {"iteration": "Regret Analysis (Iterations)", "time": "Regret Analysis (Time)"}
            improvement_type_labels = {
                "objective_value": ("Relative Improvement", "Objective Value"),
                "time_to_target": ("Relative Improvement: Speedup Factor", "Time"),
                "iteration_to_target": ("Relative Improvement: Speedup Factor", "Iterations")
            }

            for regret_type in self.config.comparative_analysis.get_regret_types():
                fig = self.comparative_plotter.plot_regret_curves(
                    comparison_list,
                    title=f"{objective} - {regret_type_labels.get(regret_type, 'Regret Analysis')}",
                    regret_type=regret_type
                )
                if fig:
                    plots.append(fig)

            for imp_type in self.config.comparative_analysis.get_improvement_types():
                metric_type, dimension = improvement_type_labels.get(imp_type, ("Relative Improvement", imp_type))
                fig = self.comparative_plotter.plot_relative_improvement(
                    comparison_list,
                    title=f"{objective} - {metric_type} ({dimension})",
                    improvement_type=imp_type
                )
                if fig:
                    plots.append(fig)

            if plots:
                comparative_plots[objective] = plots

        return comparative_plots

    def _generate_global_performance_profile(self, comparative_results: Dict[str, List[Any]]) -> Optional[go.Figure]:
        """
        Generate global performance profile comparing test cases across all objectives.

        Each test case (test_case_0, test_case_2, test_case_9, random-search, grid-search)
        is treated as an "algorithm" and compared across multiple "problems" (objectives).

        Returns:
            Performance profile figure, or None if not enough data
        """
        from analyzer.comparison.comparative_metrics import PerformanceProfileCalculator

        test_case_performance = {}
        baseline_performance = {}

        for objective, comparison_list in comparative_results.items():
            for result in comparison_list:
                if not result.experiment_trajectory:
                    continue

                test_case_name = result.display_name or result.experiment_name
                best_exp_value = min(result.experiment_trajectory)

                test_case_performance.setdefault(test_case_name, {})
                test_case_performance[test_case_name].setdefault(objective, best_exp_value)

                if result.baseline_trajectory and result.baseline_type:
                    baseline_name = self.parser.build_display_name(result.baseline_type)
                    baseline_performance.setdefault(baseline_name, {})
                    baseline_performance[baseline_name].setdefault(objective, min(result.baseline_trajectory))

        for baseline_name, perf_data in baseline_performance.items():
            test_case_performance.setdefault(baseline_name, perf_data)

        if len(test_case_performance) < 2:
            logger.warning(f"Performance profile: Need at least 2 test cases (have {len(test_case_performance)})")
            return None

        objectives = sorted({obj for tc_data in test_case_performance.values() for obj in tc_data})

        if self.config.comparative_analysis.performance_profile and self.config.comparative_analysis.performance_profile.objectives_to_profile:
            configured = self.config.comparative_analysis.performance_profile.objectives_to_profile
            objectives = [obj for obj in objectives if obj in configured]

        if len(objectives) < 2:
            logger.warning(f"Performance profile: Need at least 2 objectives (have {len(objectives)})")
            return None

        df = pd.DataFrame(
            [{tc: test_case_performance[tc].get(obj) for tc in test_case_performance} for obj in objectives],
            index=objectives
        )

        df_complete = df.dropna(axis=1)
        excluded = [col for col in df.columns if col not in df_complete.columns]
        if excluded:
            logger.info(f"Excluded test cases with incomplete data: {excluded}")

        if len(df_complete.columns) < 2:
            logger.warning(f"Performance profile: Need at least 2 complete test cases (have {len(df_complete.columns)})")
            return None

        logger.info(f"Performance profile: {len(df_complete.columns)} test cases × {len(df_complete)} objectives")

        calculator = PerformanceProfileCalculator()
        try:
            algo_dict = {col: df_complete[col].tolist() for col in df_complete.columns}
            ratios_df = calculator.calculate_performance_ratios(algo_dict, minimize=True)

            performance_profiles = calculator.generate_performance_profile(
                ratios_df,
                tau_range=(1.0, self.config.comparative_analysis.get_tau_max()),
                tau_steps=self.config.comparative_analysis.get_tau_steps()
            )

            if not performance_profiles:
                return None

            objectives_str = ", ".join(objectives)
            return self.comparative_plotter.plot_performance_profile(
                performance_profiles,
                title=f"Performance Profile - Test Case Comparison Across {{{objectives_str}}}"
            )
        except Exception as e:
            logger.error(f"Failed to generate global performance profile: {e}", exc_info=True)
            return None

    def _create_convergence_plot(self, experiments: List[Any], objective: str, plot_config: Any,
                                baselines: Dict[str, Any] = None,
                                known_optimum: Optional[float] = None,
                                title_suffix: str = "") -> Optional[go.Figure]:
        names, data_series, time_series = self._extract_plot_data(experiments, objective, plot_config)

        if not data_series:
            return None

        normalized_baselines = baselines
        if plot_config.normalize and baselines:
            data_series, normalized_baselines = self._normalize_with_baselines(
                data_series, baselines, objective, plot_config.normalization_strategy
            )
        elif plot_config.normalize:
            data_series = self.processor.normalize_series(data_series, plot_config.normalization_strategy)

        if plot_config.plot_type == 'box_plot':
            return self.plotter.create_box_plot(objective, names, data_series, plot_config,
                                                normalized_baselines, known_optimum=known_optimum)
        elif plot_config.uses_time_metric():
            return self.plotter.create_custom_plot(objective, names, data_series, time_series, plot_config,
                                                   normalized_baselines, known_optimum=known_optimum)
        else:
            return self.plotter.create_convergence_plot(objective, names, data_series, plot_config,
                                                        normalized_baselines, known_optimum=known_optimum,
                                                        title_suffix=title_suffix)

    def _normalize_with_baselines(
        self,
        data_series: List[List[float]],
        baselines: Dict[str, Any],
        objective: str,
        normalization_strategy: str
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Normalize experiments and baselines together using the same normalization factor.

        This ensures baselines and experiments are on the same scale when plotted.
        """
        from analyzer.comparison.comparison_processor import ComparisonProcessor
        from analyzer.comparison.baseline_manager import BaselineResult

        if normalization_strategy == NormalizationType.NONE.value:
            return data_series, baselines

        # Extract baseline trajectories using the shared static method
        baseline_trajectories = [
            t for bl in baselines.values()
            for t in [ComparisonProcessor._extract_baseline_trajectory(bl, objective)]
            if t
        ]

        all_series = data_series + baseline_trajectories

        if normalization_strategy == NormalizationType.MIN_OVER_ALL.value:
            all_mins = [min((y for y in s if y is not None), default=None) for s in all_series]
            global_min = min((m for m in all_mins if m is not None), default=None)

            if global_min is None or global_min == 0:
                return data_series, baselines

            normalized_experiments = [[(y / global_min) if y is not None else None for y in s] for s in data_series]

            normalized_baselines = {}
            for baseline_key, baseline_result in baselines.items():
                traj = ComparisonProcessor._extract_baseline_trajectory(baseline_result, objective)
                if traj:
                    normalized_traj = [(y / global_min) if y is not None else None for y in traj]
                    normalized_baselines[baseline_key] = BaselineResult(
                        baseline_id=baseline_result.baseline_id,
                        baseline_type=baseline_result.baseline_type,
                        trajectory=normalized_traj,
                        best_value=min((v for v in normalized_traj if v is not None), default=float('inf')),
                        raw_experiment=baseline_result.raw_experiment
                    )
                else:
                    normalized_baselines[baseline_key] = baseline_result

            return normalized_experiments, normalized_baselines

        if normalization_strategy == NormalizationType.MAX_OVER_ALL.value:
            all_maxes = [max((y for y in s if y is not None), default=None) for s in all_series]
            global_max = max((m for m in all_maxes if m is not None), default=None)

            if global_max is None or global_max == 0:
                return data_series, baselines

            normalized_experiments = [[(y / global_max) if y is not None else None for y in s] for s in data_series]

            normalized_baselines = {}
            for baseline_key, baseline_result in baselines.items():
                traj = ComparisonProcessor._extract_baseline_trajectory(baseline_result, objective)
                if traj:
                    normalized_traj = [(y / global_max) if y is not None else None for y in traj]
                    normalized_baselines[baseline_key] = BaselineResult(
                        baseline_id=baseline_result.baseline_id,
                        baseline_type=baseline_result.baseline_type,
                        trajectory=normalized_traj,
                        best_value=max((v for v in normalized_traj if v is not None), default=float('-inf')),
                        raw_experiment=baseline_result.raw_experiment
                    )
                else:
                    normalized_baselines[baseline_key] = baseline_result

            return normalized_experiments, normalized_baselines

        return data_series, baselines


    def _extract_plot_data(self, experiments: List[Any], objective: str, plot_config: Any) -> Tuple[
        List[str], List[List[float]], List[List[Optional[float]]]]:
        names, data_series, time_series = [], [], []

        for exp in experiments:
            obj_values = self.extractor.extract_objective_series(exp, objective)
            if not obj_values:
                continue

            names.append(self.parser.build_display_name(self.parser.get_name(exp)))
            data_series.append(obj_values)

            if plot_config.uses_time_metric():
                time_series.append(self.extractor.extract_time_series(exp))

        return names, data_series, time_series

    def _build_tables(self, objective_partitions: Dict[str, List[Any]]) -> Dict[str, List[Dict[str, Any]]]:
        logger.info("Building summary tables...")
        tables_by_objective = {}
        for objective, experiments in objective_partitions.items():
            result_key = self._resolve_result_key(objective, experiments)
            rows = self.table_gen.create_table(experiments, result_key, self.config.table_config)
            if rows:
                if result_key != objective:
                    for row in rows:
                        row['Objective'] = objective
                tables_by_objective[objective] = rows
        return tables_by_objective

    def _build_comparative_tables(self, comparative_results: Dict[str, List[Any]]) -> Dict[str, List[Dict[str, Any]]]:
        table_config = self.config.comparative_analysis.comparative_table
        is_minimizing_fn = None
        if self.comparative_orchestrator:
            is_minimizing_fn = self.comparative_orchestrator.comparison_processor._is_minimizing
        return ComparativeTableService.build(comparative_results, table_config, is_minimizing_fn)

    def _save_csv_files(
        self,
        tables_by_objective: Dict[str, List[Dict[str, Any]]],
        comparative_tables: Dict[str, List[Dict[str, Any]]],
        output_csv: str,
    ) -> Tuple[Dict[str, str], Dict[str, str], Optional[str]]:
        return self.export_service.save_csv_files(tables_by_objective, output_csv, comparative_tables)

    def _generate_html_report(self, objective_plots, tables_by_objective, csv_files, comparative_csv_files, zip_file,
                              output_html, output_csv, comparative_results=None,
                              comparative_plots=None, comparative_tables=None,
                              performance_profile_plot=None):
        logger.info("Generating HTML report...")
        html_content = self.report_gen.generate(
            objective_plots, tables_by_objective, csv_files, zip_file,
            comparative_plots, comparative_tables or {}, performance_profile_plot,
            comparative_csv_files=comparative_csv_files,
        )
        html_path = Path(output_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_content, encoding='utf-8')

        logger.info(f"Report generated: {html_path}")
        logger.info(f"CSV (combined): {output_csv}")
        if comparative_results:
            num_comparisons = sum(len(v) for v in comparative_results.values())
            logger.info(f"Comparative analysis completed with {num_comparisons} comparison(s)")
        logger.info("Analysis complete!")
