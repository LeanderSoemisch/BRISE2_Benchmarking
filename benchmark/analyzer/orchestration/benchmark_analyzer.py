import os
import zipfile
import logging
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objs as go

from analyzer.config import BenchmarkConfig, NormalizationType
from analyzer.data_pipeline import (ExperimentLoader, ExperimentParser, MetricExtractor, DataProcessor)
from analyzer.visualization import (PlotGenerator, TableGenerator, ReportGenerator)
from analyzer.visualization.comparative_plots import ComparativePlotGenerator
from analyzer.orchestration.comparative_orchestrator import ComparativeAnalysisOrchestrator

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

        self.comparative_orchestrator = None
        if self.config.comparative_metrics.is_active():
            logger.info("Initializing comparative analysis mode...")
            self.comparative_orchestrator = ComparativeAnalysisOrchestrator(config)

    def analyze(self, output_html: str, output_csv: str):
        """Main analysis entry point"""
        experiments = self._load_experiments()
        objectives = self._determine_objectives(experiments)

        baselines = {}
        baseline_names = set()
        if self.comparative_orchestrator:
            baselines = self.comparative_orchestrator.load_baseline_experiments()
            for baseline_result in baselines.values():
                if hasattr(baseline_result, 'raw_experiment'):
                    exp = baseline_result.raw_experiment
                    name = getattr(exp, 'name', None) or getattr(exp, 'ed_id', None)
                    if name:
                        baseline_names.add(name)

        filtered_experiments = [
            exp for exp in experiments
            if (getattr(exp, 'name', None) or getattr(exp, 'ed_id', None)) not in baseline_names
        ]

        if baseline_names:
            logger.info(f"Excluded {len(experiments) - len(filtered_experiments)} baseline experiment(s) from analysis")

        experiment_groups = self.loader.group_experiments(filtered_experiments)
        logger.info(f"Grouped {len(filtered_experiments)} experiments into {len(experiment_groups)} groups")

        objective_plots = self._generate_plots(filtered_experiments, experiment_groups, objectives, baselines)
        tables_by_objective = self._build_tables(filtered_experiments, objectives)

        comparative_results = {}
        comparative_plots = {}
        performance_profile_plot = None

        if self.comparative_orchestrator and self.config.comparative_metrics.is_active():
            logger.info("Computing comparative metrics...")
            comparative_results = self.comparative_orchestrator.compute_comparative_metrics(
                filtered_experiments, baselines
            )

            if comparative_results:
                logger.info("Generating comparative plots...")
                comparative_plots = self._generate_comparative_plots(comparative_results)

                if self.config.comparative_metrics.performance_profile is not None:
                    logger.info("Computing global performance profile...")
                    performance_profile_plot = self._generate_global_performance_profile(comparative_results)

        csv_files, zip_file = self._save_csv_files(tables_by_objective, output_csv)
        self._generate_html_report(
            objective_plots, tables_by_objective, csv_files, zip_file,
            output_html, output_csv, comparative_results, comparative_plots,
            performance_profile_plot
        )

    def _load_experiments(self) -> List[Any]:
        logger.info(f"Loading experiments from {self.config.results_folder}...")
        experiments = self.loader.load_all_experiments()
        logger.info(f"Loaded {len(experiments)} experiments")
        return experiments

    def _determine_objectives(self, experiments: List[Any]) -> List[str]:
        """Discover and filter objectives based on configuration"""
        all_objectives = self.extractor.discover_objectives(experiments)
        logger.info(f"Discovered objectives: {all_objectives}")

        if self.config.objectives_to_measure:
            objectives = [obj for obj in self.config.objectives_to_measure if obj in all_objectives]
            if not objectives:
                logger.warning("None of configured objectives found. Using all discovered.")
                objectives = sorted(list(all_objectives))
        else:
            objectives = sorted(list(all_objectives))

        logger.info(f"Analyzing objectives: {objectives}")
        return objectives

    def _generate_plots(self, experiments: List[Any], experiment_groups: Dict[str, List[Any]],
                        objectives: List[str], baselines: Dict[str, Any] = None) -> Dict[str, List[go.Figure]]:
        objective_plots = {}
        if baselines is None:
            baselines = {}

        for objective in objectives:
            figures = []
            objective_baselines = self._filter_baselines_for_objective(baselines, objective)

            for plot_config in self.config.plots:
                if not plot_config.should_plot_objective(objective):
                    continue

                if plot_config.enable_grouping:
                    fig = self.plotter.create_grouped_plot(objective, experiment_groups, plot_config, self.extractor, objective_baselines)
                else:
                    fig = self._create_improvement_plot(experiments, objective, plot_config, objective_baselines)

                if fig:
                    figures.append(fig)

            if figures:
                objective_plots[objective] = figures

        return objective_plots

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
                "objective_value": ("Normalized Improvement", "Objective Value"),
                "time_to_target": ("Normalized Improvement: Speedup Factor", "Time"),
                "iteration_to_target": ("Normalized Improvement: Speedup Factor", "Iterations")
            }

            for regret_type in self.config.comparative_metrics.get_regret_types():
                fig = self.comparative_plotter.plot_regret_curves(
                    comparison_list,
                    title=f"{objective} - {regret_type_labels.get(regret_type, 'Regret Analysis')}",
                    regret_type=regret_type
                )
                if fig:
                    plots.append(fig)

            for imp_type in self.config.comparative_metrics.get_improvement_types():
                metric_type, dimension = improvement_type_labels.get(imp_type, ("Normalized Improvement", imp_type))
                fig = self.comparative_plotter.plot_normalized_improvement(
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
                    baseline_name = result.baseline_type.replace('_', '-')
                    baseline_performance.setdefault(baseline_name, {})
                    baseline_performance[baseline_name].setdefault(objective, min(result.baseline_trajectory))

        for baseline_name, perf_data in baseline_performance.items():
            test_case_performance.setdefault(baseline_name, perf_data)

        if len(test_case_performance) < 2:
            logger.warning(f"Performance profile: Need at least 2 test cases (have {len(test_case_performance)})")
            return None

        objectives = sorted({obj for tc_data in test_case_performance.values() for obj in tc_data})

        if self.config.comparative_metrics.performance_profile and self.config.comparative_metrics.performance_profile.objectives_to_profile:
            configured = self.config.comparative_metrics.performance_profile.objectives_to_profile
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
                tau_range=(1.0, self.config.comparative_metrics.get_tau_max()),
                tau_steps=self.config.comparative_metrics.get_tau_steps()
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

    def _create_improvement_plot(self, experiments: List[Any], objective: str, plot_config: Any,
                                baselines: Dict[str, Any] = None) -> Optional[go.Figure]:
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
            return self.plotter.create_box_plot(objective, names, data_series, plot_config, normalized_baselines)
        elif plot_config.uses_time_metric():
            return self.plotter.create_custom_plot(objective, names, data_series, time_series, plot_config, normalized_baselines)
        else:
            return self.plotter.create_improvement_plot(objective, names, data_series, plot_config, normalized_baselines)

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

    def _build_tables(self, experiments: List[Any], objectives: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        logger.info("Building summary tables...")
        tables_by_objective = {}
        for objective in objectives:
            rows = self.table_gen.create_table(experiments, objective, self.config.table_config)
            if rows:
                tables_by_objective[objective] = rows
        return tables_by_objective

    def _build_comparative_tables(self, comparative_results: Dict[str, List[Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build summary tables for comparative metrics.

        Args:
            comparative_results: Dictionary mapping objectives to lists of ComparisonResult objects

        Returns:
            Dictionary mapping objectives to comparative metrics table rows
        """
        comparative_tables = {}
        table_config = self.config.comparative_metrics.comparative_table

        for objective, comparison_list in comparative_results.items():
            if not comparison_list:
                continue

            rows = []
            for result in comparison_list:
                row = {}

                # Add columns based on configuration
                if not table_config or table_config.experiment:
                    row['Experiment'] = result.display_name or result.experiment_name

                if not table_config or table_config.baseline:
                    row['Baseline'] = result.baseline_type

                if (not table_config or table_config.final_regret) and result.final_regret is not None:
                    row['Final Regret'] = f"{result.final_regret:.6f}"

                if not table_config or table_config.normalized_improvement:
                    if result.normalized_improvement is not None:
                        row['NI (Objective)'] = f"{result.normalized_improvement:.4f}"
                    if result.normalized_improvement_time is not None:
                        row['NI (Time)'] = f"{result.normalized_improvement_time:.4f}"
                    if result.normalized_improvement_iterations is not None:
                        row['NI (Iterations)'] = f"{result.normalized_improvement_iterations:.4f}"

                if result.converged_at_iteration is not None and (not table_config or table_config.converged_at_iteration):
                    row['Converged at Iter'] = result.converged_at_iteration

                minimize = (self.comparative_orchestrator.comparison_processor._is_minimizing(objective)
                            if self.comparative_orchestrator else True)
                if result.experiment_trajectory and (not table_config or table_config.experiment_best):
                    # Get best value based on optimization direction
                    exp_best = min(result.experiment_trajectory) if minimize else max(result.experiment_trajectory)
                    row['Experiment Best'] = f"{exp_best:.6f}"
                if result.baseline_trajectory and (not table_config or table_config.baseline_best):
                    # Get best value based on optimization direction
                    base_best = min(result.baseline_trajectory) if minimize else max(result.baseline_trajectory)
                    row['Baseline Best'] = f"{base_best:.6f}"

                rows.append(row)

            if rows:
                comparative_tables[objective] = rows

        return comparative_tables

    def _save_csv_files(self, tables_by_objective: Dict[str, List[Dict[str, Any]]], output_csv: str) -> Tuple[
        Dict[str, str], Optional[str]]:
        logger.info("Saving CSV files...")
        output_dir = os.path.dirname(output_csv) or '.'
        os.makedirs(output_dir, exist_ok=True)

        self._save_combined_csv(tables_by_objective, output_csv)
        csv_files = self._save_per_objective_csvs(tables_by_objective, output_dir)
        zip_file = self._create_zip_archive(output_csv, csv_files, output_dir)

        return csv_files, zip_file

    def _save_combined_csv(self, tables_by_objective: Dict[str, List[Dict[str, Any]]], output_csv: str):
        all_rows = [row for rows in tables_by_objective.values() for row in rows]
        df = pd.DataFrame(all_rows)
        self._round_numeric_columns(df)
        df.to_csv(output_csv, index=False)

    def _save_per_objective_csvs(self, tables_by_objective: Dict[str, List[Dict[str, Any]]], output_dir: str) -> Dict[str, str]:
        csv_files = {}
        for objective, rows in tables_by_objective.items():
            df = pd.DataFrame(rows)
            self._round_numeric_columns(df)
            filename = f"benchmark_objective_{objective}.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False)
            csv_files[objective] = filename
        return csv_files

    @staticmethod
    def _round_numeric_columns(df: pd.DataFrame):
        for col in ['Initial', 'Final best', 'Absolute improvement', 'Improvement %']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(6)


    def _create_zip_archive(self, output_csv: str, csv_files: Dict[str, str], output_dir: str) -> Optional[str]:
        logger.info("Creating ZIP archive...")
        zip_filename = os.path.join(output_dir, "benchmark_all_tables.zip")

        try:
            with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(output_csv, arcname=os.path.basename(output_csv))
                for objective, filename in csv_files.items():
                    zf.write(os.path.join(output_dir, filename), arcname=filename)
            return os.path.basename(zip_filename)
        except Exception as e:
            logger.warning(f"ZIP creation failed: {e}")
            return None

    def _generate_html_report(self, objective_plots, tables_by_objective, csv_files, zip_file,
                              output_html, output_csv, comparative_results=None,
                              comparative_plots=None, performance_profile_plot=None):
        logger.info("Generating HTML report...")

        comparative_tables = self._build_comparative_tables(comparative_results) if comparative_results else {}

        html_content = self.report_gen.generate(
            objective_plots, tables_by_objective, csv_files, zip_file,
            comparative_plots, comparative_tables, performance_profile_plot
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
