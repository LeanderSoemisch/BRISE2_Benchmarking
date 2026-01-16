import os
import zipfile
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objs as go

from analyzer.config import BenchmarkConfig
from analyzer.data_pipeline import (ExperimentLoader, ExperimentParser, MetricExtractor, DataProcessor)
from analyzer.visualization import (PlotGenerator, TableGenerator, ReportGenerator)


class BenchmarkAnalyzer:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = ExperimentLoader(config.results_folder)
        self.parser = ExperimentParser()
        self.extractor = MetricExtractor()
        self.processor = DataProcessor()
        self.plotter = PlotGenerator()
        self.table_generator = TableGenerator(self.parser, self.extractor)
        self.report_gen = ReportGenerator(config, self.table_generator)

    def analyze(self, output_html: str, output_csv: str):
        experiments = self._load_experiments()
        objectives = self._determine_objectives(experiments)
        experiment_groups = self.loader.group_experiments(experiments)

        print(f"Grouped {len(experiments)} experiments into {len(experiment_groups)} groups")

        objective_plots = self._generate_plots(experiments, experiment_groups, objectives)
        tables_by_objective = self._build_tables(experiments, objectives)
        csv_files, zip_file = self._save_csv_files(tables_by_objective, output_csv)

        self._generate_html_report(objective_plots, tables_by_objective, csv_files, zip_file, output_html, output_csv)

    def _load_experiments(self) -> List[Any]:
        print(f"Loading experiments from {self.config.results_folder}...")
        experiments = self.loader.load_all_experiments()
        print(f"Loaded {len(experiments)} experiments")
        return experiments

    def _determine_objectives(self, experiments: List[Any]) -> List[str]:
        """Discover and filter objectives based on configuration"""
        all_objectives = self.extractor.discover_objectives(experiments)
        print(f"Discovered objectives: {all_objectives}")

        if self.config.objectives_to_measure:
            objectives = [obj for obj in self.config.objectives_to_measure if obj in all_objectives]
            if not objectives:
                print(f"Warning: None of configured objectives found. Using all discovered.")
                objectives = sorted(list(all_objectives))
        else:
            objectives = sorted(list(all_objectives))

        print(f"Analyzing objectives: {objectives}")
        return objectives

    def _generate_plots(self, experiments: List[Any], experiment_groups: Dict[str, List[Any]], objectives: List[str]) -> \
            Dict[str, List[go.Figure]]:
        objective_plots = {}

        for objective in objectives:
            figures = []
            for plot_config in self.config.plots:
                if not plot_config.should_plot_objective(objective):
                    continue

                if plot_config.enable_grouping:
                    fig = self._create_grouped_plot(experiment_groups, objective, plot_config)
                else:
                    fig = self._create_improvement_plot(experiments, objective, plot_config)

                if fig:
                    figures.append(fig)

            if figures:
                objective_plots[objective] = figures

        return objective_plots

    def _create_grouped_plot(self, experiment_groups: Dict[str, List[Any]], objective: str, plot_config: Any) -> \
            Optional[go.Figure]:
        return self.plotter.create_grouped_plot(objective, experiment_groups, plot_config, self.extractor)

    def _create_improvement_plot(self, experiments: List[Any], objective: str, plot_config: Any) -> Optional[go.Figure]:
        names, data_series, time_series = self._extract_plot_data(experiments, objective, plot_config)

        if not data_series:
            return None

        if plot_config.normalize:
            data_series = self.processor.normalize_series(data_series, plot_config.normalization_strategy)

        if plot_config.uses_time_metric():
            return self.plotter.create_custom_plot(objective, names, data_series, time_series, plot_config)
        else:
            return self.plotter.create_improvement_plot(objective, names, data_series, plot_config)

    def _extract_plot_data(self, experiments: List[Any], objective: str, plot_config: Any) -> Tuple[
        List[str], List[List[float]], List[List[Optional[float]]]]:
        names = []
        data_series = []
        time_series = []

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
        print("Building summary tables...")
        tables_by_objective = {}

        for objective in objectives:
            rows = self.table_generator.create_table(experiments, objective, self.config.table_config)
            if rows:
                tables_by_objective[objective] = rows

        return tables_by_objective

    def _save_csv_files(self, tables_by_objective: Dict[str, List[Dict[str, Any]]], output_csv: str) -> Tuple[
        Dict[str, str], Optional[str]]:
        print("Saving CSV files...")
        output_dir = os.path.dirname(output_csv) or '.'
        os.makedirs(output_dir, exist_ok=True)

        self._save_combined_csv(tables_by_objective, output_csv)
        csv_files = self._save_per_objective_csvs(tables_by_objective, output_dir)
        zip_file = self._create_zip_archive(output_csv, csv_files, output_dir)

        return csv_files, zip_file

    def _save_combined_csv(self, tables_by_objective: Dict[str, List[Dict[str, Any]]], output_csv: str):
        all_rows = []
        for rows in tables_by_objective.values():
            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        self._round_numeric_columns(df)
        df.to_csv(output_csv, index=False)

    def _save_per_objective_csvs(self, tables_by_objective: Dict[str, List[Dict[str, Any]]], output_dir: str) -> Dict[
        str, str]:
        csv_files = {}
        for objective, rows in tables_by_objective.items():
            df = pd.DataFrame(rows)
            self._round_numeric_columns(df)

            filename = f"benchmark_objective_{objective}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            csv_files[objective] = filename

        return csv_files

    @staticmethod
    def _round_numeric_columns(df: pd.DataFrame):
        numeric_cols = ['Initial', 'Final best', 'Absolute improvement', 'Improvement %']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(6)

    def _create_zip_archive(self, output_csv: str, csv_files: Dict[str, str], output_dir: str) -> Optional[str]:
        print("Creating ZIP archive...")
        zip_filename = os.path.join(output_dir, "benchmark_all_tables.zip")

        try:
            with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(output_csv, arcname=os.path.basename(output_csv))
                for objective, filename in csv_files.items():
                    zf.write(os.path.join(output_dir, filename), arcname=filename)
            return os.path.basename(zip_filename)
        except Exception as e:
            print(f"Warning: ZIP creation failed: {e}")
            return None

    def _generate_html_report(self, objective_plots: Dict[str, List[go.Figure]],
                              tables_by_objective: Dict[str, List[Dict[str, Any]]], csv_files: Dict[str, str],
                              zip_file: Optional[str], output_html: str, output_csv: str):
        print("Generating HTML report...")
        html_content = self.report_gen.generate(objective_plots, tables_by_objective, csv_files, zip_file)

        html_path = Path(output_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_content, encoding='utf-8')

        print(f"\n✓ Report generated: {html_path}")
        print(f"✓ CSV (combined): {output_csv}")
        print(f"✓ Analysis complete!")
