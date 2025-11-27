import sys
import json
import math
import os
import pickle
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
import webbrowser
import numpy as np

from utils.template_loader import TemplateLoader


# Ensure pickled modules like 'core_entities' are importable when unpickling
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_NODE_PATH = str(PROJECT_ROOT / 'main_node')
if MAIN_NODE_PATH not in sys.path:
    sys.path.insert(0, MAIN_NODE_PATH)



@dataclass
class PlotConfig:
    """Configuration for a single plot from v3 template."""
    plot_type: str  # 'improvement_plot'
    metric_description: str
    metric_label: str
    metric_scale: str  # 'linear' | 'log10'
    metric_type: str  # 'iteration' | 'time'
    objectives_to_plot: List[str]
    normalize: bool
    normalization_strategy: str  # 'min_over_all_experiments' | 'max_over_all_experiments'
    objective_label: str
    objective_scale: str  # 'linear' | 'log10'


@dataclass
class TableConfig:
    """Configuration for table columns from v3 template."""
    task: bool = True
    model: bool = True
    sampler: bool = True
    configuration_strategy: bool = True
    stop_condition: bool = True
    experiment: bool = True
    objective: bool = True
    iterations: bool = True
    initial_value: bool = True
    final_best_value: bool = True
    improvement_percentage: bool = True
    improvement_absolute: bool = True
    runtime: bool = True


@dataclass
class BenchmarkConfig:
    """Main configuration for benchmark analysis - V3 template only."""
    results_folder: str
    output_directory: str
    experiment_name: str
    experiment_description: str
    objectives_to_measure: List[str]
    plots: List[PlotConfig]
    table_config: TableConfig

    @staticmethod
    def from_json(cfg: Dict[str, Any]) -> "BenchmarkConfig":
        """Parse V3 benchmark template JSON."""
        benchmark = cfg.get("Benchmark", {})

        # Resources folder (default: ./results/serialized/)
        folder = benchmark.get("Resources", {}).get("Folder", "./results/serialized/")

        # Output directory (default: ./results/reports/)
        output_dir = benchmark.get("Report", {}).get("outputDirectory", "./results/reports/")

        # Experiment metadata
        experiment = benchmark.get("Experiment", {})
        exp_name = experiment.get("name", "BRISE Benchmark Report")
        exp_description = experiment.get("description", "Benchmark analysis results")
        objectives = experiment.get("objectivesToMeasure", [])

        # Table configuration
        table_dict = benchmark.get("Table", {})
        table_config = TableConfig(
            task=table_dict.get("task", True),
            model=table_dict.get("model", True),
            sampler=table_dict.get("sampler", True),
            configuration_strategy=table_dict.get("configurationStrategy", True),
            stop_condition=table_dict.get("stopCondition", True),
            experiment=table_dict.get("experiment", True),
            objective=table_dict.get("objective", True),
            iterations=table_dict.get("iterations", True),
            initial_value=table_dict.get("initialValue", True),
            final_best_value=table_dict.get("finalBestValue", True),
            improvement_percentage=table_dict.get("improvementPercentage", True),
            improvement_absolute=table_dict.get("improvementAbsolute", True),
            runtime=table_dict.get("runtime", True)
        )

        # Plot configurations (Plot_0, Plot_1, etc.)
        plots = []
        plot_keys = sorted([k for k in benchmark.keys() if k.startswith("Plot_")])

        for plot_key in plot_keys:
            plot_data = benchmark[plot_key]
            plot_type_data = plot_data.get("PlotType", {})

            if "ImprovementPlot" in plot_type_data:
                impr = plot_type_data["ImprovementPlot"]
                metric_axis = impr.get("MetricAxis", {})
                objective_axis = impr.get("ObjectiveAxis", {})

                # Determine metric type from description
                metric_desc = metric_axis.get("metricDescription", "iterations completed").lower()
                metric_type = "time" if "time" in metric_desc else "iteration"

                # Extract normalization strategy
                norm_strategy_data = objective_axis.get("NormalizationStrategy", {})
                if "MinOverAll" in norm_strategy_data:
                    norm_strategy = "min_over_all_experiments"
                elif "MaxOverAll" in norm_strategy_data:
                    norm_strategy = "max_over_all_experiments"
                else:
                    norm_strategy = "none"

                plot = PlotConfig(
                    plot_type="improvement_plot",
                    metric_description=metric_axis.get("metricDescription", "iterations completed"),
                    metric_label=metric_axis.get("label", "iteration"),
                    metric_scale=metric_axis.get("scale", "linear"),
                    metric_type=metric_type,
                    objectives_to_plot=objective_axis.get("objectivesToPlot", []),
                    normalize=objective_axis.get("normalize", True),
                    normalization_strategy=norm_strategy,
                    objective_label=objective_axis.get("label", "Objective value"),
                    objective_scale=objective_axis.get("scale", "linear")
                )
                plots.append(plot)

        return BenchmarkConfig(
            results_folder=folder,
            output_directory=output_dir,
            experiment_name=exp_name,
            experiment_description=exp_description,
            objectives_to_measure=objectives,
            plots=plots,
            table_config=table_config
        )


class ExperimentLoader:
    """Loads serialized experiment files."""

    def __init__(self, folder: str):
        self.folder = folder

    def load_all(self) -> List[Any]:
        """Load all .pkl experiment files from folder."""
        files = [f for f in os.listdir(self.folder) if f.endswith('.pkl')]
        if not files:
            raise FileNotFoundError(f"No .pkl files found in {self.folder}")

        experiments = []
        for filename in sorted(files):
            filepath = os.path.join(self.folder, filename)
            with open(filepath, 'rb') as f:
                exp = pickle.load(f)
                experiments.append(exp)

        # Sort by start time if available
        try:
            experiments.sort(key=lambda e: getattr(e, 'start_time', 0))
        except Exception:
            pass

        return experiments


class ExperimentParser:
    """Parses experiment naming and extracts features."""

    @staticmethod
    def get_name(exp) -> str:
        """Get experiment name or ID."""
        return getattr(exp, 'name', None) or getattr(exp, 'ed_id', None) or 'experiment'

    @staticmethod
    def parse_features(raw_name: str) -> Dict[str, Any]:
        """Parse features from experiment naming pattern.

        Expected pattern: exp_<task>_<model>_<sampler>_<configStrategy>_<stopCondition>[_<idx>]
        Example: exp_test_GPR_sobol_quantitybased_timebased_1
        """
        features = {
            'Task': None,
            'Model': None,
            'Sampler': None,
            'ConfigurationStrategy': None,
            'StopCondition': None,
            'Index': None
        }

        try:
            pattern = r'^exp_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+?)(?:_(\d+))?$'
            match = re.match(pattern, raw_name)
            if match:
                features['Task'] = match.group(1)
                features['Model'] = match.group(2)
                features['Sampler'] = match.group(3)
                features['ConfigurationStrategy'] = match.group(4)
                features['StopCondition'] = match.group(5)
                if match.group(6):
                    features['Index'] = int(match.group(6))
        except Exception:
            pass

        return features

    @staticmethod
    def build_display_name(raw_name: str) -> str:
        """Build display name from raw experiment name.

        Since file names follow clean naming convention, use them directly.
        """
        return raw_name


class MetricExtractor:
    """Extracts metrics and objectives from experiments."""

    @staticmethod
    def discover_objectives(experiments: List[Any]) -> Set[str]:
        """Discover all objective keys from experiments."""
        objectives = set()

        for exp in experiments:
            # Sample first few configurations to find objectives
            for conf in getattr(exp, 'measured_configurations', [])[:3]:
                results = getattr(conf, 'results', {})
                for key, value in results.items():
                    if isinstance(value, (int, float)) and not math.isnan(value):
                        objectives.add(key)

        return objectives

    @staticmethod
    def extract_objective_series(exp, objective: str) -> List[float]:
        """Extract objective value series from experiment."""
        values = []

        for conf in getattr(exp, 'measured_configurations', []):
            results = getattr(conf, 'results', {})
            if objective in results:
                val = results[objective]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    values.append(float(val))

        return values

    @staticmethod
    def extract_time_series(exp) -> List[Optional[float]]:
        """Extract time series from experiment.

        Uses iteration_timestamp if available (time from start to each iteration).
        """
        times = []
        start_time = getattr(exp, 'start_time', None)

        for conf in getattr(exp, 'measured_configurations', []):
            # Use iteration_timestamp if available
            if hasattr(conf, 'iteration_timestamp') and start_time:
                try:
                    delta = conf.iteration_timestamp - start_time
                    time_val = delta.total_seconds()
                    times.append(time_val)
                    continue
                except Exception:
                    pass

            # If no timestamp, append None
            times.append(None)

        return times

    @staticmethod
    def extract_runtime(exp) -> Optional[float]:
        """Extract total runtime from experiment."""
        if hasattr(exp, 'start_time') and hasattr(exp, 'end_time'):
            try:
                delta = exp.end_time - exp.start_time
                return delta.total_seconds()
            except Exception:
                pass
        return None


class DataProcessor:
    """Processes and normalizes data series."""

    @staticmethod
    def normalize_series(series_list: List[List[float]], method: str) -> List[List[float]]:
        """Normalize series based on strategy."""
        if method == 'none':
            return series_list

        if method == 'min_over_all_experiments':
            # Find global minimum across all series
            mins = [min([y for y in s if y is not None], default=None) for s in series_list]
            global_min = min([m for m in mins if m is not None], default=None)

            if global_min is None or global_min == 0:
                return series_list

            # Normalize by global minimum
            normalized = []
            for series in series_list:
                normalized.append([(y / global_min) if y is not None else None for y in series])
            return normalized

        return series_list

    @staticmethod
    def compute_best_so_far(values: List[Optional[float]], direction: str = 'minimize') -> List[Optional[float]]:
        """Compute best-so-far series."""
        best_series = []
        current_best = None

        for val in values:
            if val is None:
                best_series.append(current_best)
                continue

            if current_best is None:
                current_best = val
            else:
                if direction == 'minimize':
                    current_best = min(current_best, val)
                else:
                    current_best = max(current_best, val)

            best_series.append(current_best)

        return best_series


class PlotGenerator:
    """Generates Plotly figures for benchmark results."""

    @staticmethod
    def _compute_robust_y_range(values: List[float]) -> Optional[List[float]]:
        """Compute robust y-axis range with padding."""
        finite_vals = np.array([v for v in values if v is not None and np.isfinite(v)])

        if finite_vals.size == 0:
            return None

        min_y = float(np.min(finite_vals))
        max_y = float(np.max(finite_vals))

        if not np.isfinite(min_y) or not np.isfinite(max_y):
            return None

        if max_y == min_y:
            pad = 0.05 * (abs(min_y) if min_y != 0 else 1.0)
            return [min_y - pad, max_y + pad]

        pad = 0.05 * (max_y - min_y)
        return [min_y - pad, max_y + pad]

    def create_improvement_plot(
        self,
        objective: str,
        experiment_names: List[str],
        data_series: List[List[float]],
        plot_config: PlotConfig
    ) -> go.Figure:
        """Create improvement plot with iterations on x-axis."""
        traces = []
        max_iterations = 0
        all_values = []

        for series, name in zip(data_series, experiment_names):
            x_vals = list(range(len(series)))
            max_iterations = max(max_iterations, len(series))

            # Compute best-so-far
            best_series = DataProcessor.compute_best_so_far(series, direction='minimize')
            all_values.extend([v for v in best_series if v is not None])

            # Determine plot mode
            mode = 'markers' if len(series) <= 1 else 'lines+markers'

            trace = go.Scatter(
                x=x_vals,
                y=best_series,
                mode=mode,
                name=f"{name} best"
            )

            if len(series) == 1:
                trace.update(marker=dict(size=12))
            traces.append(trace)

        # Build layout
        x_range = [-0.5, max_iterations - 0.5 if max_iterations > 1 else 0.5]
        y_range = self._compute_robust_y_range(all_values)

        title = f'{objective} - {plot_config.metric_description}'
        layout = dict(
            title=title,
            xaxis=dict(title=plot_config.metric_label, range=x_range),
            yaxis=dict(title=plot_config.objective_label)
        )

        if y_range:
            layout['yaxis']['range'] = y_range

        # Apply axis scale
        if plot_config.objective_scale == 'log10':
            layout['yaxis']['type'] = 'log'

        return go.Figure(data=traces, layout=layout)

    def create_time_based_plot(
        self,
        objective: str,
        experiment_names: List[str],
        objective_series: List[List[float]],
        time_series: List[List[Optional[float]]],
        plot_config: PlotConfig
    ) -> Optional[go.Figure]:
        # Create improvement plot with time on x-axis
        traces = []
        all_time = []
        all_objective = []

        for obj_vals, time_vals, name in zip(objective_series, time_series, experiment_names):
            # Compute best-so-far for objectives
            best_series = DataProcessor.compute_best_so_far(obj_vals, direction='minimize')

            # Pair with time values, filtering out None entries
            valid_pairs = []
            for i in range(min(len(best_series), len(time_vals))):
                if best_series[i] is not None and time_vals[i] is not None:
                    valid_pairs.append((time_vals[i], best_series[i]))

            if not valid_pairs:
                continue

            x_vals, y_vals = zip(*valid_pairs)
            all_time.extend(x_vals)
            all_objective.extend(y_vals)

            mode = 'markers' if len(valid_pairs) <= 1 else 'lines+markers'
            trace = go.Scatter(
                x=x_vals,
                y=y_vals,
                mode=mode,
                name=f"{name} best"
            )

            if len(valid_pairs) == 1:
                trace.update(marker=dict(size=12))
            traces.append(trace)

        if not traces:
            return None

        # Build layout
        title = f'{objective} - {plot_config.metric_description}'
        layout = dict(
            title=title,
            xaxis=dict(title=plot_config.metric_label),
            yaxis=dict(title=plot_config.objective_label)
        )

        # Apply axis scales
        if plot_config.metric_scale == 'log10':
            layout['xaxis']['type'] = 'log'

        if plot_config.objective_scale == 'log10':
            layout['yaxis']['type'] = 'log'

        # Add robust y-range
        y_range = self._compute_robust_y_range(all_objective)
        if y_range:
            layout['yaxis']['range'] = y_range

        return go.Figure(data=traces, layout=layout)


class TableBuilder:
    """Builds data tables for the report."""

    @staticmethod
    def build_summary_table(
        experiments: List[Any],
        objective: str,
        table_config: TableConfig,
        parser: ExperimentParser,
        extractor: MetricExtractor
    ) -> List[Dict[str, Any]]:
        """Build summary table for a specific objective."""
        rows = []

        for exp in experiments:
            # Extract objective series
            values = extractor.extract_objective_series(exp, objective)
            if not values:
                continue

            # Compute metrics
            initial = values[0]
            best_series = DataProcessor.compute_best_so_far(values, direction='minimize')
            final_best = best_series[-1] if best_series else None

            # Compute improvement
            if initial is not None and final_best is not None:
                improvement_abs = initial - final_best
                improvement_pct = (improvement_abs / initial * 100) if initial != 0 else None
            else:
                improvement_abs = None
                improvement_pct = None

            # Parse experiment features
            exp_name = parser.get_name(exp)
            features = parser.parse_features(exp_name)

            # Extract runtime
            runtime = extractor.extract_runtime(exp)

            # Build row
            row = {
                'Task': features.get('Task'),
                'Model': features.get('Model'),
                'Sampler': features.get('Sampler'),
                'ConfigurationStrategy': features.get('ConfigurationStrategy'),
                'StopCondition': features.get('StopCondition'),
                'Experiment': parser.build_display_name(exp_name),
                'Objective': objective,
                'Iterations': len(values),
                'Initial': initial,
                'Final best': final_best,
                'Absolute improvement': improvement_abs,
                'Improvement %': round(improvement_pct, 2) if improvement_pct is not None else None,
                'Runtime (s)': runtime
            }

            rows.append(row)

        return rows

    @staticmethod
    def format_table_html(rows: List[Dict[str, Any]], table_config: TableConfig) -> str:
        """Format table rows as HTML with column filtering."""
        if not rows:
            return "<p>No data available</p>"

        # Column mapping
        column_map = {
            'task': 'Task',
            'model': 'Model',
            'sampler': 'Sampler',
            'configuration_strategy': 'ConfigurationStrategy',
            'stop_condition': 'StopCondition',
            'experiment': 'Experiment',
            'objective': 'Objective',
            'iterations': 'Iterations',
            'initial_value': 'Initial',
            'final_best_value': 'Final best',
            'improvement_percentage': 'Improvement %',
            'improvement_absolute': 'Absolute improvement',
            'runtime': 'Runtime (s)'
        }

        # Filter columns based on config
        allowed_columns = set()
        for config_field, column_name in column_map.items():
            if getattr(table_config, config_field, True):
                allowed_columns.add(column_name)

        # Preferred column order
        preferred_order = [
            'Task', 'Model', 'Sampler', 'ConfigurationStrategy', 'StopCondition',
            'Experiment', 'Objective', 'Iterations', 'Initial', 'Final best',
            'Absolute improvement', 'Improvement %', 'Runtime (s)'
        ]

        # Get all columns from rows
        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())

        # Order headers
        headers = [h for h in preferred_order if h in all_columns and h in allowed_columns]
        headers += [c for c in sorted(all_columns) if c not in headers and c in allowed_columns]

        # Format value
        def format_value(val):
            if val is None:
                return ''
            if isinstance(val, float):
                if math.isnan(val):
                    return ''
                return f"{val:.4g}"
            return str(val)

        # Build HTML table
        html_parts = ["<table class='summary-table'>"]

        # Header row
        header_html = '<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>'
        html_parts.append(header_html)

        # Data rows
        for row in rows:
            cells = []
            for header in headers:
                value = format_value(row.get(header))
                css_class = 'exp-name' if header == 'Experiment' else 'num'
                cells.append(f"<td class='{css_class}'>{value}</td>")

            row_html = '<tr>' + ''.join(cells) + '</tr>'
            html_parts.append(row_html)

        html_parts.append("</table>")
        return ''.join(html_parts)


class ReportGenerator:
    # Generates the final HTML report

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.template_loader = TemplateLoader()

    def generate(
        self,
        objective_plots: Dict[str, List[go.Figure]],
        table_data: Dict[str, List[Dict[str, Any]]],
        csv_files: Dict[str, str],
        zip_file: Optional[str]
    ) -> str:
        """ Generate complete HTML report.

        Args:
            objective_plots: Dict mapping objective -> list of figures (Plot_0, Plot_1, etc.)
            table_data: Dict mapping objective -> table rows
            csv_files: Dict mapping objective -> CSV filename
            zip_file: Optional zip file containing all CSVs

        Returns:
            Complete HTML report as string
        """
        tab_buttons = []
        tab_contents = []

        for idx, (objective, figures) in enumerate(objective_plots.items()):
            # Tab button
            active_class = 'active' if idx == 0 else ''
            tab_id = f'obj_tab_{idx}'
            tab_buttons.append(
                f"<button class='tab-btn {active_class}' onclick=showTab('{tab_id}',this)>{objective}</button>"
            )

            # Tab content
            plots_html_parts = []
            for plot_idx, fig in enumerate(figures):
                plot_id = f"plot_{tab_id}_{plot_idx}"
                plot_type = "improvement" if plot_idx == 0 else f"plot_{plot_idx}"

                plots_html_parts.append(
                    f"<div class='plot-container'>"
                    f"<div class='plot-wrapper' id='{plot_id}'>"
                    f"{fig.to_html(include_plotlyjs=False, full_html=False)}"
                    f"</div>"
                    f"<button class='export-btn' onclick='exportPlotAsSVG(\"{plot_id}\", \"{objective}_{plot_type}\")'>Export as SVG</button>"
                    f"</div>"
                )

            plots_html = ''.join(plots_html_parts)

            # Table HTML
            table_rows = table_data.get(objective, [])
            table_html = TableBuilder.format_table_html(table_rows, self.config.table_config)

            # CSV download link
            csv_filename = csv_files.get(objective, '')
            download_link = (
                f"<div class='download-row'>"
                f"<a class='download-link' href='{csv_filename}' download>Download {objective} CSV</a>"
                f"</div>"
            ) if csv_filename else ''

            # Combine tab content
            display_style = 'block' if idx == 0 else 'none'
            tab_content = (
                f"<div id='{tab_id}' class='tab-content' style='display:{display_style}'>"
                f"{plots_html}"
                f"{download_link}"
                f"<div class='table-wrapper'>{table_html}</div>"
                f"</div>"
            )
            tab_contents.append(tab_content)

        # Global download link
        global_download = ''
        if zip_file:
            global_download = (
                f"<div class='global-download'>"
                f"<a class='download-link all' href='{zip_file}' download>Download all tables (.zip)</a>"
                f"</div>"
            )

        # Build tabs section
        tabs_section = (
            f"<div class='tabs'>"
            f"{global_download}"
            f"<div class='tab-buttons'>{''.join(tab_buttons)}</div>"
            f"{''.join(tab_contents)}"
            f"</div>"
        )

        # Render template
        context = {
            'generated_time': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'experiment_name': self.config.experiment_name,
            'experiment_description': self.config.experiment_description,
            'objectives': ', '.join(objective_plots.keys()),
            'tabs_section': tabs_section,
            'auto_status': 'disabled' if self._is_headless() else 'enabled'
        }

        html = self.template_loader.render_template(
            'report_template.html',
            context=context,
            inline_assets=True,
            css_files=['report.css'],
            js_files=['report.js']
        )

        return html

    @staticmethod
    def _is_headless() -> bool:
        """Check if running in headless/container environment."""
        if os.environ.get('DISPLAY') is None and sys.platform.startswith('linux'):
            return True
        if os.path.exists('/.dockerenv'):
            return True
        return False


class BenchmarkAnalyzer:
    """Main benchmark analyzer orchestrating all components."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.loader = ExperimentLoader(config.results_folder)
        self.parser = ExperimentParser()
        self.extractor = MetricExtractor()
        self.processor = DataProcessor()
        self.plotter = PlotGenerator()
        self.report_gen = ReportGenerator(config)

    def run(self, output_html: str, output_csv: str):
        """Run complete benchmark analysis pipeline."""
        print(f"Loading experiments from {self.config.results_folder}...")
        experiments = self.loader.load_all()
        print(f"Loaded {len(experiments)} experiments")

        # Discover objectives
        all_objectives = self.extractor.discover_objectives(experiments)
        print(f"Discovered objectives: {all_objectives}")

        # Filter objectives based on configuration
        if self.config.objectives_to_measure:
            objectives = [obj for obj in self.config.objectives_to_measure if obj in all_objectives]
            if not objectives:
                print(f"Warning: None of configured objectives found in data. Using all discovered.")
                objectives = sorted(list(all_objectives))
        else:
            objectives = sorted(list(all_objectives))

        print(f"Analyzing objectives: {objectives}")

        # Generate plots for each objective
        objective_plots: Dict[str, List[go.Figure]] = {}

        for objective in objectives:
            figures = []

            for plot_config in self.config.plots:
                # Check if this objective should be plotted
                if plot_config.objectives_to_plot and objective not in plot_config.objectives_to_plot:
                    continue

                # Extract data for all experiments
                names = []
                data_series = []
                time_series = []

                for exp in experiments:
                    obj_values = self.extractor.extract_objective_series(exp, objective)
                    if not obj_values:
                        continue

                    names.append(self.parser.build_display_name(self.parser.get_name(exp)))
                    data_series.append(obj_values)

                    if plot_config.metric_type == 'time':
                        time_series.append(self.extractor.extract_time_series(exp))

                if not data_series:
                    continue

                # Apply normalization
                if plot_config.normalize:
                    data_series = self.processor.normalize_series(
                        data_series,
                        plot_config.normalization_strategy
                    )

                # Generate appropriate plot
                if plot_config.metric_type == 'iteration':
                    fig = self.plotter.create_improvement_plot(
                        objective, names, data_series, plot_config
                    )
                elif plot_config.metric_type == 'time':
                    fig = self.plotter.create_time_based_plot(
                        objective, names, data_series, time_series, plot_config
                    )
                else:
                    # Default to iteration-based
                    fig = self.plotter.create_improvement_plot(
                        objective, names, data_series, plot_config
                    )

                if fig:
                    figures.append(fig)

            if figures:
                objective_plots[objective] = figures

        # Build tables
        print("Building summary tables...")
        tables_by_objective: Dict[str, List[Dict[str, Any]]] = {}
        all_rows = []

        for objective in objectives:
            rows = TableBuilder.build_summary_table(
                experiments, objective, self.config.table_config,
                self.parser, self.extractor
            )
            if rows:
                tables_by_objective[objective] = rows
                all_rows.extend(rows)

        # Save CSVs
        print("Saving CSV files...")
        output_dir = os.path.dirname(output_csv) or '.'
        os.makedirs(output_dir, exist_ok=True)

        # Combined CSV
        combined_df = pd.DataFrame(all_rows)
        for col in ['Initial', 'Final best', 'Absolute improvement', 'Improvement %']:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').round(6)
        combined_df.to_csv(output_csv, index=False)

        # Per-objective CSVs
        csv_files = {}
        for objective, rows in tables_by_objective.items():
            df = pd.DataFrame(rows)
            for col in ['Initial', 'Final best', 'Absolute improvement', 'Improvement %']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').round(6)

            filename = f"benchmark_objective_{objective}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            csv_files[objective] = filename

        # Create zip archive
        print("Creating ZIP archive...")
        zip_filename = os.path.join(output_dir, "benchmark_all_tables.zip")
        try:
            with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(output_csv, arcname=os.path.basename(output_csv))
                for objective, filename in csv_files.items():
                    zf.write(os.path.join(output_dir, filename), arcname=filename)
            zip_file = os.path.basename(zip_filename)
        except Exception as e:
            print(f"Warning: ZIP creation failed: {e}")
            zip_file = None

        # Generate HTML report
        print("Generating HTML report...")
        html_content = self.report_gen.generate(
            objective_plots,
            tables_by_objective,
            csv_files,
            zip_file
        )

        # Write HTML file
        html_path = Path(output_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_content, encoding='utf-8')

        # Auto-open in browser if not headless
        if not ReportGenerator._is_headless() and html_path.exists():
            try:
                uri = html_path.resolve().as_uri()
                if sys.platform.startswith('linux') and shutil.which('xdg-open'):
                    import subprocess
                    subprocess.Popen(
                        ['xdg-open', uri],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    webbrowser.open(uri)
            except Exception as e:
                print(f'Warning: Auto-open failed: {e}')

        print(f"\n✓ Report generated: {html_path}")
        print(f"✓ CSV (combined): {output_csv}")
        print(f"✓ Analysis complete!")


# Main Entry Point

def main(
    template_json_path: str = './configs/benchmark_templates/benchmark_template_v3.json',
    output_html: str = './results/reports/benchmark_report.html',
    output_csv: str = './results/reports/benchmark_all_objectives.csv'
):
    """ Main entry point for benchmark analysis

    Args:
        template_json_path: Path to V3 benchmark template JSON
        output_html: Path for output HTML report
        output_csv: Path for combined CSV output
    """
    # Load configuration
    with open(template_json_path, 'r') as f:
        config = BenchmarkConfig.from_json(json.load(f))

    # Run analysis
    analyzer = BenchmarkAnalyzer(config)
    analyzer.run(output_html, output_csv)


if __name__ == '__main__':
    main()
