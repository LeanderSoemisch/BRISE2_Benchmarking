from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import plotly.graph_objs as go

from analyzer.config import Constants, ScaleType
from analyzer.config.benchmark_config import PlotConfig
from analyzer.data_pipeline import DataProcessor


class PlotGenerator:
    """Generates Plotly figures for benchmark results"""

    @staticmethod
    def _compute_robust_y_range(values: List[float]) -> Optional[List[float]]:
        """Compute robust y-axis range with 5% padding"""
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

    @staticmethod
    def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
        """Convert hex color to rgba with specified alpha"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def _create_scatter_trace(x_vals: List[Any], y_vals: List[Any], name: str,
            single_point: bool = False) -> go.Scatter:
        """Create a scatter trace with appropriate mode"""
        mode = 'markers' if single_point else 'lines+markers'
        trace = go.Scatter(x=x_vals, y=y_vals, mode=mode, name=name)
        if single_point:
            trace.update(marker=dict(size=12))
        return trace

    @staticmethod
    def _apply_axis_config(layout: Dict[str, Any], plot_config: PlotConfig):
        """Apply axis scale configuration to layout"""
        if plot_config.metric_scale == ScaleType.LOG10.value:
            layout['xaxis']['type'] = 'log'
        if plot_config.objective_scale == ScaleType.LOG10.value:
            layout['yaxis']['type'] = 'log'

    def create_improvement_plot(self, objective: str, experiment_names: List[str], data_series: List[List[float]],
            plot_config: PlotConfig) -> go.Figure:
        traces = []
        all_values = []
        max_iterations = 0

        for series, name in zip(data_series, experiment_names):
            x_vals = list(range(len(series)))
            max_iterations = max(max_iterations, len(series))

            best_series = DataProcessor.compute_best_so_far(series)
            all_values.extend([v for v in best_series if v is not None])

            trace = self._create_scatter_trace(x_vals, best_series, f"{name} best", len(series) <= 1)
            traces.append(trace)

        x_range = [-0.5, max_iterations - 0.5 if max_iterations > 1 else 0.5]
        y_range = self._compute_robust_y_range(all_values)

        layout = dict(title=f'{objective} - {plot_config.metric_description}',
            xaxis=dict(title=plot_config.metric_label, range=x_range), yaxis=dict(title=plot_config.objective_label))

        if y_range:
            layout['yaxis']['range'] = y_range

        self._apply_axis_config(layout, plot_config)
        return go.Figure(data=traces, layout=layout)

    def create_custom_plot(self, objective: str, experiment_names: List[str], objective_series: List[List[float]],
            time_series: List[List[Optional[float]]], plot_config: PlotConfig) -> Optional[go.Figure]:
        traces = []
        all_time = []
        all_objective = []

        for obj_vals, time_vals, name in zip(objective_series, time_series, experiment_names):
            best_series = DataProcessor.compute_best_so_far(obj_vals)

            valid_pairs = [(time_vals[i], best_series[i]) for i in range(min(len(best_series), len(time_vals))) if
                           best_series[i] is not None and time_vals[i] is not None]

            if not valid_pairs:
                continue

            x_vals, y_vals = zip(*valid_pairs)
            all_time.extend(x_vals)
            all_objective.extend(y_vals)

            trace = self._create_scatter_trace(x_vals, y_vals, f"{name} best", len(valid_pairs) <= 1)
            traces.append(trace)

        if not traces:
            return None

        layout = dict(title=f'{objective} - {plot_config.metric_description}',
            xaxis=dict(title=plot_config.metric_label), yaxis=dict(title=plot_config.objective_label))

        self._apply_axis_config(layout, plot_config)

        y_range = self._compute_robust_y_range(all_objective)
        if y_range:
            layout['yaxis']['range'] = y_range

        return go.Figure(data=traces, layout=layout)

    def create_grouped_plot(self, objective: str, experiment_groups: Dict[str, List[Any]], plot_config: PlotConfig,
            extractor: Any) -> Optional[go.Figure]:
        """Create grouped plot showing min/max bands and mean for test case repetitions"""
        traces = []
        all_values = []

        for group_idx, (group_name, exp_list) in enumerate(experiment_groups.items()):
            color = Constants.DEFAULT_COLORS[group_idx % len(Constants.DEFAULT_COLORS)]
            fill_color = self._hex_to_rgba(color, alpha=0.2)

            grouped_data = extractor.extract_grouped_data(exp_list, objective, plot_config.metric_type)

            if not grouped_data:
                continue

            plot_data = self._prepare_grouped_plot_data(grouped_data)
            if not plot_data:
                continue

            x_vals, min_y, max_y, mean_y = plot_data
            all_values.extend([v for v in min_y if v is not None])
            all_values.extend([v for v in max_y if v is not None])

            group_traces = self._create_grouped_traces(x_vals, min_y, max_y, mean_y, group_name, color, fill_color)
            traces.extend(group_traces)

        if not traces:
            return None

        layout = self._create_grouped_layout(objective, plot_config, all_values)
        return go.Figure(data=traces, layout=layout)

    @staticmethod
    def _prepare_grouped_plot_data(grouped_data: Dict[str, Any]) -> Optional[Tuple[List, List, List, List]]:
        """Prepare and filter grouped data for plotting"""
        metric_vals = grouped_data['metric_values']
        min_vals = grouped_data['min_values']
        max_vals = grouped_data['max_values']
        mean_vals = grouped_data['mean_values']

        valid_indices = [i for i in range(len(metric_vals)) if
            metric_vals[i] is not None and min_vals[i] is not None and max_vals[i] is not None]

        if not valid_indices:
            return None

        x_vals = [metric_vals[i] for i in valid_indices]
        min_y = [min_vals[i] for i in valid_indices]
        max_y = [max_vals[i] for i in valid_indices]
        mean_y = [mean_vals[i] for i in valid_indices]

        min_y_best = DataProcessor.compute_best_so_far(min_y)
        max_y_best = DataProcessor.compute_best_so_far(max_y)
        mean_y_best = DataProcessor.compute_best_so_far(mean_y)

        return x_vals, min_y_best, max_y_best, mean_y_best

    @staticmethod
    def _create_grouped_traces(x_vals: List, min_y: List, max_y: List, mean_y: List, group_name: str, color: str,
            fill_color: str) -> List[go.Scatter]:
        """Create traces for min-max band and mean line in grouped plots"""
        traces = []

        traces.append(
            go.Scatter(x=x_vals, y=max_y, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip',
                legendgroup=group_name))

        traces.append(
            go.Scatter(x=x_vals, y=min_y, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=fill_color,
                name=f'{group_name} (min-max)', hovertemplate='%{x}, %{y:.4f}<extra></extra>', legendgroup=group_name))

        traces.append(
            go.Scatter(x=x_vals, y=mean_y, mode='lines+markers', name=f'{group_name} (mean)', line=dict(color=color),
                marker=dict(color=color), hovertemplate='%{x}, %{y:.4f}<extra></extra>', legendgroup=group_name))

        return traces

    def _create_grouped_layout(self, objective: str, plot_config: PlotConfig, all_values: List[float]) -> Dict[
        str, Any]:
        """Create layout for grouped plot"""
        layout = dict(title=f'{objective} - Grouped Test Cases ({plot_config.metric_description})',
            xaxis=dict(title=plot_config.metric_label), yaxis=dict(title=plot_config.objective_label))

        self._apply_axis_config(layout, plot_config)

        y_range = self._compute_robust_y_range(all_values)
        if y_range:
            layout['yaxis']['range'] = y_range

        return layout
