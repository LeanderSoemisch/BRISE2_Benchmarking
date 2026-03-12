from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import plotly.graph_objs as go

from analyzer.config import Constants, ScaleType
from analyzer.config.benchmark_config import PlotConfig
from analyzer.data_pipeline import ExperimentParser


class PlotGenerator:
    """Generates Plotly figures for benchmark results"""

    def __init__(self):
        self.parser = ExperimentParser()

    @staticmethod
    def _compute_robust_y_range(values: List[float]) -> Optional[List[float]]:
        finite_vals = np.array([v for v in values if v is not None and np.isfinite(v)])
        if finite_vals.size == 0:
            return None
        min_y, max_y = float(np.min(finite_vals)), float(np.max(finite_vals))
        if not np.isfinite(min_y) or not np.isfinite(max_y):
            return None
        pad = 0.05 * (max_y - min_y) if max_y != min_y else 0.05 * (abs(min_y) if min_y != 0 else 1.0)
        return [min_y - pad, max_y + pad]


    @staticmethod
    def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def _create_scatter_trace(x_vals: List[Any], y_vals: List[Any], name: str, single_point: bool = False) -> go.Scatter:
        mode = 'markers' if single_point else 'lines+markers'
        trace = go.Scatter(x=x_vals, y=y_vals, mode=mode, name=name)
        if single_point:
            trace.update(marker=dict(size=12))
        return trace

    @staticmethod
    def _apply_axis_config(layout: Dict[str, Any], plot_config: PlotConfig):
        if plot_config.metric_scale == ScaleType.LOG10.value:
            layout['xaxis']['type'] = 'log'
        if plot_config.objective_scale == ScaleType.LOG10.value:
            layout['yaxis']['type'] = 'log'

    @staticmethod
    def _optimum_trace(known_optimum: float) -> go.Scatter:
        """Invisible scatter used to add a green dashed optimum line to the legend."""
        return go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='green', dash='dash', width=1.5),
            name=f'Optimum ({known_optimum:g})',
        )

    @staticmethod
    def _optimum_shape(known_optimum: float) -> Dict[str, Any]:
        """Plotly shape dict for a horizontal dashed green reference line."""
        return dict(
            type='line', xref='paper', x0=0, x1=1,
            yref='y', y0=known_optimum, y1=known_optimum,
            line=dict(color='green', dash='dash', width=1.5),
        )

    @staticmethod
    def _add_optimum_to_layout(layout: Dict[str, Any], known_optimum: float):
        layout.setdefault('shapes', [])
        layout['shapes'].append(PlotGenerator._optimum_shape(known_optimum))

    def _create_baseline_traces(self, baselines: Dict[str, Any], objective: str) -> Tuple[List[go.Scatter], List[float]]:
        """Create baseline traces showing raw measured values per iteration."""
        traces = []
        all_values = []
        baseline_colors = ['#888888', '#2d2d2d', '#d62728', '#ff7f0e', '#9467bd']
        baseline_dashes = ['dash', 'dot', 'dashdot']

        for idx, (baseline_key, baseline_result) in enumerate(baselines.items()):
            color = baseline_colors[idx % len(baseline_colors)]
            dash = baseline_dashes[idx % len(baseline_dashes)]
            display_name = f"{self.parser.build_display_name(baseline_key)} Baseline"

            trajectory = None
            if hasattr(baseline_result, 'trajectory') and baseline_result.trajectory:
                if not all(v == float('inf') for v in baseline_result.trajectory):
                    trajectory = baseline_result.trajectory

            if not trajectory and hasattr(baseline_result, 'raw_experiment'):
                exp = baseline_result.raw_experiment
                measured_configs = getattr(exp, 'measured_configurations', [])
                if measured_configs:
                    trajectory = []
                    for config in measured_configs:
                        results = getattr(config, 'averaged_result', None) or getattr(config, 'results', {})
                        if results:
                            value = results.get(objective)
                            if value is None and hasattr(results, 'keys'):
                                value = results[list(results.keys())[0]]
                            if value is not None:
                                trajectory.append(value)

            if trajectory:
                x_vals = list(range(len(trajectory)))
                all_values.extend([v for v in trajectory if v is not None and np.isfinite(v)])
                traces.append(go.Scatter(
                    x=x_vals, y=trajectory, mode='lines', name=display_name,
                    line=dict(color=color, dash=dash, width=2.5),
                    hovertemplate='%{x}, %{y:.4f}<extra></extra>'
                ))

        return traces, all_values

    def create_improvement_plot(self, objective: str, experiment_names: List[str], data_series: List[List[float]],
            plot_config: PlotConfig, baselines: Dict[str, Any] = None,
            known_optimum: Optional[float] = None,
            title_suffix: str = "") -> go.Figure:
        """Plot best-so-far objective values per iteration with optional known-optimum reference."""
        traces = []
        all_values = []
        max_iterations = 0

        for series, name in zip(data_series, experiment_names):
            x_vals = list(range(len(series)))
            max_iterations = max(max_iterations, len(series))
            all_values.extend([v for v in series if v is not None])
            traces.append(self._create_scatter_trace(x_vals, series, name, len(series) <= 1))

        if baselines:
            baseline_traces, baseline_values = self._create_baseline_traces(baselines, objective)
            traces.extend(baseline_traces)
            all_values.extend(baseline_values)
            for baseline_result in baselines.values():
                if hasattr(baseline_result, 'trajectory'):
                    max_iterations = max(max_iterations, len(baseline_result.trajectory))

        layout = dict(
            title=f'{objective} - {plot_config.metric_description}{title_suffix}',
            xaxis=dict(title=plot_config.metric_label,
                       range=[-0.5, max_iterations - 0.5 if max_iterations > 1 else 0.5]),
            yaxis=dict(title=plot_config.objective_label)
        )
        y_range = self._compute_robust_y_range(all_values)
        if y_range:
            layout['yaxis']['range'] = y_range
        if known_optimum is not None:
            self._add_optimum_to_layout(layout, known_optimum)
            traces.append(self._optimum_trace(known_optimum))

        self._apply_axis_config(layout, plot_config)
        return go.Figure(data=traces, layout=layout)

    def create_custom_plot(self, objective: str, experiment_names: List[str], objective_series: List[List[float]],
            time_series: List[List[Optional[float]]], plot_config: PlotConfig,
            baselines: Dict[str, Any] = None,
            known_optimum: Optional[float] = None) -> Optional[go.Figure]:
        traces = []
        all_objective = []

        for obj_vals, time_vals, name in zip(objective_series, time_series, experiment_names):
            valid_pairs = [(time_vals[i], obj_vals[i]) for i in range(min(len(obj_vals), len(time_vals)))
                           if obj_vals[i] is not None and time_vals[i] is not None]
            if not valid_pairs:
                continue
            x_vals, y_vals = zip(*valid_pairs)
            all_objective.extend(y_vals)
            traces.append(self._create_scatter_trace(x_vals, y_vals, name, len(valid_pairs) <= 1))

        if baselines:
            baseline_traces, baseline_values = self._create_baseline_traces(baselines, objective)
            traces.extend(baseline_traces)
            all_objective.extend(baseline_values)

        if not traces:
            return None

        layout = dict(
            title=f'{objective} - {plot_config.metric_description}',
            xaxis=dict(title=plot_config.metric_label),
            yaxis=dict(title=plot_config.objective_label)
        )
        y_range = self._compute_robust_y_range(all_objective)
        if y_range:
            layout['yaxis']['range'] = y_range
        if known_optimum is not None:
            self._add_optimum_to_layout(layout, known_optimum)
            traces.append(self._optimum_trace(known_optimum))

        self._apply_axis_config(layout, plot_config)
        return go.Figure(data=traces, layout=layout)

    def create_grouped_plot(self, objective: str, experiment_groups: Dict[str, List[Any]],
            plot_config: PlotConfig, extractor: Any,
            baselines: Dict[str, Any] = None,
            title_suffix: str = "",
            known_optimum: Optional[float] = None) -> Optional[go.Figure]:
        """Create grouped improvement plot (mean ± std band) or box plot per group.

        The y-axis is auto-scaled to the data range (min/max + 5 % padding).
        A dashed green reference line is drawn at ``known_optimum`` when provided.
        """
        if plot_config.plot_type == 'box_plot':
            return self._create_grouped_box_plot(
                objective, experiment_groups, plot_config, extractor, baselines,
                title_suffix=title_suffix, known_optimum=known_optimum)

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

            x_vals, mean_y, upper_y, lower_y = plot_data
            all_values.extend(v for v in mean_y + upper_y + lower_y if v is not None)
            traces.extend(self._create_grouped_traces(x_vals, lower_y, upper_y, mean_y, group_name, color, fill_color))

        if baselines:
            baseline_traces, baseline_values = self._create_baseline_traces(baselines, objective)
            traces.extend(baseline_traces)
            all_values.extend(baseline_values)

        if not traces:
            return None

        layout = self._create_grouped_layout(objective, plot_config, all_values,
                                             title_suffix=title_suffix, known_optimum=known_optimum)
        if known_optimum is not None:
            self._add_optimum_to_layout(layout, known_optimum)
            traces.append(self._optimum_trace(known_optimum))

        return go.Figure(data=traces, layout=layout)

    @staticmethod
    def _prepare_grouped_plot_data(grouped_data: Dict[str, Any]) -> Optional[Tuple[List, List, List, List]]:
        """Return ``(x_vals, mean_y, upper_y, lower_y)`` where the band is mean ± std."""
        metric_vals = grouped_data['metric_values']
        mean_vals = grouped_data['mean_values']
        std_vals = grouped_data.get('std_values', [None] * len(mean_vals))

        valid_indices = [i for i in range(len(metric_vals))
            if metric_vals[i] is not None and mean_vals[i] is not None]
        if not valid_indices:
            return None

        x_vals = [metric_vals[i] for i in valid_indices]
        mean_y = [mean_vals[i] for i in valid_indices]
        std_y = [std_vals[i] if std_vals[i] is not None else 0.0 for i in valid_indices]

        return x_vals, mean_y, [m + s for m, s in zip(mean_y, std_y)], [m - s for m, s in zip(mean_y, std_y)]

    @staticmethod
    def _create_grouped_traces(x_vals: List, lower_y: List, upper_y: List, mean_y: List,
            group_name: str, color: str, fill_color: str) -> List[go.Scatter]:
        """Three traces per group: invisible upper bound, ±std shaded band, mean line.

        Only the mean line appears in the legend (one entry per group).
        The std band is linked to the same legend group so hovering highlights both.
        """
        return [
            go.Scatter(x=x_vals, y=upper_y, mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip', legendgroup=group_name),
            go.Scatter(x=x_vals, y=lower_y, mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=fill_color,
                showlegend=False, hovertemplate='%{x}, %{y:.4f}<extra></extra>',
                legendgroup=group_name),
            go.Scatter(x=x_vals, y=mean_y, mode='lines+markers',
                name=group_name, line=dict(color=color), marker=dict(color=color),
                hovertemplate='%{x}, %{y:.4f}<extra></extra>', legendgroup=group_name),
        ]

    def _create_grouped_layout(self, objective: str, plot_config: PlotConfig, all_values: List[float],
                                title_suffix: str = "",
                                known_optimum: Optional[float] = None) -> Dict[str, Any]:
        layout = dict(
            title=f'{objective}: ({plot_config.metric_description}){title_suffix}',
            xaxis=dict(title=plot_config.metric_label),
            yaxis=dict(title=plot_config.objective_label),
        )
        self._apply_axis_config(layout, plot_config)
        y_range = self._compute_robust_y_range(all_values)
        if y_range:
            layout['yaxis']['range'] = y_range
        return layout

    def _create_grouped_box_plot(self, objective: str, experiment_groups: Dict[str, List[Any]],
                                 plot_config: PlotConfig, extractor: Any,
                                 baselines: Dict[str, Any] = None,
                                 title_suffix: str = "",
                                 known_optimum: Optional[float] = None) -> Optional[go.Figure]:
        traces = []
        all_values = []

        for group_name, exp_list in experiment_groups.items():
            group_values = []
            for exp in exp_list:
                trajectory = extractor.extract_objective_series(exp, objective)
                if trajectory:
                    group_values.extend(v for v in trajectory if v is not None and np.isfinite(v))
            if group_values:
                all_values.extend(group_values)
                traces.append(go.Box(y=group_values, name=group_name, boxmean='sd',
                    marker=dict(opacity=0.7), hovertemplate='%{y:.4f}<extra></extra>'))

        if baselines:
            for baseline_key, baseline_result in baselines.items():
                trajectory = None
                if hasattr(baseline_result, 'trajectory') and baseline_result.trajectory:
                    if not all(v == float('inf') for v in baseline_result.trajectory):
                        trajectory = baseline_result.trajectory
                if trajectory:
                    valid_values = [v for v in trajectory if v is not None and np.isfinite(v)]
                    if valid_values:
                        all_values.extend(valid_values)
                        baseline_name = self.parser.build_display_name(baseline_key)
                        traces.append(go.Box(y=valid_values, name=f"{baseline_name} Baseline",
                            boxmean='sd', marker=dict(opacity=0.5, color='gray'),
                            hovertemplate='%{y:.4f}<extra></extra>'))

        if not traces:
            return None

        layout = dict(
            title=f'{objective} - Grouped Distribution Comparison{title_suffix}',
            xaxis=dict(title='Test Case'), yaxis=dict(title=plot_config.objective_label),
            showlegend=True, boxmode='overlay',
        )
        y_range = self._compute_robust_y_range(all_values)
        if y_range:
            layout['yaxis']['range'] = y_range
        if known_optimum is not None:
            self._add_optimum_to_layout(layout, known_optimum)
            traces.append(self._optimum_trace(known_optimum))

        self._apply_axis_config(layout, plot_config)
        return go.Figure(data=traces, layout=layout)

    def create_box_plot(self, objective: str, experiment_names: List[str], data_series: List[List[float]],
            plot_config: PlotConfig, baselines: Dict[str, Any] = None,
            known_optimum: Optional[float] = None) -> go.Figure:
        traces = []
        all_values = []

        for series, name in zip(data_series, experiment_names):
            if not series:
                continue
            valid_values = [v for v in series if v is not None and np.isfinite(v)]
            if not valid_values:
                continue
            all_values.extend(valid_values)
            traces.append(go.Box(y=valid_values, name=name, boxmean='sd',
                marker=dict(opacity=0.7), hovertemplate='%{y:.4f}<extra></extra>'))

        if baselines:
            for baseline_key, baseline_result in baselines.items():
                trajectory = None
                if hasattr(baseline_result, 'trajectory') and baseline_result.trajectory:
                    if not all(v == float('inf') for v in baseline_result.trajectory):
                        trajectory = baseline_result.trajectory
                if trajectory:
                    valid_values = [v for v in trajectory if v is not None and np.isfinite(v)]
                    if valid_values:
                        all_values.extend(valid_values)
                        display_name = f"{self.parser.build_display_name(baseline_key)} Baseline"
                        traces.append(go.Box(y=valid_values, name=display_name, boxmean='sd',
                            marker=dict(opacity=0.6, color='#808080'), line=dict(color='#606060'),
                            hovertemplate='%{y:.4f}<extra></extra>'))

        layout = dict(
            title=f'{objective} - Distribution Comparison',
            xaxis=dict(title='Algorithm'), yaxis=dict(title=plot_config.objective_label),
            showlegend=True, boxmode='overlay',
        )
        y_range = self._compute_robust_y_range(all_values)
        if y_range:
            layout['yaxis']['range'] = y_range
        if known_optimum is not None:
            self._add_optimum_to_layout(layout, known_optimum)
            traces.append(self._optimum_trace(known_optimum))

        self._apply_axis_config(layout, plot_config)

        return go.Figure(data=traces, layout=layout)
