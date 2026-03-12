from typing import List, Dict, Optional, Tuple
import numpy as np
import plotly.graph_objs as go

from analyzer.comparison.comparison_processor import ComparisonResult
from analyzer.config.benchmark_config import Constants


class ComparativePlotGenerator:
    """Generator for comparative analysis plots"""

    def plot_regret_curves(
        self,
        comparison_results: List[ComparisonResult],
        title: str = "Regret Curves",
        log_scale: bool = False,
        regret_type: str = "iteration"
    ) -> Optional[go.Figure]:
        traces = []
        seen_experiments = {}
        for result in comparison_results:
            exp_name = result.display_name or result.experiment_name
            if exp_name not in seen_experiments:
                seen_experiments[exp_name] = result

        for idx, (exp_name, result) in enumerate(seen_experiments.items()):
            color = Constants.DEFAULT_COLORS[idx % len(Constants.DEFAULT_COLORS)]
            if regret_type == "time":
                if result.regret_curve_time is None:
                    continue
                x_vals = [t for t, _ in result.regret_curve_time]
                y_vals = [r for _, r in result.regret_curve_time]
            else:
                if result.regret_curve is None:
                    continue
                x_vals = list(range(len(result.regret_curve)))
                y_vals = result.regret_curve

            traces.append(go.Scatter(
                x=x_vals, y=y_vals, mode='lines+markers', name=exp_name,
                line=dict(color=color), marker=dict(size=4)
            ))

        if not traces:
            return None

        layout = dict(
            title=title,
            xaxis=dict(title='Time (seconds)' if regret_type == "time" else 'Iteration'),
            yaxis=dict(title='Regret (distance to optimum)'),
            hovermode='closest', showlegend=True
        )
        if log_scale:
            layout['yaxis']['type'] = 'log'

        return go.Figure(data=traces, layout=layout)

    BASELINE_COLORS = ['#2980b9', '#e67e22', '#27ae60', '#c0392b', '#8e44ad']

    def plot_normalized_improvement(
        self,
        comparison_results: List[ComparisonResult],
        title: str = "Normalized Improvement",
        improvement_type: str = "objective_value"
    ) -> Optional[go.Figure]:
        improvement_attr = {
            "objective_value": "normalized_improvement",
            "time_to_target": "normalized_improvement_time",
            "iteration_to_target": "normalized_improvement_iterations"
        }
        attr_name = improvement_attr.get(improvement_type, "normalized_improvement")
        is_ratio_based = improvement_type in ["time_to_target", "iteration_to_target"]

        experiment_names, improvements, baseline_types = [], [], []
        for result in comparison_results:
            value = getattr(result, attr_name, None)
            if value is not None:
                experiment_names.append(result.display_name or result.experiment_name)
                improvements.append(value)
                baseline_types.append(result.baseline_type)

        if not improvements:
            return None

        baseline_type_groups = {}
        for name, improvement, baseline in zip(experiment_names, improvements, baseline_types):
            baseline_type_groups.setdefault(baseline, {'names': [], 'improvements': []})
            baseline_type_groups[baseline]['names'].append(name)
            baseline_type_groups[baseline]['improvements'].append(improvement)

        ordered_experiments = list(dict.fromkeys(experiment_names))
        traces = []

        for idx, (baseline_type, data) in enumerate(baseline_type_groups.items()):
            color = self.BASELINE_COLORS[idx % len(self.BASELINE_COLORS)]
            imp_map = dict(zip(data['names'], data['improvements']))
            aligned_y = [imp_map.get(exp) for exp in ordered_experiments]
            text_labels = [
                f"{val:.2f}x" if (val is not None and abs(val) >= 0.1) else ''
                for val in aligned_y
            ]

            traces.append(go.Bar(
                x=ordered_experiments,
                y=aligned_y,
                name=f"vs {baseline_type}",
                marker=dict(color=color, line=dict(color=color, width=1)),
                text=text_labels,
                textposition='outside',
                textfont=dict(size=12, color='#2c3e50'),
                offsetgroup=str(idx),
            ))

        all_improvements = [v for v in improvements if v is not None]
        y_max = max(2.0, max(all_improvements))
        y_axis_max = y_max + max(0.3, y_max * 0.2)

        layout = dict(
            title=title,
            xaxis=dict(title='Experiment', tickangle=45),
            yaxis=dict(
                title='Speedup Factor' if is_ratio_based else 'Normalized Improvement',
                range=[0, y_axis_max],
            ),
            barmode='group',
            bargap=0.2,
            bargroupgap=0.05,
            showlegend=True,
            legend=dict(
                orientation='v', yanchor='top', y=1, xanchor='right', x=1.15,
            ),
            margin=dict(l=60, r=160, t=80, b=150),
        )

        return go.Figure(data=traces, layout=layout)


    def plot_performance_profile(
        self,
        performance_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "Performance Profile"
    ) -> Optional[go.Figure]:
        if not performance_profiles:
            return None

        traces = [
            go.Scatter(
                x=tau_values.tolist(), y=rho_values.tolist(),
                mode='lines+markers', name=test_case_name,
                line=dict(color=Constants.DEFAULT_COLORS[idx % len(Constants.DEFAULT_COLORS)], width=2),
                marker=dict(size=4)
            )
            for idx, (test_case_name, (tau_values, rho_values)) in enumerate(performance_profiles.items())
        ]

        all_tau_max = max(max(tau) for tau, _ in performance_profiles.values())

        layout = dict(
            title=title,
            xaxis=dict(title='Performance Ratio (τ)', range=[1, min(all_tau_max, 10)]),
            yaxis=dict(title='Fraction of Problems Solved (ρ)', range=[0, 1.05]),
            hovermode='closest', showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1)
        )

        return go.Figure(data=traces, layout=layout)

