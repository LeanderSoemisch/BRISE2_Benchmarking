import sys
import json
import math
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Set

import pandas as pd
import plotly.graph_objs as go
import webbrowser
from pathlib import Path
import numpy as np

# Import hypervolume analysis modules
try:
    from hypervolume_analysis import HypervolumeCalculator, HypervolumeTracker
    from hypervolume_visualization import HypervolumePlotter
    HYPERVOLUME_AVAILABLE = True
except ImportError:
    HYPERVOLUME_AVAILABLE = False
    print("Warning: Hypervolume analysis not available. Install pygmo for hypervolume support.")


# Ensure pickled modules like 'core_entities' are importable when unpickling
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_NODE_PATH = str(PROJECT_ROOT / 'main_node')
if MAIN_NODE_PATH not in sys.path:
    sys.path.insert(0, MAIN_NODE_PATH)


# Minimal Experiment/Configuration protocol expected from BRISE PKL


def _safe(obj: Dict, path: List[str], default=None):
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


@dataclass
class AnalyzerConfig:
    results_folder: str
    output_directory: str  # Where to save reports (.html, .csv, .zip)
    improvement_objective: Optional[str]
    improvement_direction: str  # 'minimize' | 'maximize'
    improvement_normalize: bool
    improvement_normalization: str  # 'none' | 'min_over_all_experiments'
    improvement_axis_label: str
    time_metric_name: Optional[str]  # e.g. 'time'
    time_axis_label: Optional[str]
    time_axis_scale: str  # 'linear' | 'log10'
    label_by: str  # 'experiment_name' | 'ed_id'

    @staticmethod
    def from_json(cfg: Dict[str, Any]) -> "AnalyzerConfig":
        b = cfg.get("Benchmark", {})
        # Support new schema: Resources.Folder; fallback to old ResultsSource.Folder
        folder = (
            b.get("Resources", {}).get("Folder")
            or b.get("ResultsSource", {}).get("Folder")
            or "./results/serialized/"
        )
        # Support new schema: Report.outputDirectory; default to ./results/reports/
        output_dir = (
            b.get("Report", {}).get("outputDirectory")
            or "./results/reports/"
        )
        # ExperimentSeries vs Series
        series = b.get("ExperimentSeries", b.get("Series", {})) or {}
        plots = b.get("Plots", {}) or {}
        # Improvement vs ImprovementPlot
        impr = plots.get("Improvement", plots.get("ImprovementPlot", {})) or {}
        time_plot = plots.get("Time", plots.get("TimePlot", {})) or None
        y_impr = (impr.get("Y") or {}) if impr else {}
        # ObjectiveName is optional now (we can discover objectives automatically)
        improvement_objective = y_impr.get("ObjectiveName")
        improvement_axis_label = y_impr.get("Label", "Objective value")

        # Extract time plot configuration
        time_y_cfg = time_plot.get("Y", {}) if time_plot else {}
        # Support both MetricName (for task-level metrics) and ObjectiveName: "Time" (for iteration timestamps)
        time_metric_name = time_y_cfg.get("MetricName")
        if not time_metric_name and time_y_cfg.get("ObjectiveName") == "Time":
            time_metric_name = "iteration_time"  # Special marker for iteration-based timing
        time_axis_label = time_y_cfg.get("Label", "Elapsed time (s)") if time_metric_name else None
        time_axis_scale = time_y_cfg.get("Scale", "linear") if time_metric_name else "linear"

        return AnalyzerConfig(
            results_folder=folder,
            output_directory=output_dir,
            improvement_objective=improvement_objective,
            improvement_direction=y_impr.get("Direction", "minimize"),
            improvement_normalize=y_impr.get("Normalize", True),
            improvement_normalization=y_impr.get("Normalization", "min_over_all_experiments"),
            improvement_axis_label=improvement_axis_label,
            time_metric_name=time_metric_name,
            time_axis_label=time_axis_label,
            time_axis_scale=time_axis_scale,
            label_by=series.get("LabelBy", "experiment_name"),
        )


# Utilities to extract needed arrays from BRISE Experiment objects


def load_experiments(folder: str):
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    exps = []
    for f in sorted(files):
        with open(os.path.join(folder, f), 'rb') as inp:
            exp = pickle.load(inp)
            exps.append(exp)
    if not exps:
        raise FileNotFoundError(f"No .pkl dumps found in {folder}")
    return exps


def exp_name(exp) -> str:
    # prefer explicit name, then ed_id
    return getattr(exp, 'name', None) or getattr(exp, 'ed_id', None) or 'experiment'


def format_experiment_name(name: str) -> str:
    """Format raw experiment identifiers into a readable label.
    - Convert embedded timestamps _yymmddHHMM[SS]_ to dd/MM/YYYY HH:MM
    - Truncate 40-char hashes to 8 chars
    - Remove stray markers like '[:8]'
    """
    try:
        ts = re.search(r'_(\d{10,12})(?:_|$)', name)
        if ts:
            ts_str = ts.group(1)
            for fmt in ('%y%m%d%H%M%S', '%y%m%d%H%M'):
                try:
                    dt = datetime.strptime(ts_str, fmt)
                    name = name.replace(ts_str, dt.strftime('%d/%m/%Y %H:%M'))
                    break
                except ValueError:
                    continue
        parts = name.split('_')
        cleaned = []
        for p in parts:
            if re.fullmatch(r'[0-9a-f]{40}', p):
                cleaned.append(p[:8])
            else:
                cleaned.append(p.replace('[:8]', ''))
        return '_'.join(cleaned)
    except Exception:
        return name


def parse_experiment_features(raw_name: str) -> Dict[str, Any]:
    """Parse features from clean experiment naming pattern:
    exp_<task>_<model>_<sampler>_<configStrategy>_<stopCondition>[_<idx>]
    Example: exp_test_GPR_sobol_quantitybased_timebased or exp_test_GPR_sobol_quantitybased_timebased_1
    """
    features: Dict[str, Any] = {
        'Task': None,
        'Model': None,
        'Sampler': None,
        'ConfigurationStrategy': None,
        'StopCondition': None,
        'Index': None
    }
    try:
        # Pattern: exp_<task>_<model>_<sampler>_<configStrategy>_<stopCondition>[_<idx>]
        pattern = r'^exp_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+?)(?:_(\d+))?$'
        m = re.match(pattern, raw_name)
        if m:
            features['Task'] = m.group(1)
            features['Model'] = m.group(2)
            features['Sampler'] = m.group(3)
            features['ConfigurationStrategy'] = m.group(4)
            features['StopCondition'] = m.group(5)
            if m.group(6):
                features['Index'] = int(m.group(6))
    except Exception:
        pass
    return features


def build_display_name(raw_name: str) -> str:
    """Use the clean experiment name directly.
    Since file names are now generated as exp_<task>_<model>_<sampler>_<config>_<stop>[_<idx>],
    they are already readable and descriptive."""
    return raw_name


# Generic objective discovery

def discover_objective_keys(exps) -> Set[str]:
    keys = set()
    for exp in exps:
        for conf in getattr(exp, 'measured_configurations', [])[:3]:  # sample first few
            res = getattr(conf, 'results', {})
            for k, v in res.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    keys.add(k)
    return keys


def extract_objective_series(exp, objective: str, direction: str):
    ys = []
    for conf in getattr(exp, 'measured_configurations', []):
        res = getattr(conf, 'results', {})
        if objective in res:
            val = res[objective]
            if isinstance(val, (int, float)) and not math.isnan(val):
                ys.append(float(val))
    return ys


def extract_time_series(exp, metric_name: str):
    """Extract time series from experiment.

    First tries to use iteration_timestamp if available (time from start to each iteration).
    Falls back to aggregating per-task metric if iteration_timestamp not available.
    """
    ys = []
    start_time = getattr(exp, 'start_time', None)

    for conf in getattr(exp, 'measured_configurations', []):
        # Try iteration_timestamp first (time from experiment start to this iteration)
        if hasattr(conf, 'iteration_timestamp') and start_time:
            try:
                delta = conf.iteration_timestamp - start_time
                time_val = delta.total_seconds()
                ys.append(time_val)
                continue
            except Exception:
                pass

        # Fallback: aggregate time over tasks if metric exists
        total = 0.0
        for t in conf.get_tasks().values():
            val = _safe(t, ['result', metric_name])
            if isinstance(val, (int, float)) and not math.isnan(val):
                total += float(val)
        ys.append(total if total > 0 else None)
    return ys


def normalize_series(series_list: List[List[float]], method: str) -> List[List[float]]:
    if method == 'none':
        return series_list
    if method == 'min_over_all_experiments':
        # find global min ignoring None
        mins = [min([y for y in s if y is not None], default=None) for s in series_list]
        global_min = min([m for m in mins if m is not None]) if any(m is not None for m in mins) else None
        if global_min in (None, 0):
            return series_list
        return [[(y / global_min) if (y is not None) else None for y in s] for s in series_list]
    return series_list


def best_so_far(ys: List[Optional[float]]):
    best = []
    cur = None
    for y in ys:
        if y is None:
            best.append(cur)
            continue
        cur = y if cur is None else min(cur, y)
        best.append(cur)
    return best


def best_so_far_direction(ys: List[Optional[float]], direction: str):
    best = []
    cur = None
    for y in ys:
        if y is None:
            best.append(cur)
            continue
        if cur is None:
            cur = y
        else:
            if direction == 'minimize':
                cur = min(cur, y)
            else:
                cur = max(cur, y)
        best.append(cur)
    return best


def build_improvement_figures(exps, cfg: AnalyzerConfig) -> Dict[str, go.Figure]:
    objective_candidates = discover_objective_keys(exps)
    # Ensure configured objective first if present
    ordered_objectives = []
    if cfg.improvement_objective in objective_candidates:
        ordered_objectives.append(cfg.improvement_objective)
    for o in sorted(objective_candidates):
        if o not in ordered_objectives:
            ordered_objectives.append(o)
    figures = {}
    for obj in ordered_objectives:
        series = []
        names = []
        for exp in exps:
            ys = extract_objective_series(exp, obj, cfg.improvement_direction)
            if len(ys) == 0:
                continue  # skip experiments without this objective
            series.append(ys)
            names.append(build_display_name(exp_name(exp)))
        if not series:
            continue
        if cfg.improvement_normalize:
            series = normalize_series(series, cfg.improvement_normalization)
        traces = []
        max_iterations = 0
        for ys, name in zip(series, names):
            x = list(range(len(ys)))
            max_iterations = max(max_iterations, len(ys))
            mode = 'markers' if len(ys) <= 1 else 'lines+markers'
            trace = go.Scatter(x=x, y=best_so_far(ys), mode=mode, name=f"{name} best")
            if len(ys) == 1:
                trace.update(marker=dict(size=12))
            traces.append(trace)
        x_range = [-0.5, max_iterations - 0.5 if max_iterations > 1 else 0.5]
        fig = go.Figure(data=traces, layout=dict(title=f'Objective "{obj}" best-so-far', xaxis=dict(title='Iteration', range=x_range), yaxis=dict(title='Normalised objective')))
        figures[obj] = fig
    return figures


def build_time_plot(exps, cfg: AnalyzerConfig):
    if not cfg.time_metric_name:
        return None
    traces = []
    has_numeric = False
    max_iterations = 0
    for exp in exps:
        ys = extract_time_series(exp, cfg.time_metric_name)
        if any(y is not None for y in ys):
            has_numeric = True
        x = list(range(len(ys)))
        max_iterations = max(max_iterations, len(ys))
        mode = 'markers' if len(ys) <= 1 else 'lines+markers'
        trace = go.Scatter(x=x, y=ys, mode=mode, name=build_display_name(exp_name(exp)))
        if len(ys) == 1:
            trace.update(marker=dict(size=12))
        traces.append(trace)
    if not has_numeric:
        return None
    x_range = [-0.5, max_iterations - 0.5 if max_iterations > 1 else 0.5]
    fig = go.Figure(data=traces, layout=dict(title='Computation time per iteration', xaxis=dict(title='Iteration', range=x_range), yaxis=dict(title='Computation time (s)')))
    return fig


# Utility to detect headless/container environment

def _is_headless():
    if os.environ.get('DISPLAY') is None and sys.platform.startswith('linux'):
        return True
    if os.path.exists('/.dockerenv'):  # inside docker
        return True
    return False


# Lightweight modular wrappers (backwards-compatible)
class DataLoader:
    def __init__(self, results_folder: str):
        self.results_folder = results_folder

    def load(self):
        return load_experiments(self.results_folder)


class MetricExtractor:
    def discover(self, exps):
        return discover_objective_keys(exps)

    def series(self, exp, objective: str, direction: str):
        return extract_objective_series(exp, objective, direction)


class Normalizer:
    def normalize(self, series_list: List[List[float]], method: str, enabled: bool):
        return normalize_series(series_list, method) if enabled else series_list


class PlotBuilder:
    @staticmethod
    def _robust_y_range(all_y: List[float]) -> Optional[List[float]]:
        vals = np.array([y for y in all_y if y is not None and np.isfinite(y)], dtype=float)
        if vals.size == 0:
            return None
        min_y = float(np.min(vals))
        max_y = float(np.max(vals))
        if not np.isfinite(min_y) or not np.isfinite(max_y):
            return None
        if max_y == min_y:
            pad = 0.05 * (abs(min_y) if min_y != 0 else 1.0)
            return [min_y - pad, max_y + pad]
        pad = 0.05 * (max_y - min_y)
        return [min_y - pad, max_y + pad]

    def improvement_figure(self, obj: str, names: List[str], series: List[List[float]]):
        traces = []
        max_iterations = 0
        all_bsf = []
        for ys, name in zip(series, names):
            x = list(range(len(ys)))
            max_iterations = max(max_iterations, len(ys))
            bsf = best_so_far(ys)
            all_bsf.extend([v for v in bsf if v is not None])
            mode = 'markers' if len(ys) <= 1 else 'lines+markers'
            trace = go.Scatter(x=x, y=bsf, mode=mode, name=f"{name} best")
            if len(ys) == 1:
                trace.update(marker=dict(size=12))
            traces.append(trace)
        x_range = [-0.5, max_iterations - 0.5 if max_iterations > 1 else 0.5]
        y_range = self._robust_y_range(all_bsf)
        layout = dict(title=f'Objective "{obj}" best-so-far', xaxis=dict(title='Iteration', range=x_range))
        if y_range:
            layout['yaxis'] = dict(title='Objective value', range=y_range)
        else:
            layout['yaxis'] = dict(title='Objective value')
        return go.Figure(data=traces, layout=layout)

    def runtime_figure(self, names: List[str], runtimes: List[float]):
        """Create a bar chart showing total runtime per experiment/algorithm.

        Args:
            names: Experiment names
            runtimes: Runtime in seconds for each experiment

        Returns:
            Plotly Figure or None if no valid data
        """
        if not runtimes or all(r is None for r in runtimes):
            return None

        # Filter out None values and keep matching names
        valid_pairs = [(n, r) for n, r in zip(names, runtimes) if r is not None]
        if not valid_pairs:
            return None

        names_valid, runtimes_valid = zip(*valid_pairs)

        trace = go.Bar(
            x=list(names_valid),
            y=list(runtimes_valid),
            marker=dict(color='#3498db'),
            text=[f"{r:.2f}s" for r in runtimes_valid],
            textposition='auto'
        )

        layout = dict(
            title='Total Runtime per Experiment',
            xaxis=dict(title='Experiment', tickangle=-45),
            yaxis=dict(title='Runtime (seconds)'),
            margin=dict(b=150)  # Extra bottom margin for rotated labels
        )

        return go.Figure(data=[trace], layout=layout)

    def time_figure(self, names: List[str], series: List[List[Optional[float]]]):
        traces = []
        max_iterations = 0
        all_y = []
        for ys, name in zip(series, names):
            x = list(range(len(ys)))
            max_iterations = max(max_iterations, len(ys))
            all_y.extend([v for v in ys if v is not None])
            mode = 'markers' if len(ys) <= 1 else 'lines+markers'
            trace = go.Scatter(x=x, y=ys, mode=mode, name=name)
            if len(ys) == 1:
                trace.update(marker=dict(size=12))
            traces.append(trace)
        if not any(len(s) for s in series):
            return None
        x_range = [-0.5, max_iterations - 0.5 if max_iterations > 1 else 0.5]
        y_range = self._robust_y_range(all_y)
        layout = dict(title='Elapsed time per iteration', xaxis=dict(title='Iteration', range=x_range))
        if y_range:
            layout['yaxis'] = dict(title='Elapsed time (s)', range=y_range)
        else:
            layout['yaxis'] = dict(title='Elapsed time (s)')
        return go.Figure(data=traces, layout=layout)

    def time_vs_objective_figure(self, obj: str, names: List[str], objective_series: List[List[float]],
                                  time_series: List[List[Optional[float]]], time_label: str, time_scale: str):
        """Create scatter plot of objective value vs computation time.

        Args:
            obj: Objective name
            names: Experiment names
            objective_series: List of objective value series (one per experiment)
            time_series: List of time metric series (one per experiment)
            time_label: Label for time axis (e.g., "Computation time (s)")
            time_scale: Scale type for time axis ('linear' or 'log10')

            Returns:
            Plotly Figure or None if no valid data
        """
        traces = []
        all_time = []
        all_obj = []

        for obj_ys, time_ys, name in zip(objective_series, time_series, names):
            # Pair objective values with time values, skipping None entries
            valid_pairs = []
            for i in range(min(len(obj_ys), len(time_ys))):
                if obj_ys[i] is not None and time_ys[i] is not None:
                    valid_pairs.append((time_ys[i], obj_ys[i]))

            if not valid_pairs:
                continue

            x_vals, y_vals = zip(*valid_pairs)
            all_time.extend(x_vals)
            all_obj.extend(y_vals)

            mode = 'markers' if len(valid_pairs) <= 1 else 'lines+markers'
            trace = go.Scatter(x=x_vals, y=y_vals, mode=mode, name=name)
            if len(valid_pairs) == 1:
                trace.update(marker=dict(size=12))
            traces.append(trace)

        if not traces:
            return None

        # Build layout with appropriate axis settings
        layout = dict(
            title=f'Objective "{obj}" vs {time_label}',
            xaxis=dict(title=time_label),
            yaxis=dict(title=f'{obj} value')
        )

        # Set x-axis to log scale if specified
        if time_scale == 'log10':
            layout['xaxis']['type'] = 'log'

        # Add robust y-range
        y_range = self._robust_y_range(all_obj)
        if y_range:
            layout['yaxis']['range'] = y_range

        return go.Figure(data=traces, layout=layout)


class ReportBuilder:
    def build(self, objective_figs: Dict[str, go.Figure], time_fig: Optional[go.Figure], tables_by_objective: Dict[str, List[Dict[str, Any]]], csv_map: Dict[str, str], zip_name: Optional[str], objective_time_figs: Optional[Dict[str, go.Figure]] = None, runtime_fig: Optional[go.Figure] = None, runtime_csv: Optional[str] = None):
        # Build tabbed HTML for multiple objectives and time; each objective tab shows its table with a download link
        tabs_html = []
        tab_buttons = []
        ordered_objs = list(objective_figs.keys())  # stable order
        objective_time_figs = objective_time_figs or {}

        def build_table_html(rows: List[Dict[str, Any]]):
            def _fmt(v):
                return '' if v is None or (isinstance(v, float) and (math.isnan(v) if isinstance(v, float) else False)) else (f"{v:.4g}" if isinstance(v, (int, float)) else str(v))
            # dynamic headers with preferred order
            preferred = ['Task','Model','Sampler','ConfigurationStrategy','StopCondition','Experiment','Objective','Iterations','Initial','Final best','Absolute improvement','Improvement %','Runtime (s)']
            keys = set()
            for r in rows:
                keys.update(r.keys())
            headers = [h for h in preferred if h in keys] + [k for k in sorted(keys) if k not in preferred]
            trs = ['<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>']
            for r in rows:
                tds = ''.join(f"<td class='{'exp-name' if h=='Experiment' else 'num'}'>{_fmt(r.get(h))}</td>" for h in headers)
                trs.append(f'<tr>{tds}</tr>')
            return "<table class='summary-table'>" + ''.join(trs) + "</table>"

        for i, (obj, fig) in enumerate(objective_figs.items()):
            div_id = f'obj_tab_{i}'
            active_class = 'active' if i == 0 else ''
            tab_buttons.append(f"<button class='tab-btn {active_class}' onclick=showTab('{div_id}',this)>{obj}</button>")
            table_html = build_table_html(tables_by_objective.get(obj, []))
            csv_rel = os.path.basename(csv_map.get(obj, ''))
            download_link = f"<div class='download-row'><a class='download-link' href='{csv_rel}' download>Download {obj} CSV</a></div>" if csv_rel else ''

            # Build tab content: improvement plot + time-vs-objective plot (if available) + table
            plot_id = f"plot_{div_id}"
            plots_html = f"<div class='plot-container'><div class='plot-wrapper' id='{plot_id}'>{fig.to_html(include_plotlyjs=False, full_html=False)}</div><button class='export-btn' onclick='exportPlotAsSVG(\"{plot_id}\", \"{obj}_improvement\")'>Export as SVG</button></div>"

            # Add time vs objective plot if available for this objective
            if obj in objective_time_figs:
                time_obj_fig = objective_time_figs[obj]
                time_plot_id = f"plot_{div_id}_time"
                plots_html += f"<div class='plot-container'><div class='plot-wrapper' id='{time_plot_id}'>{time_obj_fig.to_html(include_plotlyjs=False, full_html=False)}</div><button class='export-btn' onclick='exportPlotAsSVG(\"{time_plot_id}\", \"{obj}_vs_time\")'>Export as SVG</button></div>"

            tab_inner = f"{plots_html}{download_link}<div class='table-wrapper'>{table_html}</div>"
            tabs_html.append(f"<div id='{div_id}' class='tab-content' style='display:{'block' if i==0 else 'none'}'>{tab_inner}</div>")

        if time_fig:
            time_plot_id = 'plot_time_tab'
            tabs_html.append(f"<div id='time_tab' class='tab-content' style='display:none'><div class='plot-container'><div class='plot-wrapper' id='{time_plot_id}'>{time_fig.to_html(include_plotlyjs=False, full_html=False)}</div><button class='export-btn' onclick='exportPlotAsSVG(\"{time_plot_id}\", \"elapsed_time\")'>Export as SVG</button></div></div>")
            tab_buttons.append("<button class='tab-btn' onclick=showTab('time_tab',this)>time</button>")


        # Add runtime tab
        if runtime_fig:
            div_id = 'runtime_tab'
            runtime_plot_id = 'plot_runtime_tab'
            runtime_csv_link = f"<div class='download-row'><a class='download-link' href='{runtime_csv}' download>Download Runtime CSV</a></div>" if runtime_csv else ''
            tab_inner = f"<div class='plot-container'><div class='plot-wrapper' id='{runtime_plot_id}'>{runtime_fig.to_html(include_plotlyjs=False, full_html=False)}</div><button class='export-btn' onclick='exportPlotAsSVG(\"{runtime_plot_id}\", \"runtime\")'>Export as SVG</button></div>{runtime_csv_link}"
            tabs_html.append(f"<div id='{div_id}' class='tab-content' style='display:none'>{tab_inner}</div>")
            tab_buttons.append("<button class='tab-btn' onclick=showTab('runtime_tab',this)>runtime</button>")

        global_download = f"<div class='global-download'><a class='download-link all' href='{zip_name}' download>Download all tables (.zip)</a></div>" if zip_name else ''
        tabs_section = f"<div class='tabs'>{global_download}<div class='tab-buttons'>{''.join(tab_buttons)}</div>{''.join(tabs_html)}</div>"

        # Use .format template with escaped curly braces for CSS/JS blocks
        page_template = """<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
        <title>BRISE Benchmark Report</title>
        <style>
        body {{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:20px 40px;background:#f5f7fa;}}
        h1 {{text-align:center;margin:0 0 4px;font-size:2.1em;color:#2c3e50;font-weight:600;}}
        .subtitle {{text-align:center;color:#607080;font-size:.9em;margin:0 0 28px;}}
        .table-wrapper {{display:flex;justify-content:center;margin:10px auto;max-width:1160px;}}
        table.summary-table {{border-collapse:collapse;font-size:.78em;box-shadow:0 2px 6px rgba(0,0,0,.08);background:#fff;border-radius:8px;overflow:hidden;}}
        .summary-table th {{background:#2d6db3;color:#fff;padding:10px 14px;text-transform:uppercase;font-weight:600;letter-spacing:.5px;}}
        .summary-table td {{padding:8px 14px;border-bottom:1px solid #e3e8ed;}}
        .summary-table tr:last-child td {{border-bottom:none;}}
        .summary-table tr:hover td {{background:#f3f7fb;}}
        .summary-table td.exp-name {{text-align:center;font-weight:500;color:#1f2d3d;max-width:340px;white-space:nowrap;text-overflow:ellipsis;overflow:hidden;}}
        .summary-table td.num {{text-align:center;}}
        .tabs {{margin-top:30px;}}
        .tab-buttons {{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;}}
        .tab-btn {{background:#e0e6ed;border:none;padding:8px 14px;border-radius:20px;cursor:pointer;font-size:.75em;font-weight:600;letter-spacing:.5px;color:#2c3e50;}}
        .tab-btn.active {{background:#3498db;color:#fff;box-shadow:0 2px 4px rgba(0,0,0,.15);}}
        .tab-btn:hover {{background:#3aa0e3;color:#fff;}}
        .tab-content {{background:#fff;padding:18px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);}}
        .download-row {{text-align:right;margin-top:4px;}}
        .download-link {{display:inline-block;margin:6px 0 4px;padding:6px 12px;background:#4a89d8;color:#fff;text-decoration:none;border-radius:6px;font-size:.7em;font-weight:600;}}
        .download-link:hover {{background:#3a7cc8;}}
        .download-link.all {{background:#2d6db3;margin-bottom:12px;}}
        .download-link.all:hover {{background:#245b91;}}
        .global-download {{text-align:right;}}
        .plot-container {{margin-bottom:20px;}}
        .plot-wrapper {{margin-bottom:8px;}}
        .export-btn {{background:#27ae60;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:.75em;font-weight:600;margin-bottom:10px;}}
        .export-btn:hover {{background:#229954;}}
        footer {{margin-top:46px;padding-top:14px;font-size:.65em;color:#6b7b8c;text-align:center;border-top:1px solid #dfe6ec;}}
        </style>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
        <script>
        function showTab(id,btn){{
            document.querySelectorAll('.tab-content').forEach(function(e){{e.style.display='none';}});
            document.getElementById(id).style.display='block';
            document.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('active');}});
            btn.classList.add('active');
            setTimeout(function(){{
                document.querySelectorAll('.js-plotly-plot').forEach(function(p){{
                    if(p.offsetParent !== null) Plotly.Plots.resize(p);
                }});
            }},50);
        }}
        function exportPlotAsSVG(containerId, fileName){{
            // Find the plotly div within the container
            var container = document.getElementById(containerId);
            if (!container) {{
                console.error('Container not found:', containerId);
                return;
            }}
            var plotDiv = container.querySelector('.js-plotly-plot');
            if (!plotDiv) {{
                console.error('Plotly plot not found in container:', containerId);
                return;
            }}
            // Use Plotly's downloadImage function to export as SVG
            Plotly.downloadImage(plotDiv, {{
                format: 'svg',
                width: 1200,
                height: 600,
                filename: fileName
            }});
        }}
        window.addEventListener('load', function(){{
            setTimeout(function(){{
                document.querySelectorAll('.js-plotly-plot').forEach(function(p){{Plotly.Plots.resize(p);}});
            }},100);
        }});
        </script>
        </head><body>
        <h1>BRISE Benchmark Report</h1>
        <p class='subtitle'>Generated {generated_time} | Objectives: {objectives}</p>
        {tabs_section}
        <footer>Analyzer v2 • Auto-open is {auto_status} • Per-objective tables & downloads included</footer>
        </body></html>"""

        page = page_template.format(
            generated_time=datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            objectives=', '.join(ordered_objs),
            tabs_section=tabs_section,
            auto_status=('disabled' if _is_headless() else 'enabled')
        )
        return page


def main(template_json_path: str = './configs/benchmark_templates/benchmark_template.json', output_html: str = './results/reports/benchmark_poc.html', output_csv: str = './results/reports/benchmark_all_objectives.csv'):
    with open(template_json_path, 'r') as f:
        cfg = AnalyzerConfig.from_json(json.load(f))

    data = DataLoader(cfg.results_folder)
    exps = data.load()
    try:
        exps.sort(key=lambda e: getattr(e, 'start_time', 0))
    except Exception:
        pass

    extractor = MetricExtractor()
    Normalizer()
    plotter = PlotBuilder()

    objectives = extractor.discover(exps)
    ordered_objs = sorted(list(objectives))
    if cfg.improvement_objective and cfg.improvement_objective in objectives:
        ordered_objs = [cfg.improvement_objective] + [o for o in ordered_objs if o != cfg.improvement_objective]

    objective_figs = {}
    for obj in ordered_objs:
        names, series = [], []
        for exp in exps:
            ys = extractor.series(exp, obj, cfg.improvement_direction)
            series.append(ys)
            names.append(build_display_name(exp_name(exp)))
        objective_figs[obj] = plotter.improvement_figure(obj, names, series)

    # Time plot
    time_series = []
    time_names = []
    if cfg.time_metric_name:
        for exp in exps:
            time_names.append(build_display_name(exp_name(exp)))
            time_series.append(extract_time_series(exp, cfg.time_metric_name))
        time_fig = plotter.time_figure(time_names, time_series)
    else:
        time_fig = None

    # Time vs Objective plots (one per objective)
    objective_time_figs = {}
    if cfg.time_metric_name and cfg.time_axis_label:
        for obj in ordered_objs:
            obj_series = []
            time_obj_series = []
            names_obj = []
            for exp in exps:
                obj_ys = extractor.series(exp, obj, cfg.improvement_direction)
                time_ys = extract_time_series(exp, cfg.time_metric_name)
                if len(obj_ys) > 0:  # Only include experiments with this objective
                    obj_series.append(obj_ys)
                    time_obj_series.append(time_ys)
                    names_obj.append(build_display_name(exp_name(exp)))

            if obj_series:  # Only create plot if we have data
                fig = plotter.time_vs_objective_figure(
                    obj, names_obj, obj_series, time_obj_series,
                    cfg.time_axis_label, cfg.time_axis_scale
                )
                if fig:
                    objective_time_figs[obj] = fig

    # Build table rows per objective (raw values, not normalized)
    tables_by_objective: Dict[str, List[Dict[str, Any]]] = {}
    combined_rows: List[Dict[str, Any]] = []

    # Compute runtime for each experiment once
    exp_runtimes = {}
    for exp in exps:
        runtime = None
        if hasattr(exp, 'start_time') and hasattr(exp, 'end_time'):
            try:
                delta = exp.end_time - exp.start_time
                runtime = delta.total_seconds()
            except Exception:
                pass
        exp_runtimes[id(exp)] = runtime

    for obj in ordered_objs:
        rows = []
        for exp in exps:
            raw_ys = []
            for conf in getattr(exp, 'measured_configurations', []):
                res = getattr(conf, 'results', {})
                if obj in res:
                    val = res[obj]
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        raw_ys.append(float(val))
            if len(raw_ys) == 0:
                continue
            bsf_dir = best_so_far_direction(raw_ys, cfg.improvement_direction)
            final_best = bsf_dir[-1] if bsf_dir else None
            initial = raw_ys[0] if raw_ys else None
            if initial is not None and final_best is not None:
                if cfg.improvement_direction == 'minimize':
                    improvement_abs = initial - final_best
                else:
                    improvement_abs = final_best - initial
            else:
                improvement_abs = None
            if improvement_abs is not None and initial not in (None, 0):
                improvement_pct = improvement_abs / initial * 100
            else:
                improvement_pct = None
            feats = parse_experiment_features(exp_name(exp))
            row = {
                'Task': feats.get('Task'),
                'Model': feats.get('Model'),
                'Sampler': feats.get('Sampler'),
                'ConfigurationStrategy': feats.get('ConfigurationStrategy'),
                'StopCondition': feats.get('StopCondition'),
                'Experiment': build_display_name(exp_name(exp)),
                'Objective': obj,
                'Iterations': len(raw_ys),
                'Initial': initial,
                'Final best': final_best,
                'Absolute improvement': improvement_abs,
                'Improvement %': round(improvement_pct, 2) if improvement_pct is not None else None,
                'Runtime (s)': exp_runtimes[id(exp)]
            }
            rows.append(row)
            combined_rows.append(row.copy())
        if rows:
            tables_by_objective[obj] = rows

    # CSV write (combined)
    summary_df = pd.DataFrame(combined_rows)
    for col in ['Initial','Final best','Absolute improvement','Improvement %']:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').round(6)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    # Per-objective CSVs with meaningful names and map
    out_dir = os.path.dirname(output_csv) or '.'
    csv_map: Dict[str, str] = {}
    for obj, rows in tables_by_objective.items():
        df_obj = pd.DataFrame(rows)
        for col in ['Initial','Final best','Absolute improvement','Improvement %']:
            if col in df_obj.columns:
                df_obj[col] = pd.to_numeric(df_obj[col], errors='coerce').round(6)
        obj_filename = os.path.join(out_dir, f"benchmark_objective_{obj}.csv")
        df_obj.to_csv(obj_filename, index=False)
        csv_map[obj] = os.path.basename(obj_filename)

    # Build runtime table and plot
    runtime_rows = []
    runtime_names = []
    runtime_values = []
    for exp in exps:
        feats = parse_experiment_features(exp_name(exp))
        runtime = exp_runtimes[id(exp)]
        runtime_row = {
            'Task': feats.get('Task'),
            'Model': feats.get('Model'),
            'Sampler': feats.get('Sampler'),
            'ConfigurationStrategy': feats.get('ConfigurationStrategy'),
            'StopCondition': feats.get('StopCondition'),
            'Experiment': build_display_name(exp_name(exp)),
            'Runtime (s)': runtime
        }
        runtime_rows.append(runtime_row)
        runtime_names.append(build_display_name(exp_name(exp)))
        runtime_values.append(runtime)

    # Write runtime CSV
    runtime_csv_path = None
    runtime_fig = None
    if runtime_rows:
        runtime_df = pd.DataFrame(runtime_rows)
        if 'Runtime (s)' in runtime_df.columns:
            runtime_df['Runtime (s)'] = pd.to_numeric(runtime_df['Runtime (s)'], errors='coerce').round(4)
        runtime_csv_path = os.path.join(out_dir, "benchmark_total_runtime.csv")
        runtime_df.to_csv(runtime_csv_path, index=False)

        # Create runtime bar chart
        runtime_fig = plotter.runtime_figure(runtime_names, runtime_values)

    # Build zip of all tables (including runtime)
    base_name_no_ext = os.path.splitext(os.path.basename(output_csv))[0]
    zip_name = os.path.join(out_dir, f"{base_name_no_ext.replace('all_objectives','all_tables')}.zip")
    try:
        import zipfile
        with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # add combined
            zf.write(output_csv, arcname=os.path.basename(output_csv))
            for obj, fn in csv_map.items():
                zf.write(os.path.join(out_dir, fn), arcname=fn)
    except Exception as e:
        print('ZIP creation failed:', e)
        zip_name = None

    report = ReportBuilder()
    page = report.build(objective_figs, time_fig, tables_by_objective, csv_map, os.path.basename(zip_name) if zip_name else None, objective_time_figs)
    # Write outputs
    out_html_path = Path(output_html)
    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_html_path.write_text(page, encoding='utf-8')

    # Auto open safely
    if not _is_headless() and out_html_path.exists():
        try:
            uri = out_html_path.resolve().as_uri()
            if sys.platform.startswith('linux') and shutil.which('xdg-open'):
                import subprocess
                subprocess.Popen(['xdg-open', uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                webbrowser.open(uri)
        except Exception as e:
            print('Auto-open failed:', e)
    print(f'Report: {out_html_path} \nCSV (combined): {output_csv}')


if __name__ == '__main__':
    main()
