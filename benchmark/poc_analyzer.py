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
import sys
from pathlib import Path
import numpy as np


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
    improvement_objective: str
    improvement_direction: str  # 'minimize' | 'maximize'
    improvement_normalize: bool
    improvement_normalization: str  # 'none' | 'min_over_all_experiments'
    time_metric_name: Optional[str]  # e.g. 'time'
    label_by: str  # 'experiment_name' | 'ed_id'

    @staticmethod
    def from_json(cfg: Dict[str, Any]) -> "AnalyzerConfig":
        b = cfg["Benchmark"]
        folder = b["ResultsSource"]["Folder"]
        series = b.get("Series", {})
        plots = b.get("Plots", {})
        impr = plots.get("ImprovementPlot", {})
        time_plot = plots.get("TimePlot", None)
        y_impr = impr.get("Y", {})
        return AnalyzerConfig(
            results_folder=folder,
            improvement_objective=y_impr.get("ObjectiveName", "Y1"),
            improvement_direction=y_impr.get("Direction", "minimize"),
            improvement_normalize=y_impr.get("Normalize", True),
            improvement_normalization=y_impr.get("Normalization", "min_over_all_experiments"),
            time_metric_name=time_plot.get("Y", {}).get("MetricName") if time_plot else None,
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
    # Simplified & corrected formatter: truncate 40-char hashes, format timestamps, remove residual markers
    timestamp_match = re.search(r'_(\d{10,12})(?:_|$)', name)
    if timestamp_match:
        ts_str = timestamp_match.group(1)
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
            cleaned.append(p.replace('[:8]', ''))  # remove artifacts if any
    return '_'.join(cleaned)


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
        value = None
        if objective in res:
            value = res[objective]
        else:
            # fallback: try numeric keys
            for k, v in res.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    value = v
                    break
        if value is not None:
            ys.append(float(value))
    # NOTE: no sign inversion; direction handled in best_so_far_direction
    return ys


def extract_time_series(exp, metric_name: str):
    ys = []
    for conf in getattr(exp, 'measured_configurations', []):
        # aggregate time over tasks if present
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
            series.append(ys)
            names.append(format_experiment_name(exp_name(exp)))
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
        trace = go.Scatter(x=x, y=ys, mode=mode, name=format_experiment_name(exp_name(exp)))
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
        layout = dict(title='Computation time per iteration', xaxis=dict(title='Iteration', range=x_range))
        if y_range:
            layout['yaxis'] = dict(title='Computation time (s)', range=y_range)
        else:
            layout['yaxis'] = dict(title='Computation time (s)')
        return go.Figure(data=traces, layout=layout)


class ReportBuilder:
    def build(self, objective_figs: Dict[str, go.Figure], time_fig: Optional[go.Figure], table_rows: List[Dict[str, Any]]):
        # Build tabbed HTML for multiple objectives and time
        tabs_html = []
        tab_buttons = []
        ordered_objs = list(objective_figs.keys())  # ensure stable order already constructed upstream
        for i, (obj, fig) in enumerate(objective_figs.items()):
            div_id = f'obj_tab_{i}'
            active_class = 'active' if i == 0 else ''
            tab_buttons.append(f"<button class='tab-btn {active_class}' onclick=showTab('{div_id}',this)>{obj}</button>")
            tabs_html.append(f"<div id='{div_id}' class='tab-content' style='display:{'block' if i==0 else 'none'}'>{fig.to_html(include_plotlyjs=False, full_html=False)}</div>")
        if time_fig:
            tabs_html.append(f"<div id='time_tab' class='tab-content' style='display:none'>{time_fig.to_html(include_plotlyjs=False, full_html=False)}</div>")
            tab_buttons.append("<button class='tab-btn' onclick=showTab('time_tab',this)>time</button>")
        tabs_section = f"<div class='tabs'><div class='tab-buttons'>{''.join(tab_buttons)}</div>{''.join(tabs_html)}</div>"

        # Render clean HTML summary table
        def _fmt(v):
            return '' if v is None or (isinstance(v,float) and (math.isnan(v) if isinstance(v,float) else False)) else (f"{v:.4g}" if isinstance(v,(int,float)) else str(v))
        headers = ['Experiment','Objective','Iterations','Initial','Final best','Absolute improvement','Improvement %']
        trs = ['<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>']
        for r in table_rows:
            tds = ''.join(f"<td class='{'exp-name' if h=='Experiment' else 'num'}'>{_fmt(r[h])}</td>" for h in headers)
            trs.append(f'<tr>{tds}</tr>')
        table_html = "<table class='summary-table'>" + ''.join(trs) + "</table>"

        # HTML page
        page = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
        <title>BRISE Benchmark Report</title>
        <style>
        /* ...existing styles... */
        body {{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:20px 40px;background:#f5f7fa;}}
        h1 {{text-align:center;margin:0 0 4px;font-size:2.1em;color:#2c3e50;font-weight:600;}}
        .subtitle {{text-align:center;color:#607080;font-size:.9em;margin:0 0 28px;}}
        .table-wrapper {{display:flex;justify-content:center;margin:10px auto;max-width:1160px;}}
        table.summary-table {{border-collapse:collapse;font-size:.78em;box-shadow:0 2px 6px rgba(0,0,0,.08);background:#fff;border-radius:8px;overflow:hidden;}}
        .summary-table th {{background:#2d6db3;color:#fff;padding:10px 14px;text-transform:uppercase;font-weight:600;letter-spacing:.5px;}}
        .summary-table td {{padding:8px 14px;border-bottom:1px solid #e3e8ed;}}
        .summary-table tr:last-child td {{border-bottom:none;}}
        .summary-table tr:hover td {{background:#f3f7fb;}}
        .summary-table td.exp-name {{text-align:left;font-weight:500;color:#1f2d3d;max-width:340px;white-space:nowrap;text-overflow:ellipsis;overflow:hidden;}}
        .summary-table td.num {{text-align:right;}}
        .tabs {{margin-top:30px;}}
        .tab-buttons {{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;}}
        .tab-btn {{background:#e0e6ed;border:none;padding:8px 14px;border-radius:20px;cursor:pointer;font-size:.75em;font-weight:600;letter-spacing:.5px;color:#2c3e50;}}
        .tab-btn.active {{background:#3498db;color:#fff;box-shadow:0 2px 4px rgba(0,0,0,.15);}}
        .tab-btn:hover {{background:#3aa0e3;color:#fff;}}
        .tab-content {{background:#fff;padding:18px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.08);}}
        footer {{margin-top:46px;padding-top:14px;font-size:.65em;color:#6b7b8c;text-align:center;border-top:1px solid #dfe6ec;}}
        </style>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
        <script>
        function showTab(id,btn){{
            document.querySelectorAll('.tab-content').forEach(e=>e.style.display='none');
            document.getElementById(id).style.display='block';
            document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
            btn.classList.add('active');
            setTimeout(()=>{{
                document.querySelectorAll('.js-plotly-plot').forEach(p=>{{
                    if(p.offsetParent !== null) Plotly.Plots.resize(p);
                }});
            }},50);
        }}
        window.addEventListener('load',()=>{{
            setTimeout(()=>{{
                document.querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.Plots.resize(p));
            }},100);
        }});
        </script>
        </head><body>
        <h1>BRISE Benchmark Report</h1>
        <p class='subtitle'>Generated {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Objectives: {', '.join(ordered_objs)}</p>
        <div class='table-wrapper'>{table_html}</div>
        {tabs_section}
        <footer>Analyzer v2 • Auto-open is {'disabled' if _is_headless() else 'enabled'}</footer>
        </body></html>"""

        return page


def main(template_json_path: str = './benchmark_template.json', output_html: str = './results/benchmark_poc.html', output_csv: str = './results/benchmark_poc.csv'):
    with open(template_json_path, 'r') as f:
        cfg = AnalyzerConfig.from_json(json.load(f))

    data = DataLoader(cfg.results_folder)
    exps = data.load()
    try:
        exps.sort(key=lambda e: getattr(e, 'start_time', 0))
    except Exception:
        pass

    extractor = MetricExtractor()
    normalizer = Normalizer()
    plotter = PlotBuilder()

    objectives = extractor.discover(exps)
    ordered_objs = [o for o in ([cfg.improvement_objective] + sorted(list(objectives))) if o in objectives or o == cfg.improvement_objective]

    objective_figs = {}
    for obj in ordered_objs:
        names, series = [], []
        for exp in exps:
            names.append(format_experiment_name(exp_name(exp)))
            ys = extractor.series(exp, obj, cfg.improvement_direction)
            series.append(ys)
        objective_figs[obj] = plotter.improvement_figure(obj, names, series)

    # Time plot
    time_series = []
    time_names = []
    if cfg.time_metric_name:
        for exp in exps:
            time_names.append(format_experiment_name(exp_name(exp)))
            time_series.append(extract_time_series(exp, cfg.time_metric_name))
        time_fig = plotter.time_figure(time_names, time_series)
    else:
        time_fig = None

    # Prepare summary table for the primary objective (first)
    primary_obj = ordered_objs[0] if ordered_objs else cfg.improvement_objective
    table_rows = []
    for exp in exps:
        # raw (non-normalized, non-direction-flipped) series for improvement calculation
        raw_ys = []
        for conf in getattr(exp, 'measured_configurations', []):
            res = getattr(conf, 'results', {})
            value = None
            if primary_obj in res:
                value = res[primary_obj]
            else:
                for k, v in res.items():
                    if isinstance(v, (int, float)) and not math.isnan(v):
                        value = v
                        break
            if value is not None:
                raw_ys.append(float(value))
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
        table_rows.append({
            'Experiment': format_experiment_name(exp_name(exp)),
            'Objective': primary_obj,
            'Iterations': len(raw_ys),
            'Initial': initial,
            'Final best': final_best,
            'Absolute improvement': improvement_abs,
            'Improvement %': round(improvement_pct, 2) if improvement_pct is not None else None
        })

    # CSV write (rounded)
    summary_df = pd.DataFrame(table_rows)
    for col in ['Initial','Final best','Absolute improvement','Improvement %']:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').round(6)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    report = ReportBuilder()
    page = report.build(objective_figs, time_fig, table_rows)

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
    print(f'Report: {out_html_path} \nCSV: {output_csv}')


if __name__ == '__main__':
    main()
