import math
from typing import List, Dict, Any, Set, Optional, Tuple

from analyzer.config.benchmark_config import TableConfig
from analyzer.data_pipeline import ExperimentParser, MetricExtractor, DataProcessor


class TableGenerator:
    """Builds data tables for the report"""

    COLUMN_MAP = {'task': 'Task', 'model': 'Model', 'sampler': 'Sampler',
        'configuration_strategy': 'ConfigurationStrategy', 'stop_condition': 'StopCondition', 'test_case': 'TestCase',
        'experiment': 'Experiment', 'objective': 'Objective', 'iterations': 'Iterations', 'initial_value': 'Initial',
        'final_best_value': 'Final best', 'improvement_percentage': 'Improvement %',
        'improvement_absolute': 'Absolute improvement', 'runtime': 'Runtime (s)'}

    PREFERRED_COLUMN_ORDER = ['Task', 'Model', 'Sampler', 'ConfigurationStrategy', 'StopCondition', 'TestCase',
        'Experiment', 'Objective', 'Iterations', 'Initial', 'Final best', 'Absolute improvement', 'Improvement %',
        'Runtime (s)']

    def __init__(self, parser: ExperimentParser, extractor: MetricExtractor):
        self.parser = parser
        self.extractor = extractor

    def create_table(self, experiments: List[Any], objective: str, table_config: TableConfig) -> List[Dict[str, Any]]:
        """Build summary table for a specific objective"""
        rows = []
        for exp in experiments:
            row = self._build_experiment_row(exp, objective, table_config)
            if row:
                rows.append(row)
        return rows

    def _build_experiment_row(self, exp: Any, objective: str, table_config: TableConfig) -> Optional[Dict[str, Any]]:
        """Build table row for single experiment"""
        values = self.extractor.extract_objective_series(exp, objective)
        if not values:
            return None

        initial = values[0]
        best_series = DataProcessor.compute_best_so_far(values)
        final_best = best_series[-1] if best_series else None

        improvement_abs, improvement_pct = self._compute_improvement(initial, final_best)

        exp_name = self.parser.get_name(exp)
        features = self.parser.parse_features(exp_name)
        runtime = self.extractor.extract_runtime(exp)

        return {'Task': features.get('Task'), 'Model': features.get('Model'), 'Sampler': features.get('Sampler'),
            'ConfigurationStrategy': features.get('ConfigurationStrategy'),
            'StopCondition': features.get('StopCondition'), 'TestCase': features.get('TestCase'),
            'Experiment': self.parser.build_display_name(exp_name), 'Objective': objective, 'Iterations': len(values),
            'Initial': initial, 'Final best': final_best, 'Absolute improvement': improvement_abs,
            'Improvement %': round(improvement_pct, 2) if improvement_pct is not None else None, 'Runtime (s)': runtime}

    @staticmethod
    def _compute_improvement(initial: Optional[float], final: Optional[float]) -> Tuple[
        Optional[float], Optional[float]]:
        """Compute absolute and percentage improvement"""
        if initial is None or final is None:
            return None, None

        improvement_abs = initial - final
        improvement_pct = (improvement_abs / initial * 100) if initial != 0 else None

        return improvement_abs, improvement_pct

    def format_table(self, rows: List[Dict[str, Any]], table_config: TableConfig) -> str:
        """Format table rows as HTML with column filtering"""
        if not rows:
            return "<p>No data available</p>"

        allowed_columns = self._get_allowed_columns(table_config)
        headers = self._order_headers(rows, allowed_columns)

        html_parts = ["<table class='summary-table'>", self._build_header_row(headers)]
        html_parts.extend(self._build_data_rows(rows, headers))
        html_parts.append("</table>")

        return ''.join(html_parts)

    @staticmethod
    def _get_allowed_columns(table_config: TableConfig) -> Set[str]:
        """Get set of allowed columns based on config"""
        allowed = set()
        for config_field, column_name in TableGenerator.COLUMN_MAP.items():
            if getattr(table_config, config_field, True):
                allowed.add(column_name)
        return allowed

    def _order_headers(self, rows: List[Dict[str, Any]], allowed_columns: Set[str]) -> List[str]:
        """Order headers according to preferred order"""
        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())

        headers = [h for h in self.PREFERRED_COLUMN_ORDER if h in all_columns and h in allowed_columns]
        headers += [c for c in sorted(all_columns) if c not in headers and c in allowed_columns]

        return headers

    @staticmethod
    def _build_header_row(headers: List[str]) -> str:
        """Build HTML header row"""
        return '<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>'

    @staticmethod
    def _build_data_rows(rows: List[Dict[str, Any]], headers: List[str]) -> List[str]:
        """Build HTML data rows"""
        html_rows = []
        for row in rows:
            cells = []
            for header in headers:
                value = TableGenerator._format_value(row.get(header))
                css_class = 'exp-name' if header == 'Experiment' else 'num'
                cells.append(f"<td class='{css_class}'>{value}</td>")

            html_rows.append('<tr>' + ''.join(cells) + '</tr>')

        return html_rows

    @staticmethod
    def _format_value(val: Any) -> str:
        """Format value for HTML display"""
        if val is None:
            return ''
        if isinstance(val, float):
            if math.isnan(val):
                return ''
            return f"{val:.4g}"
        return str(val)
