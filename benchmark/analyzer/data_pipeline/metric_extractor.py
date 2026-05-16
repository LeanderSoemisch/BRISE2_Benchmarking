import math
from typing import Any, List, Optional, Set, Dict, Tuple

from analyzer.config import MetricType


class MetricExtractor:
    """Extracts metrics and objectives from experiments"""

    @staticmethod
    def discover_objectives(experiments: List[Any]) -> Set[str]:
        """Discover all objective keys from experiments"""
        objectives = set()
        for exp in experiments:
            configs = getattr(exp, 'measured_configurations', [])[:5]
            for conf in configs:
                results = getattr(conf, 'results', {})
                for key, value in results.items():
                    if MetricExtractor._is_valid_numeric(value):
                        objectives.add(key)
        return objectives

    @staticmethod
    def _is_valid_numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not math.isnan(value)

    @staticmethod
    def extract_objective_series(exp: Any, objective: str) -> List[float]:
        """Extract best-so-far objective series from experiment (cumulative minimum)"""
        values: List[float] = []
        current_best: Optional[float] = None

        for conf in getattr(exp, 'measured_configurations', []):
            results = getattr(conf, 'results', {})
            is_enabled = getattr(conf, 'is_enabled', True)
            val = results.get(objective) if results else None
            if val is not None and MetricExtractor._is_valid_numeric(val):
                fval = float(val)
                if current_best is None:
                    current_best = fval
                elif is_enabled and fval < current_best:
                    current_best = fval
                values.append(current_best)

        return values

    @staticmethod
    def extract_time_series(exp: Any) -> List[Optional[float]]:
        """Extract time series from experiment using iteration timestamps"""
        times = []
        start_time = getattr(exp, 'start_time', None)

        for conf in getattr(exp, 'measured_configurations', []):
            time_val = MetricExtractor._calculate_time_delta(conf, start_time)
            times.append(time_val)

        return times

    @staticmethod
    def _calculate_time_delta(conf: Any, start_time: Any) -> Optional[float]:
        if not hasattr(conf, 'iteration_timestamp') or not start_time:
            return None
        try:
            delta = conf.iteration_timestamp - start_time
            return delta.total_seconds()
        except Exception:
            return None

    @staticmethod
    def extract_runtime(exp: Any) -> Optional[float]:
        if not (hasattr(exp, 'start_time') and hasattr(exp, 'end_time')):
            return None
        try:
            delta = exp.end_time - exp.start_time
            return delta.total_seconds()
        except Exception:
            return None

    @staticmethod
    def extract_grouped_data(experiments: List[Any], objective: str,
            metric_type: str = MetricType.ITERATION.value) -> Dict[str, Any]:
        """Extract min/max/mean objective values across grouped test case repetitions.

        Shorter repetitions are NOT forward-filled. At each iteration index we
        aggregate only over the repetitions that still have real data there; the
        returned ``sample_counts`` lets downstream code filter sparsely-sampled
        indices.

        Returns:
            Dict with min_values, max_values, mean_values, std_values,
            sample_counts and metric_values (all aligned by index).
        """
        if not experiments:
            return {}

        all_series = [MetricExtractor.extract_objective_series(exp, objective) for exp in experiments]
        all_series = [s for s in all_series if s]

        if not all_series:
            return {}

        all_time_series = []
        if metric_type == MetricType.TIME.value:
            all_time_series = [MetricExtractor.extract_time_series(exp) for exp in experiments]

        max_length = max(len(s) for s in all_series)
        min_values, max_values, mean_values, std_values, sample_counts = (
            MetricExtractor._compute_statistics(all_series, max_length)
        )
        metric_values = MetricExtractor._generate_metric_values(metric_type, max_length, all_time_series)

        return {'min_values': min_values, 'max_values': max_values, 'mean_values': mean_values,
                'std_values': std_values, 'sample_counts': sample_counts,
                'metric_values': metric_values}

    @staticmethod
    def extract_grouped_series_data(
        series_list: List[List[float]],
        metric_type: str = MetricType.ITERATION.value,
        time_series_list: Optional[List[List[Optional[float]]]] = None,
    ) -> Dict[str, Any]:
        if not series_list:
            return {}

        all_series = [s for s in series_list if s]
        if not all_series:
            return {}

        max_length = max(len(s) for s in all_series)
        min_values, max_values, mean_values, std_values, sample_counts = (
            MetricExtractor._compute_statistics(all_series, max_length)
        )

        time_series_list = time_series_list or []
        metric_values = MetricExtractor._generate_metric_values(metric_type, max_length, time_series_list)

        return {
            'min_values': min_values,
            'max_values': max_values,
            'mean_values': mean_values,
            'std_values': std_values,
            'sample_counts': sample_counts,
            'metric_values': metric_values,
        }

    @staticmethod
    def _compute_statistics(series_list: List[List[float]], length: int) -> Tuple[
        List[Optional[float]], List[Optional[float]], List[Optional[float]],
        List[Optional[float]], List[int]]:
        """Compute min, max, mean, std-dev and per-index sample count.

        Reps shorter than ``length`` contribute NaN at indices past their end, so
        nanmean/nanstd aggregate only over reps that actually reached that index.
        """
        import numpy as np
        arr = np.full((len(series_list), length), np.nan, dtype=float)
        for row, s in enumerate(series_list):
            for col, v in enumerate(s):
                if v is not None:
                    arr[row, col] = float(v)

        def _to_list(a: np.ndarray) -> List[Optional[float]]:
            return [None if np.isnan(v) else float(v) for v in a]

        sample_counts = (~np.isnan(arr)).sum(axis=0).astype(int).tolist()

        # Suppress "Mean of empty slice" / "Degrees of freedom <= 0" warnings at
        # indices with zero samples — those become NaN -> None in the output.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return (
                _to_list(np.nanmin(arr, axis=0)),
                _to_list(np.nanmax(arr, axis=0)),
                _to_list(np.nanmean(arr, axis=0)),
                _to_list(np.nanstd(arr, axis=0)),
                sample_counts,
            )

    @staticmethod
    def _generate_metric_values(metric_type: str, length: int, time_series_list: List[List[Optional[float]]]) -> List[
        Optional[float]]:
        if metric_type == MetricType.TIME.value and time_series_list:
            metric_values = []
            for i in range(length):
                times_at_i = [ts[i] for ts in time_series_list if i < len(ts) and ts[i] is not None]
                if times_at_i:
                    metric_values.append(sum(times_at_i) / len(times_at_i))
                else:
                    metric_values.append(None)
            return metric_values
        return list(range(length))
