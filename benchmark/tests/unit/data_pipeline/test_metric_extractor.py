import math
from datetime import datetime

from analyzer.config import MetricType
from analyzer.data_pipeline.metric_extractor import MetricExtractor
from tests.helpers.fakes import create_fake_experiment


def test_extract_objective_series_tracks_best_so_far_with_enabled_guard():
    exp = create_fake_experiment(
        "exp_case",
        values=[{"Y1": 5.0}, {"Y1": 3.0}, {"Y1": 4.0}, {"Y1": 2.0}],
        is_enabled=[True, False, True, True],
    )

    series = MetricExtractor.extract_objective_series(exp, "Y1")

    assert series == [5.0, 5.0, 4.0, 2.0]


def test_extract_grouped_data_time_metric_computes_statistics_and_mean_time_axis():
    start = datetime(2024, 1, 1, 0, 0, 0)
    exp_a = create_fake_experiment("exp_a", values=[{"Y1": 6.0}, {"Y1": 4.0}], start_time=start)
    exp_b = create_fake_experiment("exp_b", values=[{"Y1": 5.0}], start_time=start)

    grouped = MetricExtractor.extract_grouped_data([exp_a, exp_b], "Y1", metric_type=MetricType.TIME.value)

    assert grouped["mean_values"] == [5.5, 4.5]
    assert grouped["min_values"] == [5.0, 4.0]
    assert grouped["max_values"] == [6.0, 5.0]
    assert grouped["metric_values"] == [0.0, 10.0]


def test_discover_objectives_filters_non_numeric_and_nan_values():
    exp = create_fake_experiment(
        "exp_case",
        values=[
            {"Y1": 1.0, "tag": "n/a", "bad": math.nan},
            {"Y2": 2, "text": "x"},
            {"Y3": 3.5},
            {"Y4": 4.5},
        ],
    )

    objectives = MetricExtractor.discover_objectives([exp])
    assert objectives == {"Y1", "Y2", "Y3", "Y4"}


