"""Configuration module for benchmark analysis

Contains all configuration classes and constants for the benchmark analyzer.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class PlotType(Enum):
    """Plot type enumeration"""
    IMPROVEMENT = 'improvement_plot'
    HYPERVOLUME = 'hypervolume_plot'


class MetricType(Enum):
    """Metric type enumeration"""
    ITERATION = 'iteration'
    TIME = 'time'


class ScaleType(Enum):
    """Scale type enumeration"""
    LINEAR = 'linear'
    LOG10 = 'log10'


class NormalizationType(Enum):
    """Normalization strategy enumeration"""
    MIN_OVER_ALL = 'min_over_all_experiments'
    MAX_OVER_ALL = 'max_over_all_experiments'
    NONE = 'none'


class OptimizationDirection(Enum):
    """Optimization direction enumeration"""
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


class Constants:
    """Application constants"""
    DEFAULT_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]


@dataclass
class PlotConfig:
    """Configuration for a single plot"""
    plot_type: str
    metric_description: str
    metric_label: str
    metric_scale: str
    metric_type: str
    objectives_to_plot: List[str]
    normalize: bool
    normalization_strategy: str
    objective_label: str
    objective_scale: str

    def is_hypervolume_plot(self) -> bool:
        return self.plot_type == PlotType.HYPERVOLUME.value

    def uses_time_metric(self) -> bool:
        return self.metric_type == MetricType.TIME.value

    def should_plot_objective(self, objective: str) -> bool:
        return not self.objectives_to_plot or objective in self.objectives_to_plot


@dataclass
class TableConfig:
    """Configuration for table columns"""
    task: bool = True
    model: bool = True
    sampler: bool = True
    configuration_strategy: bool = True
    stop_condition: bool = True
    test_case: bool = False
    experiment: bool = True
    objective: bool = True
    iterations: bool = True
    initial_value: bool = True
    final_best_value: bool = True
    improvement_percentage: bool = True
    improvement_absolute: bool = True
    runtime: bool = True


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration"""
    output_directory: str
    name: str
    description: str


@dataclass
class BenchmarkConfig:
    """Main configuration for benchmark analysis"""
    results_folder: str
    output_directory: str
    experiment_name: str
    experiment_description: str
    objectives_to_measure: List[str]
    plots: List[PlotConfig]
    table_config: TableConfig

    @staticmethod
    def _parse_metric_type(metric_description: str) -> str:
        """Determine metric type from description"""
        return MetricType.TIME.value if 'time' in metric_description.lower() else MetricType.ITERATION.value

    @staticmethod
    def _parse_normalization_strategy(norm_strategy_data: Dict[str, Any]) -> str:
        """Extract normalization strategy from config"""
        if 'MinOverAll' in norm_strategy_data:
            return NormalizationType.MIN_OVER_ALL.value
        elif 'MaxOverAll' in norm_strategy_data:
            return NormalizationType.MAX_OVER_ALL.value
        return NormalizationType.NONE.value

    @staticmethod
    def _create_plot_config(plot_type: str, plot_data: Dict[str, Any]) -> PlotConfig:
        """Create PlotConfig from plot data dictionary"""
        metric_axis = plot_data.get("MetricAxis", {})
        objective_axis = plot_data.get("ObjectiveAxis", {})

        metric_desc = metric_axis.get("metricDescription", "iterations completed")
        metric_type = BenchmarkConfig._parse_metric_type(metric_desc)

        norm_strategy_data = objective_axis.get("NormalizationStrategy", {})
        norm_strategy = BenchmarkConfig._parse_normalization_strategy(norm_strategy_data)

        return PlotConfig(
            plot_type=plot_type,
            metric_description=metric_desc,
            metric_label=metric_axis.get("label", "iteration"),
            metric_scale=metric_axis.get("scale", ScaleType.LINEAR.value),
            metric_type=metric_type,
            objectives_to_plot=objective_axis.get("objectivesToPlot", []),
            normalize=objective_axis.get("normalize", True),
            normalization_strategy=norm_strategy,
            objective_label=objective_axis.get("label", "Objective value"),
            objective_scale=objective_axis.get("scale", ScaleType.LINEAR.value)
        )

    @staticmethod
    def from_json(cfg: Dict[str, Any]) -> "BenchmarkConfig":
        """Parse benchmark template JSON"""
        benchmark = cfg.get("Benchmark", {})

        folder = benchmark.get("Resources", {}).get("Folder", "./results/serialized/")
        output_dir = benchmark.get("Report", {}).get("outputDirectory", "./results/reports/")

        experiment = benchmark.get("Experiment", {})
        exp_name = experiment.get("name", "BRISE Benchmark Report")
        exp_description = experiment.get("description", "Benchmark analysis results")
        objectives = experiment.get("objectivesToMeasure", [])

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

        plots = []
        plot_keys = sorted([k for k in benchmark.keys() if k.startswith("Plot_")])

        for plot_key in plot_keys:
            plot_data = benchmark[plot_key]
            plot_type_data = plot_data.get("PlotType", {})

            if "ImprovementPlot" in plot_type_data:
                plot = BenchmarkConfig._create_plot_config(
                    PlotType.IMPROVEMENT.value,
                    plot_type_data["ImprovementPlot"]
                )
                plots.append(plot)

            elif "HypervolumePlot" in plot_type_data:
                plot = BenchmarkConfig._create_plot_config(
                    PlotType.HYPERVOLUME.value,
                    plot_type_data["HypervolumePlot"]
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
