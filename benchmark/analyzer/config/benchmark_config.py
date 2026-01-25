import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PlotType(Enum):
    IMPROVEMENT = 'improvement_plot'
    CUSTOM = 'custom_plot'


class MetricType(Enum):
    ITERATION = 'iteration'
    TIME = 'time'


class ScaleType(Enum):
    LINEAR = 'linear'
    LOG10 = 'log10'


class NormalizationType(Enum):
    MIN_OVER_ALL = 'min_over_all_experiments'
    MAX_OVER_ALL = 'max_over_all_experiments'
    NONE = 'none'


class OptimizationDirection(Enum):
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


class Constants:
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
    enable_grouping: bool = False

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
class RegretAnalysisConfig:
    known_optimum: Optional[float] = None
    optimum_per_objective: Optional[Dict[str, float]] = None
    regret_type: List[str] = field(default_factory=lambda: ["iteration"])


@dataclass
class NormalizedImprovementConfig:
    improvement_type: List[str] = field(default_factory=lambda: ["objective_value"])


@dataclass
class PerformanceProfileConfig:
    tau_max: float = 10.0
    tau_steps: int = 100
    objectives_to_profile: List[str] = field(default_factory=list)


@dataclass
class ComparativeTableConfig:
    experiment: bool = True
    baseline: bool = True
    normalized_improvement: bool = True
    converged_at_iteration: bool = True
    experiment_best: bool = True
    baseline_best: bool = True
    final_regret: bool = False


@dataclass
class ComparativeMetricsConfig:
    show_summary_table: bool = True
    comparative_table: Optional[ComparativeTableConfig] = None
    regret_analysis: Optional[RegretAnalysisConfig] = None
    normalized_improvement: Optional[NormalizedImprovementConfig] = None
    performance_profile: Optional[PerformanceProfileConfig] = None

    def is_active(self) -> bool:
        return any([
            self.regret_analysis is not None,
            self.normalized_improvement is not None,
            self.performance_profile is not None,
            self.comparative_table is not None
        ])

    def get_regret_types(self) -> List[str]:
        return self.regret_analysis.regret_type if self.regret_analysis else ["iteration"]

    def get_improvement_types(self) -> List[str]:
        return self.normalized_improvement.improvement_type if self.normalized_improvement else ["objective_value"]

    def get_tau_max(self) -> float:
        return self.performance_profile.tau_max if self.performance_profile else 10.0

    def get_tau_steps(self) -> int:
        return self.performance_profile.tau_steps if self.performance_profile else 100


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
    comparative_metrics: ComparativeMetricsConfig = field(default_factory=ComparativeMetricsConfig)

    @staticmethod
    def _parse_metric_type(metric_description: str) -> str:
        return MetricType.TIME.value if 'time' in metric_description.lower() else MetricType.ITERATION.value

    @staticmethod
    def _parse_normalization_strategy(norm_strategy_data: Dict[str, Any]) -> str:
        if 'MinOverAll' in norm_strategy_data:
            return NormalizationType.MIN_OVER_ALL.value
        elif 'MaxOverAll' in norm_strategy_data:
            return NormalizationType.MAX_OVER_ALL.value
        return NormalizationType.NONE.value

    @staticmethod
    def _create_plot_config(plot_type: str, plot_data: Dict[str, Any]) -> PlotConfig:
        metric_axis = plot_data.get("MetricAxis", {})
        objective_axis = plot_data.get("ObjectiveAxis", {})

        metric_desc = metric_axis.get("metricDescription", "iterations completed")
        norm_strategy = BenchmarkConfig._parse_normalization_strategy(objective_axis.get("NormalizationStrategy", {}))

        return PlotConfig(
            plot_type=plot_type,
            metric_description=metric_desc,
            metric_label=metric_axis.get("label", "iteration"),
            metric_scale=metric_axis.get("scale", ScaleType.LINEAR.value),
            metric_type=BenchmarkConfig._parse_metric_type(metric_desc),
            objectives_to_plot=objective_axis.get("objectivesToPlot", []),
            normalize=objective_axis.get("normalize", True),
            normalization_strategy=norm_strategy,
            objective_label=objective_axis.get("label", "Objective value"),
            objective_scale=objective_axis.get("scale", ScaleType.LINEAR.value),
            enable_grouping=plot_data.get("enableGrouping", False)
        )

    @staticmethod
    def from_json(cfg: Dict[str, Any]) -> "BenchmarkConfig":
        benchmark = cfg.get("Benchmark", {})

        folder = benchmark.get("Resources", {}).get("Folder", "./results/serialized/")
        output_dir = benchmark.get("Report", {}).get("outputDirectory", "./results/reports/")

        experiment = benchmark.get("Experiment", {})

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
        for plot_key in sorted(k for k in benchmark if k.startswith("Plot_")):
            plot_data = benchmark[plot_key]
            plot_type_data = plot_data.get("PlotType", {})

            if "ImprovementPlot" in plot_type_data:
                plots.append(BenchmarkConfig._create_plot_config(PlotType.IMPROVEMENT.value, plot_type_data["ImprovementPlot"]))
            elif "CustomPlot" in plot_type_data:
                plots.append(BenchmarkConfig._create_plot_config(PlotType.CUSTOM.value, plot_type_data["CustomPlot"]))
            elif "BoxPlot" in plot_type_data:
                plots.append(BenchmarkConfig._create_plot_config("box_plot", plot_type_data["BoxPlot"]))

        comparative_metrics = BenchmarkConfig._parse_comparative_metrics(benchmark.get("ComparativeMetrics", {}))

        return BenchmarkConfig(
            results_folder=folder,
            output_directory=output_dir,
            experiment_name=experiment.get("name", "BRISE Benchmark Report"),
            experiment_description=experiment.get("description", "Benchmark analysis results"),
            objectives_to_measure=experiment.get("objectivesToMeasure", []),
            plots=plots,
            table_config=table_config,
            comparative_metrics=comparative_metrics
        )

    @staticmethod
    def _parse_comparative_metrics(comp_metrics_dict: Dict[str, Any]) -> ComparativeMetricsConfig:
        if not comp_metrics_dict:
            return ComparativeMetricsConfig()

        comp_table_dict = comp_metrics_dict.get("ComparativeTable", {})
        comp_table_config = ComparativeTableConfig(
            experiment=comp_table_dict.get("experiment", True),
            baseline=comp_table_dict.get("baseline", True),
            normalized_improvement=comp_table_dict.get("normalizedImprovement", True),
            converged_at_iteration=comp_table_dict.get("convergedAtIteration", True),
            experiment_best=comp_table_dict.get("experimentBest", True),
            baseline_best=comp_table_dict.get("baselineBest", True),
            final_regret=comp_table_dict.get("finalRegret", False)
        ) if comp_table_dict else None

        regret_dict = comp_metrics_dict.get("RegretAnalysis", {})
        regret_config = RegretAnalysisConfig(
            known_optimum=regret_dict.get("knownOptimum"),
            optimum_per_objective=regret_dict.get("optimumPerObjective"),
            regret_type=regret_dict.get("regretType", ["iteration"])
        ) if regret_dict else None

        normalized_dict = comp_metrics_dict.get("NormalizedImprovement", {})
        normalized_config = NormalizedImprovementConfig(
            improvement_type=normalized_dict.get("improvementType", ["objective_value"])
        ) if normalized_dict is not None else None

        performance_dict = comp_metrics_dict.get("PerformanceProfile", {})
        performance_config = PerformanceProfileConfig(
            tau_max=performance_dict.get("tauMax", 10.0),
            tau_steps=performance_dict.get("tauSteps", 100),
            objectives_to_profile=performance_dict.get("objectivesToProfile", [])
        ) if performance_dict else None

        return ComparativeMetricsConfig(
            comparative_table=comp_table_config,
            regret_analysis=regret_config,
            normalized_improvement=normalized_config,
            performance_profile=performance_config
        )
