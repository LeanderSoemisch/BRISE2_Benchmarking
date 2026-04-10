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
    # Per-plot overrides (all optional)
    title: Optional[str] = None
    filter_conditions: List[Any] = field(default_factory=list)   # List[MatchCondition]
    plot_grouping: Optional[Any] = None                           # Optional[CustomGroupingConfig]

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
    speedup_factor: bool = True
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
class MatchCondition:
    """
    A single generic condition that checks one attribute of an experiment's metadata.

    ``path``     – dot-separated path into the experiment metadata dict produced by
                   ``ExperimentMetadata.extract()``, e.g. ``"problem_instance"`` or
                   ``"description.TaskConfiguration.Scenario.Hyperparameters"``.
    ``value``    – expected value (string equality / substring match when ``contains=True``).
    ``contains`` – if True, the extracted value is checked with ``value in extracted``
                   instead of strict equality.
    ``pattern``  – alternative to ``value``: a regex pattern matched against the extracted
                   string (takes precedence over ``value`` when set).
    """
    path: str
    value: Optional[str] = None
    contains: bool = False
    pattern: Optional[str] = None


@dataclass
class AutoGroupDimension:
    """
    Describes one dimension used when auto-grouping (no explicit rules).

    ``path``   – dot-separated path into experiment metadata (same as MatchCondition.path).
    ``label``  – human-readable prefix used in the generated group label.
                 Defaults to the last segment of ``path``.
    ``transform`` – optional: ``"basename"`` strips directory components from path-like
                    values (e.g. ``"scenarios/tsp/kroA100.tsp"`` → ``"kroA100.tsp"``).
    """
    path: str
    label: Optional[str] = None
    transform: Optional[str] = None   # "basename" | None


@dataclass
class CustomGroupingConfig:
    """
    Configuration for how experiments are grouped into legend lines within a plot.

    Grouping is driven by ``auto_group_by`` dimensions. Experiments that share
    the same extracted dimension values are placed in the same group.

    A config is considered *active* (``is_configured``) when it defines at
    least one auto-group dimension.
    """
    auto_group_by: List[AutoGroupDimension] = field(default_factory=list)

    @property
    def is_configured(self) -> bool:
        return bool(self.auto_group_by)

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
    known_optima: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _parse_metric_type(metric_description: str) -> str:
        return MetricType.TIME.value if 'time' in metric_description.lower() else MetricType.ITERATION.value

    @staticmethod
    def _parse_normalization_strategy(norm_strategy_data: Dict[str, Any]) -> str:
        if not isinstance(norm_strategy_data, dict):
            return NormalizationType.NONE.value

        if 'MinOverAll' in norm_strategy_data:
            return NormalizationType.MIN_OVER_ALL.value
        if 'MaxOverAll' in norm_strategy_data:
            return NormalizationType.MAX_OVER_ALL.value
        return NormalizationType.NONE.value

    @staticmethod
    def _create_plot_config(plot_type: str, plot_data: Dict[str, Any]) -> PlotConfig:
        objective_axis = plot_data.get("ObjectiveAxis", {})
        norm_strategy = BenchmarkConfig._parse_normalization_strategy(objective_axis.get("NormalizationStrategy", {}))

        filter_conditions = BenchmarkConfig._parse_match_conditions(
            plot_data.get("filterConditions", [])
        )

        # Per-plot grouping override (overrides global CustomGrouping for this plot)
        plot_grouping = None
        if "CustomGrouping" in plot_data:
            plot_grouping = BenchmarkConfig._parse_custom_grouping(plot_data.get("CustomGrouping", {}))

        is_box_plot = plot_type == "box_plot"
        if is_box_plot:
            if "MetricAxis" in plot_data:
                logger.warning("BoxPlot ignores MetricAxis because its x-axis is categorical (algorithm/test case groups)")
            metric_desc = "categorical groups"
            metric_label = "Algorithm"
            metric_scale = ScaleType.LINEAR.value
            metric_type = MetricType.ITERATION.value
        else:
            metric_axis = plot_data.get("MetricAxis", {})
            metric_desc = metric_axis.get("metricDescription", "iterations completed")
            metric_label = metric_axis.get("label", "iteration")
            metric_scale = metric_axis.get("scale", ScaleType.LINEAR.value)
            metric_type = BenchmarkConfig._parse_metric_type(metric_desc)

        return PlotConfig(
            plot_type=plot_type,
            metric_description=metric_desc,
            metric_label=metric_label,
            metric_scale=metric_scale,
            metric_type=metric_type,
            objectives_to_plot=objective_axis.get("objectivesToPlot", []),
            normalize=objective_axis.get("normalize", True),
            normalization_strategy=norm_strategy,
            objective_label=objective_axis.get("label", "Objective value"),
            objective_scale=objective_axis.get("scale", ScaleType.LINEAR.value),
            enable_grouping=plot_data.get("enableGrouping", False),
            title=plot_data.get("title"),
            filter_conditions=filter_conditions,
            plot_grouping=plot_grouping,
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

        # Parse KnownOptima once; forward to RegretAnalysis.optimum_per_objective
        known_optima: Dict[str, float] = {}
        for key, val in benchmark.get("KnownOptima", {}).items():
            try:
                known_optima[key] = float(val)
            except (TypeError, ValueError):
                pass

        comparative_metrics = BenchmarkConfig._parse_comparative_metrics(
            benchmark.get("ComparativeMetrics", {}),
            known_optima=known_optima,
        )

        return BenchmarkConfig(
            results_folder=folder,
            output_directory=output_dir,
            experiment_name=experiment.get("name", "BRISE Benchmark Report"),
            experiment_description=experiment.get("description", "Benchmark analysis results"),
            objectives_to_measure=experiment.get("objectivesToMeasure", []),
            plots=plots,
            table_config=table_config,
            comparative_metrics=comparative_metrics,
            known_optima=known_optima,
        )

    @staticmethod
    def _parse_custom_grouping(grouping_dict: Dict[str, Any]) -> Optional["CustomGroupingConfig"]:
        """Parse a CustomGrouping block.

        Returns a configured :class:`CustomGroupingConfig` when the block
        contains ``autoGroupBy`` dimensions, or ``None`` when absent/empty.
        An explicit ``"enabled"`` flag is not required.
        """
        if not grouping_dict:
            return None

        auto_dims = BenchmarkConfig._parse_auto_group_dimensions(grouping_dict.get("autoGroupBy", []))

        if not auto_dims:
            return None

        return CustomGroupingConfig(auto_group_by=auto_dims)

    @staticmethod
    def _parse_match_conditions(conditions: List[Dict[str, Any]]) -> List[MatchCondition]:
        parsed_conditions: List[MatchCondition] = []
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            parsed_conditions.append(MatchCondition(
                path=cond.get("path", ""),
                value=cond.get("value"),
                contains=cond.get("contains", False),
                pattern=cond.get("pattern"),
            ))
        return parsed_conditions

    @staticmethod
    def _parse_auto_group_dimensions(dimensions: List[Any]) -> List[AutoGroupDimension]:
        parsed_dimensions: List[AutoGroupDimension] = []
        for dim in dimensions:
            if isinstance(dim, str):
                parsed_dimensions.append(AutoGroupDimension(path=dim))
                continue
            if isinstance(dim, dict):
                path = dim.get("path", dim.get("name", ""))
                if not path:
                    continue
                parsed_dimensions.append(AutoGroupDimension(
                    path=path,
                    label=dim.get("label"),
                    transform=dim.get("transform"),
                ))
        return parsed_dimensions

    @staticmethod
    def _parse_comparative_metrics(comp_metrics_dict: Dict[str, Any],
                                    known_optima: Optional[Dict[str, float]] = None) -> ComparativeMetricsConfig:
        if not comp_metrics_dict:
            # Still create a RegretAnalysisConfig if we have known_optima to forward
            if known_optima:
                return ComparativeMetricsConfig(
                    regret_analysis=RegretAnalysisConfig(optimum_per_objective=known_optima)
                )
            return ComparativeMetricsConfig()

        comp_table_dict = comp_metrics_dict.get("ComparativeTable", {})
        comp_table_config = ComparativeTableConfig(
            experiment=comp_table_dict.get("experiment", True),
            baseline=comp_table_dict.get("baseline", True),
            normalized_improvement=comp_table_dict.get("normalizedImprovement", True),
            speedup_factor=comp_table_dict.get("speedupFactor", comp_table_dict.get("speedup_factor", True)),
            converged_at_iteration=comp_table_dict.get("convergedAtIteration", True),
            experiment_best=comp_table_dict.get("experimentBest", True),
            baseline_best=comp_table_dict.get("baselineBest", True),
            final_regret=comp_table_dict.get("finalRegret", False)
        ) if comp_table_dict else None

        show_summary_table = comp_metrics_dict.get("showSummaryTable", comp_metrics_dict.get("show_summary_table", True))

        regret_dict = comp_metrics_dict.get("RegretAnalysis", {})
        if regret_dict:
            explicit_per_obj = regret_dict.get("optimumPerObjective")
            merged_per_obj = dict(known_optima or {})
            if explicit_per_obj:
                merged_per_obj.update(explicit_per_obj)
            regret_config = RegretAnalysisConfig(
                known_optimum=regret_dict.get("knownOptimum"),
                optimum_per_objective=merged_per_obj or None,
                regret_type=regret_dict.get("regretType", ["iteration"])
            )
        elif known_optima:
            regret_config = RegretAnalysisConfig(optimum_per_objective=known_optima)
        else:
            regret_config = None

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
            show_summary_table=show_summary_table,
            comparative_table=comp_table_config,
            regret_analysis=regret_config,
            normalized_improvement=normalized_config,
            performance_profile=performance_config
        )
