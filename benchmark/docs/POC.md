# Benchmark Module Architecture

Date: 09 Nov 2025 (updated)

## Current Components

- `orchestrate_benchmark.py`
  - Provides `run_benchmark()`, `analyze()` and high-level `orchestrate()` combining both
  - CLI: `--mode benchmark|analyse`, optional `--skip-analyzer`.
  - Responsible for life-cycle scenario execution and analyzer run.

- `benchmark_runner.py`
  - `BRISEBenchmarkRunner` encapsulates benchmark scenarios and interaction with Main Node via `MainAPIClient`.
  - Scenario examples: `benchmark_test()`, `fill_db()`.
  - Produces `.pkl` experiment dumps under `./results/serialized/`.

- `poc_analyzer.py`
  - Modular analyzer pipeline (v1) => will be split into different files later.
    - DataLoader: load dumps
    - MetricExtractor: discover numeric objectives (Y1..YN), extract per-iteration series
    - Normalizer: apply chosen normalization (e.g. minimize over experiments)
    - PlotBuilder: robust axis scaling (quantile padding) and best-so-far computation
    - ReportBuilder: multi-tab HTML report (one tab per objective) + summary table + csv file
  - Features:
    - Multi-objective auto-discovery (numeric keys within configuration results)
    - Direction-aware improvement computation (minimize / maximize)
    - Clean table & plot rendering
  - Work in progress:
    - Auto-open HTML report
    - Refactoring (into multiple files and folders)
    - Enhance naming of test files and experiments for clarity

- `configs/benchmark_templates/benchmark_template.json` && `/benchmark/configs/benchmark_feature_model/benchmark_feature_model.wfl`
  - Analyzer configuration: results folder, preferred improvement objective, direction, normalization method, optional time metric.
  - Waffle feature model for configuration wizard
## Waffle Benchmark Schema
![Diagram 1 Description](feature_model/Attributed_Waffle_Benchmark_Feature_Model.png)

## Architecture Diagrams

### Simplified Overview (Recommended Starting Point)
- **[Detail_04c_Simplified_Analysis_Architecture.puml](presentation/Detail_04c_Simplified_Analysis_Architecture.puml)** ⭐ **Start here!** - Simplified component diagram showing the core POC analyzer architecture without legacy components. Perfect for understanding the system at a glance.

### Detailed Architecture
- **[Detail_04_Benchmark_Analysis_Architecture.puml](presentation/Detail_04_Benchmark_Analysis_Architecture.puml)** - Complete class diagram with all components, data models, and relationships (includes legacy analyzer)

### Flow Diagrams
- **[Detail_04b_Benchmark_Analysis_Flow.puml](presentation/Detail_04b_Benchmark_Analysis_Flow.puml)** - Detailed sequence diagram showing the interaction flow between analyzer components from loading to report generation
- **[Benchmark_Sequence_BRISE_High_Level_Flow.png](benchmark_workflow/Benchmark_Sequence_BRISE_High_Level_Flow.png)** - High-level benchmark execution flow
