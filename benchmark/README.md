# BRISE Benchmark Module

A comprehensive benchmarking and comparative analysis system for optimization algorithms. 
This module enables automated experiment evaluation, statistical analysis, and interactive visualization of optimization performance.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Workflows](#workflows)
- [Plot Types](#plot-types)
- [Comparative Analysis](#comparative-analysis)
- [Advanced Topics](#advanced-topics)
- [Commands](#commands)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
- Docker and docker-compose
- BRISE services running (main-node, worker, event-service)
- Python 3.7+ (for local development)

### Run a Complete Benchmark

```bash
# 1. Configure using Waffle (recommended)
./init.sh waffle

# 2. Run benchmark and analysis
./init.sh up benchmark

# 3. View results (auto-opens in browser)
./init.sh show_report

# 4. Clean up when done
./init.sh cleanup
```

### Analysis Only (If execution results already exist)

```bash
# Analyze existing experiment dumps
./init.sh up analyse
```

---

## Features

### Core Capabilities

#### **Multi-Objective Optimization Analysis**
- Automatic detection of all numeric objectives (Y1, Y2, Y3, ...)
- Direction-aware analysis (minimize or maximize)
- Per-objective visualization and metrics
- Aggregate performance profiles across objectives

#### **Visualization Suite**
- **Improvement Plots**: Track best-so-far progression over iterations
- **Time-Based Plots**: Performance evolution over wall-clock time
- **Box Plots**: Statistical distribution comparison across algorithms
- **Performance Profiles**: Algorithm ranking across multiple problems
- **Regret Analysis**: Distance from known optimum over time/iterations

#### **Comparative Analysis**
- Interactive baseline selection from executed experiments
- Normalized improvement metrics (objective value, time, iterations)
- Performance profiles across multiple test cases
- Regret analysis (iteration-based and time-based)
- Statistical distribution comparison via box plots

#### **Advanced Metrics**
- **Normalized Improvement**: 
  - Objective value improvement
  - Time-to-target speedup
  - Iteration-to-target speedup
- **Regret Analysis**:
  - Iteration-based regret
  - Time-based regret
- **Performance Profiles**: Cross-problem algorithm ranking

#### **Interactive Reports**
- Multi-tab HTML reports
- SVG export for all plots
- Sortable, filterable tables
- CSV data export
- Automatic browser preview

---

## Architecture

### Module Structure

```
benchmark/
├── orchestrate_benchmark.py    # Main entry point
├── runner/
│   └── benchmark_runner.py     # Experiment execution engine
│
├── analyzer/                   # Analysis pipeline
│   ├── config/                 # Configuration management
│   │   └── benchmark_config.py
│   │
│   ├── data_pipeline/          # Data processing
│   │   ├── experiment_loader.py
│   │   ├── experiment_parser.py
│   │   ├── metric_extractor.py
│   │   └── data_processor.py
│   │
│   ├── visualization/          # Plot and report generation
│   │   ├── plot_generator.py
│   │   ├── table_generator.py
│   │   └── report_generator.py
│   │
│   ├── orchestration/          # High-level coordination
│   │   ├── benchmark_analyzer.py
│   │   ├── comparative_integration.py
│   │   └── comparative_orchestrator.py
│   │
│   └── comparison/             # Comparative analysis
│       ├── README.md           # Detailed comparison docs
│       ├── baseline_manager.py
│       ├── comparison_processor.py
│       ├── comparative_metrics.py
│       └── ...
│
├── configs/                    # Configuration templates
│   ├── benchmark_templates/
│   │   ├── benchmark_template.json
│   │   ├── comparative_benchmark_template.json
│   │   └── ...
│   └── benchmark_feature_model/
│       └── benchmark_feature_model.wfl
│
└── results/                    # Output directory
    ├── serialized/             # Experiment dumps (.pkl)
    └── reports/                # Generated reports (HTML, CSV, ZIP)
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **orchestrate_benchmark.py** | CLI entry point, workflow coordination |
| **benchmark_runner.py** | Execute experiments via BRISE API |
| **experiment_loader.py** | Load .pkl files from disk |
| **experiment_parser.py** | Validate and structure experiment data |
| **metric_extractor.py** | Auto-discover objectives, extract trajectories |
| **data_processor.py** | Normalize, transform, compute best-so-far |
| **plot_generator.py** | Create all plot types (improvement, box, etc.) |
| **table_generator.py** | Generate summary statistics tables |
| **report_generator.py** | Assemble multi-tab HTML reports |
| **benchmark_analyzer.py** | Orchestrate entire analysis pipeline |
| **comparison/** | Baseline execution and comparative metrics |

---

## Configuration

### Using Waffle (Recommended)

Waffle provides a visual configuration wizard:

```bash
./init.sh waffle
```

Then:
1. Open http://localhost:8001/wizard/initialize/
2. Paste `benchmark_feature_model.wfl` content
3. Click "Configure product manually"
4. Fill in fields and download `configuration.json`
5. Save to `benchmark/configuration.json`

### Manual Configuration

See `configs/benchmark_templates/` for examples.

#### Basic Configuration

```json
{
  "Benchmark": {
    "Report": {
      "outputDirectory": "./results/reports/"
    },
    "Experiment": {
      "name": "My Benchmark",
      "description": "Optimization algorithm comparison",
      "objectivesToMeasure": ["Y1", "Y2", "Y3"]
    },
    "Table": {
      "task": true,
      "model": true,
      "iterations": true,
      "finalBestValue": true,
      "runtime": true
    },
    "Plot_0": {
      "PlotType": {
        "ImprovementPlot": { ... }
      }
    }
  }
}
```

### Configuration Fields

#### Experiment Settings
- `name`: Benchmark name
- `description`: Description for reports
- `objectivesToMeasure`: List of objectives to analyze

#### Plot Settings
- `PlotType`: One of `ImprovementPlot`, `CustomPlot`, `BoxPlot`
- `enableGrouping`: Group multiple runs of same test case
- `normalize`: Apply normalization
- `objectivesToPlot`: Which objectives to include

#### Baseline Settings
- `RandomSearchBaseline.samplingSize`: Number of random samples
- `GridSearchBaseline.gridResolution`: Grid points per dimension

#### Comparative Metrics
- `RegretAnalysis.regretType`: `["iteration"]`, `["time"]`, or both
- `NormalizedImprovement.improvementType`: Types of improvement to calculate
- `PerformanceProfile.objectivesToProfile`: Objectives for profile

---

## Workflows

### 1. Standard Benchmark Workflow

```
┌─────────────────┐
│ Configure       │
│ (Waffle/Manual) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Experiments │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate .pkl   │
│ dumps           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Results │
│ (analyzer)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HTML Report     │
│ (auto-opens)    │
└─────────────────┘
```

### 2. Analysis-Only Workflow

```
┌─────────────────┐
│ Existing .pkl   │
│ files           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ./init.sh up    │
│ analyse         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HTML Report     │
│ (auto-opens)    │
└─────────────────┘
```

### 3. Comparative Benchmark Workflow

```
┌─────────────────┐
│ Configure with  │
│ Baselines       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Experiments │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execute         │
│ Baselines       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Comparative     │
│ Analysis        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report with     │
│ Comparisons     │
└─────────────────┘
```

---

## Plot Types

### 1. Improvement Plot
**Purpose**: Track optimization progress over iterations

**Features**:
- Best-so-far trajectory
- Multiple algorithms on same plot
- Baseline overlays (random/grid search)
- Automatic Y-axis scaling

**Use Case**: See how quickly algorithms converge

**Example**:
```json
"Plot_0": {
  "PlotType": {
    "ImprovementPlot": {
      "Type": "improvement_plot",
      "enableGrouping": false,
      "MetricAxis": {
        "metricDescription": "iterations completed",
        "label": "Iteration",
        "scale": "linear"
      },
      "ObjectiveAxis": {
        "objectivesToPlot": ["Y1", "Y2"],
        "normalize": true,
        "label": "Objective Value"
      }
    }
  }
}
```

### 2. Box Plot
**Purpose**: Statistical distribution comparison

**Features**:
- Quartiles, median, mean, outliers
- Algorithm-to-algorithm comparison
- Baseline inclusion
- Variance analysis

**Use Case**: Compare algorithm robustness and consistency

**Key Insight**: Narrow boxes = consistent performance

**Example**:
```json
"Plot_1": {
  "PlotType": {
    "BoxPlot": {
      "Type": "box_plot",
      "enableGrouping": false,
      "ObjectiveAxis": {
        "objectivesToPlot": ["Y1"]
      }
    }
  }
}
```

### 3. Custom Plot 
**Purpose**: Performance over wall-clock time

**Features**:
- Time-based X-axis (seconds)
- Accounts for varying iteration costs
- Real-world performance view

**Use Case**: When iterations have different computational costs

### 4. Performance Profile
**Purpose**: Algorithm ranking across multiple problems

**Features**:
- Cumulative distribution of performance ratios
- Cross-problem comparison
- Statistical significance

**Use Case**: Determine best overall algorithm

**Auto-Generated**: When multiple test cases exist

### 5. Regret Analysis Plots
**Purpose**: Distance from known optimum

**Features**:
- Iteration-based regret
- Time-based regret
- Convergence rate analysis

**Use Case**: Validate algorithm quality when optimum is known

---

## Comparative Analysis

### Overview

Comparative analysis evaluates optimization algorithms by comparing their performance against user-selected baselines. The system computes normalized improvement, speedup factors, performance profiles, and regret metrics.

**See [`analyzer/comparison/README.md`](analyzer/comparison/README.md) for detailed metric definitions.**

---

### Interactive Baseline Selection

The benchmark module now features **dynamic baseline selection**, allowing you to choose any executed experiment as a comparison baseline.

#### Workflow

1. **Execute experiments** - Run your test cases:
   ```bash
   ./init.sh up benchmark
   ```

2. **Interactive selection** - A browser interface opens automatically, showing all executed experiments with their metadata (objectives measured, iterations completed, etc.)

3. **Select baselines** - Choose one or more experiments to serve as baselines for comparison

4. **Automatic analysis** - The system computes all comparative metrics using your selected baselines

5. **View report** - The HTML report displays normalized improvement, performance profiles, regret analysis, and comparison tables

#### Configuration

Enable comparative analysis by adding `ComparativeMetrics` to your configuration:

```json
{
  "Benchmark": {
    "ComparativeMetrics": {
      "ComparativeTable": {
        "experiment": true,
        "baseline": true,
        "normalizedImprovement": true,
        "convergedAtIteration": true,
        "experimentBest": true,
        "baselineBest": true,
        "finalRegret": true
      },
      "NormalizedImprovement": {
        "improvementType": ["objective_value", "time_to_target", "iteration_to_target"]
      },
      "RegretAnalysis": {
        "knownOptimum": 0.0,
        "optimumPerObjective": {
          "Y1": 0.0
        },
        "regretType": ["iteration", "time"]
      },
      "PerformanceProfile": {
        "tauMax": 5.0,
        "tauSteps": 50,
        "objectivesToProfile": ["Y1", "Y2", "Y3"]
      }
    }
  }
}
```
#### Benefits

- **Flexible**: Use any experiment as baseline
- **Iterative**: Compare new versions against previous runs
- **Multi-baseline**: Compare against multiple references simultaneously
- **Persistent**: Selection is saved for subsequent analyses
- **No re-execution**: Change baseline without rerunning experiments

#### Use Cases

**Algorithm Evolution**:
- Run version 1.0 of your algorithm
- Run improved version 2.0
- Select 1.0 as baseline to measure improvement

**Multi-algorithm Comparison**:
- Execute algorithms A, B, and C
- Select A and B as baselines
- Analyze C against both simultaneously

**Domain Baselines**:
- Include domain-specific reference implementations
- Use production system results as baseline
- Compare against known good configurations
---

#### Metrics

**Normalized Improvement**: How much better than baseline
- `objective_value`: Quality improvement
- `time_to_target`: Speed improvement (time)
- `iteration_to_target`: Speed improvement (iterations)

**Regret**: Distance from known optimum
- `iteration`: Regret over iterations
- `time`: Regret over time

**Performance Profile**: Algorithm ranking
- Across multiple test cases
- Statistical robustness

### Example Output

**Comparative Table**:
```
Experiment    Baseline      NI (Obj)  NI (Time)  NI (Iter)  Converged
test_case_0   random-search 1.40      0.03       -0.17      27
test_case_0   grid-search   1.65      -0.01      -3.38      27
```

**Normalized Improvement Plots**: Bar charts showing improvement factors
**Regret Plots**: Convergence to optimum over time/iterations
**Performance Profile**: Cumulative distribution curves

---

### Normalization Strategies

**MinOverAll**: Scale by minimum across all experiments
```
normalized = (value - min_all) / (max_all - min_all)
```

**MaxOverAll**: Scale by maximum across all experiments
```
normalized = value / max_all
```

**Use Case**: Compare objectives with different scales

### Grouping

**Purpose**: Aggregate multiple runs of same test case

**Effect**:
- Min/max bands show run variability
- Mean line shows average behavior
- Better statistical robustness

**Enable**: Set `enableGrouping: true` in plot config

### Data Format

Experiment dumps (`.pkl`) must contain:
- `measured_configurations`: List of evaluated configurations
- `results`: Dict mapping objective names to values
- `start_time`, `end_time`: Timestamps for time-based metrics

### Output Files

**HTML Report**: `results/reports/benchmark_report.html`
- Multi-tab interface
- Embedded interactive plots
- Sortable tables

**CSV Files**:
- `benchmark_all_objectives.csv`: Combined data
- `benchmark_objective_Y1.csv`: Per-objective data

**ZIP Archive**: `benchmark_all_tables.zip`
- All CSV files bundled

---

## Commands

### Basic Commands

| Command | Description |
|---------|-------------|
| `./init.sh up benchmark` | Execute benchmark and generate report |
| `./init.sh analyse` | Analyze existing experiments (no execution) |
| `./init.sh show_report` | Open latest report in browser |
| `./init.sh cleanup` | Remove all generated files (.pkl, .csv, .html, .zip) |
| `./init.sh cleanup_report` | Remove reports and baseline selection only |
| `./init.sh waffle` | Start Waffle configuration wizard |

## Contributing

When adding new features:

1. Update feature model (`.wfl`)
2. Add config parsing in `benchmark_config.py`
3. Implement feature in appropriate module
4. Add tests and documentation
5. Update README