# Benchmark
 
Service for running benchmark tests and performing a comparative analysis of experiment results.
Designed for understanding, interpreting and assessment of the experiments' results.

---
### Requirements
To use `benchmark` mode:
 - Run [main-node](../main_node/README.md "Main node Readme."), at least one [worker](../worker/README.md), [worker-service](../worker_service/README.md "Worker service Readme."), [event-service](../event_service/README.md). The easiest way to do so is by using the next `../brise.sh` command:  
    `../brise.sh up -m docker-compose -s main-node worker worker_service event_service`
 - Define the Domain Name for the `event-service` on the host-machine as IP of a machine where the event service is running. In case of running on the same machine set it the same as `localhost` in your environment.
 - NOTE: The benchmark looking for `event-service` on the 49153 port. For that reason be careful with changing AMQP port by using `../brise.sh`.
___
### Usage
__We strongly recommend to execute and control benchmark tests and analysis only in a dockerized way using provided `./init.sh` control script.__

##### Configure benchmark using Waffle (recommended)
1. Start the Waffle configuration wizard:
   ```bash
   ./init.sh waffle
   ```
   This will:
   - Start the Waffle service via docker-compose
   - Open your browser to http://localhost:8001/wizard/initialize/
   
2. In the Waffle wizard:
   - Copy and paste the content of `benchmark_template.wfl` into the text field
   - Click "Configure product manually"
   - Fill in the required fields:
     - `Benchmark.ExperimentSeries.Name`: Name for your experiment series
     - `Benchmark.ExperimentSeries.Description`: Description of your benchmark
     - `Benchmark.Resources.Folder`: Path to results folder (e.g., `./results/serialized/`)
     - ...
     - Configure plot settings for Improvement and Time plots as needed
   - Click "Download configured product" to get `configuration.json`
   
3. Save the downloaded file as `./benchmark/configuration.json`

4. Run the benchmark and analysis:
   ```bash
   ./init.sh up benchmark
   ```
   
   After completion, the results will automatically open in your browser!
   You can also manually open the report anytime with:
   ```bash
   ./init.sh show_report
   ```

##### Cleaning Up Generated Files

After and before running benchmarks, you can clean up all generated files (`.pkl`, `.csv`, `.html`, `.zip`):

```bash
./init.sh cleanup
```

This removes:
- All experiment dumps (`.pkl` files) from `./results/serialized/`
- All CSV files (benchmark results) from `./results/`
- All HTML reports from `./results/`
- All ZIP archives from `./results/`

You can also use the Python orchestrator directly:
```bash
python3 orchestrate_benchmark.py --mode cleanup
```

Or include cleanup automatically after each benchmark run:
```bash
python3 orchestrate_benchmark.py --cleanup
```

##### Exporting Plots as SVG

The generated HTML report includes "Export as SVG" buttons for each plot. Simply:
1. Open the report in your browser (`./init.sh show_report`)
2. Navigate to the plot you want to export
3. Click the "Export as SVG" button below the plot
4. The SVG file will be downloaded to your default downloads folder

This allows you to use high-quality vector graphics in presentations and publications.

## Configuration File Structure

The downloaded `configuration.json` should have this structure:

```json
{
  "Benchmark": {
    "Resources": {
      "Folder": "./results/serialized/"
    },
    "ExperimentSeries": {
      "Name": "MyBenchmark",
      "Description": "Description of the benchmark"
    },
    "Plots": {
      "Improvement": {
        "X": { ... },
        "Y": { ... }
      },
      "Time": {
        "X": { ... },
        "Y": { ... }
      }
    }
  }
}
```

See `configuration.json.example` for a complete example.

## Workflow Diagram

```
┌─────────────────┐
│  Start Waffle   │
│ ./init.sh waffle│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Open http://localhost:8001  │
│  /wizard/initialize/        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Paste benchmark_template.wfl│
│ Click "Configure manually"  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Fill in configuration fields│
│ - Name, Description         │
│ - Folder paths              │
│ - Plot settings             │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Download configuration.json │
│ Save to benchmark/ folder   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Run benchmark analysis     │
│  ./init.sh up benchmark     │
└─────────────────────────────┘
```

##### Plan and run benchmark tests
The general idea of writing benchmark scenario - automation of Experiment Description generation and execution.
1. Write your own benchmarking scenario as a separate method in class `BRISEBenchmarkRunner`.
    - The atomic step of scenario - you have generated complete valid Experiment Description and
    called `self.execute_experiment` passing this description as an argument.
    - You could mark your benchmarking scenario as `@benchmarkable` to calculate the number of Experiments
    that are going to be performed before their actual execution and check the overall process of Scenarios generation.
2. Add execution of this scenario in `run_benchmark` function of `orchestrate_benchmark.py` module.
3. Build an image, create a container and run the benchmark by calling `./init.sh up benchmark`
    * in case of failure you could restart benchmarking by running `./init.sh restart benchmark`.
    * benchmark enabled with __warm startup__ feature - in case of restart, the Experiments that were already performed
    and stored in storage folder will be detected and skipped. Be aware that this feature fully relies on a content of
    the Experiment Description - benchmarking script calculates hash of Experiment that is going to be executed and
    checks if Experiment Dump(s) with this hash is(are) already in a storage. If you change base Experiment Description content
    between benchmarking it will not work.

##### ~~Run analysis~~ Not available in 2.6.0
1. Put Experiment dumps in folder `./results/serialized/` (if not exists - create it).
2. Build an image, create container and run the analysis by calling `./init.sh up analyse`:
3. Use the browser to open the reports files. Reports files are stored in `./results/reports` directory.
NOTE: By default the build_detailed_report() method provided in `BRISEBenchmarkAnalyser`, but you could add your own analyser.
___
### Structure
- `./init.sh` provides control commands. For more information `./init.sh help`
- `./orchestrate_benchmark.py` is the logical entry point. Used to perform actual benchmark tests or for result analysis: in this case it combines a template with actual figures and generates a report file.
- `./benchmark_runner.py` - module with functionality for running benchmark tests.
- ~~`./benchmark_analyser.py` - module with functionality to perform benchmark results analysis.~~ is not available in 2.6.0
- `./shared_tools.py` - storage with helper tools.

###### Benchmark part
The benchmark part realized in two logical entities `MainAPIClient` and `BRISEBenchmarkRunner` located in `benchmark/benchmark_runner.py` module.

BRISEBenchmarkRunner is a main logical class that constructs Experiment Descriptions according to user defined scenario 
and executes them with BRISE using an instance of BRISE API client class MainAPIClient.

Responsibilities:
- `BRISEBenchmarkRunner`: Class for building and running benchmarking scenarios.
During initialization step it also initializes Main node API client (using provided URL address).
`BRISEBenchmarkRunner` class internally stores the Experiment Description, loaded on class instantiation
(the `Resources` folder with all available Descriptions will be copied inside of working directory on Container creation).

- `MainAPIClient`: Performs communication process with Main node. `BRISEBenchmarkRunner` class relies on main API client class
`MainAPIClient` (benchmark_runner.py module), the `perform_experiment` method that encapsulates communication logic
(starts BRISE with provided Experiment Description, waits for the `termination` event and downloads the Experiment dump
file or terminates execution after timeout).

###### ~~Analysis part~~ Is not available in 2.6.0
- `./results/serialized` default folder for storing dump of experiments.
- `./results/reports` output folder for report files
- `./templates` template for the report file, provides slots for the atomic figures.
- `./plots` code, generating atomic figures and tables:
  - Main metrics of the experiment. (`table.py`)
  - Improvements for the best result on each iteration of the experiment (`improvements.py`)
  - Statistical distribution for all and average results on each configuration in the experiment. (`box_statistic.py`)
  - Average results and measurement repeats for experiments. Statistical significance and invariability. Effect of the repeater (`repeat_vs_avg.py`)
  - Experiments configurations (`exp_config.py`)
___
### Play Around with Code

Benchmark module currently can be extended and modified with the dependencies listed in the [environment.yml](environment.yml) file.