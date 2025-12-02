import argparse
import json
import logging
import os
import sys
from pathlib import Path

from runner.benchmark_runner import BRISEBenchmarkRunner
from logger.default_logger import BRISELogConfigurator
from util.shared_tools import chown_files_in_dir, cleanup_benchmark_results

# Add main_node to path for unpickling experiment objects
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_NODE_PATH = str(PROJECT_ROOT / 'main_node')
if MAIN_NODE_PATH not in sys.path:
    sys.path.insert(0, MAIN_NODE_PATH)

from analyzer.config import BenchmarkConfig
from analyzer.orchestration import BenchmarkAnalyzer

BRISELogConfigurator()  # Configuring logging

host_event_service = "event-service"
port_event_service = 49153
results_storage = "./results/serialized/"

def run_benchmark():
    """Run the benchmark scenarios and produce dumps under results_storage."""
    # Container creation performs --volume on `./results/` folder. Change wisely results_storage.
    try:
        runner = BRISEBenchmarkRunner(host_event_service, port_event_service, results_storage)
        try:
            # ---    Add User defined benchmark scenarios execution below  ---#
            # --- Possible variants: benchmark_test, fill_db ---#
            runner.fill_db()

            # --- Helper method to move outdated experiments from `./results` folder ---#
            #runner.move_redundant_experiments(location=runner.results_storage + "repeater_outdated/")
        except Exception as exception:
            logging.error("Benchmarking interrupted: %s" % exception, exc_info=True)
        finally:
            runner.main_api_client.stop_main()
            runner.main_api_client.stop_client()
            chown_files_in_dir(results_storage)
            logging.info("Ownership of dump files changed, exiting.")
    except Exception as exception:
        logging.error("Unable to create BRISEBenchmarkRunner: %s" % exception, exc_info=True)


def analyze(
    results_storage: str = "./results/serialized/",
    output_html: str = "./results/reports/benchmark_report.html",
    output_csv: str = "./results/reports/benchmark_all_objectives.csv"
):
    """Run analyzer over produced experiment dumps.

    Args:
        results_storage: Path to folder with experiment dumps
        output_html: Path for output HTML report
        output_csv: Path for combined CSV output
    """
    try:
        # Check for configuration.json (from Waffle) first, then fall back to benchmark_template.json
        config_path = "./configuration.json" if os.path.exists("./configuration.json") else "./configs/benchmark_templates/benchmark_template_with_hypervolume.json"
        logging.info(f"Running analyzer on dumps in {results_storage} using config: {config_path}")

        with open(config_path, 'r') as f:
            config = BenchmarkConfig.from_json(json.load(f))

        analyzer = BenchmarkAnalyzer(config)
        analyzer.analyze(output_html, output_csv)

        logging.info(f"Analyzer completed: {output_html} , {output_csv}")
    except FileNotFoundError as fnf_err:
        logging.warning("Analyzer skipped: %s" % fnf_err)
    except Exception as exception:
        logging.error("Analyzer failed: %s" % exception, exc_info=True)


def orchestrate(skip_analyzer: bool = False, cleanup_before_run: bool = False):
    if cleanup_before_run:
        logging.info("Cleaning up generated files...")
        cleanup_benchmark_results(results_storage.rstrip('/serialized/'))
        logging.info("Cleanup completed.")

    run_benchmark()
    if not skip_analyzer:
        analyze(results_storage)
    else:
        logging.info("Skipping analyzer as requested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BRISE Benchmark orchestrator")
    parser.add_argument("--mode", choices=["analyse", "benchmark", "cleanup"], default="benchmark", help="Select run mode.")
    parser.add_argument("--skip-analyzer", action="store_true", help="Skip analyzer before benchmark run.")
    parser.add_argument("--cleanup", action="store_true", help="Clean up all generated files before benchmark run.")
    args = parser.parse_args()

    if args.mode == "analyse":
        analyze(results_storage)
    elif args.mode == "cleanup":
        cleanup_benchmark_results(results_storage.rstrip('/serialized/'))
    else:
        orchestrate(args.skip_analyzer, args.cleanup)
