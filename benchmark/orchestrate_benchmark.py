import argparse
import logging

from benchmark_runner import BRISEBenchmarkRunner
from logger.default_logger import BRISELogConfigurator
from shared_tools import chown_files_in_dir
from poc_analyzer import main as poc_analyzer_main

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


def analyze(results_storage: str = "./results/serialized/"):
    """Run analyzer over produced experiment dumps."""
    try:
        logging.info("Running analyzer on dumps in %s" % results_storage)
        poc_analyzer_main()
        logging.info("Analyzer completed: ./results/benchmark_poc.html , ./results/benchmark_poc.csv")
    except FileNotFoundError as fnf_err:
        logging.warning("Analyzer skipped: %s" % fnf_err)
    except Exception as exception:
        logging.error("Analyzer failed: %s" % exception, exc_info=True)


def orchestrate(skip_analyzer: bool = False):
    run_benchmark()
    if not skip_analyzer:
        analyze(results_storage)
    else:
        logging.info("Skipping analyzer as requested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BRISE Benchmark orchestrator")
    parser.add_argument("--mode", choices=["analyse", "benchmark"], default="benchmark", help="Select run mode.")
    parser.add_argument("--skip-analyzer", action="store_true", help="Skip analyzer after benchmark run.")
    args = parser.parse_args()

    if args.mode == "analyse":
        analyze(results_storage)
    else:
        orchestrate(args.skip_analyzer)
