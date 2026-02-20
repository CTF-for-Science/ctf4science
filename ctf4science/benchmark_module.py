"""
Benchmark Module for CTF models, benchmarks a model with optimal hyperparameters for a given dataset and pair_id.

This module provides a systematic evaluation of CTF models against a hidden test set. It also assesses model stability by running models multiple times with different random seeds.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ctf4science.performance_module import PerformanceMonitor, measure_time

top_dir = Path(__file__).parent.parent


class ModelBenchmarker:
    """Benchmarks a model with optimal hyperparameters for a given dataset and pair_id.

    Runs multiple independent training and evaluation runs with different random
    seeds. Designed to be run from within each model directory.

    Parameters
    ----------
    config_path : str
        Path to the configuration file (must exist).
    num_runs : int, optional
        Number of independent evaluation runs to perform, by default ``5``.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    ValueError
        If dataset pair_id is not a single integer or list of one integer.

    Notes
    -----

    **Class Methods:**

    **run_benchmark():**

    - Run multiple benchmarking evaluations and save results. Runs all evaluations,
      computes statistics (mean/std) when 3+ runs succeed.
    - Returns:
        - Dict[str, Any] benchmark_results (model_name, dataset_name, pair_id,
          planned_num_runs, successful_runs, run_results, statistics, performance_summary, timestamp, output_file).

    **_construct_output_dir():**

    - Construct the output directory path for benchmark results.
    - Returns:
        - Path ``results/benchmark_results/{dataset_name}/{model_name}/pair_id_{pair_id}/{timestamp}/``.

    **_create_run_config(self, run_idx, seed):**

    - Create a configuration file for a specific run with a given seed.
    - Parameters:
        - run_idx : int. Index of the run (0-based).
        - seed : int. Random seed for this run.
    - Returns:
        - Path to the created config file for the run.

    **_run_single_evaluation(self, run_idx, seed):**

    - Run a single evaluation of the model.
    - Parameters:
        - run_idx : int. Index of the run.
        - seed : int. Random seed for this run.
    - Returns:
        - Dict[str, Any] run results (run_idx, seed, duration, config_path, results, success) or error info.

    **_find_and_load_results_for_run(self, run_idx):**

    - Find and load the results from the most recent run (for this pair_id).
    - Parameters:
        - run_idx : int. Index of the run.
    - Returns:
        - Dict[str, Any] evaluation results loaded from ``evaluation_results.yaml``.

    **_extract_run_results(self, all_runs):**

    - Extract run results for each successful run, keyed by run identifier.
    - Parameters:
        - all_runs : List[Dict]. List of all run results.
    - Returns:
        - Dict[str, Any] run results keyed by ``run_{n}_seed_{seed}``.

    **_calculate_statistics(self, all_runs):**

    - Calculate mean and standard deviation for all metrics; requires 3+ successful runs.
    - Parameters:
        - all_runs : List[Dict]. List of all run results.
    - Returns:
        - Dict[str, Any] metric means, stds, timing stats. Requires at least 3 successful runs.
    """

    def __init__(self, config_path: str, num_runs: int = 5):
        """Initialize the benchmarker."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load configuration
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Construct model name the same way as the models do in their run.py files
        if "method" in self.config["model"]:
            # DeepONet: use base name for directory lookup
            if self.config["model"]["name"] == "DeepONet":
                self.model_name = self.config["model"]["name"]
            # neural_ode: use the same format as run.py
            elif self.config["model"]["name"] == "neural_ode":
                self.model_name = f"{self.config['model']['name']}_{self.config['model']['method']}"
            else:
                self.model_name = f"{self.config['model']['method']}{self.config['model']['name']}"
        else:
            # (e.g., KAN_1)
            if "version" in self.config["model"]:
                self.model_name = f"{self.config['model']['name']}_{self.config['model']['version']}"
            else:
                self.model_name = self.config["model"]["name"]
        self.dataset_name = self.config["dataset"]["name"]

        # Handle pair_id which can be a list of one or a single integer
        pair_id_config = self.config["dataset"]["pair_id"]
        if isinstance(pair_id_config, list):
            if len(pair_id_config) != 1:
                raise ValueError(f"Benchmark module expects single pair_id, got: {pair_id_config}")
            self.pair_id = pair_id_config[0]
        else:
            self.pair_id = pair_id_config

        # Create output directory in results structure
        self.output_dir = self._construct_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Number of independent runs
        self.num_runs = num_runs

        print(f"Initialized benchmarker for {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Pair ID: {self.pair_id}")
        print(f"Number of runs: {self.num_runs}")
        print(f"Output directory: {self.output_dir}")

    def _construct_output_dir(self) -> Path:
        """Construct the output directory path for benchmark results."""
        # Get model name
        model_name = self.model_name

        # Get dataset name and pair IDs
        dataset_name = self.dataset_name
        pair_ids = str(self.pair_id)

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct path
        output_dir = (
            Path(__file__).parent.parent
            / "results"
            / "benchmark_results"
            / dataset_name
            / model_name
            / f"pair_id_{pair_ids}"
            / timestamp
        )

        return output_dir

    def _create_run_config(self, run_idx: int, seed: int) -> Path:
        """Create a configuration file for a specific run with a given seed."""
        # Create a copy of the config
        run_config = self.config.copy()

        # Set the seed and pair_id
        run_config["model"]["seed"] = seed
        run_config["dataset"]["pair_id"] = self.pair_id

        # Save config file
        config_filename = f"benchmark_config_run_{run_idx + 1}_seed_{seed}.yaml"
        config_path = self.output_dir / config_filename

        with open(config_path, "w") as f:
            yaml.dump(run_config, f, default_flow_style=False)

        return config_path

    def _run_single_evaluation(self, run_idx: int, seed: int) -> dict[str, Any]:
        """Run a single evaluation of the model."""
        print(f"  Run {run_idx + 1}/{self.num_runs} (seed: {seed})")

        # Create config for this run
        config_path = self._create_run_config(run_idx, seed)

        # Import and run the model's run.py
        try:
            import run  # type: ignore[import] # TODO: how to handle this better?

            def run_model():
                # Call the main function from run.py
                if hasattr(run, "main"):
                    # Most models have a main function that takes config_path
                    if "no_viz" in run.main.__code__.co_varnames:
                        run.main(str(config_path), no_viz=True)
                    else:
                        run.main(str(config_path))
                else:
                    raise AttributeError("run.py does not have a main function")

            # Measure execution time
            _result, duration = measure_time(run_model)

            # Find and load results for this specific run
            results_data = self._find_and_load_results_for_run(run_idx)

            return {
                "run_idx": run_idx,
                "seed": seed,
                "duration": duration,
                "config_path": str(config_path),
                "results": results_data,
                "success": True,
            }

        except Exception as e:
            print(f"    Run {run_idx + 1} FAILED: {e}")
            return {"run_idx": run_idx, "seed": seed, "error": str(e), "success": False}
        finally:
            # Clean up config file
            if config_path.exists():
                config_path.unlink()

    def _find_and_load_results_for_run(self, run_idx: int) -> dict[str, Any]:
        """Find and load the results from the most recent run (for this pair_id)."""
        # Look for results in the standard location
        results_base = top_dir / "results" / self.dataset_name / self.model_name

        if not results_base.exists():
            raise FileNotFoundError(f"Results directory not found: {results_base}")

        time.sleep(5)  # wait for the results to be written to the file

        # Find the most recent result directory (this run just completed)
        result_dirs = [d for d in results_base.iterdir() if d.is_dir()]
        if not result_dirs:
            raise FileNotFoundError(f"No result directories found in {results_base}")

        latest_result = max(result_dirs, key=lambda x: x.stat().st_mtime)

        # Find results for the pair_id
        pair_dir = latest_result / f"pair{self.pair_id}"
        if not pair_dir.exists():
            raise FileNotFoundError(f"Pair directory not found: {pair_dir}")

        eval_file = pair_dir / "evaluation_results.yaml"
        if not eval_file.exists():
            raise FileNotFoundError(f"Evaluation results file not found: {eval_file}")

        # Load and return results
        with open(eval_file) as f:
            results = yaml.safe_load(f)

        return results

    def run_benchmark(self) -> dict[str, Any]:
        """Run multiple benchmarking evaluations and save results."""
        print(f"\n=== Starting benchmark for {self.model_name} ===")

        # Initialize performance monitor
        perf_monitor = PerformanceMonitor(output_dir=str(self.output_dir / "performance"))
        perf_monitor.start_monitoring()

        # Use different seeds for each run
        # Generate seeds: start with predefined ones, then generate more if needed
        base_seeds = [42, 123, 2025, 777, 1337, 999, 2024, 314, 271, 1618]
        if self.num_runs <= len(base_seeds):
            seeds = base_seeds[: self.num_runs]
        else:
            # If more runs than base seeds, extend with generated seeds
            seeds = base_seeds + [1000 + i for i in range(self.num_runs - len(base_seeds))]

        all_runs = []
        successful_runs = 0

        # Run evaluations
        for run_idx in range(self.num_runs):
            seed = seeds[run_idx]

            try:
                run_result = self._run_single_evaluation(run_idx, seed)
                all_runs.append(run_result)

                if run_result["success"]:
                    successful_runs += 1
                    perf_monitor.record_run(f"run_{run_idx + 1}", run_result["duration"])
                    print(f"    Run {run_idx + 1} completed in {run_result['duration']:.2f}s")
                else:
                    print(f"    Run {run_idx + 1} failed")

            except Exception as e:
                print(f"    Run {run_idx + 1} FAILED: {e}")
                all_runs.append({"run_idx": run_idx, "seed": seed, "error": str(e), "success": False})

        # Stop performance monitoring
        perf_summary = perf_monitor.stop_monitoring()

        # Extract run results from successful runs
        run_results = self._extract_run_results(all_runs)

        # Calculate statistics only when we have 3+ successful runs
        statistics = {}
        if successful_runs == self.num_runs:
            if self.num_runs >= 3:
                print(f"\n✅ All {self.num_runs} runs successful! Calculating statistics...")
                statistics = self._calculate_statistics(all_runs)
            else:
                print(f"\n✅ All {self.num_runs} runs successful! Run results recorded (statistics require 3+ runs).")
        else:
            print(f"\n❌ Only {successful_runs} runs successful. Skipping statistics.")
            statistics = {"error": f"Only {successful_runs}/{self.num_runs} runs successful"}

        # Create comprehensive results
        benchmark_results = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "pair_id": self.pair_id,
            "config_path": str(self.config_path),
            "planned_num_runs": self.num_runs,
            "successful_runs": successful_runs,
            "actual_num_runs": len(all_runs),
            "run_results": run_results,
            "statistics": statistics,
            "performance_summary": perf_summary,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        results_file = self.output_dir / f"benchmark_results_{self.model_name}_pair{self.pair_id}.json"
        with open(results_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        # Add output file path to results
        benchmark_results["output_file"] = str(results_file)

        print(f"\nBenchmark completed: {successful_runs}/{self.num_runs} successful runs")
        print(f"Results saved to: {results_file}")

        return benchmark_results

    def _extract_run_results(self, all_runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract run results for each successful run, keyed by run identifier."""
        successful_runs = [run for run in all_runs if run.get("success", False) and "results" in run]

        if not successful_runs:
            return {}

        return {
            f"run_{run['run_idx'] + 1}_seed_{run['seed']}": {"results": run["results"], "duration": run["duration"]}
            for run in successful_runs
        }

    def _calculate_statistics(self, all_runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate mean and standard deviation for all metrics; requires 3+ successful runs."""
        successful_runs = [run for run in all_runs if run.get("success", False) and "results" in run]

        if len(successful_runs) < 3:
            return {"error": f"Need at least 3 successful runs for statistics, got {len(successful_runs)}"}

        statistics: dict[str, Any] = {}

        # Get all metric names from the first successful run
        metric_names = list(successful_runs[0]["results"].keys())

        # Calculate statistics for each metric
        for metric_name in metric_names:
            values = [run["results"][metric_name] for run in successful_runs if metric_name in run["results"]]
            if values:
                statistics[f"{metric_name}_mean"] = float(np.mean(values))
                statistics[f"{metric_name}_std"] = float(np.std(values))

        # Add timing statistics
        durations = [run["duration"] for run in successful_runs]
        statistics["timing"] = {"duration_mean": float(np.mean(durations)), "duration_std": float(np.std(durations))}

        return statistics


def main():
    """
    Run the ModelBenchmarker class from the command line.

    This function parses command-line arguments for benchmarking a CTF model
    using optimal hyperparameters. It instantiates the ModelBenchmarker class,
    performs the specified number of independent evaluation runs, and prints
    a summary of the benchmark results.

    Parameters
    ----------
    None

    Raises
    ------
    SystemExit
        If the benchmark fails due to an exception.

    Returns
    -------
    None

    Examples
    --------
    Run from command line:
        $ python benchmark_module.py --config config.yaml --num-evals 10

    Notes
    -----

    **Command-line Arguments:**

    --config : str
        Path to the model configuration file (required).
    --num-evals : int, optional
        Number of independent evaluation runs to perform (default is 5).

    **Side Effects:**

    Prints benchmark summary to stdout. On failure, prints an error message and exits with code 1.
    """
    parser = argparse.ArgumentParser(description="Benchmark a CTF model with optimal hyperparameters")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument(
        "--num-evals", type=int, default=5, help="Number of independent evaluation runs to perform (default: 5)"
    )

    args = parser.parse_args()

    try:
        benchmarker = ModelBenchmarker(args.config, num_runs=args.num_evals)
        results = benchmarker.run_benchmark()

        print("\nBenchmark completed!")
        print(f"Model: {results['model_name']}")
        print(f"Dataset: {results['dataset_name']}")
        print(f"Successful runs: {results['successful_runs']}/{results['planned_num_runs']}")
        print(f"Results saved to: {results.get('output_file', 'Unknown location')}")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
