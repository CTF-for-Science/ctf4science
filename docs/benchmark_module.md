# Benchmark Module

The Benchmark Module (`benchmark_module.py`) provides systematic evaluation of CTF models against a hidden test set. It also assesses model stability by running models multiple times with different random seeds.

---

## Overview

The `ModelBenchmarker` class orchestrates benchmarking by:

* Running multiple independent evaluations with different random seeds (default: 5, configurable)
* Computing statistical summaries (mean and standard deviation) when 3+ runs are successful
* Extracting individual run results for detailed analysis
* Monitoring wall-clock time performance during execution
* Saving results for analysis

---

## Usage

### Command Line Interface

Run benchmarking from within a model directory:

```bash
cd models/YourModel
python -m ctf4science.benchmark_module --config path/to/your/config.yaml
```

To specify the number of evaluation runs:

```bash
python -m ctf4science.benchmark_module --config path/to/your/config.yaml --num-evals 10
```

### Programmatic Usage

```python
from ctf4science.benchmark_module import ModelBenchmarker

# Default: 5 runs
benchmarker = ModelBenchmarker("path/to/config.yaml")
results = benchmarker.run_benchmark()

# Custom number of runs
benchmarker = ModelBenchmarker("path/to/config.yaml", num_runs=10)
results = benchmarker.run_benchmark()
```

---

## Configuration Requirements

The benchmark module requires a standard CTF configuration file:

```yaml
dataset:
  name: ODE_Lorenz
  pair_id: 1        # Single pair ID (not a list)

model:
  name: YourModel
  method: your_method  # Optional
```

### Parameters

* **`num_runs`** (default: 5): Number of independent evaluation runs to perform

---

## Output Structure

Benchmark results are saved in:

```
results/benchmark_results/
    {dataset_name}/
        {model_name}/
            pair_id_{pair_id}/
                {timestamp}/
                    benchmark_results_{model_name}_pair{pair_id}.json
```

### Results File Contents

The main results file contains:

```json
{
  "model_name": "YourModel",
  "dataset_name": "ODE_Lorenz",
  "pair_id": 1,
  "planned_num_runs": 5,
  "successful_runs": 5,
  "actual_num_runs": 5,
  "run_results": {
    "run_1_seed_42": {
      "results": {"short_time": 85.2},
      "duration": 45.1
    },
    "run_2_seed_123": {
      "results": {"short_time": 84.8},
      "duration": 44.9
    }
  },
  "statistics": {
    "short_time_mean": 85.2,
    "short_time_std": 2.1,
    "timing": {
      "duration_mean": 45.2,
      "duration_std": 2.8
    }
  },
  "performance_summary": {},
  "timestamp": "2025-01-XX..."
}
```

---

## Statistical Analysis

The benchmark module calculates statistics for all evaluation metrics:

* **Mean**: Average score across successful runs
* **Standard Deviation**: Measure of score variability
* **Timing Statistics**: Mean and standard deviation of execution times

**Success Criteria:**
* Requires at least 3 successful runs to calculate statistics
* Individual run failures are logged but don't stop the benchmark
* All successful runs are recorded in `run_results`

---

## Model Compatibility

The benchmark module works with any CTF model that follows the standard interface:

* Model directory must contain a `run.py` file
* `run.py` must have a `main(config_path)` function
* Model should save results in the standard CTF format

---

## Troubleshooting

### Common Issues

**"Config file not found"**
* Verify the config file path is correct

**"No result directories found"**
* Check that the model ran successfully
* Verify the model saves results in the expected format

**"Only X runs successful"**
* Review error logs for failed runs
* Check model stability and error handling

---

## Integration

The benchmark module integrates with CTF components:

* **Performance Module**: Uses `PerformanceMonitor` for system monitoring
* **Evaluation Module**: Relies on standard evaluation results format
* **Data Module**: Works with standard dataset loading
