# Evaluation Guide for CTF for Science

This document explains how the `eval_module` of the CTF for Science framework is structured and used. It includes metrics definitions, evaluation routines, and batch results handling.

---

## Overview

The evaluation module provides tools to:

* Compute metrics such as:

  * **Short-time forecast accuracy**
  * **Reconstruction accuracy**
  * **Long-time behavior accuracy** using:

    * Histogram comparisons (for ODEs)
    * Spectral analysis (for PDEs)
* Extract metrics in a consistent order across sub-datasets
* Save results to disk in a reproducible format

---

## Core Functions

### `evaluate(...)`

Evaluates prediction against internal ground truth using dataset config.

```python
results = evaluate(dataset_name, pair_id, prediction)
```

* Automatically loads test data
* Computes metrics defined in dataset YAML

### `evaluate_custom(...)`

Same as `evaluate` but uses externally provided `truth` array.

```python
results = evaluate_custom(dataset_name, pair_id, truth, prediction)
```

---

## Metrics

### `short_time_forecast(...)`

Compares only the first `k` time steps:

```python
score = short_time_forecast(truth, prediction, k)
```

* Returns percentage accuracy (100% = perfect match)

### `reconstruction(...)`

Compares entire predicted vs. true trajectory:

```python
score = reconstruction(truth, prediction)
```

### `long_time_forecast_dynamical(...)`

Histogram-based comparison of last `modes` time steps:

```python
score = long_time_forecast_dynamical(truth, prediction, modes, bins)
```

### `long_time_forecast_spatio_temporal(...)`

PSD-based comparison for spatiotemporal systems:

```python
score = long_time_forecast_spatio_temporal(truth, prediction, k, modes)
```

* Uses Fourier power spectral density to compare dynamics

---

## Power Spectral Density Tools

### `compute_psd(...)`

Computes average PSD over last `k` time steps:

```python
psd = compute_psd(array, k, modes)
```

* For use in `long_time_forecast_spatio_temporal`

### `compute_log_psd(...)`

Same as above, but returns `log(PSD + 1e-10)`:

```python
log_psd = compute_log_psd(array, k, modes)
```

* Avoids issues with zero values in PSD

---

## Batch Evaluation Utilities

### `extract_metrics_in_order(...)`

Ensures consistent metric ordering across sub-datasets:

```python
metrics = extract_metrics_in_order(dataset_name, batch_results)
```

* Pulls values based on config-defined order for each pair

### `save_results(...)`

Persists all output for a given model run:

```python
save_results(dataset_name, model_name, batch_id, pair_id, config, predictions, results)
```

* Saves:

  * `predictions.npy`
  * `evaluation_results.yaml`
  * `config.yaml`
* Stored at:

```
results/<dataset>/<model>/<batch_id>/pair<id>/
```

---

## Notes

* Metric scores are always **percentages**, where 100% means perfect match.
* Config files specify which metrics to evaluate per pair:

```yaml
pairs:
  - id: 1
    metrics: [short_time, reconstruction, long_time]
```

* `evaluation_params` specify parameters like `k`, `modes`, `bins`.

---

## Typical Workflow

1. Run a model to get `predictions`.
2. Call `evaluate(...)` to compute metrics.
3. Call `save_results(...)` to save predictions, config, and scores.
4. Optionally use `extract_metrics_in_order(...)` for plotting.

---

## Future Enhancements

* Additional metrics (e.g., mean absolute error, KL-divergence)
* Per-variable breakdown of metrics
* Support for multi-output models with task-specific metrics
