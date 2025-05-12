# Visualization Guide for CTF for Science

This guide describes how to use the `Visualization` module within the CTF for Science framework to generate, customize, and save plots that help assess and understand model performance.

---

## Overview

The `Visualization` module provides tools to:

* Plot predicted vs. ground truth **trajectories**
* Compare **error metrics** across time or sub-datasets
* Visualize **histograms** of variable distributions
* Display **power spectral densities (PSDs)** for spatiotemporal data
* Perform **side-by-side 2D comparisons** of predictions vs. truth
* Automate plot generation for entire batch runs

These plots are saved under each run's `visualizations/` directory inside the `results/` folder.

---

## Usage

### 1. Import and Initialize

```python
from ctf4science.visualization_module import Visualization
viz = Visualization()  # Uses default config
```

To use a custom configuration:

```python
viz = Visualization(config_path='path/to/your/visualization_config.yaml')
```

---

## Plotting Functions

### `plot_trajectories(...)`

Plot time-series trajectories of each variable.

```python
fig = viz.plot_trajectories(truth, [predictions], labels=['Model'])
```

* Compares each prediction to the ground truth.
* One subplot per variable.

### `plot_errors(...)`

Visualize error metrics (e.g., short\_time, reconstruction) over time or sub-datasets.

```python
fig = viz.plot_errors(errors_dict)
```

* Accepts a dictionary of error lists: `{metric_name: [values]}`.

### `plot_histograms(...)`

Show histograms of variable values for the final `modes` time steps.

```python
fig = viz.plot_histograms(truth, [predictions], modes=1000, bins=50)
```

### `plot_psd(...)`

Display log-scaled Power Spectral Densities (PSD) for spatiotemporal systems.

```python
fig = viz.plot_psd(truth, [predictions], k=20, modes=100)
```

* Analyzes the frequency content of the final `k` time steps.

### `compare_prediction(...)`

2D image-style comparison of prediction, truth, and error.

```python
fig = viz.compare_prediction(truth, [predictions])
```

* Useful for image-like or spatiotemporal data.

---

## Batch and File-Based Plotting

### `plot_from_batch(...)`

Generates a single plot by reading prediction and evaluation results from disk.

```python
fig = viz.plot_from_batch(dataset_name, pair_id, batch_id, plot_type='trajectories')
```

* Supports `plot_type`: `trajectories`, `histograms`, `psd`, `errors`, `2d_comparison`
* Uses `results/<dataset>/<model>/<batch_id>/pair<id>/` to load data.

### `generate_all_plots(...)`

Generates and saves **all applicable plots** for a batch run.

```python
viz.generate_all_plots(dataset_name='ODE_Lorenz', batch_path='results/ODE_Lorenz/MyModel/batch_xyz')
```

* Plots are saved under each `pair*/visualizations/` directory.

### `save_figure_results(...)`

Saves a `matplotlib.Figure` to the appropriate results folder.

```python
viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type)
```

---

## Customization

Visualization appearance is governed by a config file (`default_visualization_config.yaml`). You can override settings like:

* `figure_size`
* Line `colors` and `linestyles`
* Image `colormap`

### Example override:

```python
fig = viz.plot_trajectories(truth, [predictions], figure_size=(12, 8), colors={'truth': 'black'})
```

---

## Output Locations

Plots are saved in:

```
results/<dataset>/<model>/<batch_id>/pair<id>/visualizations/
```

Each run will contain the visualizations that are defined for the dataset in the dataset.yaml file

* `trajectories.png`
* `histograms.png`
* `psd.png`
* `errors.png`
* `2d_comparison.png`

---

## Tips

* Use `generate_all_plots()` to automate full visualization.
* Make sure `predictions.npy` and `evaluation_results.yaml` exist before calling batch plotting.
* If using 2D comparisons, your data must be 2D arrays.
* Use Jupyter to call plotting functions interactively and tweak parameters.

---

## Additional Resources

* See `visualization_module.py` for source code
* Customize colors and layout using your own YAML configuration
* For evaluating metrics, refer to `evaluation.md` (planned)
* For dataset formats, refer to `datasets.md` (planned)
