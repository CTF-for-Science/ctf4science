"""
Visualization Module for CTF models, generates visualizations of model predictions and evaluation metrics.
"""

import importlib.resources as pkg_resources
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import colormaps, patches

from ctf4science.data_module import _load_test_data, get_applicable_plots, load_nodes
from ctf4science.eval_module import compute_log_psd

file_dir = Path(__file__).parent
top_dir = Path(__file__).parent.parent

# Rectangles (x, y, width, height) to cover on contour plots, keyed by dataset name
MASKED_REGIONS = {
    "msfr": [(1.13, -0.94, 0.7, 1.88)],  # Blanket
}


def resolve_field_layout(
    n_columns: int,
    num_fields: int = 1,
    field_sizes: list[int] | None = None,
    field_names: list[str] | None = None,
) -> tuple[list[slice], list[str]]:
    r"""Split the spatial (column) axis into the column range and label of each field.

    Multi-field datasets store their fields side by side along the column axis,
    in the order given by ``metadata.fields``. Fields of unequal width (e.g.
    ``dynasty``) give their widths in ``metadata.field_sizes``. Without
    `field_sizes`, the columns are split into `num_fields` equal parts.

    Parameters
    ----------
    n_columns : int
        Number of columns (spatial points) in the data.
    num_fields : int, optional
        Number of fields (``evaluation_params.num_fields``). Default 1. Ignored
        when `field_sizes` is given, except that it must match ``len(field_sizes)``
        if it is not 1.
    field_sizes : list of int, optional
        Width of each field along the column axis (``metadata.field_sizes``).
    field_names : list of str, optional
        Name of each field (``metadata.fields``), used as plot labels. Defaults
        to ``"Field 1"``, ``"Field 2"``, etc.

    Returns
    -------
    slices : list of slice
        Column slice of each field.
    labels : list of str
        Label of each field.

    Raises
    ------
    ValueError
        If `field_sizes` does not sum to `n_columns`, contains non-positive
        entries, or disagrees with `num_fields`; if `n_columns` is not divisible
        by `num_fields` (equal split); or if `field_names` has the wrong length.
    """
    if field_sizes is not None:
        sizes = [int(x) for x in field_sizes]
        if num_fields not in (1, len(sizes)):
            raise ValueError(
                f"num_fields ({num_fields}) does not match the number of field_sizes ({len(sizes)}: {sizes})"
            )
        if any(x <= 0 for x in sizes):
            raise ValueError(f"field_sizes must be positive, got {sizes}")
        if sum(sizes) != n_columns:
            raise ValueError(f"field_sizes {sizes} sum to {sum(sizes)}, but the data has {n_columns} columns")
    else:
        if num_fields < 1:
            raise ValueError(f"num_fields must be >= 1, got {num_fields}")
        if n_columns % num_fields != 0:
            raise ValueError(
                f"Cannot split {n_columns} columns into {num_fields} equal fields; "
                "set metadata.field_sizes in the dataset YAML for unequal field widths"
            )
        sizes = [n_columns // num_fields] * num_fields

    bounds = np.cumsum([0] + sizes)
    slices = [slice(int(a), int(b)) for a, b in itertools.pairwise(bounds)]

    if field_names is not None:
        labels = [str(name) for name in field_names]
        if len(labels) != len(slices):
            raise ValueError(f"Got {len(labels)} field names {labels} for {len(slices)} fields")
    else:
        labels = [f"Field {ii + 1}" for ii in range(len(slices))]
    return slices, labels


class Visualization:
    r"""Generates visualizations of model predictions and evaluation metrics.

    Configuration is loaded from a YAML file (default or custom path) and stored
    in `config`.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a custom YAML config file. If ``None``, uses default from package.

    Raises
    ------
    FileNotFoundError
        If the custom config file does not exist.

    Notes
    -----
    **Class Methods:**

    **plot_trajectories(self, truth, predictions, labels=None, \*\*kwargs):**

    - Plot stacked trajectories for each variable, comparing truth and predictions.
    - Parameters:
        - truth : ndarray. Ground truth, shape (time_steps, variables).
        - predictions : list of ndarray. List of prediction arrays, each same shape as `truth`.
        - labels : list of str, optional. Labels for each prediction.
        - \*\*kwargs : Passed to config (e.g. figure_size).
    - Returns:
        - plt.Figure. The generated figure.
    - Raises ``ValueError`` if prediction shapes do not match `truth`.

    **plot_errors(self, errors, labels=None, \*\*kwargs):**

    - Plot error metrics over time or across sub-datasets.
    - Parameters:
        - errors : dict. Keys are metric names; values are lists of floats.
        - labels : list of str, optional. Labels for each error type.
        - \*\*kwargs : Custom config overrides.
    - Returns:
        - plt.Figure. The generated figure.
    - Raises ``ValueError`` if `errors` is empty, lengths differ, or labels count mismatch.

    **plot_histograms(self, truth, predictions, modes, bins, labels=None, \*\*kwargs):**

    - Plot histograms over the last `modes` time steps per variable.
    - Parameters:
        - truth : ndarray. Ground truth, shape (time_steps, variables).
        - predictions : list of ndarray. Same shape as `truth`.
        - modes : int. Number of last time steps to use.
        - bins : int. Number of histogram bins.
        - labels : list of str, optional. Labels for each prediction.
        - \*\*kwargs : Custom config overrides.
    - Returns:
        - plt.Figure. The generated figure.
    - Raises ``ValueError`` if modes > time_steps or shapes mismatch.

    **plot_psd(self, truth, predictions, k, modes, labels=None, \*\*kwargs):**

    - Plot log power spectral density over the last `k` time steps.
    - Parameters:
        - truth : ndarray. Ground truth, shape (time_steps, spatial_points).
        - predictions : list of ndarray. Same shape as `truth`.
        - k : int. Number of last time steps for PSD.
        - modes : int. Number of frequency modes to plot.
        - labels : list of str, optional. Labels for each prediction.
        - \*\*kwargs : Custom config overrides.
    - Returns:
        - plt.Figure. The generated figure.
    - Raises ``ValueError`` if k or modes are invalid or shapes mismatch.

    **plot_from_batch(self, dataset_name, pair_id, batch_id, plot_type='trajectories', \*\*kwargs):**

    - Load data from a batch directory and produce one of the supported plot types.
    - Parameters:
        - dataset_name : str. Dataset name (e.g. ``'ODE_Lorenz'``).
        - pair_id : int. Pair ID for the sub-dataset.
        - batch_id : str or path-like. Path to batch dir (predictions.npy, optionally evaluation_results.yaml).
        - plot_type : str, optional. One of ``'trajectories'``, ``'histograms'``, ``'psd'``, ``'errors'``, ``'2d_comparison'``, ``'spatial_averages'``, ``'contour2d'``. Default ``'trajectories'``.
        - \*\*kwargs : Passed to the underlying plot method.
    - Returns:
        - plt.Figure. The generated figure.
    - Raises ``FileNotFoundError`` if required files missing; ``ValueError`` if `plot_type` unsupported or data invalid.

    **generate_all_plots(self, dataset_name, batch_path, \*\*kwargs):**

    - Generate all applicable plot types for the dataset and save under each pair dir in `batch_path`.
    - Parameters:
        - dataset_name : str. Dataset name.
        - batch_path : str or path-like. Path to the batch directory (containing pair* subdirs).
        - \*\*kwargs : Passed to `plot_from_batch`.
    - Returns:
        - None.

    **save_figure_results(self, fig, dataset_name, model_name, batch_name, pair_id, plot_type, results_dir=None):**

    - Save the figure to the results directory under a visualizations subfolder.
    - Parameters:
        - fig : plt.Figure. The figure to save.
        - dataset_name : str. Name of the dataset.
        - model_name : str. Name of the model.
        - batch_name : str. Batch identifier.
        - pair_id : int. Sub-dataset identifier.
        - plot_type : str. Type of plot (e.g. ``'trajectories'``, ``'histograms'``).
        - results_dir : str or Path, optional. Base path for results; default is ``results/{dataset}/{model}/{batch}/pair{pair_id}/visualizations``.
    - Returns:
        - None.

    **plot_prediction(self, ax, data, vmin=None, vmax=None, show_ticks=True, show_xlabel=False, show_ylabel=False):**

    - Plot a 2D array on the given axes (e.g. for spatio-temporal data).
    - Parameters:
        - ax : matplotlib.axes.Axes. Axes to plot on.
        - data : ndarray. 2D array, shape (time_steps, spatial_dim).
        - vmin, vmax : float, optional. Color scale limits.
        - show_ticks : bool, optional. Whether to show axis ticks. Default True.
        - show_xlabel : bool, optional. Whether to show x-axis label. Default False.
        - show_ylabel : bool, optional. Whether to show y-axis label. Default False.
    - Returns:
        - matplotlib.image.AxesImage. The image from ``imshow``.

    **compare_prediction(self, truth, predictions, cbar_options=None, show_ticks=True, show_titles=True):**

    - Create a side-by-side comparison of truth, prediction(s), and error (2D data).
    - Parameters:
        - truth : ndarray. Ground truth, shape (time_steps, spatial_dim).
        - predictions : list of ndarray. Prediction arrays, same shape as `truth`.
        - cbar_options : dict, optional. Colorbar options (show, orientation, shrink, ticks, label).
        - show_ticks : bool, optional. Whether to show axis ticks. Default True.
        - show_titles : bool, optional. Whether to show subplot titles. Default True.
    - Returns:
        - plt.Figure. The comparison figure.

    **plot_spatial_averages(self, truth, predictions, num_fields, field_sizes=None, field_names=None, \*\*kwargs):**

    - Plot spatial averages over time for truth and predictions, one subplot per field.
    - Parameters:
        - truth : ndarray. Ground truth, shape (time_steps, spatial_dim), fields side by side along the columns.
        - predictions : list of ndarray. Prediction arrays, same shape as `truth`.
        - num_fields : int. Number of fields; columns are split equally when `field_sizes` is not given.
        - field_sizes : list of int, optional. Width of each field, for fields of unequal width.
        - field_names : list of str, optional. Field names used as labels.
        - \*\*kwargs : Custom config overrides.
    - Returns:
        - plt.Figure. The generated figure.
    - Raises ``ValueError`` if shapes mismatch or the field layout is inconsistent with the data.

    **plot_2d_contours(self, nodes, truth, predictions, num_fields, time_step=-1, levels=30, field_sizes=None, field_names=None, \*\*kwargs):**

    - Plot contours of truth and predictions at one time step, one column per field.
    - Parameters:
        - nodes : ndarray. Spatial node coordinates, shape (num_nodes, 2).
        - truth : ndarray. Ground truth, shape (time_steps, spatial_dim), fields side by side along the columns.
        - predictions : list of ndarray. Prediction arrays, same shape as `truth`.
        - num_fields : int. Number of fields; columns are split equally when `field_sizes` is not given.
        - time_step : int, optional. Time step to plot. Default -1 (last).
        - levels : int, optional. Number of contour levels. Default 30.
        - field_sizes : list of int, optional. Width of each field, for fields of unequal width.
        - field_names : list of str, optional. Field names used as column titles.
        - \*\*kwargs : Custom config overrides.
    - Returns:
        - plt.Figure. The generated figure.
    - Raises ``ValueError`` if shapes mismatch, the field layout is inconsistent with the data, or a field's width
      differs from the number of nodes.
    """

    def __init__(self, config_path: Path | str | None = None):
        r"""Initialize the visualizer with default or custom config path.

        Loads YAML config from package default or from `config_path` and
        stores it in `self.config`.
        """
        if config_path is None:
            with pkg_resources.open_text("ctf4science.config", "default_visualization_config.yaml") as f:
                self.config = yaml.safe_load(f)
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

    def plot_trajectories(
        self, truth: np.ndarray, predictions: list[np.ndarray], labels: list[str] | None = None, **kwargs
    ) -> plt.Figure:
        r"""Plot stacked trajectories for each variable, comparing truth and predictions.

        One subplot per variable; truth and each prediction are overlaid with
        configurable colors and linestyles from `self.config`.

        Parameters
        ----------
        truth : ndarray
            Ground truth data, shape (time_steps, variables).
        predictions : list of ndarray
            List of prediction arrays, each matching `truth` shape.
        labels : list of str, optional
            Labels for each prediction. Defaults to ``"Prediction 1"``, etc.
        \*\*kwargs
            Override config (e.g. ``figure_size``).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Raises
        ------
        ValueError
            If any prediction shape does not match `truth`.
        """
        config = {**self.config, **kwargs}
        time_steps, num_vars = truth.shape
        for pred in predictions:
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")

        labels = labels or [f"Prediction {i + 1}" for i in range(len(predictions))]

        fig, axes = plt.subplots(num_vars, 1, figsize=config.get("figure_size", (10, 2 * num_vars)), sharex=True)
        time = np.arange(time_steps)
        cmap = colormaps.get_cmap(config["colors"]["predictions"])
        colors = [cmap(i / max(1, len(predictions) - 1)) for i in range(len(predictions))]

        for i in range(num_vars):
            ax = axes[i] if num_vars > 1 else axes  # type: ignore[index]
            ax.plot(
                time,
                truth[:, i],
                label="Truth",
                color=config["colors"]["truth"],
                linestyle=config["linestyles"]["truth"],
            )
            for j, pred in enumerate(predictions):
                ax.plot(
                    time, pred[:, i], label=labels[j], color=colors[j], linestyle=config["linestyles"]["predictions"]
                )
            ax.set_ylabel(f"Variable {i + 1}" if num_vars > 3 else ["x", "y", "z"][i])
            ax.legend()

        axes[-1].set_xlabel("Time")  # type: ignore[index]
        plt.tight_layout()
        return fig

    def plot_errors(self, errors: dict[str, list[float]], labels: list[str] | None = None, **kwargs) -> plt.Figure:
        r"""Plot error metrics over time or across sub-datasets.

        Parameters
        ----------
        errors : dict
            Keys are metric names; values are lists of floats (e.g. ``{'short_time': [values]}``).
        labels : list of str, optional
            Labels for each error curve. Defaults to `errors` keys.
        \*\*kwargs
            Override config (e.g. ``figure_size``).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Raises
        ------
        ValueError
            If `errors` is empty, error lists have different lengths, or
            `labels` length does not match number of errors.
        """
        config = {**self.config, **kwargs}
        if not errors:
            raise ValueError("Errors dictionary cannot be empty")
        lengths = {len(err) for err in errors.values()}
        if len(lengths) > 1:
            raise ValueError("All error lists must have the same length")

        fig, ax = plt.subplots(figsize=config.get("figure_size", (10, 6)))
        labels = labels or list(errors.keys())
        if len(labels) != len(errors):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of errors ({len(errors)})")

        for i, (key, err) in enumerate(errors.items()):
            ax.plot(err, label=labels[i], color=config["colors"]["errors"][i % len(config["colors"]["errors"])])
        ax.set_xlabel("Time or Sub-Dataset")
        ax.set_ylabel("Error")
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_histograms(
        self,
        truth: np.ndarray,
        predictions: list[np.ndarray],
        modes: int,
        bins: int,
        labels: list[str] | None = None,
        **kwargs,
    ) -> plt.Figure:
        r"""Plot histograms of variables over the last `modes` time steps.

        One subplot per variable; truth and each prediction are histogrammed
        over the last `modes` steps.

        Parameters
        ----------
        truth : ndarray
            Ground truth, shape (time_steps, variables).
        predictions : list of ndarray
            Prediction arrays, same shape as `truth`.
        modes : int
            Number of last time steps to include.
        bins : int
            Number of histogram bins.
        labels : list of str, optional
            Labels for each prediction. Defaults to ``"Prediction 1"``, etc.
        \*\*kwargs
            Override config (e.g. ``figure_size``).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Raises
        ------
        ValueError
            If `modes` exceeds time steps or any prediction shape does not
            match `truth`.
        """
        config = {**self.config, **kwargs}
        time_steps, num_vars = truth.shape
        if modes > time_steps:
            raise ValueError(f"modes ({modes}) cannot exceed time_steps ({time_steps})")
        for pred in predictions:
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")

        labels = labels or [f"Prediction {i + 1}" for i in range(len(predictions))]

        fig, axes = plt.subplots(num_vars, 1, figsize=config.get("figure_size", (10, 2 * num_vars)))
        cmap = colormaps.get_cmap(config["colors"]["predictions"])
        colors = [cmap(i / max(1, len(predictions) - 1)) for i in range(len(predictions))]

        for i in range(num_vars):
            ax = axes[i] if num_vars > 1 else axes  # type: ignore[index]
            truth_last = truth[-modes:, i]
            ax.hist(truth_last, bins=bins, alpha=0.5, label="Truth", color=config["colors"]["truth"])
            for j, pred in enumerate(predictions):
                pred_last = pred[-modes:, i]
                ax.hist(pred_last, bins=bins, alpha=0.5, label=labels[j], color=colors[j])
            ax.set_title(f"Variable {i + 1}" if num_vars > 3 else ["x", "y", "z"][i])
            ax.legend()
        plt.tight_layout()
        return fig

    def plot_psd(
        self,
        truth: np.ndarray,
        predictions: list[np.ndarray],
        k: int,
        modes: int,
        labels: list[str] | None = None,
        **kwargs,
    ) -> plt.Figure:
        r"""Plot log power spectral density over the last `k` time steps.

        Uses `compute_log_psd` from the eval module; one line per series
        (truth and each prediction).

        Parameters
        ----------
        truth : ndarray
            Ground truth, shape (time_steps, spatial_points).
        predictions : list of ndarray
            Prediction arrays, same shape as `truth`.
        k : int
            Number of last time steps for PSD computation.
        modes : int
            Number of frequency modes to plot.
        labels : list of str, optional
            Labels for each prediction. Defaults to ``"Prediction 1"``, etc.
        \*\*kwargs
            Override config (e.g. ``figure_size``).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Raises
        ------
        ValueError
            If `k` > time_steps, `modes` > spatial_points, or any prediction
            shape does not match `truth`.
        """
        config = {**self.config, **kwargs}
        time_steps, N = truth.shape
        if k > time_steps:
            raise ValueError(f"k ({k}) cannot exceed time_steps ({time_steps})")
        if modes > N:
            raise ValueError(f"modes ({modes}) cannot exceed spatial points ({N})")
        for pred in predictions:
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")

        labels = labels or [f"Prediction {i + 1}" for i in range(len(predictions))]

        freqs = np.fft.fftshift(np.fft.fftfreq(N))[N // 2 : N // 2 + modes]
        fig, ax = plt.subplots(figsize=config.get("figure_size", (10, 6)))
        cmap = colormaps.get_cmap(config["colors"]["predictions"])
        colors = [cmap(i / max(1, len(predictions) - 1)) for i in range(len(predictions))]

        log_psd_truth = compute_log_psd(truth, k, modes)
        ax.plot(freqs, log_psd_truth, label="Truth", color=config["colors"]["truth"])

        for j, pred in enumerate(predictions):
            log_psd_pred = compute_log_psd(pred, k, modes)
            ax.plot(freqs, log_psd_pred, label=labels[j], color=colors[j])

        ax.set_xlabel("Frequency")
        ax.set_ylabel("Log PSD")
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_from_batch(
        self, dataset_name: str, pair_id: int, batch_id: str, plot_type: str = "trajectories", **kwargs
    ) -> plt.Figure:
        r"""Plot data from a batch directory for a given plot type.

        Loads test data and predictions from the batch dir; for ``'errors'``
        also loads ``evaluation_results.yaml``. Dispatches to the appropriate
        plot method.

        Parameters
        ----------
        dataset_name : str
            Dataset name (e.g. ``'ODE_Lorenz'``).
        pair_id : int
            Pair ID for the sub-dataset.
        batch_id : str or path-like
            Path to the batch directory (must contain ``predictions.npy``;
            ``evaluation_results.yaml`` required for ``'errors'``).
        plot_type : str, optional
            One of ``'trajectories'``, ``'histograms'``, ``'psd'``,
            ``'errors'``, ``'2d_comparison'``, ``'spatial_averages'``,
            ``'contour2d'``. Default ``'trajectories'``. The multi-field plots
            (``'spatial_averages'``, ``'contour2d'``) use
            ``evaluation_params.num_fields`` and, when present,
            ``metadata.field_sizes`` (unequal field widths) and
            ``metadata.fields`` (field labels) from the dataset YAML.
        \*\*kwargs
            Passed to the underlying plot method.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Raises
        ------
        FileNotFoundError
            If ``predictions.npy`` or (for ``'errors'``) ``evaluation_results.yaml`` is missing.
        ValueError
            If `plot_type` is unsupported or data is invalid (e.g. non-2D for ``'2d_comparison'``).
        """
        test_data = _load_test_data(dataset_name, pair_id)

        batch_dir = Path(batch_id)
        predictions_path = batch_dir / "predictions.npy"
        if not predictions_path.exists():
            raise FileNotFoundError(f"predictions.npy not found in {batch_id}")
        predictions = [np.load(predictions_path)]

        config_path = top_dir / "data" / dataset_name / f"{dataset_name}.yaml"
        with open(config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        eval_params = dataset_config.get("evaluation_params", {})
        metadata = dataset_config.get("metadata", {}) or {}
        # Field widths and names for the multi-field plots
        field_kwargs = {
            "field_sizes": metadata.get("field_sizes"),
            "field_names": metadata.get("fields"),
        }

        if plot_type == "trajectories":
            return self.plot_trajectories(test_data, predictions, labels=["Model Prediction"], **kwargs)
        elif plot_type == "histograms":
            modes = eval_params.get("modes", 1000)
            bins = eval_params.get("bins", 50)
            return self.plot_histograms(test_data, predictions, modes, bins, labels=["Model Prediction"], **kwargs)
        elif plot_type == "psd":
            k = eval_params.get("k_long", 20)
            modes = eval_params.get("modes", 100)
            return self.plot_psd(test_data, predictions, k, modes, labels=["Model Prediction"], **kwargs)
        elif plot_type == "errors":
            results_path = batch_dir / "evaluation_results.yaml"
            if not results_path.exists():
                raise FileNotFoundError(f"evaluation_results.yaml not found in {batch_id}")
            with open(results_path, "r") as f:
                errors = yaml.safe_load(f)
            return self.plot_errors(errors, **kwargs)
        elif plot_type == "2d_comparison":
            # Check that data is 2D
            for prediction in predictions:
                if len(test_data.shape) != 2 or len(prediction.shape) != 2:
                    raise ValueError(
                        f"2d_comparison requires 2D data, but shapes are {test_data.shape} and {predictions[0].shape}"
                    )
            return self.compare_prediction(test_data, predictions, **kwargs)
        elif plot_type == "spatial_averages":
            num_fields = eval_params.get("num_fields", 1)
            return self.plot_spatial_averages(test_data, predictions, num_fields, **{**field_kwargs, **kwargs})
        elif plot_type == "contour2d":
            num_fields = eval_params.get("num_fields", 1)

            # Load nodes for contour plotting (e.g. for MSFR)
            nodes = load_nodes(dataset_name)

            fig = self.plot_2d_contours(nodes, test_data, predictions, num_fields, **{**field_kwargs, **kwargs})

            # Cover regions that are not part of the fields (e.g. the MSFR blanket)
            add_masked_regions(fig, MASKED_REGIONS.get(dataset_name, []))

            return fig
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def generate_all_plots(self, dataset_name: str, batch_path: str, **kwargs) -> None:
        r"""Generate all applicable plot types for the dataset and save under each pair dir.

        Uses `get_applicable_plots(dataset_name)` to determine plot types;
        iterates over ``pair*`` subdirs of `batch_path` and saves figures as
        ``{plot_type}_plot.png`` in each pair directory.

        Parameters
        ----------
        dataset_name : str
            Dataset name (e.g. ``'ODE_Lorenz'``).
        batch_path : str or path-like
            Path to the batch directory containing ``pair*`` subdirectories.
        \*\*kwargs
            Passed to `plot_from_batch`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no visualizations are defined for the dataset (from data module).
        FileNotFoundError
            If required batch files are missing when plotting.
        """
        applicable_plots = get_applicable_plots(dataset_name)
        batch_dir = Path(batch_path)
        for pair_dir in batch_dir.glob("pair*"):
            pair_id = int(pair_dir.name.replace("pair", ""))
            for plot_type in applicable_plots:
                fig = self.plot_from_batch(dataset_name, pair_id, str(pair_dir), plot_type=plot_type, **kwargs)
                fig.savefig(pair_dir / f"{plot_type}_plot.png")
                plt.close(fig)

    def save_figure_results(
        self,
        fig: plt.Figure,
        dataset_name: str,
        model_name: str,
        batch_name: str,
        pair_id: int,
        plot_type: str,
        results_dir: str | Path | None = None,
    ) -> None:
        r"""Save the figure to the results directory under a visualizations subfolder.

        Writes ``{plot_type}.png`` into the visualizations folder. If
        `results_dir` is not given, uses
        ``results/{dataset_name}/{model_name}/{batch_name}/pair{pair_id}/visualizations``.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        dataset_name : str
            Name of the dataset.
        model_name : str
            Name of the model.
        batch_name : str
            Batch identifier.
        pair_id : int
            Sub-dataset identifier.
        plot_type : str
            Type of plot (e.g. ``'trajectories'``, ``'histograms'``). Used as filename base.
        results_dir : str or Path, optional
            Base directory for results; ``visualizations`` is appended. If ``None``, inferred from dataset/model/batch/pair_id.

        Returns
        -------
        None
        """
        if results_dir is None:
            results_dir = (
                top_dir / "results" / dataset_name / model_name / batch_name / f"pair{pair_id}" / "visualizations"
            )
        else:
            results_dir = Path(results_dir) / "visualizations"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"{plot_type}.png"
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved {plot_type} plot to {save_path}")

    def plot_prediction(self, ax, data, vmin=None, vmax=None, show_ticks=True, show_xlabel=False, show_ylabel=False):
        r"""Plot a 2D array on the given axes with optional color scale and labels.

        Uses ``imshow`` with colormap from config; extent is set from data shape.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on.
        data : ndarray
            2D array, shape (time_steps, spatial_dim).
        vmin : float, optional
            Minimum value for color scale.
        vmax : float, optional
            Maximum value for color scale.
        show_ticks : bool, optional
            Whether to show axis ticks. Default True.
        show_xlabel : bool, optional
            Whether to show x-axis label (spatial dimension). Default False.
        show_ylabel : bool, optional
            Whether to show y-axis label (time dimension). Default False.

        Returns
        -------
        matplotlib.image.AxesImage
            The image returned by ``imshow``.
        """
        cmap = colormaps.get_cmap(self.config["images"]["colormap"])
        extent = [0, data.shape[1], data.shape[0], 0]
        img = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)

        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        if show_xlabel:
            ax.set_xlabel("Spatial Dimension (x)")
        if show_ylabel:
            ax.set_ylabel("Time Dimension (t)")

        return img

    def compare_prediction(self, truth, predictions, cbar_options=None, show_ticks=True, show_titles=True):
        r"""Create a side-by-side comparison of truth, prediction(s), and error (2D data).

        One row per prediction: columns are test data, predicted data, and
        error. Shared color scale; optional colorbar and subplot titles.

        Parameters
        ----------
        truth : ndarray
            Ground truth, shape (time_steps, spatial_dim).
        predictions : list of ndarray
            Prediction arrays, each same shape as `truth`.
        cbar_options : dict, optional
            Colorbar options. May include ``show`` (bool), ``orientation``
            (``'horizontal'`` or ``'vertical'``), ``shrink`` (float),
            ``ticks`` (int or array-like), ``label`` (str).
        show_ticks : bool, optional
            Whether to show axis ticks. Default True.
        show_titles : bool, optional
            Whether to show subplot titles (Test Data, Predicted Data, Error Data). Default True.

        Returns
        -------
        matplotlib.figure.Figure
            The comparison figure.
        """
        figsize = self.config["figure_size"]

        default_cbar_options = {"show": True, "orientation": "horizontal", "shrink": 0.8}

        # Safely merge with user input
        if cbar_options is None:
            cbar_options = default_cbar_options
        else:
            cbar_options = {**default_cbar_options, **cbar_options}

        nr_predictions = len(predictions)
        fig, axs = plt.subplots(1 * nr_predictions, 3, figsize=figsize)

        # Convert axs to 2D array even when nr_predictions is 1
        if nr_predictions == 1:
            axs = np.array([axs])

        for i, prediction in enumerate(predictions):
            vmin, vmax = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
            img_test = self.plot_prediction(
                axs[i, 0], truth, vmin=vmin, vmax=vmax, show_ticks=show_ticks, show_xlabel=True, show_ylabel=True
            )
            self.plot_prediction(
                axs[i, 1], prediction, vmin=vmin, vmax=vmax, show_ticks=show_ticks, show_xlabel=True, show_ylabel=False
            )
            error_data = truth - prediction
            self.plot_prediction(
                axs[i, 2], error_data, vmin=vmin, vmax=vmax, show_ticks=show_ticks, show_xlabel=True, show_ylabel=False
            )

            if show_titles:
                axs[i, 0].set_title("Test Data")
                axs[i, 1].set_title("Predicted Data")
                axs[i, 2].set_title("Error Data")

        if cbar_options.get("show", True):
            cbar = fig.colorbar(
                img_test,  # type: ignore[arg-type]
                ax=axs,
                orientation=cbar_options.get("orientation", "horizontal"),
                shrink=cbar_options.get("shrink", 0.8),
            )

            ticks = cbar_options.get("ticks")
            if ticks is not None:
                if isinstance(ticks, (np.ndarray, list)):
                    cbar.set_ticks(ticks)  # type: ignore[arg-type]
                elif isinstance(ticks, int):
                    cbar.set_ticks(np.linspace(truth.min(), truth.max(), ticks))

            label = cbar_options.get("label")
            if label:
                cbar.set_label(label)  # type: ignore[arg-type]
        return fig

    def plot_spatial_averages(
        self,
        truth: np.ndarray,
        predictions: list[np.ndarray],
        num_fields: int,
        field_sizes: list[int] | None = None,
        field_names: list[str] | None = None,
        **kwargs,
    ) -> plt.Figure:
        r"""Plot spatial averages over time for truth and predictions, one subplot per field.

        Each subplot shows the mean over that field's columns.

        Parameters
        ----------
        truth : ndarray
            Ground truth, shape (time_steps, spatial_dim). Fields are stored side
            by side along the column axis.
        predictions : list of ndarray
            Prediction arrays, each same shape as `truth`.
        num_fields : int
            Number of fields to average. Without `field_sizes` the columns are
            split into `num_fields` equal parts.
        field_sizes : list of int, optional
            Width of each field along the column axis (``metadata.field_sizes``),
            for fields of unequal width. Must sum to ``truth.shape[1]``.
        field_names : list of str, optional
            Field names (``metadata.fields``) used as y-axis labels. Defaults to
            ``"Field 1"``, ``"Field 2"``, etc.
        \*\*kwargs
            Override config (e.g. ``figure_size``). Without ``figure_size``, the
            figure height scales with the number of fields.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure comparing spatial averages.

        Raises
        ------
        ValueError
            If a prediction shape does not match `truth`, or the field layout is
            inconsistent with the data (see `resolve_field_layout`).
        """
        config = {**self.config, **kwargs}
        field_slices, field_labels = resolve_field_layout(truth.shape[1], num_fields, field_sizes, field_names)
        num_fields = len(field_slices)
        time_steps = truth.shape[0]

        for pred in predictions:
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")

        labels = [f"Prediction {i + 1}" for i in range(len(predictions))]

        # The config's default figure_size does not scale with the number of fields,
        # so only an explicit override is used
        figsize = kwargs.get("figure_size", (10, 2 * num_fields))
        fig, axs = plt.subplots(num_fields, 1, figsize=figsize, sharex=True, squeeze=False)
        axs = axs[:, 0]
        time = np.arange(time_steps)
        cmap = colormaps.get_cmap(config["colors"]["predictions"])
        colors = [cmap(i / max(1, len(predictions) - 1)) for i in range(len(predictions))]

        for ax, cols, field_label in zip(axs, field_slices, field_labels):
            truth_avg = truth[:, cols].mean(axis=1)
            ax.plot(
                time, truth_avg, label="Truth", color=config["colors"]["truth"], linestyle=config["linestyles"]["truth"]
            )
            for j, pred in enumerate(predictions):
                pred_avg = pred[:, cols].mean(axis=1)
                ax.plot(time, pred_avg, label=labels[j], color=colors[j], linestyle=config["linestyles"]["predictions"])
            ax.set_ylabel(field_label)
            ax.legend()

        axs[-1].set_xlabel("Time")

        plt.tight_layout()
        return fig

    def plot_2d_contours(
        self,
        nodes: np.ndarray,
        truth: np.ndarray,
        predictions: list[np.ndarray],
        num_fields: int,
        time_step: int = -1,
        levels: int = 30,
        field_sizes: list[int] | None = None,
        field_names: list[str] | None = None,
        **kwargs,
    ) -> plt.Figure:
        r"""Plot contours of truth and predictions at one time step, one column per field.

        The first row shows the truth and each following row one prediction.
        Every field must have one value per node.

        Parameters
        ----------
        nodes : ndarray
            Spatial node coordinates, shape (num_nodes, 2).
        truth : ndarray
            Ground truth, shape (time_steps, spatial_dim). Fields are stored side
            by side along the column axis.
        predictions : list of ndarray
            Prediction arrays, each same shape as `truth`.
        num_fields : int
            Number of fields to plot. Without `field_sizes` the columns are
            split into `num_fields` equal parts.
        time_step : int, optional
            Index of the time step to plot. Default -1 (last).
        levels : int, optional
            Number of contour levels. Default 30.
        field_sizes : list of int, optional
            Width of each field along the column axis (``metadata.field_sizes``),
            for fields of unequal width. Must sum to ``truth.shape[1]``.
        field_names : list of str, optional
            Field names (``metadata.fields``) used as column titles. Defaults to
            ``"Field 1"``, ``"Field 2"``, etc.
        \*\*kwargs
            Override config (e.g. ``images``, ``figure_size``). Without
            ``figure_size``, the figure size scales with the number of fields
            and predictions.

        Returns
        -------
        matplotlib.figure.Figure
            The generated contour figure.

        Raises
        ------
        ValueError
            If a prediction shape does not match `truth`, the field layout is
            inconsistent with the data (see `resolve_field_layout`), or a field's
            width differs from the number of nodes.
        """
        config = {**self.config, **kwargs}
        field_slices, field_labels = resolve_field_layout(truth.shape[1], num_fields, field_sizes, field_names)
        num_fields = len(field_slices)
        for cols, field_label in zip(field_slices, field_labels):
            width = cols.stop - cols.start
            if width != nodes.shape[0]:
                raise ValueError(
                    f"Field '{field_label}' has {width} columns but there are {nodes.shape[0]} nodes, "
                    "contour plots need one value per node for every field"
                )

        for pred in predictions:
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")

        labels = [f"Prediction {i + 1}" for i in range(len(predictions))]

        nrows = len(predictions) + 1  # one row for truth, one for each prediction
        ncols = num_fields

        # The default figure_size from the config is too small for a grid of contours,
        # so only an explicit override is used
        figsize = kwargs.get("figure_size", (ncols * 5, nrows * 4))
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)

        for ii, cols in enumerate(field_slices):
            _min = truth[time_step, cols].min()
            _max = truth[time_step, cols].max()

            _min *= 0.9 if _min > 0 else 1.1
            _max *= 1.1 if _max > 0 else 0.9

            # A field that is zero everywhere gives equal levels, which tricontourf rejects
            if _min == _max:
                _min, _max = -1.0, 1.0

            _levels = np.linspace(_min, _max, levels)

            # Plot truth contour
            c = axs[0, ii].tricontourf(
                nodes[:, 0], nodes[:, 1], truth[time_step, cols], levels=_levels, cmap=config["images"]["colormap"]
            )
            fig.colorbar(c, ax=axs[0, ii], orientation="vertical", shrink=0.8, format="%.2f")

            # Plot prediction contours
            for j, pred in enumerate(predictions):
                c = axs[j + 1, ii].tricontourf(
                    nodes[:, 0], nodes[:, 1], pred[time_step, cols], levels=_levels, cmap=config["images"]["colormap"]
                )
                fig.colorbar(c, ax=axs[j + 1, ii], orientation="vertical", shrink=0.8, format="%.2f")

            # Set labels and title
            axs[0, ii].set_title(field_labels[ii])
            if ii == 0:
                axs[0, ii].set_ylabel("Truth")
                for j in range(len(predictions)):
                    axs[j + 1, ii].set_ylabel(labels[j])

        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        return fig


def add_masked_regions(fig: plt.Figure, regions: list[tuple[float, float, float, float]]) -> None:
    r"""Cover rectangular regions with white patches on every axes of a contour figure.

    Used for regions that are not part of the plotted fields, such as the
    MSFR blanket.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to modify in place.
    regions : list of tuple of float
        Rectangles as ``(x, y, width, height)`` in node coordinates, with
        ``(x, y)`` the lower-left corner (see `MASKED_REGIONS`).
    """
    for ax in fig.get_axes():
        for x, y, width, height in regions:
            rec = patches.Rectangle((x, y), width, height, linewidth=1, facecolor="white", edgecolor="black", zorder=3)
            ax.add_patch(rec)
