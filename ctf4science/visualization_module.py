"""
Visualization Module for CTF models, generates visualizations of model predictions and evaluation metrics.
"""

import importlib.resources as pkg_resources
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import colormaps

from ctf4science.data_module import _load_test_data, get_applicable_plots
from ctf4science.eval_module import compute_log_psd

file_dir = Path(__file__).parent
top_dir = Path(__file__).parent.parent


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
        - plot_type : str, optional. One of ``'trajectories'``, ``'histograms'``, ``'psd'``, ``'errors'``, ``'2d_comparison'``. Default ``'trajectories'``.
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

        labels = labels if labels else [f"Prediction {i + 1}" for i in range(len(predictions))]

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
        labels = labels if labels else list(errors.keys())
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

        labels = labels if labels else [f"Prediction {i + 1}" for i in range(len(predictions))]

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

        labels = labels if labels else [f"Prediction {i + 1}" for i in range(len(predictions))]

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
            ``'errors'``, ``'2d_comparison'``. Default ``'trajectories'``.
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
            vmin, vmax = min(truth.min(), prediction[i].min()), max(truth.max(), prediction.max())
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
