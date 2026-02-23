"""
Evaluation Module for CTF models, provides evaluation metrics and routines for CTF datasets.

This module handles evaluation of CTF models against a hidden test set. It also assesses model stability by running models multiple times with different random seeds.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ctf4science.data_module import _load_test_data, get_config

file_dir = Path(__file__).parent
top_dir = Path(__file__).parent.parent

# Map (pair_id, metric_name) to E1–E12 index
pair_id_to_e = {
    (1, "short_time"): 1,
    (1, "long_time"): 2,
    (2, "reconstruction"): 3,
    (3, "long_time"): 4,
    (4, "reconstruction"): 5,
    (5, "long_time"): 6,
    (6, "short_time"): 7,
    (6, "long_time"): 8,
    (7, "short_time"): 9,
    (7, "long_time"): 10,
    (8, "short_time"): 11,
    (9, "short_time"): 12,
}


def extract_metrics_in_order(dataset_name: str, batch_results: dict[str, Any]) -> list[float]:
    r"""Extract metric values from batch results in the order defined by the dataset config.

    Pairs are processed in ascending ``pair_id``; within each pair, metrics
    follow the order in the dataset configuration.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'PDE_KS'``).
    batch_results : dict
        Dictionary loaded from batch results (e.g. ``batch_results.yaml``),
        with a ``pairs`` list keyed by ``pair_id`` and ``metrics``.

    Returns
    -------
    list of float
        Metric values in the order specified by the dataset config.

    Raises
    ------
    ValueError
        If a pair or metric is missing in `batch_results`.
    """
    # Load dataset configuration
    config = get_config(dataset_name)

    # Sort pairs by ID to ensure consistent order
    pairs = sorted(config["pairs"], key=lambda x: x["id"])

    metric_values = []
    for pair in pairs:
        pair_id = pair["id"]
        metrics_order = pair["metrics"]  # e.g., ["short_time", "long_time"]

        # Find the corresponding pair
        selected_pair = next((sd for sd in batch_results["pairs"] if sd["pair_id"] == pair_id), None)
        if selected_pair is None:
            raise ValueError(f"Pair ID {pair_id} not found in batch_results")

        # Extract metrics in the specified order
        for metric in metrics_order:
            value = selected_pair["metrics"].get(metric, None)
            if value is None:
                raise ValueError(f"Metric {metric} not found for pair ID {pair_id}")
            metric_values.append(value)

    return metric_values


def short_time_forecast(truth: np.ndarray, prediction: np.ndarray, k: int) -> float:
    r"""Compute the short-time forecast score (relative L2 over first k steps, as percentage).

    Uses the first `k` time steps of `truth` and `prediction`; score is
    ``100 * (1 - relative_L2_error)`` so that 100 is perfect.

    Parameters
    ----------
    truth : (T, F) ndarray
        Ground truth, shape (time steps, features).
    prediction : (T, F) ndarray
        Predicted data, same shape as `truth`.
    k : int
        Number of initial time steps to use.

    Returns
    -------
    float
        Short-time forecast score as a percentage (100 = perfect).
    """
    Est = np.linalg.norm(truth[:k, :] - prediction[:k, :], ord=2) / np.linalg.norm(truth[:k, :], ord=2)
    return float(100 * (1 - Est))


def reconstruction(truth: np.ndarray, prediction: np.ndarray) -> float:
    r"""Compute the reconstruction score (relative L2 over full trajectory, as percentage).

    Score is ``100 * (1 - relative_L2_error)`` so that 100 is perfect.

    Parameters
    ----------
    truth : (T, F) ndarray
        Ground truth, shape (time steps, features).
    prediction : (T, F) ndarray
        Predicted data, same shape as `truth`.

    Returns
    -------
    float
        Reconstruction score as a percentage (100 = perfect).
    """
    Est = np.linalg.norm(truth - prediction, ord=2) / np.linalg.norm(truth, ord=2)
    return float(100 * (1 - Est))


def long_time_forecast_dynamical(truth: np.ndarray, prediction: np.ndarray, modes: int, bins: int) -> float:
    r"""Compute the long-time forecast score for dynamical systems (histogram-based).

    Uses the last `modes` time steps per feature, builds L1-normalized histograms
    with `bins` bins, and compares truth vs prediction. Score is
    ``100 * (1 - relative_L1_error)`` so that 100 is perfect.

    Parameters
    ----------
    truth : (T, F) ndarray
        Ground truth, shape (time steps, features).
    prediction : (T, F) ndarray
        Predicted data, same shape as `truth`.
    modes : int
        Number of last time steps to use per feature.
    bins : int
        Number of histogram bins.

    Returns
    -------
    float
        Long-time forecast score as a percentage (100 = perfect).
    """
    num_features: int = truth.shape[1]
    truth_last: np.ndarray = truth[-modes:, :]
    pred_last: np.ndarray = prediction[-modes:, :]
    Elt_sum: float = 0
    for i in range(num_features):
        range_min = min(truth_last[:, i].min(), pred_last[:, i].min())
        range_max = max(truth_last[:, i].max(), pred_last[:, i].max())
        hist_truth, _ = np.histogram(truth_last[:, i], bins=bins, range=(range_min, range_max), density=False)
        hist_pred, _ = np.histogram(pred_last[:, i], bins=bins, range=(range_min, range_max), density=False)
        Elt_i: float = float(np.linalg.norm(hist_truth - hist_pred, ord=1) / np.linalg.norm(hist_truth, ord=1))
        Elt_sum += Elt_i
    Elt = Elt_sum / num_features
    return float(100 * (1 - Elt))


def compute_psd(array: np.ndarray, k: int, modes: int) -> np.ndarray:
    r"""Compute the averaged power spectral density over the last k time steps for the given modes.

    For each of the last `k` time steps, takes the FFT of the spatial dimension,
    computes the power spectrum, FFT-shifts it, and averages the PSD over those
    steps for `modes` Fourier modes starting from the center (zero) frequency.

    Parameters
    ----------
    array : (T, S) ndarray
        Data with shape (time_steps, spatial_points).
    k : int
        Number of last time steps to average over. Must not exceed time_steps.
    modes : int
        Number of Fourier modes from the center frequency. Must not exceed
        spatial_points.

    Returns
    -------
    ndarray
        Averaged PSD for the specified modes, shape ``(modes,)``.

    Raises
    ------
    ValueError
        If `k` exceeds time_steps or `modes` exceeds spatial_points.
    """
    # Extract dimensions of the input array
    time_steps, spatial_points = array.shape

    # Validate input parameters
    if k > time_steps:
        raise ValueError(f"k ({k}) exceeds time_steps ({time_steps})")
    if modes > spatial_points:
        raise ValueError(f"modes ({modes}) exceeds spatial_points ({spatial_points})")

    # Calculate the center index of the FFT-shifted spectrum
    center = spatial_points // 2

    # Initialize array to accumulate the PSD sum
    psd_sum = np.zeros(modes)

    # Compute PSD for each of the last k time steps and accumulate
    for i in range(k):
        # Compute FFT for the spatial data at the current time step
        fft = np.fft.fft(array[time_steps - k + i, :])
        # Calculate power spectrum (magnitude squared of FFT)
        ps = np.abs(fft) ** 2
        # Shift the power spectrum to center the zero-frequency component
        ps_shifted = np.fft.fftshift(ps)
        # Add the specified modes starting from the center to the sum
        psd_sum += ps_shifted[center : center + modes]

    # Compute the average PSD over k time steps
    psd_avg = psd_sum / k

    return psd_avg


def compute_log_psd(array: np.ndarray, k: int, modes: int) -> np.ndarray:
    r"""Compute the natural log of the averaged PSD over the last k time steps.

    Calls `compute_psd` and returns ``log(psd + 1e-10)`` to avoid log(0).
    See `compute_psd` for the PSD calculation.

    Parameters
    ----------
    array : (T, S) ndarray
        Data with shape (time_steps, spatial_points).
    k : int
        Number of last time steps to average over.
    modes : int
        Number of Fourier modes from the center frequency.

    Returns
    -------
    ndarray
        Averaged log-PSD for the specified modes, shape ``(modes,)``.

    Raises
    ------
    ValueError
        If `k` exceeds time_steps or `modes` exceeds spatial_points.
    """
    # Compute the averaged PSD using the existing function
    psd = compute_psd(array, k, modes)

    # Compute the log of the PSD, adding a small constant to avoid log(0)
    log_psd = np.log(psd + 1e-10)

    return log_psd


def long_time_forecast_spatio_temporal(truth: np.ndarray, prediction: np.ndarray, k: int, modes: int) -> float:
    r"""Compute the long-time forecast score for spatio-temporal systems (spectral).

    Compares averaged PSD of the last `k` time steps for `truth` and `prediction`
    over `modes` Fourier modes; score is ``100 * (1 - psd_error)``
    so that 100 is perfect.

    Parameters
    ----------
    truth : (T, S) ndarray
        Ground truth, shape (time steps, spatial points).
    prediction : (T, S) ndarray
        Predicted data, same shape as `truth`.
    k : int
        Number of last time steps to use for PSD.
    modes : int
        Number of Fourier modes from the center frequency.

    Returns
    -------
    float
        Long-time forecast score as a percentage (100 = perfect).
    """
    Pt = compute_psd(truth, k, modes)
    Pp = compute_psd(prediction, k, modes)
    Elt = np.linalg.norm(Pt - Pp, ord=2) / np.linalg.norm(Pt, ord=2)
    return float(100 * (1 - Elt))


def evaluate(
    dataset_name: str, pair_id: int, prediction: np.ndarray, metrics: list[str] | None = None
) -> dict[str, float]:
    r"""Evaluate the prediction using specified metrics; ground truth is loaded internally.

    Loads test data for `dataset_name` and `pair_id`, then computes the requested
    metrics. Use `evaluate_custom` to supply your own truth array.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``, ``'PDE_KS'``).
    pair_id : int
        ID of the train-test pair to use.
    prediction : ndarray
        Predicted data array (same shape as the test data for this pair).
    metrics : list of str, optional
        Metrics to compute (e.g. ``['short_time', 'long_time', 'reconstruction']``).
        If None, uses the pair's default metrics from config.

    Returns
    -------
    dict
        Mapping from metric name to computed score (float).

    Raises
    ------
    ValueError
        If `pair_id` is invalid, an unknown metric is requested, or the dataset
        long-time evaluation type is unknown.
    """
    # Retrieve the ground truth test data internally
    truth = _load_test_data(dataset_name, pair_id)

    # Evaluate
    results = evaluate_custom(dataset_name, pair_id, truth, prediction, metrics)

    # Return
    return results


def evaluate_custom(
    dataset_name: str,
    pair_id: int,
    truth: np.ndarray,
    prediction: np.ndarray,
    metrics: list[str] | None = None,
    flexible_k: bool = False,
) -> dict[str, float]:
    r"""Evaluate the prediction against a provided truth array using specified metrics.

    Uses the given `truth` and `prediction` arrays and the dataset config to
    determine evaluation parameters and long-time evaluation type. Use `evaluate`
    to load ground-truth test data internally instead.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``, ``'PDE_KS'``).
    pair_id : int
        ID of the train-test pair (used to select config and metrics).
    truth : ndarray
        Ground truth data array.
    prediction : ndarray
        Predicted data array, same shape as `truth`.
    metrics : list of str, optional
        Metrics to compute. If None, uses the pair's default metrics from config.
    flexible_k : bool, optional
        Whether to use a flexible k value. If True, the k value is min(timesteps, k) for the metric..
        If False, the k value is the value from the dataset config. Used for when hyperparameter
        optimization results in less timesteps than the k value in the dataset config.

    Returns
    -------
    dict
        Mapping from metric name to computed score (float).

    Raises
    ------
    ValueError
        If `pair_id` is invalid, an unknown metric is requested, or the dataset
        long-time evaluation type is unknown.
    """
    config = get_config(dataset_name)

    evaluation_params = config["evaluation_params"]
    pair = next((p for p in config["pairs"] if p["id"] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    default_metrics = pair["metrics"]
    metrics = default_metrics if metrics is None else metrics
    results = {}
    for metric in metrics:
        if metric == "short_time":
            if flexible_k:
                k_short = min(truth.shape[0], evaluation_params["k_short"])
                if k_short != evaluation_params["k_short"]:
                    print(
                        f"WARNING: flexible_k is True, using k_short={k_short} instead of {evaluation_params['k_short']} from {dataset_name} config"
                    )
            else:
                k_short = evaluation_params["k_short"]
            results["short_time"] = short_time_forecast(truth, prediction, k_short)
        elif metric == "long_time":
            long_time_eval_type = config["evaluations"]["long_time"]
            if long_time_eval_type == "histogram_L2_error":
                results["long_time"] = long_time_forecast_dynamical(
                    truth, prediction, evaluation_params["modes"], evaluation_params["bins"]
                )
            elif long_time_eval_type == "spectral_L2_error":
                if flexible_k:
                    k_long = min(truth.shape[0], evaluation_params["k_long"])
                    if k_long != evaluation_params["k_long"]:
                        print(
                            f"WARNING: flexible_k is True, using k_long={k_long} instead of {evaluation_params['k_long']} from {dataset_name} config"
                        )
                else:
                    k_long = evaluation_params["k_long"]
                results["long_time"] = long_time_forecast_spatio_temporal(
                    truth, prediction, k_long, evaluation_params["modes"]
                )
            else:
                raise ValueError(f"Unknown dataset long time evaluation type: {long_time_eval_type}")
        elif metric == "reconstruction":
            results["reconstruction"] = reconstruction(truth, prediction)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return results


def save_results(
    dataset_name: str,
    method_name: str,
    batch_id: str,
    pair_id: int | str,
    config: dict[str, Any],
    predictions: np.ndarray,
    results: dict[str, float] | None = None,
) -> Path:
    r"""Save configuration, predictions, and optional evaluation results for a run.

    Writes ``config.yaml``, ``predictions.npy``, and optionally
    ``evaluation_results.yaml`` under ``results/{dataset_name}/{method_name}/{batch_id}/pair{pair_id}/``.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    method_name : str
        Name of the method or model.
    batch_id : str
        Batch identifier and folder name for the batch run.
    pair_id : int or str
        Sub-dataset (pair) identifier.
    config : dict
        Configuration dictionary used for the run.
    predictions : ndarray
        Predicted data array to save.
    results : dict, optional
        Evaluation results (metric name -> score). If None, no
        ``evaluation_results.yaml`` is written.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the run results.
    """
    results_dir = top_dir / "results" / dataset_name / method_name / batch_id / f"pair{pair_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    np.save(results_dir / "predictions.npy", predictions)
    if results is not None:
        results_for_yaml = {key: float(value) for key, value in results.items()}
        with open(results_dir / "evaluation_results.yaml", "w") as f:
            yaml.dump(results_for_yaml, f)
    return results_dir


def evaluate_kaggle_csv(csv_path: str, dataset_name: str) -> dict[str, float]:
    r"""Evaluate the predictions from a Kaggle CSV file.

    Loads the CSV, groups rows by pair_id, converts each group to a (T, 3)
    prediction matrix, and runs the standard CTF evaluation for each pair.
    Returns per-pair metrics, E1–E12 in order, and their average (score).
    Only works when test data is available for the selected dataset.

    Parameters
    ----------
    csv_path : str
        Path to the Kaggle CSV file.
    dataset_name : str
        Dataset name used for config and ground-truth loading.

    Returns
    -------
    dict
        Computed metrics: keys ``pair_{id}_{metric}`` for each pair and metric,
        ``E1``–``E12`` for the ordered metric list, and ``average`` for the
        mean of E1–E12.

    Notes
    -----
    CSVs have the following format:

    id,pair_id,timestep,x,y,z
    1_0,1,0,value,value,value
    1_1,1,1,value,value,value
    1_2,1,2,value,value,value
    ...
    1_999,1,999,value,value,value
    2_0,2,0,value,value,value
    2_1,2,1,value,value,value
    ...
    9_999,9,999,value,value,value
    """
    # Load CSV and check required columns
    required = ["id", "pair_id", "timestep", "x", "y", "z"]
    df = pd.read_csv(csv_path)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV must contain columns {required}; missing: {col}")

    # Load config and get pairs and matrix shapes
    config = get_config(dataset_name)
    pairs = sorted(config["pairs"], key=lambda p: p["id"])
    metadata = config.get("metadata", {})
    matrix_shapes = metadata.get("matrix_shapes", {})

    # Initialize results dictionary
    all_pair_results: dict[int, dict[str, float]] = {}

    # Loop through pairs and evaluate
    for pair in pairs:
        # Get pair ID and expected shape
        pair_id = pair["id"]
        test_key = f"X{pair_id}test.mat"
        expected_shape = matrix_shapes.get(test_key)
        if expected_shape is None:
            raise ValueError(f"Expected shape not found for pair_id={pair_id} in {csv_path}")

        # Get subset of dataframe for this pair and sort by timestep
        sub_df = df.loc[df["pair_id"] == pair_id].sort_values("timestep")
        if sub_df.empty:
            raise ValueError(f"No rows found for pair_id={pair_id} in {csv_path}")

        # Convert to numpy array and check shapes
        pred_mat = sub_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        pred_t, pred_f = pred_mat.shape
        exp_t, exp_f = expected_shape
        if pred_t != exp_t or pred_f != exp_f:
            raise ValueError(
                f"pair_id={pair_id}: prediction shape ({pred_t}, {pred_f}) does not match "
                f"expected test shape {exp_t}, {exp_f}"
            )

        # Evaluate and store results
        pair_results = evaluate(dataset_name, pair_id, pred_mat, metrics=None)
        all_pair_results[pair_id] = pair_results

    # Map (pair_id, metric_name) to E1–E12 using pair_id_to_e
    results: dict[str, float] = {}
    for pair_id, metrics in all_pair_results.items():
        for metric_name, value in metrics.items():
            key = (pair_id, metric_name)
            if key not in pair_id_to_e:
                raise ValueError(f"(pair_id={pair_id}, metric={metric_name}) not found in pair_id_to_e")
            e_idx = pair_id_to_e[key]
            results[f"E{e_idx}"] = float(value)

    # Average of E1–E12
    e_keys_sorted = [f"E{i}" for i in range(1, 13)]
    metric_values = [results[k] for k in e_keys_sorted]
    results["average"] = float(np.mean(metric_values))

    return results
