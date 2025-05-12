import os
import yaml
import numpy as np
from pathlib import Path
from ctf4science.data_module import get_config, _load_test_data
from typing import Dict, Any, Optional, List

file_dir = Path(__file__).parent
top_dir = Path(__file__).parent.parent

def extract_metrics_in_order(dataset_name: str, batch_results: Dict[str, Any]) -> List[float]:
    """
    Extract metric values from batch_results in the order defined by the dataset configuration.

    Args:
        dataset_name (str): Name of the dataset (e.g., "PDE_KS").
        batch_results (Dict[str, Any]): Dictionary loaded from batch_results.yaml.

    Returns:
        List[float]: List of metric values in the order specified by the dataset config.

    Raises:
        ValueError: If a pair or metric is missing in batch_results.
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
    """
    Compute the short-time forecast error.

    This function calculates the relative L2 norm of the difference between the first k time steps
    of the truth and prediction arrays, scaled to a percentage.

    Args:
        truth (np.ndarray): Ground truth data array with shape (time steps, features).
        prediction (np.ndarray): Predicted data array with shape (time steps, features).
        k (int): Number of initial time steps to consider.

    Returns:
        float: Short-time forecast score as a percentage.
    """
    Est = np.linalg.norm(truth[:k, :] - prediction[:k, :], ord=2) / np.linalg.norm(truth[:k, :], ord=2)
    return float(100 * (1 - Est))

def reconstruction(truth: np.ndarray, prediction: np.ndarray) -> float:
    """
    Compute the reconstruction error.

    This function calculates the relative L2 norm of the difference between the entire truth and
    prediction arrays, scaled to a percentage.

    Args:
        truth (np.ndarray): Ground truth data array with shape (time steps, features).
        prediction (np.ndarray): Predicted data array with shape (time steps, features).

    Returns:
        float: Reconstruction score as a percentage.
    """
    Est = np.linalg.norm(truth - prediction, ord=2) / np.linalg.norm(truth, ord=2)
    return float(100 * (1 - Est))

def long_time_forecast_dynamical(truth: np.ndarray, prediction: np.ndarray, modes: int, bins: int) -> float:
    """
    Compute the long-time forecast error for dynamical systems.

    This function calculates the error based on histograms of the last 'modes' time steps for each
    feature, comparing the L-1 normalized histograms of truth and prediction.

    Args:
        truth (np.ndarray): Ground truth data array with shape (time steps, features).
        prediction (np.ndarray): Predicted data array with shape (time steps, features).
        modes (int): Number of last time steps to consider.
        bins (int): Number of bins for the histograms.

    Returns:
        float: Long-time forecast score as a percentage.
    """
    num_features = truth.shape[1]
    truth_last = truth[-modes:, :]
    pred_last = prediction[-modes:, :]
    Elt_sum = 0
    for i in range(num_features):
        range_min = min(truth_last[:, i].min(), pred_last[:,i].min())
        range_max = max(truth_last[:, i].max(), pred_last[:,i].max())
        hist_truth, _ = np.histogram(truth_last[:,i], bins=bins, range=(range_min, range_max), density=False)
        hist_pred, _ = np.histogram(pred_last[:, i], bins=bins, range=(range_min, range_max), density=False)
        Elt_i = np.linalg.norm(hist_truth - hist_pred, ord=1) / np.linalg.norm(hist_truth, ord=1)
        Elt_sum += Elt_i
    Elt = Elt_sum / num_features
    return float(100 * (1 - Elt))

def compute_psd(array: np.ndarray, k: int, modes: int) -> np.ndarray:
    """
    Compute the averaged power spectral density (PSD) over the last k time steps for the specified number of modes.

    This function calculates the PSD for each of the last k time steps by performing a Fast Fourier Transform (FFT),
    computing the power spectrum, shifting it to center the zero-frequency component, and averaging the PSD over these
    time steps for the specified number of modes starting from the center frequency.

    Args:
        array (np.ndarray): Data array with shape (time_steps, spatial_points), where spatial_points is the number
                            of spatial data points and time_steps is the number of time steps.
        k (int): Number of last time steps to average over. Must be less than or equal to time_steps.
        modes (int): Number of Fourier modes to include in the PSD, starting from the center frequency. Must be less
                     than or equal to spatial_points.

    Returns:
        np.ndarray: Averaged PSD for the specified modes, with shape (modes,).

    Raises:
        ValueError: If k exceeds the number of time_steps or modes exceeds the number of spatial_points.
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
        psd_sum += ps_shifted[center:center + modes]

    # Compute the average PSD over k time steps
    psd_avg = psd_sum / k

    return psd_avg

def compute_log_psd(array: np.ndarray, k: int, modes: int) -> np.ndarray:
    """
    Compute the natural logarithm of the averaged power spectral density (PSD) over the last k time steps.

    This function uses `compute_psd` to calculate the averaged PSD and then applies the natural logarithm to the result,
    adding a small constant (1e-10) to avoid taking the logarithm of zero.

    See `compute_psd` for details on the PSD calculation.

    Args:
        array (np.ndarray): Data array with shape (time_steps, spatial_points).
        k (int): Number of last time steps to average over.
        modes (int): Number of Fourier modes to include in the PSD.

    Returns:
        np.ndarray: Averaged log-PSD for the specified modes.

    Raises:
        ValueError: If k exceeds time_steps or modes exceeds spatial points.
    """
    # Compute the averaged PSD using the existing function
    psd = compute_psd(array, k, modes)

    # Compute the log of the PSD, adding a small constant to avoid log(0)
    log_psd = np.log(psd + 1e-10)

    return log_psd

def long_time_forecast_spatio_temporal(truth: np.ndarray, prediction: np.ndarray, k: int, modes: int) -> float:
    """
    Compute the long-time forecast error for spatio-temporal systems.

    This function calculates the error based on the power spectral density (PSD) of the last k
    time steps, comparing the log-PSD of truth and prediction.

    Args:
        truth (np.ndarray): Ground truth data array with shape (time steps, spatial points).
        prediction (np.ndarray): Predicted data array with shape (time steps, spatial points).
        k (int): Number of last time steps to consider.
        modes (int): Number of modes around the center frequency to consider.

    Returns:
        float: Long-time forecast score as a percentage.
    """
    Pt = compute_psd(truth, k, modes)
    Pp = compute_psd(prediction, k, modes)
    Elt = np.linalg.norm(Pt - Pp, ord=2) / np.linalg.norm(Pt, ord=2)
    return float(100 * (1 - Elt))

def evaluate(dataset_name: str, pair_id: int, prediction: np.ndarray, metrics: Optional[List[str]] = None) -> Dict[
    str, float]:
    """
    Evaluate the prediction using specified metrics. The truth array is loaded in this function, use `evaluate_custom` for providing your own truth array.

    This function loads the dataset configuration, retrieves the ground truth test data internally,
    determines the metrics to compute, and calculates the evaluation scores based on the dataset type
    and specified parameters.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz', 'PDE_KS').
        pair_id (int): ID of the train-test pair to use.
        truth (np.ndarray): True data array.
        prediction (np.ndarray): Predicted data array.
        metrics (Optional[List[str]]): List of metrics to compute. If None, uses defaults from config.

    Returns:
        Dict[str, float]: Dictionary containing the computed metrics.

    Raises:
        ValueError: If an unknown dataset type, metric, or invalid pair_id is specified.
    """
    # Retrieve the ground truth test data internally
    truth = _load_test_data(dataset_name, pair_id)

    # Evaluate
    results = evaluate_custom(dataset_name, pair_id, truth, prediction, metrics)

    # Return
    return results

def evaluate_custom(dataset_name: str, pair_id: int, truth: np.ndarray, prediction: np.ndarray, metrics: Optional[List[str]] = None) -> Dict[
    str, float]:
    """
    Evaluate the prediction using specified metrics and truth array. The truth array is provided in this function, use `evaluate` for loading ground-truth truth array.

    This function loads the dataset configuration, retrieves the ground truth test data internally,
    determines the metrics to compute, and calculates the evaluation scores based on the dataset type
    and specified parameters.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz', 'PDE_KS').
        pair_id (int): ID of the train-test pair to use.
        truth (np.ndarray): True data array.
        prediction (np.ndarray): Predicted data array.
        metrics (Optional[List[str]]): List of metrics to compute. If None, uses defaults from config.

    Returns:
        Dict[str, float]: Dictionary containing the computed metrics.

    Raises:
        ValueError: If an unknown dataset type, metric, or invalid pair_id is specified.
    """
    config = get_config(dataset_name)

    evaluation_params = config['evaluation_params']
    pair = next((p for p in config['pairs'] if p['id'] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    default_metrics = pair['metrics']
    metrics = default_metrics if metrics is None else metrics
    results = {}
    for metric in metrics:
        if metric == 'short_time':
            results['short_time'] = short_time_forecast(truth, prediction, evaluation_params['k'])
        elif metric == 'long_time':
            long_time_eval_type = config['evaluations']['long_time']
            if long_time_eval_type == 'histogram_L2_error':
                results['long_time'] = long_time_forecast_dynamical(truth, prediction, evaluation_params['modes'],
                                                                    evaluation_params['bins'])
            elif long_time_eval_type == 'spectral_L2_error':
                results['long_time'] = long_time_forecast_spatio_temporal(truth, prediction, evaluation_params['k'],
                                                                          evaluation_params['modes'])
            else:
                raise ValueError(f"Unknown dataset long time evaluation type: {long_time_eval_type}")
        elif metric == 'reconstruction':
            results['reconstruction'] = reconstruction(truth, prediction)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return results

def save_results(dataset_name: str, method_name: str, batch_id: str, pair_id: int, config: Dict[str, Any], predictions: np.ndarray, results: Dict[str, float] = None) -> None:
    """
    Save configuration, predictions, and evaluation results for a specific sub-dataset.

    Args:
        dataset_name (str): Name of the dataset.
        method_name (str): Name of the method or model.
        batch_id (str): Batch identifier and folder name for the batch run.
        pair_id (int): Sub-dataset identifier.
        config (Dict[str, Any]): Configuration dictionary used for the run.
        predictions (np.ndarray): Predicted data array.
        results (Dict[str, float]): Evaluation results dictionary.
    returns:
        results_dir (Path): Path to the directory containing the run results
    """
    results_dir = top_dir / 'results' / dataset_name / method_name / batch_id / f'pair{pair_id}'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    np.save(results_dir / 'predictions.npy', predictions)
    if results is not None:
        results_for_yaml = {key: float(value) for key, value in results.items()}
        with open(results_dir / 'evaluation_results.yaml', 'w') as f:
            yaml.dump(results_for_yaml, f)
    return results_dir
