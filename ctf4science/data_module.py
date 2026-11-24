"""
Data Module for CTF models, provides loading and configuration utilities for CTF datasets.

This module handles train/test pairs, timestep generation for training and evaluation, validation splits, and dataset metadata used by models and the evaluation pipeline.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat

file_dir = Path(__file__).parent
top_dir = Path(__file__).parent.parent


def parse_pair_ids(dataset_config: dict[str, Any]) -> list[int]:
    r"""Parse the pair_id configuration to determine which sub-datasets to process.

    Resolves `dataset_config` to a list of pair IDs. Supports a single integer,
    a list of integers, a range string (e.g. ``"1-3"``), or ``"all"``.

    Parameters
    ----------
    dataset_config : dict
        The ``dataset`` section from the config file. Must contain ``name``;
        may contain ``pair_id`` (int, list of int, range string, or ``"all"``).

    Returns
    -------
    list of int
        Pair IDs to process for the dataset.

    Raises
    ------
    ValueError
        If ``name`` is missing, `pair_id` is invalid, or requested pair_ids
        are not in the dataset configuration.
    """
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Dataset name must be specified in dataset_config")

    dataset_yaml = get_config(dataset_name)

    available_pair_ids = [pair["id"] for pair in dataset_yaml.get("pairs", [])]
    if not available_pair_ids:
        raise ValueError(
            f"No pairs defined in dataset configuration: {top_dir / 'data' / dataset_name / f'{dataset_name}.yaml'}"
        )

    pair_id_config = dataset_config.get("pair_id", "all")

    if pair_id_config == "all":
        return available_pair_ids
    elif isinstance(pair_id_config, int):
        if pair_id_config not in available_pair_ids:
            raise ValueError(f"Requested pair_id {pair_id_config} not in {dataset_name} dataset")
        return [pair_id_config]
    elif isinstance(pair_id_config, str) and "-" in pair_id_config:
        try:
            start, end = map(int, pair_id_config.split("-"))
            requested_ids = list(range(start, end + 1))
        except ValueError:
            raise ValueError(f"Invalid range format in pair_id: {pair_id_config}")
        invalid_ids = [pid for pid in requested_ids if pid not in available_pair_ids]
        if invalid_ids:
            raise ValueError(f"Requested pair_ids {invalid_ids} not in {dataset_name} dataset")
        return requested_ids
    elif isinstance(pair_id_config, list):
        invalid_ids = [pid for pid in pair_id_config if pid not in available_pair_ids]
        if invalid_ids:
            raise ValueError(f"Requested pair_ids {invalid_ids} not in {dataset_name} dataset")
        return pair_id_config
    else:
        raise ValueError(f"Invalid pair_id configuration: {pair_id_config}")


def get_prediction_timesteps(dataset_name: str, pair_id: int, subset: str = "test") -> np.ndarray:
    r"""Return physical time values at which predictions must be evaluated.

    Computes absolute physical times using the formula::

        timesteps[i] = (start_index + i) * delta_t,  i = 0, ..., N-1

    where ``start_index = metadata['matrix_start_index'][matrix_name]``,
    ``N = metadata['matrix_shapes'][matrix_name][0]`` (number of rows), and
    ``delta_t = metadata['delta_t']``.  These are **absolute** times along the
    underlying trajectory, not zero-based indices.  The first value is
    ``start_index * delta_t`` and the last is ``(start_index + N - 1) * delta_t``.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``, ``'PDE_KS'``).
    pair_id : int
        ID of the train-test pair.
    subset : {'test', 'initialization'}, optional
        Which matrix to use. Default is ``'test'``.

    Returns
    -------
    ndarray
        1D array of length ``N`` with physical time values spanning
        ``[start_index * delta_t, (start_index + N - 1) * delta_t]``.

    Notes
    -----
    **Example — ODE_Lorenz, pair 1, subset='test'**:
    ``X1test.mat`` has shape ``[1000, 3]``, ``start_index=10000``,
    ``delta_t=0.05``.  Returns 1000 values in ``[500.0, 549.95]``.

    **Example — ODE_Lorenz, pair 6, subset='test'**:
    ``X6test.mat`` has shape ``[1000, 3]``, ``start_index=100``,
    ``delta_t=0.05``.  Returns 1000 values in ``[5.0, 54.95]``.

    See the dataset YAML configs under ``data/<dataset_name>/`` for the
    ``matrix_shapes``, ``matrix_start_index``, and ``delta_t`` values for
    every matrix.

    Raises
    ------
    ValueError
        If `pair_id` is missing, `subset` matrix or metadata (e.g.
        ``matrix_shapes``, ``matrix_start_index``, ``delta_t``) is missing.
    """
    # Get config
    config = get_config(dataset_name)

    # Get config pair_id entry
    pair = next((p for p in config["pairs"] if p["id"] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    if subset not in ["test", "initialization"]:
        raise ValueError(f"Subset {subset} is not valid.")

    # Get testing matrix name
    test_mat_name = pair.get(subset, None)
    if test_mat_name is None:
        raise ValueError(f"Provided pair_id {pair_id} does not have a {subset} matrix in {dataset_name} config")

    # Load metadata and ensure it has necessary entries
    metadata = get_metadata(dataset_name)
    if "matrix_shapes" not in metadata or metadata["matrix_shapes"] is None:
        raise ValueError(f"Provided {dataset_name} config does not have 'matrix_shapes' or it is empty")
    if "matrix_start_index" not in metadata or metadata["matrix_shapes"] is None:
        raise ValueError(f"Provided {dataset_name} config does not have 'matrix_start_index' or it is empty")

    # Get testing matrix shape
    test_mat_shape = metadata["matrix_shapes"].get(test_mat_name, None)
    if test_mat_shape is None:
        raise ValueError(
            f"Provided {subset} matrix {test_mat_name} for pair_id {pair_id} in {dataset_name} config does not have a shape"
        )
    test_mat_len = test_mat_shape[0]

    # Get testing matrix start index
    test_mat_start_idx = metadata["matrix_start_index"].get(test_mat_name, None)
    if test_mat_start_idx is None:
        raise ValueError(
            f"Provided {subset} matrix {test_mat_name} for pair_id {pair_id} in {dataset_name} config does not have a starting index"
        )

    # Get delta_t
    delta_t = metadata.get("delta_t", None)
    if delta_t is None:
        raise ValueError(f"Provided {dataset_name} config does not have a delta_t")

    # Generate prediction_timesteps
    prediction_timesteps = (
        np.linspace(test_mat_start_idx, test_mat_start_idx + test_mat_len - 1, test_mat_len) * delta_t
    )

    return prediction_timesteps


def get_training_timesteps(dataset_name: str, pair_id: int) -> list[np.ndarray]:
    r"""Return physical time values for each training matrix of the given pair.

    For each training matrix, computes absolute physical times using::

        timesteps[i] = (start_index + i) * delta_t,  i = 0, ..., N-1

    where ``start_index = metadata['matrix_start_index'][matrix_name]``,
    ``N = metadata['matrix_shapes'][matrix_name][0]`` (number of rows), and
    ``delta_t = metadata['delta_t']``.  Values are **absolute** physical times
    along the underlying trajectory; the first value is
    ``start_index * delta_t`` and the last is ``(start_index + N - 1) * delta_t``.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``, ``'PDE_KS'``).
    pair_id : int
        ID of the train-test pair.

    Returns
    -------
    list of ndarray
        One 1D array per training matrix.  Each array has ``N`` elements
        spanning ``[start_index * delta_t, (start_index + N - 1) * delta_t]``.

    Notes
    -----
    **Example — ODE_Lorenz, pair 1**:
    ``X1train.mat`` has shape ``[10000, 3]``, ``start_index=0``,
    ``delta_t=0.05``.  Returns a list with one array of 10000 values in
    ``[0.0, 499.95]``.

    **Example — ODE_Lorenz, pair 8**:
    Three training matrices ``X6train``/``X7train``/``X8train``, each with
    shape ``[10000, 3]``, ``start_index=0``, ``delta_t=0.05``.  Returns a
    list of three arrays, each spanning ``[0.0, 499.95]``.

    See the dataset YAML configs under ``data/<dataset_name>/`` for the
    ``matrix_shapes``, ``matrix_start_index``, and ``delta_t`` values for
    every matrix.

    Raises
    ------
    ValueError
        If `pair_id` is missing, train matrices or metadata are missing.
    """
    # Get config
    config = get_config(dataset_name)

    # Get config pair_id entry
    pair = next((p for p in config["pairs"] if p["id"] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    # Get training matrix name
    train_mat_names = pair.get("train", None)
    if train_mat_names is None:
        raise ValueError(f"Provided pair_id {pair_id} does not have a train matrix in {dataset_name} config")

    # Load metadata and ensure it has necessary entries
    metadata = get_metadata(dataset_name)
    if "matrix_shapes" not in metadata or metadata["matrix_shapes"] is None:
        raise ValueError(f"Provided {dataset_name} config does not have 'matrix_shapes' or it is empty")
    if "matrix_start_index" not in metadata or metadata["matrix_shapes"] is None:
        raise ValueError(f"Provided {dataset_name} config does not have 'matrix_start_index' or it is empty")

    # Get training matrix shape
    prediction_timesteps_list: list[np.ndarray] = []
    for train_mat_name in train_mat_names:
        train_mat_shape = metadata["matrix_shapes"].get(train_mat_name, None)
        if train_mat_shape is None:
            raise ValueError(
                f"Provided train matrix {train_mat_name} for pair_id {pair_id} in {dataset_name} config does not have a shape"
            )
        train_mat_len = train_mat_shape[0]

        # Get training matrix start index
        train_mat_start_idx = metadata["matrix_start_index"].get(train_mat_name, None)
        if train_mat_start_idx is None:
            raise ValueError(
                f"Provided train matrix {train_mat_name} for pair_id {pair_id} in {dataset_name} config does not have a starting index"
            )

        # Get delta_t
        delta_t = metadata.get("delta_t", None)
        if delta_t is None:
            raise ValueError(f"Provided {dataset_name} config does not have a delta_t")

        # Generate prediction_timesteps
        prediction_timesteps = (
            np.linspace(train_mat_start_idx, train_mat_start_idx + train_mat_len - 1, train_mat_len) * delta_t
        )

        # Save to list
        prediction_timesteps_list.append(prediction_timesteps)

    return prediction_timesteps_list


def _load_mat_file(file_path: Path) -> np.ndarray:
    r"""Load a .mat file and return its single main variable.

    Keys starting with ``__`` are ignored; exactly one other variable must exist.

    Parameters
    ----------
    file_path : path-like
        Path to the .mat file.

    Returns
    -------
    ndarray
        The main variable from the .mat file.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    ValueError
        If the file does not contain exactly one main variable.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    mat_data = loadmat(str(file_path))
    keys = [key for key in mat_data if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected one main variable in {file_path}")

    return mat_data[keys[0]]


def _load_npy_file(file_path: Path) -> np.ndarray:
    r"""Load a .npy file and return its content as an array.

    Parameters
    ----------
    file_path : path-like
        Path to the .npy file.

    Returns
    -------
    ndarray
        The array loaded from `file_path`.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return np.load(str(file_path))


def _load_npz_file(file_path: Path) -> np.ndarray:
    r"""Load a .npz file and return its single main variable.

    Expects exactly one key in the archive; that array is returned.

    Parameters
    ----------
    file_path : path-like
        Path to the .npz file.

    Returns
    -------
    ndarray
        The single array from the .npz file.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    ValueError
        If the file does not contain exactly one variable.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    npz_data = np.load(str(file_path))
    keys = list(npz_data.keys())
    if len(keys) != 1:
        raise ValueError(f"Expected one main variable in {file_path}, found {len(keys)}: {keys}")

    return npz_data[keys[0]]


def _load_data_file(file_path: Path) -> np.ndarray:
    r"""Load a data file (.mat, .npy, or .npz) and return its main variable.

    Format is inferred from `file_path.suffix`. For .mat and .npz, exactly
    one main variable is required; .npy files contain a single array.

    Parameters
    ----------
    file_path : path-like
        Path to the data file (``.mat``, ``.npy``, or ``.npz``).

    Returns
    -------
    ndarray
        The main variable from the data file.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    ValueError
        If the file has an unsupported suffix (only ``.mat``, ``.npy``, ``.npz``
        supported) or contains more than one main variable (.mat/.npz).
    """
    if file_path.suffix == ".npz":
        return _load_npz_file(file_path)
    elif file_path.suffix == ".mat":
        return _load_mat_file(file_path)
    elif file_path.suffix == ".npy":
        return _load_npy_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Only .mat and .npz files are supported.")


def _load_test_data(dataset_name: str, pair_id: int, transpose=False) -> np.ndarray:
    r"""Load test data for a given dataset and pair ID.

    Intended for internal use by the evaluation module. Reads the test matrix
    for `pair_id` from the dataset ``test/`` directory.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``, ``'PDE_KS'``).
    pair_id : int
        ID of the train-test pair.
    transpose : bool, optional
        If True, transpose the result to (N, T). Default is False.

    Returns
    -------
    ndarray
        Test data array (optionally transposed).

    Raises
    ------
    ValueError
        If `pair_id` does not exist or has no test matrix, or the data file
        does not contain exactly one main variable.
    FileNotFoundError
        If the test data file is not found.
    """
    config = get_config(dataset_name)
    pair = next((p for p in config["pairs"] if p["id"] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    test_matrix_name = pair.get("test", None)
    if test_matrix_name is None:
        raise ValueError(f"Provided pair_id {pair_id} does not have a test matrix in {dataset_name} config")

    test_file = top_dir / "data" / dataset_name / "test" / test_matrix_name
    test_data = _load_data_file(test_file)
    if transpose:
        test_data = test_data.T
    return test_data


def load_dataset(dataset_name: str, pair_id: int, transpose=False) -> tuple[list, np.ndarray | None]:
    r"""Load train and initialization data for a given dataset and pair ID.

    Training matrices are loaded from the dataset ``train/`` directory according
    to the pair config; optional initialization matrix is loaded if present.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``, ``'PDE_KS'``).
    pair_id : int
        ID of the train-test pair.
    transpose : bool, optional
        If True, transpose each loaded matrix. Default is False.

    Returns
    -------
    train_data : list of ndarray
        List of training data arrays (one per ``train`` entry in the pair).
    init_data : ndarray or None
        Initialization data if the pair defines it, otherwise None.

    Raises
    ------
    ValueError
        If `pair_id` does not exist, has no training matrices, or a data file
        does not contain exactly one main variable.
    FileNotFoundError
        If any train or initialization data file is not found.
    """
    config = get_config(dataset_name)
    pair = next((p for p in config["pairs"] if p["id"] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    # Load train data
    train_data = []
    if "train" in pair:
        for data_file_name in pair["train"]:
            train_file = top_dir / "data" / dataset_name / "train" / data_file_name
            train_data_i = _load_data_file(train_file)
            if transpose:
                train_data_i = train_data_i.T
            train_data.append(train_data_i)
    else:
        raise ValueError(f"Provided pair_id {pair_id} does not have any training matrices in {dataset_name} config")

    # Load initialization data if available
    initialization_data = None
    if "initialization" in pair:
        initialization_file = top_dir / "data" / dataset_name / "train" / pair["initialization"]
        initialization_data = _load_data_file(initialization_file)
        if transpose:
            initialization_data = initialization_data.T

    return train_data, initialization_data


def load_nodes(dataset_name: str) -> np.ndarray:
    r"""Load the mesh/node information for a given dataset.

    Reads the ``nodes.npy`` file located in the root directory of the
    specified dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'PDE_KS'``).

    Returns
    -------
    ndarray
        The node/mesh data array.

    Raises
    ------
    FileNotFoundError
        If ``nodes.npy`` does not exist in the dataset folder.
    """
    node_file = top_dir / "data" / dataset_name / "nodes.npy"

    return _load_npy_file(node_file)


def get_applicable_plots(dataset_name: str) -> list[str]:
    r"""Return the list of applicable visualization types for the given dataset.

    Reads the ``visualizations`` key from the dataset config for `dataset_name`.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``).

    Returns
    -------
    list of str
        Applicable plot types (e.g. ``['trajectories', 'histograms']``), or an
        empty list if the dataset configuration defines no visualizations (as
        for ``buildings``, whose 3D arrays the plot functions do not support).
    """
    dataset_config = get_config(dataset_name)

    applicable_plots = dataset_config.get("visualizations") or []

    return applicable_plots


def get_metadata(dataset_name: str) -> dict:
    r"""Return metadata for the given dataset.

    Returns the ``metadata`` section from the dataset YAML (e.g. ``delta_t``,
    ``matrix_shapes``, ``matrix_start_index``).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``).

    Returns
    -------
    dict
        Dataset metadata (e.g. ``delta_t``, ``matrix_shapes``,
        ``matrix_start_index``).

    Raises
    ------
    ValueError
        If the ``metadata`` key is missing or empty in the dataset config.
    """
    dataset_config = get_config(dataset_name)

    metadata = dataset_config.get("metadata", [])
    if not metadata:
        raise ValueError(f"No metadata defined for dataset: {dataset_name}")

    return metadata


def get_config(dataset_name: str) -> dict[str, Any]:
    r"""Load and return the dataset configuration for the specified dataset.

    Reads ``data/{dataset_name}/{dataset_name}.yaml`` and returns the parsed
    YAML (pairs, metadata, visualizations, etc.).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'ODE_Lorenz'``).

    Returns
    -------
    dict
        Full dataset configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the dataset YAML file does not exist.
    """

    config_path = top_dir / "data" / dataset_name / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {config_path}")

    with open(top_dir / "data" / dataset_name / f"{dataset_name}.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config


def is_lower_is_better(dataset_config: dict[str, Any]) -> bool:
    r"""Return whether a dataset's metrics are errors (lower is better).

    True only if the dataset YAML sets ``lower_is_better: true`` (the YAML
    boolean, not merely a truthy value). Read by `evaluate_custom` (score
    clipping) and the tuner (default mode).

    Parameters
    ----------
    dataset_config : dict
        Parsed dataset YAML (as returned by `get_config`).

    Returns
    -------
    bool
        ``True`` if ``lower_is_better`` is ``True``, otherwise ``False``.
    """
    return dataset_config.get("lower_is_better", False) is True


def get_validation_training_timesteps(dataset_name, pair_id, train_split=0.8):
    r"""Return training timesteps used for the validation split.

    Returns physical time values (same units and formula as
    :func:`get_training_timesteps`) after applying the validation split
    logic.  The split behaviour depends on ``pair_id``:

    * **pair_ids 2, 4** (reconstruction): no split; training timesteps are
      returned unchanged.
    * **pair_id 8**: the second training matrix is held out for validation;
      its timesteps are removed from the returned list.
    * **pair_id 9**: the third training matrix is held out; its timesteps are
      removed.
    * **pair_ids 1, 3, 5, 6, 7**: the single training matrix is split at
      ``floor(train_split * N)`` rows; only the first portion's timesteps
      are returned.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.
    pair_id : int
        Data pair (1–9; special handling for 2, 4, 8, 9).
    train_split : float, optional
        Fraction of data for training in (0, 1). Default is 0.8. Ignored for
        pair_ids 2, 4, 8, 9.

    Returns
    -------
    list of ndarray
        Training timesteps for the validation setup (one array per training
        matrix after applying the split logic).  Each array contains absolute
        physical times computed as ``(start_index + i) * delta_t``.

    Raises
    ------
    ValueError
        If `train_split` is not in (0, 1) or `pair_id` is not in 1–9.
    """
    # Load sub-dataset
    train_data, _init_data = load_dataset(dataset_name, pair_id)

    # Get training and prediction timesteps
    training_timesteps = get_training_timesteps(dataset_name, pair_id)

    # Stack all training matrices to get a single training matrix
    train_data_all = np.concatenate(train_data, axis=0)

    # Generate validation split
    if pair_id in [1, 3, 5, 6, 7, 8, 9]:
        # Forecasting problem
        if pair_id == 8:
            training_timesteps.pop(1)
        elif pair_id == 9:
            training_timesteps.pop(2)
        else:
            # Verify input
            if train_split <= 0.0 or train_split >= 1.0:
                raise ValueError(f"train_split of {train_split} is not in (0.0, 1.0).")
            # Calculate total number of training points
            train_num = int(train_split * train_data_all.shape[0])
            # Validation split is obtained from the only matrix in the train_data list
            training_timesteps_tmp = training_timesteps[0].copy()
            training_timesteps[0] = training_timesteps_tmp[0:train_num]
    elif pair_id in [2, 4]:
        pass
    else:
        raise ValueError(f"The provided pair_id {pair_id} is invalid.")

    return training_timesteps


def get_validation_prediction_timesteps(dataset_name, pair_id, train_split=0.8):
    r"""Return prediction timesteps for the validation split.

    Returns a 1D array of absolute physical times (same units and formula as
    :func:`get_prediction_timesteps`) for the held-out validation window.
    The source of these timesteps depends on ``pair_id``:

    * **pair_ids 2, 4** (reconstruction): prediction timesteps equal the
      full training matrix timesteps.
    * **pair_id 8**: timesteps of the second training matrix (the held-out
      validation matrix).
    * **pair_id 9**: timesteps of the third training matrix.
    * **pair_ids 1, 3, 5, 6, 7**: the tail of the single training matrix
      (rows ``floor(train_split * N)`` onward).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.
    pair_id : int
        Data pair (1–9; special handling for 2, 4, 8, 9).
    train_split : float, optional
        Fraction of data for training in (0, 1). Default is 0.8. Ignored for
        pair_ids 2, 4, 8, 9.

    Returns
    -------
    ndarray
        1D array of absolute physical times for the validation prediction
        window, computed as ``(start_index + i) * delta_t``.

    Raises
    ------
    ValueError
        If `train_split` is not in (0, 1) or `pair_id` is not in 1–9.
    """
    # Load sub-dataset
    train_data, _ = load_dataset(dataset_name, pair_id)

    # Get training and prediction timesteps
    training_timesteps = get_training_timesteps(dataset_name, pair_id)

    # Stack all training matrices to get a single training matrix
    train_data_all = np.concatenate(train_data, axis=0)

    # Generate validation split
    if pair_id in [1, 3, 5, 6, 7, 8, 9]:
        # Forecasting problem
        if pair_id == 8:
            # Get prediction timesteps for initialization matrix
            prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id, subset="initialization")
            # Validation split is obtained from the matrices X7trian.mat
            prediction_timesteps = training_timesteps.pop(1)
        elif pair_id == 9:
            # Get prediction timesteps for initialization matrix
            prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id, subset="initialization")
            # Validation split is obtained from the matrices X8trian.mat
            prediction_timesteps = training_timesteps.pop(2)
        else:
            # Verify input
            if train_split <= 0.0 or train_split >= 1.0:
                raise ValueError(f"train_split of {train_split} is not in (0.0, 1.0).")
            # Calculate total number of training points
            train_num = int(train_split * train_data_all.shape[0])
            # Validation split is obtained from the only matrix in the train_data list
            training_timesteps_tmp = training_timesteps[0].copy()
            prediction_timesteps = training_timesteps_tmp[train_num:]
    elif pair_id in [2, 4]:
        # Reconstruction problem
        prediction_timesteps = training_timesteps[0].copy()
    else:
        raise ValueError(f"The provided pair_id {pair_id} is invalid.")

    return prediction_timesteps


def load_validation_dataset(dataset_name, pair_id, train_split=0.8, transpose=False):
    r"""Load dataset and split into training, validation, and initialization.

    For pair_ids 2 and 4, validation is a copy of training. For pair_ids 8 and
    9, one training matrix becomes validation and `init_data` is the burn-in
    subset; for 1, 3, 5, 6, 7, the single matrix is split by `train_split`.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.
    pair_id : int
        Data pair (1–9; special handling for 2, 4, 8, 9).
    train_split : float, optional
        Fraction of data for training in (0, 1). Default is 0.8. Ignored for
        pair_ids 2, 4, 8, 9.
    transpose : bool, optional
        If True, return arrays as (features, timesteps); otherwise
        (timesteps, features). Default is False.

    Returns
    -------
    train_data : list of ndarray
        Training data arrays (after split).
    val_data : ndarray
        Validation data matrix.
    init_data : ndarray or None
        Initialization (burn-in) data; only non-None for pair_id 8 and 9.

    Raises
    ------
    ValueError
        If `train_split` is not in (0, 1), `pair_id` is not in 1–9, or
        pair_id 8/9 has no initialization data.
    """
    # Load sub-dataset
    train_data, init_data = load_dataset(dataset_name, pair_id)

    # Get training and prediction timesteps
    training_timesteps = get_training_timesteps(dataset_name, pair_id)

    # Stack all training matrices to get a single training matrix
    train_data_all = np.concatenate(train_data, axis=0)

    # Generate validation split
    if pair_id in [1, 3, 5, 6, 7, 8, 9]:
        # Forecasting problem
        if pair_id == 8:
            # Get prediction timesteps for initialization matrix
            # Validation split is obtained from the matrices X7trian.mat
            val_data = train_data.pop(1)
        elif pair_id == 9:
            # Get prediction timesteps for initialization matrix
            # Validation split is obtained from the matrices X8trian.mat
            val_data = train_data.pop(2)
        else:
            # Verify input
            if train_split <= 0.0 or train_split >= 1.0:
                raise ValueError(f"train_split of {train_split} is not in (0.0, 1.0).")
            # Calculate total number of training points
            train_num = int(train_split * train_data_all.shape[0])
            # Validation split is obtained from the only matrix in the train_data list
            train_data_tmp = train_data[0].copy()
            train_data[0] = train_data_tmp[0:train_num, :]
            training_timesteps_tmp = training_timesteps[0].copy()
            training_timesteps[0] = training_timesteps_tmp[0:train_num]
            val_data = train_data_tmp[train_num:, :]
    elif pair_id in [2, 4]:
        # Reconstruction problem
        val_data = train_data[0].copy()
    else:
        raise ValueError(f"The provided pair_id {pair_id} is invalid.")

    # Extract burn in split when applicable (using same size as original burn_in matrix)
    if pair_id in [8, 9]:
        if init_data is None:
            raise ValueError(f"The provided pair_id {pair_id} does not have initialization data.")
        burn_in_num = init_data.shape[0]
        init_data = val_data.copy()
        init_data = init_data[0:burn_in_num, :]
    if transpose:
        return [td.T for td in train_data], val_data.T, (init_data.T if init_data is not None else None)

    return train_data, val_data, init_data


# ---------------------------------------------------------------------------
# Windowed forecasting datasets (``buildings``)
# ---------------------------------------------------------------------------

# Values the ``covariates`` key of a windowed pair may take (absent means "none")
COVARIATE_MODES = ("none", "exact", "noisy")

# Channels of the continuous (time, building, channel) arrays that a noisy-weather file replaces,
# in the order the noisy-weather file stores them. Every other covariate channel (``u``) always
# comes from the clean file.
NOISY_WEATHER_CHANNELS = (0, 1, 2)

# Suffix appended to the stem of a train file to name its noisy-weather file (used by validation)
NOISY_WEATHER_SUFFIX = "_noisy_weather"

# Name of the file in the dataset directory that lists the building ID of every column of each set
BUILDING_IDS_FILE = "building_ids.txt"
BUILDING_IDS_HEADER = ("set", "column", "building_id")


def _window_starts(n_rows: int, lookback: int, horizon: int, stride: int) -> np.ndarray:
    r"""Return the first row of every forecast window that fits in a series of `n_rows` rows.

    Window ``i`` starts at row ``i * stride``. Its lookback is rows
    ``[i * stride, i * stride + lookback)`` and its horizon is the next
    `horizon` rows. Only windows whose horizon ends inside the series are kept.

    Parameters
    ----------
    n_rows : int
        Number of rows (time steps) in the continuous series.
    lookback : int
        Number of rows a model sees before each forecast.
    horizon : int
        Number of rows forecast after each lookback.
    stride : int
        Number of rows between consecutive window starts.

    Returns
    -------
    ndarray
        1D integer array of window start rows, ``[0, stride, 2 * stride, ...]``.
        Empty if the series is shorter than ``lookback + horizon``.

    Raises
    ------
    ValueError
        If `lookback`, `horizon` or `stride` is not a positive integer.
    """
    for name, value in (("lookback", lookback), ("horizon", horizon), ("stride", stride)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    span = lookback + horizon
    n_windows = 0 if n_rows < span else (n_rows - span) // stride + 1
    return np.arange(n_windows) * stride


def _forecast_windows(series: np.ndarray, lookback: int, horizon: int, stride: int, part: str) -> np.ndarray:
    r"""Cut a continuous series into forecast windows, as a view without copying.

    This is the one place where forecast windows are cut: the forecast loaders
    and `ctf4science.eval_module.evaluate_custom` all call it, so lookbacks,
    covariates and truth always line up. Window start rows come from
    `_window_starts`.

    Parameters
    ----------
    series : ndarray
        Continuous series with time on axis 0, e.g. ``(T, buildings, channels)``
        or ``(T, buildings)``.
    lookback : int
        Number of lookback rows per window.
    horizon : int
        Number of forecast rows per window.
    stride : int
        Number of rows between consecutive window starts.
    part : {'lookback', 'horizon'}
        Which rows of each window to return: the `lookback` rows, or the
        `horizon` rows that follow them.

    Returns
    -------
    ndarray
        Read-only view of shape ``(W, length, *series.shape[1:])``, where ``W``
        is the number of windows and ``length`` is `lookback` or `horizon`.

    Raises
    ------
    ValueError
        If `part` is not ``'lookback'`` or ``'horizon'``, or a window
        parameter is not a positive integer.
    """
    if part == "lookback":
        offset, length = 0, lookback
    elif part == "horizon":
        offset, length = lookback, horizon
    else:
        raise ValueError(f"part must be 'lookback' or 'horizon', got {part!r}")

    n_windows = len(_window_starts(series.shape[0], lookback, horizon, stride))
    if n_windows == 0:
        return np.empty((0, length, *series.shape[1:]), dtype=series.dtype)

    # Rows covered by this part of all windows; sliding_window_view puts the window axis last
    segment = series[offset : offset + (n_windows - 1) * stride + length]
    windows = np.lib.stride_tricks.sliding_window_view(segment, length, axis=0)[::stride]
    return np.moveaxis(windows, -1, 1)


def _get_pair(config: dict[str, Any], dataset_name: str, pair_id: int) -> dict[str, Any]:
    r"""Return the config entry of a pair, raising ``ValueError`` if it does not exist."""
    pair = next((p for p in config["pairs"] if p["id"] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")
    return pair


def _get_windowed_pair(config: dict[str, Any], dataset_name: str, pair_id: int) -> dict[str, Any]:
    r"""Return the config entry of a windowed forecasting pair (one with a ``horizon`` key)."""
    pair = _get_pair(config, dataset_name, pair_id)
    if "horizon" not in pair:
        raise ValueError(f"Provided pair_id {pair_id} in {dataset_name} config is not a windowed pair (no 'horizon')")
    return pair


def is_windowed_pair(dataset_name: str, pair_id: int) -> bool:
    r"""Return whether a pair is a windowed forecasting pair (it has a ``horizon`` key).

    Windowed pairs (e.g. buildings) are forecast with `load_forecast_windows`
    and scored on windows; all other pairs use `load_dataset` as before.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'buildings'``).
    pair_id : int
        ID of the train-test pair.

    Returns
    -------
    bool
        ``True`` if the pair defines ``horizon``, otherwise ``False``.

    Raises
    ------
    ValueError
        If `pair_id` does not exist in the dataset config.
    """
    return "horizon" in _get_pair(get_config(dataset_name), dataset_name, pair_id)


def get_window_params(dataset_name: str, pair_id: int) -> dict[str, int]:
    r"""Return the forecast window parameters of a windowed pair.

    The ``lookback`` comes from ``evaluation_params``; the ``stride`` comes
    from the pair's own ``stride`` key if it has one, otherwise from
    ``evaluation_params``; the ``horizon`` comes from the pair.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'buildings'``).
    pair_id : int
        ID of the train-test pair. The pair must define ``horizon``.

    Returns
    -------
    dict
        ``{'lookback': int, 'horizon': int, 'stride': int}``.

    Raises
    ------
    ValueError
        If `pair_id` does not exist or the pair has no ``horizon``.
    """
    config = get_config(dataset_name)
    return _window_params(config, _get_windowed_pair(config, dataset_name, pair_id))


def _window_params(config: dict[str, Any], pair: dict[str, Any]) -> dict[str, int]:
    r"""Resolve lookback, horizon and stride (with the per-pair stride override) for a pair."""
    evaluation_params = config["evaluation_params"]
    return {
        "lookback": evaluation_params["lookback"],
        "horizon": pair["horizon"],
        "stride": pair.get("stride", evaluation_params["stride"]),
    }


def _covariate_mode(pair: dict[str, Any], dataset_name: str) -> str:
    r"""Return the pair's ``covariates`` setting, ``'none'`` if absent."""
    mode = pair.get("covariates", "none")
    if mode not in COVARIATE_MODES:
        raise ValueError(
            f"Unknown covariates setting {mode!r} for pair_id {pair['id']} in {dataset_name} config; "
            f"expected one of {COVARIATE_MODES}"
        )
    return mode


def _covariate_series(clean: np.ndarray, noisy_weather: np.ndarray | None, covariate_channels: list[int]) -> np.ndarray:
    r"""Assemble the continuous covariate series ``(T, buildings, len(covariate_channels))``.

    Weather channels (`NOISY_WEATHER_CHANNELS`) are taken from `noisy_weather`
    when it is given; every other channel (``u``) is taken from `clean`.
    """
    if noisy_weather is None:
        return clean[..., covariate_channels]
    if noisy_weather.shape[:-1] != clean.shape[:-1] or noisy_weather.shape[-1] != len(NOISY_WEATHER_CHANNELS):
        raise ValueError(
            f"Noisy weather shape {noisy_weather.shape} does not match clean shape {clean.shape} "
            f"with {len(NOISY_WEATHER_CHANNELS)} weather channels"
        )
    columns = [
        noisy_weather[..., NOISY_WEATHER_CHANNELS.index(c)] if c in NOISY_WEATHER_CHANNELS else clean[..., c]
        for c in covariate_channels
    ]
    return np.stack(columns, axis=-1)


def _window_inputs(
    config: dict[str, Any],
    pair: dict[str, Any],
    dataset_name: str,
    series: np.ndarray,
    noisy_weather: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Cut the lookback windows and (if the pair uses them) covariate windows from a continuous series."""
    params = _window_params(config, pair)
    lookback = _forecast_windows(series, params["lookback"], params["horizon"], params["stride"], "lookback")
    mode = _covariate_mode(pair, dataset_name)
    if mode == "none":
        return lookback, None
    covariate_channels = config["evaluation_params"]["covariate_channels"]
    covariates = _covariate_series(series, noisy_weather if mode == "noisy" else None, covariate_channels)
    covariates = _forecast_windows(covariates, params["lookback"], params["horizon"], params["stride"], "horizon")
    return lookback, covariates


def load_forecast_windows(dataset_name: str, pair_id: int) -> tuple[np.ndarray, np.ndarray | None]:
    r"""Load the test-set forecast windows a model predicts from, for a windowed pair.

    Cuts the pair's continuous test series ``(T, buildings, channels)`` into
    windows with the dataset's ``lookback``, the pair's ``horizon`` and the
    pair's ``stride`` (or the dataset default). Window ``i`` has lookback rows
    ``[i * stride, i * stride + lookback)``; the forecast covers the next
    ``horizon`` rows. Models never cut windows themselves, so every model is
    scored on the same windows by `ctf4science.eval_module.evaluate`.

    Both returned window arrays (lookback and covariates) are read-only views;
    copy a batch before modifying it in place. Predict each window from its
    own lookback and covariates only, and fit nothing on these arrays:
    adjacent windows overlap, so another window's lookback can contain this
    window's answer.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'buildings'``).
    pair_id : int
        ID of the train-test pair. The pair must define ``horizon``.

    Returns
    -------
    lookback : ndarray
        Read-only view into the test series, shape
        ``(W, lookback, buildings, channels)``, all channels. It is not
        copied, so predict over ``W`` in batches: copying it in one go can
        take many GB.
    covariates : ndarray or None
        Read-only view of the known future inputs over each forecast horizon,
        shape ``(W, horizon, buildings, len(covariate_channels))``, in the order of
        ``evaluation_params.covariate_channels``. None if the pair's
        ``covariates`` is ``none`` (or absent). For ``noisy`` the weather
        channels come from the pair's ``noisy_weather`` file in ``test/`` and
        the other channels from the clean test series.

    Raises
    ------
    ValueError
        If `pair_id` does not exist, the pair has no ``horizon``, its
        ``covariates`` setting is unknown, or a ``noisy`` pair has no
        ``noisy_weather`` file.
    FileNotFoundError
        If a data file is missing.
    """
    config = get_config(dataset_name)
    pair = _get_windowed_pair(config, dataset_name, pair_id)

    series = _load_test_data(dataset_name, pair_id)
    noisy_weather = None
    if _covariate_mode(pair, dataset_name) == "noisy":
        if "noisy_weather" not in pair:
            raise ValueError(
                f"Provided pair_id {pair_id} in {dataset_name} config has noisy covariates but no 'noisy_weather' file"
            )
        noisy_weather = _load_data_file(top_dir / "data" / dataset_name / "test" / pair["noisy_weather"])

    return _window_inputs(config, pair, dataset_name, series, noisy_weather)


def load_building_ids(dataset_name: str) -> dict[str, list[str]]:
    r"""Load the building ID of every column of each set from ``building_ids.txt``.

    The file is tab-separated with the header line ``set<TAB>column<TAB>building_id``
    and one row per column of each set. ``set`` names the arrays whose
    building axis (axis 1) the row describes: ``sources`` for
    ``train/sources.npz``, and ``targets_PI`` / ``targets_RW`` for the target
    files. ``column`` is the index on axis 1 and ``building_id`` is the
    building's parameter string. Rows of one set that share a ``building_id``
    are variants of one building (e.g. the excitation variants of a source
    building).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'buildings'``).

    Returns
    -------
    dict
        Mapping from set name to its building IDs, ordered by column.

    Raises
    ------
    FileNotFoundError
        If the dataset has no ``building_ids.txt``.
    ValueError
        If the header is wrong, a row is malformed, two rows give the same set
        and column, or a set's columns are not exactly ``0, 1, ..., n - 1``.
    """
    file_path = top_dir / "data" / dataset_name / BUILDING_IDS_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"Building ID file not found: {file_path}")

    with open(file_path, "r") as f:
        lines = [line.rstrip("\r\n") for line in f if line.strip()]
    if not lines or lines[0].split("\t") != list(BUILDING_IDS_HEADER):
        raise ValueError(f"{file_path} must start with the header {chr(9).join(BUILDING_IDS_HEADER)!r}")

    columns: dict[str, dict[int, str]] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(BUILDING_IDS_HEADER):
            raise ValueError(f"{file_path} line {line_number}: expected 3 tab-separated fields, got {line!r}")
        set_name, column, building_id = fields
        by_column = columns.setdefault(set_name, {})
        if int(column) in by_column:
            raise ValueError(f"{file_path} line {line_number}: set {set_name!r} lists column {column} twice")
        by_column[int(column)] = building_id

    building_ids: dict[str, list[str]] = {}
    for set_name, by_column in columns.items():
        if sorted(by_column) != list(range(len(by_column))):
            raise ValueError(f"{file_path}: columns of set {set_name!r} are not 0..{len(by_column) - 1}")
        building_ids[set_name] = [by_column[i] for i in range(len(by_column))]
    return building_ids


def _noisy_weather_name(file_name: str) -> str:
    r"""Return the name of a train file's noisy-weather file, e.g. ``sources_noisy_weather.npz``."""
    path = Path(file_name)
    return f"{path.stem}{NOISY_WEATHER_SUFFIX}{path.suffix}"


def load_validation_forecast_windows(
    dataset_name: str, pair_id: int, n_holdout: int, n_val_rows: int = 2688, seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray | None, np.ndarray]:
    r"""Split a windowed pair's training data into a training set and validation windows.

    Used for hyperparameter tuning, where the test set must not be used. The
    split mirrors the pair's task. A pair is a **forecasting** pair if its
    ``train`` list has a second file (the target buildings' own earlier data);
    otherwise it is a **generalization** pair.

    * **Generalization** (one train file, the source buildings): `n_holdout`
      building IDs are drawn by a shuffle with `seed` and held out with all
      their columns (every variant of a held-out building leaves training
      together, see `load_building_ids`). The held-out buildings' whole
      series is the validation series. `n_val_rows` is ignored.
    * **Forecasting** (two train files: sources, then target data): all
      sources stay in training. The target file is split in time: training
      gets its first ``T - n_val_rows`` rows and the last `n_val_rows` rows
      are the validation series for all target buildings. `n_holdout` is
      ignored.

    The validation series is windowed exactly as the test series is by
    `load_forecast_windows`. For pairs with ``covariates: noisy`` the weather
    covariates come from the train file's noisy-weather file
    (``<stem>_noisy_weather.npz`` in ``train/``).

    As for the test set, both window arrays (lookback and covariates) are
    read-only views; copy a batch before modifying it in place. Predict each
    window from its own lookback and covariates only, and fit nothing on
    these arrays or on `truth`: adjacent windows overlap, so another window's
    lookback can contain this window's answer.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. ``'buildings'``).
    pair_id : int
        ID of the train-test pair. The pair must define ``horizon``.
    n_holdout : int
        Number of source building IDs to hold out (generalization pairs).
    n_val_rows : int, optional
        Number of trailing target rows to hold out (forecasting pairs).
        Default is 2688 (28 days of 15-minute steps).
    seed : int, optional
        Seed of the building shuffle (generalization pairs). Default is 0.

    Returns
    -------
    train_data : list of ndarray
        Training arrays, in the order of the pair's ``train`` list, with the
        validation part removed.
    lookback : ndarray
        Read-only lookback windows of the validation series, shape
        ``(W, lookback, buildings, channels)``.
    covariates : ndarray or None
        Read-only covariate windows, shape ``(W, horizon, buildings, n_covariates)``,
        or None if the pair uses no covariates.
    truth : ndarray
        The continuous validation series ``(T_val, buildings, channels)``,
        not windowed. Pass it to `ctf4science.eval_module.evaluate_custom`,
        which cuts the truth windows itself.

    Raises
    ------
    ValueError
        If the pair is not windowed, has more than two train files,
        `n_holdout` or `n_val_rows` is not an integer (e.g. ``45.0`` from a
        YAML file), or they leave an empty training or validation set.
    FileNotFoundError
        If a data file or ``building_ids.txt`` is missing.
    """
    for name, value in (("n_holdout", n_holdout), ("n_val_rows", n_val_rows)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            # ValueError, as for the other bad window and split parameters
            raise ValueError(f"{name} must be an integer, got {value!r} ({type(value).__name__})")  # noqa: TRY004
    config = get_config(dataset_name)
    pair = _get_windowed_pair(config, dataset_name, pair_id)
    train_names = pair.get("train", [])
    if len(train_names) not in (1, 2):
        raise ValueError(
            f"Provided pair_id {pair_id} in {dataset_name} config must have one train file (generalization) "
            f"or two (forecasting), got {len(train_names)}"
        )
    train_dir = top_dir / "data" / dataset_name / "train"
    noisy = _covariate_mode(pair, dataset_name) == "noisy"
    params = _window_params(config, pair)

    sources = _load_data_file(train_dir / train_names[0])

    if len(train_names) == 1:
        # Generalization: hold out whole source buildings (all their variants) by building ID
        set_name = Path(train_names[0]).stem
        ids = load_building_ids(dataset_name).get(set_name)
        if ids is None:
            raise ValueError(f"No rows for set {set_name!r} in {BUILDING_IDS_FILE} of {dataset_name}")
        if len(ids) != sources.shape[1]:
            raise ValueError(
                f"{BUILDING_IDS_FILE} lists {len(ids)} IDs for {set_name}, but {train_names[0]} has "
                f"{sources.shape[1]} buildings"
            )
        unique_ids = list(dict.fromkeys(ids))
        if not 1 <= n_holdout < len(unique_ids):
            raise ValueError(
                f"n_holdout must be at least 1 and below the {len(unique_ids)} source building IDs, got {n_holdout}"
            )
        shuffled = np.random.default_rng(seed).permutation(len(unique_ids))
        held_out = {unique_ids[i] for i in shuffled[:n_holdout]}
        val_cols = [i for i, building_id in enumerate(ids) if building_id in held_out]
        train_cols = [i for i, building_id in enumerate(ids) if building_id not in held_out]

        train_data = [sources[:, train_cols]]
        truth = sources[:, val_cols]
        noisy_weather = None
        if noisy:
            noisy_weather = _load_data_file(train_dir / _noisy_weather_name(train_names[0]))[:, val_cols]
    else:
        # Forecasting: keep all sources, split the target data in time
        targets = _load_data_file(train_dir / train_names[1])
        n_rows = targets.shape[0]
        if not 1 <= n_val_rows < n_rows:
            raise ValueError(f"n_val_rows must be in [1, {n_rows - 1}], got {n_val_rows}")
        split = n_rows - n_val_rows
        train_data = [sources, targets[:split]]
        truth = targets[split:]
        noisy_weather = None
        if noisy:
            noisy_weather = _load_data_file(train_dir / _noisy_weather_name(train_names[1]))[split:]

    if len(_window_starts(truth.shape[0], params["lookback"], params["horizon"], params["stride"])) == 0:
        raise ValueError(
            f"Validation series of {truth.shape[0]} rows is shorter than lookback + horizon "
            f"({params['lookback']} + {params['horizon']})"
        )

    lookback, covariates = _window_inputs(config, pair, dataset_name, truth, noisy_weather)
    return train_data, lookback, covariates, truth
