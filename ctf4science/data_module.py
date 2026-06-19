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
        Applicable plot types (e.g. ``['trajectories', 'histograms']``).

    Raises
    ------
    ValueError
        If no visualizations are defined in the dataset configuration.
    """
    dataset_config = get_config(dataset_name)

    applicable_plots = dataset_config.get("visualizations", [])
    if not applicable_plots:
        raise ValueError(f"No visualizations defined for dataset: {dataset_name}")

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
