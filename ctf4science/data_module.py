import os
import yaml
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from typing import Dict, Any, List, Tuple, Optional

file_dir = Path(__file__).parent
top_dir = Path(__file__).parent.parent

def parse_pair_ids(dataset_config: Dict[str, Any]) -> List[int]:
    """
    Parse the pair_id configuration to determine which sub-datasets to process.

    Args:
        dataset_config (Dict[str, Any]): The 'dataset' section from the config file,
                                         containing 'name' and optionally 'pair_id'.

    Returns:
        List[int]: A list of pair_ids to process.

    Raises:
        ValueError: If the pair_id configuration is invalid or requested pair_ids are unavailable.
    """
    dataset_name = dataset_config.get('name')
    if not dataset_name:
        raise ValueError("Dataset name must be specified in dataset_config")

    dataset_yaml = get_config(dataset_name)
    
    available_pair_ids = [pair['id'] for pair in dataset_yaml.get('pairs', [])]
    if not available_pair_ids:
        raise ValueError(f"No pairs defined in dataset configuration: {top_dir / 'data' / dataset_name / f'{dataset_name}.yaml'}")

    pair_id_config = dataset_config.get('pair_id', 'all')

    if pair_id_config == 'all':
        return available_pair_ids
    elif isinstance(pair_id_config, int):
        if pair_id_config not in available_pair_ids:
            raise ValueError(f"Requested pair_id {pair_id_config} not in {dataset_name} dataset")
        return [pair_id_config]
    elif isinstance(pair_id_config, str) and '-' in pair_id_config:
        try:
            start, end = map(int, pair_id_config.split('-'))
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
    """
    Provide a numpy vector containing the time steps at which the prediction
    matrices need to be evaluated at once a model has been trained.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz', 'PDE_KS').
        pair_id (int): ID of the train-test pair to load.
        subset (str): Either "test" or "initialization"

    Returns:
        np.ndarray: A 1D numpy vector containing time-steps to evaluate a model
                    at given the dataset_name and pair_id

    Raises:
        ValueError: If the dataset configuration does not contain enough information
                    to generate the output prediction timesteps vector. This can
                    happen from any of the following:
                    - 'pair_id' doesn't exist in config
                    - 'test' matrix not defined in pair_id
                    - 'metadata' does not exist in config
                    - 'metadata' does not have 'matrix_shapes' or 'matrix_start_index
                    - 'metadata' does not have test matrix shape
                    - 'metadata' does not have test matrix starting index
                    - 'delta_t' does not exist in metadata
    """
    # Get config
    config = get_config(dataset_name)

    # Get config pair_id entry
    pair = next((p for p in config['pairs'] if p['id'] == pair_id), None)
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
    if 'matrix_shapes' not in metadata or metadata['matrix_shapes'] is None:
        raise ValueError(f"Provided {dataset_name} config does not have 'matrix_shapes' or it is empty")
    if 'matrix_start_index' not in metadata or metadata['matrix_shapes'] is None:
        raise ValueError(f"Provided {dataset_name} config does not have 'matrix_start_index' or it is empty")

    # Get testing matrix shape
    test_mat_shape = metadata['matrix_shapes'].get(test_mat_name, None)
    if test_mat_shape is None:
        raise ValueError(f"Provided {subset} matrix {test_mat_name} for pair_id {pair_id} in {dataset_name} config does not have a shape")
    test_mat_len = test_mat_shape[0]

    # Get testing matrix start index
    test_mat_start_idx = metadata['matrix_start_index'].get(test_mat_name, None)
    if test_mat_start_idx is None:
        raise ValueError(f"Provided {subset} matrix {test_mat_name} for pair_id {pair_id} in {dataset_name} config does not have a starting index")

    # Get delta_t
    delta_t = metadata.get('delta_t', None)
    if delta_t is None:
        raise ValueError(f"Provided {dataset_name} config does not have a delta_t")

    # Generate prediction_timesteps
    prediction_timesteps = np.linspace(test_mat_start_idx, test_mat_start_idx + test_mat_len - 1, test_mat_len) * delta_t

    return prediction_timesteps

def get_training_timesteps(dataset_name: str, pair_id: int) -> np.ndarray:
    """
    Provide a list of numpy vectors containing the time steps of the training
    matrices.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz', 'PDE_KS').
        pair_id (int): ID of the train-test pair to load.

    Returns:
        list(np.ndarray): A list of 1D numpy vector containing time-steps of the training
                    matrices from the given the dataset_name and pair_id

    Raises:
        ValueError: If the dataset configuration does not contain enough information
                    to generate the output prediction timesteps vector. This can
                    happen from any of the following:
                    - 'pair_id' doesn't exist in config
                    - 'train' matrix not defined in pair_id
                    - 'metadata' does not exist in config
                    - 'metadata' does not have 'matrix_shapes' or 'matrix_start_index
                    - 'metadata' does not have train matrix shape
                    - 'metadata' does not have train matrix starting index
                    - 'delta_t' does not exist in metadata
    """
    # Get config
    config = get_config(dataset_name)

    # Get config pair_id entry
    pair = next((p for p in config['pairs'] if p['id'] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    # Get training matrix name
    train_mat_names = pair.get('train', None)
    if train_mat_names is None:
        raise ValueError(f"Provided pair_id {pair_id} does not have a train matrix in {dataset_name} config")

    # Load metadata and ensure it has necessary entries
    metadata = get_metadata(dataset_name)
    if 'matrix_shapes' not in metadata or metadata['matrix_shapes'] is None:
        raise ValueError(f"Provided {dataset_name} config does not have 'matrix_shapes' or it is empty")
    if 'matrix_start_index' not in metadata or metadata['matrix_shapes'] is None:
        raise ValueError(f"Provided {dataset_name} config does not have 'matrix_start_index' or it is empty")

    # Get training matrix shape
    prediction_timesteps_list = list()
    for train_mat_name in train_mat_names:
        train_mat_shape = metadata['matrix_shapes'].get(train_mat_name, None)
        if train_mat_shape is None:
            raise ValueError(f"Provided train matrix {train_mat_name} for pair_id {pair_id} in {dataset_name} config does not have a shape")
        train_mat_len = train_mat_shape[0]

        # Get training matrix start index
        train_mat_start_idx = metadata['matrix_start_index'].get(train_mat_name, None)
        if train_mat_start_idx is None:
            raise ValueError(f"Provided train matrix {train_mat_name} for pair_id {pair_id} in {dataset_name} config does not have a starting index")

        # Get delta_t
        delta_t = metadata.get('delta_t', None)
        if delta_t is None:
            raise ValueError(f"Provided {dataset_name} config does not have a delta_t")

        # Generate prediction_timesteps
        prediction_timesteps = np.linspace(train_mat_start_idx, train_mat_start_idx + train_mat_len - 1, train_mat_len) * delta_t

        # Save to list
        prediction_timesteps_list.append(prediction_timesteps)

    return prediction_timesteps_list

def _load_mat_file(file_path: Path) -> np.ndarray:
    """
    Load a .mat file and extract the single main variable.

    This function checks if the specified .mat file exists, loads it, and extracts the main variable.
    It expects exactly one main variable (non-metadata) in the .mat file.

    Args:
        file_path (Path): Path to the .mat file.

    Returns:
        np.ndarray: The main variable from the .mat file.

    Raises:
        FileNotFoundError: If the .mat file does not exist.
        ValueError: If the .mat file does not contain exactly one main variable.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    mat_data = loadmat(str(file_path))
    keys = [key for key in mat_data.keys() if not key.startswith('__')]
    if len(keys) != 1:
        raise ValueError(f"Expected one main variable in {file_path}")

    return mat_data[keys[0]]

def _load_test_data(dataset_name: str, pair_id: int, transpose=False) -> np.ndarray:
    """
    Load the test data for a specified dataset and pair ID.

    This function is intended for internal use by the evaluation module only.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz', 'PDE_KS').
        pair_id (int): ID of the train-test pair to load.
        transpose (bool): Whether to transpose the test data to (N, T). Defaults to false.
    Returns:
        np.ndarray: Test data array.

    Raises:
        ValueError: If the pair_id does not exist or the .mat file does not contain exactly one main variable.
        FileNotFoundError: If the test .mat file is not found.
    """
    config = get_config(dataset_name)
    pair = next((p for p in config['pairs'] if p['id'] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    test_matrix_name = pair.get('test', None)
    if test_matrix_name is None:
        raise ValueError(f"Provided pair_id {pair_id} does not have a test matrix in {dataset_name} config")

    test_file = top_dir / 'data' / dataset_name / 'test' / test_matrix_name
    test_data = _load_mat_file(test_file)
    if transpose:
        test_data = test_data.T
    return test_data

def load_dataset(dataset_name: str, pair_id: int, transpose=False) -> Tuple[list, Optional[np.ndarray]]:
    """
    Load the train and initialization data for a specified dataset and pair ID.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz', 'PDE_KS').
        pair_id (int): ID of the train-test pair to load.
        transpose (bool, optional): Whether to transpose the matrix. Defaults to True.
    Returns:
        Tuple[list, Optional[np.ndarray]]: A tuple containing:
            - A list of training data arrays.
            - An array of initialization data if available, otherwise None.

    Raises:
        ValueError: If the pair_id does not exist or any .mat file does not contain exactly one main variable.
        FileNotFoundError: If any of the train or initialization .mat files are not found.
    """
    config = get_config(dataset_name)
    pair = next((p for p in config['pairs'] if p['id'] == pair_id), None)
    if pair is None:
        raise ValueError(f"Provided pair_id {pair_id} does not exist in {dataset_name} config")

    # Load train data
    train_data = []
    if 'train' in pair:
        for data_file_name in pair['train']:
            train_file = top_dir / 'data' / dataset_name / 'train' / data_file_name
            train_data_i = _load_mat_file(train_file)
            if transpose:
                train_data_i = train_data_i.T
            train_data.append(train_data_i)
    else:
        raise ValueError(f"Provided pair_id {pair_id} does not have any training matrices in {dataset_name} config")

    # Load initialization data if available
    initialization_data = None
    if 'initialization' in pair:
        initialization_file = top_dir / 'data' / dataset_name / 'train' / pair['initialization']
        initialization_data = _load_mat_file(initialization_file)
        if transpose:
            initialization_data = initialization_data.T

    return train_data, initialization_data

def get_applicable_plots(dataset_name: str) -> List[str]:
    """
    Retrieve the list of applicable visualizations for the given dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz').

    Returns:
        List[str]: List of applicable plot types (e.g., ['trajectories', 'histograms']).

    Raises:
        ValueError: If no visualizations are defined in the dataset configuration.
    """
    dataset_config = get_config(dataset_name)
    
    applicable_plots = dataset_config.get('visualizations', [])
    if not applicable_plots:
        raise ValueError(f"No visualizations defined for dataset: {dataset_name}")
    
    return applicable_plots


def get_metadata(dataset_name: str) -> Dict:
    """
    Retrieve a dictionary of applicable metadata for the given dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz').

    Returns:
        Dict: Dictionary containing metadata (delta_t, spatial_dimension, etc.)

    Raises:
        ValueError: If no metadata are defined in the dataset configuration.
    """
    dataset_config = get_config(dataset_name)

    metadata = dataset_config.get('metadata', [])
    if not metadata:
        raise ValueError(f"No metadata defined for dataset: {dataset_name}")

    return metadata

def get_config(dataset_name: str) -> Dict[str, Any]:
    """
    Retrieve the dataset configuration file for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ODE_Lorenz').

    Returns:
        Dict: dataset configuration file

    Raises:
        FileNotFoundError: If the dataset configuration file is not found.
    """

    config_path = top_dir / 'data' / dataset_name / f'{dataset_name}.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {config_path}")

    with open(top_dir / 'data' / dataset_name / f'{dataset_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_validation_training_timesteps(dataset_name, pair_id, train_split=0.8):
    """
    Returns training timesteps for validation data.

    For pair_ids 2 and 4, validation data is equal to training data  
    For pair_ids 8 and 9, validation data is one of the training matrices (second matrix for pair_id 8, third matrix for pair_id 9) with burn_in matrix being a subset of the selected validation matrix (the same size as the original burn_in matrix)
    For the other pair_ids, split the single training matrix into the training and validation subset by using the train_split parameter
    split is performed.

    Args:
        dataset_name (str): Name of the dataset to load.
        pair_id (int): Identifier for the specific data pair to process (special handling for 8 and 9).
        train_split (float, optional): Proportion of data to use for training (0-1). Defaults to 0.8. (ignored for pair ids 2,4,8,9)

    Returns:
        (ndarray): Numpy array containing timesteps of validation matrix

    Raises:
        ValueError: If train_split is too small to accommodate the required data splits for the given pair_id.
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
            training_timesteps.pop(1)
        elif pair_id == 9:
            training_timesteps.pop(2)
        else:
            # Verify input
            if train_split <= 0.0 or train_split >= 1.0:
                raise ValueError(f"train_split of {train_split} is not in (0.0, 1.0).")
            # Calculate total number of training points
            train_num = int(train_split*train_data_all.shape[0])
            # Validation split is obtained from the only matrix in the train_data list
            training_timesteps_tmp = training_timesteps[0].copy()
            training_timesteps[0] = training_timesteps_tmp[0:train_num]
    elif pair_id in [2, 4]:
        pass
    else:
        raise ValueError(f"The provided pair_id {pair_id} is invalid.")

    return training_timesteps

def get_validation_prediction_timesteps(dataset_name, pair_id, train_split=0.8):
    """
    Returns prediction timesteps for validation data.

    For pair_ids 2 and 4, validation data is equal to training data  
    For pair_ids 8 and 9, validation data is one of the training matrices (second matrix for pair_id 8, third matrix for pair_id 9) with burn_in matrix being a subset of the selected validation matrix (the same size as the original burn_in matrix)
    For the other pair_ids, split the single training matrix into the training and validation subset by using the train_split parameter
    split is performed.

    Args:
        dataset_name (str): Name of the dataset to load.
        pair_id (int): Identifier for the specific data pair to process (special handling for 8 and 9).
        train_split (float, optional): Proportion of data to use for training (0-1). Defaults to 0.8. (ignored for pair ids 2,4,8,9)

    Returns:
        (ndarray): Numpy array containing timesteps of validation matrix

    Raises:
        ValueError: If train_split is too small to accommodate the required data splits for the given pair_id.
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
            train_num = int(train_split*train_data_all.shape[0])
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
    """
    Load and split dataset into training, validation, and initialization portions.
    Also returns training and prediction timesteps.

    For pair_ids 2 and 4, validation data is equal to training data  
    For pair_ids 8 and 9, validation data is one of the training matrices (second matrix for pair_id 8, third matrix for pair_id 9) with burn_in matrix being a subset of the selected validation matrix (the same size as the original burn_in matrix)
    For the other pair_ids, split the single training matrix into the training and validation subset by using the train_split parameter
    split is performed.

    Args:
        dataset_name (str): Name of the dataset to load.
        pair_id (int): Identifier for the specific data pair to process (special handling for 8 and 9).
        train_split (float, optional): Proportion of data to use for training (0-1). Defaults to 0.8. (ignored for pair ids 2,4,8,9)
        transpose: transpose the data (True) => (features, timesteps) or (False) => (timesteps, features)

    Returns:
        tuple: Contains three elements:
            - train_data (list): List of numpy arrays containing training data
            - val_data (ndarray): Validation data matrix
            - init_data (ndarray): Initialization data matrix (only populated for pair_id 8/9)

    Raises:
        ValueError: If train_split is too small to accommodate the required data splits for the given pair_id.
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
            train_num = int(train_split*train_data_all.shape[0])
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
        burn_in_num = init_data.shape[0]
        init_data = val_data.copy()
        init_data = init_data[0:burn_in_num, :]
    if transpose:
        return [td.T for td in train_data], val_data.T, (init_data.T if init_data is not None else None)

    return train_data, val_data, init_data
