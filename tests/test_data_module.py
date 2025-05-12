import os
import yaml
import tempfile
import unittest
import numpy as np
from scipy.io import savemat
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, mock_open
from ctf4science.data_module import parse_pair_ids, load_dataset, get_applicable_plots, get_metadata, get_config, _load_test_data, _load_mat_file, get_prediction_timesteps, get_training_timesteps, load_validation_dataset, get_validation_training_timesteps, get_validation_prediction_timesteps

class TestDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Called once before running all unit tests
        """

        # Create a temporary directory for testing
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.top_dir = Path(cls.temp_dir.name)
 
        # Mock the top_dir used in data_module
        cls.patcher = patch('ctf4science.data_module.top_dir', cls.top_dir)
        cls.patcher.start()

        # Parameters
        cls.delta_t = 0.025
        cls.train_start_idx = 0
        cls.train_len = 100
        cls.train_end_idx = cls.train_len - 1
        cls.test_start_idx = cls.train_end_idx + 1
        cls.test_len = 100
        cls.test_end_idx = cls.test_start_idx + cls.test_len - 1
        cls.init_start_idx = cls.train_end_idx + 1
        cls.init_len = 20
        cls.init_end_idx = cls.init_start_idx + cls.init_len - 1

        # Sample dataset configurations
        cls.dataset_config_0 = {
            'type': 'spatio-temporal',
            'evaluation_params': {
                'k': 2,
                'modes': 10
            },
            'pairs': [
                {
                    'id': 1,
                    'train': ['X1train.mat'],
                    'test': 'X1test.mat',
                    'metrics': ['short_time', 'long_time']
                },
                {
                    'id': 2,
                    'train': ['X1train.mat', 'X2train.mat', 'X3train.mat'],
                    'test': 'X1test.mat',
                    'initialization': 'X5train.mat',
                    'metrics': ['short_time', 'long_time'],
                }
            ],
            'metadata': {
                'delta_t': cls.delta_t,
                'spatial_dimension': 1,
                'matrix_shapes': {
                    'X1test.mat': [cls.test_len, 1],
                    'X1train.mat': [cls.train_len, 1]
                },
                'matrix_start_index': {
                    'X1test.mat': cls.test_start_idx,
                    'X1train.mat': cls.train_start_idx
                }
            },
            'evaluations': {
                'long_time': 'spectral_L2_error'
            },
            'visualizations': ['psd']
        }

        cls.dataset_config_1 = {
            'pairs': {
            },
        }

        # Tries to load bad matrices
        cls.dataset_config_2 = {
            'type': 'spatio-temporal',
            'evaluation_params': {
                'k': 2,
                'modes': 10
            },
            'pairs': [
                { # bad train matrix
                    'id': 1,
                    'train': ['X4train.mat'],
                    'test': 'X1test.mat',
                    'metrics': ['short_time', 'long_time']
                },
                { # bad test matrix
                    'id': 2,
                    'train': ['X1train.mat'],
                    'test': 'X2test.mat',
                    'metrics': ['short_time', 'long_time']
                },
                { # bad initialization matrix
                    'id': 3,
                    'train': ['X1train.mat'],
                    'test': 'X1test.mat',
                    'initialization': 'X4train.mat',
                    'metrics': ['short_time', 'long_time']
                },
                { # bad all matrices (train don't exist)
                    'id': 4,
                    'train': ['X999train.mat'],
                    'test': 'X1test.mat',
                    'initialization': 'X5train.mat',
                    'metrics': ['short_time', 'long_time']
                },
                { # bad all matrices (test doesn't exist)
                    'id': 5,
                    'train': ['X1train.mat'],
                    'test': 'X999test.mat',
                    'initialization': 'X5train.mat',
                    'metrics': ['short_time', 'long_time']
                },
                { # bad all matrices (init doesn't exist)
                    'id': 6,
                    'train': ['X1train.mat'],
                    'test': 'X1test.mat',
                    'initialization': 'X999train.mat',
                    'metrics': ['short_time', 'long_time']
                },
                { # test not even defined
                    'id': 7
                }
            ],
            'metadata': {
                'delta_t': cls.delta_t,
                'spatial_dimension': 1
            },
            'evaluations': {
                'long_time': 'spectral_L2_error'
            },
            'visualizations': ['psd']
        }

        # Bad config files for testing get_prediction_timesteps, get_training_timesteps
        cls.dataset_config_3 = {
            'pairs': [
                { # no test matrix
                    'id': 1,
                },
                { # yes test matrix
                    'id': 2,
                    'test': 'X1test.mat',
                    'train': ['X1train.mat']
                },
            ],
            'metadata': { # no 'matrix_shapes' or 'matrix_start_index
                'delta_t': cls.delta_t,
                'spatial_dimension': 1
            },
        }

        cls.dataset_config_4 = {
            'pairs': [
                { # no test matrix
                    'id': 1,
                },
                { # yes test matrix
                    'id': 2,
                    'test': 'X1test.mat',
                    'train': ['X1train.mat']
                },
            ],
            'metadata': { # yes 'matrix_shapes' no 'matrix_start_index (without test matrix)
                'matrix_shapes': {},
                'delta_t': cls.delta_t,
                'spatial_dimension': 1
            },
        }

        cls.dataset_config_5 = {
            'pairs': [
                { # no test matrix
                    'id': 1,
                },
                { # yes test matrix
                    'id': 2,
                    'test': 'X1test.mat',
                    'train': ['X1train.mat']
                },
            ],
            'metadata': { # yes 'matrix_shapes' yes 'matrix_start_index (without test matrix)
                'matrix_shapes': {},
                'matrix_start_index': {},
                'delta_t': cls.delta_t,
                'spatial_dimension': 1
            },
        }

        cls.dataset_config_6 = {
            'pairs': [
                { # no test matrix
                    'id': 1,
                },
                { # yes test matrix
                    'id': 2,
                    'test': 'X1test.mat',
                    'train': ['X1train.mat']
                },
            ],
            'metadata': { # yes 'matrix_shapes' yes 'matrix_start_index (with test matrix, wihout value in matrix_start_index)
                'matrix_shapes': {
                    'X1test.mat': [cls.test_len, 1],
                    'X1train.mat': [cls.train_len, 1]
                },
                'matrix_start_index': {},
                'delta_t': cls.delta_t,
                'spatial_dimension': 1
            },
        }

        cls.dataset_config_7 = {
            'pairs': [
                { # no test matrix
                    'id': 1,
                },
                { # yes test matrix
                    'id': 2,
                    'test': 'X1test.mat',
                    'train': ['X1train.mat']
                },
            ],
            'metadata': { # yes 'matrix_shapes' yes 'matrix_start_index (with test matrix, wihout value in matrix_start_index)
                'matrix_shapes': {
                    'X1test.mat': [cls.test_len, 1],
                    'X1train.mat': [cls.train_len, 1]
                },
                'matrix_start_index': {
                    'X1test.mat': cls.test_start_idx,
                    'X1train.mat': cls.train_start_idx
                },
                'spatial_dimension': 1
            },
        }

        # For testing load_validation-dataset
        cls.dataset_config_8 = {
            'pairs': [
                {
                    'id': 1,
                    'train': ['X1train.mat'],
                    'test': 'X1test.mat',
                    'metrics': ['short_time', 'long_time'],
                },
                {
                    'id': 2,
                    'train': ['X1train.mat'],
                    'test': 'X2test.mat',
                    'metrics': ['reconstruction'],
                },
                {
                    'id': 8,
                    'train': ['X1train.mat', 'X2train.mat', 'X3train.mat'],
                    'test': 'X1test.mat',
                    'initialization': 'X5train.mat',
                    'metrics': ['short_time', 'long_time'],
                },
                {
                    'id': 9,
                    'train': ['X1train.mat', 'X2train.mat', 'X3train.mat'],
                    'test': 'X1test.mat',
                    'initialization': 'X6train.mat',
                    'metrics': ['short_time', 'long_time'],
                },
                {
                    'id': 999,
                    'train': ['X1train.mat', 'X2train.mat', 'X3train.mat'],
                    'test': 'X1test.mat',
                    'initialization': 'X6train.mat',
                    'metrics': ['short_time', 'long_time'],
                }
            ],
            'metadata': { # yes 'matrix_shapes' yes 'matrix_start_index (with test matrix, wihout value in matrix_start_index)
                'matrix_shapes': {
                    'X1test.mat': [cls.test_len, 1],
                    'X1train.mat': [cls.train_len, 1],
                    'X2train.mat': [cls.train_len, 1],
                    'X3train.mat': [cls.train_len, 1],
                    'X5train.mat': [cls.init_len, 1],
                    'X6train.mat': [cls.init_len, 1],
                },
                'matrix_start_index': {
                    'X1test.mat': cls.test_start_idx,
                    'X1train.mat': cls.train_start_idx,
                    'X2train.mat': cls.train_start_idx+cls.train_len,
                    'X3train.mat': cls.train_start_idx+2*cls.train_len+cls.test_len,
                    'X5train.mat': cls.train_start_idx+cls.train_len*2,
                    'X6train.mat': cls.train_start_idx+cls.train_len*3+cls.test_len,
                },
                'spatial_dimension': 1,
                'delta_t': cls.delta_t,
            },
        }

        # Sample training and testing data
        cls.mock_train_data_1 = np.arange(cls.train_start_idx,cls.train_end_idx+1).reshape(-1,1)
        cls.mock_train_data_2 = np.arange(cls.train_start_idx,cls.train_end_idx+1).reshape(-1,1)*0.5
        cls.mock_train_data_3 = np.arange(cls.train_start_idx,cls.train_end_idx+1).reshape(-1,1)*0.25
        cls.mock_test_data = np.arange(cls.test_start_idx,cls.test_end_idx+1).reshape(-1,1)
        cls.mock_init_data_1 = np.arange(cls.init_start_idx,cls.init_end_idx+1).reshape(-1,1)
        cls.mock_init_data_2 = np.arange(cls.init_start_idx,cls.init_end_idx+1).reshape(-1,1)*0.5
        
        # Create mock .mat file structures
        cls.mock_train_mat_1 = {'__header__': b'header', '__version__': '1.0', '__globals__': None, 'ytrain': cls.mock_train_data_1} # X1train.mat
        cls.mock_train_mat_2 = {'__header__': b'header', '__version__': '1.0', '__globals__': None, 'ytrain': cls.mock_train_data_2} # X2train.mat
        cls.mock_train_mat_3 = {'__header__': b'header', '__version__': '1.0', '__globals__': None, 'ytrain': cls.mock_train_data_3} # X3train.mat
        cls.mock_train_mat_4 = {'__header__': b'header', '__version__': '1.0', '__globals__': None} # X4train.mat (bad matrix)
        cls.mock_init_mat_1 = {'__header__': b'header', '__version__': '1.0', '__globals__': None, 'ytrain': cls.mock_init_data_1} # X5train.mat
        cls.mock_init_mat_2 = {'__header__': b'header', '__version__': '1.0', '__globals__': None, 'ytrain': cls.mock_init_data_2} # X6train.mat
        cls.mock_test_mat_1 = {'__header__': b'header', '__version__': '1.0', '__globals__': None, 'ytest': cls.mock_test_data} # X1test.mat
        cls.mock_test_mat_2 = {'__header__': b'header', '__version__': '1.0', '__globals__': None} # X2test.mat (bad matrix)

        # Write to files in expected directory structure
        for i, yaml_content in enumerate([cls.dataset_config_0, cls.dataset_config_1, cls.dataset_config_2, cls.dataset_config_3, cls.dataset_config_4, cls.dataset_config_5, cls.dataset_config_6, cls.dataset_config_7, cls.dataset_config_8]):
            # Create directories
            (cls.top_dir / 'data' / f'dataset_{i}' / 'test').mkdir(parents=True, exist_ok=True)
            (cls.top_dir / 'data' / f'dataset_{i}' / 'train').mkdir(parents=True, exist_ok=True)
            # Write YAML dataset config files
            with open(cls.top_dir / 'data' / f'dataset_{i}' / f'dataset_{i}.yaml', 'w') as f:
                yaml.dump(yaml_content, f)
            # Write .mat files
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'train' / 'X1train.mat', cls.mock_train_mat_1)
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'train' / 'X2train.mat', cls.mock_train_mat_2)
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'train' / 'X3train.mat', cls.mock_train_mat_3)
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'train' / 'X4train.mat', cls.mock_train_mat_4)
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'train' / 'X5train.mat', cls.mock_init_mat_1)
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'train' / 'X6train.mat', cls.mock_init_mat_2)
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'test' / 'X1test.mat', cls.mock_test_mat_1)
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'test' / 'X2test.mat', cls.mock_test_mat_2)

    @classmethod
    def tearDownClass(cls):
        """
        Called once after running all unit tests
        """
        cls.temp_dir.cleanup()
        cls.patcher.stop()
        pass

    def setUp(self):
        """
        Called once before each unit test
        """
        pass

    def tearDown(self):
        """
        Called once after each unit test
        """
        pass

    def test_missing_dataset_name(self):
        """
        Test that ValueError is raised when dataset name is missing
        """
        with self.assertRaises(ValueError) as context:
            parse_pair_ids({})
        self.assertIn("Dataset name must be specified in dataset_config", str(context.exception))

    def test_missing_config_file(self):
        """
        Test that FileNotFoundError is raised when config file doesn't exist
        """
        with self.assertRaises(FileNotFoundError) as context:
            get_config('dataset_999')
        self.assertIn("Dataset configuration file not found", str(context.exception))

    def test_no_pairs(self):
        """
        Test that ValueError is raised when config file exists but has no pairs defined
        """
        with self.assertRaises(ValueError) as context:
            parse_pair_ids({'name': 'dataset_1'})
        self.assertIn("No pairs defined in dataset configuration", str(context.exception))

    def test_no_requested_pair_id_int(self):
        """
        Test that ValueError is raised when config file exists, it has pairs, but the specific pair does not exist (case where pair_id is an int)
        """
        run_config = {
            "name": "dataset_0",
            "pair_id": 999
        }
        with self.assertRaises(ValueError) as context:
            parse_pair_ids(run_config)
        self.assertIn(f"Requested pair_id 999 not in dataset_0 dataset", str(context.exception))

    def test_no_requested_pair_id_range(self):
        """
        Test that ValueError is raised when config file exists, it has pairs, but the specific pair does not exist (case where pair_id is a range of integers)
        """
        run_config = {
            "name": "dataset_0",
            "pair_id": "1-3"
        }
        with self.assertRaises(ValueError) as context:
            parse_pair_ids(run_config)
        self.assertIn(f"Requested pair_ids [3] not in dataset_0 dataset", str(context.exception))

    def test_invalid_pair_id_range(self):
        """
        Test that ValueError is raised when config file exists, it has pairs, but the specific pair does not exist (case where pair_id is an invalid range)
        """
        run_config = {
            "name": "dataset_0",
            "pair_id": "a-b"
        }
        with self.assertRaises(ValueError) as context:
            parse_pair_ids(run_config)
        self.assertIn(f"Invalid range format in pair_id: a-b", str(context.exception))

    def test_invalid_pair_id_list(self):
        """
        Test that ValueError is raised when config file exists, it has pairs, but the specific pair does not exist (case where pair_id is a list with an invalid pair_id)
        """
        run_config = {
            "name": "dataset_0",
            "pair_id": [1,2,999]
        }
        with self.assertRaises(ValueError) as context:
            parse_pair_ids(run_config)
        self.assertIn(f"Requested pair_ids [999] not in dataset_0 dataset", str(context.exception))

    def test_invalid_pair_id_configuration(self):
        """
        Test that ValueError is raised when config file exists, it has pairs, but the specific pair does not exist (case where pair_id is just nonsense)
        """
        run_config = {
            "name": "dataset_0",
            "pair_id": "beans"
        }
        with self.assertRaises(ValueError) as context:
            parse_pair_ids(run_config)
        self.assertIn(f"Invalid pair_id configuration: beans", str(context.exception))

    def test_valid_pair_id_constant(self):
        """
        Test that ValueError is raised when config file exists, it has pairs, but the specific pair does not exist (case where pair_id is an invalid range)
        """
        run_config = {
            "name": "dataset_0",
            "pair_id": 1
        }
        self.assertListEqual([1], parse_pair_ids(run_config))

    def test_valid_pair_id_range(self):
        """
        Test that ValueError is raised when config file exists, it has pairs, but the specific pair does not exist (case where pair_id is an invalid range)
        """
        run_config = {
            "name": "dataset_0",
            "pair_id": "1-2"
        }
        self.assertListEqual([1,2], parse_pair_ids(run_config))

    def test_valid_pair_id_list(self):
        """
        Test that ValueError is raised when config file exists, it has pairs, but the specific pair does not exist (case where pair_id is a list with an invalid pair_id)
        """
        run_config = {
            "name": "dataset_0",
            "pair_id": [1,2]
        }
        self.assertListEqual([1,2], parse_pair_ids(run_config))

    def test_load_mat_file_fail(self):
        """
        Fail on loading a matrix because it doesn't have a main variable
        """
        file_path = self.top_dir / 'data' / 'dataset_0' / 'train' / "X4train.mat"
        with self.assertRaises(ValueError) as context:
            _load_mat_file(file_path)
        self.assertIn(f"Expected one main variable in", str(context.exception))

    def test_load_test_data_success(self):
        """
        Successfully load test data for a valid dataset and pair ID
        """
        test_data = _load_test_data('dataset_0', 1)
        np.testing.assert_array_equal(test_data, self.mock_test_data)

    def test_load_test_data_fail_not_in_config(self):
        """
        Fail to load test data because it's not defined in the config file
        """
        with self.assertRaises(ValueError) as context:
            _load_test_data('dataset_2', 7)
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not have a test matrix in", str(context.exception))
        self.assertIn("config", str(context.exception))

    def test_load_test_data_fail_file_not_found(self):
        """
        Fail to load test data because the .mat file does not exist
        """
        with self.assertRaises(FileNotFoundError) as context:
            _load_test_data('dataset_2', 5)
        self.assertIn("Dataset file not found", str(context.exception))

    def test_load_test_data_fail_no_main_variable(self):
        """
        Fail to load test data because the .mat file does not contain exactly one main variable
        """
        with self.assertRaises(ValueError) as context:
            _load_test_data('dataset_2', 2)
        self.assertIn("Expected one main variable in", str(context.exception))

    def test_load_test_data_large_pair_id(self):
        """
        Fail to load test data because the pair_id doesn't exist
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(_load_test_data('dataset_0', 999))
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not exist in", str(context.exception))

    def test_load_dataset_success(self):
        """
        Load two training, a testing, and an initialization matrix successfully
        """
        train_data, initialization_data = load_dataset('dataset_0', 2)

        self.assertEqual(len(train_data), 3)
        np.testing.assert_array_equal(train_data[0], self.mock_train_data_1)
        np.testing.assert_array_equal(train_data[1], self.mock_train_data_2)
        np.testing.assert_array_equal(train_data[2], self.mock_train_data_3)
        np.testing.assert_array_equal(initialization_data, self.mock_init_data_1)

    def test_load_test_data_fail_not_in_config(self):
        """
        Fail to load test data because it's not defined in the config file
        """
        with self.assertRaises(ValueError) as context:
            _load_test_data('dataset_2', 7)
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not have a test matrix in", str(context.exception))
        self.assertIn("config", str(context.exception))

    def test_load_dataset_fail_train(self):
        """
        Fail on loading training matrix because it doesn't have a main variable
        """
        with self.assertRaises(ValueError) as context:
            load_dataset('dataset_2', 1)
        self.assertIn(f"Expected one main variable in", str(context.exception))

    def test_load_dataset_fail_initialization_bad_matrix(self):
        """
        Fail on loading initialization matrix because it doesn't have a main variable
        """
        with self.assertRaises(ValueError) as context:
            load_dataset('dataset_2', 3)
        self.assertIn(f"Expected one main variable in", str(context.exception))

    def test_load_dataset_fail_train_not_found(self):
        """
        Fail on loading training matrix because it doesn't exist
        """
        with self.assertRaises(FileNotFoundError) as context:
            load_dataset('dataset_2', 4)
        self.assertIn("Dataset file not found", str(context.exception))

    def test_load_dataset_fail_initialization_not_found(self):
        """
        Fail on loading initialization matrix it doesn't exist
        """
        with self.assertRaises(FileNotFoundError) as context:
            load_dataset('dataset_2', 6)
        self.assertIn("Dataset file not found", str(context.exception))

    def test_load_dataset_None_initialization(self):
        """
        Loading initialization matrix returns None because it doesn't exist
        """
        _, initialization_data = load_dataset('dataset_0', 1)
        self.assertIsNone(initialization_data)

    def test_load_dataset_large_pair_id(self):
        """
        Fail to load dataset because the pair_id doesn't exist
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(load_dataset('dataset_0', 999))
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not exist in", str(context.exception))

    def test_get_applicable_plots_success(self):
        """
        Successfully get a list of applicable visualizations for the given dataset
        """
        plots = get_applicable_plots('dataset_0')
        self.assertListEqual(['psd'], plots)

    def test_get_applicable_plots_failure_undefined(self):
        """
        Fail getting a list of applicable visualizations for the given dataset because a list of visualizations is not defined in the configuration file
        """
        with self.assertRaises(ValueError) as context:
            get_applicable_plots('dataset_1')
        self.assertIn("No visualizations defined for dataset", str(context.exception))

    def test_get_metadata_success(self):
        """
        Successfully get a list of metadata for the given dataset
        """
        d =  {
            'delta_t': self.delta_t,
            'spatial_dimension': 1
        }
        metadata = get_metadata('dataset_2')
        self.assertDictEqual(d, metadata)

    def test_get_metadata_failure_undefined(self):
        """
        Fail getting a list of metadata for the given dataset because a list of metadata is not defined in the configuration file
        """
        with self.assertRaises(ValueError) as context:
            get_metadata('dataset_1')
        self.assertIn("No metadata defined for dataset", str(context.exception))

    def test_get_prediction_timesteps_success(self):
        """
        Successfully get the numpy vector containing the time steps at which the prediction matrices need to be evaluated
        """
        expected_timesteps = np.linspace(self.test_start_idx, self.test_end_idx, self.test_len)*self.delta_t
        produced_timesteps = get_prediction_timesteps('dataset_0', 1)

        np.testing.assert_array_almost_equal(expected_timesteps, produced_timesteps, decimal=6)

    def test_get_prediction_timesteps_fail_pair_id_not_in_config(self):
        """
        Fail because pair_id doesn't exist in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_prediction_timesteps('dataset_3', 999))
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not exist in", str(context.exception))
        self.assertIn("config", str(context.exception))

    def test_get_prediction_timesteps_fail_test_not_in_config(self):
        """
        Fail because test does not exist in pair_id in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_prediction_timesteps('dataset_3', 1))
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not have a test matrix in", str(context.exception))
        self.assertIn("config", str(context.exception))

    def test_get_prediction_timesteps_fail_matrix_shapes_not_in_config(self):
        """
        Fail because matrix_shapes does not exist in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_prediction_timesteps('dataset_3', 2))
        self.assertIn("Provided", str(context.exception))
        self.assertIn("config does not have 'matrix_shapes'", str(context.exception))

    def test_get_prediction_timesteps_fail_matrix_start_index_not_in_config(self):
        """
        Fail because matrix_start_index does not exist in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_prediction_timesteps('dataset_4', 2))
        self.assertIn("Provided", str(context.exception))
        self.assertIn("config does not have 'matrix_start_index'", str(context.exception))

    def test_get_prediction_timesteps_fail_test_matrix_shape_not_in_config(self):
        """
        Fail because test matrix does not have a matrix shape in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_prediction_timesteps('dataset_5', 2))
        self.assertIn("Provided test matrix", str(context.exception))
        self.assertIn("for pair_id", str(context.exception))
        self.assertIn("config does not have a shape", str(context.exception))

    def test_get_prediction_timesteps_fail_test_matrix_start_index_not_in_config(self):
        """
        Fail because test matrix does not have a start index in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_prediction_timesteps('dataset_6', 2))
        self.assertIn("Provided test matrix", str(context.exception))
        self.assertIn("for pair_id", str(context.exception))
        self.assertIn("config does not have a starting index", str(context.exception))

    def test_get_prediction_timesteps_fail_delta_t_not_in_config(self):
        """
        Fail because delta_t does not exist in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_prediction_timesteps('dataset_7', 2))
        self.assertIn("Provided", str(context.exception))
        self.assertIn("config does not have a delta_t", str(context.exception))

    def test_get_prediction_timesteps_invalid_subset(self):
        """
        Fail because subset is not valid
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_prediction_timesteps('dataset_7', 2, subset="beans"))
        self.assertIn("Subset", str(context.exception))
        self.assertIn("is not valid", str(context.exception))

    def test_get_training_timesteps_success(self):
        """
        Successfully get the numpy vector containing the time steps at which the training matrices need to be evaluated
        """
        expected_timesteps = np.linspace(self.train_start_idx, self.train_end_idx, self.train_len)*self.delta_t
        produced_timesteps = get_training_timesteps('dataset_0', 1)

        self.assertEqual(len(produced_timesteps), 1)
        np.testing.assert_array_almost_equal(expected_timesteps, produced_timesteps[0], decimal=6)

    def test_get_training_timesteps_fail_pair_id_not_in_config(self):
        """
        Fail because pair_id doesn't exist in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_training_timesteps('dataset_3', 999))
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not exist in", str(context.exception))
        self.assertIn("config", str(context.exception))

    def test_get_training_timesteps_fail_test_not_in_config(self):
        """
        Fail because test does not exist in pair_id in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_training_timesteps('dataset_3', 1))
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not have a train matrix in", str(context.exception))
        self.assertIn("config", str(context.exception))

    def test_get_training_timesteps_fail_matrix_shapes_not_in_config(self):
        """
        Fail because matrix_shapes does not exist in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_training_timesteps('dataset_3', 2))
        self.assertIn("Provided", str(context.exception))
        self.assertIn("config does not have 'matrix_shapes'", str(context.exception))

    def test_get_training_timesteps_fail_matrix_start_index_not_in_config(self):
        """
        Fail because matrix_start_index does not exist in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_training_timesteps('dataset_4', 2))
        self.assertIn("Provided", str(context.exception))
        self.assertIn("config does not have 'matrix_start_index'", str(context.exception))

    def test_get_training_timesteps_fail_test_matrix_shape_not_in_config(self):
        """
        Fail because test matrix does not have a matrix shape in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_training_timesteps('dataset_5', 2))
        self.assertIn("Provided train matrix", str(context.exception))
        self.assertIn("for pair_id", str(context.exception))
        self.assertIn("config does not have a shape", str(context.exception))

    def test_get_training_timesteps_fail_test_matrix_start_index_not_in_config(self):
        """
        Fail because test matrix does not have a start index in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_training_timesteps('dataset_6', 2))
        self.assertIn("Provided train matrix", str(context.exception))
        self.assertIn("for pair_id", str(context.exception))
        self.assertIn("config does not have a starting index", str(context.exception))

    def test_get_training_timesteps_fail_delta_t_not_in_config(self):
        """
        Fail because delta_t does not exist in metadata in config
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_training_timesteps('dataset_7', 2))
        self.assertIn("Provided", str(context.exception))
        self.assertIn("config does not have a delta_t", str(context.exception))

    def test_get_validation_training_timesteps_success_8(self):
        """
        Successfully get validation training timesteps
        Pair_id=8 case (forecasting, interpolative)
        """
        # Generate output
        training_timesteps = get_validation_training_timesteps('dataset_8', 8, 0.8)

        # Make sure training timesteps look as expected
        self.assertEqual(len(training_timesteps), 2)
        expected_training_timesteps = (np.linspace(self.train_start_idx, self.train_end_idx, self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[0], decimal=6)

        expected_training_timesteps = (np.linspace(self.train_start_idx+self.train_len*3, self.train_end_idx+self.train_len*3, self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[1], decimal=6)

    def test_get_validation_prediction_timesteps_success_8(self):
        """
        Successfully get validation prediction timesteps
        Pair_id=8 case (forecasting, interpolative)
        """
        # Generate output
        prediction_timesteps = get_validation_prediction_timesteps('dataset_8', 8, 0.8)

        # Make sure prediction timesteps look as expected
        expected_prediction_timesteps = (np.arange(self.train_start_idx+self.train_len, self.train_start_idx+2*self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_prediction_timesteps, prediction_timesteps, decimal=6)

    def test_load_validation_dataset_success_8(self):
        """
        Successfully split dataset into training, validation, and initialization portions
        Pair_id=8 case (forecasting, interpolative)

        We expect the roughly the following:
            The original data has 4 matrices:
                train[0]:  [x1 x2 x3 x4]
                train[1]:  [y1 y2 y3 y4]
                train[2]:  [z1 z2 z3 z4]
                init_data: [b1 b2 b3 b4]
            The output should look like this:
                train[0]:  [x1 x2 x3 x4]
                train[2]:  [z1 z2 z3 z4]
                init_data: [y1 y2]
                val_data:  [y1 y2 y3 y4]
        """
        # Expected output
        exp_train_data = list()
        exp_train_data.append(self.mock_train_data_1)
        exp_train_data.append(self.mock_train_data_3)

        exp_init_data = self.mock_train_data_2[0:self.init_len,:]
        exp_val_data = self.mock_train_data_2

        # Generate output
        train_data, val_data, initialization_data = load_validation_dataset('dataset_8', 8, 0.8)

        # Make data sure it looks as expected
        self.assertEqual(len(train_data), 2)
        np.testing.assert_array_equal(train_data[0], exp_train_data[0])
        np.testing.assert_array_equal(train_data[1], exp_train_data[1])
        np.testing.assert_array_equal(val_data, exp_val_data)
        np.testing.assert_array_equal(initialization_data, exp_init_data)

    def test_get_validation_prediction_timesteps_success_9(self):
        """
        Successfully get validation prediction timesteps
        Pair_id=9 case (forecasting, interpolative)
        """
        # Generate output
        prediction_timesteps = get_validation_prediction_timesteps('dataset_8', 9, 0.8)

        # Make sure prediction timesteps look as expected
        expected_prediction_timesteps = (np.arange(self.train_start_idx+self.train_len*3, self.train_start_idx+self.train_len*4)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_prediction_timesteps, prediction_timesteps, decimal=6)

    def test_get_validation_training_timesteps_success_9(self):
        """
        Successfully get validation prediction timesteps
        Pair_id=9 case (forecasting, interpolative)
        """
        # Generate output
        training_timesteps = get_validation_training_timesteps('dataset_8', 9, 0.8)

        # Make sure training timesteps look as expected
        self.assertEqual(len(training_timesteps), 2)
        expected_training_timesteps = (np.linspace(self.train_start_idx, self.train_end_idx, self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[0], decimal=6)

        expected_training_timesteps = (np.linspace(self.train_start_idx+self.train_len, self.train_end_idx+self.train_len, self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[1], decimal=6)

    def test_load_validation_dataset_success_9(self):
        """
        Successfully split dataset into training, validation, and initialization portions
        Pair_id=9 case (forecasting, extrapolative)

        We expect the roughly the following:
            The original data has 4 matrices:
                train[0]:  [x1 x2 x3 x4]
                train[1]:  [y1 y2 y3 y4]
                train[2]:  [z1 z2 z3 z4]
                init_data: [b1 b2 b3 b4]
            The output should look like this:
                train[0]:  [x1 x2 x3 x4]
                train[1]:  [y1 y2 y3 y4]
                init_data: [z1 z2]
                val_data:  [z1 z2 z3 z4]
        """
        # Expected output
        exp_train_data = list()
        exp_train_data.append(self.mock_train_data_1)
        exp_train_data.append(self.mock_train_data_2)

        exp_init_data = self.mock_train_data_3[0:self.init_len,:]
        exp_val_data = self.mock_train_data_3

        # Generate output
        train_data, val_data, initialization_data = load_validation_dataset('dataset_8', 9, 0.8)
        training_timesteps = get_validation_training_timesteps('dataset_8', 9, 0.8)
        prediction_timesteps = get_validation_prediction_timesteps('dataset_8', 9, 0.8)

        # Make data sure it looks as expected
        self.assertEqual(len(train_data), 2)
        np.testing.assert_array_equal(train_data[0], exp_train_data[0])
        np.testing.assert_array_equal(train_data[1], exp_train_data[1])
        np.testing.assert_array_equal(val_data, exp_val_data)
        np.testing.assert_array_equal(initialization_data, exp_init_data)

        # Make sure training timesteps look as expected
        self.assertEqual(len(training_timesteps), 2)
        expected_training_timesteps = (np.linspace(self.train_start_idx, self.train_end_idx, self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[0], decimal=6)

        expected_training_timesteps = (np.linspace(self.train_start_idx+self.train_len, self.train_end_idx+self.train_len, self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[1], decimal=6)

        # Make sure prediction timesteps look as expected
        expected_prediction_timesteps = (np.arange(self.train_start_idx+self.train_len*3, self.train_start_idx+self.train_len*4)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_prediction_timesteps, prediction_timesteps, decimal=6)

    def test_get_validation_prediction_timesteps_success_1(self):
        """
        Successfully get validation prediction timesteps
        Pair_id=1 case (forecasting, interpolative)
        """
        # Generate output
        prediction_timesteps = get_validation_prediction_timesteps('dataset_8', 1, 0.8)

        # Make sure prediction timesteps look as expected
        expected_prediction_timesteps = (np.arange(self.train_start_idx+80, self.train_start_idx+100)*self.delta_t)
        np.testing.assert_array_almost_equal(prediction_timesteps, expected_prediction_timesteps, decimal=6)

    def test_get_validation_training_timesteps_success_1(self):
        """
        Successfully get validation prediction timesteps
        Pair_id=1 case (forecasting, interpolative)
        """
        # Generate output
        training_timesteps = get_validation_training_timesteps('dataset_8', 1, 0.8)

        # Make sure training timesteps look as expected
        self.assertEqual(len(training_timesteps), 1)
        expected_training_timesteps = (np.linspace(self.train_start_idx, self.train_end_idx, self.train_len)*self.delta_t)[0:80]
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[0], decimal=6)

    def test_load_validation_dataset_success_1(self):
        """
        Successfully split dataset into training, validation, and initialization portions
        Pair_id=1 case (forecasting)

        We expect the roughly the following:
            The original data has 4 matrices:
                train[0]:  [x1 x2 x3 x4]
            The output should look like this:
                train[0]:  [x1 x2 x3]
                val_data:  [x4]
        """
        # Expected output
        exp_train_data = list()
        exp_train_data.append(self.mock_train_data_1[0:80,:])

        exp_val_data = self.mock_train_data_1[80:,:]

        # Generate output
        train_data, val_data, initialization_data = load_validation_dataset('dataset_8', 1, 0.8)
        training_timesteps = get_validation_training_timesteps('dataset_8', 1, 0.8)
        prediction_timesteps = get_validation_prediction_timesteps('dataset_8', 1, 0.8)

        # Make sure it looks as expected
        self.assertEqual(len(train_data), 1)
        np.testing.assert_array_equal(train_data[0], exp_train_data[0])
        np.testing.assert_array_equal(val_data, exp_val_data)
        self.assertIsNone(initialization_data)

        # Make sure training timesteps look as expected
        self.assertEqual(len(training_timesteps), 1)
        expected_training_timesteps = (np.linspace(self.train_start_idx, self.train_end_idx, self.train_len)*self.delta_t)[0:80]
        np.testing.assert_array_almost_equal(training_timesteps[0], expected_training_timesteps, decimal=6)

        # Make sure prediction timesteps look as expected
        expected_prediction_timesteps = (np.arange(self.train_start_idx+80, self.train_start_idx+100)*self.delta_t)
        np.testing.assert_array_almost_equal(prediction_timesteps, expected_prediction_timesteps, decimal=6)

    def test_get_validation_prediction_timesteps_success_2(self):
        """
        Successfully get validation prediction timesteps
        Pair_id=2 case (forecasting, interpolative)
        """
        # Generate output
        prediction_timesteps = get_validation_prediction_timesteps('dataset_8', 2, 0.8)

        # Make sure prediction timesteps look as expected
        expected_prediction_timesteps = (np.arange(self.train_start_idx, self.train_start_idx+self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_prediction_timesteps, prediction_timesteps, decimal=6)

    def test_get_validation_training_timesteps_success_2(self):
        """
        Successfully get validation prediction timesteps
        Pair_id=2 case (forecasting, interpolative)
        """
        # Generate output
        training_timesteps = get_validation_training_timesteps('dataset_8', 2, 0.8)

        # Make sure training timesteps look as expected
        self.assertEqual(len(training_timesteps), 1)
        expected_training_timesteps = (np.linspace(self.train_start_idx, self.train_end_idx, self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[0], decimal=6)

    def test_load_validation_dataset_success_2(self):
        """
        Successfully split dataset into training, validation, and initialization portions
        Pair_id=2 case (reconstruction)

        We expect the roughly the following:
            The original data has 4 matrices:
                train[0]:  [x1 x2 x3 x4]
            The output should look like this:
                train[0]:  [x1 x2 x3 x4]
                val_data:  [x1 x2 x3 x4]
        """
        # Expected output
        exp_train_data = list()
        exp_train_data.append(self.mock_train_data_1)

        exp_val_data = self.mock_train_data_1

        # Generate output
        train_data, val_data, initialization_data = load_validation_dataset('dataset_8', 2, 0.8)
        training_timesteps = get_validation_training_timesteps('dataset_8', 2, 0.8)
        prediction_timesteps = get_validation_prediction_timesteps('dataset_8', 2, 0.8)

        # Make sure it looks as expected
        self.assertEqual(len(train_data), 1)
        np.testing.assert_array_equal(train_data[0], exp_train_data[0])
        np.testing.assert_array_equal(val_data, exp_val_data)
        self.assertIsNone(initialization_data)

        # Make sure training timesteps look as expected
        self.assertEqual(len(training_timesteps), 1)
        expected_training_timesteps = (np.linspace(self.train_start_idx, self.train_end_idx, self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_training_timesteps, training_timesteps[0], decimal=6)

        # Make sure prediction timesteps look as expected
        expected_prediction_timesteps = (np.arange(self.train_start_idx, self.train_start_idx+self.train_len)*self.delta_t)
        np.testing.assert_array_almost_equal(expected_prediction_timesteps, prediction_timesteps, decimal=6)

    def test_get_validation_training_timesteps_fail_train_split(self):
        """
        Fail splitting validation training timesteps

        Fail because training split is not between 0.0 and 1.0
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_validation_training_timesteps('dataset_8', 1, 2.0))
        self.assertIn("train_split of", str(context.exception))
        self.assertIn("is not in", str(context.exception))

    def test_get_validation_prediction_timesteps_fail_train_split(self):
        """
        Fail splitting validation prediction timesteps

        Fail because training split is not between 0.0 and 1.0
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_validation_prediction_timesteps('dataset_8', 1, 2.0))
        self.assertIn("train_split of", str(context.exception))
        self.assertIn("is not in", str(context.exception))

    def test_load_validation_dataset_fail_train_split(self):
        """
        Fail splitting dataset into training, validation, and initialization portions

        Fail because training split is not between 0.0 and 1.0
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(load_validation_dataset('dataset_8', 1, 2.0))
        self.assertIn("train_split of", str(context.exception))
        self.assertIn("is not in", str(context.exception))

    def test_get_validation_training_timesteps_fail_pair_id(self):
        """
        Fail splitting validation training timesteps

        Fail because pair_id of 999 is invalid (but exists in dataset)
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_validation_training_timesteps('dataset_8', 999, .8))
        self.assertIn("The provided pair_id", str(context.exception))
        self.assertIn("is invalid", str(context.exception))

    def test_get_validation_prediction_timesteps_fail_pair_id(self):
        """
        Fail splitting validation prediction timesteps

        Fail because pair_id of 999 is invalid (but exists in dataset)
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(get_validation_prediction_timesteps('dataset_8', 999, .8))
        self.assertIn("The provided pair_id", str(context.exception))
        self.assertIn("is invalid", str(context.exception))

    def test_load_validation_dataset_fail_pair_id(self):
        """
        Fail splitting dataset into training, validation, and initialization portions

        Fail because pair_id of 999 is invalid (but exists in dataset)
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(load_validation_dataset('dataset_8', 999, .8))
        self.assertIn("The provided pair_id", str(context.exception))
        self.assertIn("is invalid", str(context.exception))
