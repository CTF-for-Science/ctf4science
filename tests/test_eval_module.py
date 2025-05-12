import os
import yaml
import tempfile
import unittest
import numpy as np
from scipy.io import savemat
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, mock_open
from ctf4science.eval_module import short_time_forecast, reconstruction, long_time_forecast_dynamical, compute_log_psd, long_time_forecast_spatio_temporal, evaluate, evaluate_custom, save_results, compute_psd, extract_metrics_in_order

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
        cls.patcher1 = patch('ctf4science.eval_module.top_dir', cls.top_dir)
        cls.patcher2 = patch('ctf4science.data_module.top_dir', cls.top_dir)
        cls.patcher1.start()
        cls.patcher2.start()

        # Set up test parameters
        cls.spectral_k = 20
        cls.histogram_k = 1000
        cls.short_time_k = 20
        cls.modes = 5
        cls.bins = 50

        # Sample dataset configurations
        cls.dataset_config_0 = {
            'type': 'spatio-temporal',
            'evaluation_params': {
                'k': 20,
                'modes': 5
            },
            'pairs': [
                {
                    'test': 'X1test.mat',
                    'id': 1,
                    'metrics': ['short_time', 'long_time', 'reconstruction']
                },
            ],
            'evaluations': {
                'long_time': 'spectral_L2_error'
            }
        }

        cls.dataset_config_1 = {
            'type': 'dynamical',
            'evaluation_params': {
                'k': 20,
                'modes': 1000,
                'bins': 50
            },
            'pairs': [
                {
                    'test': 'X1test.mat',
                    'id': 1,
                    'metrics': []
                },
            ],
            'evaluations': {
                'long_time': 'histogram_L2_error'
            }
        }

        cls.dataset_config_2 = {
            'type': 'dynamical',
            'evaluation_params': {
                'k': 1000,
                'bins': 50
            },
            'pairs': [
                {
                    'test': 'X1test.mat',
                    'id': 1,
                    'metrics': ['long_time']
                },
            ],
            'evaluations': {
                'long_time': 'not_implemented_error'
            }
        }

        cls.dataset_config_3 = {
            'type': 'dynamical',
            'evaluation_params': {
                'k': 1000,
                'bins': 50
            },
            'pairs': [
                {
                    'id': 1,
                    'metrics': ['long_time', 'short_time']
                },
                {
                    'id': 2,
                    'metrics': ['long_time', 'short_time']
                },
            ],
            'evaluations': {
                'long_time': 'not_implemented_error'
            }
        }

        cls.batch_results_1 = {
            "batch_id": "batch_example",
            "dataset": "dataset_example",
            "model": "model_example",
            "pairs": [
                {
                    "metrics": {
                        "long_time": 50.0,
                        "short_time": 10.0
                    },
                    "pair_id": 1
                },
                {
                    "metrics": {
                        "long_time": 50.0,
                    },
                    "pair_id": 2
                },
            ]
        }

        cls.batch_results_2 = {
            "batch_id": "batch_example",
            "dataset": "dataset_example",
            "model": "model_example",
            "pairs": [
                {
                    "metrics": {
                        "long_time": 50.0,
                        "short_time": 10.0
                    },
                    "pair_id": 1
                },
            ]
        }

        # Create train/test matrices
        cls.mock_test_data = np.tile(np.arange(100,200),(20,1)).T
        cls.mock_test_mat_1 = {'__header__': b'header', '__version__': '1.0', '__globals__': None, 'ytest': cls.mock_test_data} # X1test.mat

        # Write to files in expected directory structure
        for i, yaml_content in enumerate([cls.dataset_config_0, cls.dataset_config_1, cls.dataset_config_2, cls.dataset_config_3]):
            # Write YAML dataset config files
            (cls.top_dir / 'data' / f'dataset_{i}' / 'test').mkdir(parents=True, exist_ok=True)
            with open(cls.top_dir / 'data' / f'dataset_{i}' / f'dataset_{i}.yaml', 'w') as f:
                yaml.dump(yaml_content, f)
            # Write .mat files
            savemat(cls.top_dir / 'data' / f'dataset_{i}' / 'test' / 'X1test.mat', cls.mock_test_mat_1)

    @classmethod
    def tearDownClass(cls):
        """
        Called once after running all unit tests
        """
        cls.temp_dir.cleanup()
        cls.patcher1.stop()
        cls.patcher2.stop()
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

    def test_extract_metrics_in_order_success(self):
        """
        Call extract_metrics_in_order successfully
        """
        self.assertListEqual(extract_metrics_in_order('dataset_2', self.batch_results_1), [50.0])

    def test_extract_metrics_in_order_pair_id_not_found(self):
        """
        Call extract_metrics_in_order unsuccessfully
        Tests to make sure we throw a ValueError when a pair_id is not found
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(extract_metrics_in_order('dataset_3', self.batch_results_2))
        self.assertIn(f"not found in batch_results", str(context.exception))

    def test_extract_metrics_in_order_metric_not_found(self):
        """
        Call extract_metrics_in_order unsuccessfully
        Tests to make sure we throw a ValueError when a metric for a pair_id is not found
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(extract_metrics_in_order('dataset_3', self.batch_results_1))
        self.assertIn(f"not found for pair ID", str(context.exception))

    def test_short_time_forecast_success(self):
        """
        Call short_time_forecast successfully
        Tests to make sure we get a zero result and a 100 result
        """
        truth = np.ones((100,10))
        prediction_zero = np.zeros((100,10))
        prediction_100 = np.copy(truth)

        res_zero = short_time_forecast(truth, prediction_zero, self.short_time_k)
        res_100 = short_time_forecast(truth, prediction_100, self.short_time_k)

        self.assertAlmostEqual(res_zero, 0., places=6)
        self.assertAlmostEqual(res_100, 100., places=6)

    def test_reconstruction_success(self):
        """
        Call reconstruction successfully
        Tests to make sure we get a zero result and a 100 result
        """
        truth = np.ones((100,10))
        prediction_zero = np.zeros((100,10))
        prediction_100 = np.copy(truth)

        res_zero = reconstruction(truth, prediction_zero)
        res_100 = reconstruction(truth, prediction_100)

        self.assertAlmostEqual(res_zero, 0., places=6)
        self.assertAlmostEqual(res_100, 100., places=6)

    def test_long_time_forecast_dynamical_success(self):
        """
        Call long_time_forecast_dynamical successfully
        Tests to make sure we get a zero result and a 100 result
        """
        truth = np.ones((2000,10))
        prediction_100 = np.copy(truth)

        res_100 = long_time_forecast_dynamical(truth, prediction_100, self.histogram_k, self.bins)
        self.assertAlmostEqual(res_100, 100., places=6)

        # TODO: Double check histogram score shouldn't return zero
        prediction_zero = np.zeros((2000,10))
        res_zero = long_time_forecast_dynamical(truth, prediction_zero, self.histogram_k, self.bins)
        self.assertNotEqual(res_zero, 100.)

    def test_compute_psd_success(self):
        """
        Call compute_psd and make sure it runs
        (not doing any value testing here)
        """
        input_arr = np.ones((2000,10))
        compute_psd(input_arr, self.spectral_k, self.modes)

    def test_compute_psd_failure(self):
        """
        Call compute_psd and check that we hit both ValueErrors
        """
        input_arr = np.ones((2000,10))

        with self.assertRaises(ValueError) as context:
            compute_psd(input_arr, 2001, self.modes)
        self.assertIn("exceeds time_steps", str(context.exception))

        with self.assertRaises(ValueError) as context:
            compute_psd(input_arr, self.spectral_k, 11)
        self.assertIn("exceeds spatial_points", str(context.exception))

    def test_compute_log_psd_success(self):
        """
        Call compute_log_psd and make sure it runs
        (not doing any value testing here)
        """
        input_arr = np.ones((2000,10))
        compute_log_psd(input_arr, self.spectral_k, self.modes)

    def test_compute_log_psd_failure(self):
        """
        Call compute_log_psd and check that we hit both ValueErrors
        """
        input_arr = np.ones((2000,10))

        with self.assertRaises(ValueError) as context:
            compute_log_psd(input_arr, 2001, self.modes)
        self.assertIn("exceeds time_steps", str(context.exception))

        with self.assertRaises(ValueError) as context:
            compute_log_psd(input_arr, self.spectral_k, 11)
        self.assertIn("exceeds spatial_points", str(context.exception))

    def test_long_time_forecast_spatio_temporal_success(self):
        """
        Call long_time_forecast_dynamical successfully
        Tests to make sure we get a zero result and a 100 result
        """
        truth = np.ones((2000,10))
        prediction_100 = np.copy(truth)

        res_100 = long_time_forecast_spatio_temporal(truth, prediction_100, self.spectral_k, self.modes)
        self.assertAlmostEqual(res_100, 100., places=6)

        # TODO: Double check histogram score shouldn't return zero
        prediction_zero = np.zeros((2000,10))
        res_zero = long_time_forecast_spatio_temporal(truth, prediction_zero, self.spectral_k, self.modes)
        self.assertAlmostEqual(res_zero, 0.)

    def test_evaluate_custom_success(self):
        """
        Call evaluate_custom successfully
        Tests to make sure we get a zero result and a 100 result
        """
        prediction_100 = self.mock_test_data

        # Short time, long time, and reconstruction from config
        res = evaluate_custom('dataset_0', 1, self.mock_test_data, prediction_100)
        self.assertIn('short_time', res)
        self.assertIn('long_time', res)
        self.assertIn('reconstruction', res)
        self.assertAlmostEqual(res['short_time'], 100., places=6)
        self.assertAlmostEqual(res['long_time'], 100., places=6)
        self.assertAlmostEqual(res['reconstruction'], 100., places=6)

        # Short time, long time, and reconstruction from provided
        res = evaluate_custom('dataset_1', 1, self.mock_test_data, prediction_100, ['short_time', 'long_time', 'reconstruction'])
        self.assertIn('short_time', res)
        self.assertIn('long_time', res)
        self.assertIn('reconstruction', res)
        self.assertAlmostEqual(res['short_time'], 100., places=6)
        self.assertAlmostEqual(res['long_time'], 100., places=6)
        self.assertAlmostEqual(res['reconstruction'], 100., places=6)

    def test_evaluate_success(self):
        """
        Call evaluate successfully
        Tests to make sure we get a zero result and a 100 result
        """
        prediction_100 = self.mock_test_data

        # Short time, long time, and reconstruction from config
        res = evaluate('dataset_0', 1, prediction_100)
        self.assertIn('short_time', res)
        self.assertIn('long_time', res)
        self.assertIn('reconstruction', res)
        self.assertAlmostEqual(res['short_time'], 100., places=6)
        self.assertAlmostEqual(res['long_time'], 100., places=6)
        self.assertAlmostEqual(res['reconstruction'], 100., places=6)

        # Short time, long time, and reconstruction from provided
        res = evaluate('dataset_1', 1, prediction_100, ['short_time', 'long_time', 'reconstruction'])
        self.assertIn('short_time', res)
        self.assertIn('long_time', res)
        self.assertIn('reconstruction', res)
        self.assertAlmostEqual(res['short_time'], 100., places=6)
        self.assertAlmostEqual(res['long_time'], 100., places=6)
        self.assertAlmostEqual(res['reconstruction'], 100., places=6)

    def test_evaluate_fail_pair_id(self):
        """
        Call evaluate unsuccessfully
        Tests to make sure we fail on an invalid pair_id
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(evaluate('dataset_0', 999, None, None))
        self.assertIn(f"Provided pair_id", str(context.exception))
        self.assertIn(f"does not exist in", str(context.exception))

    def test_evaluate_fail_long_time_evaluation_type(self):
        """
        Call evaluate unsuccessfully
        Tests to make sure we fail on an invalid long time evaluation type
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(evaluate('dataset_2', 1, None, None))
        self.assertIn(f"Unknown dataset long time evaluation type", str(context.exception))

    def test_evaluate_fail_unknown_metric(self):
        """
        Call evaluate unsuccessfully
        Tests to make sure we fail on an invalid metric
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(evaluate('dataset_2', 1, None, ["random_metric"]))
        self.assertIn(f"Unknown metric", str(context.exception))

    def test_save_results(self):
        """
        Call save_results successfully
        Tests to make sure we run save_results without fail
        """
        predictions = np.ones((10,10))
        save_results('dataset_1', 'method', 'batch_id', '1', {'test': 1}, predictions, {'result': 1})
