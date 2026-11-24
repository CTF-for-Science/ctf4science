import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from scipy.io import savemat

from ctf4science.data_module import _load_test_data
from ctf4science.eval_module import (
    compute_log_psd,
    compute_psd,
    evaluate,
    evaluate_custom,
    evaluate_kaggle_csv,
    extract_metrics_in_order,
    long_time_forecast_dynamical,
    long_time_forecast_spatio_temporal,
    mae,
    mase,
    mase_scale,
    reconstruction,
    rmse,
    save_results,
    short_time_forecast,
)


def _savemat_no_meta(path, mdict):
    """Save .mat dict without __header__/__version__/__globals__ to avoid MatWriteWarning."""
    data_only = {k: v for k, v in mdict.items() if not (k.startswith("__") and k.endswith("__"))}
    savemat(path, data_only)


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
        cls.patcher1 = patch("ctf4science.eval_module.top_dir", cls.top_dir)
        cls.patcher2 = patch("ctf4science.data_module.top_dir", cls.top_dir)
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
            "type": "spatio-temporal",
            "evaluation_params": {"k_short": 20, "k_long": 20, "modes": 5},
            "pairs": [
                {"test": "X1test.mat", "id": 1, "metrics": ["short_time", "long_time", "reconstruction"]},
            ],
            "evaluations": {"long_time": "spectral_L2_error"},
        }

        cls.dataset_config_1 = {
            "type": "dynamical",
            "evaluation_params": {"k_short": 20, "k_long": 20, "modes": 1000, "bins": 50},
            "pairs": [
                {"test": "X1test.mat", "id": 1, "metrics": []},
            ],
            "evaluations": {"long_time": "histogram_L2_error"},
        }

        cls.dataset_config_2 = {
            "type": "dynamical",
            "evaluation_params": {"k_short": 1000, "k_long": 1000, "modes": 1000, "bins": 50},
            "pairs": [
                {"test": "X1test.mat", "id": 1, "metrics": ["long_time"]},
            ],
            "evaluations": {"long_time": "not_implemented_error"},
        }

        cls.dataset_config_3 = {
            "type": "dynamical",
            "evaluation_params": {"k_short": 1000, "k_long": 1000, "modes": 1000, "bins": 50},
            "pairs": [
                {"id": 1, "metrics": ["long_time", "short_time"]},
                {"id": 2, "metrics": ["long_time", "short_time"]},
            ],
            "evaluations": {"long_time": "not_implemented_error"},
        }

        cls.batch_results_1 = {
            "batch_id": "batch_example",
            "dataset": "dataset_example",
            "model": "model_example",
            "pairs": [
                {"metrics": {"long_time": 50.0, "short_time": 10.0}, "pair_id": 1},
                {
                    "metrics": {
                        "long_time": 50.0,
                    },
                    "pair_id": 2,
                },
            ],
        }

        cls.batch_results_2 = {
            "batch_id": "batch_example",
            "dataset": "dataset_example",
            "model": "model_example",
            "pairs": [
                {"metrics": {"long_time": 50.0, "short_time": 10.0}, "pair_id": 1},
            ],
        }

        # Create train/test matrices
        cls.mock_test_data = np.tile(np.arange(100, 200), (20, 1)).T
        cls.mock_test_mat_1 = {
            "__header__": b"header",
            "__version__": "1.0",
            "__globals__": None,
            "ytest": cls.mock_test_data,
        }  # X1test.mat

        # Write to files in expected directory structure
        for i, yaml_content in enumerate(
            [cls.dataset_config_0, cls.dataset_config_1, cls.dataset_config_2, cls.dataset_config_3]
        ):
            # Write YAML dataset config files
            (cls.top_dir / "data" / f"dataset_{i}" / "test").mkdir(parents=True, exist_ok=True)
            with open(cls.top_dir / "data" / f"dataset_{i}" / f"dataset_{i}.yaml", "w") as f:
                yaml.dump(yaml_content, f)
            # Write .mat files
            _savemat_no_meta(cls.top_dir / "data" / f"dataset_{i}" / "test" / "X1test.mat", cls.mock_test_mat_1)

        # Lorenz_Official-like dataset for evaluate_kaggle_csv tests
        lorenz_dir = cls.top_dir / "data" / "Lorenz_Official"
        lorenz_test_dir = lorenz_dir / "test"
        lorenz_test_dir.mkdir(parents=True, exist_ok=True)
        lorenz_config = {
            "type": "dynamical",
            "evaluation_params": {"k_short": 20, "k_long": 20, "modes": 500, "bins": 41},
            "pairs": [
                {"id": 1, "train": ["X1train.mat"], "test": "X1test.mat", "metrics": ["short_time", "long_time"]},
                {"id": 2, "train": ["X2train.mat"], "test": "X2test.mat", "metrics": ["reconstruction"]},
                {"id": 3, "train": ["X2train.mat"], "test": "X3test.mat", "metrics": ["long_time"]},
                {"id": 4, "train": ["X3train.mat"], "test": "X4test.mat", "metrics": ["reconstruction"]},
                {"id": 5, "train": ["X3train.mat"], "test": "X5test.mat", "metrics": ["long_time"]},
                {"id": 6, "train": ["X4train.mat"], "test": "X6test.mat", "metrics": ["short_time", "long_time"]},
                {"id": 7, "train": ["X5train.mat"], "test": "X7test.mat", "metrics": ["short_time", "long_time"]},
                {
                    "id": 8,
                    "train": ["X6train.mat", "X7train.mat", "X8train.mat"],
                    "test": "X8test.mat",
                    "initialization": "X9train.mat",
                    "metrics": ["short_time"],
                },
                {
                    "id": 9,
                    "train": ["X6train.mat", "X7train.mat", "X8train.mat"],
                    "test": "X9test.mat",
                    "initialization": "X10train.mat",
                    "metrics": ["short_time"],
                },
            ],
            "metadata": {
                "delta_t": 0.05,
                "spatial_dimension": 3,
                "matrix_shapes": {
                    "X1test.mat": [1000, 3],
                    "X2test.mat": [10000, 3],
                    "X3test.mat": [1000, 3],
                    "X4test.mat": [10000, 3],
                    "X5test.mat": [1000, 3],
                    "X6test.mat": [1000, 3],
                    "X7test.mat": [1000, 3],
                    "X8test.mat": [1000, 3],
                    "X9test.mat": [1000, 3],
                },
            },
            "evaluations": {"long_time": "histogram_L2_error"},
        }
        with open(lorenz_dir / "Lorenz_Official.yaml", "w") as f:
            yaml.dump(lorenz_config, f)
        for pair_id, shape in lorenz_config["metadata"]["matrix_shapes"].items():
            # Write .mat files for test data
            test_t, test_f = shape
            data = np.arange(test_t * test_f, dtype=np.float64).reshape(test_t, test_f)  # dummy test data
            _savemat_no_meta(lorenz_test_dir / pair_id, {"data": data})

    @classmethod
    def tearDownClass(cls):
        """
        Called once after running all unit tests
        """
        cls.temp_dir.cleanup()
        cls.patcher1.stop()
        cls.patcher2.stop()

    def setUp(self):
        """
        Called once before each unit test
        """

    def tearDown(self):
        """
        Called once after each unit test
        """

    def test_extract_metrics_in_order_success(self):
        """
        Call extract_metrics_in_order successfully
        """
        self.assertListEqual(extract_metrics_in_order("dataset_2", self.batch_results_1), [50.0])

    def test_extract_metrics_in_order_pair_id_not_found(self):
        """
        Call extract_metrics_in_order unsuccessfully
        Tests to make sure we throw a ValueError when a pair_id is not found
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(extract_metrics_in_order("dataset_3", self.batch_results_2))
        self.assertIn("not found in batch_results", str(context.exception))

    def test_extract_metrics_in_order_metric_not_found(self):
        """
        Call extract_metrics_in_order unsuccessfully
        Tests to make sure we throw a ValueError when a metric for a pair_id is not found
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(extract_metrics_in_order("dataset_3", self.batch_results_1))
        self.assertIn("not found for pair ID", str(context.exception))

    def test_short_time_forecast_success(self):
        """
        Call short_time_forecast successfully
        Tests to make sure we get a zero result and a 100 result
        """
        truth = np.ones((100, 10))
        prediction_zero = np.zeros((100, 10))
        prediction_100 = np.copy(truth)

        res_zero = short_time_forecast(truth, prediction_zero, self.short_time_k)
        res_100 = short_time_forecast(truth, prediction_100, self.short_time_k)

        self.assertAlmostEqual(res_zero, 0.0, places=6)
        self.assertAlmostEqual(res_100, 100.0, places=6)

    def test_reconstruction_success(self):
        """
        Call reconstruction successfully
        Tests to make sure we get a zero result and a 100 result
        """
        truth = np.ones((100, 10))
        prediction_zero = np.zeros((100, 10))
        prediction_100 = np.copy(truth)

        res_zero = reconstruction(truth, prediction_zero)
        res_100 = reconstruction(truth, prediction_100)

        self.assertAlmostEqual(res_zero, 0.0, places=6)
        self.assertAlmostEqual(res_100, 100.0, places=6)

    def test_long_time_forecast_dynamical_success(self):
        """
        Call long_time_forecast_dynamical successfully
        Tests to make sure we get a zero result and a 100 result
        """
        truth = np.ones((2000, 10))
        prediction_100 = np.copy(truth)

        res_100 = long_time_forecast_dynamical(truth, prediction_100, self.histogram_k, self.bins)
        self.assertAlmostEqual(res_100, 100.0, places=6)

        # TODO: Double check histogram score shouldn't return zero
        prediction_zero = np.zeros((2000, 10))
        res_zero = long_time_forecast_dynamical(truth, prediction_zero, self.histogram_k, self.bins)
        self.assertNotEqual(res_zero, 100.0)

    def test_compute_psd_success(self):
        """
        Call compute_psd and make sure it runs
        (not doing any value testing here)
        """
        input_arr = np.ones((2000, 10))
        compute_psd(input_arr, self.spectral_k, self.modes)

    def test_compute_psd_failure(self):
        """
        Call compute_psd and check that we hit both ValueErrors
        """
        input_arr = np.ones((2000, 10))

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
        input_arr = np.ones((2000, 10))
        compute_log_psd(input_arr, self.spectral_k, self.modes)

    def test_compute_log_psd_failure(self):
        """
        Call compute_log_psd and check that we hit both ValueErrors
        """
        input_arr = np.ones((2000, 10))

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
        truth = np.ones((2000, 10))
        prediction_100 = np.copy(truth)

        res_100 = long_time_forecast_spatio_temporal(truth, prediction_100, self.spectral_k, self.modes)
        self.assertAlmostEqual(res_100, 100.0, places=6)

        # TODO: Double check histogram score shouldn't return zero
        prediction_zero = np.zeros((2000, 10))
        res_zero = long_time_forecast_spatio_temporal(truth, prediction_zero, self.spectral_k, self.modes)
        self.assertAlmostEqual(res_zero, 0.0)

    def test_evaluate_custom_success(self):
        """
        Call evaluate_custom successfully
        Tests to make sure we get a zero result and a 100 result
        """
        prediction_100 = self.mock_test_data

        # Short time, long time, and reconstruction from config
        res = evaluate_custom("dataset_0", 1, self.mock_test_data, prediction_100)
        self.assertIn("short_time", res)
        self.assertIn("long_time", res)
        self.assertIn("reconstruction", res)
        self.assertAlmostEqual(res["short_time"], 100.0, places=6)
        self.assertAlmostEqual(res["long_time"], 100.0, places=6)
        self.assertAlmostEqual(res["reconstruction"], 100.0, places=6)

        # Short time, long time, and reconstruction from provided
        res = evaluate_custom(
            "dataset_1", 1, self.mock_test_data, prediction_100, ["short_time", "long_time", "reconstruction"]
        )
        self.assertIn("short_time", res)
        self.assertIn("long_time", res)
        self.assertIn("reconstruction", res)
        self.assertAlmostEqual(res["short_time"], 100.0, places=6)
        self.assertAlmostEqual(res["long_time"], 100.0, places=6)
        self.assertAlmostEqual(res["reconstruction"], 100.0, places=6)

    def test_evaluate_custom_nonfinite_prediction(self):
        """
        A prediction containing NaN or inf scores -100 on every metric
        """
        for bad in (np.nan, np.inf):
            prediction = self.mock_test_data.astype(float).copy()
            prediction[-1, 0] = bad
            res = evaluate_custom("dataset_0", 1, self.mock_test_data, prediction)
            self.assertEqual(res, {"short_time": -100.0, "long_time": -100.0, "reconstruction": -100.0})

            # Unknown metrics still raise
            with self.assertRaises(ValueError) as context:
                evaluate_custom("dataset_0", 1, self.mock_test_data, prediction, ["random_metric"])
            self.assertIn("Unknown metric", str(context.exception))

    def test_evaluate_custom_clips_scores(self):
        """
        A very poor finite prediction is clipped to -100 instead of scoring far below it
        """
        prediction = self.mock_test_data.astype(float) * 1000
        # The raw scores are far below -100, so clipping is what brings them to -100
        self.assertLess(reconstruction(self.mock_test_data, prediction), -100.0)
        res = evaluate_custom("dataset_0", 1, self.mock_test_data, prediction, ["short_time", "reconstruction"])
        self.assertEqual(res, {"short_time": -100.0, "reconstruction": -100.0})
        for score in res.values():
            self.assertIsInstance(score, float)

    def test_evaluate_custom_flexible_k(self):
        """
        Call evaluate_custom with flexible_k=True when trajectory has fewer timesteps than config k.
        Ensures k is capped to min(timesteps, k) so evaluation succeeds and scores are correct.
        """
        # Fewer timesteps (10) than dataset_0's k_short/k_long (20)
        short_truth = np.ones((10, 20))
        short_prediction = np.copy(short_truth)

        with patch("builtins.print"):
            # flexible_k=True: should use k=10 instead of 20, no index error
            res = evaluate_custom(
                "dataset_0",
                1,
                short_truth,
                short_prediction,
                metrics=["short_time", "long_time", "reconstruction"],
                flexible_k=True,
            )
        self.assertIn("short_time", res)
        self.assertIn("long_time", res)
        self.assertIn("reconstruction", res)
        self.assertAlmostEqual(res["short_time"], 100.0, places=6)
        self.assertAlmostEqual(res["long_time"], 100.0, places=6)
        self.assertAlmostEqual(res["reconstruction"], 100.0, places=6)

    def test_evaluate_success(self):
        """
        Call evaluate successfully
        Tests to make sure we get a zero result and a 100 result
        """
        prediction_100 = self.mock_test_data

        # Short time, long time, and reconstruction from config
        res = evaluate("dataset_0", 1, prediction_100)
        self.assertIn("short_time", res)
        self.assertIn("long_time", res)
        self.assertIn("reconstruction", res)
        self.assertAlmostEqual(res["short_time"], 100.0, places=6)
        self.assertAlmostEqual(res["long_time"], 100.0, places=6)
        self.assertAlmostEqual(res["reconstruction"], 100.0, places=6)

        # Short time, long time, and reconstruction from provided
        res = evaluate("dataset_1", 1, prediction_100, ["short_time", "long_time", "reconstruction"])
        self.assertIn("short_time", res)
        self.assertIn("long_time", res)
        self.assertIn("reconstruction", res)
        self.assertAlmostEqual(res["short_time"], 100.0, places=6)
        self.assertAlmostEqual(res["long_time"], 100.0, places=6)
        self.assertAlmostEqual(res["reconstruction"], 100.0, places=6)

    def test_evaluate_fail_pair_id(self):
        """
        Call evaluate unsuccessfully
        Tests to make sure we fail on an invalid pair_id
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(evaluate("dataset_0", 999, None, None))  # pyrefly: ignore
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not exist in", str(context.exception))

    def test_evaluate_fail_long_time_evaluation_type(self):
        """
        Call evaluate unsuccessfully
        Tests to make sure we fail on an invalid long time evaluation type
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(evaluate("dataset_2", 1, None, None))  # pyrefly: ignore
        self.assertIn("Unknown dataset long time evaluation type", str(context.exception))

    def test_evaluate_fail_unknown_metric(self):
        """
        Call evaluate unsuccessfully
        Tests to make sure we fail on an invalid metric
        """
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(evaluate("dataset_2", 1, None, ["random_metric"]))  # pyrefly: ignore
        self.assertIn("Unknown metric", str(context.exception))

    def test_save_results(self):
        """
        Call save_results successfully
        Tests to make sure we run save_results without fail
        """
        predictions = np.ones((10, 10))
        save_results("dataset_1", "method", "batch_id", "1", {"test": 1}, predictions, {"result": 1})

    def test_evaluate_custom_fail(self):
        """
        Call evaluate_custom and fails
        Tests to make sure we get a zero result and a 100 result
        """
        # Short time, long time, and reconstruction from config
        with self.assertRaises(ValueError) as context:
            self.assertIsNone(evaluate_custom("dataset_0", 999, None, None))  # pyrefly: ignore
        self.assertIn("Provided pair_id", str(context.exception))
        self.assertIn("does not exist in", str(context.exception))

    def _lorenz_csv_path(self, rows_by_pair):
        """Build a temporary Kaggle-style submission CSV from per-pair prediction arrays.

        Each key in rows_by_pair is a pair_id (1-9); each value is an array of shape
        (T, 3) with x, y, z coordinates. Rows are written in ascending pair_id and
        timestep order. The file has columns: pair_id, timestep, x, y, z.

        Returns
        -------
        str
            Path to the temporary CSV file (caller should unlink when done).
        """
        records = []
        for pair_id in sorted(rows_by_pair.keys()):
            arr = rows_by_pair[pair_id]
            records.extend(
                [
                    {
                        "id": f"{pair_id}_{t}",
                        "pair_id": pair_id,
                        "timestep": t,
                        "x": arr[t, 0],
                        "y": arr[t, 1],
                        "z": arr[t, 2],
                    }
                    for t in range(arr.shape[0])
                ]
            )
        df = pd.DataFrame(records)
        fd = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)  # noqa: SIM115
        fd.close()
        df.to_csv(fd.name, index=False)
        return fd.name

    def test_evaluate_kaggle_csv_missing_column(self):
        """evaluate_kaggle_csv raises ValueError when a required column is missing."""
        df = pd.DataFrame({"pair_id": [1], "timestep": [0], "x": [0.0], "y": [0.0]})  # missing z
        fd = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)  # noqa: SIM115
        fd.close()
        df.to_csv(fd.name, index=False)
        try:
            with self.assertRaises(ValueError) as ctx:
                evaluate_kaggle_csv(fd.name, dataset_name="Lorenz_Official")
            self.assertIn("CSV must contain columns", str(ctx.exception))
            self.assertIn("missing", str(ctx.exception))
        finally:
            Path(fd.name).unlink(missing_ok=True)

    def test_evaluate_kaggle_csv_no_rows_for_pair(self):
        """evaluate_kaggle_csv raises ValueError when CSV has no rows for a pair_id. Fails because the only pair_id with rows is 1, but there are other pairs in the config."""
        rows = {}
        rows[1] = _load_test_data("Lorenz_Official", 1)
        csv_path = self._lorenz_csv_path(rows)
        try:
            with self.assertRaises(ValueError) as ctx:
                evaluate_kaggle_csv(csv_path, dataset_name="Lorenz_Official")
            self.assertIn("No rows found for pair_id=", str(ctx.exception))
        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_evaluate_kaggle_csv_shape_mismatch(self):
        """evaluate_kaggle_csv raises ValueError when prediction shape does not match expected."""
        rows = {}
        for pid in range(1, 10):
            arr = _load_test_data("Lorenz_Official", pid)
            if pid == 1:
                arr = arr[:999]  # wrong length for pair 1 (expect 1000)
            rows[pid] = arr
        csv_path = self._lorenz_csv_path(rows)
        try:
            with self.assertRaises(ValueError) as ctx:
                evaluate_kaggle_csv(csv_path, dataset_name="Lorenz_Official")
            self.assertIn("prediction shape", str(ctx.exception))
            self.assertIn("does not match", str(ctx.exception))
        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_evaluate_kaggle_csv_expected_shape_not_found(self):
        """evaluate_kaggle_csv raises ValueError when config has no matrix_shapes for a pair."""
        df = pd.DataFrame(
            {
                "id": list(range(20)),
                "pair_id": [1] * 20,
                "timestep": list(range(20)),
                "x": [0.0] * 20,
                "y": [0.0] * 20,
                "z": [0.0] * 20,
            }
        )
        fd = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)  # noqa: SIM115
        fd.close()
        df.to_csv(fd.name, index=False)
        try:
            with self.assertRaises(ValueError) as ctx:
                evaluate_kaggle_csv(fd.name, dataset_name="dataset_0")
            self.assertIn("Expected shape not found", str(ctx.exception))
        finally:
            Path(fd.name).unlink(missing_ok=True)

    def test_evaluate_kaggle_csv_pair_id_to_e_not_found(self):
        """evaluate_kaggle_csv raises ValueError when a (pair_id, metric) is not in pair_id_to_e."""
        from ctf4science.data_module import _load_test_data

        rows = {}
        for pid in range(1, 10):
            rows[pid] = _load_test_data("Lorenz_Official", pid)
        csv_path = self._lorenz_csv_path(rows)
        with patch("ctf4science.eval_module.evaluate") as mock_evaluate:
            mock_evaluate.return_value = {"unknown_metric": 1.0}
            try:
                with self.assertRaises(ValueError) as ctx:
                    evaluate_kaggle_csv(csv_path, dataset_name="Lorenz_Official")
                self.assertIn("not found in pair_id_to_e", str(ctx.exception))
            finally:
                Path(csv_path).unlink(missing_ok=True)

    def test_evaluate_kaggle_csv_valid_success_100(self):
        """evaluate_kaggle_csv runs successfully on a valid CSV and returns E1–E12 and average. Scores 100."""
        from ctf4science.data_module import _load_test_data

        rows = {}
        for pid in range(1, 10):
            rows[pid] = _load_test_data("Lorenz_Official", pid)
        csv_path = self._lorenz_csv_path(rows)
        try:
            results = evaluate_kaggle_csv(csv_path, dataset_name="Lorenz_Official")
            self.assertIsInstance(results, dict)
            for i in range(1, 13):
                self.assertIn(f"E{i}", results)
                self.assertIsInstance(results[f"E{i}"], (int, float))
            self.assertIn("average", results)
            self.assertIsInstance(results["average"], (int, float))
            self.assertAlmostEqual(results["average"], sum(results[f"E{i}"] for i in range(1, 13)) / 12)
            self.assertAlmostEqual(results["average"], 100.0)
        finally:
            Path(csv_path).unlink(missing_ok=True)

    def test_evaluate_kaggle_csv_valid_success_0(self):
        """evaluate_kaggle_csv runs successfully on a valid CSV and returns E1–E12 and average. Doesn't score 100."""
        from ctf4science.data_module import _load_test_data

        rows = {}
        for pid in range(1, 10):
            rows[pid] = _load_test_data("Lorenz_Official", pid) * 0.0
        csv_path = self._lorenz_csv_path(rows)
        try:
            results = evaluate_kaggle_csv(csv_path, dataset_name="Lorenz_Official")
            self.assertIsInstance(results, dict)
            for i in range(1, 13):
                self.assertIn(f"E{i}", results)
                self.assertIsInstance(results[f"E{i}"], (int, float))
            self.assertIn("average", results)
            self.assertIsInstance(results["average"], (int, float))
            self.assertAlmostEqual(results["average"], sum(results[f"E{i}"] for i in range(1, 13)) / 12)
            self.assertLess(results["average"], 100.0)
        finally:
            Path(csv_path).unlink(missing_ok=True)


class TestWindowedMetrics(unittest.TestCase):
    """MAE, RMSE and MASE on windowed forecasting pairs (buildings), with lower_is_better clipping."""

    lookback = 2
    # Errors added to each building's truth windows; building 2 is constant, so its MASE scale is 0
    offsets = np.array([1.0, -3.0, 5.0])

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.top_dir = Path(cls.temp_dir.name)
        cls.patcher1 = patch("ctf4science.eval_module.top_dir", cls.top_dir)
        cls.patcher2 = patch("ctf4science.data_module.top_dir", cls.top_dir)
        cls.patcher1.start()
        cls.patcher2.start()

        # Continuous truth (time, building, channel): indoor temperature (channel 4) is t, 2t and 7
        n_rows = cls.lookback + 3 * 96
        t = np.arange(n_rows, dtype=np.float32)
        rng = np.random.default_rng(0)
        cls.truth = rng.normal(size=(n_rows, 3, 6)).astype(np.float32)
        cls.truth[:, 0, 4] = t
        cls.truth[:, 1, 4] = 2 * t
        cls.truth[:, 2, 4] = 7.0

        cls.config = {
            "lower_is_better": True,
            "evaluation_params": {
                "lookback": cls.lookback,
                "stride": 96,
                "target_channel": 4,
                "covariate_channels": [0, 1, 2, 5],
            },
            "pairs": [
                # G1-like: one-step forecast from every step (stride override)
                {
                    "id": 1,
                    "train": ["x.npz"],
                    "test": "truth.npz",
                    "horizon": 1,
                    "stride": 1,
                    "metrics": ["mae", "rmse", "mase"],
                },
                # G2-like: 96-step forecast at the default stride
                {"id": 2, "train": ["x.npz"], "test": "truth.npz", "horizon": 96, "metrics": ["mae", "rmse", "mase"]},
                # Not windowed (no horizon)
                {"id": 3, "train": ["x.npz"], "test": "truth.npz", "metrics": ["reconstruction"]},
            ],
        }
        root = cls.top_dir / "data" / "bld_eval"
        (root / "test").mkdir(parents=True)
        with open(root / "bld_eval.yaml", "w") as f:
            yaml.dump(cls.config, f)
        np.savez(root / "test" / "truth.npz", data=cls.truth)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        cls.patcher1.stop()
        cls.patcher2.stop()

    def _truth_windows(self, horizon: int, stride: int) -> np.ndarray:
        """Truth windows (W, horizon, B) of the target channel, cut by hand."""
        target = self.truth[:, :, 4]
        starts = range(0, target.shape[0] - self.lookback - horizon + 1, stride)
        return np.stack([target[s + self.lookback : s + self.lookback + horizon] for s in starts])

    def _offset_prediction(self, horizon: int, stride: int) -> np.ndarray:
        return self._truth_windows(horizon, stride) + self.offsets

    def test_metric_functions(self):
        """mae, rmse, mase_scale and mase against hand-computed values."""
        truth = np.zeros((2, 3, 3))
        prediction = truth + self.offsets
        self.assertAlmostEqual(mae(truth, prediction), 3.0)
        self.assertAlmostEqual(rmse(truth, prediction), np.sqrt(35 / 3))
        np.testing.assert_allclose(mase_scale(self.truth[:, :, 4], 1), [1.0, 2.0, 0.0])
        np.testing.assert_allclose(mase_scale(self.truth[:, :, 4], 96), [96.0, 192.0, 0.0])
        # Building 2 has scale 0 and is left out of the mean
        self.assertAlmostEqual(mase(truth, prediction, np.array([1.0, 2.0, 0.0])), (1 / 1 + 3 / 2) / 2)
        with self.assertRaises(ValueError):
            mase(truth, prediction, np.zeros(3))
        with self.assertRaises(ValueError):
            mase_scale(np.zeros((5, 3)), 5)

    def test_evaluate_custom_lag_1(self):
        """Horizon-1 pair: MASE uses the lag-1 naive scale from the continuous truth."""
        prediction = self._offset_prediction(1, 1)
        self.assertEqual(prediction.shape, (self.truth.shape[0] - self.lookback, 1, 3))
        res = evaluate_custom("bld_eval", 1, self.truth, prediction)
        self.assertAlmostEqual(res["mae"], 3.0, places=5)
        self.assertAlmostEqual(res["rmse"], np.sqrt(35 / 3), places=5)
        self.assertAlmostEqual(res["mase"], (1 / 1 + 3 / 2) / 2, places=5)

    def test_evaluate_custom_lag_96(self):
        """Horizon-96 pair: MASE uses the lag-96 naive scale; windows use the default stride."""
        prediction = self._offset_prediction(96, 96)
        self.assertEqual(prediction.shape, (3, 96, 3))
        res = evaluate_custom("bld_eval", 2, self.truth, prediction)
        self.assertAlmostEqual(res["mae"], 3.0, places=5)
        self.assertAlmostEqual(res["rmse"], np.sqrt(35 / 3), places=5)
        self.assertAlmostEqual(res["mase"], (1 / 96 + 3 / 192) / 2, places=7)

    def test_evaluate_matches_evaluate_custom(self):
        """evaluate (truth loaded from test/) and evaluate_custom score the same data identically."""
        for pair_id, horizon, stride in ((1, 1, 1), (2, 96, 96)):
            prediction = self._offset_prediction(horizon, stride)
            self.assertEqual(
                evaluate("bld_eval", pair_id, prediction), evaluate_custom("bld_eval", pair_id, self.truth, prediction)
            )

    def test_perfect_prediction_scores_zero(self):
        """A perfect prediction scores 0 on every error metric."""
        res = evaluate_custom("bld_eval", 2, self.truth, self._truth_windows(96, 96))
        self.assertEqual(res, {"mae": 0.0, "rmse": 0.0, "mase": 0.0})

    def test_lower_is_better_clip(self):
        """With lower_is_better, very poor scores are clipped to 100."""
        prediction = self._truth_windows(96, 96) + 1.0e6
        self.assertGreater(mae(self._truth_windows(96, 96), prediction), 100.0)
        res = evaluate_custom("bld_eval", 2, self.truth, prediction)
        self.assertEqual(res, {"mae": 100.0, "rmse": 100.0, "mase": 100.0})
        for score in res.values():
            self.assertIsInstance(score, float)

    def test_lower_is_better_nonfinite(self):
        """With lower_is_better, a NaN or inf prediction scores 100 on every metric."""
        for bad in (np.nan, np.inf, -np.inf):
            prediction = self._truth_windows(96, 96).copy()
            prediction[0, 0, 0] = bad
            with patch("builtins.print"):
                res = evaluate_custom("bld_eval", 2, self.truth, prediction)
            self.assertEqual(res, {"mae": 100.0, "rmse": 100.0, "mase": 100.0})

    def test_windowed_failures(self):
        """Wrong prediction shapes, windowed truth and metrics of the wrong kind raise."""
        prediction = self._truth_windows(96, 96)
        with self.assertRaises(ValueError) as context:
            evaluate_custom("bld_eval", 2, self.truth, prediction[:-1])
        self.assertIn("does not match the truth windows", str(context.exception))
        with self.assertRaises(ValueError) as context:
            evaluate_custom("bld_eval", 2, self.truth[:, :, 4], prediction)
        self.assertIn("continuous (time, building, channel) series", str(context.exception))
        with self.assertRaises(ValueError) as context:
            evaluate_custom("bld_eval", 2, self.truth, prediction, ["short_time"])
        self.assertIn("only supported for non-windowed pairs", str(context.exception))
        with self.assertRaises(ValueError) as context:
            evaluate_custom("bld_eval", 3, np.ones((10, 3)), np.ones((10, 3)), ["mae"])
        self.assertIn("only supported for windowed pairs", str(context.exception))
