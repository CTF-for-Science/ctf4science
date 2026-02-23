import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from unittest.mock import patch

from ctf4science.visualization_module import Visualization


class TestVisualizationModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Called once before running all unit tests.
        Sets up a temporary top_dir with a minimal dataset and batch layout
        so that visualization utilities can be exercised end-to-end.
        """
        cls.temp_dir = tempfile.TemporaryDirectory(delete=False)
        cls.top_dir = Path(cls.temp_dir.name)

        # Patch top_dir in both visualization and data modules so that
        # _load_test_data and config loading use the temporary structure.
        cls.patcher_vis = patch("ctf4science.visualization_module.top_dir", cls.top_dir)
        cls.patcher_data = patch("ctf4science.data_module.top_dir", cls.top_dir)
        cls.patcher_vis.start()
        cls.patcher_data.start()

        # Minimal dataset configuration for visualization tests
        cls.dataset_name = "dataset_vis"
        cls.pair_id = 1
        cls.time_steps = 50
        cls.spatial_dim = 10

        data_root = cls.top_dir / "data" / cls.dataset_name
        test_dir = data_root / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create simple test data used by _load_test_data (npy-based)
        cls.test_data = np.linspace(0.0, 1.0, cls.time_steps * cls.spatial_dim, dtype=float).reshape(
            cls.time_steps, cls.spatial_dim
        )
        np.save(test_dir / "X1test.npy", cls.test_data)

        dataset_config = {
            "type": "dynamical",
            "evaluation_params": {
                "k_short": 10,
                "k_long": 10,
                "modes": 5,
                "bins": 20,
            },
            "pairs": [
                {"id": cls.pair_id, "test": "X1test.npy", "metrics": ["short_time", "long_time"]},
            ],
            # Visualization types understood by Visualization.generate_all_plots
            "visualizations": ["trajectories", "histograms", "psd", "errors"],
        }

        with open(data_root / f"{cls.dataset_name}.yaml", "w") as f:
            yaml.safe_dump(dataset_config, f)

        # Batch directory with predictions and evaluation_results for plotting
        cls.batch_root = cls.top_dir / "batches" / "batch1"
        cls.pair_dir = cls.batch_root / f"pair{cls.pair_id}"
        cls.pair_dir.mkdir(parents=True, exist_ok=True)

        # predictions.npy: shape must match test data
        predictions = cls.test_data * 0.9
        np.save(cls.pair_dir / "predictions.npy", predictions)

        # evaluation_results.yaml for "errors" plot
        errors = {
            "short_time": [0.1, 0.2, 0.3],
            "long_time": [0.05, 0.15, 0.25],
        }
        with open(cls.pair_dir / "evaluation_results.yaml", "w") as f:
            yaml.safe_dump(errors, f)

        # Instantiate visualizer (uses default visualization config)
        cls.visualizer = Visualization()
        cls.visualizer.config.setdefault("colors", {})
        cls.visualizer.config["colors"].setdefault("errors", ["red", "green", "blue"])

    @classmethod
    def tearDownClass(cls):
        """
        Called once after running all unit tests.
        """
        cls.temp_dir.cleanup()
        cls.patcher_vis.stop()
        cls.patcher_data.stop()

    def test_plot_trajectories_runs(self):
        """
        Ensure plot_trajectories executes without error on simple inputs.
        """
        truth = self.test_data
        predictions = [truth * 0.9]
        fig = self.visualizer.plot_trajectories(truth, predictions, labels=["Model"])
        # One truth line plus one prediction line should be drawn.
        ax0 = fig.axes[0]
        self.assertEqual(len(ax0.lines), 2)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_trajectories_shape_mismatch_raises(self):
        """
        plot_trajectories raises ValueError when prediction shape mismatches truth.
        """
        truth = self.test_data
        bad_prediction = self.test_data[:, :-1]  # wrong spatial dimension
        with self.assertRaises(ValueError):
            self.visualizer.plot_trajectories(truth, [bad_prediction])

    def test_plot_errors_runs(self):
        """
        Ensure plot_errors executes without error on simple inputs.
        """
        errors = {"short_time": [0.1, 0.2, 0.3], "long_time": [0.05, 0.15, 0.25]}
        fig = self.visualizer.plot_errors(errors, labels=["Short Time", "Long Time"])
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_errors_empty_raises(self):
        """
        plot_errors raises ValueError when errors dict is empty.
        """
        with self.assertRaises(ValueError):
            self.visualizer.plot_errors({})

    def test_plot_errors_length_mismatch_raises(self):
        """
        plot_errors raises ValueError when error list lengths differ.
        """
        errors = {"short_time": [0.1, 0.2], "long_time": [0.05, 0.15, 0.25]}
        with self.assertRaises(ValueError):
            self.visualizer.plot_errors(errors)

    def test_plot_errors_label_count_mismatch_raises(self):
        """
        plot_errors raises ValueError when label count does not match errors.
        """
        errors = {"short_time": [0.1, 0.2, 0.3], "long_time": [0.05, 0.15, 0.25]}
        with self.assertRaises(ValueError):
            self.visualizer.plot_errors(errors, labels=["Only one label"])

    def test_plot_histograms_runs(self):
        """
        Ensure plot_histograms executes without error on simple inputs.
        """
        truth = self.test_data
        predictions = [truth * 1.1]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Tight layout not applied.*",
                category=UserWarning,
            )
            fig = self.visualizer.plot_histograms(truth, predictions, modes=10, bins=10, labels=["Model"])
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_histograms_invalid_modes_raises(self):
        """
        plot_histograms raises ValueError when modes exceed time steps.
        """
        truth = self.test_data
        predictions = [truth]
        with self.assertRaises(ValueError):
            self.visualizer.plot_histograms(truth, predictions, modes=self.time_steps + 1, bins=10)

    def test_plot_histograms_shape_mismatch_raises(self):
        """
        plot_histograms raises ValueError when prediction shape mismatches truth.
        """
        truth = self.test_data
        bad_prediction = self.test_data[:, :-1]
        with self.assertRaises(ValueError):
            self.visualizer.plot_histograms(truth, [bad_prediction], modes=10, bins=10)

    def test_plot_psd_runs(self):
        """
        Ensure plot_psd executes without error on simple inputs.
        """
        # Use square-ish array to exercise PSD computation
        truth = np.ones((32, 16))
        predictions = [truth * 0.8]
        fig = self.visualizer.plot_psd(truth, predictions, k=10, modes=8, labels=["Model"])
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_psd_invalid_k_and_modes_and_shape_raises(self):
        """
        plot_psd raises ValueError when k/modes are invalid or shapes mismatch.
        """
        truth = np.ones((10, 5))
        predictions = [truth]

        with self.assertRaises(ValueError):
            self.visualizer.plot_psd(truth, predictions, k=11, modes=3)

        with self.assertRaises(ValueError):
            self.visualizer.plot_psd(truth, predictions, k=5, modes=6)

        bad_prediction = np.ones((10, 4))
        with self.assertRaises(ValueError):
            self.visualizer.plot_psd(truth, [bad_prediction], k=5, modes=3)

    def test_plot_from_batch_runs_for_supported_types(self):
        """
        Ensure plot_from_batch executes for each supported plot_type used in config.
        """
        for plot_type in ["trajectories", "histograms", "psd", "errors"]:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Tight layout not applied.*",
                    category=UserWarning,
                )
                fig = self.visualizer.plot_from_batch(
                    self.dataset_name,
                    self.pair_id,
                    str(self.pair_dir),
                    plot_type=plot_type,
                )
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)

    def test_plot_from_batch_missing_predictions_raises(self):
        """
        plot_from_batch raises FileNotFoundError when predictions.npy is missing.
        """
        missing_dir = self.batch_root / "missing_pair"
        missing_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            self.visualizer.plot_from_batch(self.dataset_name, self.pair_id, str(missing_dir), plot_type="trajectories")

    def test_plot_from_batch_missing_errors_file_raises(self):
        """
        plot_from_batch raises FileNotFoundError when evaluation_results.yaml is missing for errors plot.
        """
        missing_err_dir = self.batch_root / "pair_missing_errors"
        missing_err_dir.mkdir(parents=True, exist_ok=True)
        # predictions.npy required, but omit evaluation_results.yaml
        np.save(missing_err_dir / "predictions.npy", self.test_data)
        with self.assertRaises(FileNotFoundError):
            self.visualizer.plot_from_batch(self.dataset_name, self.pair_id, str(missing_err_dir), plot_type="errors")

    def test_plot_from_batch_2d_comparison_branch_and_validation(self):
        """
        plot_from_batch '2d_comparison' dispatches correctly and enforces 2D requirement.
        """
        # Success case: use existing 2D data
        fig = self.visualizer.plot_from_batch(
            self.dataset_name,
            self.pair_id,
            str(self.pair_dir),
            plot_type="2d_comparison",
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

        # Failure case: predictions with non-2D shape
        bad_dir = self.batch_root / "pair_bad_2d"
        bad_dir.mkdir(parents=True, exist_ok=True)
        bad_predictions = self.test_data.reshape(self.time_steps, self.spatial_dim, 1)
        np.save(bad_dir / "predictions.npy", bad_predictions)
        with self.assertRaises(ValueError):
            self.visualizer.plot_from_batch(self.dataset_name, self.pair_id, str(bad_dir), plot_type="2d_comparison")

        # Unknown plot type should raise ValueError
        with self.assertRaises(ValueError):
            self.visualizer.plot_from_batch(self.dataset_name, self.pair_id, str(self.pair_dir), plot_type="unknown")

    def test_generate_all_plots_runs_and_saves_files(self):
        """
        Ensure generate_all_plots executes without error and creates PNG files.
        """
        # Matplotlib may emit a "Tight layout not applied" UserWarning when the
        # layout is constrained; this is benign for our purposes, so we ignore it.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Tight layout not applied.*",
                category=UserWarning,
            )
            self.visualizer.generate_all_plots(self.dataset_name, str(self.batch_root))

        # Expect one PNG per visualization type in the pair directory
        generated = {p.name for p in self.pair_dir.glob("*.png")}
        for name in ["trajectories_plot.png", "histograms_plot.png", "psd_plot.png", "errors_plot.png"]:
            self.assertIn(name, generated)

    def test_save_figure_results_runs_and_saves_file(self):
        """
        Ensure save_figure_results executes without error and writes an image.
        """
        fig = plt.figure()
        custom_results_dir = self.top_dir / "custom_results"
        plot_type = "trajectories"

        self.visualizer.save_figure_results(
            fig,
            dataset_name=self.dataset_name,
            model_name="modelX",
            batch_name="batchY",
            pair_id=self.pair_id,
            plot_type=plot_type,
            results_dir=custom_results_dir,
        )

        expected_path = custom_results_dir / "visualizations" / f"{plot_type}.png"
        self.assertTrue(expected_path.exists())

    def test_save_figure_results_default_results_dir(self):
        """
        save_figure_results uses the default results directory when results_dir is None.
        """
        fig = plt.figure()
        plot_type = "histograms"
        self.visualizer.save_figure_results(
            fig,
            dataset_name=self.dataset_name,
            model_name="modelY",
            batch_name="batchZ",
            pair_id=self.pair_id,
            plot_type=plot_type,
            results_dir=None,
        )
        expected_path = (
            self.top_dir
            / "results"
            / self.dataset_name
            / "modelY"
            / "batchZ"
            / f"pair{self.pair_id}"
            / "visualizations"
            / f"{plot_type}.png"
        )
        self.assertTrue(expected_path.exists())

    def test_compare_prediction_runs(self):
        """
        Ensure compare_prediction executes without error on simple 2D data.
        """
        truth = np.linspace(0.0, 1.0, 40 * 20, dtype=float).reshape(40, 20)
        predictions = [truth * 0.95]

        fig = self.visualizer.compare_prediction(
            truth,
            predictions,
            cbar_options={"show": True, "orientation": "horizontal", "shrink": 0.7, "ticks": 3, "label": "Value"},
            show_ticks=False,
            show_titles=True,
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_prediction_default_cbar_options(self):
        """
        compare_prediction uses default colorbar options when cbar_options is None.
        """
        truth = np.linspace(0.0, 1.0, 30 * 10, dtype=float).reshape(30, 10)
        predictions = [truth * 1.05]
        fig = self.visualizer.compare_prediction(
            truth, predictions, cbar_options=None, show_ticks=True, show_titles=False
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_prediction_ticks_list(self):
        """
        compare_prediction accepts list/array ticks for the colorbar.
        """
        truth = np.linspace(0.0, 1.0, 30 * 10, dtype=float).reshape(30, 10)
        predictions = [truth * 0.9]
        fig = self.visualizer.compare_prediction(
            truth,
            predictions,
            cbar_options={"show": True, "ticks": [0.0, 0.5, 1.0]},
            show_ticks=True,
            show_titles=False,
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualization_init_with_custom_config_and_missing_file(self):
        """
        Visualization __init__ loads a custom config path and raises on missing file.
        """
        # Valid temporary config file
        tmp_cfg = self.top_dir / "vis_config.yaml"
        cfg_data = {
            "figure_size": [8, 4],
            "colors": {"truth": "blue", "predictions": "viridis", "errors": ["red", "green"]},
            "linestyles": {"truth": "-", "predictions": "--"},
            "images": {"colormap": "hot"},
        }
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg_data, f)

        vis = Visualization(config_path=tmp_cfg)
        self.assertEqual(vis.config["figure_size"], [8, 4])

        # Missing config file should raise FileNotFoundError
        missing_cfg = self.top_dir / "does_not_exist.yaml"
        with self.assertRaises(FileNotFoundError):
            Visualization(config_path=missing_cfg)
