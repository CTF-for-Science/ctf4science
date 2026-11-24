import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from ctf4science.data_module import get_config
from ctf4science.eval_module import SCORE_MIN
from ctf4science.tune_module import ModelTuner, _build_parser


def _make_tuner(dataset_name, pair_ids, mode):
    """Build a ModelTuner without running __init__ (which starts Ray), with the attributes the helpers use."""
    tuner = ModelTuner.__new__(ModelTuner)
    tuner.hp_config = {"dataset": {"name": dataset_name}}
    tuner.pair_ids = pair_ids
    tuner.dataset_config = get_config(dataset_name)
    tuner.tuning_metric = tuner.dataset_config.get("tuning_metric")
    tuner.mode = ModelTuner._resolve_mode(mode, tuner.dataset_config)
    return tuner


class TestTuneModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.top_dir = Path(cls.temp_dir.name)
        cls.patcher = patch("ctf4science.data_module.top_dir", cls.top_dir)
        cls.patcher.start()

        pairs = [
            {"id": 1, "metrics": ["mae", "rmse", "mase"]},
            {"id": 2, "metrics": ["mae", "rmse", "mase"]},
            {"id": 3, "metrics": ["mae", "rmse"]},
        ]
        datasets = {
            # Default dataset: no lower_is_better, no tuning_metric
            "plain": {"pairs": pairs},
            # Lower-is-better dataset that tunes on one metric
            "errors": {"lower_is_better": True, "tuning_metric": "mae", "pairs": pairs},
            "errors_bad_metric": {"lower_is_better": True, "tuning_metric": "mase", "pairs": pairs},
            "explicit_false": {"lower_is_better": False, "pairs": pairs},
        }
        for name, config in datasets.items():
            (cls.top_dir / "data" / name).mkdir(parents=True, exist_ok=True)
            with open(cls.top_dir / "data" / name / f"{name}.yaml", "w") as f:
                yaml.dump(config, f)

        cls.results = {
            "pairs": [
                {"pair_id": 1, "metrics": {"mae": 1.0, "rmse": 2.0, "mase": 3.0}},
                {"pair_id": 2, "metrics": {"mae": 4.0, "rmse": 5.0, "mase": 6.0}},
            ]
        }

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        cls.temp_dir.cleanup()

    # _resolve_mode

    def test_mode_from_lower_is_better(self):
        self.assertEqual(ModelTuner._resolve_mode(None, get_config("errors")), "min")

    def test_mode_default_max(self):
        self.assertEqual(ModelTuner._resolve_mode(None, get_config("plain")), "max")
        self.assertEqual(ModelTuner._resolve_mode(None, get_config("explicit_false")), "max")

    def test_explicit_mode_wins(self):
        self.assertEqual(ModelTuner._resolve_mode("max", get_config("errors")), "max")
        self.assertEqual(ModelTuner._resolve_mode("min", get_config("plain")), "min")

    # CLI and __init__ wiring

    def test_cli_mode_defaults_to_none(self):
        """--mode defaults to None, so ModelTuner reads the mode from the dataset."""
        parser = _build_parser("test")
        self.assertIsNone(parser.parse_args([]).mode)
        self.assertEqual(parser.parse_args(["--mode", "max"]).mode, "max")

    def _init_tuner(self, dataset_name, mode=None):
        """Run the real ModelTuner.__init__ without starting Ray, writing its output directory to the temp dir."""
        config_path = self.top_dir / f"tune_{dataset_name}.yaml"
        hp_config = {
            "dataset": {"name": dataset_name, "pair_id": [1, 2]},
            "model": {"name": "TestModel"},
            "hyperparameters": {"alpha": {"type": "uniform", "lower_bound": 0.0, "upper_bound": 1.0}},
        }
        with open(config_path, "w") as f:
            yaml.dump(hp_config, f)
        with (
            patch("ctf4science.tune_module.ray.is_initialized", return_value=True),
            patch("ctf4science.tune_module.top_dir", self.top_dir),
        ):
            tuner = ModelTuner(config_path=str(config_path), mode=mode, run_opt_main=lambda path: None)
        self.assertTrue(tuner.output_dir.is_relative_to(self.top_dir))
        return tuner

    def test_init_mode_from_dataset(self):
        """The real __init__ resolves the mode and failure score from the dataset YAML."""
        tuner = self._init_tuner("errors")
        self.assertEqual(tuner.mode, "min")
        self.assertEqual(tuner.failure_score, -SCORE_MIN * 2)  # one tuning_metric for each of 2 pairs

        tuner = self._init_tuner("plain")
        self.assertEqual(tuner.mode, "max")
        self.assertEqual(tuner.failure_score, SCORE_MIN * 6)  # 3 metrics for each of 2 pairs

        # An explicit mode wins
        self.assertEqual(self._init_tuner("errors", mode="max").mode, "max")

    # _sum_results

    def test_sum_results_all_metrics(self):
        tuner = _make_tuner("plain", [1, 2], None)
        self.assertEqual(tuner._sum_results(self.results), 21.0)

    def test_sum_results_tuning_metric(self):
        tuner = _make_tuner("errors", [1, 2], None)
        self.assertEqual(tuner._sum_results(self.results), 5.0)

    def test_sum_results_tuning_metric_missing(self):
        tuner = _make_tuner("errors", [1, 2], None)
        results = {"pairs": [{"pair_id": 1, "metrics": {"rmse": 2.0}}, {"pair_id": 2, "metrics": {"mae": 4.0}}]}
        with self.assertRaises(KeyError):
            tuner._sum_results(results)

    def test_sum_results_missing_pair(self):
        """Results that do not cover every tuned pair raise, instead of giving a smaller sum."""
        incomplete = {"pairs": self.results["pairs"][:1]}
        for dataset_name, pair_ids, results in (
            ("errors", [1, 2], incomplete),
            ("errors", [1, 2], {"pairs": []}),
            ("plain", [1, 2], incomplete),
            ("errors", None, self.results),  # all pairs tuned, but pair 3 is missing
        ):
            with (
                self.subTest(dataset=dataset_name, pair_ids=pair_ids, n_results=len(results["pairs"])),
                self.assertRaisesRegex(ValueError, "no entry for tuned pair"),
            ):
                _make_tuner(dataset_name, pair_ids, None)._sum_results(results)

    # _objective

    def _objective_score(self, dataset_name, mode, results):
        """Run `_objective` for one trial whose run_opt writes `results`, without Ray or a model."""
        tuner = _make_tuner(dataset_name, [1, 2], mode)
        tuner.metric = "score"
        tuner.failure_score = tuner._failure_score()
        tuner.blank_config = {"model": {}}
        tuner._generate_config = lambda config, template, name: str(self.top_dir / f"{name}.yaml")

        def run_opt_main(config_path):
            with open(results_path, "w") as f:
                yaml.dump(results, f)

        # _objective reads results_<trial>.yaml next to the file that defines run_opt_main
        run_opt_main.__code__ = run_opt_main.__code__.replace(co_filename=str(self.top_dir / "run_opt.py"))
        results_path = self.top_dir / "results_trial0.yaml"
        tuner._run_opt_main = run_opt_main
        context = MagicMock()
        context.get_trial_id.return_value = "trial0"
        with patch("ctf4science.tune_module.tune.get_context", return_value=context):
            score = tuner._objective({})[tuner.metric]
        self.assertFalse(results_path.exists())
        return score, tuner.failure_score

    def test_objective_missing_pair_scores_failure(self):
        """A trial whose results miss a tuned pair gets the failure score, in both modes."""
        incomplete = {"pairs": self.results["pairs"][:1]}
        for mode in ("min", "max"):
            for results in (incomplete, {"pairs": []}):
                with self.subTest(mode=mode, n_results=len(results["pairs"])):
                    score, failure_score = self._objective_score("errors", mode, results)
                    self.assertEqual(score, failure_score)
                    self.assertEqual(failure_score, SCORE_MIN * 2 if mode == "max" else -SCORE_MIN * 2)

    def test_objective_complete_results_unchanged(self):
        """Results covering every tuned pair are summed as before."""
        self.assertEqual(self._objective_score("errors", None, self.results)[0], 5.0)
        self.assertEqual(self._objective_score("plain", None, self.results)[0], 21.0)

    # _failure_score

    def test_failure_score_all_metrics(self):
        # 3 + 3 + 2 metrics over all pairs
        self.assertEqual(_make_tuner("plain", None, None)._failure_score(), SCORE_MIN * 8)
        self.assertEqual(_make_tuner("plain", [1, 3], "min")._failure_score(), -SCORE_MIN * 5)

    def test_failure_score_tuning_metric(self):
        # One metric per tuned pair; mode resolved to min from lower_is_better
        self.assertEqual(_make_tuner("errors", [1, 2], None)._failure_score(), -SCORE_MIN * 2)
        self.assertEqual(_make_tuner("errors", None, None)._failure_score(), -SCORE_MIN * 3)
        self.assertEqual(_make_tuner("errors", [1, 2], "max")._failure_score(), SCORE_MIN * 2)

    def test_failure_score_tuning_metric_not_in_pair(self):
        # Pair 3 does not list mase
        self.assertEqual(_make_tuner("errors_bad_metric", [1, 2], None)._failure_score(), -SCORE_MIN * 2)
        with self.assertRaises(ValueError):
            _make_tuner("errors_bad_metric", [1, 3], None)._failure_score()

    def test_failure_score_unknown_pairs(self):
        with self.assertRaises(ValueError):
            _make_tuner("plain", [99], None)._failure_score()

    # _parse_pair_ids

    def test_parse_pair_ids(self):
        self.assertIsNone(ModelTuner._parse_pair_ids({"pair_id": "all"}))
        self.assertIsNone(ModelTuner._parse_pair_ids({}))
        self.assertEqual(ModelTuner._parse_pair_ids({"pair_id": 4}), [4])
        self.assertEqual(ModelTuner._parse_pair_ids({"pair_id": [1, 3]}), [1, 3])

    def test_parse_pair_ids_range(self):
        self.assertEqual(ModelTuner._parse_pair_ids({"pair_id": "1-6"}), [1, 2, 3, 4, 5, 6])
        self.assertEqual(ModelTuner._parse_pair_ids({"pair_id": "7-7"}), [7])

    def test_parse_pair_ids_invalid(self):
        for bad in ["1-a", "1-2-3", "x", 1.5, [1, "2"]]:
            with self.subTest(pair_id=bad), self.assertRaises(ValueError):
                ModelTuner._parse_pair_ids({"pair_id": bad})


if __name__ == "__main__":
    unittest.main()
