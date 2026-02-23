r"""Tune module for CTF models, orchestrates hyperparameter tuning using Ray Tune.

Supports a single config
file or automatic discovery of configs in a model's
``tuning_config/config_*.yaml``.
"""

import argparse
import datetime
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ray
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from ctf4science.performance_module import PerformanceMonitor

top_dir = Path(__file__).parent.parent


class ModelTuner:
    r"""Orchestrates hyperparameter tuning for CTF models using Ray Tune.

    Supports tuning with a specific config file or automatically detecting
    all config files in a model's ``tuning_config/config_*.yaml``.

    Parameters
    ----------
    config_path : str
        Path to the configuration file (dataset, model, hyperparameters).
    model_name : str, optional
        Model name; if not provided, inferred from config or directory.
    save_final_config : bool, optional
        Whether to save the final optimal config. Default is ``True``.
    metric : str, optional
        Metric to optimize. Default is ``"score"``.
    mode : str, optional
        Optimization mode ``"min"`` or ``"max"``. Default is ``"max"``.
    ignore_reinit_error : bool, optional
        Whether to ignore Ray reinit errors (dev only). Default is ``False``.
    time_budget_hours : float, optional
        Max time budget in hours. Default is ``24.0``.
    use_asha : bool, optional
        Whether to use ASHA scheduler. Default is ``False``.
    asha_config : dict, optional
        ASHA config (max_t, grace_period, reduction_factor, brackets).
    gpus_per_trial : int, optional
        GPUs per trial (0 = all available). Default is ``0``.
    enable_performance_monitoring : bool, optional
        Enable performance (time) monitoring. Default is ``False``.
    performance_output_dir : str, optional
        Directory for performance results.
    ray_results_dir : str, optional
        Directory for Ray temporary results.
    run_opt_main : callable, optional
        Callable that runs a single optimization trial given a config path
        (typically ``run_opt.main`` from the model directory). Required.

    Raises
    ------
    ValueError
        If config is missing required sections (dataset, model, hyperparameters)
        or if `run_opt_main` is not provided.

    Notes
    -----
    **Class Methods:**

    **run_optimization():**

    - Run the full tuning workflow: create search space, Tuner, fit(), save best config and history when applicable.
    - Returns:
        - None.

    **run_from_cli(description):** (static)

    - Run tuning from the command line: parse CLI, import model-local ``run_opt.main``, run ModelTuner per config.
    - Parameters:
        - description : str, optional. Description for the argument parser. Default ``"CTF Model Hyperparameter Tuner"``.
    - Returns:
        - None.

    **_construct_output_dir():**

    - Construct the output directory path from model, dataset, and pair_id.
      Structure: ``results/tune_results/{model_name}/{dataset_name}/pair_id_{pair_ids}/{timestamp}/``.
    - Returns:
        - Path. Constructed output directory path.

    **_infer_model_name(config_path):** (static)

    - Infer model name from directory structure (e.g. parent of ``tuning_config``).
    - Parameters:
        - config_path : str. Path to the config file.
    - Returns:
        - str. Inferred model name.
    - Raises ``ValueError`` if model name cannot be inferred.

    **_validate_config(self, config):**

    - Validate that the configuration contains required sections: dataset, model, hyperparameters.
    - Parameters:
        - config : dict. Configuration dictionary to validate.
    - Returns:
        - None.
    - Raises ``ValueError`` if required sections are missing or invalid.

    **_validate_param_space(self, param_space):**

    - Validate the parameter space (types, bounds, choices, etc.).
    - Parameters:
        - param_space : dict. Parameter space dictionary to validate.
    - Returns:
        - None.
    - Raises ``ValueError`` if parameter space is empty or invalid.

    **_objective(self, config):**

    - Objective function for Ray Tune: generate config, run model via ``run_opt_main``, sum results.
    - Parameters:
        - config : dict. Trial hyperparameter configuration.
    - Returns:
        - Dict[str, float]. Dictionary with the optimization metric (e.g. score).

    **_parse_pair_ids(dataset_config):** (static)

    - Parse pair_id from dataset config (int, list of int, or ``'all'``).
    - Parameters:
        - dataset_config : dict. The ``dataset`` section from the config file.
    - Returns:
        - list of int or None. Pair IDs to optimize for, or None for all pairs.
    - Raises ``ValueError`` if pair_id is not int, list of ints, or ``'all'``.

    **_sum_results(self, results):**

    - Sum metric values from the results dictionary (all pairs in ``results['pairs']``).
    - Parameters:
        - results : dict. Evaluation results from run_opt (pairs with metrics).
    - Returns:
        - float. Sum of all metric values.

    **_create_search_space(self, tuning_config):**

    - Build a Ray Tune search space dictionary from the tuning config.
    - Parameters:
        - tuning_config : dict. Hyperparameter spec (type, lower_bound, upper_bound, choices, etc.).
    - Returns:
        - dict. Ray Tune search space.
    - Raises ``Exception`` if required keys or types are missing or unsupported.

    **_generate_config(self, config, template, name):**

    - Generate a YAML config file with the given hyperparameters and save to output_dir.
    - Parameters:
        - config : dict. Selected hyperparameters.
        - template : dict. Config template to fill.
        - name : str. Output filename base (without extension).
    - Returns:
        - str. Path to the generated configuration file.

    **_get_resources():**

    - Get resource configuration (cpu, gpu) for Ray Tune trials.
    - Returns:
        - dict. Keys ``cpu`` and ``gpu`` for per-trial resources.
    """

    def __init__(
        self,
        config_path: str,
        model_name: str | None = None,
        save_final_config: bool = True,
        metric: str = "score",
        mode: str = "max",
        ignore_reinit_error: bool = False,
        time_budget_hours: float = 24.0,  # Default time budget of 24 hours
        use_asha: bool = False,  # Whether to use ASHA scheduler
        asha_config: dict[str, Any] | None = None,  # Configuration for ASHA scheduler
        gpus_per_trial: int = 0,  # Number of GPUs to use per trial (0 means use all available)
        enable_performance_monitoring: bool = False,  # Whether to enable performance monitoring (average time per run)
        performance_output_dir: str | None = None,  # Directory for performance results
        ray_results_dir: str | None = None,  # Directory for Ray temporary results
        run_opt_main: Callable[[str], None] | None = None,
    ) -> None:
        r"""Initialize the ModelTuner; see class docstring for parameters."""

        if run_opt_main is None:
            raise ValueError(
                "ModelTuner requires a run_opt_main callable that accepts a config path. "
                "Import it from your model's run_opt.py and pass it to ModelTuner."
            )
        self._run_opt_main: Callable[[str], None] = run_opt_main
        # Load and validate configuration
        with open(config_path, "r") as f:
            self.hp_config = yaml.safe_load(f)
        self._validate_config(self.hp_config)

        # Extract parameter space from config
        self.param_space = self.hp_config.get("hyperparameters", {})
        self._validate_param_space(self.param_space)

        # Determine model_name with fallback hierarchy: provided -> config file -> directory structure
        # First try to get from config file if not explicitly provided
        if not model_name:
            model_name = self.hp_config.get("model", {}).get("name")
        # If still no model_name, infer it from directory structure
        if not model_name:
            model_name = self._infer_model_name(config_path=config_path)

        self.model_name = model_name
        print(f"Model_name: {self.model_name}")

        # Parse and store pair_ids for objective function filtering
        self.pair_ids = self._parse_pair_ids(self.hp_config["dataset"])

        self.save_final_config = save_final_config
        self.metric = metric
        self.mode = mode
        self.ignore_reinit_error = ignore_reinit_error
        self.time_budget_hours = time_budget_hours

        # Inform user if multiple pair_ids are being used
        if self.pair_ids is not None and len(self.pair_ids) > 1:
            print(
                f"Note: Processing {len(self.pair_ids)} pair_ids ({self.pair_ids}). Each trial will take approximately {len(self.pair_ids)}x longer."
            )
        self.use_asha = use_asha
        self.gpus_per_trial = gpus_per_trial
        self.ray_results_dir = ray_results_dir

        # Initialize output directory
        self.output_dir = self._construct_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Performance monitoring setup
        self.enable_performance_monitoring = enable_performance_monitoring
        self.performance_output_dir = performance_output_dir or str(self.output_dir / "performance_results")
        self.performance_monitor = None
        if self.enable_performance_monitoring:
            print(f"Performance monitoring enabled (time only), output directory: {self.performance_output_dir}")

        # Set up optional ASHA configuration
        self.asha_config = asha_config or {
            "max_t": 100,  # Maximum number of training iterations
            "grace_period": 10,  # Minimum number of iterations before stopping
            "reduction_factor": 3,  # Factor to reduce the number of trials
            "brackets": 1,  # Number of brackets for ASHA
        }

        # Initialize Ray if not already initialized
        if not ray.is_initialized():  # pyrefly: ignore
            try:
                ray_init_kwargs: dict[str, Any] = {
                    "ignore_reinit_error": self.ignore_reinit_error,
                    "include_dashboard": False,  # Disable dashboard for local runs
                    "_system_config": {
                        "object_spilling_threshold": 0.8,  # 80% memory threshold
                        "object_store_full_delay_ms": 100,  # Delay when store is full
                    },
                }

                # Add custom results directory if specified
                if self.ray_results_dir:
                    ray_init_kwargs["_temp_dir"] = self.ray_results_dir

                ray.init(**ray_init_kwargs)
                resources = ray.cluster_resources()  # pyrefly: ignore
                print("Ray initialized successfully with resources:")
                print(f"  - CPUs: {resources.get('CPU', 0)}")
                print(f"  - GPUs: {resources.get('GPU', 0)}")
                if self.ray_results_dir:
                    print(f"  - Ray results directory: {self.ray_results_dir}")
            except Exception as e:
                print(f"Warning: Ray initialization had issues: {e!s}")
                print("Attempting to continue with local execution...")
                ray_init_kwargs = {"ignore_reinit_error": self.ignore_reinit_error, "local_mode": True}
                if self.ray_results_dir:
                    ray_init_kwargs["_temp_dir"] = self.ray_results_dir
                ray.init(**ray_init_kwargs)

    def _construct_output_dir(self) -> Path:
        r"""Construct the output directory path for tune results.

        Path is built from `model_name`, dataset name, and pair_id. Structure:
        ``results/tune_results/{model_name}/{dataset_name}/pair_id_{pair_ids}/{timestamp}/``.

        Returns
        -------
        Path
            Constructed output directory path.
        """
        # Get model name
        model_name = self.model_name

        # Get dataset name and pair IDs
        dataset_name = self.hp_config["dataset"]["name"]
        pair_ids_str = "_".join(map(str, self.pair_ids)) if self.pair_ids else "all"

        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct path
        output_dir = (
            top_dir / "results" / "tune_results" / model_name / dataset_name / f"pair_id_{pair_ids_str}" / timestamp
        )

        return output_dir

    @staticmethod
    def _infer_model_name(config_path: str) -> str:
        r"""Infer model name from directory structure.

        Expects config under a ``tuning_config`` directory; the parent of
        that directory is taken as the model name (e.g. ``models/MyModel/tuning_config/``).

        Parameters
        ----------
        config_path : str
            Path to the config file.

        Returns
        -------
        str
            Inferred model name.

        Raises
        ------
        ValueError
            If model name cannot be inferred from `config_path`.
        """

        # Config files are typically in: models/{model_name}/tuning_config/config_*.yaml
        config_path_obj = Path(config_path)
        parts = config_path_obj.resolve().parts

        try:
            idx = parts.index("tuning_config")
            if idx > 0:
                return parts[idx - 1]
        except ValueError:
            pass

        raise ValueError(
            f"Could not infer model_name from config_path {config_path}. "
            f"Please provide model_name parameter, ensure config file contains model.name, "
            f"or ensure config_path contains 'tuning_config' directory"
        )

    def _validate_config(self, config: dict[str, Any]) -> None:
        r"""Validate that the configuration has required sections.

        Checks for the presence of ``dataset``, ``model``, and ``hyperparameters``
        in `config`.

        Parameters
        ----------
        config : dict
            Configuration dictionary to validate.

        Raises
        ------
        ValueError
            If any required section is missing.
        """
        required_sections = ["dataset", "model", "hyperparameters"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in config: {section}")

    def _validate_param_space(self, param_space: dict[str, Any]) -> None:
        r"""Validate the parameter space configuration.

        Ensures each parameter has a valid ``type`` (e.g. uniform, loguniform,
        choice, grid_search), required bounds or choices, and ``q`` for
        q-prefixed types.

        Parameters
        ----------
        param_space : dict
            Parameter space dictionary to validate.

        Raises
        ------
        ValueError
            If parameter space is empty or any parameter has invalid type,
            bounds, or missing keys.
        """
        if not param_space:
            raise ValueError("Parameter space cannot be empty")

        for param_name, param_config in param_space.items():
            if not isinstance(param_config, dict):
                raise TypeError(f"Parameter {param_name} must be a dictionary")

            param_type = param_config.get("type")
            if not param_type:
                raise ValueError(f"Missing type for parameter {param_name}")

            if param_config["type"] not in [
                "uniform",
                "quniform",
                "loguniform",
                "qloguniform",
                "randn",
                "qrandn",
                "randint",
                "qrandint",
                "lograndint",
                "qlograndint",
                "choice",
                "grid_search",
            ]:
                raise ValueError(f"Invalid type for hyperparameter {param_name}")

            # Validate bounds for numeric parameters
            if param_config["type"] in [
                "uniform",
                "quniform",
                "loguniform",
                "qloguniform",
                "randn",
                "qrandn",
                "randint",
                "qrandint",
                "lograndint",
                "qlograndint",
            ]:
                if "lower_bound" not in param_config or "upper_bound" not in param_config:
                    raise ValueError(f"Missing bounds for hyperparameter {param_name}")
                if param_config["lower_bound"] >= param_config["upper_bound"]:
                    raise ValueError(
                        f"Invalid bounds for hyperparameter {param_name}: lower_bound must be less than upper_bound"
                    )

            # Validate q for q-prefixed parameters
            if param_type.startswith("q") and "q" not in param_config:
                raise ValueError(f"Missing q value for parameter {param_name}")

            # Validate choices for choice type
            if param_config["type"] == "choice" and "choices" not in param_config:
                raise ValueError(f"Missing choices for parameter {param_name}")

            # Validate grid for grid_search type
            if param_config["type"] == "grid_search" and "grid" not in param_config:
                raise ValueError(f"Missing grid values for parameter {param_name}")

    def _objective(self, config: dict[str, Any]) -> dict[str, float]:
        r"""Objective function for Ray Tune: run one trial and return the metric.

        Called by Ray Tune for each trial. Generates a config with the trial's
        hyperparameters, runs the model via `run_opt_main`, loads results from
        the run's YAML, and returns a dict with the optimization metric (e.g. score).
        On failure returns a poor score (e.g. ``-inf`` for maximize).

        Parameters
        ----------
        config : dict
            Trial hyperparameter configuration from Ray Tune.

        Returns
        -------
        dict
            Single key (the metric name) with the trial's score (float).
        """
        try:
            # Get batch_id
            batch_id = str(tune.get_context().get_trial_id())

            # Create a copy of the blank config to avoid modifying the original
            trial_config = self.blank_config.copy()

            # Add batch_id to model config
            trial_config["model"]["batch_id"] = batch_id

            # Create config file
            config_path = self._generate_config(config, trial_config, f"hp_config_{batch_id}")

            # Run model via injected callable
            try:
                self._run_opt_main(config_path)
            except Exception as e:
                print(f"Training failed: {e!s}")
                traceback.print_exc()
                # Return a very poor score to indicate failure
                return {self.metric: float("-inf") if self.mode == "max" else float("inf")}

            # Extract results and clean up files
            # Get the directory where run_opt.py is located
            run_opt_dir = Path(self._run_opt_main.__code__.co_filename).parent
            results_path = run_opt_dir / f"results_{batch_id}.yaml"

            if not results_path.exists():
                print(f"Results file not found: {results_path}")
                return {self.metric: float("-inf") if self.mode == "max" else float("inf")}

            with open(results_path, "r") as f:
                results = yaml.safe_load(f)
            results_path.unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)

            # Sum results (run_opt.py only evaluates the pair_ids specified in the config)
            score = self._sum_results(results)

        except Exception as e:
            print(f"Error in objective function: {e!s}")
            # Return a very poor score to indicate failure
            return {self.metric: float("-inf") if self.mode == "max" else float("inf")}

        # Return score with metric name
        return {self.metric: score}

    @staticmethod
    def _parse_pair_ids(dataset_config: dict[str, Any]) -> list[int] | None:
        r"""Parse pair_id from the dataset section of the config.

        Accepts a single int, a list of ints, or ``'all'``. Returns ``None``
        for ``'all'`` to mean all pairs.

        Parameters
        ----------
        dataset_config : dict
            The ``dataset`` section from the config file.

        Returns
        -------
        list of int or None
            Pair IDs to optimize for, or ``None`` if all pairs should be used.

        Raises
        ------
        ValueError
            If `pair_id` is not an int, list of ints, or ``'all'``.
        """
        pair_id_config = dataset_config.get("pair_id", "all")

        if pair_id_config == "all":
            return None
        elif isinstance(pair_id_config, int):
            return [pair_id_config]
        elif isinstance(pair_id_config, list):
            # Validate that all elements are integers
            if not all(isinstance(item, int) for item in pair_id_config):
                raise ValueError(f"Invalid pair_id: {pair_id_config}. All elements must be integers.")
            return pair_id_config
        else:
            raise ValueError(f"Invalid pair_id configuration: {pair_id_config}. Expected int, list of ints, or 'all'")

    def _sum_results(self, results):
        r"""Sum all metric values from the evaluation results.

        Iterates over ``results['pairs']`` and sums each pair's metric values.
        run_opt only evaluates the pair_ids in the config, so all entries
        are for the requested pairs.

        Parameters
        ----------
        results : dict
            Evaluation results from run_opt (must have a ``pairs`` key with
            list of dicts containing ``metrics``).

        Returns
        -------
        float
            Sum of all metric values across all pairs.
        """
        total = 0
        for pair_dict in results["pairs"]:
            metric_dict = pair_dict["metrics"]
            for metric in metric_dict:
                total += metric_dict[metric]
        return total

    def _create_search_space(self, tuning_config: dict[str, Any]) -> dict[str, Any]:
        r"""Build a Ray Tune search space from the tuning config.

        Maps each parameter in `tuning_config` to a Ray Tune sampler (e.g.
        ``tune.uniform``, ``tune.choice``, ``tune.grid_search``) based on
        ``type`` and optional ``lower_bound``, ``upper_bound``, ``choices``,
        ``grid``, ``q``, etc.

        Parameters
        ----------
        tuning_config : dict
            Hyperparameter spec keyed by parameter name. Each value is a dict
            with at least ``type``; numeric types need ``lower_bound`` and
            ``upper_bound``; ``choice`` needs ``choices``; ``grid_search``
            needs ``grid``; q-prefixed types need ``q``.

        Returns
        -------
        dict
            Ray Tune search space (suitable for `param_space`).

        Raises
        ------
        Exception
            If ``type`` is missing, unsupported, or required keys (e.g. bounds,
            choices) are missing for a parameter.
        """
        search_space: dict[str, Any] = {}
        for name, value in tuning_config.items():
            param_dict = value
            if "type" not in param_dict:
                raise KeyError(f"'type' not in {param_dict} keys")

            if param_dict["type"] == "uniform":
                search_space[name] = tune.uniform(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                )
            elif param_dict["type"] == "quniform":
                search_space[name] = tune.quniform(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                    param_dict["q"],
                )
            elif param_dict["type"] == "loguniform":
                search_space[name] = tune.loguniform(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                )
            elif param_dict["type"] == "qloguniform":
                search_space[name] = tune.qloguniform(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                    param_dict["q"],
                )
            elif param_dict["type"] == "randn":
                search_space[name] = tune.randn(param_dict["lower_bound"], param_dict["upper_bound"])
            elif param_dict["type"] == "qrandn":
                search_space[name] = tune.qrandn(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                    param_dict["q"],
                )
            elif param_dict["type"] == "randint":
                search_space[name] = tune.randint(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                )
            elif param_dict["type"] == "qrandint":
                search_space[name] = tune.qrandint(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                    param_dict["q"],
                )
            elif param_dict["type"] == "lograndint":
                search_space[name] = tune.lograndint(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                )
            elif param_dict["type"] == "qlograndint":
                search_space[name] = tune.qlograndint(
                    param_dict["lower_bound"],
                    param_dict["upper_bound"],
                    param_dict["q"],
                )
            elif param_dict["type"] == "choice":
                search_space[name] = tune.choice(param_dict["choices"])
            elif param_dict["type"] == "grid_search":
                search_space[name] = tune.grid_search(param_dict["grid"])
            else:
                raise ValueError(f"Parameter type {param_dict['type']} not supported.")

        return search_space

    def _generate_config(self, config: dict[str, Any], template: dict[str, Any], name: str) -> str:
        r"""Generate a YAML config file with the given hyperparameters.

        Writes `config` into ``template['model']`` and saves the result as
        ``{name}.yaml`` in `output_dir`. Modifies `template` in place.

        Parameters
        ----------
        config : dict
            Selected hyperparameters to write into the template.
        template : dict
            Configuration template; ``template['model']`` is updated with
            keys from `config`.
        name : str
            Output filename without extension (e.g. ``"hp_config_trial_1"``).

        Returns
        -------
        str
            Path to the generated YAML file.
        """
        # Fill out dictionary
        for blank_key, value in config.items():
            template["model"][blank_key] = value
        # Save config
        config_path = self.output_dir / f"{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(template, f)
        return str(config_path)

    def _get_resources(self) -> dict[str, Any]:
        r"""Get per-trial resource configuration for Ray Tune.

        Uses Ray cluster resources; reserves 2 CPUs for overhead and divides
        the rest across trials. When GPUs are available, respects
        `gpus_per_trial` (0 means use all GPUs).

        Returns
        -------
        dict
            Keys ``cpu`` and ``gpu``: number of CPUs and GPUs per trial.
        """
        # Let Ray automatically detect available resources
        resources = ray.cluster_resources()  # pyrefly: ignore
        cpu_count = int(resources.get("CPU", 1))
        gpu_count = int(resources.get("GPU", 0))

        # Log available resources
        print("\nAvailable resources:")
        print(f"  - CPUs: {cpu_count}")
        print(f"  - GPUs: {gpu_count}")

        # Reserve 2 CPUs for overhead
        reserved_cpus = 2
        available_cpus = max(1, cpu_count - reserved_cpus)

        if gpu_count > 0:
            # If gpus_per_trial is 0, use all available GPUs
            gpus_to_use = gpu_count if self.gpus_per_trial == 0 else min(self.gpus_per_trial, gpu_count)
            # Calculate how many parallel trials we can run
            num_parallel_trials = max(1, gpu_count // gpus_to_use)
            cpus_per_trial = max(1, available_cpus // num_parallel_trials)

            # Log resource allocation
            print("\nResource allocation:")
            print(f"  - GPUs per trial: {gpus_to_use}")
            print(f"  - CPUs per trial: {cpus_per_trial}")
            print(f"  - Number of parallel trials: {num_parallel_trials}")

            return {"cpu": cpus_per_trial, "gpu": gpus_to_use}
        else:
            # When no GPUs are available, use a reasonable default number of CPUs per trial
            default_trials = 4  # Default number of parallel trials when no GPUs are available
            cpus_per_trial = max(1, available_cpus // default_trials)

            # Log resource allocation
            print("\nResource allocation (CPU-only):")
            print(f"  - CPUs per trial: {cpus_per_trial}")
            print(f"  - Number of parallel trials: {default_trials}")

            return {"cpu": cpus_per_trial, "gpu": 0}

    def run_optimization(self) -> None:
        r"""Run the complete hyperparameter optimization workflow.

        Creates the search space from config, builds a Ray Tune Tuner with
        optional ASHA scheduler, runs ``tuner.fit()``, and saves the best
        config and tuning history when `save_final_config` is True. Trial
        count is controlled by ``n_trials`` in config and/or `time_budget_hours`;
        tuning stops when either limit is reached. Raises if no trials complete
        or best result cannot be obtained.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If no trials complete successfully or best result/config/metrics
            are unavailable.
        """
        # Initialize performance monitoring if enabled
        if self.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor(output_dir=self.performance_output_dir)
            self.performance_monitor.start_monitoring()
        # Create a copy of the configuration to avoid modifying the original
        self.blank_config = self.hp_config.copy()

        # Separate hyperparameters from the main config
        hyperparameters = self.blank_config.pop("hyperparameters", {})

        # Generate parameter dictionary for Ray Tune
        param_dict = self._create_search_space(hyperparameters)

        # Create Ray Tune object
        trainable = tune.with_resources(self._objective, self._get_resources())

        # Convert time budget from hours to seconds
        time_budget_s = int(self.time_budget_hours * 3600)

        # Configure scheduler if ASHA is enabled
        scheduler = None
        if self.use_asha:
            print("\nUsing ASHA scheduler for early stopping with configuration:")
            for key, value in self.asha_config.items():
                print(f"- {key}: {value}")
            print()
            scheduler = ASHAScheduler(
                max_t=self.asha_config["max_t"],
                grace_period=self.asha_config["grace_period"],
                reduction_factor=self.asha_config["reduction_factor"],
                brackets=self.asha_config["brackets"],
            )

        # Create tune config
        tune_config = tune.TuneConfig(metric=self.metric, mode=self.mode, scheduler=scheduler)

        # Get n_trials from config if present
        n_trials = self.blank_config["model"].get("n_trials")
        if n_trials is not None:
            tune_config.num_samples = n_trials
            print(f"\nUsing n_trials from config: {n_trials}")
        else:
            # If no n_trials specified, set a very large number so time budget controls stopping
            tune_config.num_samples = 100000

        # Always set time budget
        if time_budget_s > 0:
            tune_config.time_budget_s = time_budget_s
            print(f"Using time budget: {self.time_budget_hours} hours")

        # Create tuner with custom results directory
        tuner_kwargs: dict[str, Any] = {"trainable": trainable, "param_space": param_dict, "tune_config": tune_config}

        if self.ray_results_dir:
            tuner_kwargs["run_config"] = tune.RunConfig(storage_path=self.ray_results_dir)

        tuner = tune.Tuner(**tuner_kwargs)

        # Run optimization
        results = tuner.fit()

        # Stop performance monitoring if enabled
        if self.enable_performance_monitoring and self.performance_monitor:
            # Get total trials and time in simple one-liners
            total_trials = len(results)
            total_time = sum(result.metrics.get("time_total_s", 0.0) for result in results)  # pyrefly: ignore

            # Update performance monitor
            self.performance_monitor.run_count = total_trials
            self.performance_monitor.total_time = total_time

            performance_summary = self.performance_monitor.stop_monitoring()
            print("\nPerformance Summary:")
            print(f"  Total trials: {performance_summary['total_num_runs']}")
            print(f"  Average time per trial: {performance_summary['average_time_per_run_seconds']:.2f}s")
            print(f"  Total trial time: {performance_summary['total_run_time_hours']:.2f}h")

        # Check if any trials completed successfully
        if not results:
            raise RuntimeError("No trials completed successfully. Check the logs for more details.")

        try:
            # Try to get the best result
            result = results.get_best_result(metric=self.metric, mode=self.mode)

            if result is None:
                raise RuntimeError("No best result found. Check the logs for more details.")
            if result.config is None:
                raise RuntimeError("No best config found. Check the logs for more details.")
            if result.metrics is None:
                raise RuntimeError("No best metrics found. Check the logs for more details.")

            best_config = result.config
            best_value = result.metrics[self.metric]
            print(f"Best {self.metric}: {best_value} (params: {best_config})")

            # Save results
            if self.save_final_config:  # Only False when unit testing
                # Save optimal parameters
                pair_ids_str = "".join(map(str, self.pair_ids)) if self.pair_ids else "all"
                self.blank_config["model"].pop("batch_id", None)
                self.blank_config["model"].pop("n_trials", None)
                self.blank_config["model"].pop("train_split", None)
                config_path = self._generate_config(
                    best_config,
                    self.blank_config,
                    f"optimal_params_{self.blank_config['dataset']['name']}_{pair_ids_str}",
                )
                print("Optimal parameters saved to:", config_path)

                # Save tuning history
                history_path = self.output_dir / f"tuning_history_{self.model_name}.yaml"
                with open(history_path, "w") as f:
                    yaml.dump(
                        {
                            "best_config": best_config,
                            "best_value": best_value,
                            "all_results": results,
                            "final_config": self.blank_config,
                        },
                        f,
                    )
                print("Tuning history saved to:", history_path)
            else:
                print("Not saving results (unit testing mode).")
        except Exception as e:
            # If we can't get the best result, print all available results for debugging
            print("\nTuning failed. Available results:")
            for i, result in enumerate(results):  # pyrefly: ignore
                print(f"\nTrial {i}:")
                print(f"Config: {result.config}")
                print(f"Metrics: {result.metrics}")
                if hasattr(result, "error") and result.error:
                    print(f"Error: {result.error}")
            raise RuntimeError(f"Failed to get best result: {e!s}")

    @staticmethod
    def run_from_cli(description: str = "CTF Model Hyperparameter Tuner") -> None:
        r"""Run tuning from the command line.

        Parses CLI args (e.g. ``--config-path``, ``--time-budget-hours``),
        imports the model-local ``run_opt.main`` from the caller's directory,
        and runs `ModelTuner` for each specified config. For parallel runs
        across nodes, use SLURM scripts; this CLI is for single-node use.

        Parameters
        ----------
        description : str, optional
            Description for the argument parser. Default is
            ``"CTF Model Hyperparameter Tuner"``.

        Returns
        -------
        None
        """
        # Get the directory of the calling script
        # Assuming the calling script is in the model directory: optimize_parameters.py
        caller_frame = sys._getframe(1)
        caller_path = caller_frame.f_code.co_filename
        caller_dir = Path(caller_path).parent

        parser = argparse.ArgumentParser(description=description)

        # Basic arguments
        parser.add_argument("--config-path", help="Path to the model's config file")
        parser.add_argument(
            "--model-name",
            help="Specific model to tune (optional, will be inferred from config file orcurrent directory if not provided)",
        )
        parser.add_argument("--ray-results-dir", help="Directory for Ray temporary results (default: ~/ray_results)")

        # Tuning parameters
        tuning_group = parser.add_argument_group("Tuning Parameters")
        tuning_group.add_argument(
            "--time-budget-hours",
            type=float,
            default=24.0,
            help="Maximum time budget for tuning in hours (default: 24.0)",
        )
        tuning_group.add_argument("--metric", default="score", help="Metric to optimize (default: score)")
        tuning_group.add_argument(
            "--mode",
            choices=["min", "max"],
            default="max",
            help="Optimization mode: 'min' to minimize or 'max' to maximize the metric (default: max)",
        )
        tuning_group.add_argument(
            "--gpus-per-trial",
            type=int,
            default=0,
            help="Number of GPUs to use per trial (default: 0, meaning use all available GPUs)",
        )

        # Performance monitoring arguments
        performance_group = parser.add_argument_group("Performance Monitoring (optional)")
        performance_group.add_argument(
            "--enable-performance-monitoring",
            action="store_true",
            help="Enable performance monitoring (average time per run)",
        )
        performance_group.add_argument(
            "--performance-output-dir",
            help="Directory to save performance results (default: results/performance_results)",
        )

        # ASHA scheduler arguments
        asha_group = parser.add_argument_group("ASHA Scheduler (optional)")
        asha_group.add_argument("--use-asha", action="store_true", help="Use ASHA scheduler for early stopping")
        asha_group.add_argument(
            "--asha-max-t", type=int, default=100, help="Maximum number of training iterations for ASHA (default: 100)"
        )
        asha_group.add_argument(
            "--asha-grace-period",
            type=int,
            default=10,
            help="Minimum number of iterations before stopping for ASHA (default: 10)",
        )
        asha_group.add_argument(
            "--asha-reduction-factor",
            type=int,
            default=3,
            help="Factor to reduce the number of trials for ASHA (default: 3)",
        )
        asha_group.add_argument("--asha-brackets", type=int, default=1, help="Number of brackets for ASHA (default: 1)")

        args = parser.parse_args()

        # Prepare ASHA config if enabled
        asha_config = None
        if args.use_asha:
            asha_config = {
                "max_t": args.asha_max_t,
                "grace_period": args.asha_grace_period,
                "reduction_factor": args.asha_reduction_factor,
                "brackets": args.asha_brackets,
            }

        # Get config files
        if args.config_path:
            # Use the provided config file
            config_files = [Path(args.config_path)]
        else:
            # Find all config files in tuning_config directory relative to caller
            tuning_config_dir = caller_dir / "tuning_config"
            if not tuning_config_dir.exists():
                raise FileNotFoundError(
                    f"tuning_config directory not found at {tuning_config_dir}. "
                    f"Please provide --config-path or run from a model directory with tuning_config."
                )
            config_files = sorted(tuning_config_dir.glob("config_*.yaml"))
            if not config_files:
                raise FileNotFoundError(
                    f"No config_*.yaml files found in {tuning_config_dir}. "
                    f"Please provide --config-path or ensure tuning config files exist."
                )

        # Import the model-local run_opt.main; this assumes run_from_cli is
        # called from a model directory that contains run_opt.py.
        try:
            from run_opt import main as run_opt_main  # type: ignore[import] # TODO: how to handle this better?
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment issue
            raise FileNotFoundError(
                "Could not import 'run_opt'. Make sure ModelTuner.run_from_cli() is "
                "called from a model directory that contains run_opt.py."
            ) from exc

        print(f"Found {len(config_files)} config file(s):")
        for i, cfg in enumerate(config_files, 1):
            print(f"  {i}. {cfg}")

        # Run tuning for each config file
        for i, config_path in enumerate(config_files, 1):
            print(f"\nRunning tuning with config file {i}/{len(config_files)}: {config_path}")
            try:
                # Load config to get dataset and pair_id info
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                dataset_name = config["dataset"]["name"]

                # Validate and parse pair_ids
                pair_ids = ModelTuner._parse_pair_ids(config["dataset"])
                pair_ids_str = "_".join(map(str, pair_ids)) if pair_ids else "all"
                print(f"Dataset: {dataset_name}, Pair ID: {pair_ids_str}")

                tuner = ModelTuner(
                    model_name=args.model_name,  # Pass None if not provided, ModelTuner will infer from config file or directory structure
                    config_path=str(config_path),
                    time_budget_hours=args.time_budget_hours,
                    use_asha=args.use_asha,
                    asha_config=asha_config,
                    mode=args.mode,
                    metric=args.metric,
                    gpus_per_trial=args.gpus_per_trial,
                    enable_performance_monitoring=args.enable_performance_monitoring,
                    performance_output_dir=args.performance_output_dir,
                    ray_results_dir=args.ray_results_dir,
                    run_opt_main=run_opt_main,
                )
                tuner.run_optimization()
            except Exception as e:
                print(f"Error tuning with config file {config_path}: {e!s}")
                print("Continuing with next config file...")
                continue


if __name__ == "__main__":
    ModelTuner.run_from_cli()
