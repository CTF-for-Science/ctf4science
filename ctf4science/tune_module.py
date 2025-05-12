import os
import yaml
import numpy as np
import ray
import json
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
import importlib
import datetime
import argparse
import logging
import time
import sys

from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids
from ctf4science.eval_module import evaluate
from run_opt import main as run_opt_main

class TuningRunner:
    """
    A class for running hyperparameter tuning of CTF models using Ray Tune.
    
    This class provides functionality for:
    - Loading and validating tuning configurations
    - Defining and validating parameter spaces
    - Running hyperparameter optimization with Ray Tune
    - Saving and managing tuning results
    """
    def __init__(
        self,
        config_path: str,
        save_final_config: bool = True,
        metric: str = "score",
        mode: str = "max",
        ignore_reinit_error: bool = False,
        time_budget_hours: float = 24.0,  # Default time budget of 24 hours
        use_asha: bool = False,  # Whether to use ASHA scheduler
        asha_config: Optional[Dict[str, Any]] = None,  # ASHA configuration
        output_dir: Optional[str] = None,  # Optional custom output directory
        gpus_per_trial: int = 0  # Number of GPUs to use per trial (0 means use all available)
    ) -> None:
        """
        Initialize the TuningRunner with configuration file.

        Args:
            config_path: Path to the configuration file containing dataset, model, and hyperparameter specifications.
            save_final_config: Whether to save the final configuration file to output_dir (default: True).
            metric: Metric to optimize (default: "score").
            mode: Optimization mode, "min" or "max" (default: "max").
            ignore_reinit_error: Whether to ignore Ray reinitialization errors (default: False).
                Set to True only during development/testing. Not recommended for production.
            time_budget_hours: Maximum time budget for tuning in hours (default: 24.0).
                If both time_budget_hours (from cli) and n_trials (from config) are specified, tuning will stop when either limit is reached.
            use_asha: Whether to use ASHA scheduler for early stopping (default: False).
            asha_config: Optional configuration for ASHA scheduler. If None and use_asha is True, default values will be used:
                {
                    'max_t': 100,  # Maximum number of training iterations
                    'grace_period': 10,  # Minimum number of iterations before stopping
                    'reduction_factor': 3,  # Factor to reduce the number of trials
                    'brackets': 1  # Number of brackets for ASHA
                }
            output_dir: Optional custom output directory. If None, a default directory will be constructed.
            gpus_per_trial: Number of GPUs to use per trial (default: 0). Set to 0 to use all available GPUs.

        Raises:
            ValueError: If config is missing required fields.
        """
        
        # Load and validate configuration
        with open(config_path, 'r') as f:
            self.hp_config = yaml.safe_load(f)
        self._validate_config(self.hp_config)
        
        # Extract parameter space from config
        self.param_space = self.hp_config.get('hyperparameters', {})
        self._validate_param_space(self.param_space)
        self.model_name = self.hp_config['model']['name']
        
        self.save_final_config = save_final_config
        self.metric = metric
        self.mode = mode
        self.ignore_reinit_error = ignore_reinit_error
        self.time_budget_hours = time_budget_hours
        self.use_asha = use_asha
        self.gpus_per_trial = gpus_per_trial
        
        # Set up optional ASHA configuration
        self.asha_config = asha_config or {
            'max_t': 100,  # Maximum number of training iterations
            'grace_period': 10,  # Minimum number of iterations before stopping
            'reduction_factor': 3,  # Factor to reduce the number of trials
            'brackets': 1  # Number of brackets for ASHA
        }

        # Initialize output directory
        self.output_dir = self._construct_output_dir(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            try:
                ray.init(
                    ignore_reinit_error=self.ignore_reinit_error,
                    include_dashboard=False,  # Disable dashboard for local runs
                    _system_config={
                        "object_spilling_threshold": 0.8,  # 80% memory threshold
                        "object_store_full_delay_ms": 100,  # Delay when store is full
                    }
                )
                resources = ray.cluster_resources()
                print(f"Ray initialized successfully with resources:")
                print(f"  - CPUs: {resources.get('CPU', 0)}")
                print(f"  - GPUs: {resources.get('GPU', 0)}")
            except Exception as e:
                print(f"Warning: Ray initialization had issues: {str(e)}")
                print("Attempting to continue with local execution...")
                ray.init(
                    ignore_reinit_error=self.ignore_reinit_error,
                    local_mode=True  # Fall back to local mode if there are issues
                )

    def _construct_output_dir(self, output_dir: Optional[str] = None) -> Path:
        """
        Construct the output directory path programmatically based on model, dataset and pair_id information.

        The directory structure is:
        results/tune_results/
            {model_name}/
                {dataset_name}/
                    pair_id_{pair_id}/
                        {timestamp}/

        Args:
            output_dir: Optional custom output directory. If provided, this will be used instead of constructing a path.

        Returns:
            Path: Constructed output directory path.
        """
        if output_dir is not None:
            # Use the custom output directory if provided
            return Path(output_dir)
        
        # Get model name
        model_name = self.model_name
        
        # Get dataset name and pair IDs
        dataset_name = self.hp_config['dataset']['name']
        pair_ids = '_'.join(map(str, self.hp_config['dataset']['pair_id']))
        
        # Create timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Construct path
        output_dir = Path(__file__).parent.parent / 'results' / 'tune_results' / model_name / dataset_name / f'pair_id_{pair_ids}' / timestamp
        
        return output_dir

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary.

        This function checks that the configuration contains all required sections:
        dataset, model, and hyperparameters.

        Args:
            config: Configuration dictionary to validate.

        Raises:
            ValueError: If required fields are missing or have invalid values.
        """
        required_sections = ['dataset', 'model', 'hyperparameters']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in config: {section}")
        # TODO: add more validation

    def _validate_param_space(self, param_space: Dict[str, Any]) -> None:
        """
        Validate the parameter space configuration.

        This function checks that the parameter space is properly defined with valid
        parameter types and bounds. It supports various parameter types including
        uniform, loguniform, and randn distributions.

        Args:
            param_space: Parameter space dictionary to validate.

        Raises:
            ValueError: If parameter space is empty or contains invalid definitions.
        """
        if not param_space:
            raise ValueError("Parameter space cannot be empty")
            
        for param_name, param_config in param_space.items():
            if not isinstance(param_config, dict):
                raise ValueError(f"Parameter {param_name} must be a dictionary")
                
            param_type = param_config.get('type')
            if not param_type:
                raise ValueError(f"Missing type for parameter {param_name}")
                
            if param_config['type'] not in ['uniform', 'quniform', 'loguniform', 'qloguniform', 
                                          'randn', 'qrandn', 'randint', 'qrandint', 
                                          'lograndint', 'qlograndint', 'choice', 'grid_search']:
                raise ValueError(f"Invalid type for hyperparameter {param_name}")
                
            # Validate bounds for numeric parameters
            if param_config['type'] in ['uniform', 'quniform', 'loguniform', 'qloguniform', 
                                      'randn', 'qrandn', 'randint', 'qrandint', 
                                      'lograndint', 'qlograndint']:
                if 'lower_bound' not in param_config or 'upper_bound' not in param_config:
                    raise ValueError(f"Missing bounds for hyperparameter {param_name}")
                if param_config['lower_bound'] >= param_config['upper_bound']:
                    raise ValueError(f"Invalid bounds for hyperparameter {param_name}: lower_bound must be less than upper_bound")
                    
            # Validate q for q-prefixed parameters
            if param_type.startswith('q'):
                if 'q' not in param_config:
                    raise ValueError(f"Missing q value for parameter {param_name}")

            # Validate choices for choice type
            if param_config['type'] == 'choice':
                if 'choices' not in param_config:
                    raise ValueError(f"Missing choices for parameter {param_name}")

            # Validate grid for grid_search type
            if param_config['type'] == 'grid_search':
                if 'grid' not in param_config:
                    raise ValueError(f"Missing grid values for parameter {param_name}")

    def _objective(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Objective function for hyperparameter optimization.

        This function is called by Ray Tune for each trial. It:
        1. Gets the trial ID
        2. Generates a configuration file with the trial's hyperparameters
        3. Runs the model with the configuration
        4. Extracts and returns the results

        Args:
            config (dict): Dictionary containing the hyperparameter configuration.

        Returns:
            Dict[str, float]: Dictionary containing the optimization metric (score).
        """
        try:
            # Get batch_id
            batch_id = str(tune.get_context().get_trial_id())
            
            # Create a copy of the blank config to avoid modifying the original
            trial_config = self.blank_config.copy()
            
            # Add batch_id to model config
            trial_config['model']['batch_id'] = batch_id
            
            # Create config file
            config_path = self._generate_config(config, trial_config, f'hp_config_{batch_id}')
            
            # Run model
            try:
                run_opt_main(config_path)
            except Exception as e:
                print(f"Training failed: {str(e)}")
                # Return a very poor score to indicate failure
                return {self.metric: float('-inf') if self.mode == 'max' else float('inf')}
            
            # Extract results and clean up files
            # Get the directory where run_opt.py is located
            run_opt_dir = Path(run_opt_main.__code__.co_filename).parent
            results_path = run_opt_dir / f'results_{batch_id}.yaml'
            
            if not results_path.exists():
                print(f"Results file not found: {results_path}")
                return {self.metric: float('-inf') if self.mode == 'max' else float('inf')}
                
            with open(results_path, 'r') as f:
                results = yaml.safe_load(f)
            results_path.unlink(missing_ok=True)
            Path(config_path).unlink(missing_ok=True)
            
            score = self._sum_results(results)
            # Return score with metric name
            return {self.metric: score}
            
        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            # Return a very poor score to indicate failure
            return {self.metric: float('-inf') if self.mode == 'max' else float('inf')}

    def _sum_results(self, results):
        """
        Sums all metric values from a results dictionary containing evaluation metrics.
        
        Iterates through all pairs in the results dictionary and sums all metric values
        found in each pair's 'metrics' dictionary. This is used to aggregate
        evaluation metrics from a batch_results.yaml file.

        Args:
            results (dict): A dictionary containing evaluation results.
        
        Returns:
            float: The sum of all metric values across all pairs in the results dictionary.
        """
        total = 0
        for pair_dict in results['pairs']:
            metric_dict = pair_dict['metrics']
            for metric in metric_dict.keys():
                total += metric_dict[metric]
        return total

    def _create_search_space(self, tuning_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a Ray Tune search space dictionary from the tuning config file.

        Args:
            tuning_config (dict):
                Dictionary containing the parameter specification with the following keys:
                - 'type': str, either 'float' or 'int' indicating the parameter type
                - 'lower_bound': float/int, the minimum value for the parameter
                - 'upper_bound': float/int, the maximum value for the parameter
                - 'log': bool, whether to sample in log space

        Returns:
            dict: Ray Tune expected search_space dictionary

        Raises:
            Exception:
                If any of the required keys ('type', 'lower_bound', 'upper_bound', 'log')
                are missing from tuning_config for a parameter.
                If the parameter type is neither 'float' nor 'int'.
        """
        search_space = {}
        for name in tuning_config.keys():
            param_dict = tuning_config[name]
            if 'type' not in param_dict:
                raise Exception(f"\'type\' not in {param_dict} keys")

            if param_dict['type'] == "uniform":
                search_space[name] = tune.uniform(param_dict['lower_bound'], param_dict['upper_bound'])
            elif param_dict['type'] == "quniform":
                search_space[name] = tune.quniform(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
            elif param_dict['type'] == "loguniform":
                search_space[name] = tune.loguniform(param_dict['lower_bound'], param_dict['upper_bound'])
            elif param_dict['type'] == "qloguniform":
                search_space[name] = tune.qloguniform(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
            elif param_dict['type'] == "randn":
                search_space[name] = tune.randn(param_dict['lower_bound'], param_dict['upper_bound'])
            elif param_dict['type'] == "qrandn":
                search_space[name] = tune.qrandn(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
            elif param_dict['type'] == "randint":
                search_space[name] = tune.randint(param_dict['lower_bound'], param_dict['upper_bound'])
            elif param_dict['type'] == "qrandint":
                search_space[name] = tune.qrandint(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
            elif param_dict['type'] == "lograndint":
                search_space[name] = tune.lograndint(param_dict['lower_bound'], param_dict['upper_bound'])
            elif param_dict['type'] == "qlograndint":
                search_space[name] = tune.qlograndint(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
            elif param_dict['type'] == "choice":
                search_space[name] = tune.choice(param_dict['choices'])
            elif param_dict['type'] == "grid_search":
                search_space[name] = tune.grid_search(param_dict['grid'])
            else:
                raise Exception(f"Parameter type {param_dict['type']} not supported.")

        return search_space

    def _generate_config(self, config: Dict[str, Any], template: Dict[str, Any], name: str) -> str:
        """
        Generates a configuration file with suggested hyperparameter values.

        This function suggests a value for the constant parameter using Raytun's config,
        updates the configuration template with this value, and saves the resulting
        configuration to a YAML file.

        Args:
            config (dict): Dictionary containing selected hyperparameters.
            template (dict): Configuration template dictionary that will be populated with
                the suggested values.
            name (str): Name to use for the output configuration file (without extension).

        Returns:
            str: Path to the generated configuration file.

        Side Effects:
            - Writes a new YAML configuration file to ./config/{name}.yaml
            - Modifies the input template dictionary by adding the suggested constant value
        """
        # Fill out dictionary
        for blank_key in config.keys():
            template['model'][blank_key] = config[blank_key]
        # Save config
        config_path = self.output_dir / f'{name}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(template, f)
        return str(config_path)

    def _get_resources(self) -> Dict[str, Any]:
        """
        Get the resource configuration for Ray Tune trials.

        Returns:
            Dict[str, Any]: Resource configuration dictionary for Ray Tune with keys:
                - cpu: Number of CPUs per trial
                - gpu: Number of GPUs per trial
        """
        # Let Ray automatically detect available resources
        resources = ray.cluster_resources()
        cpu_count = int(resources.get('CPU', 1))
        gpu_count = int(resources.get('GPU', 0))

        # Log available resources
        print(f"\nAvailable resources:")
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
            print(f"\nResource allocation:")
            print(f"  - GPUs per trial: {gpus_to_use}")
            print(f"  - CPUs per trial: {cpus_per_trial}")
            print(f"  - Number of parallel trials: {num_parallel_trials}")
            
            return {
                "cpu": cpus_per_trial,
                "gpu": gpus_to_use
            }
        else:
            # When no GPUs are available, use a reasonable default number of CPUs per trial
            default_trials = 4  # Default number of parallel trials when no GPUs are available
            cpus_per_trial = max(1, available_cpus // default_trials)
            
            # Log resource allocation
            print(f"\nResource allocation (CPU-only):")
            print(f"  - CPUs per trial: {cpus_per_trial}")
            print(f"  - Number of parallel trials: {default_trials}")

            return {
                "cpu": cpus_per_trial,
                "gpu": 0
            }

    def run_optimization(self) -> None:
        """
        Run the complete optimization workflow.

        This method handles the entire optimization process including:
        1. Loading hyperparameters
        2. Initializing Ray Tune tuner
        3. Running the tuning process
        4. Saving results to files

        The number of trials is determined by:
        - n_trials from config file if present
        - time_budget_hours (always used with a default value of 24 hours)
        If both are specified, tuning stops when either limit is reached.
        """
        # Create a copy of the configuration to avoid modifying the original
        self.blank_config = self.hp_config.copy()
        
        # Separate hyperparameters from the main config
        hyperparameters = self.blank_config.pop('hyperparameters', {})

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
                max_t=self.asha_config['max_t'],
                grace_period=self.asha_config['grace_period'],
                reduction_factor=self.asha_config['reduction_factor'],
                brackets=self.asha_config['brackets']
            )
        
        # Create tune config
        tune_config = tune.TuneConfig(
            metric=self.metric,
            mode=self.mode,
            scheduler=scheduler
        )
        
        # Get n_trials from config if present
        n_trials = self.blank_config['model'].get('n_trials')
        if n_trials is not None:
            tune_config.num_samples = n_trials
            print(f"\nUsing n_trials from config: {n_trials}")
        
        # Always set time budget
        if time_budget_s > 0:
            tune_config.time_budget_s = time_budget_s
            print(f"Using time budget: {self.time_budget_hours} hours")
        
        tuner = tune.Tuner(
            trainable,
            param_space=param_dict,
            tune_config=tune_config
        )
        
        # Run optimization
        results = tuner.fit()

        # Check if any trials completed successfully
        if not results:
            raise RuntimeError("No trials completed successfully. Check the logs for more details.")

        try:
            # Try to get the best result
            result = results.get_best_result(metric=self.metric, mode=self.mode)
            best_config = result.config
            best_value = result.metrics[self.metric]
            print(f"Best {self.metric}: {best_value} (params: {best_config})")

            # Save results
            if self.save_final_config:  # Only False when unit testing
                # Save optimal parameters
                pair_ids = ''.join(map(str, self.blank_config["dataset"]["pair_id"]))
                self.blank_config['model'].pop('batch_id', None)
                self.blank_config['model'].pop('n_trials', None)
                self.blank_config['model'].pop('train_split', None)
                config_path = self._generate_config(best_config, self.blank_config, f'optimal_params_{self.blank_config["dataset"]["name"]}_{pair_ids}')
                print("Optimal parameters saved to:", config_path)

                # Save tuning history
                history_path = self.output_dir / f"tuning_history_{self.model_name}.yaml"
                with open(history_path, 'w') as f:
                    yaml.dump({
                        'best_config': best_config,
                        'best_value': best_value,
                        'all_results': results,
                        'final_config': self.blank_config
                    }, f)
                print("Tuning history saved to:", history_path)
            else:
                print("Not saving results (unit testing mode).")
        except Exception as e:
            # If we can't get the best result, print all available results for debugging
            print("\nTuning failed. Available results:")
            for trial in results:
                print(f"\nTrial {trial.trial_id if hasattr(trial, 'trial_id') else 'Unknown'}:")
                print(f"Status: {trial.status if hasattr(trial, 'status') else 'Unknown'}")
                print(f"Config: {trial.config if hasattr(trial, 'config') else 'Unknown'}")
                if hasattr(trial, 'metrics'):
                    print(f"Metrics: {trial.metrics}")
                if hasattr(trial, 'error'):
                    print(f"Error: {trial.error}")
            raise RuntimeError(f"Failed to get best result: {str(e)}")

    def _get_cpu_usage(self) -> float:
        """Get average CPU usage during tuning."""
        if not ray.is_initialized():
            return 0.0
        resources = ray.cluster_resources()
        used_resources = ray.available_resources()
        return (1 - used_resources.get('CPU', 0) / resources.get('CPU', 1)) * 100

    def _get_gpu_usage(self) -> float:
        """Get average GPU usage during tuning."""
        if not ray.is_initialized():
            return 0.0
        resources = ray.cluster_resources()
        used_resources = ray.available_resources()
        return (1 - used_resources.get('GPU', 0) / resources.get('GPU', 1)) * 100

    def _get_memory_usage(self) -> float:
        """Get average memory usage during tuning."""
        if not ray.is_initialized():
            return 0.0
        resources = ray.cluster_resources()
        used_resources = ray.available_resources()
        return (1 - used_resources.get('memory', 0) / resources.get('memory', 1)) * 100

class ModelTuner:
    """
    Orchestrates hyperparameter tuning for CTF models. Config files are automatically
    detected in tuning_config/config*.yaml under each model.
    Supports three modes:
    1. Single model tuning
    2. Multiple model tuning (with specified models)
    3. All models tuning
    In single model tuning, if no config file is detected, it raises an error.
    In multiple or all model tuning, if no config files are detected, it logs a warning and skips the model.
    """
    def __init__(
        self,
        models_dir: str = "models",
        log_dir: Optional[str] = None,
        log_to_file: bool = True
    ):
        """
        Initialize the ModelTuner.
        
        Args:
            models_dir: Directory containing model directories
            log_dir: Directory to save logs. If None and log_to_file is True, logs will be saved to "logs"
            log_to_file: Whether to log to a file in addition to console output
        """
        # Convert models_dir to absolute path if it's relative
        self.models_dir = Path(models_dir).resolve()
        if not self.models_dir.exists():
            raise ValueError(f"Models directory does not exist: {self.models_dir}")
            
        # Set up logging directory if needed
        if log_to_file:
            self.log_dir = Path(log_dir) if log_dir else Path("logs")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = None
        
        # Set up logging
        self._setup_logging(log_to_file=log_to_file)
        
        # Log initialization
        self.logger.info(f"Initialized ModelTuner with models directory: {self.models_dir}")
    
    def _setup_logging(self, log_to_file: bool = True):
        """
        Set up logging configuration.
        
        Args:
            log_to_file: Whether to log to a file in addition to console output.
                        If False, only logs to console.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Always set up console logging
        handlers = [logging.StreamHandler()]
        
        # Optionally set up file logging
        if log_to_file:
            log_file = self.log_dir / f"tuning_{timestamp}.log"
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger("ModelTuner")
    
    def _find_tuning_configs(self, model_name: Optional[str] = None) -> List[Path]:
        """
        Find tuning configuration files.
        
        Args:
            model_name: Optional specific model to look for. If None, finds all models.
            
        Returns:
            List of paths to tuning configuration files.
            
        Note:
            If model_name is specified and no config files are found, raises FileNotFoundError.
            If model_name is None (finding all models), returns empty list for models without configs.
        """
        if model_name:
            search_pattern = self.models_dir / model_name / "tuning_config" / "config_*.yaml"
            config_files = list(search_pattern.parent.glob(search_pattern.name))
            if not config_files:
                raise FileNotFoundError(f"No tuning config files found for model {model_name}")
            return config_files
        else:
            # Find all model directories
            model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            config_files = []
            
            for model_dir in model_dirs:
                config_dir = model_dir / "tuning_config"
                if config_dir.exists():
                    model_configs = list(config_dir.glob("config_*.yaml"))
                    if not model_configs:
                        self.logger.warning(f"No tuning config files found in {config_dir}, skipping model {model_dir.name}")
                    else:
                        config_files.extend(model_configs)
                else:
                    self.logger.warning(f"No tuning_config directory found in {model_dir}, skipping model {model_dir.name}")
            
            if not config_files:
                self.logger.warning("No tuning config files found in any model directory")
            
            return config_files
    
    def tune_model(
        self,
        model_name: str,
        config_path: str,
        time_budget_hours: float = 24.0,
        use_asha: bool = False,
        asha_config: Optional[Dict[str, Any]] = None,
        mode: str = "max",
        metric: str = "score",
        output_dir: Optional[str] = None,
        gpus_per_trial: int = 0
    ) -> None:
        """
        Tune a single model.
        
        Args:
            model_name: Name of the model to tune
            config_path: Path to the model's config file
            time_budget_hours: Maximum time budget for tuning in hours
            use_asha: Whether to use ASHA scheduler for early stopping
            asha_config: Optional configuration for ASHA scheduler
            mode: Optimization mode, max by default
            metric: Metric to optimize, score by default
            output_dir: Optional custom output directory
            gpus_per_trial: Number of GPUs to use per trial (default: 0). Set to 0 to use all available GPUs.
        """
        self.logger.info(f"Starting tuning for model: {model_name}")
        
        try:
            # Initialize tuner
            tuner = TuningRunner(
                config_path=config_path,
                time_budget_hours=time_budget_hours,
                use_asha=use_asha,
                asha_config=asha_config,
                mode=mode,
                metric=metric,
                output_dir=output_dir,
                gpus_per_trial=gpus_per_trial
            )
            
            # Run tuning
            tuner.run_optimization()
            
            self.logger.info(f"Completed tuning for model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error tuning model {model_name}: {str(e)}")
            raise
    
    def tune_all_models_sequential(
        self,
        time_budget_hours: float = 24.0,
        use_asha: bool = False,
        asha_config: Optional[Dict[str, Any]] = None,
        mode: str = "max",
        metric: str = "score",
        output_dir: Optional[str] = None
    ) -> None:
        """
        Tune all models that have tuning configuration files sequentially.
        Not used at the moment due to conflicting package requirements among models.
        
        Note: This method runs models sequentially on a single node. For parallel execution
        across multiple nodes, use SLURM bash scripts to submit individual model tuning jobs.
        Each model will use all available resources on its node (CPUs and GPUs).
        
        Future improvements:
        - Add parallel execution support using SLURM job arrays
        - Each model would get its own node
        - Automated resource allocation and result collection
        
        Args:
            time_budget_hours: Maximum time budget for tuning in hours
            use_asha: Whether to use ASHA scheduler for early stopping
            asha_config: Optional configuration for ASHA scheduler
            mode: Optimization mode, "min" to minimize or "max" to maximize the metric
            metric: Metric to optimize
            output_dir: Optional custom output directory
        """
        self.logger.info("Starting sequential tuning for all models to ensure equal resource allocation")
        self.logger.info("Note: For parallel execution across multiple nodes, use SLURM bash scripts to submit individual model tuning jobs")
        
        # Find all tuning config files
        config_files = self._find_tuning_configs()
        
        if not config_files:
            self.logger.warning("No models to tune. Please ensure at least one model has tuning config files.")
            return
        
        # Run each model sequentially
        completed_models = []
        failed_models = []
        resource_usage = {}
        
        for config_path in config_files:
            model_name = config_path.parent.parent.name
            self.logger.info(f"\nStarting tuning for model: {model_name}")
            
            try:
                # Initialize tuner
                tuner = TuningRunner(
                    config_path=str(config_path),
                    time_budget_hours=time_budget_hours,
                    use_asha=use_asha,
                    asha_config=asha_config,
                    mode=mode,
                    metric=metric,
                    output_dir=output_dir
                )
                
                # Run tuning
                tuner.run_optimization()
                
                # Get resource usage metrics
                resource_metrics = {
                    "cpu_usage": tuner._get_cpu_usage(),
                    "gpu_usage": tuner._get_gpu_usage(),
                    "memory_usage": tuner._get_memory_usage()
                }
                
                completed_models.append(model_name)
                resource_usage[model_name] = resource_metrics
                self.logger.info(f"Completed tuning for model: {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error tuning model {model_name}: {str(e)}")
                failed_models.append(model_name)
                continue
        
        # Log results
        if completed_models:
            self.logger.info(f"\nSuccessfully tuned {len(completed_models)} models: {', '.join(completed_models)}")
            
            # Log resource usage comparison
            self.logger.info("\nResource Usage Comparison:")
            self.logger.info("=" * 50)
            for model in completed_models:
                self.logger.info(f"\nModel: {model}")
                for metric, value in resource_usage[model].items():
                    self.logger.info(f"  {metric}: {value}")
        
            if failed_models:
                self.logger.warning(f"Failed to tune {len(failed_models)} models: {', '.join(failed_models)}")
        else:
            self.logger.warning("No models were successfully tuned.")

    @staticmethod
    def run_from_cli(description: str = "CTF Model Hyperparameter Tuner") -> None:
        """
        This method provides a simple interface for running tuning from command line.
        
        Note: For parallel execution of multiple models across different nodes,
        use SLURM bash scripts to submit individual model tuning jobs.
        This CLI interface is currently designed for execution on a single node.
        
        Args:
            description: Description for the argument parser
        """
        # Get the directory of the calling script
        caller_frame = sys._getframe(1)
        caller_path = caller_frame.f_code.co_filename
        caller_dir = Path(caller_path).parent
        
        # Get the workspace root (two levels up from the model directory)
        workspace_root = caller_dir.parent.parent
        
        parser = argparse.ArgumentParser(description=description)
        
        # Basic arguments
        parser.add_argument("--models-dir", default=str(workspace_root / "models"), 
                          help="Directory containing model directories (default: models)")
        parser.add_argument("--output-dir", help="Directory to save tuning results (optional)")
        parser.add_argument("--config-path", help="Path to the model's config file")
        parser.add_argument("--model-name", help="Specific model to tune (optional, defaults to the model directory containing the script)")
        
        # Logging arguments
        logging_group = parser.add_argument_group('Logging Options')
        logging_group.add_argument("--log-dir", help="Directory to save logs (required if --log-to-file is used)")
        logging_group.add_argument("--log-to-file", action="store_true", help="Enable logging to file (requires --log-dir)")
        
        # Tuning parameters
        tuning_group = parser.add_argument_group('Tuning Parameters')
        tuning_group.add_argument("--time-budget-hours", type=float, default=24.0, 
                                help="Maximum time budget for tuning in hours (default: 24.0)")
        tuning_group.add_argument("--metric", default="score", help="Metric to optimize (default: score)")
        tuning_group.add_argument("--mode", choices=["min", "max"], default="max",
                                help="Optimization mode: 'min' to minimize or 'max' to maximize the metric (default: max)")
        tuning_group.add_argument("--gpus-per-trial", type=int, default=0,
                                help="Number of GPUs to use per trial (default: 0, meaning use all available GPUs)")
        
        # ASHA scheduler arguments
        asha_group = parser.add_argument_group('ASHA Scheduler (optional)')
        asha_group.add_argument('--use-asha', action='store_true',
                              help='Use ASHA scheduler for early stopping')
        asha_group.add_argument('--asha-max-t', type=int, default=100,
                              help='Maximum number of training iterations for ASHA (default: 100)')
        asha_group.add_argument('--asha-grace-period', type=int, default=10,
                              help='Minimum number of iterations before stopping for ASHA (default: 10)')
        asha_group.add_argument('--asha-reduction-factor', type=int, default=3,
                              help='Factor to reduce the number of trials for ASHA (default: 3)')
        asha_group.add_argument('--asha-brackets', type=int, default=1,
                              help='Number of brackets for ASHA (default: 1)')
        
        args = parser.parse_args()
        
        # Validate logging arguments
        if args.log_to_file and not args.log_dir:
            parser.error("--log-dir is required when --log-to-file is used")
        
        # Prepare ASHA config if enabled
        asha_config = None
        if args.use_asha:
            asha_config = {
                'max_t': args.asha_max_t,
                'grace_period': args.asha_grace_period,
                'reduction_factor': args.asha_reduction_factor,
                'brackets': args.asha_brackets
            }
        
        # Initialize Tuner
        modelTuner = ModelTuner(
            models_dir=args.models_dir,
            log_dir=args.log_dir,
            log_to_file=args.log_to_file
        )
        
        # If model_name not specified, try to detect it from the file structure
        if not args.model_name:
            # Check if we're in a model directory
            if caller_dir.name in [d.name for d in Path(args.models_dir).iterdir() if d.is_dir()]:
                args.model_name = caller_dir.name
                modelTuner.logger.info(f"Automatically detected model name: {args.model_name}")
            else:
                modelTuner.logger.warning("No model name specified and could not detect from file structure. Will tune all models.")
        
        # Run tuning
        if args.model_name:
            # Tune single model
            if args.config_path:
                # Use provided config path
                config_path = args.config_path
                modelTuner.logger.info(f"Using provided config path: {config_path}")
                modelTuner.tune_model(
                    model_name=args.model_name,
                    config_path=config_path,
                    time_budget_hours=args.time_budget_hours,
                    use_asha=args.use_asha,
                    asha_config=asha_config,
                    mode=args.mode,
                    metric=args.metric,
                    output_dir=args.output_dir,
                    gpus_per_trial=args.gpus_per_trial
                )
            else:
                # Find config files
                config_files = modelTuner._find_tuning_configs(args.model_name)
                if not config_files:
                    raise FileNotFoundError(f"No tuning config files found for model {args.model_name}")
                
                modelTuner.logger.info(f"Found {len(config_files)} config files:")
                for i, cfg in enumerate(config_files, 1):
                    modelTuner.logger.info(f"  {i}. {cfg}")
                
                # Run tuning for each config file
                for i, config_path in enumerate(config_files, 1):
                    modelTuner.logger.info(f"\nRunning tuning with config file {i}/{len(config_files)}: {config_path}")
                    try:
                        # Load config to get dataset and pair_id info
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                        dataset_name = config['dataset']['name']
                        pair_id = '_'.join(map(str, config['dataset']['pair_id']))
                        modelTuner.logger.info(f"Dataset: {dataset_name}, Pair ID: {pair_id}")
                        
                        modelTuner.tune_model(
                            model_name=args.model_name,
                            config_path=str(config_path),
                            time_budget_hours=args.time_budget_hours,
                            use_asha=args.use_asha,
                            asha_config=asha_config,
                            mode=args.mode,
                            metric=args.metric,
                            output_dir=args.output_dir,
                            gpus_per_trial=args.gpus_per_trial
                        )
                    except Exception as e:
                        modelTuner.logger.error(f"Error tuning with config file {config_path}: {str(e)}")
                        modelTuner.logger.info("Continuing with next config file...")
                        continue
        else:
            # Tune all models sequentially
            modelTuner.tune_all_models_sequential(
                time_budget_hours=args.time_budget_hours,
                use_asha=args.use_asha,
                asha_config=asha_config,
                mode=args.mode,
                metric=args.metric,
                output_dir=args.output_dir
            )

if __name__ == "__main__":
    ModelTuner.run_from_cli() 
