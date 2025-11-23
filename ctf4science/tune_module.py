import yaml
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
from typing import Dict, Any, Optional, List
import datetime
import argparse
import logging
import time
import sys

from run_opt import main as run_opt_main
from ctf4science.performance_module import PerformanceMonitor

top_dir = Path(__file__).parent.parent

class ModelTuner:
    """
    Orchestrates hyperparameter tuning for CTF models using Ray Tune.
    
    Supports tuning with a specific config file or automatically detecting all config files
    in a model's tuning_config/config*.yaml.
    """
    def __init__(
        self,
        config_path: str,
        model_name: Optional[str] = None,
        save_final_config: bool = True,
        metric: str = "score",
        mode: str = "max",
        ignore_reinit_error: bool = False,
        time_budget_hours: float = 24.0,  # Default time budget of 24 hours
        use_asha: bool = False,  # Whether to use ASHA scheduler
        asha_config: Optional[Dict[str, Any]] = None,  # Configuration for ASHA scheduler
        gpus_per_trial: int = 0,  # Number of GPUs to use per trial (0 means use all available)
        enable_performance_monitoring: bool = False,  # Whether to enable performance monitoring (average time per run)
        performance_output_dir: Optional[str] = None,  # Directory for performance results
        ray_results_dir: Optional[str] = None  # Directory for Ray temporary results
    ) -> None:
        """
        Initialize the ModelTuner with configuration file.

        Args:
            config_path: Path to the configuration file containing dataset, model, and hyperparameter specifications.
            model_name: Optional model name. If not provided, it will be extracted from the config file or inferred from the file structure.
            save_final_config: Whether to save the final configuration file (default: True).
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
            gpus_per_trial: Number of GPUs to use per trial (default: 0). Set to 0 to use all available GPUs.
            enable_performance_monitoring: Whether to enable performance monitoring (average time per run) (default: False).
            performance_output_dir: Directory for performance results. If None, uses default location.
            ray_results_dir: Directory for Ray temporary results. If None, uses default ~/ray_results.

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
        
        # Determine model_name with fallback hierarchy: provided -> config file -> directory structure
        # First try to get from config file if not explicitly provided
        if not model_name:
            model_name = self.hp_config.get('model', {}).get('name')
        # If still no model_name, infer it from directory structure
        if not model_name:
            model_name = self._infer_model_name(config_path=config_path)

        self.model_name = model_name
        print(f"Model_name: {self.model_name}")
        
        # Parse and store pair_ids for objective function filtering
        self.pair_ids = self._parse_pair_ids(self.hp_config['dataset'])
        
        self.save_final_config = save_final_config
        self.metric = metric
        self.mode = mode
        self.ignore_reinit_error = ignore_reinit_error
        self.time_budget_hours = time_budget_hours
        
        # Inform user if multiple pair_ids are being used
        if self.pair_ids is not None and len(self.pair_ids) > 1:
            print(f"Note: Processing {len(self.pair_ids)} pair_ids ({self.pair_ids}). Each trial will take approximately {len(self.pair_ids)}x longer.")
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
            'max_t': 100,  # Maximum number of training iterations
            'grace_period': 10,  # Minimum number of iterations before stopping
            'reduction_factor': 3,  # Factor to reduce the number of trials
            'brackets': 1  # Number of brackets for ASHA
        }
            
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            try:
                ray_init_kwargs = {
                    "ignore_reinit_error": self.ignore_reinit_error,
                    "include_dashboard": False,  # Disable dashboard for local runs
                    "_system_config": {
                        "object_spilling_threshold": 0.8,  # 80% memory threshold
                        "object_store_full_delay_ms": 100,  # Delay when store is full
                    }
                }
                
                # Add custom results directory if specified
                if self.ray_results_dir:
                    ray_init_kwargs["_temp_dir"] = self.ray_results_dir
                    
                ray.init(**ray_init_kwargs)
                resources = ray.cluster_resources()
                print(f"Ray initialized successfully with resources:")
                print(f"  - CPUs: {resources.get('CPU', 0)}")
                print(f"  - GPUs: {resources.get('GPU', 0)}")
                if self.ray_results_dir:
                    print(f"  - Ray results directory: {self.ray_results_dir}")
            except Exception as e:
                print(f"Warning: Ray initialization had issues: {str(e)}")
                print("Attempting to continue with local execution...")
                ray_init_kwargs = {
                    "ignore_reinit_error": self.ignore_reinit_error,
                    "local_mode": True
                }
                if self.ray_results_dir:
                    ray_init_kwargs["_temp_dir"] = self.ray_results_dir
                ray.init(**ray_init_kwargs)

    def _construct_output_dir(self) -> Path:
        """
        Construct the output directory path programmatically based on model, dataset and pair_id information.

        The directory structure is:
        results/tune_results/
            {model_name}/
                {dataset_name}/
                    pair_id_{pair_id}/
                        {timestamp}/

        Returns:
            Path: Constructed output directory path.
        """
        # Get model name
        model_name = self.model_name
        
        # Get dataset name and pair IDs
        dataset_name = self.hp_config['dataset']['name']
        pair_ids_str = '_'.join(map(str, self.pair_ids)) if self.pair_ids else 'all'
        
        # Create timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Construct path
        output_dir = top_dir / 'results' / 'tune_results' / model_name / dataset_name / f'pair_id_{pair_ids_str}' / timestamp
        
        return output_dir

    @staticmethod
    def _infer_model_name(config_path: str) -> str:
        """
        Infer model name from directory structure.
        
        Args:
            config_path: Path to the config file
            
        Returns:
            str: Inferred model name
            
        Raises:
            ValueError: If model name cannot be inferred
        """
        
        # Config files are typically in: models/{model_name}/tuning_config/config_*.yaml
        config_path_obj = Path(config_path)
        parts = config_path_obj.resolve().parts
        
        try:
            idx = parts.index('tuning_config')
            if idx > 0:
                return parts[idx - 1]
        except ValueError:
            pass
        
        raise ValueError(
            f"Could not infer model_name from config_path {config_path}. "
            f"Please provide model_name parameter, ensure config file contains model.name, "
            f"or ensure config_path contains 'tuning_config' directory"
        )

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
            
            # Sum results (run_opt.py only evaluates the pair_ids specified in the config)
            score = self._sum_results(results)
            # Return score with metric name
            return {self.metric: score}
            
        except Exception as e:
            print(f"Error in objective function: {str(e)}")
            # Return a very poor score to indicate failure
            return {self.metric: float('-inf') if self.mode == 'max' else float('inf')}

    @staticmethod
    def _parse_pair_ids(dataset_config: Dict[str, Any]) -> Optional[List[int]]:
        """
        Parse the pair_id configuration to determine which pair_ids to optimize for.
        
        Validates that pair_id is either an integer, a list of integers, or 'all'.
        
        Args:
            dataset_config: The 'dataset' section from the config file.
        
        Returns:
            Optional[List[int]]: A list of pair_ids to optimize for, or None if 'all' pairs should be used.
            
        Raises:
            ValueError: If pair_id is not an int, list of ints, or 'all'
        """
        pair_id_config = dataset_config.get('pair_id', 'all')
        
        if pair_id_config == 'all':
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
        """
        Sums metric values from a results dictionary containing evaluation metrics.
        
        Note: run_opt.py only evaluates the pair_ids specified in the config,
        so all results in the dictionary are for the specified pair_ids.

        Args:
            results (dict): A dictionary containing evaluation results.
        
        Returns:
            float: The sum of all metric values from all pairs in the results.
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
            - Writes a new YAML configuration file to the output directory
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
        1. Extracting hyperparameters from config
        2. Initializing Ray Tune tuner
        3. Running the tuning process
        4. Saving results to files

        The number of trials is determined by:
        - n_trials from config file if present
        - time_budget_hours (always used with a default value of 24 hours)
        If both are specified, tuning stops when either limit is reached.
        """
        # Initialize performance monitoring if enabled
        if self.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor(output_dir=self.performance_output_dir)
            self.performance_monitor.start_monitoring()
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
        else:
            # If no n_trials specified, set a very large number so time budget controls stopping
            tune_config.num_samples = 100000
        
        # Always set time budget
        if time_budget_s > 0:
            tune_config.time_budget_s = time_budget_s
            print(f"Using time budget: {self.time_budget_hours} hours")
        
        # Create tuner with custom results directory
        tuner_kwargs = {
            "trainable": trainable,
            "param_space": param_dict,
            "tune_config": tune_config
        }
        
        if self.ray_results_dir:
            tuner_kwargs["run_config"] = tune.RunConfig(storage_path=self.ray_results_dir)
        
        tuner = tune.Tuner(**tuner_kwargs)
        
        # Run optimization
        results = tuner.fit()

        # Stop performance monitoring if enabled
        if self.enable_performance_monitoring and self.performance_monitor:
            # Get total trials and time in simple one-liners
            total_trials = len(results)
            total_time = sum(result.metrics.get("time_total_s", 0.0) for result in results)
            
            # Update performance monitor
            self.performance_monitor.run_count = total_trials
            self.performance_monitor.total_time = total_time
            
            performance_summary = self.performance_monitor.stop_monitoring()
            print(f"\nPerformance Summary:")
            print(f"  Total trials: {performance_summary['total_num_runs']}")
            print(f"  Average time per trial: {performance_summary['average_time_per_run_seconds']:.2f}s")
            print(f"  Total trial time: {performance_summary['total_run_time_hours']:.2f}h")

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
                pair_ids_str = ''.join(map(str, self.pair_ids)) if self.pair_ids else 'all'
                self.blank_config['model'].pop('batch_id', None)
                self.blank_config['model'].pop('n_trials', None)
                self.blank_config['model'].pop('train_split', None)
                config_path = self._generate_config(best_config, self.blank_config, f'optimal_params_{self.blank_config["dataset"]["name"]}_{pair_ids_str}')
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
            for i, result in enumerate(results):
                print(f"\nTrial {i}:")
                print(f"Config: {result.config}")
                print(f"Metrics: {result.metrics}")
                if hasattr(result, 'error') and result.error:
                    print(f"Error: {result.error}")
            raise RuntimeError(f"Failed to get best result: {str(e)}")

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
        # Assuming the calling script is in the model directory: optimize_parameters.py
        caller_frame = sys._getframe(1)
        caller_path = caller_frame.f_code.co_filename
        caller_dir = Path(caller_path).parent
        
        parser = argparse.ArgumentParser(description=description)
        
        # Basic arguments
        parser.add_argument("--config-path", help="Path to the model's config file")
        parser.add_argument("--model-name", help="Specific model to tune (optional, will be inferred from config file orcurrent directory if not provided)")
        parser.add_argument("--ray-results-dir", help="Directory for Ray temporary results (default: ~/ray_results)")
        
        # Tuning parameters
        tuning_group = parser.add_argument_group('Tuning Parameters')
        tuning_group.add_argument("--time-budget-hours", type=float, default=24.0, 
                                help="Maximum time budget for tuning in hours (default: 24.0)")
        tuning_group.add_argument("--metric", default="score", help="Metric to optimize (default: score)")
        tuning_group.add_argument("--mode", choices=["min", "max"], default="max",
                                help="Optimization mode: 'min' to minimize or 'max' to maximize the metric (default: max)")
        tuning_group.add_argument("--gpus-per-trial", type=int, default=0,
                                help="Number of GPUs to use per trial (default: 0, meaning use all available GPUs)")
        
        # Performance monitoring arguments
        performance_group = parser.add_argument_group('Performance Monitoring (optional)')
        performance_group.add_argument("--enable-performance-monitoring", action="store_true",
                                     help="Enable performance monitoring (average time per run)")
        performance_group.add_argument("--performance-output-dir", 
                                     help="Directory to save performance results (default: results/performance_results)")
        
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
        
        # Prepare ASHA config if enabled
        asha_config = None
        if args.use_asha:
            asha_config = {
                'max_t': args.asha_max_t,
                'grace_period': args.asha_grace_period,
                'reduction_factor': args.asha_reduction_factor,
                'brackets': args.asha_brackets
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
        
        print(f"Found {len(config_files)} config file(s):")
        for i, cfg in enumerate(config_files, 1):
            print(f"  {i}. {cfg}")
        
        # Run tuning for each config file
        for i, config_path in enumerate(config_files, 1):
            print(f"\nRunning tuning with config file {i}/{len(config_files)}: {config_path}")
            try:
                # Load config to get dataset and pair_id info
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                dataset_name = config['dataset']['name']
                
                # Validate and parse pair_ids
                pair_ids = ModelTuner._parse_pair_ids(config['dataset'])
                pair_ids_str = '_'.join(map(str, pair_ids)) if pair_ids else 'all'
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
                    ray_results_dir=args.ray_results_dir
                )
                tuner.run_optimization()
            except Exception as e:
                print(f"Error tuning with config file {config_path}: {str(e)}")
                print("Continuing with next config file...")
                continue

if __name__ == "__main__":
    ModelTuner.run_from_cli() 
