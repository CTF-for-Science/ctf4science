# Getting Started with Hyperparameter optimization

We will be using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) for our hyperparameter optimization framework. A minimum example is implemented in the [CTF_NaiveBaselines](https://github.com/CTF-for-Science/CTF_NaiveBaselines) repository. This document will explain the core principles.

## Installation

Make sure your environment contains the optional packages by running `pip install -e .[all]` from the top-level `ctf4science` repository.

## Usage

To run a hyperparameter optimization, run `optimize_parameters.py tuning_config/<filename>.yaml`. This will read in the hyperparameter configuration file and perform hyperparameter optimization using the specified values and ranges provided. It is your responsibility to modify the code in `optimize_parameters.py` to generate a suitable configuration file for your model.

Inside of `optimize_parameters.py`, `run_opt.py` is run. `run_opt.py` is like `run.py` except that it does not load the testing dataset, instead it generates and evaluates your model on a train/validation split of the original training dataset. Several helper functions have been implemented to get this to work: `load_validation_dataset`, `get_validation_training_timesteps`, and `get_validation_prediction_timesteps`.

The implementation of the two files `optimize_parameters.py` and `run_opt.py` are described in more detail in the next section.

## Files

### `optimize_parameters.py`
This is the high level optimization function. Its one argument is a hyperparameter specific configuration file.  

The code works by calling `ctf4science/tune_module`, which:
1. Uses a created a template configuration file (in `tuning_config/`)
2. Sets up RayTune
3. Runs an objective function, which takes the template configuration, populates it with the trial's selected hyperparameters
4. Runs `run_opt.py` with the generated configuration file
5. Reads in the generated `results_{...}.yaml` file from `run_opt.py` and sums all the scores
6. Go back to (2.) until all trials have completed

Note that the configuration files all have a unique batch_id to allow for parallelization  
Note also that all intermediate files are deleted and only a single final configuration file is saved  

### `run_opt.py`
This is run in the inner loop of the `optimize_parameters.py` script.
The code works exactly as `run.py` except it saves the `results_{...}.yaml` file using the trial ID and only uses the training/validation split of the original dataset.

### `tuning_config/`
This directory contains hyperparameter optimization configurations. An example file is here:

```yaml
dataset:
  name: PDE_KS
  pair_id: [8]
model:
  name: model
  train_split: 0.8
  seed: 0
  n_trials: 3
hyperparameters:
  lag:
    type: randint
    lower_bound: 5
    upper_bound: 15
  horizon:
    type: randint
    lower_bound: 5
    upper_bound: 15
  n_kernels:
    type: choice
    choices: [8, 16, 32, 64, 128]
  n_blocks:
    type: randint
    lower_bound: 3
    upper_bound: 6
  weight_decay:
    type: choice 
    choices: [0, 0.0001]
  lr:
    type: loguniform
    lower_bound: 0.0001
    upper_bound: 0.01
  dropout:
    type: choice
    choices: [0.1, 0.2]
```

All "types" of hyperparameters come from the `_create_search_space(...)` function in `tune_module.py`. This function takes a `tuning_config` file and returns a Ray Tune dictionary that is directly used in hyperparameter optimization.

Note: the `pair_id` parameter is a list of pair_ids to run your hyperparameter optimization on. The "score" that is calculates is a sum of all evaluation metrics for all pair_ids. If you want to optimize one model over many pair_ids, make this parameter a list of those pair_ids. If you want to optimize one model per pair_id, make it a list of a single integer.
