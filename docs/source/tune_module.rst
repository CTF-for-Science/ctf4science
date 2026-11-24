Tune Module
===========

We will be using `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_ for our hyperparameter optimization framework. A minimum example is implemented in the `CTF_NaiveBaselines <https://github.com/CTF-for-Science/CTF_NaiveBaselines>`_ repository. This document will explain the core principles.

Installation
------------

Make sure your environment contains the optional packages by running ``pip install -e .[all]`` from the top-level ``ctf4science`` repository.

Usage
-----

To run a hyperparameter optimization, run ``optimize_parameters.py tuning_config/<filename>.yaml``. This will read in the hyperparameter configuration file and perform hyperparameter optimization using the specified values and ranges provided. It is your responsibility to modify the code in ``optimize_parameters.py`` to generate a suitable configuration file for your model.

Inside of ``optimize_parameters.py``, ``run_opt.py`` is run. ``run_opt.py`` is like ``run.py`` except that it does not load the testing dataset, instead it generates and evaluates your model on a train/validation split of the original training dataset. Several helper functions have been implemented to get this to work: ``load_validation_dataset``, ``get_validation_training_timesteps``, and ``get_validation_prediction_timesteps``.

The implementation of the two files ``optimize_parameters.py`` and ``run_opt.py`` are described in more detail in the next section.

Files
-----

``optimize_parameters.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the high level optimization function. Its one argument is a hyperparameter specific configuration file.

The code works by calling ``ctf4science/tune_module``, which:

1. Uses a created a template configuration file (in ``tuning_config/``)
2. Sets up RayTune
3. Runs an objective function, which takes the template configuration, populates it with the trial's selected hyperparameters
4. Runs ``run_opt.py`` with the generated configuration file
5. Reads in the generated ``results_{...}.yaml`` file from ``run_opt.py`` and sums the scores (see `Objective and optimization mode`_)
6. Go back to (2.) until all trials have completed

Note that the configuration files all have a unique batch_id to allow for parallelization

Note also that all intermediate files are deleted and only a single final configuration file is saved

``run_opt.py``
~~~~~~~~~~~~~~

This is run in the inner loop of the ``optimize_parameters.py`` script.
The code works exactly as ``run.py`` except it saves the ``results_{...}.yaml`` file using the trial ID and only uses the training/validation split of the original dataset.

``tuning_config/``
~~~~~~~~~~~~~~~~~~

This directory contains hyperparameter optimization configurations. An example file is here:

.. code-block:: yaml

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

All "types" of hyperparameters come from the ``_create_search_space(...)`` function in ``tune_module.py``. This function takes a ``tuning_config`` file and returns a Ray Tune dictionary that is directly used in hyperparameter optimization.

Note: the ``pair_id`` parameter selects the pair_ids to run your hyperparameter optimization on. It can be a single integer, a list of integers, an inclusive range string such as ``"1-6"``, or ``all``. The "score" that is calculated is a sum of the evaluation metrics over all of these pair_ids (see `Objective and optimization mode`_). If you want to optimize one model over many pair_ids, make this parameter a list (or range) of those pair_ids. If you want to optimize one model per pair_id, make it a list of a single integer.

Objective and optimization mode
-------------------------------

Each trial's score is the sum of the metrics that ``run_opt.py`` writes for the tuned pairs. Two optional top-level keys of the dataset YAML (``data/<dataset>/<dataset>.yaml``) control how that score is built and in which direction it is optimized:

.. code-block:: yaml

   lower_is_better: true   # metrics are errors: the tuner minimises
   tuning_metric: mae      # sum only this metric over the tuned pairs

``lower_is_better``
   Sets the default optimization mode. ``ModelTuner`` takes ``mode=None`` by default (and ``--mode`` on the command line defaults to unset), which means "read it from the dataset": ``min`` if the dataset YAML sets ``lower_is_better: true``, otherwise ``max``. The existing datasets do not set the key, so they are still maximized. An explicit ``mode`` argument or ``--mode min|max`` always wins over the dataset setting.

``tuning_metric``
   When set, the score sums only this metric over the tuned pairs, one value per pair, instead of every metric of every pair. Use it when a pair's metrics measure the same thing in different ways (e.g. MAE, RMSE and MASE), where summing them would weight them arbitrarily. Every tuned pair must list ``tuning_metric`` in its ``metrics``; the tuner raises an error at start-up otherwise. Without the key, all metrics are summed.

A trial that fails, produces no results, or produces results that miss a tuned pair (matched on each entry's ``pair_id``) gets the worst score a real trial can reach: each counted metric at its clipped worst value (-100 when maximizing, +100 when minimizing). The number of counted metrics is every default metric of each tuned pair, or one per tuned pair when ``tuning_metric`` is set. Failed trials therefore rank last without using infinities.

The ``--metric`` option (default ``score``) is unrelated to ``tuning_metric``: it is only the name under which the summed score is reported to Ray Tune.
