Data Module
===========

The Data Module (``data_module.py``) provides loading and configuration utilities for CTF datasets. It handles train/test pairs, timestep generation for training and evaluation, validation splits, and dataset metadata used by models and the evaluation pipeline.

Overview
--------

The data module provides tools to:

* **Load datasets**: Load training and initialization (or test) data for a given dataset name and pair ID
* **Resolve pair IDs**: Parse ``pair_id`` from config (single ID, list, range, or ``"all"``)
* **Timesteps**: Get training timesteps and prediction timesteps for a pair
* **Validation**: Support validation splits (training/validation timesteps and ``load_validation_dataset``)
* **Config and metadata**: Read dataset YAML config and metadata (e.g. ``delta_t``, matrix shapes)
* **Visualization**: Get which plot types apply to a dataset (``get_applicable_plots``)

Dataset layout is under ``data/{dataset_name}/`` with a ``{dataset_name}.yaml`` config and ``train/`` (and optionally test) data files. See :doc:`datasets` for available datasets.

Usage
-----

Programmatic Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ctf4science.data_module import (
       get_config,
       parse_pair_ids,
       load_dataset,
       get_training_timesteps,
       get_prediction_timesteps,
       get_metadata,
   )

   # Dataset config and which pairs to process
   config = get_config("ODE_Lorenz")
   pair_ids = parse_pair_ids({"name": "ODE_Lorenz", "pair_id": 1})

   # Load train and initialization data for one pair
   train_data, init_data = load_dataset("ODE_Lorenz", pair_id=1)

   # Timesteps for training and for evaluation
   train_timesteps = get_training_timesteps("ODE_Lorenz", 1)
   pred_timesteps = get_prediction_timesteps("ODE_Lorenz", 1, subset="test")

   # Metadata (delta_t, matrix_shapes, etc.)
   metadata = get_metadata("ODE_Lorenz")

Validation Splits
~~~~~~~~~~~~~~~~~

For validation (e.g. hyperparameter tuning), use the validation helpers:

.. code-block:: python

   from ctf4science.data_module import (
       get_validation_training_timesteps,
       get_validation_prediction_timesteps,
       load_validation_dataset,
   )

   val_train_t = get_validation_training_timesteps("ODE_Lorenz", 1, train_split=0.8)
   val_pred_t = get_validation_prediction_timesteps("ODE_Lorenz", 1, train_split=0.8)
   train_data, val_data, init_data = load_validation_dataset("ODE_Lorenz", 1, train_split=0.8)

Timestep Convention
-------------------

All ``get_*_timesteps`` functions return **absolute physical times**, not
zero-based row indices.  The formula is::

    timesteps[i] = (start_index + i) * delta_t,   i = 0, ..., N-1

where:

* ``N`` — number of rows in the matrix (``metadata['matrix_shapes'][name][0]``)
* ``start_index`` — absolute row offset in the underlying trajectory
  (``metadata['matrix_start_index'][name]``)
* ``delta_t`` — physical time step (``metadata['delta_t']``)

So the returned array spans
``[start_index * delta_t, (start_index + N - 1) * delta_t]``.

This means that a test matrix whose ``start_index`` equals the length of the
training matrix will have timesteps that begin exactly where training ends —
the standard forecasting setup.  Reconstruction pairs (pair IDs 2 and 4)
have ``start_index=0`` for both train and test, so their timestep ranges
overlap completely.

For concrete T ranges per dataset and pair, see :doc:`datasets`.

Pair ID Configuration
---------------------

The ``pair_id`` in the dataset section of a run config can be:

* A single integer: process that pair only
* A list of integers: process those pairs
* A range string (e.g. ``"1-3"``): process pairs 1, 2, 3
* ``"all"``: process all pairs defined in the dataset YAML

Use ``parse_pair_ids(dataset_config)`` to resolve this to a ``List[int]``.

Key Functions
-------------

Please refer to :doc:`api` for the full API of the data module. Main entry points:

* **Config and discovery**: ``get_config``, ``get_metadata``, ``parse_pair_ids``, ``get_applicable_plots``
* **Loading**: ``load_dataset``, ``load_validation_dataset``
* **Timesteps**: ``get_training_timesteps``, ``get_prediction_timesteps``, ``get_validation_training_timesteps``, ``get_validation_prediction_timesteps``

Data Formats
------------

The module supports:

* ``.mat`` (MATLAB) files with a single main variable
* ``.npy`` (NumPy) and ``.npz`` files

Training and test file names and structure are defined per pair in the dataset ``{dataset_name}.yaml`` under ``pairs``.

Integration
-----------

The data module is used by:

* **Benchmark and evaluation**: Config and pair resolution; loading test data for metrics
* **Models**: Loading training/initialization data and timesteps
* **Visualization**: ``get_applicable_plots`` and metadata for plot configuration
* **Tune module**: Validation data and timesteps for hyperparameter search
