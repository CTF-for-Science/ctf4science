# Configuration File Overview

This document goes into detail on the expected configuration file formatting for running and hyperparameter tuning all models.

---

## Run files

Running a model is typically done using the associated `run.py` script. An example of this for the LSTM model is:

```yaml
dataset:
  name: PDE_KS
  pair_ids: [1] # or 1-3, or [1, 2, 3], or empty
model:
  batch_size: 128
  epochs: 50
  gradient_clip_val: 1.0
  hidden_state_size: 30
  log_every_n_steps: 1
  lr: 0.001
  model: lstm
  name: lstm
  seed: 0
  seq_length: 30
  solver: fixed_rk4

```

Notice that there are two top-level fields, the `dataset` and the `model` field. The `dataset` field specifies the associated dataset and pair_ids to train the model on. If `pair_ids` is omitted, it trains the model on _all_ pair_ids. The `model` field contain the hyperparameters for training the model.

## Tuning files

Hyperparameter tuning for a model is typically done using the associated `run_opt.py` script. An example of this for the LSTM model is:

```yaml
dataset:
  name: KS_Official
  pair_id: 1
hyperparameters:
  hidden_state_size:
    lower_bound: 8
    type: randint
    upper_bound: 256
  lr:
    lower_bound: 1.0e-05
    type: loguniform
    upper_bound: 0.01
  seq_length:
    lower_bound: 5
    type: randint
    upper_bound: 512
model:
  batch_size: 128
  epochs: 200
  gradient_clip_val: 1.0
  model: lstm
  n_trials: 100
  name: lstm
  seed: 0
  solver: fixed_rk4
  train_split: 0.8
```

There are three top-level fields: `dataset`, `hyperparameters`, and `model`. The `dataset` field is the same as above, as well as the `model` field. The new field, `hyperparameters`, contains the names of the hyperparameters to optimize as well as the search spaces. The search space information uses language and notation from RayTune's [Search Space API](https://docs.ray.io/en/latest/tune/api/search_space.html#tune-search-space).

More details on hyperparameter optimization can be found in [hyperparameter_optimization.md](hyperparameter_optimization.md).