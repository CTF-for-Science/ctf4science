Datasets
========

This document summarizes the core datasets used in the CTF for Science framework. Each dataset comprises a collection of train/test pairs, each associated with one or more evaluation metrics (E1–E12, or the error metrics of `buildings`_). For a detailed explanation of these metrics, see :doc:`evaluation_metrics`. For visual outputs associated with each dataset, see :doc:`visualization_module`.

Dataset Summary Table
---------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Type
     - Delta t
     - Spatial Dim
     - Long-Time Eval
     - Visualizations
   * - ODE\_Lorenz
     - Dynamical
     - 0.05
     - 3
     - histogram\_L2\_error
     - trajectories, histograms
   * - PDE\_KS
     - Spatio-temporal
     - 0.025
     - 1024
     - spectral\_L2\_error
     - psd, 2d\_comparison
   * - Lorenz\_Official
     - Dynamical
     - 0.05
     - 3
     - histogram\_L2\_error
     - trajectories, histograms
   * - KS\_Official
     - Spatio-temporal
     - 0.025
     - 1024
     - spectral\_L2\_error
     - psd, 2d\_comparison
   * - sst
     - Spatio-temporal
     - 1.0
     - 90601
     - spectral\_L2\_error
     - psd, 2d\_comparison
   * - seismo
     - Spatio-temporal
     - 1.0
     - 2048
     - spectral\_L2\_error
     - psd, 2d\_comparison
   * - ocean\_das
     - Spatio-temporal
     - 1.0
     - 3000
     - spectral\_L2\_error
     - psd, 2d\_comparison
   * - crustal_3d
     - Spatio-temporal
     - 1.0
     - 62451, 26508
     - spectral\_L2\_error
     - psd, 2d\_comparison
   * - buildings
     - Time series (windowed)
     - 900 s
     - 144 target buildings × 6 channels
     - none (mae, rmse, mase)
     - none

ODE\_Lorenz
-----------

A 3D dynamical system based on the Lorenz attractor. This dataset tests forecasting and reconstruction capabilities across varied noise levels and training regimes.

* **Time step** (``delta_t``): 0.05
* **Spatial dimension**: 3
* **Evaluation**: histogram L2 error for long-time metrics
* **Visualizations**: Trajectories, Histograms

Physical times are computed as ``t[i] = (start_index + i) * delta_t``.

.. list-table:: ODE\_Lorenz pairs
   :header-rows: 1
   :widths: 8 30 25 30 25 28

   * - ID
     - Train file(s)
     - Train T range
     - Test file
     - Test T range
     - Metrics
   * - 1
     - X1train [10000 rows]
     - [0.00, 499.95]
     - X1test [1000 rows]
     - [500.00, 549.95]
     - short\_time, long\_time
   * - 2
     - X2train [10000 rows]
     - [0.00, 499.95]
     - X2test [10000 rows]
     - [0.00, 499.95]
     - reconstruction
   * - 3
     - X2train [10000 rows]
     - [0.00, 499.95]
     - X3test [1000 rows]
     - [500.00, 549.95]
     - long\_time
   * - 4
     - X3train [10000 rows]
     - [0.00, 499.95]
     - X4test [10000 rows]
     - [0.00, 499.95]
     - reconstruction
   * - 5
     - X3train [10000 rows]
     - [0.00, 499.95]
     - X5test [1000 rows]
     - [500.00, 549.95]
     - long\_time
   * - 6
     - X4train [100 rows]
     - [0.00, 4.95]
     - X6test [1000 rows]
     - [5.00, 54.95]
     - short\_time, long\_time
   * - 7
     - X5train [100 rows]
     - [0.00, 4.95]
     - X7test [1000 rows]
     - [5.00, 54.95]
     - short\_time, long\_time
   * - 8
     - X6/X7/X8train [3×10000 rows]; init X9train [100 rows, t=[495.00, 499.95]]
     - [0.00, 499.95]
     - X8test [1000 rows]
     - [500.00, 549.95]
     - short\_time
   * - 9
     - X6/X7/X8train [3×10000 rows]; init X10train [100 rows, t=[495.00, 499.95]]
     - [0.00, 499.95]
     - X9test [1000 rows]
     - [500.00, 549.95]
     - short\_time

PDE\_KS
-------

A spatio-temporal dataset based on the Kuramoto-Sivashinsky (KS) partial differential equation. It challenges models to learn dynamics over space and time using dense 1024-dimensional spatial grids.

* **Time step** (``delta_t``): 0.025
* **Spatial dimension**: 1024
* **Evaluation**: spectral L2 error for long-term behavior
* **Visualizations**: Power Spectral Density (PSD)

Physical times are computed as ``t[i] = (start_index + i) * delta_t``.

.. list-table:: PDE\_KS pairs
   :header-rows: 1
   :widths: 8 30 25 30 25 28

   * - ID
     - Train file(s)
     - Train T range
     - Test file
     - Test T range
     - Metrics
   * - 1
     - X1train [10000 rows]
     - [0.000, 249.975]
     - X1test [1000 rows]
     - [250.000, 274.975]
     - short\_time, long\_time
   * - 2
     - X2train [10000 rows]
     - [0.000, 249.975]
     - X2test [10000 rows]
     - [0.000, 249.975]
     - reconstruction
   * - 3
     - X2train [10000 rows]
     - [0.000, 249.975]
     - X3test [1000 rows]
     - [250.000, 274.975]
     - long\_time
   * - 4
     - X3train [10000 rows]
     - [0.000, 249.975]
     - X4test [10000 rows]
     - [0.000, 249.975]
     - reconstruction
   * - 5
     - X3train [10000 rows]
     - [0.000, 249.975]
     - X5test [1000 rows]
     - [250.000, 274.975]
     - long\_time
   * - 6
     - X4train [100 rows]
     - [0.000, 2.475]
     - X6test [1000 rows]
     - [2.500, 27.475]
     - short\_time, long\_time
   * - 7
     - X5train [100 rows]
     - [0.000, 2.475]
     - X7test [1000 rows]
     - [2.500, 27.475]
     - short\_time, long\_time
   * - 8
     - X6/X7/X8train [3×10000 rows]; init X9train [100 rows, t=[247.500, 249.975]]
     - [0.000, 249.975]
     - X8test [1000 rows]
     - [250.000, 274.975]
     - short\_time
   * - 9
     - X6/X7/X8train [3×10000 rows]; init X10train [100 rows, t=[247.500, 249.975]]
     - [0.000, 249.975]
     - X9test [1000 rows]
     - [250.000, 274.975]
     - short\_time

Lorenz\_Official
----------------

The official Lorenz dataset with longer sequences and standardized splits for benchmarking. The testing data is not included and predictions need to be submitted for scoring on the test set.

* **Time step** (``delta_t``): 0.05
* **Spatial dimension**: 3
* **Evaluation**: histogram L2 error for long-time metrics
* **Visualizations**: Trajectories, Histograms

Pair structure, matrix sizes, and T ranges are identical to `ODE\_Lorenz`_ above.

KS\_Official
------------

The official Kuramoto-Sivashinsky dataset designed for rigorous testing of spatio-temporal forecasting and generalization. The testing data is not included in this dataset. Predictions need to be submitted for scoring on the test set.

* **Time step** (``delta_t``): 0.025
* **Spatial dimension**: 1024
* **Evaluation**: spectral L2 error for long-term behavior
* **Visualizations**: PSD, 2D comparison

Pair structure, matrix sizes, and T ranges are identical to `PDE\_KS`_ above.

seismo
------

A spatio-temporal dataset of synthetic seismic waveforms generated using the Instaseis library. This dataset challenges models to learn complex wave propagation dynamics across 2048 virtual seismometer stations, testing forecasting capabilities for earthquake-induced ground motion patterns.

* **Time step** (``delta_t``): 1.0
* **Spatial dimension**: 2048 sensors
* **Evaluation**: spectral L2 error for long-term behavior
* **Data characteristics**:
  * Velocity seismograms (m/s) in vertical (Z) component
  * Synthetic earthquakes with randomized magnitude, location, etc.
  * Normalized for each earthquake event

Physical times are computed as ``t[i] = (start_index + i) * delta_t``.

.. list-table:: seismo pairs
   :header-rows: 1
   :widths: 8 30 22 30 22 28

   * - ID
     - Train file(s)
     - Train T range
     - Test file
     - Test T range
     - Metrics
   * - 1
     - X1train [2000 rows]
     - [0, 1999]
     - X1test [1000 rows]
     - [2000, 2999]
     - short\_time, long\_time
   * - 2
     - X2train [2000 rows]
     - [0, 1999]
     - X2test [2000 rows]
     - [0, 1999]
     - reconstruction
   * - 3
     - X2train [2000 rows]
     - [0, 1999]
     - X3test [1000 rows]
     - [2000, 2999]
     - long\_time
   * - 4
     - X3train [2000 rows]
     - [0, 1999]
     - X4test [2000 rows]
     - [0, 1999]
     - reconstruction
   * - 5
     - X3train [2000 rows]
     - [0, 1999]
     - X5test [1000 rows]
     - [2000, 2999]
     - long\_time
   * - 6
     - X4train [500 rows]
     - [1500, 1999]
     - X6test [1000 rows]
     - [2000, 2999]
     - short\_time, long\_time
   * - 7
     - X5train [500 rows]
     - [1500, 1999]
     - X7test [1000 rows]
     - [2000, 2999]
     - short\_time, long\_time
   * - 8
     - X6/X7/X8train [3×2000 rows]; init X9train [500 rows, t=[1500, 1999]]
     - [0, 1999]
     - X8test [1000 rows]
     - [2000, 2999]
     - short\_time
   * - 9
     - X6/X7/X8train [3×2000 rows]; init X10train [500 rows, t=[1500, 1999]]
     - [0, 1999]
     - X9test [1000 rows]
     - [2000, 2999]
     - short\_time

ocean_das
---------

A spatio-temporal dataset using a novel geophysical sensing technology called DAS (Distributed Acoustic Sensing). This dataset is comprised of data from a shallow offshore DAS about 30m below sea level where surface gravity waves are particularly dispersive. The data is sampled at 5Hz but low-pass filtered to 1Hz.

* **Time step** (``delta_t``): 1.0
* **Spatial dimension**: 3000 sensors
* **Evaluation**: spectral L2 error for long-term behavior
* **Data characteristics**:
  * Real-world sensor measurements measuring acoustic frequency strain signals.

Pair structure, matrix sizes, and T ranges are identical to `seismo`_ above (with spatial dimension 3000 instead of 2048).

crustal_3d
----------

A spatio-temporal from synthetic 3D seismic wavefields in a heterogeneous 3D crustal model. Each simulation yields three-component velocity seismograms on a :math:`32\times32\times32` heterogeneous grid. Virtual sensors form a :math:`94\times94` grid arranged on top of the model volume with 100m spacing. These seismograms are sampled for 6 seconds at 50Hz.

* **Time step** (``delta_t``): 1.0
* **Spatial dimension**: 62451 (pairs 1–7) or 26508 (pairs 8–9)
* **Evaluation**: spectral L2 error for long-term behavior
* **Data characteristics**:
  * For tasks :math:`E_{1}`-:math:`E_{10}`, the velocity seismograms, virtual sensors, and point sources are provided, yielding 62451 data points per timestep. For tasks :math:`E_{11}`-:math:`E_{12}` only the velocity seismograms are provided, yielding 26508 data points per timestep.

Physical times are computed as ``t[i] = (start_index + i) * delta_t``.

.. list-table:: crustal\_3d pairs
   :header-rows: 1
   :widths: 8 30 22 30 22 28

   * - ID
     - Train file(s)
     - Train T range
     - Test file
     - Test T range
     - Metrics
   * - 1
     - X1train [500 rows, 62451 dim]
     - [0, 499]
     - X1test [100 rows]
     - [500, 599]
     - short\_time, long\_time
   * - 2
     - X2train [500 rows, 62451 dim]
     - [0, 499]
     - X2test [500 rows]
     - [0, 499]
     - reconstruction
   * - 3
     - X2train [500 rows, 62451 dim]
     - [0, 499]
     - X3test [100 rows]
     - [500, 599]
     - long\_time
   * - 4
     - X3train [500 rows, 62451 dim]
     - [0, 499]
     - X4test [500 rows]
     - [0, 499]
     - reconstruction
   * - 5
     - X3train [500 rows, 62451 dim]
     - [0, 499]
     - X5test [100 rows]
     - [500, 599]
     - long\_time
   * - 6
     - X4train [200 rows, 62451 dim]
     - [300, 499]
     - X6test [100 rows]
     - [500, 599]
     - short\_time, long\_time
   * - 7
     - X5train [200 rows, 62451 dim]
     - [300, 499]
     - X7test [100 rows]
     - [500, 599]
     - short\_time, long\_time
   * - 8
     - X6/X7/X8train [3×500 rows, 26508 dim]; init X9train [200 rows, t=[300, 499]]
     - [0, 499]
     - X8test [100 rows]
     - [500, 599]
     - short\_time
   * - 9
     - X6/X7/X8train [3×500 rows, 26508 dim]; init X10train [200 rows, t=[300, 499]]
     - [0, 499]
     - X9test [100 rows]
     - [500, 599]
     - short\_time

SST
---

This dataset contains Global Sea Surface Temperature (SST) data from NASA's Group for High Resolution Sea Surface Temperature (GHRSST) product. SST data exhibits complex, multiscale features of turbulent flows with intermittent events and quasi-periodic behavior, making it a challenging benchmark for forecasting, reconstruction, and prediction tasks. Unlike the synthetic KS and Lorenz datasets, this represents real-world geophysical observations, providing a critical testbed for evaluating data-driven methods on actual scientific data.

* **Time step** (``delta_t``): 1.0
* **Spatial dimension**: 90601
* **Evaluation**: spectral L2 error for long-term behavior
* **Data characteristics**:
  * Real-world physical data

Physical times are computed as ``t[i] = (start_index + i) * delta_t``.

.. list-table:: sst / sst\_kaggle pairs
   :header-rows: 1
   :widths: 8 30 22 30 22 28

   * - ID
     - Train file(s)
     - Train T range
     - Test file
     - Test T range
     - Metrics
   * - 1
     - X1train [800 rows]
     - [0, 799]
     - X1test [400 rows]
     - [800, 1199]
     - short\_time, long\_time
   * - 2
     - X2train [800 rows]
     - [0, 799]
     - X2test [800 rows]
     - [0, 799]
     - reconstruction
   * - 3
     - X2train [800 rows]
     - [0, 799]
     - X3test [400 rows]
     - [800, 1199]
     - long\_time
   * - 4
     - X3train [800 rows]
     - [0, 799]
     - X4test [800 rows]
     - [0, 799]
     - reconstruction
   * - 5
     - X3train [800 rows]
     - [0, 799]
     - X5test [400 rows]
     - [800, 1199]
     - long\_time
   * - 6
     - X4train [100 rows]
     - [0, 99]
     - X6test [400 rows]
     - [100, 499]
     - short\_time, long\_time
   * - 7
     - X5train [100 rows]
     - [0, 99]
     - X7test [400 rows]
     - [100, 499]
     - short\_time, long\_time
   * - 8
     - X6/X7/X8train [3×800 rows]; init X9train [100 rows, t=[700, 799]]
     - [0, 799]
     - X8test [400 rows]
     - [800, 1199]
     - short\_time
   * - 9
     - X6/X7/X8train [3×800 rows]; init X10train [100 rows, t=[700, 799]]
     - [0, 799]
     - X9test [400 rows]
     - [800, 1199]
     - short\_time

buildings
---------

Thermal behaviour of buildings: weather, heating power and indoor temperature at 15-minute steps. Models are trained on *source* buildings and forecast the indoor temperature of 144 *target* buildings in their second year. Unlike the other datasets, every array is ``(time, building, channel)`` and a pair is scored on many short forecast windows rather than one long trajectory.

* **Time step** (``delta_t``): 900 s (15 minutes); one day is 96 steps.
* **Times**: ``get_training_timesteps`` and ``get_prediction_timesteps`` return the CSV ``time`` column: year 1 starts at 900 s and year 2 at 31,536,000 s (exactly 365 days).
* **Buildings**: 144 targets, in a PI and an RW variant with the same building order, and the source buildings (one column per source file).
* **Channels** (axis 2, float32): 0 diffuse solar, 1 direct solar, 2 outdoor temperature, 3 heating power, 4 indoor temperature (the forecast target), 5 ``u`` (heating power is ``u`` times a per-building constant, so covariates use ``u``).
* **Evaluation**: ``mae``, ``rmse`` and ``mase`` on the indoor temperature, lower is better (see :doc:`evaluation_metrics`).
* **Visualizations**: none. The plot functions expect 2D ``(T, F)`` arrays, so ``get_applicable_plots('buildings')`` returns an empty list.

Files
~~~~~

Every time series is stored once, as a continuous array; the forecast windows are cut at load time.

.. list-table:: buildings files
   :header-rows: 1
   :widths: 40 25 35

   * - File
     - Shape
     - Content
   * - ``train/sources.npz``
     - (35039, S, 6)
     - Source buildings, year 1
   * - ``train/targets_{PI,RW}_octdec.npz``
     - (8832, 144, 6)
     - Target buildings, Oct–Dec of year 1 (forecasting pairs only)
   * - ``test/targets_{PI,RW}_year2.npz``
     - (35040, 144, 6)
     - Target buildings, year 2: model input (lookbacks) and ground truth
   * - ``*_noisy_weather.npz``
     - (rows, buildings, 3)
     - Noisy copies of the three weather channels of each file above, for the G3 pairs
   * - ``building_ids.txt``
     - —
     - Tab-separated ``set``, ``column``, ``building_id``: the building of each column of ``sources``, ``targets_PI`` and ``targets_RW``

``S`` is the number of source files. When a source building comes in several excitation variants, the variants share one building ID.

Pairs
~~~~~

There are 12 pairs, one per task. *Generalization* pairs train on the source buildings only; *forecasting* pairs also train on the targets' own Oct–Dec data. The horizon and covariates set the difficulty: G1 forecasts one step ahead with no future inputs, G2 forecasts a day ahead with the exact future weather and ``u``, and G3 forecasts a day ahead with noisy future weather and the exact ``u``.

.. list-table:: buildings pairs
   :header-rows: 1
   :widths: 8 16 10 30 10 10 16

   * - ID
     - Task
     - Group
     - Train file(s)
     - Horizon
     - Stride
     - Covariates
   * - 1 / 2
     - Generalization
     - G1 PI / RW
     - sources
     - 1
     - 1
     - none
   * - 3 / 4
     - Generalization
     - G2 PI / RW
     - sources
     - 96
     - 96
     - exact
   * - 5 / 6
     - Generalization
     - G3 PI / RW
     - sources
     - 96
     - 96
     - noisy
   * - 7 / 8
     - Forecasting
     - G1 PI / RW
     - sources, targets\_{PI,RW}\_octdec
     - 1
     - 1
     - none
   * - 9 / 10
     - Forecasting
     - G2 PI / RW
     - sources, targets\_{PI,RW}\_octdec
     - 96
     - 96
     - exact
   * - 11 / 12
     - Forecasting
     - G3 PI / RW
     - sources, targets\_{PI,RW}\_octdec
     - 96
     - 96
     - noisy

Every pair is tested on ``targets_{PI,RW}_year2.npz``. Use ``pair_id: "1-6"`` to run all generalization pairs.

Forecast windows and the stride rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API cuts the year-2 series into windows. Window ``i`` has:

* **lookback**: rows ``[i * stride, i * stride + lookback)``, all 6 channels (``lookback`` = 96, one day);
* **target**: the next ``horizon`` rows of indoor temperature;
* **covariates**: the same ``horizon`` rows of channels 0, 1, 2 and 5 (G2 and G3 only). For G3 the weather channels come from the noisy file and ``u`` from the clean one; the noise is stored once per time step, so a step has the same noisy value in every window.

The stride is a dataset setting, not a model setting, so every model is scored on the same windows. It is ``evaluation_params.stride`` (96: one forecast per day, starting at midnight on the test set) unless a pair sets its own ``stride``. The four G1 pairs set ``stride: 1``: with a one-step horizon and a stride of 96, every scored step would be 00:00, so they instead score a one-step forecast from every possible lookback. Only windows that end inside the series are kept, so year 2 gives 34,944 windows per building at stride 1 and 364 at stride 96.

Models never cut windows themselves:

.. code-block:: python

   from ctf4science.data_module import load_dataset, load_forecast_windows
   from ctf4science.eval_module import evaluate

   train_data, _ = load_dataset("buildings", pair_id)
   lookback, covariates = load_forecast_windows("buildings", pair_id)
   #   lookback   (W, 96, 144, 6), a view into the year-2 array
   #   covariates (W, horizon, 144, 4), or None for G1
   prediction = model.predict(lookback, covariates)  # (W, horizon, 144)
   results = evaluate("buildings", pair_id, prediction)

``lookback`` is not copied. For the G1 pairs it is nominally ``(34944, 96, 144, 6)``, about 11.6 GB as float32 if copied in one go (for example by moving it to a GPU), so predict over ``W`` in batches.

Both ``lookback`` and ``covariates`` are read-only; copy a batch before modifying it in place. Predict each window from its own lookback and covariates only, and fit nothing on these arrays: adjacent windows overlap, so another window's lookback can contain this window's answer.

For hyperparameter tuning, ``load_validation_forecast_windows("buildings", pair_id, n_holdout, n_val_rows=2688, seed=0)`` returns ``train_data, lookback, covariates, truth`` without touching the test set. Generalization pairs hold out ``n_holdout`` source building IDs (all variants of a building together) and window their whole year; forecasting pairs keep all sources and hold out the last ``n_val_rows`` rows (28 days) of the targets' Oct–Dec data. ``truth`` is the continuous held-out series; pass it to ``evaluate_custom``, which cuts the truth windows itself. The same two rules apply to the validation windows. Generalization-pair validation windows start at 00:15, not midnight, because the source series start at 900 s.

The test set is not hidden
~~~~~~~~~~~~~~~~~~~~~~~~~~

The year-2 files are both the model input (the lookbacks) and the ground truth, so ``test/`` ships with the dataset and ``buildings`` has no hidden leaderboard. This follows from storing each series once and allowing any stride. Models should read the test set only through ``load_forecast_windows`` and must not train on it.

Download
~~~~~~~~

``data/`` is not tracked in git. Download ``buildings.tar.gz`` from the `OSF page <https://osf.io/6rzhm/overview>`_, extract it with ``tar -xzvf buildings.tar.gz`` and place the ``buildings/`` directory in the top-level ``data/`` directory. The tarball is built from the authors' CSV files by ``scripts/convert_buildings.py``.

----

Each dataset configuration file (e.g., ``ODE_Lorenz.yaml``) includes:

* The full list of train/test matrix files.
* Pair ID mappings to metrics.
* Matrix shapes and time offsets.

To inspect these settings programmatically, see ``ctf4science/data_module.py``. For guidance on configuration format, see :doc:`configuration`.
