Datasets
========

This document summarizes the core datasets used in the CTF for Science framework. Each dataset comprises a collection of train/test pairs, each associated with one or more evaluation metrics (E1–E12). For a detailed explanation of these metrics, see :doc:`evaluation_metrics`. For visual outputs associated with each dataset, see :doc:`visualization_module`.

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

----

Each dataset configuration file (e.g., ``ODE_Lorenz.yaml``) includes:

* The full list of train/test matrix files.
* Pair ID mappings to metrics.
* Matrix shapes and time offsets.

To inspect these settings programmatically, see ``ctf4science/data_module.py``. For guidance on configuration format, see :doc:`configuration`.
