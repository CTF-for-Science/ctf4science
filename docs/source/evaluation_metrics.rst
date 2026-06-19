Evaluation Metrics
===================

The Common Task Framework (CTF) for Science employs a standardized suite of 12 evaluation metrics (E1–E12) to assess model performance across various data regimes and prediction tasks. All metrics return a score between 0 and 100, where:

* 100 indicates a perfect match to the ground truth,
* 0 corresponds to predicting all zeros, and
* Negative values indicate performance worse than the zero baseline.

Note: Not all dataset get their scores to correspond to the ``[-100, 100]`` range. The Lorenz dataset for example can produce large negative scores.

This document outlines each metric's purpose, method of evaluation, and its corresponding dataset pair.


Dataset Pairs and Files
-----------------------

.. list-table:: Files and corresponding evaluation metrics (E\ :sub:`1`–E\ :sub:`12`) for benchmark datasets.
   :header-rows: 1

   * - Score
     - Pair ID
     - Test
     - Task
     - Train / Burn-in File(s)
     - Ground Truth File
   * - E\ :sub:`1`
     - 1
     - Forecasting
     - Short-time
     - :math:`\mathbf{X}_{1\text{train}}`
     - :math:`\mathbf{X}_{1\text{test}}`
   * - E\ :sub:`2`
     - 1
     - Forecasting
     - Long-time
     - :math:`\mathbf{X}_{1\text{train}}`
     - :math:`\mathbf{X}_{1\text{test}}`
   * - E\ :sub:`3`
     - 2
     - Noisy (medium)
     - Reconstruction (denoising)
     - :math:`\mathbf{X}_{2\text{train}}`
     - :math:`\mathbf{X}_{2\text{test}}`
   * - E\ :sub:`4`
     - 3
     - Noisy (medium)
     - Forecast (long-time)
     - :math:`\mathbf{X}_{2\text{train}}`
     - :math:`\mathbf{X}_{3\text{test}}`
   * - E\ :sub:`5`
     - 4
     - Noisy (high)
     - Reconstruction (denoising)
     - :math:`\mathbf{X}_{3\text{train}}`
     - :math:`\mathbf{X}_{4\text{test}}`
   * - E\ :sub:`6`
     - 5
     - Noisy (high)
     - Forecast (long-time)
     - :math:`\mathbf{X}_{3\text{train}}`
     - :math:`\mathbf{X}_{5\text{test}}`
   * - E\ :sub:`7`
     - 6
     - Limited Data (clean)
     - Forecast (short-time)
     - :math:`\mathbf{X}_{4\text{train}}`
     - :math:`\mathbf{X}_{6\text{test}}`
   * - E\ :sub:`8`
     - 6
     - Limited Data (clean)
     - Forecast (long-time)
     - :math:`\mathbf{X}_{4\text{train}}`
     - :math:`\mathbf{X}_{6\text{test}}`
   * - E\ :sub:`9`
     - 7
     - Limited Data (noisy)
     - Forecast (short-time)
     - :math:`\mathbf{X}_{5\text{train}}`
     - :math:`\mathbf{X}_{7\text{test}}`
   * - E\ :sub:`10`
     - 7
     - Limited Data (noisy)
     - Forecast (long-time)
     - :math:`\mathbf{X}_{5\text{train}}`
     - :math:`\mathbf{X}_{7\text{test}}`
   * - E\ :sub:`11`
     - 8
     - Parametric Generalization
     - Interpolation forecast
     - :math:`\mathbf{X}_{6,7,8\text{train}}` / :math:`\mathbf{X}_{9\text{train}}`
     - :math:`\mathbf{X}_{8\text{test}}`
   * - E\ :sub:`12`
     - 9
     - Parametric Generalization
     - Extrapolation forecast
     - :math:`\mathbf{X}_{6,7,8\text{train}}` / :math:`\mathbf{X}_{10\text{train}}`
     - :math:`\mathbf{X}_{9\text{test}}`

Metric Descriptions
-------------------

**E1 – Short-Time Forecast Accuracy**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 1
* **Measures**: Accuracy over initial prediction steps.
* **How**: Computes the relative L2 error over the first *k* time steps between forecast and truth.

**E2 – Long-Time Forecast Accuracy**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 1
* **Measures**: Fidelity of long-term behavior via statistics.
* **How**: For spatio-temporal datasets, computes the relative L2 distance between averaged power spectra (PSD) of forecast and truth over dominant Fourier modes. For dynamical datasets, computes the relative L1 distance between histograms of forecast and truth over each variable.

**E3 – Reconstruction (Medium Noise)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 2
* **Measures**: Ability to reconstruct clean signals from moderately noisy data.
* **How**: Relative L2 error between denoised output and noise-free reference.

**E4 – Long-Time Forecast (Medium Noise)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 3
* **Measures**: Long-time statistical accuracy when forecasting from medium-noise initial conditions.
* **How**: Same as E2 but starting from medium-noise initial conditions.

**E5 – Reconstruction (High Noise)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 4
* **Measures**: Denoising capability under high noise conditions.
* **How**: Same as E3, but on data with stronger degradation.

**E6 – Long-Time Forecast (High Noise)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 5
* **Measures**: Long-time statistical accuracy when forecasting from high-noise initial conditions.
* **How**: Same as E2, but with high-noise input data.

**E7 – Short-Time Forecast (Low Data, Clean)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 6
* **Measures**: Forecasting accuracy from small clean datasets.
* **How**: Same as E1.

**E8 – Long-Time Forecast (Low Data, Clean)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 6
* **Measures**: Long-time accuracy from limited clean data.
* **How**: Same as E2.

**E9 – Short-Time Forecast (Low Data, Noisy)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 7
* **Measures**: Short-term forecasting from short, noisy input.
* **How**: Same as E1.

**E10 – Long-Time Forecast (Low Data, Noisy)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 7
* **Measures**: Long-range statistical alignment under low data + noise.
* **How**: Same as E2.

**E11 – Parametric Generalization (Interpolation)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 8
* **Measures**: Predictive generalization to interpolated physical parameters.
* **How**: Forecast accuracy in unseen but interpolated parametric regime.

**E12 – Parametric Generalization (Extrapolation)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Pair**: ID 9
* **Measures**: Generalization to extrapolated dynamics.
* **How**: Forecast skill in unseen extrapolated physical regimes.

----


For implementation details of how each metric is computed, see the source in ``eval_module.py``. The evaluation logic automatically selects the appropriate metric per dataset using the ``metrics`` list specified in each dataset YAML configuration.
