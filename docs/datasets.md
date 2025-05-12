# Dataset Overview

This document summarizes the core datasets used in the CTF for Science framework. Each dataset comprises a collection of train/test pairs, each associated with one or more evaluation metrics (E1–E12). For a detailed explanation of these metrics, see [evaluation.md](evaluation.md). For visual outputs associated with each dataset, see [visualization.md](visualization.md).

---

## Dataset Summary Table

| Name             | Type            | Delta t | Spatial Dim | Long-Time Eval       | Visualizations           |
| ---------------- | --------------- | ------- | ----------- | -------------------- | ------------------------ |
| ODE\_Lorenz      | Dynamical       | 0.05    | 3           | histogram\_L2\_error | trajectories, histograms |
| PDE\_KS          | Spatio-temporal | 0.025   | 1024        | spectral\_L2\_error  | psd                      |
| Lorenz\_Official | Dynamical       | 0.05    | 3           | histogram\_L2\_error | trajectories, histograms |
| KS\_Official     | Spatio-temporal | 0.025   | 1024        | spectral\_L2\_error  | psd, 2d\_comparison      |

---

## ODE\_Lorenz

A 3D dynamical system based on the Lorenz attractor. This dataset tests forecasting and reconstruction capabilities across varied noise levels and training regimes.

* **Time step**: 0.05
* **Spatial dimension**: 3
* **Evaluation**: histogram L2 error for long-time metrics
* **Relevant metrics**:

  * ID 1: E1 (short\_time), E2 (long\_time)
  * ID 2: E3 (reconstruction)
  * ID 3: E4 (long\_time)
  * ID 4: E5 (reconstruction)
  * ID 5: E6 (long\_time)
  * ID 6: E7, E8 (short\_time, long\_time)
  * ID 7: E9, E10 (short\_time, long\_time)
  * ID 8: E11 (short\_time)
  * ID 9: E12 (short\_time)
* **Visualizations**: Trajectories, Histograms

## PDE\_KS

A spatio-temporal dataset based on the Kuramoto-Sivashinsky (KS) partial differential equation. It challenges models to learn dynamics over space and time using dense 1024-dimensional spatial grids.

* **Time step**: 0.025
* **Spatial dimension**: 1024
* **Evaluation**: spectral L2 error for long-term behavior (e.g., E2, E8, E10)
* **Relevant metrics**:

  * ID 1: E1 (short\_time), E2 (long\_time)
  * ID 2: E3 (reconstruction)
  * ID 3: E4 (long\_time)
  * ID 4: E5 (reconstruction)
  * ID 5: E6 (long\_time)
  * ID 6: E7, E8 (short\_time, long\_time)
  * ID 7: E9, E10 (short\_time, long\_time)
  * ID 8: E11 (short\_time)
  * ID 9: E12 (short\_time)
* **Visualizations**: Power Spectral Density (PSD)

## Lorenz\_Official

The official Lorenz dataset with longer sequences and standardized splits for benchmarking. The testing data is not included and predictions need to be submitted for scoring on the test set.

* **Time step**: 0.05
* **Spatial dimension**: 3
* **Evaluation**: histogram L2 error for long-time metrics
* **Relevant metrics**:

  * IDs 1–9 map identically to E1–E12
* **Visualizations**: Trajectories, Histograms

## KS\_Official

The official Kuramoto-Sivashinsky dataset designed for rigorous testing of spatio-temporal forecasting and generalization. The testing data is not included in this dataset. Predictions need to be submitted for scoring on the test set.

* **Time step**: 0.025
* **Spatial dimension**: 1024
* **Evaluation**: spectral L2 error for long-term behavior
* **Relevant metrics**:

  * IDs 1–9 map identically to E1–E12
* **Visualizations**: PSD, 2D comparison

---

Each dataset configuration file (e.g., `ODE_Lorenz.yaml`) includes:

* The full list of train/test matrix files.
* Pair ID mappings to metrics.
* Matrix shapes and time offsets.

To inspect these settings programmatically, see `ctf4science/data_module.py`. For guidance on configuration format, see [configuration.md](configuration.md).
