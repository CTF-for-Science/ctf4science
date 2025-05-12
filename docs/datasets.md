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
| seismo           | Spatio-temporal | 1.0     | 2048        | spectral\_L2\_error  | psd, 2d\_comparison      |

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

## seismo

A spatio-temporal dataset of synthetic seismic waveforms generated using the Instaseis library. This dataset challenges models to learn complex wave propagation dynamics across 2048 virtual seismometer stations, testing forecasting capabilities for earthquake-induced ground motion patterns.

* **Time step**: 1.0
* **Spatial dimension**: 2048
* **Evaluation**: spectral L2 error for long-term behavior
* **Data characteristics**:
  * Velocity seismograms (m/s) in vertical (Z) component
  * Synthetic earthquakes with randomized magnitude, location, etc.
  * Normalized for each earthquake event
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

## SST

This dataset contains Global Sea Surface Temperature (SST) data from NASA's Group for High Resolution Sea Surface Temperature (GHRSST) product. SST data exhibits complex, multiscale features of turbulent flows with intermittent events and quasi-periodic behavior, making it a challenging benchmark for forecasting, reconstruction, and prediction tasks. Unlike the synthetic KS and Lorenz datasets, this represents real-world geophysical observations, providing a critical testbed for evaluating data-driven methods on actual scientific data.

* **Time step**: 1.0
* **Spatial dimension**: 90601
* **Evaluation**: spectral L2 error for long-term behavior
* **Data characteristics**:
  * Real-world physical data
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

---

Each dataset configuration file (e.g., `ODE_Lorenz.yaml`) includes:

* The full list of train/test matrix files.
* Pair ID mappings to metrics.
* Matrix shapes and time offsets.

To inspect these settings programmatically, see `ctf4science/data_module.py`. For guidance on configuration format, see [configuration.md](configuration.md).
