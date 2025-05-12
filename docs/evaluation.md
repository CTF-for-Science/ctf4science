# Evaluation Metrics

The Common Task Framework (CTF) for Science employs a standardized suite of 12 evaluation metrics (E1–E12) to assess model performance across various data regimes and prediction tasks. All metrics return a score between 0 and 100, where:

* 100 indicates a perfect match to the ground truth,
* 0 corresponds to predicting all zeros, and
* Negative values indicate performance worse than the zero baseline.

Note: Not all dataset get their scores to correspond to the `[-100, 100]` range. The Lorenz dataset for example can produce large negative scores.

This document outlines each metric's purpose, method of evaluation, and its corresponding dataset pair.

---
## Summary Table

| Metric | Name                                  | Task Scenario          | Input Noise | Data Regime | Forecast Type | Dataset Pair ID |
| ------ | ------------------------------------- | ---------------------- | ----------- | ----------- | ------------- | --------------- |
| E1     | Short-Time Forecast Accuracy          | Baseline               | None        | Full        | Short-term    | 1               |
| E2     | Long-Time Forecast Accuracy           | Baseline               | None        | Full        | Long-term     | 1               |
| E3     | Reconstruction (Medium Noise)         | Medium-noise denoising | Medium      | Full        | N/A           | 2               |
| E4     | Short-Time Forecast (Medium Noise)    | Forecast from noise    | Medium      | Full        | Short-term    | 3               |
| E5     | Reconstruction (High Noise)           | High-noise denoising   | High        | Full        | N/A           | 4               |
| E6     | Short-Time Forecast (High Noise)      | Forecast from noise    | High        | Full        | Short-term    | 5               |
| E7     | Short-Time Forecast (Low Data, Clean) | Few-shot clean         | None        | Sparse      | Short-term    | 6               |
| E8     | Long-Time Forecast (Low Data, Clean)  | Few-shot clean         | None        | Sparse      | Long-term     | 6               |
| E9     | Short-Time Forecast (Low Data, Noisy) | Few-shot noisy         | Medium/High | Sparse      | Short-term    | 7               |
| E10    | Long-Time Forecast (Low Data, Noisy)  | Few-shot noisy         | Medium/High | Sparse      | Long-term     | 7               |
| E11    | Parametric Gen. (Interpolation)       | Interpolation across λ | None        | Full        | Short-term    | 8               |
| E12    | Parametric Gen. (Extrapolation)       | Extrapolation beyond λ | None        | Full        | Short-term    | 9               |


---

## Metric Descriptions

### **E1 – Short-Time Forecast Accuracy**

* **Pair**: ID 1
* **Measures**: Accuracy over initial prediction steps.
* **How**: Computes the root-mean-square error over the first *k* time steps between forecast and truth.

### **E2 – Long-Time Forecast Accuracy**

* **Pair**: ID 1
* **Measures**: Fidelity of long-term behavior via statistics.
* **How**: L2 distance between log power spectra (log-PSD) of forecast and truth over dominant modes.

### **E3 – Reconstruction (Medium Noise)**

* **Pair**: ID 2
* **Measures**: Ability to reconstruct clean signals from moderately noisy data.
* **How**: L2 error between denoised output and noise-free reference.

### **E4 – Short-Time Forecast (Medium Noise)**

* **Pair**: ID 3
* **Measures**: Short-term accuracy when initialized from noisy input.
* **How**: Same as E1 but starting from medium-noise initial conditions.

### **E5 – Reconstruction (High Noise)**

* **Pair**: ID 4
* **Measures**: Denoising capability under high noise conditions.
* **How**: Same as E3, but on data with stronger degradation.

### **E6 – Short-Time Forecast (High Noise)**

* **Pair**: ID 5
* **Measures**: Forecasting skill with severely noisy initializations.
* **How**: Same as E1, but with high-noise input data.

### **E7 – Short-Time Forecast (Low Data, Clean)**

* **Pair**: ID 6
* **Measures**: Forecasting accuracy from small clean datasets.
* **How**: Same as E1 with training on just 51 time steps.

### **E8 – Long-Time Forecast (Low Data, Clean)**

* **Pair**: ID 6
* **Measures**: Long-time accuracy from limited clean data.
* **How**: Same as E2.

### **E9 – Short-Time Forecast (Low Data, Noisy)**

* **Pair**: ID 7
* **Measures**: Short-term forecasting from short, noisy input.
* **How**: Same as E1.

### **E10 – Long-Time Forecast (Low Data, Noisy)**

* **Pair**: ID 7
* **Measures**: Long-range statistical alignment under low data + noise.
* **How**: Same as E2.

### **E11 – Parametric Generalization (Interpolation)**

* **Pair**: ID 8
* **Measures**: Predictive generalization to interpolated physical parameters.
* **How**: Forecast accuracy in unseen but interpolated parametric regime.

### **E12 – Parametric Generalization (Extrapolation)**

* **Pair**: ID 9
* **Measures**: Generalization to extrapolated dynamics.
* **How**: Forecast skill in unseen extrapolated physical regimes.

---

For implementation details of how each metric is computed, see the source in `eval_module.py`. The evaluation logic automatically selects the appropriate metric per dataset using the `metrics` list specified in each dataset YAML configuration.
