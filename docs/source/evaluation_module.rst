Evaluation Module
====================================

This document explains how the ``eval_module`` of the CTF for Science framework is structured and used. It includes metrics definitions, evaluation routines, and batch results handling.

Overview
--------

The evaluation module provides tools to:

* Compute metrics such as:

  * **Short-time forecast accuracy**
  * **Reconstruction accuracy**
  * **Long-time behavior accuracy** using:

    * Histogram comparisons (for ODEs)
    * Spectral analysis (for PDEs)
* Extract metrics in a consistent order across sub-datasets
* Save results to disk in a reproducible format

Please refer to :doc:`api` for the full API of the evaluation module.

Notes
-----

* Metric scores are always **percentages**, where 100% means perfect match.
* Config files specify which metrics to evaluate per pair:

.. code-block:: yaml

   pairs:
     - id: 1
       metrics: [short_time, reconstruction, long_time]

* ``evaluation_params`` specify parameters like ``k``, ``modes``, ``bins``.

Typical Workflow
----------------

1. Run a model to get ``predictions``.
2. Call ``evaluate(...)`` to compute metrics.
3. Call ``save_results(...)`` to save predictions, config, and scores.
4. Optionally use ``extract_metrics_in_order(...)`` for plotting.

Future Enhancements
-------------------

* Additional metrics (e.g., mean absolute error, KL-divergence)
* Per-variable breakdown of metrics
* Support for multi-output models with task-specific metrics
