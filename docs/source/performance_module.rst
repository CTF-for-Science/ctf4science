Performance Module
==================

This document describes the ``performance_module`` of the CTF for Science framework. It provides wall-clock time monitoring for model runs, used during benchmarking and hyperparameter tuning to track and summarize execution time.

Overview
--------

The performance module provides:

* **PerformanceMonitor** (class): Session-based monitoring that:

  * Records start time and resets counters via ``start_monitoring()``
  * Accepts per-run durations via ``record_run(run_id, duration)``
  * Computes and returns summary metrics and writes a timestamped YAML file via ``stop_monitoring()``

* **measure_time** (function): Wraps any callable, runs it once, and returns ``(result, duration_seconds)``. Useful for timing a single run or trial.
* **Output**: Summary YAML files under a configurable directory (default ``results/performance_results``), with metrics such as total runs, total/average time per run, and session duration.

Energy consumption is not measured in this module; it is handled at the SLURM job level using EAR via bash scripts.

Please refer to :doc:`api` for the full API of the performance module.

Usage
-----

Programmatic usage with **PerformanceMonitor** (e.g. inside a benchmark or tuning loop):

.. code-block:: python

   from ctf4science.performance_module import PerformanceMonitor

   monitor = PerformanceMonitor(output_dir="results/my_benchmark/performance")
   monitor.start_monitoring()

   for i in range(num_runs):
       # ... run model ...
       duration = 42.5  # seconds (e.g. from measure_time or similar)
       monitor.record_run(f"run_{i+1}", duration)

   summary = monitor.stop_monitoring()
   # summary has total_num_runs, total_run_time_seconds, average_time_per_run_seconds, etc.
   # A YAML file is also written under output_dir.

Timing a single call with **measure_time**:

.. code-block:: python

   from ctf4science.performance_module import measure_time

   result, duration = measure_time(my_function, arg1, arg2, kw=value)
   print(f"Completed in {duration:.2f}s")

Summary Output
--------------

When ``stop_monitoring()`` is called, the monitor writes a YAML file (e.g. ``performance_summary_YYYYMMDD_HHMMSS.yaml``) to the output directory. The returned (and saved) dictionary includes:

* ``total_num_runs``: Number of runs recorded
* ``total_run_time_seconds`` / ``total_run_time_hours``: Cumulative run time
* ``average_time_per_run_seconds`` / ``average_time_per_run_hours``: Mean duration per run
* ``total_session_time_seconds`` / ``total_session_time_hours``: Wall time since ``start_monitoring()``
* ``timestamp``: ISO format timestamp

Integration
-----------

* **Benchmark module**: Uses ``PerformanceMonitor`` to track time across multiple evaluation runs and optionally record per-run duration.
* **Tune module**: Can enable performance monitoring (``enable_performance_monitoring``) to record average time per trial and write summaries under the tune output directory.

Notes
-----

* If ``stop_monitoring()`` is called without a prior ``start_monitoring()``, it returns an empty dict and does not write a file.
* ``record_run`` raises ``ValueError`` if `duration` is negative.
* ``measure_time`` re-raises any exception raised by the callable; the exception is logged before re-raising.
