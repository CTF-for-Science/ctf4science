Performance Module
==================

The Performance Module (``performance_module.py``) provides wall-clock time monitoring for model runs. It is used during benchmarking and hyperparameter tuning to track and summarize execution time.

Overview
--------

The performance module provides tools to:

* **PerformanceMonitor** (class): Session-based monitoring that records start time, accepts per-run durations via ``record_run(run_id, duration)``, and writes a timestamped YAML summary via ``stop_monitoring()``
* **measure_time** (function): Wraps any callable, runs it once, and returns ``(result, duration_seconds)``
* **Output**: Summary YAML files under a configurable directory (default ``results/performance_results``), with total runs, total/average time per run, and session duration

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
       duration = 42.5  # seconds (e.g. from measure_time)
       monitor.record_run(f"run_{i+1}", duration)

   summary = monitor.stop_monitoring()
   # summary has total_num_runs, total_run_time_seconds, average_time_per_run_seconds, etc.
   # A YAML file is also written under output_dir.

Timing a single call with **measure_time**:

.. code-block:: python

   from ctf4science.performance_module import measure_time

   result, duration = measure_time(my_function, arg1, arg2, kw=value)
   print(f"Completed in {duration:.2f}s")

Summary Metrics
---------------

The summary returned by ``stop_monitoring()`` (and written to YAML) includes:

* **total_run_time_seconds**: Sum of all durations passed to ``record_run()`` — i.e. the cumulative run time of all runs.
* **total_session_time_seconds**: Wall-clock time from ``start_monitoring()`` to ``stop_monitoring()`` — i.e. elapsed time of the whole session, including gaps between runs (loading, I/O, etc.).

Integration
-----------

* **Benchmark module**: Uses ``PerformanceMonitor`` to track time across multiple evaluation runs
* **Tune module**: Can enable performance monitoring (``enable_performance_monitoring``) to record average time per trial and write summaries under the tune output directory

Notes
-----

* If ``stop_monitoring()`` is called without a prior ``start_monitoring()``, it returns an empty dict and does not write a file
* ``record_run`` raises ``ValueError`` if `duration` is negative
* ``measure_time`` re-raises any exception raised by the callable (after logging)