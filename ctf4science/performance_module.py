"""
Performance monitoring module for CTF models, measures wall-clock time for model execution.

This module provides simplified performance monitoring focused on:
- Wall-clock time measurement

Tracks total wall-clock time and calculates averages across multiple trials during hyperparameter tuning.
"""

import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    r"""Performance monitoring for CTF models (wall-clock time only).

    Tracks total wall-clock time and calculates averages across multiple
    runs. Energy consumption is measured at the SLURM job level using EAR
    through bash scripts.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save performance results. Defaults to
        ``results/performance_results``.

    Notes
    -----
    **Class Methods:**

    **start_monitoring():**

    - Start monitoring a process or session. Resets total time and run count,
      and records the current time as session start.
    - Returns:
        - None.

    **record_run(self, run_id, duration):**

    - Record a completed run and update cumulative time and run count.
    - Parameters:
        - run_id : str. Unique identifier for the run (e.g. ``"run_1"``).
        - duration : float. Duration of the run in seconds.
    - Returns:
        - None.
    - Raises ``ValueError`` if `duration` is negative.

    **stop_monitoring():**

    - Stop monitoring and return summary metrics. Computes total run time,
      average time per run, and session duration; writes a performance
      summary YAML file to `output_dir`. If no session was started, returns
      an empty dict.
    - Returns:
        - dict. Summary with keys including ``total_num_runs``,
          ``total_run_time_seconds``, ``average_time_per_run_seconds``,
          ``total_session_time_seconds``, ``timestamp``, etc.

    **_save_summary_results(self, metrics):**

    - Save summary results to a YAML file in the output directory.
    - Parameters:
        - metrics : dict. Summary metrics to write (e.g. from `stop_monitoring`).
    - Returns:
        - None. Exceptions during write are logged but not raised.
    """

    def __init__(self, output_dir: str | None = None):
        r"""Initialize the performance monitor and set the output directory."""
        self.output_dir = Path(output_dir) if output_dir else Path("results/performance_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track total time and run count
        self.total_time = 0.0
        self.run_count = 0
        self.start_time = None

        logger.info("Performance monitor initialized")
        logger.info(f"Output directory: {self.output_dir}")

    def start_monitoring(self) -> None:
        r"""Start monitoring a process or session.

        Resets total time and run count, and records the current time as
        session start.
        """
        self.start_time = time.time()
        self.total_time = 0.0
        self.run_count = 0
        logger.info("Started performance monitoring")

    def record_run(self, run_id: str, duration: float) -> None:
        r"""Record a completed run and update cumulative time and run count.

        Parameters
        ----------
        run_id : str
            Unique identifier for the run (e.g. ``"run_1"``).
        duration : float
            Duration of the run in seconds.

        Raises
        ------
        ValueError
            If `duration` is negative.
        """
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")

        self.total_time += duration
        self.run_count += 1
        logger.debug(f"Recorded run {run_id}: {duration:.2f}s")

    def stop_monitoring(self) -> dict[str, Any]:
        r"""Stop monitoring and return summary metrics.

        Computes total run time, average time per run, and session duration;
        writes a performance summary YAML file to `output_dir`. If no session
        was started, returns an empty dict.

        Returns
        -------
        dict
            Summary with keys including ``total_num_runs``,
            ``total_run_time_seconds``, ``average_time_per_run_seconds``,
            ``total_session_time_seconds``, ``timestamp``, etc.
        """
        if self.start_time is None:
            logger.warning("No monitoring session to stop")
            return {}

        total_session_time = time.time() - self.start_time
        average_time_per_run = self.total_time / self.run_count if self.run_count > 0 else 0.0

        summary_metrics = {
            "total_num_runs": self.run_count,
            "total_run_time_seconds": self.total_time,
            "total_run_time_hours": self.total_time / 3600,
            "average_time_per_run_seconds": average_time_per_run,
            "average_time_per_run_hours": average_time_per_run / 3600,
            "total_session_time_seconds": total_session_time,
            "total_session_time_hours": total_session_time / 3600,
            "timestamp": datetime.now().isoformat(),
        }

        # Save summary results
        self._save_summary_results(summary_metrics)

        logger.info(f"Stopped monitoring. Completed {self.run_count} runs")
        logger.info(f"Average time per run: {average_time_per_run:.2f}s")

        return summary_metrics

    def _save_summary_results(self, metrics: dict[str, Any]) -> None:
        r"""Save summary results to a YAML file in the output directory.

        Parameters
        ----------
        metrics : dict
            Summary metrics to write (e.g. from `stop_monitoring`). Written
            as-is to a timestamped file under `output_dir`.

        Notes
        -----
        Exceptions during write are logged but not raised.
        """
        try:
            filename = f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            filepath = self.output_dir / filename

            with open(filepath, "w") as f:
                yaml.dump(metrics, f, default_flow_style=False)

            logger.info(f"Saved performance summary to {filepath}")

        except Exception as e:
            logger.error(f"Error saving performance summary: {e}")


def measure_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    r"""Measure wall-clock time for a single call to a callable.

    Executes `func(*args, **kwargs)` and returns its result together with
    the elapsed time in seconds. Any exception raised by `func` is logged
    and re-raised.

    Parameters
    ----------
    func : callable
        Callable to invoke (e.g. a function or lambda).
    *args : tuple, optional
        Positional arguments passed to `func`.
    **kwargs : dict, optional
        Keyword arguments passed to `func`.

    Returns
    -------
    result : any
        Return value of `func(*args, **kwargs)`.
    duration : float
        Elapsed wall-clock time in seconds.

    Raises
    ------
    Exception
        Re-raised if `func` raises; the exception is logged before re-raising.
    """
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_time
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Function {func.__name__} failed after {duration:.2f}s: {e}")
        raise
    return result, duration
