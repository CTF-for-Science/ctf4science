"""
Performance monitoring module for CTF models.

This module provides simplified performance monitoring focused on:
- Wall-clock time measurement

Tracks total wall-clock time and calculates averages across multiple trials during hyperparameter tuning.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

import yaml

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Performance monitoring for CTF models. (Wall-clock time only for now)
    Tracks total wall-clock time and calculates averages across multiple runs.
    Energy consumption is measured at the SLURM job level using EAR through bash scripts.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize performance monitor.
        
        Args:
            output_dir (Optional[str]): Directory to save performance results
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results/performance_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track total time and run count
        self.total_time = 0.0
        self.run_count = 0
        self.start_time = None
        
        logger.info(f"Performance monitor initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def start_monitoring(self) -> None:
        """
        Start monitoring a process or session.
        """
        self.start_time = time.time()
        self.total_time = 0.0
        self.run_count = 0
        logger.info("Started performance monitoring")
    
    def record_run(self, run_id: str, duration: float) -> None:
        """
        Record a completed run.
        
        Args:
            run_id (str): Unique identifier for the run
            duration (float): Duration of the run in seconds
            
        Raises:
            ValueError: If duration is negative
        """
        if duration < 0:
            raise ValueError(f"Duration must be non-negative, got {duration}")
            
        self.total_time += duration
        self.run_count += 1
        logger.debug(f"Recorded run {run_id}: {duration:.2f}s")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop monitoring and return summary metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing summary metrics
        """
        if self.start_time is None:
            logger.warning("No monitoring session to stop")
            return {}
        
        total_session_time = time.time() - self.start_time
        average_time_per_run = self.total_time / self.run_count if self.run_count > 0 else 0.0
        
        summary_metrics = {
            'total_num_runs': self.run_count,
            'total_run_time_seconds': self.total_time,
            'total_run_time_hours': self.total_time / 3600,
            'average_time_per_run_seconds': average_time_per_run,
            'average_time_per_run_hours': average_time_per_run / 3600,
            'total_session_time_seconds': total_session_time,
            'total_session_time_hours': total_session_time / 3600,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary results
        self._save_summary_results(summary_metrics)
        
        logger.info(f"Stopped monitoring. Completed {self.run_count} runs")
        logger.info(f"Average time per run: {average_time_per_run:.2f}s")
        
        return summary_metrics
    
    def _save_summary_results(self, metrics: Dict[str, Any]) -> None:
        """Save summary results to file."""
        try:
            filename = f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)
            
            logger.info(f"Saved performance summary to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving performance summary: {e}")


def measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure wall-clock time for any function.
    
    Args:
        func (Callable): Function to measure
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple[Any, float]: (function result, duration in seconds)
        
    Raises:
        Exception: If the function execution raises an exception
    """
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Function {func.__name__} failed after {duration:.2f}s: {e}")
        raise