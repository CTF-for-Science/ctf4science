import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Optional, Dict

import importlib.resources as pkg_resources

from ctf4science.data_module import get_applicable_plots, _load_test_data
from ctf4science.eval_module import compute_log_psd

file_dir = Path(__file__).parent
top_dir = Path(__file__).parent.parent

class Visualization:
    """
    A class for generating visualizations of model predictions and evaluation metrics.
    
    Attributes:
        config (dict): Configuration settings for visualizations loaded from a YAML file.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize with a configuration file.
        
        Args:
            config_path (Optional[str]): Path to a custom YAML config file. If None, uses default.
        
        Raises:
            FileNotFoundError: If the custom config file does not exist.
        """
        if config_path is None:
            with pkg_resources.open_text('ctf4science.config', 'default_visualization_config.yaml') as f:
                self.config = yaml.safe_load(f)
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
    
    def plot_trajectories(self, truth: np.ndarray, predictions: List[np.ndarray], 
                          labels: Optional[List[str]] = None, **kwargs) -> plt.Figure:
        """
        Plot stacked trajectories for each variable, comparing truth and predictions.
        
        Args:
            truth (np.ndarray): Ground truth data, shape (time_steps, variables).
            predictions (List[np.ndarray]): List of prediction arrays, each matching truth's shape.
            labels (Optional[List[str]]): Labels for each prediction.
            **kwargs: Custom configurations (e.g., figure_size).
        
        Returns:
            plt.Figure: The generated figure object.
        
        Raises:
            ValueError: If truth and predictions have incompatible shapes.
        """
        config = {**self.config, **kwargs}
        time_steps, num_vars = truth.shape
        for pred in predictions:
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")
        
        labels = labels if labels else [f'Prediction {i+1}' for i in range(len(predictions))]
        
        fig, axes = plt.subplots(num_vars, 1, figsize=config.get('figure_size', (10, 2 * num_vars)), sharex=True)
        time = np.arange(time_steps)
        cmap = cm.get_cmap(config['colors']['predictions'])
        colors = [cmap(i / max(1, len(predictions) - 1)) for i in range(len(predictions))]
        
        for i in range(num_vars):
            ax = axes[i] if num_vars > 1 else axes
            ax.plot(time, truth[:, i], label='Truth', color=config['colors']['truth'],
                    linestyle=config['linestyles']['truth'])
            for j, pred in enumerate(predictions):
                ax.plot(time, pred[:, i], label=labels[j], color=colors[j],
                        linestyle=config['linestyles']['predictions'])
            ax.set_ylabel(f'Variable {i+1}' if num_vars > 3 else ['x', 'y', 'z'][i])
            ax.legend()
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        return fig

    def plot_errors(self, errors: Dict[str, List[float]], labels: Optional[List[str]] = None, 
                    **kwargs) -> plt.Figure:
        """
        Plot error metrics over time or across sub-datasets.
        
        Args:
            errors (Dict[str, List[float]]): Dictionary of error metrics (e.g., {'short_time': [values]}).
            labels (Optional[List[str]]): Labels for each error type.
            **kwargs: Custom configurations.
        
        Returns:
            plt.Figure: The generated figure object.
        
        Raises:
            ValueError: If error lists have inconsistent lengths.
        """
        config = {**self.config, **kwargs}
        if not errors:
            raise ValueError("Errors dictionary cannot be empty")
        lengths = {len(err) for err in errors.values()}
        if len(lengths) > 1:
            raise ValueError("All error lists must have the same length")
        
        fig, ax = plt.subplots(figsize=config.get('figure_size', (10, 6)))
        labels = labels if labels else list(errors.keys())
        if len(labels) != len(errors):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of errors ({len(errors)})")
        
        for i, (key, err) in enumerate(errors.items()):
            ax.plot(err, label=labels[i], 
                    color=config['colors']['errors'][i % len(config['colors']['errors'])])
        ax.set_xlabel('Time or Sub-Dataset')
        ax.set_ylabel('Error')
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_histograms(self, truth: np.ndarray, predictions: List[np.ndarray], modes: int, 
                        bins: int, labels: Optional[List[str]] = None, **kwargs) -> plt.Figure:
        """
        Plot histograms of variables for dynamical systems over the last 'modes' time steps.
        
        Args:
            truth (np.ndarray): Ground truth data, shape (time_steps, variables).
            predictions (List[np.ndarray]): List of prediction arrays, shape (time_steps, variables).
            modes (int): Number of last time steps to consider.
            bins (int): Number of bins for histograms.
            labels (Optional[List[str]]): Labels for each prediction.
            **kwargs: Custom configurations.
        
        Returns:
            plt.Figure: The generated figure object.
        
        Raises:
            ValueError: If modes exceeds time steps or shapes are incompatible.
        """
        config = {**self.config, **kwargs}
        time_steps, num_vars = truth.shape
        if modes > time_steps:
            raise ValueError(f"modes ({modes}) cannot exceed time_steps ({time_steps})")
        for pred in predictions:
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")
        
        labels = labels if labels else [f'Prediction {i+1}' for i in range(len(predictions))]
        
        fig, axes = plt.subplots(num_vars, 1, figsize=config.get('figure_size', (10, 2 * num_vars)))
        cmap = cm.get_cmap(config['colors']['predictions'])
        colors = [cmap(i / max(1, len(predictions) - 1)) for i in range(len(predictions))]
        
        for i in range(num_vars):
            ax = axes[i] if num_vars > 1 else axes
            truth_last = truth[-modes:, i]
            ax.hist(truth_last, bins=bins, alpha=0.5, label='Truth', color=config['colors']['truth'])
            for j, pred in enumerate(predictions):
                pred_last = pred[-modes:, i]
                ax.hist(pred_last, bins=bins, alpha=0.5, label=labels[j], color=colors[j])
            ax.set_title(f'Variable {i+1}' if num_vars > 3 else ['x', 'y', 'z'][i])
            ax.legend()
        plt.tight_layout()
        return fig

    def plot_psd(self, truth: np.ndarray, predictions: List[np.ndarray], k: int, 
                 modes: int, labels: Optional[List[str]] = None, **kwargs) -> plt.Figure:
        """
        Plot Power Spectral Density for spatio-temporal systems over the last k time steps.
        
        Args:
            truth (np.ndarray): Ground truth data, shape (time_steps, spatial_points).
            predictions (List[np.ndarray]): List of prediction arrays each shape (time_steps, spatial_points).
            k (int): Number of last time steps to consider.
            modes (int): Number of modes to plot.
            labels (Optional[List[str]]): Labels for each prediction.
            **kwargs: Custom configurations.
        
        Returns:
            plt.Figure: The generated figure object.
        
        Raises:
            ValueError: If k or modes are invalid or shapes are incompatible.
        """
        config = {**self.config, **kwargs}
        time_steps, N = truth.shape
        if k > time_steps:
            raise ValueError(f"k ({k}) cannot exceed time_steps ({time_steps})")
        if modes > N:
            raise ValueError(f"modes ({modes}) cannot exceed spatial points ({N})")
        for pred in predictions:
            if pred.shape != truth.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}")
        
        labels = labels if labels else [f'Prediction {i+1}' for i in range(len(predictions))]
        
        freqs = np.fft.fftshift(np.fft.fftfreq(N))[N//2:N//2 + modes]
        fig, ax = plt.subplots(figsize=config.get('figure_size', (10, 6)))
        cmap = cm.get_cmap(config['colors']['predictions'])
        colors = [cmap(i / max(1, len(predictions) - 1)) for i in range(len(predictions))]
        
        log_psd_truth = compute_log_psd(truth, k, modes)
        ax.plot(freqs, log_psd_truth, label='Truth', color=config['colors']['truth'])
        
        for j, pred in enumerate(predictions):
            log_psd_pred = compute_log_psd(pred, k, modes)
            ax.plot(freqs, log_psd_pred, label=labels[j], color=colors[j])
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Log PSD')
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_from_batch(self, dataset_name: str, pair_id: int, batch_id: str,
                        plot_type: str = 'trajectories', **kwargs) -> plt.Figure:
        """
        Plot data from a run directory, supporting multiple plot types.
        
        Args:
            dataset_name (str): Dataset name (e.g., 'ODE_Lorenz').
            pair_id (int): Pair ID for the sub-dataset.
            batch_path (str): Path to the batch directory containing predictions.npy and evaluation_results.yaml.
            plot_type (str): Type of plot ('trajectories', 'histograms', 'psd', 'errors').
            **kwargs: Custom configurations.
        
        Returns:
            plt.Figure: The generated figure object.
        
        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If plot_type is unsupported or parameters are missing.
        """
        test_data = _load_test_data(dataset_name, pair_id)
        
        batch_dir = Path(batch_id)
        predictions_path = batch_dir / 'predictions.npy'
        if not predictions_path.exists():
            raise FileNotFoundError(f"predictions.npy not found in {batch_id}")
        predictions = [np.load(predictions_path)]
        
        config_path = top_dir / 'data' / dataset_name / f'{dataset_name}.yaml'
        with open(config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        eval_params = dataset_config.get('evaluation_params', {})
        
        if plot_type == 'trajectories':
            return self.plot_trajectories(test_data, predictions, labels=['Model Prediction'], **kwargs)
        elif plot_type == 'histograms':
            modes = eval_params.get('modes', 1000)
            bins = eval_params.get('bins', 50)
            return self.plot_histograms(test_data, predictions, modes, bins, labels=['Model Prediction'], **kwargs)
        elif plot_type == 'psd':
            k = eval_params.get('k', 20)
            modes = eval_params.get('modes', 100)
            return self.plot_psd(test_data, predictions, k, modes, labels=['Model Prediction'], **kwargs)
        elif plot_type == 'errors':
            results_path = batch_dir / 'evaluation_results.yaml'
            if not results_path.exists():
                raise FileNotFoundError(f"evaluation_results.yaml not found in {batch_id}")
            with open(results_path, 'r') as f:
                errors = yaml.safe_load(f)
            return self.plot_errors(errors, **kwargs)
        elif plot_type == '2d_comparison':
            # Check that data is 2D
            for prediction in predictions:
                if len(test_data.shape) != 2 or len(prediction.shape) != 2:
                    raise ValueError(
                        f"2d_comparison requires 2D data, but shapes are {test_data.shape} and {predictions[0].shape}")
            return self.compare_prediction(test_data, predictions, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def generate_all_plots(self, dataset_name: str, batch_path: str, **kwargs):
        """
        Generate all applicable plots for a dataset based on its configuration.
        
        Args:
            dataset_name (str): Dataset name.
            batch_path (str): Path to the batch directory.
            **kwargs: Custom configurations.
        
        Raises:
            FileNotFoundError: If dataset config file is missing.
        """
        applicable_plots = get_applicable_plots(dataset_name)
        batch_dir = Path(batch_path)
        for pair_dir in batch_dir.glob('pair*'):
            pair_id = int(pair_dir.name.replace('pair', ''))
            for plot_type in applicable_plots:
                fig = self.plot_from_batch(dataset_name, pair_id, str(pair_dir), plot_type=plot_type, **kwargs)
                fig.savefig(pair_dir / f'{plot_type}_plot.png')
                plt.close(fig)

    def save_figure_results(self, fig: plt.Figure, dataset_name: str, model_name: str, 
                            batch_name: str, pair_id: int, plot_type: str, results_dir: Optional[str] = None) -> None:
        """
        Save the figure to the results directory under a visualizations subfolder.
        
        Args:
            fig (plt.Figure): The figure to save.
            dataset_name (str): Name of the dataset.
            model_name (str): Name of the model.
            batch_name (str): Batch identifier.
            pair_id (int): Sub-dataset identifier.
            plot_type (str): Type of plot (e.g., 'trajectories', 'histograms').
            results_dir (optional, str): Path to the directory containing the run results
        
        Raises:
            FileNotFoundError: If the results directory cannot be created.
        """
        if results_dir is None:
            results_dir = top_dir / 'results' / dataset_name / model_name / batch_name / f'pair{pair_id}' / 'visualizations'
        else:
            results_dir = results_dir / "visualizations"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f'{plot_type}.png'
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved {plot_type} plot to {save_path}")

    def plot_prediction(self, ax, data, vmin = None, vmax = None, show_ticks = True, show_xlabel = False, show_ylabel = False):
        """
        Plot a 2D prediction on the given axes with customizable display options.
        
        Args:
            ax (matplotlib.axes.Axes): The axes to plot on.
            data (np.ndarray): 2D array of data to plot, shape (time_steps, spatial_dim).
            vmin (optional, float): Minimum value for color scaling.
            vmax (optional, float): Maximum value for color scaling.
            show_ticks (bool): Whether to show axis ticks.
            show_xlabel (bool): Whether to show x-axis label (dimension 1 of data).
            show_ylabel (bool): Whether to show y-axis label (dimension 0 of data).
        
        Returns:
            matplotlib.image.AxesImage: The image object created by imshow.
        """
        cmap = cm.get_cmap(self.config['images']['colormap'])
        extent = [0, data.shape[1], data.shape[0], 0]
        img = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower',
                       extent=extent, vmin=vmin, vmax=vmax)

        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        if show_xlabel:
            ax.set_xlabel('Spatial Dimension (x)')
        if show_ylabel:
            ax.set_ylabel('Time Dimension (t)')

        return img

    def compare_prediction(self, truth, predictions, cbar_options=None, show_ticks=True, show_titles=True):
        """
        Create a side-by-side comparison of test data and predicted data.
        
        Args:
            truth (np.ndarray): Ground truth data, shape (time_steps, variables).
            predictions (List[np.ndarray]): List of prediction arrays, each with shape (time_steps, variables).
            cbar_options (optional, dict): Options for colorbar display. Can include:
                - show (bool): Whether to show colorbar
                - orientation (str): 'horizontal' or 'vertical'
                - shrink (float): Scale factor for colorbar size
                - ticks (int/list/np.ndarray): Tick locations or number of ticks
                - label (str): Colorbar label
            show_ticks (bool): Whether to show axis ticks.
            show_titles (bool): Whether to show subplot titles.
        
        Returns:
            plt.Figure: The created figure containing the comparison plot.
        """
        figsize = self.config['figure_size']

        default_cbar_options = {
            'show': True,
            'orientation': 'horizontal',
            'shrink': 0.8
        }

        # Safely merge with user input
        if cbar_options is None:
            cbar_options = default_cbar_options
        else:
            cbar_options = {**default_cbar_options, **cbar_options}

        nr_predictions = len(predictions)
        fig, axs = plt.subplots(1*nr_predictions, 3, figsize=figsize)

        # Convert axs to 2D array even when nr_predictions is 1
        if nr_predictions == 1:
            axs = np.array([axs])

        for i, prediction in enumerate(predictions):
            vmin, vmax = min(truth.min(), prediction[i].min()), max(truth.max(), prediction.max())
            img_test = self.plot_prediction(axs[i, 0], truth, vmin=vmin, vmax=vmax, show_ticks=show_ticks, show_xlabel=True, show_ylabel=True)
            self.plot_prediction(axs[i, 1], prediction, vmin=vmin, vmax=vmax, show_ticks=show_ticks, show_xlabel=True, show_ylabel=False)
            error_data = truth - prediction
            self.plot_prediction(axs[i, 2], error_data, vmin=vmin, vmax=vmax, show_ticks=show_ticks, show_xlabel=True, show_ylabel=False)

            if show_titles:
                axs[i, 0].set_title('Test Data')
                axs[i, 1].set_title('Predicted Data')
                axs[i, 2].set_title('Error Data')

        if cbar_options.get('show', True):
            cbar = fig.colorbar(
                img_test,
                ax=axs,
                orientation=cbar_options.get('orientation', 'horizontal'),
                shrink=cbar_options.get('shrink', 0.8)
            )

            ticks = cbar_options.get('ticks')
            if ticks is not None:
                if isinstance(ticks, (np.ndarray, list)):
                    cbar.set_ticks(ticks)
                elif isinstance(ticks, int):
                    cbar.set_ticks(np.linspace(truth.min(), truth.max(), ticks))

            label = cbar_options.get('label')
            if label:
                cbar.set_label(label)
        return fig