# Getting Started with CTF for Science Framework

This guide provides detailed instructions to help new users install, use, and contribute to the CTF for Science Framework.

## Installation

To set up the framework on your system:

1. **Clone the Repository**:  
    Using SSH (Recommended)
    ```bash
    git clone --recursive git@github.com:CTF-for-Science/ctf4science.git
    ```  

    Using HTTPS (requires GitHub authentication):  
    ```bash
    git clone --recursive https://github.com/CTF-for-Science/ctf4science.git
    ```

2. **Install the Framework**:  
   Install the framework as a Python package in editable mode:
   ```bash
   pip install -e .
   ```

   This installs core dependencies:
   - `numpy`
   - `scipy`
   - `pyyaml`
   - `matplotlib`
   - `importlib-resources`

   To install all optional dependencies, run `pip install -e .[all]` instead. This installs optional dependencies:
   - `optuna`
   - `jupyterlab`
   **Note**: zsh shell users should run  `pip install -e '.[all]'` to avoid errors.
   **Note**: Some models may require additional dependencies, specified in their own `requirements.txt` files.

## Quick Start

To test the framework with a baseline model:

1. Navigate to the naive baselines directory:
   ```bash
   cd models/CTF_NaiveBaselines
   ```

2. Run the 'average' baseline on the Lorenz system:
   ```bash
   python run.py config/config_Lorenz_average_batch_1-6.yaml
   ```

   This will:
   - Load the Lorenz dataset for sub-datasets 1 through 6.
   - Generate predictions using the 'average' method for each sub-dataset.
   - Save results, including visualizations, to `results/ODE_Lorenz/CTF_NaiveBaselines_average/<batch_id>/`.

   **Note**: Check the `results/` directory for outputs like `predictions.npy`, `evaluation_results.yaml`, and visualization plots (e.g., `trajectories.png`, `histograms.png`) for each sub-dataset.

## Understanding the Core Modules

The framework relies on three key modules:

- **`data_module.py`**:
  - **Purpose**: Handles dataset loading and sub-dataset selection.
  - **Key Functions**:
    - `load_dataset(dataset_name, pair_id)`: Loads train and test data for a specific sub-dataset.
    - `parse_pair_ids(dataset_config)`: Interprets the `pair_id` configuration to determine which sub-datasets to process. It supports multiple formats (see "Configuring Your Run" below).

- **`eval_module.py`**:
  - **Purpose**: Computes evaluation metrics for model predictions.
  - **Key Functions**:
    - `evaluate(dataset_name, pair_id, truth, prediction)`: Calculates metrics like short-time forecast, reconstruction, and long-time forecast.
    - `save_results(...)`: Saves config, predictions, and metrics to the `results/` directory for each sub-dataset.

- **`visualization_module.py`**:
  - **Purpose**: Generates plots to visualize predictions and metrics.
  - **Key Features**: Auto-generates plots (e.g., trajectories, histograms, PSD) during runs, saved in `results/**/visualizations/`.

## Configuring Your Run

Configuration files (e.g., `config_Lorenz_average_batch_1-6.yaml`) control dataset and model parameters. The `dataset` section is **required** in every config file and specifies the dataset name and the sub-datasets to run on.

### Dataset Configuration Options

- **`dataset`** (Required):
  - `name`: The name of the dataset (e.g., `ODE_Lorenz`, `PDE_KS`).
  - `pair_id`: Specifies which sub-datasets to run on. This field is optional within the `dataset` section, but if omitted, the framework will run on all available sub-datasets. Supported formats include:
    - **Single integer**: Run on a specific sub-dataset.
      ```yaml
      pair_id: 3  # Run on sub-dataset 3 only
      ```
    - **List of integers**: Run on multiple specific sub-datasets.
      ```yaml
      pair_id: [1, 2, 3, 4, 5, 6]  # Run on sub-datasets 1 through 6
      ```
    - **Range string**: Run on a range of sub-datasets.
      ```yaml
      pair_id: '1-3'  # Run on sub-datasets 1, 2, and 3
      ```
    - **Omitted or `'all'`**: Run on all available sub-datasets for the dataset.
      ```yaml
      dataset:
        name: PDE_KS
        # pair_id omitted or set to 'all' to run on all sub-datasets
      ```

Example configuration:
```yaml
dataset:
  name: ODE_Lorenz
  pair_id: '1-6'  # Run on sub-datasets 1 through 6
model:
  name: CTF_NaiveBaselines
  method: average
```

See [configuration.md](configuration.md) for full details (to be created).

## Contributing a New Model

To integrate your own model into the framework, follow these steps:

### Step 1: Create a Model Directory
Create a new directory under `models/` (e.g., `models/MyModel`):
```bash
mkdir models/MyModel
```

### Step 2: Implement Your Model
In `models/MyModel/`, create a Python file (e.g., `my_model.py`) with your model’s logic. It should:
- Accept a `config` dictionary and optional `train_data` during initialization.
- Provide a `predict` method for generating predictions.

Example:
```python
class MyModel:
    def __init__(self, config, train_data=None):
        self.config = config
        self.train_data = train_data
        # Add initialization logic

    def predict(self, test_data):
        # Add prediction logic
        return predictions
```

### Step 3: Create a `run.py` File
Add a `run.py` file in `models/MyModel/` to handle batch runs across multiple sub-datasets. The framework uses a batch run approach, processing all specified sub-datasets and saving results under a unique batch identifier. Below is an example:

```python
import argparse
import yaml
from pathlib import Path
import datetime
from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from my_model import MyModel


def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract dataset name and parse pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = "MyModel"
    # Generate a unique batch_id for this run
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize visualization object
    viz = Visualization()
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:
        # Load sub-dataset
        train_data, test_data, init_data = load_dataset(dataset_name, pair_id)
        # Initialize model
        model = MyModel(config, train_data)
        # Generate predictions
        predictions = model.predict()
        # Evaluate predictions
        results = evaluate(dataset_name, pair_id, test_data, predictions)
        # Save results and get directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, predictions, results)

        # Append metrics to batch results
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type)

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    main(args.config)
```

### Step 4: Add Configuration Files
In `models/MyModel/config/`, create a YAML file (e.g., `my_config.yaml`):
```yaml
dataset:
  name: ODE_Lorenz
  pair_id: '1-3'  # Example: run on sub-datasets 1 to 3
model:
  param1: value1
```

### Step 5: Document Your Model
Add a `README.md` in `models/MyModel/` explaining your model, dependencies, and usage.

### Step 6: Test Your Model
Run your model:
```bash
python models/MyModel/run.py models/MyModel/config/my_config.yaml
```

Verify the output in `results/`.

## Running Models

To run any model:
```bash
python models/<model_name>/run.py models/<model_name>/config/<config_file>.yaml
```

## Adding Your Model to ctf4science: Best Practices

Now that you have developed your own model for the CTF in its own branch and you want to add it to the ctf4science repository. To maintain code quality and review processes, we recommend adding your model as a submodule through a development branch rather than pushing directly to the main branch.

1. First, clone the ctf4science repository:

```bash
git clone --recursive git@github.com:CTF-for-Science/ctf4science.git
cd ctf4science
```

2. Create a new development branch:

```bash
git checkout -b add-mymodel-submodule
```

3. Add your model repository as a submodule in the `models` directory:

```bash
git submodule add git@github.com:MyGithubName/MyModelRepo.git models/MyModelRepo
```

4. Commit the changes to your development branch:

```bash
git commit -m "Add MyModelRepo as a submodule"
```

5. Push your development branch to the remote repository:

```bash
git push origin add-mymodel-submodule
```

6. Create a pull request (PR) from your branch to the main branch:
   - Go to the ctf4science repository on GitHub
   - Click "Pull requests" > "New pull request"
   - Set the base branch to `main` and the compare branch to `add-mymodel-submodule`
   - Add a description explaining your model and its integration
   - Submit the pull request for review

## Results and Visualization

After running a model, the framework saves results for each sub-dataset in `results/<dataset>/<model>/<batch_id>/<pair_id>/`, including:
- `config.yaml`: Configuration used.
- `predictions.npy`: Predicted data array.
- `evaluation_results.yaml`: Metrics (e.g., short_time, reconstruction, long_time).
- `visualizations/`: Auto-generated plots (e.g., `trajectories.png`, `histograms.png`, `psd.png`).

A `batch_results.yaml` file is also saved in `results/<dataset>/<model>/<batch_id>/`, summarizing the metrics for all sub-datasets in the batch.

Use the Jupyter notebooks in the `notebooks/` directory for further analysis or custom visualizations.

## Additional Documentation

- [datasets.md](datasets.md): Datasets overview.
- [evaluation.md](evaluation.md): Metrics overview.
- [evaluation_module.md](evaluation_module.md): Evaluation Module overview.
- [visualization.md](visualization.md): Visualization instructions.
- [configuration.md](configuration.md): Config file structure (planned).
- [developer_instructions.md](developer_instructions.md): Developer instructions
- [hyperparameter_optimization.md](hyperparameteroptimization.md): Information about hyperparameter optimization.
- API docs in `docs/api/` (planned).
