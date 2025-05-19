# CTF for Science Framework

Welcome to the **CTF for Science Framework**, a modular and extensible platform designed for benchmarking modeling methods on chaotic systems, from the [AI Institute in Dynamic Systems](https://dynamicsai.org/). This framework supports the evaluation and comparison of models for systems like ordinary differential equations (ODEs, e.g., Lorenz system) and partial differential equations (PDEs, e.g., Kuramoto-Sivashinsky equation) using standardized datasets and metrics.

## Overview

The framework provides:
- A standardized environment for submitting and evaluating models.
- Predefined datasets and evaluation metrics.
- Tools for running models, saving results, and visualizing performance.

Whether you're a researcher benchmarking a new method or a contributor adding a model, this framework streamlines the process.

### Team:  

| Name            | Email                   | Github         |
|-----------------|-------------------------|----------------|
| Philippe Wyder  | pwyder@uw.edu           | GitWyd         |
| Judah Goldfeder | jag2396@columbia.edu    | Jgoldfeder     |
| Alexey Yermakov | alexeyy@uw.edu          | yyexela        |
| Yue Zhao        | yue.zhao@surf.nl        | yuezhao6371    |
| Stefano Riva    | stefano.riva@polimi.it  | steriva        |
| Jan Williams    | jmpw1@uw.edu            | Jan-Williams   |
| David Zoro      | zorodav@uw.edu          | zorodav        |
| Amy Sara Rude   | amysrude@uw.edu         | amysrude       |
| Matteo Tomasetto| matteo.tomasetto@polimi.it | MatteoTomasetto |
| Joe Germany     | jmg15@mail.aub.edu      | joeGermany     |
| Joseph Bakarji  | jb50@aub.edu.lb         | josephbakarji  |
| Georg Maierhofer| gam37@cam.ac.uk         | GeorgAUT       |
| Miles Cranmer   | mc2473.cam.ac.uk@gmail.com | MilesCranmer |
| Nathan Kutz     | kutz@uw.edu             | nathankutz     |

## 🔧 Quickstart 

Run a simple experiment on the Lorenz dataset with naive baselines:

### Prerequisites
- Git installed on your system
- Github account
- SSH key set up with GitHub (see https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

### Clone the Repository
Using SSH (Recommended)
```bash
git clone --recursive git@github.com:CTF-for-Science/ctf4science.git
```

Using HTTPS (requires GitHub authentication):
```bash
git clone --recursive https://github.com/CTF-for-Science/ctf4science.git
```

### Install the Repository and run an example
```bash
git clone --recurse-submodules https://github.com/CTF-for-Science/ctf4science.git
cd ctf4science
pip install -e .
python models/CTF_NaiveBaselines/run.py models/CTF_NaiveBaselines/config/config_Lorenz_average_batch_all.yaml
```

Note that the `--recurse-submodules` flag will clone the associated model submodule repositories as well. If you don't want to download all submodules, and only want to run the CTF_NaiveBaselines, then you can run `git submodule update models/CTF_NaiveBaselines` after running `git clone https://github.com/CTF-for-Science/ctf4science.git`, and thereby circumvent cloning all the modules.

**Note**: This runs the 'average' baseline on the Lorenz dataset for sub-datasets 1 through 6. Results, including predictions, evaluation metrics, and visualizations (e.g., trajectory and histogram plots), are automatically saved in `results/ODE_Lorenz/CTF_NaiveBaselines_average/<batch_id>/`.

**Note**: To install optional dependencies, run `pip install -e .[all]` instead.

### 📁 Results Directory Structure

After a run, results are saved to:

```
results/<dataset>/<model>/<batch_id>/
  ├── <pair_id>/               # Metrics, predictions, and visualizations for each sub-dataset
  │   ├── config.yaml          # Configuration used
  │   ├── predictions.npy      # Predicted data
  │   ├── evaluation_results.yaml # Evaluation metrics
  │   └── visualizations/      # Auto-generated plots (e.g., trajectories.png)
  └── batch_results.yaml       # Aggregated batch metrics
```

## Getting Started With Your Own Model

To install and start using the framework, follow the instructions in [docs/getting_started.md](docs/getting_started.md). This guide covers:
- Installation steps.
- Running a quick example with a baseline model.
- Adding your own model to the framework.

## Directory Structure

- `models/`: Contains model implementations (e.g., baselines and user-contributed models).
- `results/`: Stores model predictions, evaluation results, and visualizations.
- `docs/`: Additional documentation (e.g., contributing, configuration).
- `notebooks/`: Jupyter notebooks to analyze or visualize results.
- `tests/`: Contains unit tests for the `ctf4science` package.

## Contributing a Model

We welcome contributions! To add a new model or improve the framework, see the detailed steps in [docs/getting_started.md#contributing-a-new-model](docs/getting_started.md#contributing-a-new-model).

## Contributing to the ctf4science Package

Refer to [docs/developer_instructions.md](docs/developer_instructions).

## Papers that Inspired This Work

- [Descending through a Crowded Valley - Benchmarking Deep Learning Optimizers](https://arxiv.org/abs/2007.01547)
- [CoDBench: A Critical Evaluation of Data-driven Models for Continuous Dynamical Systems](https://arxiv.org/abs/2310.01650)
- [Weak baselines and reporting biases lead to over-optimism in machine learning for fluid-related partial differential equations](https://www.nature.com/articles/s42256-024-00897-5)
- [The Well: Dynamic System Dataset & Benchmarking](https://polymathic-ai.org/the_well)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. This license only covers the ctf4science package. All models linked as submodules are subject to their respective license.

## Questions?

For support or inquiries, open an issue on our [GitHub repository](https://github.com/CTF-for-Science/ctf4science).
