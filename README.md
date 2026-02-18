# CTF for Science Framework

Welcome to the **CTF for Science Framework**, a modular and extensible platform designed for benchmarking modeling methods on chaotic systems, from the [AI Institute in Dynamic Systems](https://dynamicsai.org/). This framework supports the evaluation and comparison of models for systems like ordinary differential equations (ODEs, e.g., Lorenz system) and partial differential equations (PDEs, e.g., Kuramoto-Sivashinsky equation) using standardized datasets and metrics.

Website: [ctf-for-science.github.io/ctf4science/](https://ctf-for-science.github.io/ctf4science/)

## Overview

The framework provides:
- A standardized environment for submitting and evaluating models.
- Predefined datasets and evaluation metrics.
- Tools for running models, saving results, and visualizing performance.

Whether you're a researcher benchmarking a new method or a contributor adding a model, this framework streamlines the process.

### Team:  

| Name             | Email                           | Github           | Affiliation                          |
|------------------|---------------------------------|------------------|--------------------------------------|
| Philippe Wyder   | pwyder@uw.edu                   | GitWyd           | University of Washington             |
| Judah Goldfeder  | jag2396@columbia.edu            | Jgoldfeder       | Columbia University                  |
| Alexey Yermakov  | alexeyy@uw.edu                  | yyexela          | University of Washington             |
| Yue Zhao         | yue.zhao@surf.nl                | yuezhao6371      | SURF (Netherlands)                   |
| Stefano Riva     | stefano.riva@polimi.it          | steriva          | Politecnico di Milano                |
| Jan Williams     | jmpw1@uw.edu                    | Jan-Williams     | University of Washington             |
| David Zoro       | zorodav@uw.edu                  | zorodav          | University of Washington             |
| Amy Sara Rude    | amysrude@uw.edu                 | amysrude         | University of Washington             |
| Matteo Tomasetto | matteo.tomasetto@polimi.it      | MatteoTomasetto  | Politecnico di Milano                |
| Joe Germany      | jmg15@mail.aub.edu              | joeGermany       | American University of Beirut        |
| Joseph Bakarji   | jb50@aub.edu.lb                 | josephbakarji    | American University of Beirut        |
| Georg Maierhofer | gam37@cam.ac.uk                 | GeorgAUT         | University of Cambridge              |
| Miles Cranmer    | mc2473.cam.ac.uk@gmail.com      | MilesCranmer     | University of Cambridge              |
| Nathan Kutz      | kutz@uw.edu                     | nathankutz       | University of Washington             |


## Quickstart 

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

### Download the Data
Please visit our Open Science Framework [page](https://osf.io/6rzhm/overview) to download all datasets. Each dataset can be extracted by running `tar -xzvf {dataset.tar.gz}`. Place the datasets in a top-level `data/` directory.

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

To install and start using the framework, follow the instructions in [ctf-for-science.github.io/ctf4science/getting_started.html](https://ctf-for-science.github.io/ctf4science/getting_started.html). This guide covers:
- Installation steps.
- Running a quick example with a baseline model.
- Adding your own model to the framework.

## Directory Structure

- `data/`: Contains datasets from our [OSF page](https://osf.io/6rzhm/overview).
- `models/`: Contains model implementations (e.g., baselines and user-contributed models).
- `results/`: Stores model predictions, evaluation results, and visualizations.
- `notebooks/`: Jupyter notebooks to analyze or visualize results.
- `test/`: Contains unit tests for the `ctf4science` package.

## Contributing a Model

We welcome contributions! To add a new model or improve the framework, see the detailed steps in [ctf-for-science.github.io/ctf4science/getting_started.html#contributing-a-new-model](https://ctf-for-science.github.io/ctf4science/getting_started.html#contributing-a-new-model).

## Contributing to the ctf4science Package

Refer to [ctf-for-science.github.io/ctf4science/contributing.html](https://ctf-for-science.github.io/ctf4science/contributing.html).

## Kaggle Page
Check out the Dynamic AI Institute [Kaggle Page](https://www.kaggle.com/organizations/dynamics-ai), for datasets and upcoming contests.
## Papers that Inspired This Work

- [Descending through a Crowded Valley - Benchmarking Deep Learning Optimizers](https://arxiv.org/abs/2007.01547)
- [CoDBench: A Critical Evaluation of Data-driven Models for Continuous Dynamical Systems](https://arxiv.org/abs/2310.01650)
- [Weak baselines and reporting biases lead to over-optimism in machine learning for fluid-related partial differential equations](https://www.nature.com/articles/s42256-024-00897-5)
- [The Well: Dynamic System Dataset & Benchmarking](https://polymathic-ai.org/the_well)

## Citations

If you used our package in your work, please cite:

```bibtex
@article{wydercommon,
  title={Common Task Framework For a Critical Evaluation of Scientific Machine Learning Algorithms},
  author={Wyder, Philippe Martin and Goldfeder, Judah and Yermakov, Alexey and Zhao, Yue and Riva, Stefano and Williams, Jan P and Zoro, David and Rude, Amy Sara and Tomasetto, Matteo and Germany, Joe and Maierhofer, Georg and Cranmer, Miles and Kutz, J. Nathan},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

and

```bibtex
@misc{yermakov2025seismicwavefieldcommontask,
      title={The Seismic Wavefield Common Task Framework}, 
      author={Alexey Yermakov and Yue Zhao and Marine Denolle and Yiyu Ni and Philippe M. Wyder and Judah Goldfeder and Stefano Riva and Jan Williams and David Zoro and Amy Sara Rude and Matteo Tomasetto and Joe Germany and Joseph Bakarji and Georg Maierhofer and Miles Cranmer and J. Nathan Kutz},
      year={2025},
      eprint={2512.19927},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.19927}, 
}
```

If you used our data, please also cite:

```bibtex
@misc{ctf4science_osf,
    author={Yermakov, Alexey and Zhao, Yue and Denolle, Marine and Ni, Yiyu and Wyder, Philippe Martin and Goldfeder, Judah and Riva, Stefano and Williams, Jan P and Zoro, David and Rude, Amy Sara and Tomasetto, Matteo and Germany, Joe and Maierhofer, Georg and Cranmer, Miles and Kutz, J. Nathan},
    title={ctf4science},
    year={2025},
    doi={10.17605/OSF.IO/6RZHM},
    url={https://osf.io/6rzhm},
    note={Open Science Framework project}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. This license only covers the ctf4science package. All models linked as submodules are subject to their respective license.

## Questions?

For support or inquiries, open an issue on our [GitHub repository](https://github.com/CTF-for-Science/ctf4science).
