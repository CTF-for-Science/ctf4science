Contributing to the CTF for Science Framework
=============================================

Welcome to the CTF for Science Framework! We're excited to have you contribute. To ensure a consistent and high-quality codebase, please follow the guidelines below. These standards help maintain readability, reliability, and collaboration across the project.

Environment
-----------

The environment can be created by running::

   pyenv install 3.13.7
   pyenv local 3.13.7
   python -m venv ~/.virtualenvs/ctf4science
   source ~/.virtualenvs/ctf4science/bin/activate
   pip install -e .[all,dev]

Nox Taskrunner
--------------

We use `Nox <https://nox.thea.codes/en/stable/index.html>`__ to automate all tasks in our package. Specifically, running ``nox`` runs all tests, lints and formats the code, performs typechecking, and builds the documentation. It is also used for the continuous integration pipeline to do all the aforementioned in Github Actions.

To run a specific Nox task, run ``nox --list`` to see available tasks and then ``nox --no-venv --no-install -s <task>`` to run the task.

Pre-commit
----------

We use `pre-commit <https://pre-commit.com/>`__ to verify all code is linted, formatted, and type-checked before committed. This ensures a smoother code development process. Use ``pre-commit install`` to install the appropriate hooks and verify it runs with ``pre-commit run --all-files``.

Linting and Formatting
----------------------

Linting ensures that the code logic makes sense and nothing is being called in a way that would surely break. For this, we use `Ruff <https://docs.astral.sh/ruff/>`__. We also use Ruff for code formatting, ensuring a standard format across all code. There is a Ruff VSCode extension that can be used to help with linting and formatting.

Type checking
-------------

In this repository, we use `Pyrefly <https://pyrefly.org/>`__ to check all functions are called with their specified types. We also use `jaxtyping <https://docs.kidger.site/jaxtyping/api/array/>`__ to allow for clear function typing for PyTorch array dimensions. There is a Pyrefly VSCode extension that can be used to help with type checking.

Building/Serving Documentation
------------------------------

We use `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ to build our documentation. Make sure that the ``sphinx`` package is installed in your python environment. The source files are located in ``doc/``. To build the documentation, run ``nox --no-venv --no-install -s build_docs``.

To preview documentation files, open ``docs/build/html/index.html`` locally in your browser. If you're developing on a server, first run ``python -m http.server 8000`` in ``docs/build/html`` and then on your local computer run ``ssh -L 8080:localhost:8000 <server_address>``. This will turn the server into an HTTP server and opening ``localhost:8080`` in a browser will show the generated documentation.

Adding to Documentation
-----------------------

We use `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ as our documentation generator. To add to the documentation, add to the ``docs/source/`` directory. All files in this directory will be included in the documentation. Please read the `Sphinx documentation <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`__ for more information on how to write documentation.

Coding Standards
----------------

Adhering to coding standards is crucial for a maintainable and scalable project. Please follow these guidelines:

* **Modular and Reusable Functions**:

  * Write small, focused functions that do one thing well.
  * Use descriptive names (e.g., ``load_dataset`` instead of ``load_data``).

* **Error Handling**:

  * Use specific exceptions (e.g., ``ValueError``, ``FileNotFoundError``) instead of generic ``Exception``.
  * Provide clear error messages to aid debugging.

* **Type Hints**:

  * Include type hints for function arguments and return values (e.g., ``def func(arg: int) -> str:``).
  * This improves code clarity and helps with static type checking.

* **Docstrings**:

  * Use NumPy-style docstrings for functions and classes.
  * Include sections for ``Args``, ``Returns``, and ``Raises`` where applicable.
  * Example:

    .. code-block:: python

       def example_function(arg: int) -> str:
           """
           Brief description.

           Parameters
           ----------
           arg (int): Description of arg.

           Returns
           -------
           str: Description of return value.

           Raises
           ------
           ValueError: If arg is invalid.
           """


* **Variable and File Path Naming**:

  * Use descriptive variable names (e.g., ``dataset_name`` instead of ``dn``).
  * Use ``pathlib.Path`` for file paths to ensure cross-platform compatibility.

* **Internal Functions**:

  * Prefix functions intended for internal use with an underscore (e.g., ``_helper_function``).
  * This indicates they are not part of the public API.

* **Imports**:

  * List all imports explicitly at the top of the file.
  * Avoid wildcard imports (e.g., ``from module import *``).

* **PEP 8 Compliance (Not currently enforced, but desired)**:
  * Follow PEP 8 guidelines for code style.
  * You can use tools like ``flake8`` or ``pylint`` to check compliance.

Testing Standards
-----------------

Testing is essential to ensure the reliability of the framework. We use Python's ``unittest`` module for writing and running tests.

* **Writing Tests**:

  * Write unit tests for new functions, classes, or bug fixes.
  * Place tests in the ``test`` directory, following the naming convention ``test_<module>.py``.
  * Use ``assert`` methods to verify expected behavior (e.g., ``self.assertEqual``, ``self.assertRaises``).

* **Running Tests**:

  * To run all tests, navigate to the top-level directory and execute:

    .. code-block:: bash

       pytest

  * Ensure all tests pass before submitting changes.

* **Test Coverage**:

  * Aim for high test coverage, especially for critical functions.
  * Use tools like ``coverage.py`` to measure and improve coverage.

Additional Guidelines
---------------------

Documentation
~~~~~~~~~~~~~

* Update or add documentation for new features or significant changes.
* Ensure docstrings are up-to-date and accurately reflect the code.

Version Control
~~~~~~~~~~~~~~~

* Write clear, concise commit messages (e.g., "Add function to load dataset").
* Use feature branches for development and submit pull requests for review.

Code Review
~~~~~~~~~~~

* Participate in code reviews to maintain code quality.
* Be open to feedback and make necessary adjustments.

----

Thank you for contributing to the CTF for Science Framework! Your efforts help advance scientific computing and modeling. If you have any questions, feel free to reach out to the project maintainers.
