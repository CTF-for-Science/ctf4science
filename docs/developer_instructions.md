# Contributing to the CTF for Science Framework

Welcome to the CTF for Science Framework! We’re excited to have you contribute. To ensure a consistent and high-quality codebase, please follow the guidelines below. These standards help maintain readability, reliability, and collaboration across the project.

## Coding Standards

Adhering to coding standards is crucial for a maintainable and scalable project. Please follow these guidelines:

- **Modular and Reusable Functions**:

  - Write small, focused functions that do one thing well.
  - Use descriptive names (e.g., `load_dataset` instead of `load_data`).

- **Error Handling**:

  - Use specific exceptions (e.g., `ValueError`, `FileNotFoundError`) instead of generic `Exception`.
  - Provide clear error messages to aid debugging.

- **Type Hints**:

  - Include type hints for function arguments and return values (e.g., `def func(arg: int) -> str:`).
  - This improves code clarity and helps with static type checking.

- **Docstrings**:

  - Use NumPy-style docstrings for functions and classes.
  - Include sections for `Args`, `Returns`, and `Raises` where applicable.
  - Example:

    ```python
    def example_function(arg: int) -> str:
        """
        Brief description.
    
        Args:
            arg (int): Description of arg.
    
        Returns:
            str: Description of return value.
    
        Raises:
            ValueError: If arg is invalid.
        """
    ```

- **Variable and File Path Naming**:

  - Use descriptive variable names (e.g., `dataset_name` instead of `dn`).
  - Use `pathlib.Path` for file paths to ensure cross-platform compatibility.

- **Internal Functions**:

  - Prefix functions intended for internal use with an underscore (e.g., `_helper_function`).
  - This indicates they are not part of the public API.

- **Imports**:

  - List all imports explicitly at the top of the file.
  - Avoid wildcard imports (e.g., `from module import *`).

- **PEP 8 Compliance (Not currently enforced, but desired) **:
  - Follow PEP 8 guidelines for code style.
  - You can use tools like `flake8` or `pylint` to check compliance.

## Testing Standards

Testing is essential to ensure the reliability of the framework. We use Python’s `unittest` module for writing and running tests.

- **Writing Tests**:

  - Write unit tests for new functions, classes, or bug fixes.
  - Place tests in the `tests` directory, following the naming convention `test_<module>.py`.
  - Use `assert` methods to verify expected behavior (e.g., `self.assertEqual`, `self.assertRaises`).

- **Running Tests**:

  - To run all tests, navigate to the top-level directory and execute:

    ```bash
    python -m unittest
    ```
  - Ensure all tests pass before submitting changes.

- **Test Coverage**:

  - Aim for high test coverage, especially for critical functions.
  - Use tools like `coverage.py` to measure and improve coverage.

## Additional Guidelines

### Documentation

- Update or add documentation for new features or significant changes.
- Ensure docstrings are up-to-date and accurately reflect the code.

### Version Control

- Write clear, concise commit messages (e.g., "Add function to load dataset").
- Use feature branches for development and submit pull requests for review.

### Code Review

- Participate in code reviews to maintain code quality.
- Be open to feedback and make necessary adjustments.

---

Thank you for contributing to the CTF for Science Framework! Your efforts help advance scientific computing and modeling. If you have any questions, feel free to reach out to the project maintainers.
