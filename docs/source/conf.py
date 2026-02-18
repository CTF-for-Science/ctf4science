import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CTF for Science"
copyright = "2026, CTF4Science Team"
author = "CTF4Science Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["generated/ctf4science.config.rst"]

autodoc_default_options = {
    # Do not add members on automodule pages; rely on per-object stubs instead.
    "inherited-members": False,
    "show-inheritance": False,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "CTF for Science Documentation"
html_theme = "pydata_sphinx_theme"
html_logo = "_static/images/ctf4science-logo-light.svg"
html_show_sourcelink = False

html_theme_options = {
    "logo": {
        "image_light": "_static/images/ctf4science-logo-light.svg",
        "image_dark": "_static/images/ctf4science-logo-dark.svg",
    },
    "github_url": "https://github.com/CTF-for-Science/ctf4science",
}

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Numpydoc configuration: do not show or link inherited methods
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
