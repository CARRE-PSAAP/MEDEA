# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MEDEA"
copyright = "2026 CARRE"
author = "Kyle Hansen, India Allan, Caleb Shaw"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Extracts Python docstrings
    "sphinx.ext.viewcode",  # Adds links to source code
    "breathe",  # Bridges Doxygen XML to Sphinx
    "sphinx.ext.mathjax",  # render latex math
    "sphinxcontrib.bibtex"
]

bibtex_bibfiles = ['refs.bib']

breathe_projects = {"MEDEA": "../build/doxygen/xml"}
breathe_default_project = "MEDEA"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- MathJax formatting configuration -------------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            'vec': [r'{\boldsymbol{#1}}', 1],
        }
    }
}