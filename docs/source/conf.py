# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OSEkit"
copyright = "2025, osekit"
author = "osekit"
release = "0.2.5"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "myst_nb"]

templates_path = ["_templates"]
exclude_patterns = []
autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "../logo/osekit_small.png"
html_title = "OSEKit"

html_theme_options = {
    "repository_url": "https://github.com/Project-OSmOSE/OSEkit",
    "use_repository_button": True,
}
