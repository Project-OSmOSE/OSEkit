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

extensions = ["myst_parser", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = []
autoclass_content = "both"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "searchbox.html"],
}  # Displays the global TOC in the sidebar
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "logo_only": True,
    "style_nav_header_background": "#252529",
    "navigation_depth": 6,
}
html_favicon = "_static/favicon.ico"
html_logo = "../logo/osekit_small.png"
