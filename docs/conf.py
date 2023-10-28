# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))


# -- Project information -----------------------------------------------------

project = 'OSmOSE toolkit'
copyright = '2023, OSmOSE team'
author = 'OSmOSE team'


add_module_names = False

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [  'sphinx_gallery.gen_gallery',
		'myst_parser',
	        'sphinx.ext.autodoc',
               'sphinx.ext.doctest',
               'sphinx.ext.intersphinx',
		'sphinx.ext.autosectionlabel',
	      'sphinx_copybutton'
               #'sphinx_design',
              ]

from sphinx_gallery.sorting import ExplicitOrder # NOQA
from sphinx_gallery.sorting import ExampleTitleSortKey  # NOQA
sphinx_gallery_conf = {
     'examples_dirs': ['../tutorials','../advanced_workflows'],   # path to your example scripts
     'gallery_dirs': ['gallery_tutorials','gallery_advanced_workflows'],  
     'filename_pattern': '^((?!skip_).)*$',
     'subsection_order': ExplicitOrder(['../tutorials/Dataset','../tutorials/Spectrogram','../tutorials/Auxiliary','../advanced_workflows/Soundscape','../advanced_workflows/Weather']),
     'within_subsection_order': ExampleTitleSortKey,	
}
    


source_suffix = ['.rst', '.md']



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']




# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

#html_theme = "alabaster"
import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': 'white',
    'display_version': True,
}
# html_theme_options = {'collapse_navigation': False, 'logo_only': False}
using_rtd_theme = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_logo = '_static/osmose_logo.png'








# nb_execution_mode = "cache"
nb_execution_mode = "auto"

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]


autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "member-order": "bysource",
}

default_role = "py:obj"
