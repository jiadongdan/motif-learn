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
import os
import shutil
import sys
sys.path.insert(0, os.path.abspath('..\..'))


# -- Project information -----------------------------------------------------

project = 'stempy'
copyright = '2020, Dan Jiadong'
author = 'Dan Jiadong'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Extension from here: https://github.com/sphinx-gallery/sphinx-gallery/blob/master/doc/conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
]

# graphviz dot setting, from matplotlib conf.py
graphviz_dot = shutil.which('dot')
graphviz_output_format = 'svg'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# generate autosummary even if no references
autosummary_generate = True

# napoleon configuration
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False

todo_include_todos = True


# sphinx_copybutton configuration
copybutton_prompt_text = ">>> "
copybutton_remove_prompts = True
copybutton_only_copy_prompt_lines = True


# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
#pygments_style = 'solarized-dark'
#pygments_style = 'solarized-light'
#pygments_style = 'monokai'


sphinx_gallery_conf = {
    'examples_dirs': ['../examples'],
    'gallery_dirs': ['auto_examples'],
    # avoid generating too many cross links
    'inspect_global_variables': False,
    'remove_config_comments': True,
    'download_all_examples': False,
}

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']