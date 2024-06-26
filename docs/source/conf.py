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
# from sphinx.ext.napoleon.docstring import GoogleDocstring

import os
import sys
import operator
import inspect

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'SuperGrad'
copyright = '2024, IQUBIT'
author = 'IQUBIT'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    "sphinx_remove_toctrees",
    'sphinx_copybutton',
    'sphinx_design',
    'doi_role',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
}

suppress_warnings = [
    'ref.citation',  # Many duplicated citations in numpy/scipy docstrings.
    'ref.footnote',  # Many unreferenced footnotes in numpy/scipy docstrings
]

nbsphinx_allow_errors = True

autosectionlabel_prefix_document = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

autosummary_generate = True
# napoleon options
napoleon_google_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Remove auto-generated API docs from sidebars. They take too long to build.
remove_from_toctrees = ["*/_autosummary/*"]

# List of qubit_subsets, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This qubit_subset also affects html_static_path and html_extra_path.
exclude_qubit_subsets = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_theme_options = {
    'show_toc_level': 2,
    'repository_url': 'https://github.com/iqubit-org/supergrad',
    'use_repository_button': True,  # add a "link to repository" button
    'navigation_with_keys': False,
}

# Customize code links via sphinx.ext.linkcode


def linkcode_resolve(domain, info):
    import supergrad

    if domain != 'py':
        return None
    if not info['module']:
        return None
    if not info['fullname']:
        return None
    try:
        mod = sys.modules.get(info['module'])
        obj = operator.attrgetter(info['fullname'])(mod)
        if isinstance(obj, property):
            obj = obj.fget
        while hasattr(obj, '__wrapped__'):  # decorated functions
            obj = obj.__wrapped__
        filename = inspect.getsourcefile(obj)
        source, linenum = inspect.getsourcelines(obj)
    except:
        return None
    filename = os.path.relpath(filename,
                               start=os.path.dirname(supergrad.__file__))
    lines = f"#L{linenum}-L{linenum + len(source)}" if linenum else ""
    return f"https://github.com/iqubit-org/supergrad/blob/main/supergrad/{filename}{lines}"
