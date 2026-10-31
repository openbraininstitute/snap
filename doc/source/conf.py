"""Configuration file for the Sphinx documentation builder."""
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import importlib
import re
import subprocess

# -- Project information -----------------------------------------------------

project = "Blue Brain SNAP"
author = "Blue Brain Project, EPFL"

release = importlib.metadata.distribution("bluepysnap").version
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "obi_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# ensure a useful title is used
html_title = "Blue Brain SNAP"

# hide source links
html_show_sourcelink = False

# autodoc settings
autodoc_default_options = {
    "members": True,
}

autoclass_content = "both"

# show the type annotations in the parameter and return descriptions, not in the signatures
autodoc_typehints = "description"

autodoc_mock_imports = ["libsonata"]

# autosummary settings
autosummary_generate = True

suppress_warnings = [
    "autosectionlabel.*",
    "ref.python",  # duplicate aliases exported by the package API
]

# generate the link to the notebooks on GitHub
_base_url = "https://github.com/openbraininstitute/snap"
_git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
extlinks = {
    "notebooks_source": (
        f"{_base_url}/blob/{_git_commit}/doc/source/notebooks/%s.ipynb",
        "Notebook: %s",
    )
}


# Sphinx shows typing.Self literally: replace it with the class that the method is documented on,
# which is what Self means (e.g. CircuitNodeIds.filter_population returns a CircuitNodeIds).
_SELF_RE = re.compile(r"(?<![\w.])~?(?:typing\.)?Self\b")


def _replace_self_in_typehints(app, what, name, obj, options, signature, return_annotation):
    annotations = app.env.current_document.autodoc_annotations.get(name)
    if what == "method" and annotations:
        class_name = "~" + name.rsplit(".", 1)[0]
        for key, value in annotations.items():
            annotations[key] = _SELF_RE.sub(class_name, value)


def setup(app):
    """Connect the event handlers."""
    # run after sphinx.ext.autodoc.typehints, which records the type hints with priority 500
    app.connect("autodoc-process-signature", _replace_self_in_typehints, priority=600)
