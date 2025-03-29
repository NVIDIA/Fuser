import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "nvFuser"
copyright = "2024, NVIDIA"
author = "NVIDIA"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_toolbox.more_autodoc.overloads",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Hide overload type signatures (from "sphinx_toolbox.more_autodoc.overload")
overloads_location = ["signature"]

# Display long function signatures with each param on a new line.
# Helps make annotated signatures more readable.
maximum_signature_line_length = 120

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
