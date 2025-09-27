# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "nvFuser"
copyright = "2025, NVIDIA"
author = "NVIDIA"

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_toolbox.more_autodoc.overloads",
]

myst_enable_extensions = [
    "dollarmath",
    "html_image",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_sidebars = {"**": ["globaltoc.html", "searchbox.html"]}
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# Hide overload type signatures (from "sphinx_toolbox.more_autodoc.overload")
overloads_location = ["signature"]

# Display long function signatures with each param on a new line.
# Helps make annotated signatures more readable.
maximum_signature_line_length = 120

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
