# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import os
import sys
import datetime
import pathlib

sys.path.insert(0, os.path.abspath("../.."))

project = "nvFuser"
author = "NVIDIA CORPORATION & AFFILIATES"

# Copyright statement
release_year = 2023
current_year = datetime.date.today().year
if current_year == release_year:
    copyright_year = release_year
else:
    copyright_year = str(release_year) + "-" + str(current_year)
copyright = f"{copyright_year}, NVIDIA CORPORATION & AFFILIATES. All rights reserved."

# Version
root_path = pathlib.Path(__file__).resolve().parents[3]
print(root_path)
with open(root_path / "python" / "version.txt", "r") as f:
    _raw_version = f.readline().strip()
version = str(_raw_version)
release = str(_raw_version)

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

pygments_style = "sphinx"
templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_show_sphinx = False
html_static_path = ["_static"]
html_sidebars = {"**": ["globaltoc.html", "searchbox.html"]}
html_css_files = [
    "css/nvidia_font.css",
    "css/nvidia_footer.css",
]
html_theme_options = {
    "collapse_navigation": False,
    "logo_only": False,
    "version_selector": False,
    "language_selector": False,
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
