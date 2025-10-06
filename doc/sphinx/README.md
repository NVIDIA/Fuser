<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

### Building documentation

NvFuser uses [Sphinx](https://www.sphinx-doc.org/en/master/) for documentation.

Building the documentation requires additional packages.  The following commands will install them.
```bash
sudo apt-get install pandoc
pip install --upgrade Sphinx furo pandoc myst-parser sphinx-copybutton nbsphinx nbsphinx-link sphinx-inline-tabs
```

Documentation can then be built via the following commands.
```bash
cd docs/sphinx
make html
```
