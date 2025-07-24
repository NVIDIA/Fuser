<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Build

```
# Install Pytorch with CUDA
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# --no-build-isolation flag is required to use PyTorch with CUDA
pip install --no-build-isolation .
```

# Test

```
python test.py
```
