<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Build

```
mkdir build
cd build
CMAKE_PREFIX_PATH=$(python -c 'import nvfuser_common.utils; import torch.utils; print(nvfuser_common.utils.cmake_prefix_path, torch.utils.cmake_prefix_path, sep=";")')
cmake -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -G Ninja ..
cmake --build .
```

# Test

```
./sinh_example
```
