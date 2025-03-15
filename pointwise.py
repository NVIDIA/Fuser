# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import fusion
from nvfuser import MemoryType, ParallelType

f = fusion.Fusion()
fg = fusion.FusionGuard(f)

tv0 = fusion.TensorViewBuilder().num_dims(2).contiguity(True).build()
tv1 = fusion.TensorViewBuilder().num_dims(2).contiguity(True).build()
f.add_input(tv0)
f.add_input(tv1)

tv2 = fusion.ops.add(tv0, tv1)
f.add_output(tv2)

tv3 = tv0.cache_after()
tv4 = tv1.cache_after()
tv5 = tv2.cache_before()
tv3.set_memory_type(MemoryType.shared)
tv4.set_memory_type(MemoryType.shared)
tv5.set_memory_type(MemoryType.shared)

all_tensors = [tv3, tv4, tv5, tv2]
for tensor in all_tensors:
    tensor.merge(axis=0)
    tensor.split(axis=0, factor=128)
    tensor.axis(0).parallelize(ParallelType.grid_x)
    tensor.axis(1).parallelize(ParallelType.block_x)

inputs = [
    torch.ones(4, 8, device="cuda"),
    torch.ones(4, 8, device="cuda"),
]
fec = fusion.FusionExecutorCache(f, auto_schedule=False)
print(fec.execute(inputs))
