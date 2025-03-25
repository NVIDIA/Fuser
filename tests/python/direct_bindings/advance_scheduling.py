# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser import (
    direct,
    MemoryType,
    ParallelType,
    LoadStoreOpType,
    DirectFusionDefinition,
)  # noqa: F401


def register_sharing_circular_buffering_pointwise():
    # Parameters
    number_of_stages = 4
    prefetch_distance = 1
    tensor_outer_dim = 128
    tensor_inner_dim = 1024
    # with bdimx = 256, bdimy = 1,
    # after warp specialization, bdimy = 2,
    # kernel has 256 * 2 = 512 threads, each can use 128 registers.
    # With register sharing, adjust to [64, 192]
    bulk_inner_dim = 256

    # Inputs
    t0 = torch.randn(
        tensor_outer_dim,
        tensor_inner_dim,
        dtype=torch.float,
        device=torch.device("cuda:0"),
    )
    t1 = torch.randn(
        tensor_outer_dim,
        tensor_inner_dim,
        dtype=torch.float,
        device=torch.device("cuda:0"),
    )
    inputs = [t0, t1]

    fd = DirectFusionDefinition()
    tv0 = fd.from_pytorch(t0)
    tv1 = fd.from_pytorch(t1)
    tv2 = fd.ops.add(tv0, tv1)
    fd.add_output(tv2)

    # Use TMA to load TV0 into shared memory
    tv3 = tv0.cache_after(LoadStoreOpType.tma)
    tv3.set_memory_type(MemoryType.shared)

    tv4 = tv1.cache_after(LoadStoreOpType.tma)
    tv4.set_memory_type(MemoryType.shared)

    reference = tv2

    # [M, N] -> [M, N/bid, bid]
    reference.split(-1, bulk_inner_dim)
    fd.schedule.transform_like(reference)

    tv3.axis(0).parallelize(ParallelType.grid_x)
    tv4.axis(0).parallelize(ParallelType.grid_x)

    # Set computeAt position
    fd.schedule.inline_at(tv2, pos=2)

    # Circular Buffer with TMA loads
    tv3.axis(2).parallelize(ParallelType.tma)
    tv4.axis(2).parallelize(ParallelType.tma)
    # circular_buffer_type = WarpSpecialized(ParallelType.TIDy, (64, 192))
    fd.schedule.warp_specialize(
        tv3, number_of_stages, prefetch_distance, ParallelType.block_y
    )
    fd.schedule.warp_specialize(
        tv4, number_of_stages, prefetch_distance, ParallelType.block_y
    )

    # Split reference to parallelize TMA tile
    reference.split(-1, bulk_inner_dim)
    reference.axis(0).parallelize(ParallelType.grid_x)
    reference.axis(-1).parallelize(ParallelType.block_x)

    outputs = fd.execute(inputs, auto_schedule=False)
    assert torch.allclose(outputs[0], t0 + t1)

register_sharing_circular_buffering_pointwise()
