# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
import pytest
from nvfuser import (
    direct,
    MemoryType,
    ParallelType,
    SchedulerType,
    DirectFusionDefinition,
    DataType,
    LoadStoreOpType,
)


def test_pointwise_manual():
    # Create test inputs
    inputs = [
        torch.randn(4, 8, device="cuda"),
        torch.randn(4, 8, device="cuda"),
    ]

    with DirectFusionDefinition() as fd:
        tv0 = fd.from_pytorch(inputs[0])
        tv1 = fd.from_pytorch(inputs[1])
        tv2 = fd.ops.add(tv0, tv1)
        fd.add_output(tv2)

    # Verify pointwise scheduling is possible
    status, msg = fd.schedule.can_schedule(fd.fusion, inputs, SchedulerType.pointwise)
    assert status
    assert msg == ""

    # Manual scheduling
    tv3 = tv0.cache_after()
    tv4 = tv1.cache_after()
    tv5 = tv2.cache_before()

    # Set memory types
    tv3.set_memory_type(MemoryType.shared)
    tv4.set_memory_type(MemoryType.shared)
    tv5.set_memory_type(MemoryType.shared)

    # Transform and parallelize
    selected_tensors = [tv2, tv3, tv4, tv5]
    reference_tv = tv2
    reference_tv.merge(axis=0)
    reference_tv.split(axis=0, factor=128)
    fd.schedule.transform_like(reference_tv)

    # Parallelize axes
    reference_tv.axis(0).parallelize(ParallelType.grid_x)
    reference_tv.axis(1).parallelize(ParallelType.block_x)
    fd.schedule.parallelize_like(reference_tv)

    # Inline most operations
    fd.schedule.inline_most()

    # Execute fusion and verify output
    outputs = fd.execute(inputs, auto_schedule=False)
    expected = inputs[0] + inputs[1]
    torch.testing.assert_close(outputs[0], expected)

    scheduled_fusion_ir = """Inputs:
  T0_g_float[iS28{( ceilDiv(( i0 * i1 ), 128) )}, iS29{128}]
  T1_g_float[iS22{( ceilDiv(( i3 * i4 ), 128) )}, iS23{128}]
Outputs:
  T2_g_float[iblockIdx.x13{( ceilDiv(( i0 * i1 ), 128) )}, ithreadIdx.x14{128}] ca_pos( 2 ) produce_pos( 2 )

%kernel_math {
T3_s_float[iblockIdx.x25{( ceilDiv(( i0 * i1 ), 128) )}, ithreadIdx.x26{128}] ca_pos( 2 )
   = Set( T0_g_float[iS28{( ceilDiv(( i0 * i1 ), 128) )}, iS29{128}], cache_op=Streaming )
T4_s_float[iblockIdx.x19{( ceilDiv(( i3 * i4 ), 128) )}, ithreadIdx.x20{128}] ca_pos( 2 )
   = Set( T1_g_float[iS22{( ceilDiv(( i3 * i4 ), 128) )}, iS23{128}], cache_op=Streaming )
T5_s_float[iblockIdx.x16{( ceilDiv(( i0 * i1 ), 128) )}, ithreadIdx.x17{128}] ca_pos( 2 ) produce_pos( 2 )
   = T3_s_float[iblockIdx.x25{( ceilDiv(( i0 * i1 ), 128) )}, ithreadIdx.x26{128}] ca_pos( 2 )
   + T4_s_float[iblockIdx.x19{( ceilDiv(( i3 * i4 ), 128) )}, ithreadIdx.x20{128}] ca_pos( 2 );
T2_g_float[iblockIdx.x13{( ceilDiv(( i0 * i1 ), 128) )}, ithreadIdx.x14{128}] ca_pos( 2 ) produce_pos( 2 )
   = Set( T5_s_float[iblockIdx.x16{( ceilDiv(( i0 * i1 ), 128) )}, ithreadIdx.x17{128}] ca_pos( 2 ) produce_pos( 2 ), cache_op=Streaming )
} // %kernel_math \n\n"""
    assert fd.fusion.print_math() == scheduled_fusion_ir


def test_pointwise_auto():
    with DirectFusionDefinition() as fd:
        tv0 = fd.define_tensor(
            shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False
        )
        tv1 = fd.define_tensor(
            shape=[-1, -1], contiguity=[True, True], dtype=DataType.Float, is_cpu=False
        )
        tv2 = fd.ops.add(tv0, tv1)
        fd.add_output(tv2)

    # Create test inputs
    inputs = [
        torch.randn(4, 8, device="cuda"),
        torch.randn(4, 8, device="cuda"),
    ]

    # Verify pointwise scheduling is possible
    status, msg = fd.schedule.can_schedule(fd.fusion, inputs, SchedulerType.pointwise)
    assert status
    assert msg == ""

    # Get compatible schedulers
    schedulers = fd.schedule.find_compatible_schedulers(fd.fusion, inputs)
    assert SchedulerType.pointwise in schedulers

    # Compute heuristics
    params = fd.schedule.compute_heuristics(fd.fusion, inputs, SchedulerType.pointwise)
    assert params is not None

    # Auto schedule the fusion
    fd.schedule.auto_schedule(fd.fusion, inputs, SchedulerType.pointwise, params)

    # Execute fusion and verify output
    outputs = fd.execute(inputs, auto_schedule=False)
    expected = inputs[0] + inputs[1]
    torch.testing.assert_close(outputs[0], expected)


def test_register_sharing_circular_buffering_pointwise():
    # Parameters
    number_of_stages = 4
    prefetch_distance = 1
    tensor_outer_dim = 128
    tensor_inner_dim = 1024
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

    with DirectFusionDefinition() as fd:
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
    warp_specialization_if_stmt = "if ((((nvfuser_index_t)threadIdx.y) == 1LL)) {"
    assert warp_specialization_if_stmt in fd.fusion.print_kernel()
    assert torch.allclose(outputs[0], t0 + t1)
