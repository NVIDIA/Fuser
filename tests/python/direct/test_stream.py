# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owner(s): ["module: nvfuser"]

import torch
from nvfuser_direct import FusionDefinition, ParallelType, DataType


def test_matmul(nvfuser_direct_test):
    c = 3

    with FusionDefinition() as fd:
        inp = fd.define_tensor([-1, -1], contiguity=True, dtype=DataType.Float)
        w = fd.define_tensor([-1, -1], contiguity=True, dtype=DataType.Float)
        out = fd.ops.matmul(inp, w)
        fd.add_output(out)

        out.outer_split(1, c)
        out.axis(1).parallelize(ParallelType.stream)
        # With NVFUSER_DUMP=host_ir, you'll see the host IR container like the
        # following:
        #
        # %HostIrContainer { (T0_g_float[iS0{i0}, iS1{i2}], T1_g_float[istreamIdx7{3}, iS11{i2}, iS8{( ceilDiv(i4, 3) )}]) -> (T2_g_float[istreamIdx9{3}, iS4{i0}, iS10{( ceilDiv(i4, 3) )}, rS6{i2}]) :
        #   FOR i18 from 0 to 3:
        #     T2_g_float[istreamIdx9{3}, iS4{i0}, iS10{( ceilDiv(i4, 3) )}, rS6{i2}]
        #        = matmul(T0_g_float[iS0{i0}, iS1{i2}],
        #                 T1_g_float[istreamIdx7{3}, iS11{i2}, iS8{( ceilDiv(i4, 3) )}])
        # } // %HostIrContainer

    inp = torch.testing.make_tensor(5, 7, dtype=torch.float32, device="cuda")
    w = torch.testing.make_tensor(7, c * 2, dtype=torch.float32, device="cuda")
    ref = torch.matmul(inp, w)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as profile:
        (out,) = fd.execute([inp, w], _enable_options=["host_ir_lowering"])
    torch.testing.assert_close(out, ref)

    matmul_events = [event for event in profile.events() if event.name == "aten::mm"]
    assert len(matmul_events) == c
    for event in matmul_events:
        assert event.input_shapes == [[5, 7], [7, 2], [5, 2]]


def test_two_matmuls(nvfuser_direct_test):
    c = 3

    with FusionDefinition() as fd:
        inp = fd.define_tensor([-1, -1], contiguity=True, dtype=DataType.Float)
        w1 = fd.define_tensor([-1, -1], contiguity=True, dtype=DataType.Float)
        w2 = fd.define_tensor([-1, -1], contiguity=True, dtype=DataType.Float)
        out = fd.ops.matmul(inp, w1)
        out = fd.ops.matmul(out, w2)
        fd.add_output(out)

        inp.outer_split(0, c)
        inp.axis(0).parallelize(ParallelType.stream)
        # With NVFUSER_DUMP=host_ir, you'll see the host IR container like the
        # following:
        #
        # %HostIrContainer { (T0_g_float[istreamIdx12{3}, iS13{( ceilDiv(i0, 3) )}, iS1{i2}], T1_g_float[iS14{i2}, iS3{i4}], T2_g_float[iS15{i4}, iS5{i6}]) -> (T4_g_float[istreamIdx18{3}, iS19{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}]) :
        #   T4_g_float[istreamIdx18{3}, iS19{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}] = ALLOCATE(buffer=T4_g_float[istreamIdx18{3}, iS19{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}], mem_type=global, size=( i0 * i6 ), zero_init=false, resets_to_zero=false)
        #   FOR i99 from 0 to 3:
        #     T5_l_float[istreamIdx22{3}, iS23{( ceilDiv(i0, 3) )}, iS21{i2}] = ShardByStream(T0_g_float[istreamIdx12{3}, iS13{( ceilDiv(i0, 3) )}, iS1{i2}], stream_index = i99)
        #     T3_g_float[istreamIdx16{3}, iS17{( ceilDiv(i0, 3) )}, iS7{i4}, rS8{i2}]
        #        = matmul(T5_l_float[istreamIdx22{3}, iS23{( ceilDiv(i0, 3) )}, iS21{i2}],
        #                 T1_g_float[iS14{i2}, iS3{i4}])
        #     T6_l_float[istreamIdx26{3}, iS27{( ceilDiv(i0, 3) )}, iS25{i6}] = ShardByStream(T4_g_float[istreamIdx18{3}, iS19{( ceilDiv(i0, 3) )}, iS10{i6}, rS11{i4}], stream_index = i99)
        #     T6_l_float[istreamIdx26{3}, iS27{( ceilDiv(i0, 3) )}, iS25{i6}]
        #        = matmul(T3_g_float[istreamIdx16{3}, iS17{( ceilDiv(i0, 3) )}, iS7{i4}, rS8{i2}],
        #                 T2_g_float[iS15{i4}, iS5{i6}])
        # } // %HostIrContainer

    inp = torch.testing.make_tensor(c * 2, 3, dtype=torch.float32, device="cuda")
    w1 = torch.testing.make_tensor(3, 5, dtype=torch.float32, device="cuda")
    w2 = torch.testing.make_tensor(5, 3, dtype=torch.float32, device="cuda")
    ref = torch.matmul(torch.matmul(inp, w1), w2)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as profile:
        (out,) = fd.execute([inp, w1, w2], _enable_options=["host_ir_lowering"])
        torch.testing.assert_close(out, ref)

    matmul_events = [event for event in profile.events() if event.name == "aten::mm"]
    assert len(matmul_events) == c * 2
    for event in matmul_events:
        # The `m` dimension is split into `c` chunks, so each chunk will have `m == 2`.
        assert event.input_shapes[0][0] == 2
