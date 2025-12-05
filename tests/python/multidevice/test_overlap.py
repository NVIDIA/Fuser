# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import os

import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition, CommunicatorBackend, TensorView


@pytest.mark.mpi
def test_row_parallel_linear_forward(multidevice_direct_test):
    # This is a port of CollectiveBasedOverlapTest.RowParallelLinear_Forward.
    h, s, t = 2, 3, 6
    d = multidevice_direct_test.size
    if (h * 4) % d != 0:
        pytest.skip(
            f"Row-parallel linear requires {h * 4} to be divisible by world size {d}."
        )
    assert t % s == 0

    mesh = nvfuser.multidevice.DeviceMesh(range(d))

    with FusionDefinition() as fd:
        inp = fd.define_tensor(
            shape=[-1, h * 4], contiguity=True, dtype=DataType.BFloat16
        )
        weight = fd.define_tensor(
            shape=[h, h * 4], contiguity=True, dtype=DataType.BFloat16
        )
        out = fd.ops.linear(inp, weight)
        fd.add_output(out)

        for tv in (inp, weight):
            tv.set_device_mesh(mesh)

        inp.split(0, s, inner_split=False)
        inp.axis(0).parallelize(nvfuser.ParallelType.stream)
        inp.split(2, d, inner_split=False)
        inp.axis(2).parallelize(nvfuser.ParallelType.mesh_x)
        weight.split(1, d, inner_split=False)
        weight.axis(1).parallelize(nvfuser.ParallelType.mesh_x)

    # Expected pre-segmentation IR:
    #
    #   [t, 4h]                                 [h, 4h]
    #   /\  /\                                      /\.
    #  s*  d                                       d
    #                      |
    #                      | linear
    #                      |
    #                          r{4h}
    #                          /  \.
    #                 [t, h, d, r{4h/d}]
    #                 /\.
    #                s
    #                     |
    #                     | sum
    #                     |
    #                  [t, h, r{d}]
    #                  /\.
    #                 s*

    # Expected host IR:
    #
    # %HostIrContainer { (T0_g___bfloat[istreamIdx7{3}, ideviceIdx.x9{2}, iS8{( ceilDiv(i0, 3) )}, iS10{4}] (DeviceMesh{0 1}), T1_g___bfloat[ideviceIdx.x11{2}, iS2{2}, iS12{4}] (DeviceMesh{0 1})) -> (T2_g___bfloat[istreamIdx27{3}, rdeviceIdx.x26{2}, iS28{( ceilDiv(i0, 3) )}, iS25{2}] (DeviceMesh{0 1})) :
    #   T2_g___bfloat[istreamIdx27{3}, rdeviceIdx.x26{2}, iS28{( ceilDiv(i0, 3) )}, iS25{2}] (DeviceMesh{0 1}) = ALLOCATE(buffer=T2_g___bfloat[istreamIdx27{3}, rdeviceIdx.x26{2}, iS28{( ceilDiv(i0, 3) )}, iS25{2}] (DeviceMesh{0 1}), mem_type=global, size=( i0 * 2 ), zero_init=false, resets_to_zero=false)
    #   FOR i535 from 0 to 3:
    #     T4_l___bfloat[istreamIdx31{3}, ideviceIdx.x33{2}, iS32{( ceilDiv(i0, 3) )}, iS34{4}] (DeviceMesh{0 1}) = ShardByStream(T0_g___bfloat[istreamIdx7{3}, ideviceIdx.x9{2}, iS8{( ceilDiv(i0, 3) )}, iS10{4}] (DeviceMesh{0 1}), stream_index = i535)
    #     T3_g___bfloat[istreamIdx20{3}, ideviceIdx.x22{2}rf, iS21{( ceilDiv(i0, 3) )}, iS18{2}, rS23{4}rf] (DeviceMesh{0 1})
    #        = linear(T4_l___bfloat[istreamIdx31{3}, ideviceIdx.x33{2}, iS32{( ceilDiv(i0, 3) )}, iS34{4}] (DeviceMesh{0 1}),
    #                 T1_g___bfloat[ideviceIdx.x11{2}, iS2{2}, iS12{4}] (DeviceMesh{0 1})      )
    #     T5_l___bfloat[istreamIdx37{3}, iS38{( ceilDiv(i0, 3) )}, iS36{2}] (DeviceMesh{0 1}) = ShardByStream(T2_g___bfloat[istreamIdx27{3}, rdeviceIdx.x26{2}, iS28{( ceilDiv(i0, 3) )}, iS25{2}] (DeviceMesh{0 1}), stream_index = i535)
    #     Communication 250 (type=Allreduce, team=(0 1), input=T3_g___bfloat[istreamIdx20{3}, ideviceIdx.x22{2}rf, iS21{( ceilDiv(i0, 3) )}, iS18{2}, rS23{4}rf] (DeviceMesh{0 1}), output=T5_l___bfloat[istreamIdx37{3}, iS38{( ceilDiv(i0, 3) )}, iS36{2}] (DeviceMesh{0 1}), backend=NCCL)
    #     Wait Communication 250
    # } // %HostIrContainer

    inp_ref = torch.randint(-2, 3, (t, h * 4), dtype=torch.int32).to(torch.bfloat16)
    weight_ref = torch.randint(-2, 3, (h, h * 4), dtype=torch.int32).to(torch.bfloat16)
    out_ref = torch.nn.functional.linear(inp_ref, weight_ref)

    inp = (multidevice_direct_test.shard_tensor(inp_ref, -1, mesh),)
    weight = (multidevice_direct_test.shard_tensor(weight_ref, -1, mesh),)
    (out,) = fd.execute([inp, weight], _enable_options=["host_ir_lowering"])
    torch.testing.assert_close(out.cpu(), out_ref)

    # Collect CUDA kernels after a warmup run to exclude autotuning.
    # nvfuser_direct.PythonProfiler failed with host IR lowering. The main
    # reason is that HostIrContainer doesn't keep segments while SegmentProfiler
    # is still expecting data.  It's unclear to me whether we should relax
    # SegmentProfiler's assumptions or stop creating them in the first place.
    with torch.profiler.profile(record_shapes=True) as prof:
        (out,) = fd.execute([inp, weight], _enable_options=["host_ir_lowering"])

    matmul_events = [event for event in prof.events() if event.name == "aten::mm"]
    assert len(matmul_events) == s

    m = t // s
    n = h
    k = h * 4 // d
    for event in matmul_events:
        assert event.input_shapes == [[m, k], [k, n], [m, n]]


@pytest.mark.mpi
@pytest.mark.parametrize("backend_type", [CommunicatorBackend.nccl])
@pytest.mark.parametrize("s", [1, 8])
def test_overlap_allgather_matmul_stream_outermost(
    multidevice_direct_test, benchmark, backend_type, s
):
    def fusion_definition(fd, m, k, n, s, d) -> list[TensorView]:
        x = fd.define_tensor(
            shape=[s, d, m // (s * d), k], contiguity=True, dtype=DataType.BFloat16
        )
        weight = fd.define_tensor(
            shape=[n, k], contiguity=True, dtype=DataType.BFloat16
        )
        bias = fd.define_tensor(shape=[n], contiguity=True, dtype=DataType.BFloat16)

        # [s, d, m//(s*d), n]
        out = fd.ops.linear(x, weight, bias)

        fd.add_output(out)
        return [x, weight, bias, out]

    def multidevice_schedule(fd, tensors, num_devices) -> None:
        mesh = nvfuser.multidevice.DeviceMesh(range(num_devices))
        for tv in tensors:
            tv.set_device_mesh(mesh)

        x, weight, bias, out = tensors
        x.axis(1).parallelize(nvfuser.ParallelType.mesh_x)
        out.axis(0).parallelize(nvfuser.ParallelType.stream)

    N_WARMUPS, N_ITERATIONS = 5, 25
    m, k, n, d = 2**10, 2**10, 2**10, multidevice_direct_test.size
    assert m % (s * d) == 0

    os.environ["UCC_CL_BASIC_TLS"] = "nccl"

    torch.cuda.set_device(multidevice_direct_test.local_rank)
    x_unsharded = torch.testing.make_tensor(
        s, d, m // (s * d), k, dtype=torch.bfloat16, device="cpu"
    )
    x = multidevice_direct_test.shard_tensor(
        x_unsharded,
        1,
        nvfuser.multidevice.DeviceMesh(range(multidevice_direct_test.size)),
    )
    weight = torch.testing.make_tensor(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.testing.make_tensor(n, dtype=torch.bfloat16, device="cuda")
    ins = [x, weight, bias]
    out_ref = torch.nn.functional.linear(x_unsharded, weight.cpu(), bias.cpu())

    with FusionDefinition() as fd:
        tensors = fusion_definition(fd, m, k, n, s, d)
        multidevice_schedule(fd, tensors, d)

    params = nvfuser.multidevice.MultiDeviceExecutorParams()
    params.backend_type = backend_type
    multidevice_executor = nvfuser.multidevice.MultiDeviceExecutor(fd.fusion, params)

    # warmup
    for _ in range(N_WARMUPS):
        outputs = multidevice_executor.run(ins)
        out = outputs[0].cpu()
        assert out.dtype == torch.bfloat16
        assert out.shape == torch.Size([s, d, m // (s * d), n])
        torch.testing.assert_close(out, out_ref, rtol=1e-1, atol=1e-1)

    # benchmark
    benchmark.pedantic(lambda: multidevice_executor.run(ins), rounds=N_ITERATIONS)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "backend_type", [CommunicatorBackend.nccl, CommunicatorBackend.cuda]
)
def test_overlap_allgather_matmul_shard_outermost(
    multidevice_direct_test, benchmark, backend_type
):
    def fusion_definition(fd, m, k, n, d) -> list[TensorView]:
        x = fd.define_tensor(
            shape=[d, m // d, k], contiguity=True, dtype=DataType.BFloat16
        )
        weight = fd.define_tensor(
            shape=[n, k], contiguity=True, dtype=DataType.BFloat16
        )
        bias = fd.define_tensor(shape=[n], contiguity=True, dtype=DataType.BFloat16)

        # [d, m//d, n]
        out = fd.ops.linear(x, weight, bias)

        fd.add_output(out)
        return [x, weight, bias, out]

    def multidevice_schedule(fd, tensors, num_devices) -> None:
        mesh = nvfuser.multidevice.DeviceMesh(range(num_devices))
        for tv in tensors:
            tv.set_device_mesh(mesh)

        x, weight, bias, out = tensors
        x.axis(0).parallelize(nvfuser.ParallelType.mesh_x)
        out.axis(0).parallelize(nvfuser.ParallelType.stream)

    N_WARMUPS, N_ITERATIONS = 5, 25
    m, k, n, d = 2**10, 2**10, 2**10, multidevice_direct_test.size
    assert m % d == 0

    os.environ["UCC_CL_BASIC_TLS"] = "nccl"

    torch.cuda.set_device(multidevice_direct_test.local_rank)
    x_unsharded = torch.testing.make_tensor(
        d, m // d, k, dtype=torch.bfloat16, device="cpu"
    )
    x = multidevice_direct_test.shard_tensor(
        x_unsharded,
        0,
        nvfuser.multidevice.DeviceMesh(range(multidevice_direct_test.size)),
    )
    weight = torch.testing.make_tensor(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.testing.make_tensor(n, dtype=torch.bfloat16, device="cuda")
    ins = [x, weight, bias]
    out_ref = torch.nn.functional.linear(x_unsharded, weight.cpu(), bias.cpu())

    with FusionDefinition() as fd:
        tensors = fusion_definition(fd, m, k, n, d)
        multidevice_schedule(fd, tensors, d)

    params = nvfuser.multidevice.MultiDeviceExecutorParams()
    params.backend_type = backend_type
    params.use_allocation_cache = True
    multidevice_executor = nvfuser.multidevice.MultiDeviceExecutor(fd.fusion, params)

    # warmup
    for _ in range(N_WARMUPS):
        outputs = multidevice_executor.run(ins)
        out = outputs[0].cpu()
        assert out.dtype == torch.bfloat16
        assert out.shape == torch.Size([d, m // d, n])
        torch.testing.assert_close(out, out_ref, rtol=1e-1, atol=1e-1)

    # benchmark
    benchmark.pedantic(lambda: multidevice_executor.run(ins), rounds=N_ITERATIONS)
