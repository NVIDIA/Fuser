# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, Shard

import nvfuser_direct as nvfuser
from .benchmark_utils import get_benchmark_fns
from nvfuser_direct import DataType, FusionDefinition, CommunicatorBackend, TensorView


def row_parallel_linear_forward(
    h: int, num_devices: int, num_chunks: int
) -> FusionDefinition:
    with FusionDefinition() as fd:
        inp = fd.define_tensor(
            shape=[-1, h * 4], contiguity=True, dtype=DataType.BFloat16
        )
        weight = fd.define_tensor(
            shape=[h, h * 4], contiguity=True, dtype=DataType.BFloat16
        )
        out = fd.ops.linear(inp, weight)
        fd.add_output(out)

        mesh = nvfuser.multidevice.DeviceMesh(torch.arange(num_devices))
        for tv in (inp, weight):
            tv.set_device_mesh(mesh)

        inp.outer_split(0, num_chunks)
        inp.axis(0).parallelize(nvfuser.ParallelType.stream)
        inp.outer_split(2, mesh.size)
        inp.axis(2).parallelize(nvfuser.ParallelType.mesh_x)
        weight.outer_split(1, mesh.size)
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

    # The host IR dumped with NVFUSER_DUMP=host_ir is similar to `row_parallel_linear_forward_reference`:
    #
    # %HostIrContainer { (T0_g___bfloat[istreamIdx7{3}, ideviceIdx.x9{2}, iS8{( ceilDiv(i0, 3) )}, iS10{4}] (DeviceMesh{0 1}), T1_g___bfloat[ideviceIdx.x11{2}, iS2{2}, iS12{4}] (DeviceMesh{0 1})) -> (T2_g___bfloat[istreamIdx27{3}, rdeviceIdx.x26{2}, iS28{( ceilDiv(i0, 3) )}, iS25{2}] (DeviceMesh{0 1})) :
    #   T2_g___bfloat[istreamIdx27{3}, rdeviceIdx.x26{2}, iS28{( ceilDiv(i0, 3) )}, iS25{2}] (DeviceMesh{0 1}) = ALLOCATE(buffer=T2_g___bfloat[istreamIdx27{3}, rdeviceIdx.x26{2}, iS28{( ceilDiv(i0, 3) )}, iS25{2}] (DeviceMesh{0 1}), mem_type=global, size=( i0 * 2 ), zero_init=false, resets_to_zero=false)
    #   Stream 0x174e5c80 = GetCurrentStream()
    #   FOR i535 from 0 to 3:
    #     SetCurrentStream(Stream i535)
    #     Synchronize(Stream 0x174e5c80)
    #     T4_l___bfloat[istreamIdx37{3}, iS38{( ceilDiv(i0, 3) )}, ideviceIdx.x35{2}, iS36{4}] (DeviceMesh{0 1}) = ShardByStream(T0_g___bfloat[istreamIdx7{3}, ideviceIdx.x9{2}, iS8{( ceilDiv(i0, 3) )}, iS10{4}] (DeviceMesh{0 1}), stream_index = i535)
    #     T3_g___bfloat[istreamIdx20{3}, ideviceIdx.x22{2}rf, iS21{( ceilDiv(i0, 3) )}, iS18{2}, rS23{4}rf] (DeviceMesh{0 1}) = ALLOCATE(buffer=T3_g___bfloat[istreamIdx20{3}, ideviceIdx.x22{2}rf, iS21{( ceilDiv(i0, 3) )}, iS18{2}, rS23{4}rf] (DeviceMesh{0 1}), mem_type=global, size=( ( ceilDiv(i0, 3) ) * 12 ), zero_init=false, resets_to_zero=false)
    #     T3_g___bfloat[istreamIdx20{3}, ideviceIdx.x22{2}rf, iS21{( ceilDiv(i0, 3) )}, iS18{2}, rS23{4}rf] (DeviceMesh{0 1})
    #        = linear(T4_l___bfloat[istreamIdx37{3}, iS38{( ceilDiv(i0, 3) )}, ideviceIdx.x35{2}, iS36{4}] (DeviceMesh{0 1}),
    #                 T1_g___bfloat[ideviceIdx.x11{2}, iS2{2}, iS12{4}] (DeviceMesh{0 1})      )
    #     T5_l___bfloat[istreamIdx41{3}, iS42{( ceilDiv(i0, 3) )}, iS40{2}] (DeviceMesh{0 1}) = ShardByStream(T2_g___bfloat[istreamIdx27{3}, rdeviceIdx.x26{2}, iS28{( ceilDiv(i0, 3) )}, iS25{2}] (DeviceMesh{0 1}), stream_index = i535)
    #     Communication 272 (type=Allreduce, team=(0 1), input=T3_g___bfloat[istreamIdx20{3}, ideviceIdx.x22{2}rf, iS21{( ceilDiv(i0, 3) )}, iS18{2}, rS23{4}rf] (DeviceMesh{0 1}), output=T5_l___bfloat[istreamIdx41{3}, iS42{( ceilDiv(i0, 3) )}, iS40{2}] (DeviceMesh{0 1}), backend=NCCL)
    #     Wait(Communication 272)
    #   SetCurrentStream(Stream 0x174e5c80)
    #   FOR i535 from 0 to 3:
    #     Synchronize(Stream i535)
    # } // %HostIrContainer

    return fd


@pytest.mark.mpi
def test_row_parallel_linear_forward(multidevice_test):
    # This is a port of CollectiveBasedOverlapTest.RowParallelLinear_Forward.
    h, s, t = 2, 3, 6
    d = multidevice_test.size
    if (h * 4) % d != 0:
        pytest.skip(
            f"Row-parallel linear requires {h * 4} to be divisible by world size {d}."
        )
    assert t % s == 0

    fd = row_parallel_linear_forward(h, d, s)

    inp_ref = torch.testing.make_tensor(t, h * 4, dtype=torch.int32, device="cpu").to(
        torch.bfloat16
    )
    weight_ref = torch.testing.make_tensor(
        h, h * 4, dtype=torch.int32, device="cpu"
    ).to(torch.bfloat16)
    out_ref = torch.nn.functional.linear(inp_ref, weight_ref)

    inp = multidevice_test.shard_tensor(inp_ref, fd.fusion.inputs()[0])
    weight = multidevice_test.shard_tensor(weight_ref, fd.fusion.inputs()[1])
    # nvfuser_direct.PythonProfiler failed with host IR lowering. The main
    # reason is that HostIrContainer doesn't keep segments while SegmentProfiler
    # is still expecting data.  It's unclear to me whether we should relax
    # SegmentProfiler's assumptions or stop creating them in the first place.
    with torch.profiler.profile(record_shapes=True) as prof:
        (out,) = fd.execute([inp, weight], _enable_options=["host_ir_lowering"])
    torch.testing.assert_close(out.cpu(), out_ref)

    matmul_events = [event for event in prof.events() if event.name == "aten::mm"]
    assert len(matmul_events) == s

    m = t // s
    n = h
    k = h * 4 // d
    for event in matmul_events:
        assert event.input_shapes == [[m, k], [k, n], [m, n]]


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.parametrize("s", [1, 2, 4])
def test_row_parallel_linear_forward_benchmark(multidevice_test, benchmark, s):
    # This is a port of CollectiveBasedOverlapTest.RowParallelLinear_Forward.
    h, t = 8192, 8192
    d = multidevice_test.size
    if (h * 4) % d != 0:
        pytest.skip(
            f"Row-parallel linear requires {h * 4} to be divisible by world size {d}."
        )
    assert t % s == 0

    fd = row_parallel_linear_forward(h, d, s)

    inp_ref = torch.randn(t, h * 4, dtype=torch.bfloat16, device="cpu")
    weight_ref = torch.randn(h, h * 4, dtype=torch.bfloat16, device="cpu")

    inp = multidevice_test.shard_tensor(inp_ref, fd.fusion.inputs()[0])
    weight = multidevice_test.shard_tensor(weight_ref, fd.fusion.inputs()[1])

    warmup_fn, benchmark_fn = get_benchmark_fns(
        lambda: fd.execute([inp, weight], _enable_options=["host_ir_lowering"])
    )
    warmup_fn()
    benchmark.pedantic(benchmark_fn, rounds=5)


# The caching allocator in PyTorch can't cache buffers across streams, so we
# have to reuse streams to avoid repeated cudaMalloc. torch.cuda.Stream() is
# backed by a stream pool as well but I failed to find a way to set its size.
class StreamPool:
    def __init__(self):
        self._streams = {}

    def get(self, sid: int) -> torch.cuda.Stream:
        s = self._streams.get(sid)
        if s is None:
            s = torch.cuda.Stream()
            self._streams[sid] = s
        return s


def row_parallel_linear_forward_reference(
    inp_shard: torch.Tensor,
    weight_shard: torch.Tensor,
    num_chunks: int,
    stream_pool: StreamPool,
) -> torch.Tensor:
    out = torch.empty(
        inp_shard.size(0),
        weight_shard.size(0),
        device="cuda",
        dtype=inp_shard.dtype,
    )
    inp_chunks = inp_shard.chunk(num_chunks)
    out_chunks = out.chunk(num_chunks)

    main_stream = torch.cuda.current_stream()
    worker_streams = []
    for i, (inp_chunk, out_chunk) in enumerate(zip(inp_chunks, out_chunks)):
        worker_stream = stream_pool.get(i)
        worker_streams.append(worker_stream)
        worker_stream.wait_stream(main_stream)
        with torch.cuda.stream(worker_stream):
            torch.matmul(inp_chunk, weight_shard.T, out=out_chunk)
            work = dist.all_reduce(out_chunk, op=dist.ReduceOp.SUM, async_op=True)
            work.wait()

    for worker_stream in worker_streams:
        main_stream.wait_stream(worker_stream)

    return out


@pytest.mark.mpi
def test_row_parallel_linear_forward_reference(setup_default_process_group):
    h, s, t = 2, 3, 6
    d = dist.get_world_size()
    if (h * 4) % d != 0:
        pytest.skip(
            f"Row-parallel linear requires {h * 4} to be divisible by world size {d}."
        )
    assert t % s == 0

    torch.manual_seed(0)
    inp_ref = torch.testing.make_tensor(t, h * 4, dtype=torch.int32, device="cpu").to(
        torch.bfloat16
    )
    weight_ref = torch.testing.make_tensor(
        h, h * 4, dtype=torch.int32, device="cpu"
    ).to(torch.bfloat16)
    out_ref = torch.nn.functional.linear(inp_ref.cuda(), weight_ref.cuda()).cpu()

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])
    inp_shard = distribute_tensor(inp_ref, mesh, placements=[Shard(-1)]).to_local()
    weight_shard = distribute_tensor(
        weight_ref, mesh, placements=[Shard(-1)]
    ).to_local()
    stream_pool = StreamPool()
    out = row_parallel_linear_forward_reference(inp_shard, weight_shard, s, stream_pool)

    torch.testing.assert_close(out.cpu(), out_ref)


@pytest.mark.mpi
@pytest.mark.benchmark
def test_row_parallel_linear_forward_reference_benchmark(
    setup_default_process_group, benchmark
):
    h, s, t = 8192, 2, 8192
    d = dist.get_world_size()
    if (h * 4) % d != 0:
        pytest.skip(
            f"Row-parallel linear requires {h * 4} to be divisible by world size {d}."
        )
    assert t % s == 0

    torch.manual_seed(0)
    inp_ref = torch.randn(t, h * 4, dtype=torch.bfloat16)
    weight_ref = torch.randn(h, h * 4, dtype=torch.bfloat16)

    mesh = dist.device_mesh.init_device_mesh("cuda", [d])
    inp_shard = distribute_tensor(inp_ref, mesh, placements=[Shard(-1)]).to_local()
    weight_shard = distribute_tensor(
        weight_ref, mesh, placements=[Shard(-1)]
    ).to_local()

    stream_pool = StreamPool()
    warmup_fn, benchmark_fn = get_benchmark_fns(
        lambda: row_parallel_linear_forward_reference(
            inp_shard, weight_shard, s, stream_pool
        )
    )
    warmup_fn()
    benchmark.pedantic(benchmark_fn, rounds=5)


@pytest.mark.mpi
@pytest.mark.parametrize("backend_type", [CommunicatorBackend.nccl])
@pytest.mark.parametrize("s", [1, 8])
def test_overlap_allgather_matmul_stream_outermost(
    multidevice_test, benchmark, backend_type, s
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
    m, k, n, d = 2**10, 2**10, 2**10, multidevice_test.size
    assert m % (s * d) == 0

    os.environ["UCC_CL_BASIC_TLS"] = "nccl"
    x_unsharded = torch.testing.make_tensor(
        s, d, m // (s * d), k, dtype=torch.bfloat16, device="cpu"
    )
    x = multidevice_test.shard_tensor_1d(
        x_unsharded,
        1,
        nvfuser.multidevice.DeviceMesh(range(multidevice_test.size)),
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
    multidevice_test, benchmark, backend_type
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
    m, k, n, d = 2**10, 2**10, 2**10, multidevice_test.size
    assert m % d == 0

    os.environ["UCC_CL_BASIC_TLS"] = "nccl"
    x_unsharded = torch.testing.make_tensor(
        d, m // d, k, dtype=torch.bfloat16, device="cpu"
    )
    x = multidevice_test.shard_tensor_1d(
        x_unsharded,
        0,
        nvfuser.multidevice.DeviceMesh(range(multidevice_test.size)),
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
    params.offset_stream_indexing_by_rank = True
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
