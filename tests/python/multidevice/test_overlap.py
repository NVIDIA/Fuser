# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import os

import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition, CommunicatorBackend, TensorView


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
@pytest.mark.parametrize("backend_type", [CommunicatorBackend.nccl, CommunicatorBackend.cuda])
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
