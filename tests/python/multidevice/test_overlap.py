# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import os

import nvfuser
from nvfuser import DataType, FusionDefinition, CommunicatorBackend


class OverlapAGMatmulStreamOutermost(FusionDefinition):
    def __init__(self, m, k, n, s, num_devices, communication_backend):
        super().__init__(
            use_multidevice_executor=True, backend_type=communication_backend
        )
        self.m = m
        self.k = k
        self.n = n
        self.s = s
        self._num_devices = num_devices

    def definition(self) -> None:
        m, k, n, s, d = (
            self.m,
            self.k,
            self.n,
            self.s,
            self._num_devices,
        )
        self.x = self.define_tensor(
            shape=[s, d, m // (s * d), k], contiguity=True, dtype=DataType.BFloat16
        )
        self.weight = self.define_tensor(
            shape=[n, k], contiguity=True, dtype=DataType.BFloat16
        )
        self.bias = self.define_tensor(
            shape=[n], contiguity=True, dtype=DataType.BFloat16
        )

        self.out = self.ops.linear(
            self.x, self.weight, self.bias
        )  # [s, d, m//(s*d), n]

        self.add_output(self.out)

    def multidevice_schedule(self):
        mesh = nvfuser.DeviceMesh(range(self._num_devices))
        for tv in [
            self.x,
            self.weight,
            self.bias,
            self.out,
        ]:
            self.sched._set_device_mesh(tv, mesh)

        self.sched.parallelize(self.x, 1, nvfuser.ParallelType.mesh_x)
        self.sched.parallelize(self.out, 0, nvfuser.ParallelType.stream)


@pytest.mark.mpi
@pytest.mark.parametrize("backend_type", [CommunicatorBackend.nccl])
@pytest.mark.parametrize("s", [1, 8])
def test_overlap_allgather_matmul_stream_outermost(
    multidevice_test, benchmark, backend_type, s
):
    N_WARMUPS, N_ITERATIONS = 5, 25
    m, k, n, d = 2**10, 2**10, 2**10, multidevice_test.size
    assert m % (s * d) == 0

    os.environ["UCC_CL_BASIC_TLS"] = "nccl"

    torch.cuda.set_device(multidevice_test.local_rank)
    x_unsharded = torch.testing.make_tensor(
        s, d, m // (s * d), k, dtype=torch.bfloat16, device="cpu"
    )
    x = multidevice_test.shard_tensor(
        x_unsharded, 1, nvfuser.DeviceMesh(range(multidevice_test.size))
    )
    weight = torch.testing.make_tensor(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.testing.make_tensor(n, dtype=torch.bfloat16, device="cuda")
    ins = [x, weight, bias]
    out_ref = torch.nn.functional.linear(x_unsharded, weight.cpu(), bias.cpu())

    fd = OverlapAGMatmulStreamOutermost(m, k, n, s, d, backend_type)

    # warmup
    for _ in range(N_WARMUPS):
        outputs, _ = fd.execute(ins)
        out = outputs[0].cpu()
        assert out.dtype == torch.bfloat16
        assert out.shape == torch.Size([s, d, m // (s * d), n])
        torch.testing.assert_close(out, out_ref, rtol=1e-1, atol=1e-1)

    # benchmark
    benchmark.pedantic(lambda: fd.execute(ins), rounds=N_ITERATIONS)
