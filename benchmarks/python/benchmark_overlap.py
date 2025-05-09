# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import torch

import nvfuser
from nvfuser import DataType, FusionDefinition, CommunicatorBackend


def benchmark_cuda_events(fn, inputs, num_warmup=1, num_iterations=10):
    """
    Lightweight benchmark using CUDA events for timing.
    Returns average time per iteration in milliseconds.
    """
    # Warmup
    for _ in range(num_warmup):
        fn(*inputs)

    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark
    torch.cuda.synchronize()  # Ensure all previous operations are completed
    start_event.record()

    for _ in range(num_iterations):
        fn(*inputs)

    end_event.record()
    torch.cuda.synchronize()  # Ensure all operations are completed

    # Calculate average time per iteration
    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / num_iterations


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
        m, k, n, s, d = self.m, self.k, self.n, self.s, self._num_devices
        self.x = self.define_tensor(
            shape=[s, d, m // (s * d), k], contiguity=True, dtype=DataType.BFloat16
        )
        self.weight = self.define_tensor(
            shape=[n, k], contiguity=True, dtype=DataType.BFloat16
        )
        self.bias = self.define_tensor(
            shape=[n], contiguity=True, dtype=DataType.BFloat16
        )
        self.out = self.ops.linear(self.x, self.weight, self.bias)
        self.add_output(self.out)

    def multidevice_schedule(self):
        mesh = nvfuser.DeviceMesh(range(self._num_devices))
        for tv in [self.x, self.weight, self.bias, self.out]:
            self.sched._set_device_mesh(tv, mesh)
        self.sched.parallelize(self.x, 1, nvfuser.ParallelType.mesh_x)
        self.sched.parallelize(self.out, 0, nvfuser.ParallelType.stream)


class MultideviceBenchmark:
    def __init__(self):
        self._communicator = nvfuser.Communicator.instance()
        torch.manual_seed(0)

    @property
    def communicator(self):
        return self._communicator

    @property
    def size(self):
        return self._communicator.size()

    @property
    def rank(self):
        return self._communicator.rank()

    @property
    def local_size(self):
        return self._communicator.local_size()

    @property
    def local_rank(self):
        return self._communicator.local_rank()

    def shard_tensor(
        self, t: torch.Tensor, dim: int, mesh: nvfuser.DeviceMesh
    ) -> torch.Tensor:
        assert t.is_cpu, (
            "This is not strictly required but it's a general good practice "
            "for unit tests to create unsharded data on CPU to reduce GPU "
            "memory footprint."
        )
        return mesh.shard_tensor(t, dim, self.rank).cuda(self.rank)


@pytest.fixture
def multidevice_benchmark():
    return MultideviceBenchmark()


@pytest.mark.mpi
@pytest.mark.parametrize(
    "backend_type", [CommunicatorBackend.nccl, CommunicatorBackend.ucc]
)
@pytest.mark.parametrize("s", [1, 8])
def test_overlap_allgather_matmul_stream_outermost(
    multidevice_benchmark,
    backend_type,
    s,
    disable_validation: bool,
    disable_benchmarking: bool,
):
    # Resetting the cache here is necessary to workaround a bug that would need a
    # proper fix. If not avoiding the cache, there is an issue for the second test
    # that is being run. More specifically, the second time we define the fusion,
    # we hit the cache in
    # https://github.com/NVIDIA/Fuser/blob/6ff60e2a320733a2f49de57007d6bb45000107cd/csrc/python_frontend/fusion_definition.cpp#L95 .
    # Later, when we call _set_device_mesh, we get a "thro out of range" here
    # https://github.com/NVIDIA/Fuser/blob/6ff60e2a320733a2f49de57007d6bb45000107cd/csrc/python_frontend/schedule_bindings.cpp#L60
    # because the FusionDefinition has not run so it doesn't contain any state.
    nvfuser.FusionCache.reset()

    m, k, n, d = 2**15, 2**17, 2**10, multidevice_benchmark.size
    assert m % (s * d) == 0
    os.environ["UCC_CL_BASIC_TLS"] = "nccl"
    torch.cuda.set_device(multidevice_benchmark.local_rank)

    # Create input tensors
    x_unsharded = torch.testing.make_tensor(
        s, d, m // (s * d), k, dtype=torch.bfloat16, device="cpu"
    )
    x = multidevice_benchmark.shard_tensor(
        x_unsharded, 1, nvfuser.DeviceMesh(range(multidevice_benchmark.size))
    )
    weight = torch.testing.make_tensor(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.testing.make_tensor(n, dtype=torch.bfloat16, device="cuda")
    inputs = [x, weight, bias]

    # Compute reference output
    out_ref = torch.nn.functional.linear(x_unsharded, weight.cpu(), bias.cpu())

    # Create fusion definition
    fd = OverlapAGMatmulStreamOutermost(m, k, n, s, d, backend_type)

    # Validate if needed
    if not disable_validation:
        outputs, _ = fd.execute(inputs)
        out = outputs[0].cpu()
        assert out.dtype == torch.bfloat16
        assert out.shape == torch.Size([s, d, m // (s * d), n])
        torch.testing.assert_close(out, out_ref, rtol=1e-1, atol=1e-1)

    # Run benchmark if needed
    if not disable_benchmarking:

        def benchmark_fn(*args):
            outputs, _ = fd.execute(*args)
            return outputs[0]

        avg_ms = benchmark_cuda_events(benchmark_fn, inputs)

        if multidevice_benchmark.rank == 0:
            print(f"\nBenchmark results for s={s}, backend={backend_type}:")
            print(f"Average time per iteration: {avg_ms:.3f} ms")
            print(f"Throughput: {m * k * n * 2 / (avg_ms * 1e-3) / 1e12:.3f} TFLOPs")
