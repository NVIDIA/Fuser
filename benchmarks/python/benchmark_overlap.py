# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import torch
import nvfuser
from nvfuser import FusionDefinition, CommunicatorBackend
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype
from .core import BENCHMARK_CONFIG, clear_l2_cache


class CUDAEventTimer:
    """Custom CUDA event-based timer for accurate GPU timing.

    This timer uses CUDA events to measure elapsed time between operations,
    providing more accurate GPU timing than CPU-based timing methods.
    """

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.is_running = False

    def __call__(self):
        """Record timing events and compute elapsed time.

        Returns:
            float: Elapsed time in seconds
        """
        if self.is_running:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
            self.is_running = False
            return elapsed_ms / 1000.0  # Convert ms to seconds
        else:
            self.start_event.record()
            self.is_running = True
            return 0.0

    def cleanup(self):
        """Ensure timer is not running and synchronize CUDA."""
        if self.is_running:
            self.end_event.record()
            torch.cuda.synchronize()
            self.is_running = False


def benchmark_cuda_events_pedantic(
    benchmark, benchmark_fn, inputs, rounds, warmup_rounds
):
    """Wrapper for benchmark_cuda_events that uses pytest's pedantic method with CUDA events.

    Args:
        benchmark: pytest-benchmark fixture
        benchmark_fn: Function to benchmark
        inputs: List of inputs to pass to benchmark_fn
        rounds: Number of rounds to run
        warmup_rounds: Number of warmup rounds
    """

    def setup():
        clear_l2_cache()
        return inputs, {}

    def wrapped_fn(*args):
        benchmark_fn(*args[0])
        return None

    # Set our custom CUDA event timer
    benchmark._timer = CUDAEventTimer()

    benchmark.pedantic(
        wrapped_fn,
        setup=setup,
        rounds=rounds,
        warmup_rounds=warmup_rounds,
        iterations=1,
    )


class OverlapAGMatmulStreamOutermost(FusionDefinition):
    """Fusion definition for overlapping all-gather with matrix multiplication.

    This fusion implements a matrix multiplication operation with overlapping
    all-gather communication, using stream parallelism for the outermost dimension.
    """

    def __init__(self, m, k, n, s, num_devices, communication_backend, dtype):
        super().__init__(
            use_multidevice_executor=True, backend_type=communication_backend
        )
        self.m = m
        self.k = k
        self.n = n
        self.s = s
        self._num_devices = num_devices
        self.dtype = dtype

    def definition(self) -> None:
        m, k, n, s, d = self.m, self.k, self.n, self.s, self._num_devices
        self.x = self.define_tensor(
            shape=[s, d, m // (s * d), k],
            contiguity=True,
            dtype=torch_dtype_to_nvfuser_dtype(self.dtype),
        )
        self.weight = self.define_tensor(
            shape=[n, k],
            contiguity=True,
            dtype=torch_dtype_to_nvfuser_dtype(self.dtype),
        )
        self.bias = self.define_tensor(
            shape=[n], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype)
        )
        self.out = self.ops.linear(self.x, self.weight, self.bias)
        self.add_output(self.out)

    def multidevice_schedule(self):
        mesh = nvfuser.DeviceMesh(range(self._num_devices))
        for tv in [self.x, self.weight, self.bias, self.out]:
            self.sched._set_device_mesh(tv, mesh)
        self.sched.parallelize(self.x, 1, nvfuser.ParallelType.mesh_x)
        self.sched.parallelize(self.out, 0, nvfuser.ParallelType.stream)


class MultideviceSettings:
    """Settings and utilities for multi-device execution."""

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
def multidevice_settings():
    return MultideviceSettings()


@pytest.mark.mpi
@pytest.mark.parametrize(
    "backend_type", [CommunicatorBackend.ucc, CommunicatorBackend.nccl]
)
@pytest.mark.parametrize("s", [1, 8])
@pytest.mark.parametrize("m", [2**16])
@pytest.mark.parametrize("k", [2**12, 2**16])
@pytest.mark.parametrize("n", [2**10])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_overlap_allgather_matmul_stream_outermost(
    benchmark,
    multidevice_settings,
    backend_type,
    s,
    m,
    k,
    n,
    dtype,
    validate_output=False,
):
    """Test overlapping all-gather with matrix multiplication using stream parallelism.

    Args:
        benchmark: pytest-benchmark fixture
        multidevice_settings: Settings for multi-device execution
        backend_type: Communication backend to use
        s: Number of streams
        m: Matrix dimension m
        k: Matrix dimension k
        n: Matrix dimension n
        dtype: Data type for computation
        validate_output: Whether to validate output against reference
    """
    nvfuser.FusionCache.reset()

    d = multidevice_settings.size
    assert m % (s * d) == 0
    os.environ["UCC_CL_BASIC_TLS"] = "nccl"
    torch.cuda.set_device(multidevice_settings.local_rank)

    # Create input tensors
    x_unsharded = torch.testing.make_tensor(
        s, d, m // (s * d), k, dtype=dtype, device="cpu"
    )
    x = multidevice_settings.shard_tensor(
        x_unsharded, 1, nvfuser.DeviceMesh(range(multidevice_settings.size))
    )
    weight = torch.testing.make_tensor(n, k, dtype=dtype, device="cuda")
    bias = torch.testing.make_tensor(n, dtype=dtype, device="cuda")
    inputs = [x, weight, bias]

    # Create fusion definition
    fd = OverlapAGMatmulStreamOutermost(m, k, n, s, d, backend_type, dtype)

    if validate_output:
        outputs, _ = fd.execute([inputs])
        out = outputs[0].cpu()
        assert out.dtype == dtype
        assert out.shape == torch.Size([s, d, m // (s * d), n])
        out_ref = torch.nn.functional.linear(x_unsharded, weight.cpu(), bias.cpu())
        torch.testing.assert_close(out, out_ref, rtol=float("inf"), atol=1e-1)

    def benchmark_fn(*args):
        outputs, _ = fd.execute(args)
        return outputs[0]

    benchmark_cuda_events_pedantic(
        benchmark,
        benchmark_fn,
        [inputs],
        warmup_rounds=BENCHMARK_CONFIG["warmup_rounds"],
        rounds=BENCHMARK_CONFIG["rounds"],
    )
