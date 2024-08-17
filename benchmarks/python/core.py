# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import ctypes
import gc
import pytest_benchmark
import torch
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity
from typing import List, Callable, Union, Tuple
import numpy as np
from nvfuser import FusionDefinition, FusionCache


def get_device_properties() -> Tuple[int, float]:
    """
    Computes device properties using ctypes and cuda.
    Note: Consider using CUDA-Python when CUDA support >= 12.0.
    """
    libnames = ("libcuda.so", "libcuda.dylib", "nvcuda.dll", "cuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + " ".join(libnames))

    # Device attribute enums (taken from cuda.h)
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ge12b8a782bebe21b1ac0091bf9f4e2a3

    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39

    device_properties = {}
    device = torch.cuda.current_device()
    cuda_properties = torch.cuda.get_device_properties(device)

    device_properties["gpu_name"] = cuda_properties.name
    device_properties["gpu_compute_capability_major"] = cuda_properties.major
    device_properties["gpu_compute_capability_minor"] = cuda_properties.minor
    device_properties["gpu_gmem_bytes"] = cuda_properties.total_memory
    device_properties["gpu_sm_count"] = cuda_properties.multi_processor_count

    max_threads_per_block = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(max_threads_per_block),
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        device,
    )
    device_properties["gpu_max_threads_per_block"] = max_threads_per_block.value

    smem_per_block = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(smem_per_block),
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        device,
    )
    device_properties["gpu_smem_bytes_per_block"] = smem_per_block.value

    max_reg_per_block = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(max_reg_per_block),
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        device,
    )
    device_properties["gpu_regs_per_block"] = max_reg_per_block.value

    max_clock_khz = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(max_clock_khz),
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        device,
    )
    device_properties["gpu_clock_rate_khz"] = max_clock_khz.value

    l2_cache_size = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(l2_cache_size), CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device
    )
    device_properties["gpu_l2_bytes"] = l2_cache_size.value

    memory_clock_rate = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(memory_clock_rate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device
    )
    device_properties["gpu_mem_clock_khz"] = memory_clock_rate.value

    memory_bus_width = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(memory_bus_width),
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        device,
    )
    device_properties["gpu_mem_bus_width"] = memory_bus_width.value

    max_threads_per_sm = ctypes.c_int()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(max_threads_per_sm),
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        device,
    )
    device_properties["gpu_max_threads_per_sm"] = max_threads_per_sm.value

    # Compute peak bandwidth in GBps
    peak_bandwidth = (2 * memory_bus_width.value * memory_clock_rate.value) / (1e6 * 8)
    device_properties["gpu_peak_bandwidth_gbps"] = peak_bandwidth

    return device_properties


# These variables can be overwritten through CLI commands
# --benchmark-rounds=rounds --benchmark-warmup-rounds=warmup_rounds
# --benchmark-num-inputs=num_inputs
BENCHMARK_CONFIG = {"rounds": 10, "warmup_rounds": 1, "num_inputs": None}

DEVICE_PROPERTIES = get_device_properties()
L2_CACHE_SIZE = DEVICE_PROPERTIES["gpu_l2_bytes"]
PEAK_BANDWIDTH_GBPS = DEVICE_PROPERTIES["gpu_peak_bandwidth_gbps"]


def clear_l2_cache() -> None:
    """
    Flushes the L2 cache by creating a buffer of the same size.
    """
    n_elements = L2_CACHE_SIZE // 4
    x = torch.empty(n_elements, dtype=torch.float32, device="cuda", requires_grad=False)
    y = torch.clone(x)


def clear_dynamo_cache() -> None:
    """
    Utility function to enforce re-compilation to avoid different results between
    running a serials of tests and a standalone test due to kernel re-use.
    It slows down the test but ensures the correctness of the benchmark results.
    Ref: https://github.com/pytorch/pytorch/issues/107444
    """
    torch._dynamo.reset()


def clear_cuda_cache() -> None:
    """
    Utility function to clear CUDA cache before running a test.
    """
    if (
        torch.cuda.memory_allocated()
        or torch.cuda.memory_reserved() > 0.8 * DEVICE_PROPERTIES["gpu_gmem_bytes"]
    ):
        gc.collect()
        torch.cuda.empty_cache()


# Backward function for torch baseline benchmarks.
def unary_bwd_torch(inputs: List):  # [output, grad_out]
    inputs[0].backward(inputs[1], retain_graph=True)


def compute_total_iobytes(
    tensor_props: dict[str, tuple[int | tuple[int, ...], torch.dtype]]
):
    """
    Compute IObytes for baselines from given description:
    Tensor_props has entries of the form: {'tensor_id': (size: tuple, dtype: torch.dtype)}
    """
    iobytes = 0
    for _, tensor_prop in tensor_props.items():
        size, dtype = tensor_prop[0], tensor_prop[1]
        if isinstance(size, tuple):
            iobytes += np.prod(size) * dtype.itemsize
        else:
            iobytes += size * dtype.itemsize
    return int(iobytes)


class NVFBenchmark:
    """
    A wrapper class around pytest-benchmark to support
    torchprofiler-based timer and metric computation.
    """

    def __init__(
        self, benchmark_fixture, device: str = "cuda", precision: float = 1e-6
    ):
        """
        Arguments:
            benchmark_fixture: pytest-benchmark fixture passed to every
                function intended to be run as a benchmark by pytest
            precision: Precision for the torchprofiler-based timer used.
                Set explicitly to avoid timer calibration.

        Class members:
            self.prof: torch.profiler instance used as by the custom torchprofile_timer for the current benchmark
            self.benchmark: Underlying pytest-benchmark fixture with timer modified to use torchprofile_timer
            self.current_time: Global montonic clock incremented based on elapsed CUDA time
        """

        self.device = device
        self.fd = None  # Set through setup() for host benchmarking.
        self.benchmark = benchmark_fixture

        if device == "cuda":
            # Initialize a Torch Profiler object
            self.prof = profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]
            )
            # Modify the default timer.
            benchmark_fixture._timer = self.torchprofile_timer
        else:
            benchmark_fixture._timer = self.fusionprofile_timer
        # Externally set the precision to avoid timer calibration. Since the timer uses CUDA times,
        # calibration using subsequent timer calls produces invalid results.
        # https://github.com/ionelmc/pytest-benchmark/blob/728752d2976ef53fde7e40beb3e55f09cf4d4736/src/pytest_benchmark/timers.py#L15
        benchmark_fixture._precisions[benchmark_fixture._timer] = precision

        self.benchmark = benchmark_fixture

        # Global montonic clock
        self.current_time = 0.0

        # Specifies if the timer in host measurement is called at the start/finish of execution.
        # Timings are measured at the end of execution.
        self.execution_start = True

    def __call__(self, function_to_benchmark: Callable, *args, **kwargs):
        return self.benchmark(function_to_benchmark, *args, **kwargs)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.benchmark, attr)
        return super().__getattr__(attr)

    def torchprofile_timer(self) -> float:
        """
        Custom torchprofiler-based timer used by pytest-benchmark.
        At every timer call, the profiler is stopped to compute the elapsed CUDA time
        and the global clock is incremented. The profiler is restarted before returning to continue tracing.

        Returns:
            self.current_time: Global monotonic clock variable
        """
        try:
            self.prof.stop()
            prof_averages = self.prof.key_averages()
            elapsed_cuda_time = self._get_kernel_time(prof_averages)
            self._increment_global_time(elapsed_cuda_time)
            # Clear the internal profiler object to avoid accumulating function events and then restart the profiler
            # See PR: https://github.com/pytorch/pytorch/pull/125510
            self.prof.profiler = None
            self.prof.start()
        except AssertionError:
            self.prof.start()
        return self.current_time

    def fusionprofile_timer(self) -> float:
        if not self.execution_start:
            profile = self.fd.profile()
            elapsed_host_time = profile.host_time_ms / 1e3
            self._increment_global_time(elapsed_host_time)
        self.execution_start = not self.execution_start
        return self.current_time

    def _get_kernel_time(
        self, prof_averages: torch.autograd.profiler_util.EventList
    ) -> float:
        """
        Arguments:
            prof_averages: Output of self.prof.key_averages()
        Returns:
            time_value: Elapsed CUDA time in seconds.
        """
        elapsed_cuda_time = (
            sum(
                [
                    # Re: torch profiler API changes in https://github.com/pytorch/pytorch/pull/123247
                    (
                        event.self_device_time_total
                        if hasattr(event, "self_device_time_total")
                        else event.self_cuda_time_total
                    )
                    for event in prof_averages
                    if event.device_type == DeviceType.CUDA
                ]
            )
            / 1e6
        )
        return elapsed_cuda_time

    def _increment_global_time(self, elapsed_time: float) -> None:
        self.current_time += elapsed_time

    def cleanup(self) -> None:
        """
        Stops a running torchprofiler instance if found.
        """
        try:
            self.prof.stop()
        except AssertionError:
            pass

    def set_metrics(
        self,
        inputs: Union[torch.Tensor, List],
        outputs: Union[torch.Tensor, List],
        iobytes: int = None,
    ) -> None:
        """
        Utility function to compute metrics for the target function.

        Args:
            inputs: Inputs to the target function
            outputs: Outputs of the target function
            iobytes (Optional): When given, IO bytes computation is skipped
                and this is used to compute the metrics.

        Current metrics:
            IOBytes: Total bytes in inputs + outputs
            BytesPerSecond: IOBytes * total_rounds / total_time
            Bandwdith (GBps): BytesPerSecond / (1024**3)
            % Peak Bandwidth (SOL): 100 * Bandwidth /PEAK_BANDWIDTH
        """
        if not iobytes:
            if isinstance(inputs, torch.Tensor):
                inputs = [inputs]
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]

            iobytes = 0
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    iobytes += inp.element_size() * inp.numel()
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    iobytes += out.element_size() * out.numel()

        self.benchmark.extra_info["IOBytes"] = iobytes
        bandwidth_bps = (
            iobytes * self.benchmark.stats["rounds"]
        ) / self.benchmark.stats["total"]
        self.benchmark.extra_info["Bandwidth (Bps)"] = bandwidth_bps
        self.benchmark.extra_info["Bandwidth (GBps)"] = bandwidth_bps / 1024**3
        self.benchmark.extra_info["% Peak Bandwidth (SOL)"] = (
            100 * (bandwidth_bps / 1024**3) / PEAK_BANDWIDTH_GBPS
        )


def run_benchmark(
    benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    benchmark_fn: Callable | None,
    inputs: Union[torch.Tensor, List],
    iobytes: int = None,
    device: str = "cuda",
    fusion_fn: Callable = None,
) -> Union[torch.Tensor, List]:
    """
    Benchmarks the target function using torchprofiler and stores metrics as extra information.

    Arguments:
        benchmark: pytest-benchmark fixture
        benchmark_fn: Target function
        inputs: Inputs to the target function
        iobytes (Optional): When given, IO bytes computation is skipped
                and this is used to compute SOL and bandwidth.
        device (Optional): Default: CUDA, Possible values: ["cuda", "host"].
            Using device="host" is only allowed with nvFuser FusionDefinition.
        fusion_fn (Optional): Must be provided if device = "host".
            fusion_fn should only require FusionDefinition() as the input.
            Use functools.partial if fusion_fn accepts additional arguments.
            See test_many_pointwise_ops.py for example.

    Returns:
        outputs: Output of the target function
    """

    assert device in [
        "cuda",
        "host",
    ], f"Unsupported device type: {device}. Use one of cuda/host."

    if device == "host":
        assert fusion_fn is not None and benchmark_fn is None

    nvf_benchmark = NVFBenchmark(benchmark, device=device)

    def host_benchmark_fn(inputs, fd):
        # Set the fd variable used to query the profile object
        nvf_benchmark.fd = fd
        return fd.execute(inputs, profile=True)

    def setup():
        clear_l2_cache()

        if device == "host":
            # Reset the FusionCache to measure host overhead correctly.
            FusionCache.reset()
            with FusionDefinition() as fd:
                fusion_fn(fd)
            # The benchmark_fn used is host_benchmark_fn above.
            return [inputs], {"fd": fd}

        return [inputs], {}

    benchmark_fn = benchmark_fn if benchmark_fn is not None else host_benchmark_fn
    outputs = nvf_benchmark.pedantic(
        benchmark_fn,
        setup=setup,
        rounds=BENCHMARK_CONFIG["rounds"],
        warmup_rounds=BENCHMARK_CONFIG["warmup_rounds"],
    )

    if device == "cuda":
        # Record additional metrics (IOBytes, Bandwidth)
        nvf_benchmark.set_metrics(inputs, outputs, iobytes)
        # Stop torch.profiler instance
        nvf_benchmark.cleanup()

    return outputs
