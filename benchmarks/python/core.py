# SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Iterable
import pytest_benchmark
import torch
from typing import List, Callable, Union
import numpy as np
from nvfuser import FusionDefinition, FusionCache
from nvfuser.pytorch_utils import DEVICE_PROPERTIES
import warnings
import thunder
from thunder.executors.nvfuserex import nvfuserex
from nvfuser.benchmark_utils import TorchProfileTimer, FusionProfileTimer

# These variables can be overwritten through CLI commands
# --benchmark-rounds=rounds --benchmark-warmup-rounds=warmup_rounds
# --benchmark-num-inputs=num_inputs
BENCHMARK_CONFIG = {
    "rounds": 10,
    "warmup_rounds": 1,
    "num_inputs": None,
    "with_nsys": False,
}

L2_CACHE_SIZE = DEVICE_PROPERTIES["gpu_l2_bytes"]
PEAK_BANDWIDTH_GBPS = DEVICE_PROPERTIES["gpu_peak_bandwidth_gbps"]

DEFAULT_EXECUTORS = ["eager", "torchcompile", "thunder"]


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


# Backward function for torch baseline benchmarks.
# The first two inputs are expected to be out and grad_out. The remaining are inputs of the forward pass used to clear grad between subsequent runs to avoid grad accumulation. See setup() in run_benchmark().
def unary_bwd_torch(inputs: List):  # [output, grad_out, fwd_inputs]
    inputs[0].backward(inputs[1], retain_graph=True)


def with_executor(executor: str, fwd_fn: Callable, **kwargs) -> Callable:
    assert executor in ["eager", "torchcompile", "thunder", "thunder-torchcompile"]
    if executor == "eager":
        return fwd_fn
    if executor == "torchcompile":
        return torch.compile(fwd_fn, **kwargs)
    if executor == "thunder":
        return thunder.jit(
            fwd_fn, nv_enable_bookend=False, executors=[nvfuserex], **kwargs
        )
    if executor == "thunder-torchcompile":
        return thunder.jit(fwd_fn, executors=["torchcompile"], **kwargs)


def compute_total_iobytes(
    tensor_props: dict[str, tuple[int | tuple[int, ...], torch.dtype]],
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
        self._setup_timer(benchmark_fixture, device, precision)
        self.benchmark = benchmark_fixture

    def _setup_timer(self, benchmark_fixture, device: str, precision: float):
        """Setup the appropriate timer based on device type."""
        if BENCHMARK_CONFIG["with_nsys"]:
            warnings.warn(
                "NSYS is enabled. No profiler will be used and pytest will report wall-clock time. "
                "Please refer to the nsys report for accurate timings."
            )
            return

        # Timer selection based on device
        timer_class = TorchProfileTimer if device == "cuda" else FusionProfileTimer
        benchmark_fixture._timer = timer_class()
        # Externally set the precision to avoid timer calibration. Since the timer uses CUDA times,
        # calibration using subsequent timer calls produces invalid results.
        # https://github.com/ionelmc/pytest-benchmark/blob/728752d2976ef53fde7e40beb3e55f09cf4d4736/src/pytest_benchmark/timers.py#L15
        benchmark_fixture._precisions[benchmark_fixture._timer] = precision

    def __call__(self, function_to_benchmark: Callable, *args, **kwargs):
        return self.benchmark(function_to_benchmark, *args, **kwargs)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.benchmark, attr)
        return super().__getattr__(attr)

    # Set the fd object for fusion profiling.
    # fd is returned by setup() for host benchmarking.
    def set_fd(self, fd):
        if BENCHMARK_CONFIG["with_nsys"]:
            return
        assert isinstance(self._timer, FusionProfileTimer)
        self._timer.set_fd(fd)

    def cleanup(self):
        if not BENCHMARK_CONFIG["with_nsys"]:
            self._timer.cleanup()

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
            Bandwdith (GBps): BytesPerSecond / 1e9
            % Peak Bandwidth (SOL): 100 * Bandwidth /PEAK_BANDWIDTH
        """
        if not iobytes:
            if not isinstance(inputs, Iterable) or isinstance(inputs, torch.Tensor):
                inputs = [inputs]
            if not isinstance(outputs, Iterable) or isinstance(outputs, torch.Tensor):
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
        self.benchmark.extra_info["Bandwidth (GBps)"] = bandwidth_bps / 1e9
        self.benchmark.extra_info["% Peak Bandwidth (SOL)"] = (
            100 * (bandwidth_bps / 1e9) / PEAK_BANDWIDTH_GBPS
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

    # Check that the device is `cuda` or `host:{compile/steady/dynamic}`.
    assert device.split(":")[0] in [
        "cuda",
        "host",
    ], f'Unsupported device type: {device.split(":")[0]}. Use one of cuda/host.'

    host_bench_mode = None

    # Store warmup rounds locally to modify for host:steady/dynamic cases.
    warmup_rounds = BENCHMARK_CONFIG["warmup_rounds"]

    if device.split(":")[0] == "host":
        # Host benchmarking expects a fusion function to generate fusion definitions everytime FusionCache is reset.
        assert fusion_fn is not None and benchmark_fn is None

        # Reset the FusionCache to avoid any inadvertent fusion execution from affecting measurements
        FusionCache.reset()

        # device = 'host:compile', 'host:steady', 'host:dyanamic'
        # Set the host_bench_mode -- The 3 modes require different setup calls.
        host_bench_mode = device.split(":")[-1]
        device = device.split(":")[0]  # device = 'host'

        # Set the warmup rounds if required for `steady/dynamic` host latency measurement.
        if (
            host_bench_mode in ["steady", "dynamic"]
            and BENCHMARK_CONFIG["warmup_rounds"] == 0
        ):
            # By default, warmup_rounds=1. If BENCHMARK_CONFIG['warmup_rounds'] == 0 through --benchmark-warmup-rounds, raise a warning that it was ignored.
            warnings.warn(
                "--benchmark-warmup-rounds=0 is ignored for host:steady/dynamic benchmarking. Setting warmup_rounds=1."
            )
            warmup_rounds = 1

    """
    Setup function: This is called before each benchmarking round. This function is used to:
    1. Clear L2 cache.
    2. For host latency benchmarks, the 3 modes use different setups.
        a) 'compile': FusionCache is reset at every round to measure the first time overhead before instantiating fd.
        b) 'steady': Nothing additional is required. The warmup round avoids including the first time overhead in the measurements.
        c) 'dynamic': We maintain a global counter to track which input is executed. Once all the inputs have been executed once,
        the FusionCache is reset again and we execute fd for the first input to avoid including the first time compile overhead in the dynamic measurement.
    """

    # Counter used in `dynamic` host latency benchmarking, unused in other cases.
    global counter
    counter = 0

    def setup():
        clear_l2_cache()
        if device == "cuda":
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    inp.grad = None
            return [inputs], {}

        # Device = 'host'
        # For device='host', we use the host_benchmark_fn below. It expects a list of fusion inputs, and the fd object.
        assert host_bench_mode in [
            "compile",
            "steady",
            "dynamic",
        ], f"Expected host benchmark mode to be one of compile, steady, or dynamic, found {host_bench_mode}"

        if host_bench_mode == "compile":
            # Reset the FusionCache to measure initial host overhead correctly.
            FusionCache.reset()

        # Instantiate the fusion definition
        with FusionDefinition() as fd:
            fusion_fn(fd)

        if host_bench_mode in ["compile", "steady"]:
            return [inputs], {"fd": fd}

        # For dynamic host latency benchmarking, return a particular input shape, and reset FusionCache if all inputs have been executed.
        global counter
        counter += 1
        if counter % len(inputs) == 0:
            # All inputs have been executed once.
            FusionCache.reset()
            with FusionDefinition() as fd:
                fusion_fn(fd)
            # Execute fd with the first inputs to avoid measuring first time overhead.
            fd.execute(inputs[0])
            counter += 1
        return [inputs[counter % len(inputs)]], {"fd": fd}

    # Create an instance of NVFBenchmark
    nvf_benchmark = NVFBenchmark(benchmark, device=device)

    # The host_benchmark_fn uses the `fd` object returned from setup function.
    def host_benchmark_fn(inputs, fd):
        # Set the fd variable used to query the profile object
        nvf_benchmark.set_fd(fd)
        return fd.execute(inputs, profile=not BENCHMARK_CONFIG["with_nsys"])

    benchmark_fn = benchmark_fn if benchmark_fn is not None else host_benchmark_fn

    outputs = nvf_benchmark.pedantic(
        benchmark_fn,
        setup=setup,
        rounds=BENCHMARK_CONFIG["rounds"],
        warmup_rounds=warmup_rounds,
    )

    if device == "cuda":
        # Record additional metrics (IOBytes, Bandwidth)
        nvf_benchmark.set_metrics(inputs, outputs, iobytes)
        # Stop torch.profiler instance
        nvf_benchmark.cleanup()

    return outputs
