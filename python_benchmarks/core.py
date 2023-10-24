import ctypes
import torch
from torch.profiler import profile, ProfilerActivity
from typing import List, Callable, Union, Tuple
from torch.autograd import DeviceType


def get_device_properties() -> Tuple[int, float]:
    """
    Computes L2 cache size and peak device bandwidth (GBps) using ctypes and cuda.
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
    # https://nvidia.github.io/cuda-python/module/cuda.html

    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38

    l2_cache_size = ctypes.c_int()
    device = torch.cuda.current_device()
    memory_clock_rate = ctypes.c_int()
    memory_bus_width = ctypes.c_int()

    cuda.cuDeviceGetAttribute(
        ctypes.byref(l2_cache_size), CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device
    )
    cuda.cuDeviceGetAttribute(
        ctypes.byref(memory_clock_rate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device
    )
    cuda.cuDeviceGetAttribute(
        ctypes.byref(memory_bus_width),
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        device,
    )

    # Compute peak bandwidth in GBps
    peak_bandwidth = (2 * memory_bus_width.value * memory_clock_rate.value) / (1e6 * 8)

    return l2_cache_size.value, peak_bandwidth


L2_CACHE_SIZE, PEAK_BANDWIDTH_GBPS = get_device_properties()


def clear_l2_cache() -> None:
    """
    Flushes the L2 cache by creating a buffer of the same size.
    """
    n_elements = L2_CACHE_SIZE // 4
    x = torch.empty(n_elements, dtype=torch.float32, device="cuda", requires_grad=False)
    y = torch.clone(x)


class NVFBenchmark:
    """
    A wrapper class around pytest-benchmark to support
    torchprofiler-based timer and metric computation.
    """

    def __init__(self, benchmark_fixture, precision: float = 1e-6):
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
        # Initialize a Torch Profiler object
        self.prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])

        # Modify the default timer.
        benchmark_fixture._timer = self.torchprofile_timer
        # Externally set the precision to avoid timer calibration. Since the timer uses CUDA times,
        # calibration using subsequent timer calls produces invalid results.
        # https://github.com/ionelmc/pytest-benchmark/blob/728752d2976ef53fde7e40beb3e55f09cf4d4736/src/pytest_benchmark/timers.py#L15
        benchmark_fixture._precisions[benchmark_fixture._timer] = precision

        self.benchmark = benchmark_fixture

        # Global montonic clock
        self.current_time = 0.0

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
            self.prof.start()
        except AssertionError:
            self.prof.start()
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
                    event.self_cuda_time_total
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
        self, inputs: Union[torch.Tensor, List], outputs: Union[torch.Tensor, List]
    ) -> None:
        """
        Utility function to compute metrics for the target function.
        Current metrics:
            IOBytes: Total bytes in inputs + outputs
            BytesPerSecond: IOBytes * total_rounds / total_time
            Bandwdith (GBps): BytesPerSecond / (1024**3)
            % Peak Bandwidth (SOL): 100 * Bandwidth /PEAK_BANDWIDTH
        """
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
    benchmark,
    benchmark_fn: Callable,
    inputs: Union[torch.Tensor, List],
    rounds: int = 10,
    warmup_rounds: int = 1,
) -> Union[torch.Tensor, List]:
    """
    Benchmarks the target function using torchprofiler and stores metrics as extra information.

    Arguments:
        benchmark: pytest-benchmark fixture
        benchmark_fn: Target function
        inputs: Inputs to the target function

    Returns:
        outputs: Output of the target function
    """

    def setup():
        clear_l2_cache()
        return [inputs], {}

    nvf_benchmark = NVFBenchmark(benchmark)
    outputs = nvf_benchmark.pedantic(
        benchmark_fn, setup=setup, rounds=rounds, warmup_rounds=warmup_rounds
    )
    nvf_benchmark.set_metrics(inputs, outputs)
    nvf_benchmark.cleanup()
    return outputs
