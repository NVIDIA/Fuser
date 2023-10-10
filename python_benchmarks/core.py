import ctypes
import torch
from torch.profiler import profile, ProfilerActivity

def get_l2_cache_size() -> int:
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

    # Device attribute enum for L2 cache size
    # https://nvidia.github.io/cuda-python/module/cuda.html?highlight=l2+cache+size#cuda.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
    
    l2_cache_size = ctypes.c_int()
    device = torch.cuda.current_device()
    cuda.cuDeviceGetAttribute(
        ctypes.byref(l2_cache_size), CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device
    )
    return l2_cache_size.value

L2_CACHE_SIZE = get_l2_cache_size()

def clear_l2_cache() -> None:
    '''
    Flushes the L2 cache by creating a buffer of the same size.
    '''
    n_elements = L2_CACHE_SIZE // 4
    x = torch.empty(n_elements, dtype=torch.float32, device="cuda", requires_grad=False)
    y = torch.clone(x)


class NVFBenchmark():
    '''
    A wrapper class around pytest-benchmark to support 
    torchprofiler-based timer and metric computation.
    '''
    def __init__(self, benchmark_fixture, precision: float = 1e-6):
        '''
        Arguments:
            benchmark_fixture: pytest-benchmark fixture passed to every 
                function intended to be run as a benchmark by pytest
            precision: Precision for the torchprofiler-based timer used. 
                Set explicitly to avoid timer calibration. 
        '''
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

    def __call__(self, function_to_benchmark, *args, **kwargs):
        return self.benchmark(function_to_benchmark, *args, **kwargs)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.benchmark, attr)
        return super().__getattr__(attr)

    def torchprofile_timer(self) -> float:
        '''
        Custom torchprofiler-based timer used by pytest-benchmark. 
        At every timer call, the profiler is stopped to compute the elapsed CUDA time 
        and the global clock is incremented. The profiler is restarted before returning to continue tracing.

        Returns:
            self.current_time: Global monotonic clock variable
        '''
        try:
            self.prof.stop()
            prof_output = self.prof.key_averages().table()
            elapsed_cuda_time = self._get_kernel_time(prof_output)
            self._increment_global_time(elapsed_cuda_time)
            self.prof.start()
        except:
            self.prof.start()
        return self.current_time

    def _get_kernel_time(self, prof_output: str) -> float:
        '''
        Parses the profiler output to obtain the total CUDA time.
        Returns:
            time_value: Elapsed CUDA time.
        '''
        prof_averages = prof_output.split("\n")[-2:]
        cuda_avg_str = None
        for time_avg_str in prof_averages:
            if "CUDA" not in time_avg_str:
                continue
            cuda_avg_str = time_avg_str

        time_str = cuda_avg_str.split()[-1]
        time_unit = time_str[-2:]
        time_value = float(time_str[:-2])
        if time_unit == "us":
            time_value /= 1e6
        elif time_unit == "ms":
            time_value /= 1e3
        return time_value

    def _increment_global_time(self, elapsed_time: float) -> None:
        self.current_time += elapsed_time

    def cleanup(self) -> None:
        '''
        Stops a running torchprofiler instance if found.
        '''
        try:
            self.prof.stop()
        except:
            pass

    def compute_metrics(self, inputs, outputs) -> None:
        '''
        Utility function that computes IO Bytes processed by the 
        target function and the bytes per second rate achieved. 
        '''
        iobytes = 0
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                iobytes += inp.element_size() * inp.numel()
        for out in outputs:
            iobytes += out.element_size() * out.numel()

        self.benchmark.extra_info["IOBytes"] = iobytes
        self.benchmark.extra_info["BytesPerSecond"] = (
            iobytes * self.benchmark.stats["rounds"]
        ) / self.benchmark.stats["total"]


def run_benchmark(benchmark, benchmark_fn, inputs):
    '''
    Benchmarks the target function using torchprofiler and stores metrics as extra information.

    Arguments:
        benchmark: pytest-benchmark fixture
        benchmark_fn: Target function
        inputs: Inputs to the target function

    Returns:
        outputs: Output of the target function
    '''
    nvf_bench = NVFBenchmark(benchmark)
    clear_l2_cache()
    outputs = nvf_bench(benchmark_fn, inputs)
    nvf_bench.compute_metrics(inputs, outputs)
    nvf_bench.cleanup()
    return outputs
