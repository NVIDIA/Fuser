import sys
import ctypes
import torch
from torch.profiler import profile, ProfilerActivity

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

CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
l2_cache_size = ctypes.c_int()
device = torch.cuda.current_device()
cuda.cuDeviceGetAttribute(
    ctypes.byref(l2_cache_size), CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device
)

L2_CACHE_SIZE = l2_cache_size.value


def clearL2Cache():
    n_elements = L2_CACHE_SIZE // 4
    x = torch.empty(n_elements, dtype=torch.float32, device="cuda", requires_grad=False)
    y = torch.clone(x)


class NVFBenchmark(object):
    def __init__(self, benchmark_fixture, precision: float = 1e-6):
        # Initialize a Torch Profiler object
        self.prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])

        # Set the timer and precision
        benchmark_fixture._timer = self.torchprofile_timer
        benchmark_fixture._precisions[benchmark_fixture._timer] = precision

        self.benchmark = benchmark_fixture
        self.current_time = 0.0

    def __call__(self, function_to_benchmark, *args, **kwargs):
        return self.benchmark(function_to_benchmark, *args, **kwargs)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.benchmark, attr)
        return super().__getattr__(attr)

    def torchprofile_timer(self):
        try:
            self.prof.stop()
            prof_output = self.prof.key_averages().table()
            elapsed_cuda_time = self._get_kernel_time(prof_output)
            self._increment_global_time(elapsed_cuda_time)
            self.prof.start()
        except:
            self.prof.start()
        return self.current_time

    def _get_kernel_time(self, prof_output: str):
        # Split the profiler output and get the last two lines
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

    def _increment_global_time(self, elapsed_time: float):
        self.current_time += elapsed_time

    def cleanup(self):
        try:
            self.prof.stop()
        except:
            pass

    def compute_metrics(self, inputs, outputs):
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


def runBenchmark(benchmark, benchmark_fn, inputs):
    nvf_bench = NVFBenchmark(benchmark)
    clearL2Cache()
    outputs = nvf_bench(benchmark_fn, inputs)
    nvf_bench.compute_metrics(inputs, outputs)
    nvf_bench.cleanup()
    return outputs
