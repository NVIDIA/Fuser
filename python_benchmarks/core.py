import torch
from torch.profiler import profile, ProfilerActivity
from nvfuser import FusionDefinition, DataType
import sys
from pytest_benchmark.fixture import BenchmarkFixture
import logging
# # Create a named logger
# logger = logging.getLogger('__core__')
# logger.setLevel(logging.INFO)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# logger.addHandler(console_handler)

# # Create a file handler
# file_handler = logging.FileHandler('./logs/core.log')
# file_handler.setLevel(logging.INFO)
# logger.addHandler(file_handler)

gpu_name = torch.cuda.get_device_name(0)
if 'A100' in gpu_name:
  GPU = "A100"
elif 'H100' in gpu_name:
  GPU = "H100"
elif 'V100' in gpu_name:
  GPU = "V100"
else:
  raise ValueError("Invalid GPU name")

# set different properties for different GPUs
if GPU == "A100":
  L2_CACHE_SIZE = 40*1024*1024
  SM_COUNT = 108
elif GPU == "H100":
  L2_CACHE_SIZE = 50*1024*1024
  SM_COUNT = 132
elif GPU == "V100":
  L2_CACHE_SIZE = 6*1024*1024
  SM_COUNT = 80

def clearL2Cache():
    n_elements = L2_CACHE_SIZE // 4
    x = torch.empty(n_elements, dtype=torch.float32, device='cuda', requires_grad=False)
    y = torch.clone(x)

class NVFBenchmark(object):
    def __init__(self, benchmark_fixture: pytest_benchmark.fixture.BenchmarkFixture, precision:float=1e-6):
        # Initialize a Torch Profiler object
        self.prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])

        # Set the timer and precision 
        benchmark_fixture._timer = self.torchprofile_timer
        benchmark_fixture._precisions[benchmark_fixture._timer] = precision

        self.benchmark = benchmark_fixture
        self.benchmark._timer = self.torchprofile_timer
        self.benchmark._precisions[self.benchmark._timer] = precision
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
    
    def _get_kernel_time(self, prof_output:str):
        #Split the profiler output and get the last two lines
        prof_averages = prof_output.split('\n')[-2:]
        cuda_avg_str = None
        for time_avg_str in prof_averages:
            if 'CUDA' not in time_avg_str:
                continue
            cuda_avg_str = time_avg_str

        time_str = cuda_avg_str.split()[-1]
        time_unit = time_str[-2:]
        time_value = float(time_str[:-2])
        if time_unit == 'us':
            time_value /= 1e6 
        elif time_unit == 'ms':
            time_value /= 1e3
        return time_value
    
    def _increment_global_time(self, elapsed_time:float):
        self.current_time += elapsed_time
    
    def cleanup(self):
        try:
            self.prof.stop()
        except:
            pass
    
    
