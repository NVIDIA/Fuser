import pytest
from torch.profiler import profile, ProfilerActivity
from nvfuser import FusionDefinition, DataType
import torch
import logging
from pprint import pformat

# Create a named logger
logger = logging.getLogger('__temp_convertor__')
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the formatter for the console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
datefmt='%m/%d/%Y %I:%M:%S%p')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Create a file handler
file_handler = logging.FileHandler('./logs/temp_tests_run.log')
file_handler.setLevel(logging.INFO)

# Set the formatter for the file handler
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')

# file_handler.setFormatter(formatter)
# Add the file handler to the logger

logger.addHandler(file_handler)

def fusion_func(fd: FusionDefinition):
    t0 = fd.from_pytorch(inputs[0])
    t1 = fd.from_pytorch(inputs[1])
    c0 = fd.define_scalar(3.0)

    t2 = fd.ops.add(t0, t1)
    t3 = fd.ops.mul(t2, c0)
    t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

    t5 = fd.ops.cast(t4, DataType.Half)
    fd.add_output(t5)

def test_my_stuff():
    inputs = [
            torch.ones(2, 4, 8, device="cuda"),
            torch.ones(2, 4, 8, device="cuda"),
        ]

    def fusion_func(fd: FusionDefinition):
        t0 = fd.from_pytorch(inputs[0])
        t1 = fd.from_pytorch(inputs[1])
        c0 = fd.define_scalar(3.0)

        t2 = fd.ops.add(t0, t1)
        t3 = fd.ops.mul(t2, c0)
        t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

        t5 = fd.ops.cast(t4, DataType.Half)
        fd.add_output(t5)
    
    with FusionDefinition() as fd:
        fusion_func(fd)

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        output = fd.execute(inputs)
    
    print(prof.key_averages())
    cuda_time = sum([item.cuda_time for item in prof.key_averages()])
    cpu_time = sum([item.cpu_time for item in prof.key_averages()])

    times = {}
    times['cuda'] = cuda_time
    times['cpu'] = cpu_time
    return

if __name__ == "__main__":
    test_my_stuff()