<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Debugging

## Debug a failing nvFuser script

Debugging a failing nvFuser Python script typically follows the following workflow.

1. An error in compilation is observed when running a python script. This will print a reproducer Python script as part of the error message that defines the fusion and some inputs.
3. You begin debugging by inspecting where the error came from and isolating the problematic Fusion segment that failed to compile.
4. You isolate a repro for that failing segment and try to simplify it as much as possible while checking that it still triggers the bad behavior.
2. (optional) You copy the repro error and describe what you were doing in a new here issue on the nvFuser repo.
5. Use `NVFUSER_DUMP` options and `gdb` to inspect the runtime state of nvFuser to try and determine the root cause and find a fix.

In step 1, the repro will look something like this:
````
An error occurred while executing nvFuser FusionDefinition 0.
If you believe this is a bug or need assistance, please file an issue at https://github.com/NVIDIA/Fuser/issues/new
Here's a script to reproduce the error:
```python
# CUDA devices:
#  0: NVIDIA H100 80GB HBM3
# torch version: 2.6.0a0+gitffb7a08
# cuda version: 12.6
# nvfuser version: 0.2.22+git6912435
import torch
from nvfuser import FusionDefinition, DataType

def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 28, 32768, 2], contiguity=[None, True, False, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[3, 2, 1, 0])
    T1 = fd.define_tensor(shape=[1, 32768, 2], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False, stride_order=[2, 1, 0])
    # ...
    fd.add_output(T273)

with FusionDefinition() as fd:
    nvfuser_fusion_id0(fd)

inputs = [
    torch.randn(7340026, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 28, 32768, 2), (7340032, 262144, 8, 1)),
    torch.randn(7340026, dtype=torch.bfloat16, device='cuda:0').as_strided((1, 28, 32768, 2), (7340032, 262144, 8, 1)),
]
fd.execute(inputs)
```
````
while a compile error might give a message like the following:
```
Traceback (most recent call last):
  File "/opt/pytorch/nvfuser/nvfuser/__init__.py", line 182, in execute
    results = self._execute(
              ^^^^^^^^^^^^^^
RuntimeError:  INTERNAL ASSERT FAILED at "/opt/pytorch/nvfuser/csrc/runtime/fusion_kernel_runtime.cpp":368, please report a bug with repro script to NVFuser at https://github.com/NVIDIA/Fuser/issues. Detected exception while compiling fusion segments in parallel. Error messages from all threads are printed below.

Error from segmentation group 11:  INTERNAL ASSERT FAILED at "/opt/pytorch/nvfuser/csrc/index_compute.cpp":1995, please report a bug with repro script to NVFuser at https://github.com/NVIDIA/Fuser/issues. Couldn't find allocation mapping for T125_l___bfloat[ iblockIdx.x846{( ceilDiv(2, blockDim.x) )}, ithreadIdx.x847{blockDim.x}, iS855{( ceilDiv(( ceilDiv(( ceilDiv(( ceilDiv(32768, blockDim.y) ), 16) ), 1) ), gridDim.y) )}, iblockIdx.y854{gridDim.y}, ithreadIdx.y849{blockDim.y}, iUS853{1}, iUR851{16}, bS505{1} ] ca_pos( 6 ) dim: 2 id: iS507{2}
Exception raised from getNonGlobalConsumerStridedIndices at /opt/pytorch/nvfuser/csrc/index_compute.cpp:1995 (most recent call first):
frame #0: nvfuser::nvfCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x91 (0x7ff45f092448 in /opt/pytorch/nvfuser/nvfuser/_C.cpython-312-x86_64-linux-gnu.so)
...
```
This indicates that segmentation group 11 is the one with the problem.

Step 2 is aided by launching your script like `NVFUSER_DUMP=python_definition_segments python foo.py`. This will print, for each segment, a smaller fusion definition than in the overall repro shown above:
```
Python definition for segmented group 8:

def nvfuser_fusion_id8(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(shape=[1, 32768, 56], contiguity=[None, True, True], dtype=DataType.BFloat16, is_cpu=False)
    T1 = fd.ops.squeeze(T0, dims=[0], squeeze_expanded=True)
    T2 = fd.ops.permute(T1, dims=[1, 0])
    fd.add_output(T1, stride_order=[1, 0])
    fd.add_output(T2, stride_order=[0, 1])
```
Find the group matching the problematic one shown in the error message and this will allow you to cut+paste a new, more targeted repro. Don't forget to modify the `inputs` to match those expected by the segment fusion definition.

### NVFUSER_DUMP

Use the `NVFUSER_DUMP` environment variable to control what intermediate results
to dump and verbose logging. It can be prepended to any command that launches
nvfuser, e.g., `bin/test_nvfuser`, `bin/nvfuser_bench` and `python3
a_python_script_that_imports_and_runs_nvfuser.py`.
`csrc/options.cpp` lists all dumping options and their meanings.

Examples:
* `NVFUSER_DUMP=cuda_kernel` prints the generated CUDA kernels.
* `NVFUSER_DUMP=segmenter_logging` prints which scheduler gets used.

### gdb

```shell
$ NVFUSER_BUILD_BUILD_TYPE=Debug pip install -v -e ./python --no-build-isolation
# or if you are on the PJNL docker image, DEBUG_BUILD=1 _bn
$ gdb --args bin/test_nvfuser --gtest_filter=<FILTER>
(gdb) catch throw nvfuser::nvfError
(gdb) r
```

## Debug memory corruption using `asan`

```shell
$ NVFUSER_BUILD_WITH_ASAN=1 pip install -v -e ./python --no-build-isolation
# or if you are on the PJNL docker image, NVFUSER_BUILD_WITH_ASAN=1 _bn

$ LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS=protect_shadow_gap=0 <CMD>
# The ASAN_OPTIONS is needed to work around https://github.com/google/sanitizers/issues/629.
```

### If built with clang

```shell
$ ASAN_OPTIONS=protect_shadow_gap=0 <test binary e.g. bin/test_nvfuser>
```
for C++ tests. `LD_PRELOAD` isn't needed because clang by default uses `-static-libsan` and C++ tests are statically linked.

```shell
$ LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so) ASAN_OPTIONS=protect_shadow_gap=0 pytest tests/python/<test_file e.g. test_python_frontend.py>
```
For Python tests. `LD_PRELOAD` is needed because the Python API loads nvFuser as a shared library.

## Debug memory leaks or excessive memory usage

```shell
# Install tcmalloc and some other tools.
$ sudo apt install google-perftools

# For me, tcmalloc was installed at /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
$ LD_PRELOAD=<path to libtcmalloc.so> HEAPPROFILE=/tmp/<NAME> <CMD>
```

The above command should print out "Starting tracking the heap" at the beginning. During or at the end of the program execution, you should be able to see something like "Dumping heap profile to /tmp/<NAME>.<NUMBER>.heap". These are the dumped heap profiles to be examined by `pprof`.

```shell
$ sudo apt install golang
$ go install github.com/google/pprof@latest
$ $HOME/go/bin/pprof -dot -output /tmp/<NAME>.dot /tmp/<NAME>.<NUMBER>.heap
$ dot -Tpng /tmp/<NAME>.dot -o /tmp/<NAME>.png
```

## Debug slow kernels

You can do that with `nsys` or `ncu`. For example,

```shell
$ nsys profile <CMD>
$ nsys stats --report cuda_gpu_kern_sum <the .nsys-rep file generated by the above command>
```

```shell
$ ncu -k <KERNEL_NAME_FILTER> <CMD>
```

Unlike `nsys`, `ncu` by default tries to stabilize measurement by flushing GPU caches and locking clocks. `ncu -h` for knobs to change that behavior.

For better UI, you can let `ncu` export profiling results to `.ncu-rep` remotely and open that from the Nsight Compute GUI on your host (e.g. a MacBook). Note that Nsight Compute is a different tool from Nsight Systems.

```shell
$ ncu -o <OUTPUT_NAME> <OTHER_OPTIONS> <CMD>
...
==PROF== Report: <OUTPUT_NAME>.ncu-rep
```

When examine nvrtc compiled kernel, it's useful to associate cuda source file with the lowered device code. `-lineinfo` is useful for that as well as the source code.
```shell
$ NVFUSER_ENABLE=kernel_lineinfo NVFUSER_DUMP=cuda_to_file ncu <CMD>
```

## Debug slow CPU execution

Use Google's [cpuprofile](https://gperftools.github.io/gperftools/cpuprofile.html)

```shell
$ sudo apt install google-perftools libgoogle-perftools-dev golang
# Run the binary with the profiler, e.g.,
$ LD_PRELOAD=<path to libprofiler e.g. /usr/lib/x86_64-linux-gnu/libprofiler.so> CPUPROFILE=<path to the profiling results to be created e.g /tmp/test_nvfuser.prof> bin/test_nvfuser --gtest_filter=<filter>
$ go install github.com/google/pprof@latest
$ $HOME/go/bin/pprof --pdf bin/test_nvfuser /tmp/test_nvfuser.prof
Generating report in profile001.pdf
```
