..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Data classes
============

This section documents the various data classes in NvFuser.


LaunchParams
------------
.. autoclass:: nvfuser_direct.LaunchParams
   :members:
   :undoc-members:

   LaunchParams class hold the grid, block, and dynamic shared memory used when launching a CUDA kernel.

CompileParams
-------------
.. autoclass:: nvfuser_direct.CompileParams
   :members:
   :undoc-members:

   CompileParams hold the parameters used to control cubin generation with NVRTC.
