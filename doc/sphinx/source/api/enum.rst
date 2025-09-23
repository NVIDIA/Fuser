..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Enums and POD classes
=====================

This section documents the various Enums and POD in NvFuser.


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

CommunicatorBackend Types
-------------------------

.. autoclass:: nvfuser_direct.CommunicatorBackend
   :members:

   CommunicatorBackend enum defines the different communication backends possible for NvFuser multi-gpu execution.

Data Types
----------

.. autoclass:: nvfuser_direct.DataType
   :members:

   DataType enum defines the supported data types in nvFuser.

Parallel Types
--------------

.. autoclass:: nvfuser_direct.ParallelType
   :members:

   ParallelType enum defines the different parallelization strategies available for tensor operations.

Scheduler Types
---------------

.. autoclass:: nvfuser_direct.SchedulerType
   :members:

   SchedulerType enum defines the different scheduling strategies available for optimizing fusion execution.
