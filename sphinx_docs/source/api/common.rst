..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Common API
==========

This section documents the common API components of nvFuser.

Direct Fusion Definition
-----------------------

.. autoclass:: nvfuser.direct._DirectFusionDefinition
   :members:
   :undoc-members:
   :show-inheritance:

   The DirectFusionDefinition class provides a low-level interface for defining nvFuser operations.
   It exposes two main components:

   - ``ops``: Contains the operators for defining fusion operations
   - ``sched``: Contains the scheduling operators for optimizing the fusion

TensorView
---------

.. autoclass:: nvfuser.direct.TensorView
   :members:
   :undoc-members:
   :show-inheritance:

   TensorView represents a tensor in the fusion definition. It provides methods for:
   
   - Accessing tensor properties (domain, axes, etc.)
   - Performing operations on tensors
   - Managing tensor memory and scheduling

TensorViewBuilder
---------------

.. autoclass:: nvfuser.direct.TensorViewBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   TensorViewBuilder provides a fluent interface for creating TensorView objects with specific properties.

FusionExecutorCache
-----------------

.. autoclass:: nvfuser.direct.FusionExecutorCache
   :members:
   :undoc-members:
   :show-inheritance:

   FusionExecutorCache manages the compilation and execution of fusion definitions.
   It provides methods for:
   
   - Compiling fusions
   - Executing fusions with input tensors
   - Accessing compiled kernels and IR

KernelExecutor
------------

.. autoclass:: nvfuser.direct.KernelExecutor
   :members:
   :undoc-members:
   :show-inheritance:

   KernelExecutor provides a lower-level interface for executing compiled kernels.
   It allows for:
   
   - Manual kernel compilation
   - Direct kernel execution
   - Access to kernel properties and scheduling information

Data Types
---------

.. autoclass:: nvfuser.DataType
   :members:
   :undoc-members:
   :show-inheritance:

   DataType enum defines the supported data types in nvFuser.

Memory Types
-----------

.. autoclass:: nvfuser.MemoryType
   :members:
   :undoc-members:
   :show-inheritance:

   MemoryType enum defines the different memory types available for tensor storage.

Parallel Types
------------

.. autoclass:: nvfuser.ParallelType
   :members:
   :undoc-members:
   :show-inheritance:

   ParallelType enum defines the different parallelization strategies available for tensor operations.

Scheduler Types
-------------

.. autoclass:: nvfuser.SchedulerType
   :members:
   :undoc-members:
   :show-inheritance:

   SchedulerType enum defines the different scheduling strategies available for optimizing fusion execution.
