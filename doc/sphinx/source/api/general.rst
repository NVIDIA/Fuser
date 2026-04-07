..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

General
==========

This section documents the common API components of nvFuser.

Statement
---------

.. autoclass:: nvfuser_direct.Statement
   :members:
   :undoc-members:

   Statement represents a base class for all statements in the nvFuser IR.
   It provides methods for:

   - Accessing statement properties
   - Managing statement dependencies
   - Querying statement type and attributes

Val
---

.. autoclass:: nvfuser_direct.Val
   :members:
   :undoc-members:
   :show-inheritance:

   Val represents a value in the nvFuser IR. It provides methods for:

   - Accessing value properties and type
   - Managing value dependencies
   - Querying value attributes and constraints

Expr
----

.. autoclass:: nvfuser_direct.Expr
   :members:
   :undoc-members:
   :show-inheritance:

   Expr represents an expression in the nvFuser IR. It provides methods for:

   - Accessing expression operands
   - Managing expression dependencies
   - Querying expression type and attributes

IterDomain
----------

.. autoclass:: nvfuser_direct.IterDomain
   :members:
   :undoc-members:
   :show-inheritance:

   IterDomain represents an iteration domain in the nvFuser IR. It provides methods for:

   - Accessing domain properties (extent, start, stop)
   - Managing domain dependencies
   - Querying domain attributes and constraints

TensorDomain
------------

.. autoclass:: nvfuser_direct.TensorDomain
   :members:
   :undoc-members:
   :show-inheritance:

   TensorDomain represents a tensor domain in the nvFuser IR. It provides methods for:

   - Accessing domain dimensions
   - Managing domain properties
   - Querying domain attributes and constraints

TensorView
----------

.. autoclass:: nvfuser_direct.TensorView
   :members:
   :undoc-members:
   :show-inheritance:

   TensorView represents a tensor in the fusion definition. It provides methods for:

   - Accessing tensor properties (domain, axes, etc.)
   - Performing operations on tensors
   - Managing tensor memory and scheduling

FusionExecutorCache
-------------------

.. autoclass:: nvfuser_direct.FusionExecutorCache
   :members:
   :undoc-members:

   FusionExecutorCache manages the compilation and execution of fusion definitions.
   It provides methods for:

   - Compiling fusions
   - Executing fusions with input tensors
   - Accessing compiled kernels and IR

KernelExecutor
--------------

.. autoclass:: nvfuser_direct.KernelExecutor
   :members:
   :undoc-members:

   KernelExecutor provides a lower-level interface for executing compiled kernels.
   It allows for:

   - Manual kernel compilation
   - Direct kernel execution
   - Access to kernel properties and scheduling information

Fusion Definition
-----------------

.. autoclass:: nvfuser_direct.FusionDefinition
   :members:
   :undoc-members:

   The DirectFusionDefinition class provides a low-level interface for defining nvFuser operations.
   It exposes two main components:

   - ``ops``: Contains the operators for defining fusion operations
