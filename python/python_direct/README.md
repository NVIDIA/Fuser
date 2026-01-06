<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->
# Direct Bindings API

The Direct Python API maps from CPP to Python directly, so any CPP function can be exposed to python users.

### Legacy Python API
* A Trie data structure is used to map a fusion math definition to FusionExecutorCache.
* This FusionCache creates the CPP Fusion. The user cannot directly create nor modify the CPP fusion in python.
* RecordFunctor is created for each operation to handle equivalence, hashing, printing, and CPP interoperability.

### Differences
* Caching fusions by definitions AND serialization is not supported with direct bindings.
* Supports using constant scalars with any operation.
* The python reproducers map directly to Fusion IR, so it will be more verbose. Direct bindings does not have `RecordFunctor` objects to map from Fusion IR to original python API.

## How to add Fusion IR?
* Create `nb:class_` for IR node.
* `Statement`, `Expr`, `Val`, `IterDomain`, `TensorDomain`, `TensorView`, and `Scalar` exist in `python_direct/ir.cpp`
* All other nodes from `csrc/ir/internal_nodes.h` go to `python_direct/internal_ir.cpp`
* Add functions to the IR node's `pyt::class_`. All nodes map CPP `toString` to python `__str__`, so the node is printable. Other functions are usually added to support scheduling.

## How to add new operation in Direct Bindings API?
* Add operation to `python_direct/ops.cpp` with numpy-style docstring.
* If an operation corresponds with some new Expr nodes, add the appropriate `void handle(const SomeOp*) final` to `python_direct/python_translate.cpp`. For example, `broadcast_in_dim` is a composite operations using `broadcast` and `expand`, so you would need to add two `handle` functions to `PythonTranslator`.

## How to add support for an expression not yet overriden by PythonTranslator?
1. Create handle function for expression.
2. Check if IR node pointer is not nullptr.
3. Add output values for Expr node to `visited_vals_`.
4. Create dynamic scalar input arguments. This step is for view and expand operations. TensorView input arguments are handled via DAG traversal. Constant scalars are added directly to python defintion.
5. Use `PythonPrinter::generateOperation` if the operation only uses positional arguments. This is mainly used for unary and binary operations.
6. Use `PythonPrinter::generateKwargsOperation` if the operation uses keyword arguments. If none of the keyword arguments have default arguments, create a static vector of strings. If some of the keyword arguments have default arguments, create a vector of KeywordArgument. The KeywordArgument struct hold default values for keyword arguments. Use `std::nullopt` for keyword arguments without default values.

## How to debug PythonTranslator?
1. Recompile with debug symbols with `export NVFUSER_BUILD_BUILD_TYPE=RelwithDebInfo`.
2. Run `gdb python`.
3. Catch exception in gdb `(gdb) catch throw`.
4. Run failing test with `r -m pytest test_python_frontend.py -k [your_failing_test]`.
5. At gdb Catchpoint, get backtrace for call stack using `(gdb) bt`.
6. Find and fix failure in PythonTranslate.
