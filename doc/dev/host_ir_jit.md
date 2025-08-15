<!--
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Host IR JIT Overview

## Introduction

Host IR JIT (Just-In-Time) is a new runtime targeting to reduce host side latency. 
It:
1. captures graph dependencies at compile time
2. has register aligned access in runtime comparing with hash table lookup

## JIT Compilation Process

### 1. LLVM Integration
Host IR JIT uses LLVM's ORC (On-Request Compilation) JIT framework:
LLVM IR is translated from host IR and saved as an executable in LLVM ORC JIT
at compile time. During runtime, LLVM ORC JIT calls the saved executable with
given inputs and derives results.


### 2. Compilation Pipeline

**Below code snippets are a pseudo code for host ir jit architecture.**

```cpp
void HostIrJitImpl::compile() {
  // 1. Create LLVM context and module
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("host_ir_jit_module", *context);

  // 2. Compile function declarations
  compileFunctionDeclarations(module.get(), *context);

  // 3. Generate LLVM IR for inputs
  unpackInputs(container_.get(), builder, val_to_value);

  // 4. Compile all top-level expressions
  for (auto* expr : container_->topLevelExprs()) {
    dispatcher.dispatch(expr);
  }

  // 5. Generate LLVM IR for outputs
  packOutputs(container_.get(), builder, val_to_value);

  // 6. Add module to JIT and lookup main function
  jit_->addIRModule(ThreadSafeModule(std::move(module), std::move(context)));
  main_func_ = jit_->lookup(kMainFuncName);
}
```
*Detailed Implementation:* https://github.com/NVIDIA/Fuser/blob/main/csrc/host_ir/jit.cpp#L1125-#L1176

### 3. External Function Integration
In llvm level, we wrap aten fallbacks or other llvm unsupported functions into external C++ calls.
Currently our JIT supports wrapper functions with:

- **Aten Fallbacks**: `matmul`, `linear`, `permute`, `reshape`, `at_empty_strided_cuda`
- **Memory Management**: `new_tensor`, `delete_tensor`, `set_tensor`
- **nvFuser Interals** `launchKernel`
- **Profiling**: NVTX range push/pop for performance analysis
  
*Detailed Implementation:* https://github.com/NVIDIA/Fuser/blob/main/csrc/host_ir/jit.cpp#L1195-#L1396

### 3. IR Translation
The `HostIrCompileDispatcher` translates Host IR expression nodes to LLVM IR:
Currently, host ir jit supports these expressions:
`ViewOp`, `LoadStoreOp`, `MatmulOp`, `LinearOp`, `LaunchKernel`, `Allocate`, `Deallocate`

*Detailed Implementation:* https://github.com/NVIDIA/Fuser/blob/main/csrc/host_ir/jit.cpp#L783-#L1123

## Runtime Execution

### 1. Function Interface
The compiled JIT function follows this signature:
```cpp
using main_func_t = void (*)(int64_t, const void**, void**);
// Parameters: cache_id, input_tensors, output_tensors
```

### 2. Execution Flow
```cpp
KernelArgumentHolder HostIrJitImpl::runWithInputs(const KernelArgumentHolder& args) {
  // 1. Validate inputs and prepare tensor arrays
  std::vector<const void*> input_aten_tensors;
  std::vector<void*> output_aten_tensors;

  // 2. Call compiled JIT function
  main_func_(cache_id, input_aten_tensors.data(), output_aten_tensors.data());

  // 3. Collect and return outputs
  return outputs;
}
```

## Configuration and Build Options
`python setup.py install --build-with-host-ir-jit` will enables NVFUSER_HOST_IR_JIT flag and thus select host ir jit as
backend by default. Otherwise the default backend is host ir evaluator. In the future, when llvm is fully supported in
all build machines, we are able to get rid of this opt-in flag and rather use `enableOption` to control backend switching
after build is done.

## Future Integration plan & steps to turn on by default

**Ops need to be supported:**
- Stream operations
- Communication operations
- remaining Aten fallbacks supported by nvFuser (Alternate way is to compile entire libtorch)

**Other functionalities need to be supported:**
- Alias analysis
- Dynamic shape support in launchKernel
- Correct shape and stride handling for multi device tensor

**Roadmap**
- Enable currently Host IR path (without jit) by default for multi-gpu, with loose requirement of latency
- Enable Host IR Jit for multi-gpu with small set of ops coverage.
- Enable Host IR Jit for single-gpu with small set of ops coverage, ensuring no latency regression
- Enable Host IR Jit for single gpu with full set of ops coverage
