# Thunder test segfualt

## Context on the current work

This branch is an investigation branch off of feature branch
`md/ir-container-uptr` for looking into mysterious segfualts caused by the
refactor in that branch. The aim of `md/ir-container-uptr` is to move
IrContainer from an Inheritance pattern to a composition one. In the feature
branch we kept the current inheritance hierarchy of `IrContainer` -> `Fusion` ->
`Kernel`/`HostIrContainer` to not upset the "interface" throughout the rest of
the code that assumes `IrContainer` as the base type for dynamic dispatch.

Functionally the current `IrStorage` works the same as the old `IrContainer`.
`IrContainer` in the feature branch is just a forwarding class to maintain the
interface.

The composition pattern is a non-negotiable as this is a part of a larger
refactor and this is the first phase.

Outside of the current errors the eature branch is considered "complete"
functionally, we are just tying up loose ends.

## The Problem

We are seeing a segfault in one of the Thunder CI test nvfuser is run with, this
could be similar to the error in the SERDE_SEGFAULT_INVESTIGATION. That issue
has been resolved as that specific failing test has been removed for unrelated
reasons to this IrContainer work (note we did not find the error in this branch
related to that failure).

To reproduce:
```bash
pytest /opt/pytorch/lightning-thunder/thunder/tests/test_grad.py -k "outer_nvfuser_cuda_thunder and float32" -vxs
```

## Instructions for building / running during this investigation

### Compiling

Always use the script `_bn` to compile and build the project this utilizes
`sccache` for fast installs and handles installation into the system python
environment.

**DO NOT** directly build from `cmake` or `pip install` commands.

### Env Args

I prefer using `clang-20` - this is already set in the `CC` and `CXX` env vars.
Some env flags you may want to use for `_bn`
`NVFUSER_SOURCE_DIR=<path to current source>` <-- REQUIRED
`NVFUSER_BUILD_BUILD_TYPE=RelWithDebInfo` `NVFUSER_BUILD_ENABLE_PCH=1` <-- for
faster build w/ clang clean builds e2e ~4-5 minutes.
`NVFUSER_BUILD_NO_BENCHMARK=1` <-- we probably dont need to build benchmark
targets.

For more information on available build and runtime env flags see:
`/root/workspace/Fuser/main/tools/env-config`

### Debugging

The system has lldb and gdb.

We can compile builds with address sanitizer: `NVFUSER_BUILD_WITH_ASAN=1`

Whan running python scripts for nvfuser tests / reproducers you will need:
`LD_PRELOAD=$(clang-20 -print-file-name=libclang_rt.asan-x86_64.so)`

# Investigation Report
