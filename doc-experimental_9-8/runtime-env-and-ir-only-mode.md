# Runtime Environment Issues and IR-Only Mode (Heavily Commented)

## What you’ll learn
- Common run-time failures when executing nvFuser samples
- How to fix missing shared library errors and driver mismatches
- How to run in IR-only mode to keep working without CUDA execution

Primary reference: `../doc-bot/experimenting/9-3-2025/workarounds`

---

## Symptom 1: Missing shared libraries (e.g., libc10_cuda.so)

Error example:
```
error while loading shared libraries: libc10_cuda.so: cannot open shared object file: No such file or directory
```

Why it happens:
- Your binary links against Torch/ATen CUDA libraries, but the dynamic loader can’t locate them at runtime.

Quick fix (session-local):
```bash
# Add Torch, nvFuser common, and CUDA lib64 to the search path
export LD_LIBRARY_PATH=\
/usr/local/lib/python3.12/dist-packages/torch/lib:\
/opt/pytorch/nvfuser/python/nvfuser_common/lib:\
/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Quick verify: list a known lib (path may vary)
ls /usr/local/lib/python3.12/dist-packages/torch/lib/libc10.so | cat
```

Longer-term fixes:
- Bake rpath at link time (see the build article’s `-Wl,-rpath,...` example)
- Install libraries into standard loader paths

---

## Symptom 2: NVIDIA driver / CUDA mismatch

Error example:
```
NVIDIA driver on your system is too old (found version 12040)
```

Options:
- Update the NVIDIA driver to match your CUDA/Torch
- Use a container or environment matching the installed driver
- Temporarily switch to IR-only mode (no GPU execution required)

---

## IR-only mode: keep working without CUDA execution

Purpose:
- Allow programs to run far enough to build/print Fusion IR and transforms without launching GPU kernels. Great for documentation and debugging IR.

How to enable:
```bash
export NVFUSER_IR_ONLY=1
./tv_add_2d_SAMPLE_2   # program prints IR; skips device execution
```

Notes:
- Your program’s IR printing code (e.g., `fusion.print(...)`, `tv->printTransforms()`) continues to work
- Any paths that require actual kernel launches are skipped or stubbed

---

## Handy checklist
- Missing libs: set `LD_LIBRARY_PATH` or add rpath when linking
- Driver mismatch: update driver or use IR-only mode
- Sanity run (IR-only): verify IR printing works end-to-end

See also:
- Build/Run guide: `./build-and-run-standalone-samples.md`
- Getting started sample: `./getting-started-tv-add-2d.md`
