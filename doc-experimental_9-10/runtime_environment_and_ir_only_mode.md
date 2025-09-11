## Runtime Environment Issues and IR-Only Mode

This guide documents common runtime problems when executing nvFuser samples and how to recover quickly. It also explains IR-only mode for inspection without CUDA execution.

### Quick fixes (copy/paste)

- Missing shared libs (e.g., `libc10_cuda.so`):
```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\
/opt/pytorch/nvfuser/python/nvfuser_common/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
<path-to-sample>
```

- Driver-toolkit mismatch (cannot run CUDA):
```bash
NVFUSER_IR_ONLY=1 <path-to-sample>
```

References: `../doc-bot/experimenting/9-3-2025/workarounds`, `../doc-bot/experimenting/how_to_run`.

### Symptoms and resolutions

- Error: `error while loading shared libraries: libc10_cuda.so: cannot open shared object file`
  - Cause: Runtime loader cannot locate Torch or CUDA libs.
  - Fix: Prepend `LD_LIBRARY_PATH` as above (Torch lib path, nvFuser_common lib, CUDA lib64).

- Error: NVIDIA driver mismatch (e.g., version too old/new for the toolkit)
  - Cause: Kernel launch fails due to driver-toolkit incompatibility.
  - Options:
    - Update the driver or use a matching environment.
    - Use IR-only mode (`NVFUSER_IR_ONLY=1`) to inspect the Fusion IR and transforms without CUDA execution.

### IR-only mode

- Purpose: Inspect Fusion IR, transforms, and scheduling artifacts without launching CUDA.
- Usage:
```bash
NVFUSER_IR_ONLY=1 /opt/pytorch/nvfuser/doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_1
```
- What you can see:
  - Fusion IR with/without transforms via sample prints
  - Optionally, scheduled IR/code using executor “most recent” getters if the sample requests them conditionally

### Recommended run setup

- Use the helper script to set the runtime paths, then run:
```bash
source ../doc-bot/experimenting/setup_libs_to_run
/opt/pytorch/nvfuser/doc-bot/experimenting/8-22-2025/tv_add_2d_SAMPLE_2
```

### Notes on input binding and execution context

- Fusion IR is symbolic; real sizes and pointers arrive at execution.
- If you see shape or dtype errors, ensure inputs are pushed in the same order as `fusion->inputs()`.
- For more on binding and inspection, see `../doc-bot/experimenting/9-3-2025/key_insights.md`.


